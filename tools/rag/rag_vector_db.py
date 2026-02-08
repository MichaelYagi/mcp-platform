"""
RAG Vector Database
Handles storage and retrieval of embeddings
"""
import json
import logging
import uuid
from typing import Dict, Any, List
from langchain_ollama import OllamaEmbeddings
from .rag_utils import load_rag_db, save_rag_db, get_connection

logger = logging.getLogger("mcp_server")

# Initialize embeddings model
embeddings_model = OllamaEmbeddings(model="bge-large")

# In-memory cache to avoid loading/saving on every chunk
_db_cache = None
_db_dirty = False
_pending_chunks = []

def load_rag_database():
    """Load database into memory cache"""
    global _db_cache
    if _db_cache is None:
        _db_cache = load_rag_db()
    return _db_cache


def save_rag_database():
    """Save database from memory cache to disk"""
    global _db_cache, _db_dirty
    if _db_dirty and _db_cache is not None:
        save_rag_db(_db_cache)
        _db_dirty = False


def add_to_rag(text: str, source: str = None, save: bool = True) -> Dict[str, Any]:
    """
    Add a single text chunk to the RAG database.

    Args:
        text: Text chunk to add
        source: Source identifier (e.g., "plex:12345")
        save: Whether to save immediately (False for batch operations)

    Returns:
        Dictionary with success status
    """
    global _db_cache, _db_dirty

    try:
        # Load database into cache
        db = load_rag_database()

        # Generate embedding for the text
        logger.debug(f"🔮 Generating embedding for text (length: {len(text)})")
        embedding = embeddings_model.embed_query(text)

        # Create document entry
        doc_id = str(uuid.uuid4())
        doc = {
            "id": doc_id,
            "text": text,
            "embedding": embedding,
            "metadata": {
                "source": source,
                "length": len(text),
                "word_count": len(text.split())
            }
        }

        # Add to in-memory database
        db.append(doc)
        _db_dirty = True

        # Save if requested
        if save:
            save_rag_database()

        logger.debug(f"✅ Added document {doc_id} to RAG (save={save})")

        return {
            "success": True,
            "id": doc_id,
            "length": len(text)
        }

    except Exception as e:
        logger.error(f"❌ Error adding to RAG: {e}")
        raise  # Re-raise to be caught by rag_add


def get_rag_stats() -> Dict[str, Any]:
    """
    Get statistics about the RAG database.

    Returns:
        Dictionary with database statistics
    """
    try:
        db = load_rag_database()

        if not db:
            return {
                "total_documents": 0,
                "total_words": 0,
                "sources": []
            }

        total_words = sum(doc["metadata"]["word_count"] for doc in db)
        sources = list(set(doc["metadata"]["source"] for doc in db if doc["metadata"].get("source")))

        return {
            "total_documents": len(db),
            "total_words": total_words,
            "sources": sources,
            "unique_sources": len(sources)
        }

    except Exception as e:
        logger.error(f"❌ Error getting RAG stats: {e}")
        return {
            "error": str(e)
        }

def add_to_rag_batch(text: str, source: str = None) -> Dict[str, Any]:
    """
    Add a chunk to the pending batch (doesn't save yet).

    Args:
        text: Text chunk to add
        source: Source identifier

    Returns:
        Dictionary with success status
    """
    global _pending_chunks

    try:
        # Generate embedding
        logger.debug(f"🔮 Generating embedding for text (length: {len(text)})")
        embedding = embeddings_model.embed_query(text)

        # Create document entry
        doc_id = str(uuid.uuid4())
        doc = {
            "id": doc_id,
            "text": text,
            "embedding": embedding,
            "metadata": {
                "source": source,
                "length": len(text),
                "word_count": len(text.split())
            }
        }

        # Add to pending batch
        _pending_chunks.append(doc)

        logger.debug(f"✅ Queued document {doc_id} (pending: {len(_pending_chunks)})")

        return {
            "success": True,
            "id": doc_id,
            "length": len(text)
        }

    except Exception as e:
        logger.error(f"❌ Error adding to batch: {e}")
        raise


def flush_batch():
    """
    Save all pending chunks to database.
    Call this after processing a complete movie.
    """
    global _db_cache, _pending_chunks

    if not _pending_chunks:
        return

    logger.info(f"💾 Flushing {len(_pending_chunks)} chunks to database...")

    # Load database
    db = load_rag_database()

    # Add all pending chunks
    db.extend(_pending_chunks)

    # Save to disk
    save_rag_db(db)

    # Clear pending
    _pending_chunks = []

    logger.info(f"✅ Batch saved successfully")

def should_refresh_source(source: str, max_age_days: int = 30) -> bool:
    """
    Check if source should be refreshed based on age.

    Args:
        source: Source URL
        max_age_days: Maximum age in days before refresh

    Returns:
        True if should refresh, False if recent enough
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
                       SELECT created_at
                       FROM documents
                       WHERE source = ?
                       ORDER BY created_at DESC LIMIT 1
                       """, (source,))

        row = cursor.fetchone()

        if not row:
            return True  # Not found, should fetch

        # Check age
        from datetime import datetime, timedelta
        created_at = datetime.fromisoformat(row[0])
        age = datetime.now() - created_at

        if age.days > max_age_days:
            logger.info(f"🔄 Source is {age.days} days old, refreshing...")
            return True

        logger.info(f"⏭️  Source is recent ({age.days} days old), skipping")
        return False

    except Exception as e:
        logger.error(f"❌ Error checking age: {e}")
        return True  # Default to fetching on error


def has_source(source: str) -> bool:
    """
    Check if a source URL already exists in RAG.

    Args:
        source: Source URL to check

    Returns:
        True if source exists, False otherwise
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
                       SELECT COUNT(*)
                       FROM documents
                       WHERE source = ? LIMIT 1
                       """, (source,))

        count = cursor.fetchone()[0]
        return count > 0

    except Exception as e:
        logger.error(f"❌ Error checking source: {e}")
        return False  # Default to allowing storage on error

def batch_insert_documents(documents: List[Dict[str, Any]]) -> int:
    """
    Optimized batch insert with binary embeddings.
    """
    import uuid
    import numpy as np

    try:
        conn = get_connection()
        cursor = conn.cursor()

        # Prepare data with binary embeddings (MUCH faster than JSON)
        batch_data = []
        for doc in documents:
            doc_id = str(uuid.uuid4())

            # Convert embedding to numpy array then bytes
            embedding_array = np.array(doc['embedding'], dtype=np.float32)
            embedding_bytes = embedding_array.tobytes()

            batch_data.append((
                doc_id,
                doc['text'],
                embedding_bytes,  # Binary, not JSON!
                doc.get('source'),
                doc.get('length'),
                doc.get('word_count')
            ))

        # Fast transaction
        cursor.execute("BEGIN IMMEDIATE")
        cursor.executemany("""
                           INSERT INTO documents
                               (id, text, embedding, source, length, word_count)
                           VALUES (?, ?, ?, ?, ?, ?)
                           """, batch_data)
        cursor.execute("COMMIT")

        logger.debug(f"💾 Batch inserted {len(batch_data)} documents")
        return len(batch_data)

    except Exception as e:
        logger.error(f"❌ Batch insert failed: {e}")
        try:
            cursor.execute("ROLLBACK")
        except:
            pass
        return 0