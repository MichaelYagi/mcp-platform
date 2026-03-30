"""
RAG Vector Database
Handles storage and retrieval of embeddings.
Text is never stored here — chunk text lives in sessions.db (chunks table),
looked up via chunk_id at search time.
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


def add_to_rag(text: str, source: str = None, chunk_id: int = None, save: bool = True) -> Dict[str, Any]:
    """
    Embed a single text chunk and add it to the RAG database.

    Args:
        text:     Text chunk to embed
        source:   Source identifier (e.g. URL, file path)
        chunk_id: FK into sessions.db chunks.id — must be pre-created by the caller
                  via session_manager.store_chunk() before calling this function
        save:     Whether to persist immediately (False for batch operations)

    Returns:
        Dictionary with success status and doc id
    """
    global _db_cache, _db_dirty

    try:
        db = load_rag_database()

        logger.debug(f"🔮 Generating embedding for text (length: {len(text)})")
        embedding = embeddings_model.embed_query(text)

        doc_id = str(uuid.uuid4())
        doc = {
            "id":        doc_id,
            "embedding": embedding,
            "chunk_id":  chunk_id,
            "metadata":  {"source": source},
        }

        db.append(doc)
        _db_dirty = True

        if save:
            save_rag_database()

        logger.debug(f"✅ Added document {doc_id} to RAG (chunk_id={chunk_id}, save={save})")
        return {"success": True, "id": doc_id, "length": len(text)}

    except Exception as e:
        logger.error(f"❌ Error adding to RAG: {e}")
        raise


def get_rag_stats() -> Dict[str, Any]:
    """Get statistics about the RAG database."""
    try:
        import os
        from pathlib import Path

        conn = get_connection()
        cursor = conn.cursor()

        # Attach sessions.db to access chunk text lengths
        sessions_db = os.getenv(
            "SESSIONS_DB",
            str(Path(__file__).resolve().parent.parent.parent / "data" / "sessions.db")
        )
        cursor.execute("ATTACH DATABASE ? AS sessions", (sessions_db,))

        cursor.execute("SELECT COUNT(*) FROM documents")
        total_documents = cursor.fetchone()[0]

        cursor.execute("""
            SELECT COUNT(DISTINCT d.source),
                   SUM(LENGTH(c.text)),
                   COUNT(LENGTH(c.text))
            FROM documents d
            JOIN sessions.chunks c ON d.chunk_id = c.id
        """)
        row = cursor.fetchone()
        unique_sources = row[0] or 0
        total_chars = row[1] or 0
        chunks_with_text = row[2] or 0
        # Approximate word count from char count
        total_words = total_chars // 5

        cursor.execute("""
            SELECT DISTINCT source FROM documents
            WHERE source IS NOT NULL AND source != ''
        """)
        sources = [r[0] for r in cursor.fetchall()]

        return {
            "total_documents": total_documents,
            "unique_sources": unique_sources,
            "total_words": total_words,
            "total_chars": total_chars,
            "sources": sources,
        }

    except Exception as e:
        logger.error(f"❌ Error getting RAG stats: {e}")
        return {"error": str(e)}
    finally:
        try:
            conn.close()
        except Exception:
            pass


def add_to_rag_batch(text: str, source: str = None, chunk_id: int = None) -> Dict[str, Any]:
    """
    Queue a chunk for batch insertion (does not persist yet).

    Args:
        text:     Text chunk to embed
        source:   Source identifier
        chunk_id: FK into sessions.db chunks.id

    Returns:
        Dictionary with success status
    """
    global _pending_chunks

    try:
        logger.debug(f"🔮 Generating embedding for text (length: {len(text)})")
        embedding = embeddings_model.embed_query(text)

        doc_id = str(uuid.uuid4())
        doc = {
            "id":        doc_id,
            "embedding": embedding,
            "chunk_id":  chunk_id,
            "metadata":  {"source": source},
        }

        _pending_chunks.append(doc)
        logger.debug(f"✅ Queued document {doc_id} (pending: {len(_pending_chunks)})")
        return {"success": True, "id": doc_id, "length": len(text)}

    except Exception as e:
        logger.error(f"❌ Error adding to batch: {e}")
        raise


def flush_batch():
    """Save all pending chunks to the database."""
    global _db_cache, _pending_chunks

    if not _pending_chunks:
        return

    logger.info(f"💾 Flushing {len(_pending_chunks)} chunks to database...")

    db = load_rag_database()
    db.extend(_pending_chunks)
    save_rag_db(db)
    _pending_chunks = []

    logger.info("✅ Batch saved successfully")


def should_refresh_source(source: str, max_age_days: int = 30) -> bool:
    """Check if a source should be refreshed based on age."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT created_at FROM documents
            WHERE source = ? ORDER BY created_at DESC LIMIT 1
        """, (source,))
        row = cursor.fetchone()

        if not row:
            return True

        from datetime import datetime
        created_at = datetime.fromisoformat(row[0])
        age = datetime.now() - created_at

        if age.days > max_age_days:
            logger.info(f"🔄 Source is {age.days} days old, refreshing...")
            return True

        logger.info(f"⏭️  Source is recent ({age.days} days old), skipping")
        return False

    except Exception as e:
        logger.error(f"❌ Error checking source age: {e}")
        return True
    finally:
        conn.close()


def has_source(source: str) -> bool:
    """Check if a source URL already exists in the RAG DB."""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents WHERE source = ? LIMIT 1", (source,))
        return cursor.fetchone()[0] > 0
    except Exception as e:
        logger.error(f"❌ Error checking source: {e}")
        return False
    finally:
        conn.close()


def batch_insert_documents(documents: List[Dict[str, Any]]) -> int:
    """
    Optimized batch insert with binary embeddings.

    Each document dict must contain:
        embedding  — list of floats
        source     — source identifier string
        chunk_id   — int FK into sessions.db chunks.id
    """
    import numpy as np

    try:
        conn = get_connection()
        cursor = conn.cursor()

        batch_data = []
        for doc in documents:
            doc_id = str(uuid.uuid4())
            embedding_bytes = np.array(doc['embedding'], dtype=np.float32).tobytes()
            batch_data.append((
                doc_id,
                embedding_bytes,
                doc.get('source'),
                doc.get('chunk_id'),
            ))

        cursor.execute("BEGIN IMMEDIATE")
        cursor.executemany("""
            INSERT INTO documents (id, embedding, source, chunk_id)
            VALUES (?, ?, ?, ?)
        """, batch_data)
        cursor.execute("COMMIT")

        logger.debug(f"💾 Batch inserted {len(batch_data)} documents")
        return len(batch_data)

    except Exception as e:
        logger.error(f"❌ Batch insert failed: {e}")
        try:
            cursor.execute("ROLLBACK")
        except Exception:
            pass
        return 0
    finally:
        conn.close()