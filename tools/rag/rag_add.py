"""
RAG Add Tool
Concurrent embeddings + batch inserts + proper error handling.

For each chunk:
  1. Text is persisted to sessions.db (chunks table) via session_manager.store_chunk()
  2. The returned chunk_id is stored alongside the embedding in rag_database.db
  3. Text is never written to rag_database.db
"""

import asyncio
import logging
import time
from typing import Dict, Any, List

logger = logging.getLogger("mcp_server")

# Safe limits for embedding models
MAX_CHUNK_TOKENS = 350        # Conservative limit (bge-large has ~512 max)
MAX_CHUNK_CHARS = MAX_CHUNK_TOKENS * 4   # ~1400 characters


def estimate_tokens(text: str) -> int:
    """Estimate token count (1 token ≈ 4 characters)"""
    return len(text) // 4


def split_text_safe(text: str, max_tokens: int = MAX_CHUNK_TOKENS) -> List[str]:
    """
    Split text into token-safe chunks with proper boundary detection.

    Args:
        text:       Text to split
        max_tokens: Maximum tokens per chunk (default: 350)

    Returns:
        List of text chunks guaranteed to be under token limit
    """
    max_chars = max_tokens * 4

    if len(text) <= max_chars:
        if estimate_tokens(text) <= max_tokens:
            return [text]

    chunks = []
    paragraphs = text.split('\n\n')
    current_chunk = ""

    for para in paragraphs:
        if len(para) > max_chars:
            for delimiter in ['. ', '! ', '? ', '\n']:
                para = para.replace(delimiter, delimiter + '|SPLIT|')

            for sentence in para.split('|SPLIT|'):
                sentence = sentence.strip()
                if not sentence:
                    continue
                if len(current_chunk) + len(sentence) > max_chars:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    current_chunk += (" " + sentence if current_chunk else sentence)
        else:
            if len(current_chunk) + len(para) > max_chars:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                current_chunk += ("\n\n" + para if current_chunk else para)

    if current_chunk:
        chunks.append(current_chunk.strip())

    # Safety check: verify all chunks are under limit
    safe_chunks = []
    for chunk in chunks:
        if estimate_tokens(chunk) > max_tokens:
            logger.warning(f"⚠️ Chunk too large ({estimate_tokens(chunk)} tokens), force-splitting...")
            safe_chunks.extend(force_split_chunk(chunk, max_tokens // 2))
        else:
            safe_chunks.append(chunk)

    return safe_chunks


def force_split_chunk(text: str, max_tokens: int) -> List[str]:
    """Force split a chunk at character boundaries"""
    max_chars = max_tokens * 4
    return [text[i:i + max_chars].strip() for i in range(0, len(text), max_chars) if text[i:i + max_chars].strip()]


def rag_add(text: str, source: str = None, chunk_size: int = 400) -> Dict[str, Any]:
    """
    Add text to RAG with threaded embedding generation and batch insert.

    For each chunk, the text is first stored in sessions.db to get a chunk_id,
    then the embedding + chunk_id are inserted into rag_database.db.

    Args:
        text:       Text to add
        source:     Source identifier (URL, file path, etc.)
        chunk_size: Maximum tokens per chunk

    Returns:
        Dictionary with results
    """
    from tools.rag.rag_vector_db import batch_insert_documents
    from client.session_manager import SessionManager

    start_time = time.time()

    safe_chunk_size = min(chunk_size, MAX_CHUNK_TOKENS)
    if chunk_size > MAX_CHUNK_TOKENS:
        logger.warning(f"⚠️ Chunk size {chunk_size} reduced to safe limit {MAX_CHUNK_TOKENS}")

    logger.info(f"📝 Adding text to RAG (length: {len(text)}, max_tokens: {safe_chunk_size}) for {source}")

    try:
        chunks = split_text_safe(text, max_tokens=safe_chunk_size)
        logger.info(f"📦 Split into {len(chunks)} chunks")

        # Persist chunk text to sessions.db before embedding so chunk_ids exist.
        # Skip any chunk that already exists for this source (exact text match)
        # so re-ingesting a URL never creates duplicates.
        session_manager = SessionManager()
        chunk_ids = []
        skipped_chunks = []

        # Build a set of existing chunk texts for this source in one query
        # to avoid N round-trips for large documents.
        try:
            from tools.rag.rag_vector_db import get_connection as _rag_conn
            _conn = _rag_conn()
            _sessions_db = session_manager.db_path
            _conn.execute("ATTACH DATABASE ? AS _sess_dedup", (_sessions_db,))
            _existing = set(
                row[0] for row in _conn.execute("""
                    SELECT c.text
                    FROM documents d
                    JOIN _sess_dedup.chunks c ON d.chunk_id = c.id
                    WHERE d.source = ?
                """, (source,)).fetchall()
            )
            _conn.execute("DETACH DATABASE _sess_dedup")
            _conn.close()
        except Exception as _dedup_err:
            logger.warning(f"⚠️ Could not pre-fetch existing chunks for dedup: {_dedup_err}")
            _existing = set()

        for chunk in chunks:
            if chunk in _existing:
                skipped_chunks.append(chunk)
                logger.debug(f"⏭️ Skipping duplicate chunk for source: {source}")
                continue
            chunk_id = session_manager.store_chunk(source=source, text=chunk)
            chunk_ids.append(chunk_id)

        if skipped_chunks:
            logger.info(f"⏭️ Skipped {len(skipped_chunks)} duplicate chunk(s) for source: {source}")

        # Only embed the new chunks — chunk_ids and chunks are now aligned
        chunks = [c for c in chunks if c not in _existing]

        logger.debug(f"💾 Stored {len(chunk_ids)} chunk texts in sessions.db")

        # Generate embeddings (threaded, no asyncio)
        embed_start = time.time()
        results = _generate_embeddings_threaded(chunks)
        embed_duration = time.time() - embed_start

        # Build document list pairing each embedding with its chunk_id
        successful_docs = []
        failed_count = 0

        for success, result, index in results:
            if success:
                successful_docs.append({
                    'embedding': result,
                    'source':    source,
                    'chunk_id':  chunk_ids[index],
                })
            else:
                failed_count += 1
                # Clean up the orphaned chunk row so sessions.db stays in sync
                try:
                    from tools.rag.rag_utils import get_connection
                    # We stored the chunk but embedding failed — delete it
                    sm_conn = __import__('sqlite3').connect(session_manager.db_path)
                    sm_conn.execute("DELETE FROM chunks WHERE id = ?", (chunk_ids[index],))
                    sm_conn.commit()
                    sm_conn.close()
                except Exception as cleanup_err:
                    logger.warning(f"⚠️ Could not clean up orphaned chunk {chunk_ids[index]}: {cleanup_err}")

        logger.info(
            f"⚡ Generated {len(successful_docs)} embeddings in {embed_duration:.2f}s "
            f"({failed_count} failed)"
        )

        if successful_docs:
            insert_start = time.time()
            inserted = batch_insert_documents(successful_docs)
            insert_duration = time.time() - insert_start
            logger.info(f"💾 Batch inserted {inserted} documents in {insert_duration:.2f}s")
        else:
            inserted = 0
            insert_duration = 0

        total_duration = time.time() - start_time

        if failed_count > 0:
            logger.warning(f"⚠️ Added {inserted} chunks, {failed_count} failed")
        else:
            logger.info(f"✅ Successfully added {inserted} chunks")

        return {
            "success": inserted > 0 or len(skipped_chunks) > 0,
            "chunks_added": inserted,
            "chunks_skipped": len(skipped_chunks),
            "chunks_failed": failed_count,
            "source": source,
            "original_length": len(text),
            "processing_time_seconds": round(total_duration, 2),
            "embedding_time_seconds": round(embed_duration, 2),
            "insert_time_seconds": round(insert_duration, 2),
        }

    except Exception as e:
        logger.error(f"❌ Error in rag_add: {e}")
        return {
            "success": False,
            "error": str(e),
            "chunks_added": 0,
            "chunks_skipped": 0,
            "chunks_failed": 0,
        }


def _generate_embeddings_threaded(texts: List[str]) -> List[tuple]:
    """
    Generate embeddings using ThreadPoolExecutor (pure sync, no asyncio).

    Args:
        texts: List of text chunks

    Returns:
        List of tuples (success: bool, result: embedding or error str, index: int)
    """
    from langchain_ollama import OllamaEmbeddings
    from concurrent.futures import ThreadPoolExecutor

    embeddings_model = OllamaEmbeddings(model="bge-large")

    def embed_one(args: tuple) -> tuple:
        text, index = args
        try:
            embedding = embeddings_model.embed_query(text)
            return (True, embedding, index)
        except Exception as e:
            logger.error(f"❌ Embedding failed for chunk {index}: {e}")
            return (False, str(e), index)

    with ThreadPoolExecutor(max_workers=5) as executor:
        indexed_texts = [(text, i) for i, text in enumerate(texts)]
        results = list(executor.map(embed_one, indexed_texts))

    return results