"""
RAG Add Tool
Concurrent embeddings + batch inserts + proper error handling
"""

import asyncio
import logging
import time
from typing import Dict, Any, List

logger = logging.getLogger("mcp_server")

# Safe limits for embedding models
MAX_CHUNK_TOKENS = 350  # Conservative limit (bge-large has ~512 max)
MAX_CHUNK_CHARS = MAX_CHUNK_TOKENS * 4  # ~1400 characters


def estimate_tokens(text: str) -> int:
    """Estimate token count (1 token ≈ 4 characters)"""
    return len(text) // 4


def split_text_safe(text: str, max_tokens: int = MAX_CHUNK_TOKENS) -> List[str]:
    """
    Split text into token-safe chunks with proper boundary detection.

    Args:
        text: Text to split
        max_tokens: Maximum tokens per chunk (default: 350)

    Returns:
        List of text chunks guaranteed to be under token limit
    """
    max_chars = max_tokens * 4

    if len(text) <= max_chars:
        # Verify it's actually safe
        if estimate_tokens(text) <= max_tokens:
            return [text]

    chunks = []
    paragraphs = text.split('\n\n')
    current_chunk = ""

    for para in paragraphs:
        # If paragraph itself is too large, split by sentences
        if len(para) > max_chars:
            sentences = []
            for delimiter in ['. ', '! ', '? ', '\n']:
                para = para.replace(delimiter, delimiter + '|SPLIT|')

            for sentence in para.split('|SPLIT|'):
                sentence = sentence.strip()
                if not sentence:
                    continue

                # If current chunk + sentence is too big, flush current
                if len(current_chunk) + len(sentence) > max_chars:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
                else:
                    current_chunk += (" " + sentence if current_chunk else sentence)
        else:
            # Normal paragraph
            if len(current_chunk) + len(para) > max_chars:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = para
            else:
                current_chunk += ("\n\n" + para if current_chunk else para)

    # Add final chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    # Safety check: verify all chunks are under limit
    safe_chunks = []
    for chunk in chunks:
        if estimate_tokens(chunk) > max_tokens:
            # Force split at even smaller size
            logger.warning(f"⚠️ Chunk too large ({estimate_tokens(chunk)} tokens), force-splitting...")
            sub_chunks = force_split_chunk(chunk, max_tokens // 2)
            safe_chunks.extend(sub_chunks)
        else:
            safe_chunks.append(chunk)

    return safe_chunks


def force_split_chunk(text: str, max_tokens: int) -> List[str]:
    """Force split a chunk at character boundaries"""
    max_chars = max_tokens * 4
    chunks = []

    for i in range(0, len(text), max_chars):
        chunk = text[i:i + max_chars].strip()
        if chunk:
            chunks.append(chunk)

    return chunks


async def generate_embeddings_concurrent(texts: List[str]) -> List[tuple]:
    """
    Generate embeddings for all chunks concurrently.

    Args:
        texts: List of text chunks

    Returns:
        List of tuples (success: bool, result: embedding or error)
    """
    from langchain_ollama import OllamaEmbeddings

    embeddings_model = OllamaEmbeddings(model="bge-large")

    async def embed_one(text: str, index: int) -> tuple:
        """Embed single text chunk"""
        try:
            # Run in thread pool since ollama is sync
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                embeddings_model.embed_query,
                text
            )
            return (True, embedding, index)
        except Exception as e:
            logger.error(f"❌ Embedding failed for chunk {index}: {e}")
            return (False, str(e), index)

    # Generate all embeddings concurrently
    tasks = [embed_one(text, i) for i, text in enumerate(texts)]
    results = await asyncio.gather(*tasks)

    return results


def rag_add(text: str, source: str = None, chunk_size: int = 400) -> Dict[str, Any]:
    """
    Add text to RAG with threaded embedding generation and batch insert.
    Pure sync - works in any context.

    Args:
        text: Text to add
        source: Source identifier
        chunk_size: Maximum tokens per chunk

    Returns:
        Dictionary with results
    """
    from tools.rag.rag_vector_db import batch_insert_documents
    from concurrent.futures import ThreadPoolExecutor

    start_time = time.time()

    # Override unsafe chunk sizes
    safe_chunk_size = min(chunk_size, MAX_CHUNK_TOKENS)
    if chunk_size > MAX_CHUNK_TOKENS:
        logger.warning(f"⚠️ Chunk size {chunk_size} reduced to safe limit {MAX_CHUNK_TOKENS}")

    logger.info(f"📝 Adding text to RAG (length: {len(text)}, max_tokens: {safe_chunk_size}) for {source}")

    try:
        # Split into safe chunks
        chunks = split_text_safe(text, max_tokens=safe_chunk_size)
        logger.info(f"📦 Split into {len(chunks)} chunks")

        # Generate embeddings using thread pool (NOT async)
        embed_start = time.time()
        results = _generate_embeddings_threaded(chunks)
        embed_duration = time.time() - embed_start

        # Separate successful and failed
        successful_docs = []
        failed_count = 0

        for success, result, index in results:
            if success:
                successful_docs.append({
                    'embedding': result,
                    'source': source,
                })
            else:
                failed_count += 1

        logger.info(f"⚡ Generated {len(successful_docs)} embeddings in {embed_duration:.2f}s ({failed_count} failed)")

        # Batch insert
        if successful_docs:
            insert_start = time.time()
            inserted = batch_insert_documents(successful_docs)
            insert_duration = time.time() - insert_start
            logger.info(f"💾 Batch inserted {inserted} documents in {insert_duration:.2f}s")
        else:
            inserted = 0
            insert_duration = 0

        total_duration = time.time() - start_time

        result = {
            "success": inserted > 0,
            "chunks_added": inserted,
            "chunks_failed": failed_count,
            "source": source,
            "original_length": len(text),
            "processing_time_seconds": round(total_duration, 2),
            "embedding_time_seconds": round(embed_duration, 2),
            "insert_time_seconds": round(insert_duration, 2)
        }

        if failed_count > 0:
            logger.warning(f"⚠️ Added {inserted} chunks, {failed_count} failed")
        else:
            logger.info(f"✅ Successfully added {inserted} chunks")

        return result

    except Exception as e:
        logger.error(f"❌ Error in rag_add: {e}")
        return {
            "success": False,
            "error": str(e),
            "chunks_added": 0,
            "chunks_failed": 0
        }


def _generate_embeddings_threaded(texts: List[str]) -> List[tuple]:
    """
    Generate embeddings using ThreadPoolExecutor (pure sync, no asyncio).

    Args:
        texts: List of text chunks

    Returns:
        List of tuples (success: bool, result: embedding or error, index: int)
    """
    from langchain_ollama import OllamaEmbeddings
    from concurrent.futures import ThreadPoolExecutor

    embeddings_model = OllamaEmbeddings(model="bge-large")

    def embed_one(args: tuple) -> tuple:
        """Embed single text chunk"""
        text, index = args
        try:
            embedding = embeddings_model.embed_query(text)
            return (True, embedding, index)
        except Exception as e:
            logger.error(f"❌ Embedding failed for chunk {index}: {e}")
            return (False, str(e), index)

    # Use ThreadPoolExecutor with map for ordered results
    with ThreadPoolExecutor(max_workers=5) as executor:
        indexed_texts = [(text, i) for i, text in enumerate(texts)]
        results = list(executor.map(embed_one, indexed_texts))

    return results