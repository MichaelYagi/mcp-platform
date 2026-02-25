"""
Conversation RAG Bridge
Handles storing and retrieving conversation turns in the RAG store,
scoped by session_id. Completely separate from document/external RAG.

Store:    store_turn(session_id, role, content)
Retrieve: retrieve_context(session_id, query, top_k) -> List[Dict]
"""

import logging
import time
import uuid
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("mcp_server")

# Single shared thread pool for embedding conversation turns
_embed_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="conv_rag")

# Role prefix used when embedding so the model understands speaker context
ROLE_PREFIX = {
    "user": "User said: ",
    "assistant": "Assistant said: ",
}

# Maximum characters to embed per turn (conversation turns don't need chunking,
# but very long assistant responses should be trimmed to a representative window)
MAX_TURN_CHARS = 1200


def _get_embeddings_model():
    """Lazy-load embeddings model (same model as rest of RAG)"""
    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(model="bge-large")


def _embed_text(text: str) -> Optional[List[float]]:
    """Generate embedding for a single text string. Returns None on failure."""
    try:
        model = _get_embeddings_model()
        return model.embed_query(text)
    except Exception as e:
        logger.error(f"❌ conversation_rag: embedding failed: {e}")
        return None


def store_turn(session_id: int, role: str, content: str) -> bool:
    """
    Embed a single conversation turn and store it in the RAG database,
    tagged with session_id so it can be retrieved in isolation later.

    Args:
        session_id: Integer session ID from session_manager
        role:       'user' or 'assistant'
        content:    Message text

    Returns:
        True if stored successfully, False otherwise
    """
    from tools.rag.rag_utils import get_connection

    if not content or not content.strip():
        return False

    # Prefix with role so embeddings carry speaker context
    prefix = ROLE_PREFIX.get(role, f"{role}: ")
    text_to_embed = prefix + content[:MAX_TURN_CHARS]

    try:
        embedding = _embed_text(text_to_embed)
        if embedding is None:
            return False

        import numpy as np
        embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()

        doc_id = str(uuid.uuid4())
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO documents (id, text, embedding, source, session_id, length, word_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            doc_id,
            content,                    # Store original (no prefix) for display
            embedding_bytes,
            "conversation",             # source identifies type
            str(session_id),            # session_id scopes retrieval
            len(content),
            len(content.split())
        ))
        conn.commit()

        logger.debug(f"💬 Stored {role} turn for session {session_id} (doc {doc_id[:8]})")
        return True

    except Exception as e:
        logger.error(f"❌ conversation_rag.store_turn failed: {e}")
        return False


def store_turn_async(session_id: int, role: str, content: str) -> None:
    """
    Fire-and-forget version of store_turn. Runs embedding in background
    thread so it never blocks the WebSocket response path.

    Args:
        session_id: Integer session ID
        role:       'user' or 'assistant'
        content:    Message text
    """
    def _run():
        start = time.time()
        ok = store_turn(session_id, role, content)
        elapsed = time.time() - start
        if ok:
            logger.debug(f"✅ conv_rag async store done ({elapsed:.2f}s) session={session_id} role={role}")
        else:
            logger.warning(f"⚠️ conv_rag async store failed session={session_id} role={role}")

    _embed_executor.submit(_run)


def retrieve_context(
    session_id: int,
    query: str,
    top_k: int = 5,
    min_score: float = 0.35
) -> List[Dict[str, Any]]:
    """
    Retrieve the most semantically relevant past turns from this session.

    Does NOT return the full history — only what's relevant to the current query.
    The caller (context_tracker) decides how to format this into the prompt.

    Args:
        session_id:  Integer session ID to scope retrieval
        query:       Current user message (used as query embedding)
        top_k:       Maximum results to return
        min_score:   Minimum cosine similarity threshold

    Returns:
        List of dicts with keys: text, role, score
        Sorted by score descending.
    """
    from tools.rag.rag_utils import get_connection, cosine_similarity

    if not query or not query.strip():
        return []

    try:
        # Embed the query
        query_embedding = _embed_text(query)
        if query_embedding is None:
            return []

        # Load only this session's turns from DB (not the whole RAG store)
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, text, embedding, source
            FROM documents
            WHERE session_id = ?
            ORDER BY created_at ASC
        """, (str(session_id),))

        rows = cursor.fetchall()

        if not rows:
            logger.debug(f"💬 No RAG turns found for session {session_id}")
            return []

        import numpy as np

        results = []
        for row in rows:
            embedding_data = row[2]

            # Handle binary format
            if isinstance(embedding_data, bytes):
                embedding = np.frombuffer(embedding_data, dtype=np.float32).tolist()
            else:
                import json
                embedding = json.loads(embedding_data)

            score = cosine_similarity(query_embedding, embedding)

            if score >= min_score:
                # Infer role from stored text prefix or source
                text = row[1]
                results.append({
                    "text": text,
                    "score": float(score),
                    "source": row[3],
                })

        # Sort by relevance
        results.sort(key=lambda x: x["score"], reverse=True)
        top = results[:top_k]

        logger.debug(f"🔍 conv_rag: {len(top)}/{len(rows)} turns matched for session {session_id}")
        return top

    except Exception as e:
        logger.error(f"❌ conversation_rag.retrieve_context failed: {e}")
        return []


def purge_session(session_id: int) -> int:
    """
    Remove all RAG entries for a session. Call when session is deleted
    so the RAG store doesn't accumulate orphaned conversation data.

    Args:
        session_id: Session to purge

    Returns:
        Number of rows deleted
    """
    from tools.rag.rag_utils import get_connection

    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM documents WHERE session_id = ?", (str(session_id),))
        deleted = cursor.rowcount
        conn.commit()
        logger.info(f"🗑️ Purged {deleted} RAG entries for session {session_id}")
        return deleted
    except Exception as e:
        logger.error(f"❌ conversation_rag.purge_session failed: {e}")
        return 0