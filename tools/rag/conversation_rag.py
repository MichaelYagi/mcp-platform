"""
Conversation RAG Bridge
Handles storing and retrieving conversation turns in the RAG store,
scoped by session_id. Completely separate from document/external RAG.

Store:    store_turn(session_id, role, content, message_id)
Retrieve: retrieve_context(session_id, query, top_k) -> List[Dict]

Turn text is never stored in rag_database.db. The message_id FK into
sessions.db messages.id is stored instead, and text is fetched directly
by primary key at retrieval time — no positional indexing, no drift.
"""
import asyncio
import logging
import time
import uuid
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("mcp_server")

_embed_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="conv_rag")

ROLE_PREFIX = {
    "user":      "User said: ",
    "assistant": "Assistant said: ",
}

MAX_TURN_CHARS = 1200
_embeddings_model = None

def _get_embeddings_model():
    global _embeddings_model
    if _embeddings_model is None:
        from langchain_ollama import OllamaEmbeddings
        _embeddings_model = OllamaEmbeddings(model="bge-large")
    return _embeddings_model

def _embed_text(text: str) -> Optional[List[float]]:
    """Generate embedding for a single text string. Returns None on failure."""
    try:
        return _get_embeddings_model().embed_query(text)
    except Exception as e:
        logger.error(f"❌ conversation_rag: embedding failed: {e}")
        return None


def store_turn(session_id: int, role: str, content: str, message_id: int) -> bool:
    """
    Embed a single conversation turn and store it in the RAG database,
    tagged with session_id and message_id for reliable text retrieval.

    Text is NOT stored in rag_database.db. At retrieval time, message_id
    is used to fetch content directly from sessions.db messages table.

    Args:
        session_id: Integer session ID from session_manager
        role:       'user' or 'assistant'
        content:    Message text (used only for embedding, not persisted here)
        message_id: messages.id PK from sessions.db — stored as FK in RAG DB

    Returns:
        True if stored successfully, False otherwise
    """
    from tools.rag.rag_utils import get_connection

    if not content or not content.strip():
        return False

    prefix = ROLE_PREFIX.get(role, f"{role}: ")
    text_to_embed = prefix + content[:MAX_TURN_CHARS]

    embedding = _embed_text(text_to_embed)
    if embedding is None:
        return False

    import numpy as np
    embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()

    doc_id = str(uuid.uuid4())
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO documents (id, embedding, source, session_id, message_id)
            VALUES (?, ?, ?, ?, ?)
        """, (
            doc_id,
            embedding_bytes,
            "conversation",
            str(session_id),
            message_id,
        ))
        conn.commit()
        logger.debug(
            f"💬 Stored {role} turn for session {session_id} "
            f"(doc {doc_id[:8]}, message_id={message_id})"
        )
        return True

    except Exception as e:
        logger.error(f"❌ conversation_rag.store_turn failed: {e}")
        return False
    finally:
        conn.close()


def store_turn(session_id: int, role: str, content: str, message_id: int) -> bool:
    """
    Embed a single conversation turn and store it in the RAG database,
    tagged with session_id and message_id for reliable text retrieval.

    Text is NOT stored in rag_database.db. At retrieval time, message_id
    is used to fetch content directly from sessions.db messages table.

    Args:
        session_id: Integer session ID from session_manager
        role:       'user' or 'assistant'
        content:    Message text (used only for embedding, not persisted here)
        message_id: messages.id PK from sessions.db — stored as FK in RAG DB

    Returns:
        True if stored successfully, False otherwise
    """
    import time as _time
    from pathlib import Path as _Path
    from tools.rag.rag_utils import get_connection

    if not content or not content.strip():
        return False

    # Wait for any active Ollama generation (e.g. summarize_direct) to finish
    # before attempting to embed — Ollama serializes requests so we'd just
    # queue behind it anyway, but this makes the ordering explicit and logged.
    lock_file = _Path(__file__).resolve().parents[2] / "client" / ".ollama_busy"
    waited = 0
    while lock_file.exists() and waited < 120:
        logger.debug(f"conv_rag: Ollama busy, waiting... ({waited}s)")
        _time.sleep(1)
        waited += 1
    if waited > 0:
        logger.info(f"conv_rag: waited {waited}s for Ollama to free up before embedding")

    prefix = ROLE_PREFIX.get(role, f"{role}: ")
    text_to_embed = prefix + content[:MAX_TURN_CHARS]

    embedding = _embed_text(text_to_embed)
    if embedding is None:
        return False

    import numpy as np
    embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()

    doc_id = str(uuid.uuid4())
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO documents (id, embedding, source, session_id, message_id)
            VALUES (?, ?, ?, ?, ?)
        """, (
            doc_id,
            embedding_bytes,
            "conversation",
            str(session_id),
            message_id,
        ))
        conn.commit()
        logger.debug(
            f"💬 Stored {role} turn for session {session_id} "
            f"(doc {doc_id[:8]}, message_id={message_id})"
        )
        return True

    except Exception as e:
        logger.error(f"❌ conversation_rag.store_turn failed: {e}")
        return False
    finally:
        conn.close()

def retrieve_context(
    session_id: int,
    query: str,
    top_k: int = 5,
    min_score: float = 0.35,
) -> List[Dict[str, Any]]:
    """
    Retrieve the most semantically relevant past turns from this session.

    Does NOT return full history — only what's relevant to the current query.
    Text is fetched from sessions.db by message_id (direct PK lookup).

    Args:
        session_id:  Integer session ID to scope retrieval
        query:       Current user message (used as query embedding)
        top_k:       Maximum results to return
        min_score:   Minimum cosine similarity threshold

    Returns:
        List of dicts with keys: text, role, score, reranked
        Sorted by score descending.
    """
    from tools.rag.rag_utils import get_connection, cosine_similarity
    from tools.rag.rag_search import RERANKER_MODEL, RERANK_CANDIDATES, _reranker_available, _check_reranker, _rerank
    from client.session_manager import SessionManager

    if not query or not query.strip():
        return []

    try:
        query_embedding = _embed_text(query)
        if query_embedding is None:
            return []

        conn = get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, embedding, message_id
                FROM documents
                WHERE session_id = ?
                ORDER BY created_at ASC
            """, (str(session_id),))
            rows = cursor.fetchall()
        finally:
            conn.close()

        if not rows:
            logger.debug(f"💬 No RAG turns found for session {session_id}")
            return []

        import numpy as np

        # Cosine pass
        cosine_results = []
        for row in rows:
            embedding_data = row[1]
            if isinstance(embedding_data, bytes):
                embedding = np.frombuffer(embedding_data, dtype=np.float32).tolist()
            else:
                import json
                embedding = json.loads(embedding_data)

            score = cosine_similarity(query_embedding, embedding)
            if score >= min_score:
                cosine_results.append({
                    "message_id": row[2],
                    "score":      float(score),
                    "reranked":   False,
                })

        cosine_results.sort(key=lambda x: x["score"], reverse=True)
        candidates = cosine_results[:RERANK_CANDIDATES]

        if not candidates:
            return []

        # Fetch turn text from sessions.db by message_id (direct PK lookup — no positional drift)
        session_manager = SessionManager()
        populated = []
        for candidate in candidates:
            message_id = candidate.get("message_id")
            if message_id is None:
                logger.warning(f"⚠️ RAG turn has no message_id for session {session_id} — skipping")
                continue
            msg = session_manager.get_message_by_id(message_id)
            if not msg or not msg.get("text"):
                logger.debug(f"💬 message_id={message_id} not found in sessions.db (may have been trimmed)")
                continue
            populated.append({
                **candidate,
                "text": msg["text"],
                "role": msg["role"],
            })

        if not populated:
            return []

        # Optional reranking pass — use cached flag to avoid repeated availability checks
        reranker_on = _reranker_available
        if reranker_on:
            logger.debug(f"🔀 Reranking {len(populated)} conversation candidates with {RERANKER_MODEL}")
            populated = _rerank(query, populated)

        top = populated[:top_k]
        logger.debug(
            f"🔍 conv_rag: {len(top)}/{len(rows)} turns matched for session {session_id} "
            f"(reranked={reranker_on})"
        )
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

    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM documents WHERE session_id = ?", (str(session_id),))
        deleted = cursor.rowcount
        conn.commit()
        logger.info(f"🗑️ Purged {deleted} RAG entries for session {session_id}")
        return deleted
    except Exception as e:
        logger.error(f"❌ conversation_rag.purge_session failed: {e}")
        return 0
    finally:
        conn.close()