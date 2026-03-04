"""
RAG Search Tool
Cosine similarity search over document embeddings, with optional reranking.

Pipeline:
  1. Embed the query with bge-large
  2. Score all documents by cosine similarity, filter by min_score
  3. Take top RERANK_CANDIDATES (20) for reranking, or top_k if reranker absent
  4. If bge-reranker-v2-m3 is available, rerank candidates and return top_k
  5. Fetch passage text from sessions.db (chunks table) via chunk_id

If bge-reranker-v2-m3 is not installed a single WARNING is emitted at import time
and search continues with cosine-only ranking.
"""

from typing import Dict, Any, List, Optional
from langchain_ollama import OllamaEmbeddings
import logging
import requests

from .rag_utils import load_rag_db, cosine_similarity

logger = logging.getLogger("mcp_server")

# Embeddings model — required; RAG is disabled upstream if this is unavailable
embeddings_model = OllamaEmbeddings(model="bge-large")

# Reranker config
RERANKER_MODEL      = "sam860/qwen3-reranker:0.6b-Q8_0"
RERANK_CANDIDATES   = 20      # How many cosine hits to pass to the reranker
OLLAMA_API_BASE     = "http://localhost:11434"

# Probe for reranker availability once at import time
_reranker_available: Optional[bool] = None


def _check_reranker() -> bool:
    """
    Return True if bge-reranker-v2-m3 is present in the local Ollama instance.
    Result is cached after the first call.
    """
    global _reranker_available
    if _reranker_available is not None:
        return _reranker_available

    try:
        resp = requests.get(f"{OLLAMA_API_BASE}/api/tags", timeout=3)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            # Match full name or with/without tag suffix
            _reranker_available = any(
                m == RERANKER_MODEL or m.startswith(RERANKER_MODEL.split(":")[0])
                for m in models
            )
        else:
            _reranker_available = False
    except Exception:
        _reranker_available = False

    if not _reranker_available:
        logger.warning(
            f"⚠️  Reranker model '{RERANKER_MODEL}' is not installed. "
            f"RAG search will use cosine similarity only. "
            f"Run `ollama pull {RERANKER_MODEL}` to enable reranking."
        )
    else:
        logger.info(f"✅ Reranker '{RERANKER_MODEL}' is available")

    return _reranker_available


def _rerank(query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Re-score candidates using bge-reranker-v2-m3 cross-encoder via the Ollama
    /api/embed endpoint. Returns candidates sorted by rerank score descending.

    Each candidate dict must have a 'text' field populated before calling this.
    Candidates with missing text are passed through with their original cosine score.
    """
    reranked = []

    for candidate in candidates:
        text = candidate.get("text", "")
        if not text:
            reranked.append(candidate)
            continue
        try:
            resp = requests.post(
                f"{OLLAMA_API_BASE}/api/embed",
                json={"model": RERANKER_MODEL, "input": [query, text]},
                timeout=20,
            )
            if resp.status_code == 200:
                embeddings = resp.json().get("embeddings", [])
                if len(embeddings) == 2:
                    rerank_score = cosine_similarity(embeddings[0], embeddings[1])
                    reranked.append({**candidate, "score": rerank_score, "reranked": True})
                else:
                    reranked.append(candidate)
            else:
                reranked.append(candidate)
        except Exception as e:
            logger.warning(f"⚠️ Reranker call failed for candidate: {e}")
            reranked.append(candidate)

    reranked.sort(key=lambda x: x["score"], reverse=True)
    return reranked


# Probe on module load so the warning appears at startup, not on first search
_check_reranker()


def rag_search(query: str, top_k: int = 5, min_score: float = 0.3) -> Dict[str, Any]:
    """
    Search the RAG database for relevant documents.

    Args:
        query:     The search query
        top_k:     Number of results to return (default: 5)
        min_score: Minimum cosine similarity threshold (default: 0.3)

    Returns:
        Dictionary with search results. Each result contains:
            id, text, score, metadata, reranked (bool)
    """
    from client.session_manager import SessionManager

    try:
        logger.info(f"🔍 Searching RAG for: '{query}'")

        db = load_rag_db()
        if not db:
            logger.warning("⚠️  RAG database is empty")
            return {
                "success": True,
                "query": query,
                "results": [],
                "message": "RAG database is empty",
            }

        # Embed the query
        query_embedding = embeddings_model.embed_query(query)

        # Cosine similarity pass — filter by min_score, take up to RERANK_CANDIDATES
        cosine_results = []
        for doc in db:
            score = cosine_similarity(query_embedding, doc["embedding"])
            if score >= min_score:
                cosine_results.append({
                    "id":       doc["id"],
                    "chunk_id": doc.get("chunk_id"),
                    "score":    float(score),
                    "metadata": doc["metadata"],
                    "reranked": False,
                })

        cosine_results.sort(key=lambda x: x["score"], reverse=True)
        candidates = cosine_results[:RERANK_CANDIDATES]

        if not candidates:
            logger.info("🔍 No results above similarity threshold")
            return {
                "success": True,
                "query": query,
                "results": [],
                "total_matches": 0,
                "returned": 0,
            }

        # Fetch passage text from sessions.db for all candidates
        session_manager = SessionManager()
        for candidate in candidates:
            chunk_id = candidate.get("chunk_id")
            if chunk_id is not None:
                candidate["text"] = session_manager.get_chunk(chunk_id) or ""
            else:
                candidate["text"] = ""
                logger.warning(f"⚠️ Document {candidate['id']} has no chunk_id — text unavailable")

        # Optional reranking pass
        if _check_reranker():
            logger.debug(f"🔀 Reranking {len(candidates)} candidates with {RERANKER_MODEL}")
            candidates = _rerank(query, candidates)

        top_results = candidates[:top_k]
        logger.info(f"✅ Returning {len(top_results)} results (reranked={_check_reranker()})")

        return {
            "success": True,
            "query": query,
            "results": top_results,
            "total_matches": len(cosine_results),
            "returned": len(top_results),
            "reranked": _check_reranker(),
        }

    except Exception as e:
        logger.error(f"❌ Error searching RAG: {e}")
        return {
            "success": False,
            "error": str(e),
            "query": query,
            "results": [],
        }