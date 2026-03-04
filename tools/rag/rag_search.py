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

from .rag_utils import load_rag_db, cosine_similarity

logger = logging.getLogger("mcp_server")

# Embeddings model — required; RAG is disabled upstream if this is unavailable
embeddings_model = OllamaEmbeddings(model="bge-large")

# Reranker config
# Uses sentence-transformers CrossEncoder — runs locally on CPU, no Ollama needed.
# Install: pip install sentence-transformers
# Model downloads automatically on first use (~80MB, cached in ~/.cache/huggingface)
RERANKER_MODEL    = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_CANDIDATES = 20      # How many cosine hits to pass to the reranker

# Probe for reranker availability once at import time
_reranker_available: Optional[bool] = None
_cross_encoder = None        # Lazy-loaded CrossEncoder instance


def _check_reranker() -> bool:
    """
    Return True if sentence-transformers is installed and CrossEncoder loads.
    Result is cached after the first call. Model is lazy-loaded on first use.
    """
    global _reranker_available
    if _reranker_available is not None:
        return _reranker_available

    try:
        from sentence_transformers import CrossEncoder  # noqa: F401
        _reranker_available = True
        logger.info(f"✅ Reranker available: {RERANKER_MODEL} (sentence-transformers)")
    except ImportError:
        _reranker_available = False
        logger.warning(
            "⚠️  sentence-transformers not installed — RAG search will use cosine similarity only. "
            "Run `pip install sentence-transformers` to enable reranking."
        )

    return _reranker_available


def _get_cross_encoder():
    """Lazy-load the CrossEncoder model (downloads once, cached locally)."""
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder
        logger.info(f"⏳ Loading cross-encoder model: {RERANKER_MODEL}")
        _cross_encoder = CrossEncoder(RERANKER_MODEL)
        logger.info(f"✅ Cross-encoder loaded: {RERANKER_MODEL}")
    return _cross_encoder


def _rerank(query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Re-score candidates using a sentence-transformers CrossEncoder.
    Scores all candidates in a single batched call — fast on CPU.

    Falls back to original cosine order on any error.
    Each candidate dict must have a 'text' field populated before calling this.
    """
    # Separate candidates with text from those without
    indexed  = [(i, c) for i, c in enumerate(candidates) if c.get("text")]
    no_text  = [(i, c) for i, c in enumerate(candidates) if not c.get("text")]

    if not indexed:
        return candidates

    try:
        cross_encoder = _get_cross_encoder()

        # Score all query-passage pairs in one batch
        pairs  = [(query, c.get("text")) for _, c in indexed]
        scores = cross_encoder.predict(pairs)   # returns numpy array of floats

        reranked = []
        for score, (_, candidate) in zip(scores, indexed):
            reranked.append({**candidate, "score": float(score), "reranked": True})

        # Append no-text candidates at the tail with their original cosine score
        for _, candidate in no_text:
            reranked.append(candidate)

        reranked.sort(key=lambda x: x["score"], reverse=True)
        return reranked

    except Exception as e:
        logger.warning(f"⚠️ Reranker failed ({e}), falling back to cosine order")
        return candidates


# Probe on module load so the warning appears at startup, not on first search.
# CrossEncoder itself is lazy-loaded on first actual rerank call.
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

        # Optional reranking pass — use cached flag, avoid repeated availability checks
        reranker_on = _reranker_available
        if reranker_on:
            logger.debug(f"🔀 Reranking {len(candidates)} candidates with {RERANKER_MODEL}")
            candidates = _rerank(query, candidates)

        top_results = candidates[:top_k]
        logger.info(f"✅ Returning {len(top_results)} results (reranked={reranker_on})")

        return {
            "success": True,
            "query": query,
            "results": top_results,
            "total_matches": len(cosine_results),
            "returned": len(top_results),
            "reranked": reranker_on,
        }

    except Exception as e:
        logger.error(f"❌ Error searching RAG: {e}")
        return {
            "success": False,
            "error": str(e),
            "query": query,
            "results": [],
        }