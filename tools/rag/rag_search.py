from typing import Dict, Any, List
from langchain_ollama import OllamaEmbeddings
import logging

from .rag_utils import load_rag_db, cosine_similarity

logger = logging.getLogger("mcp_server")

# Initialize embeddings model
embeddings_model = OllamaEmbeddings(model="bge-large")


def rag_search(query: str, top_k: int = 5, min_score: float = 0.3) -> Dict[str, Any]:
    """
    Search the RAG database for relevant documents.

    Args:
        query: The search query
        top_k: Number of results to return (default: 5)
        min_score: Minimum similarity score (default: 0.3)

    Returns:
        Dictionary with search results
    """
    try:
        logger.info(f"🔍 Searching RAG for: '{query}'")

        # Load database
        db = load_rag_db()

        if not db:
            logger.warning("⚠️  RAG database is empty")
            return {
                "success": True,
                "query": query,
                "results": [],
                "message": "RAG database is empty"
            }

        # Generate query embedding
        query_embedding = embeddings_model.embed_query(query)

        # Calculate similarities
        results = []
        for doc in db:
            score = cosine_similarity(query_embedding, doc["embedding"])

            if score >= min_score:
                results.append({
                    "id": doc["id"],
                    "text": doc["text"],
                    "score": float(score),
                    "metadata": doc["metadata"]
                })

        # Sort by score (highest first)
        results.sort(key=lambda x: x["score"], reverse=True)

        # Return top_k results
        top_results = results[:top_k]

        logger.info(f"✅ Found {len(top_results)} results")

        return {
            "success": True,
            "query": query,
            "results": top_results,
            "total_matches": len(results),
            "returned": len(top_results)
        }

    except Exception as e:
        logger.error(f"❌ Error searching RAG: {e}")
        return {
            "success": False,
            "error": str(e),
            "query": query,
            "results": []
        }