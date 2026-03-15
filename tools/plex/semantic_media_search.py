import os
import math
import json
import threading
import requests
from collections import Counter
from typing import Dict, List, Any

_INDEX_CACHE_LOCK = threading.Lock()
_INDEX_CACHE = {
    "docs": None,
    "idf": None,
    "doc_vectors": None
}


# ------------------------------------------------------------
#  PLEX API WRAPPER (JSON FORCED)
# ------------------------------------------------------------
def _plex_get(path: str) -> Dict[str, Any]:
    """
    Fetch JSON from Plex, forcing JSON output and pagination.
    """
    plex_url   = os.getenv("PLEX_URL")
    plex_token = os.getenv("PLEX_TOKEN")
    if not plex_url or not plex_token:
        raise RuntimeError("PLEX_URL and PLEX_TOKEN must be set in environment variables.")

    url = f"{plex_url}{path}"

    params = {
        "X-Plex-Token": plex_token,
        "X-Plex-Container-Start": 0,
        "X-Plex-Container-Size": 500
    }

    headers = {
        "Accept": "application/json"
    }

    response = requests.get(url, headers=headers, params=params, timeout=10)
    response.raise_for_status()
    return response.json()


# ------------------------------------------------------------
#  FETCH ALL MOVIES + SHOWS
# ------------------------------------------------------------
def _fetch_all_media() -> List[Dict[str, Any]]:
    """
    Fetch all movies and shows from Plex using the JSON API.
    """
    sections = _plex_get("/library/sections")["MediaContainer"]["Directory"]

    media_items = []

    for section in sections:
        section_type = section.get("type")
        key = section.get("key")

        if section_type not in ("movie", "show"):
            continue

        data = _plex_get(f"/library/sections/{key}/all")
        items = data["MediaContainer"].get("Metadata", [])

        for item in items:
            media_items.append({
                "id": item.get("ratingKey"),
                "title": item.get("title", ""),
                "summary": item.get("summary", ""),
                "year": item.get("year"),
                "genres": [g["tag"] for g in item.get("Genre", [])],
                "cast": [r["tag"] for r in item.get("Role", [])],
                "tags": [t["tag"] for t in item.get("Tag", [])]
            })

    return media_items


# ------------------------------------------------------------
#  TOKENIZATION + TEXT BUILDING
# ------------------------------------------------------------
def _tokenize(text: str) -> List[str]:
    text = text.lower()
    tokens = []
    current = []

    for ch in text:
        if ch.isalnum():
            current.append(ch)
        else:
            if current:
                tokens.append("".join(current))
                current = []
    if current:
        tokens.append("".join(current))

    return tokens


def _build_doc_text(doc: Dict[str, Any]) -> str:
    parts = [
        doc.get("title") or "",
        doc.get("summary") or ""
    ]

    for field in ("genres", "cast", "tags"):
        values = doc.get(field)
        if isinstance(values, list):
            parts.extend(values)
        elif values:
            parts.append(str(values))

    return " ".join(parts)


# ------------------------------------------------------------
#  TF‑IDF INDEX BUILDING
# ------------------------------------------------------------
def _build_tfidf_index():
    with _INDEX_CACHE_LOCK:
        if _INDEX_CACHE["docs"] is not None:
            return

        docs = _fetch_all_media()

        corpus_tokens = []
        df_counter = Counter()

        for doc in docs:
            text = _build_doc_text(doc)
            tokens = _tokenize(text)
            corpus_tokens.append(tokens)
            for term in set(tokens):
                df_counter[term] += 1

        num_docs = len(docs)
        idf = {term: math.log((num_docs + 1) / (df + 1)) + 1.0 for term, df in df_counter.items()}

        doc_vectors = []
        for tokens in corpus_tokens:
            tf_counter = Counter(tokens)
            max_tf = max(tf_counter.values()) if tf_counter else 1
            vec = {term: (tf / max_tf) * idf.get(term, 0.0) for term, tf in tf_counter.items()}
            doc_vectors.append(vec)

        _INDEX_CACHE["docs"] = docs
        _INDEX_CACHE["idf"] = idf
        _INDEX_CACHE["doc_vectors"] = doc_vectors


# ------------------------------------------------------------
#  QUERY VECTOR + COSINE SIMILARITY
# ------------------------------------------------------------
def _vectorize_query(query: str, idf: Dict[str, float]) -> Dict[str, float]:
    tokens = _tokenize(query)
    tf_counter = Counter(tokens)
    if not tf_counter:
        return {}

    max_tf = max(tf_counter.values())
    return {term: (tf / max_tf) * idf.get(term, 0.0) for term, tf in tf_counter.items() if term in idf}


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0

    dot = sum(val * b.get(term, 0.0) for term, val in a.items())
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


# ------------------------------------------------------------
#  PUBLIC TOOL FUNCTION
# ------------------------------------------------------------
def semantic_media_search(query: str, limit: int = 10) -> Dict[str, Any]:
    """
    Semantic search over the Plex media index using pure-Python TF-IDF.
    Returns the top 'limit' matching media items with similarity scores.
    """
    if not query.strip():
        return {"results": []}

    if limit <= 0:
        limit = 10

    _build_tfidf_index()

    docs = _INDEX_CACHE["docs"]
    idf = _INDEX_CACHE["idf"]
    doc_vectors = _INDEX_CACHE["doc_vectors"]

    query_vec = _vectorize_query(query, idf)
    if not query_vec:
        return {"results": []}

    scored = []
    for doc, vec in zip(docs, doc_vectors):
        score = _cosine(query_vec, vec)
        if score > 0:
            scored.append({
                "id": doc["id"],
                "title": doc["title"],
                "summary": doc["summary"],
                "genres": doc["genres"],
                "year": doc["year"],
                "score": score
            })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return {"results": scored[:limit]}