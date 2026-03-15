import os
import re
import math
import requests
import logging
from collections import Counter
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
#  PLEX API HELPERS
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


def _plex_download(path: str) -> str:
    """
    Download subtitle file or stream as raw text.
    """
    plex_url   = os.getenv("PLEX_URL")
    plex_token = os.getenv("PLEX_TOKEN")
    if not plex_url or not plex_token:
        raise RuntimeError("PLEX_URL and PLEX_TOKEN must be set in environment variables.")

    if not path.startswith("/"):
        path = "/" + path

    url = f"{plex_url}{path}"

    params = {
        "X-Plex-Token": plex_token
    }

    # IMPORTANT: No Accept: application/json here
    headers = {
        "Accept": "*/*"
    }

    r = requests.get(url, headers=headers, params=params, timeout=10)
    r.raise_for_status()
    return r.text

# ------------------------------------------------------------
#  SUBTITLE PARSING (SRT/VTT)
# ------------------------------------------------------------
TIMECODE_RE = re.compile(
    r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})"
)

def _parse_srt(text: str) -> List[Dict[str, Any]]:
    """
    Parse SRT subtitle text into a list of entries:
    { "start": seconds, "end": seconds, "text": "..." }
    """
    entries = []
    blocks = text.split("\n\n")

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue

        # Find timecode line
        match = TIMECODE_RE.search(block)
        if not match:
            continue

        start_tc, end_tc = match.groups()

        def tc_to_seconds(tc: str) -> float:
            h, m, s_ms = tc.split(":")
            s, ms = s_ms.split(",")
            return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

        start = tc_to_seconds(start_tc)
        end = tc_to_seconds(end_tc)

        # Subtitle text (skip index + timecode)
        text_lines = []
        for line in lines:
            if "-->" in line:
                continue
            if line.strip().isdigit():
                continue
            text_lines.append(line.strip())

        entry_text = " ".join(text_lines).strip()
        if entry_text:
            entries.append({"start": start, "end": end, "text": entry_text})

    return entries


# ------------------------------------------------------------
#  TEXT PROCESSING + TF-IDF
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


def _build_tfidf_vectors(chunks: List[Dict[str, Any]]):
    """
    Build TF-IDF vectors for subtitle chunks.
    """
    corpus_tokens = []
    df_counter = Counter()

    for chunk in chunks:
        tokens = _tokenize(chunk["text"])
        corpus_tokens.append(tokens)
        for term in set(tokens):
            df_counter[term] += 1

    num_docs = len(chunks)
    idf = {term: math.log((num_docs + 1) / (df + 1)) + 1.0 for term, df in df_counter.items()}

    vectors = []
    for tokens in corpus_tokens:
        tf_counter = Counter(tokens)
        max_tf = max(tf_counter.values()) if tf_counter else 1
        vec = {term: (tf / max_tf) * idf.get(term, 0.0) for term, tf in tf_counter.items()}
        vectors.append(vec)

    return idf, vectors


def _vectorize_query(query: str, idf: Dict[str, float]):
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
#  MAIN TOOL
# ------------------------------------------------------------
def scene_locator(media_id: str, query: str, limit: int = 5) -> Dict[str, Any]:
    """
    Locate semantically relevant scenes within a Plex media item using subtitle files.
    This function MUST NEVER be called with a movie title. It MUST ONLY be called with a Plex ratingKey. If the user provides a title instead of a ratingKey, ALWAYS call semantic_media_search first to resolve the correct ratingKey.
    """

    # Validate that media_id looks like a ratingKey (should be numeric or mostly numeric)
    if not media_id.isdigit() and not media_id.replace('-', '').isdigit():
        logger.error(f"Invalid media_id {media_id}")
        return {
            "error": f"Invalid media_id '{media_id}'. Expected a Plex ratingKey (numeric ID), not a title. "
                     f"Please call semantic_media_search_text first to get the correct ratingKey."
        }

    logger.info(f"In scene locator for media {media_id}")

    # 1. Get metadata for the media item
    meta = _plex_get(f"/library/metadata/{media_id}")
    media = meta["MediaContainer"]["Metadata"][0]

    logger.info(f"[scene_locator] Get metadata for the media item {media_id}")

    # 2. Find subtitle parts
    parts = media.get("Media", [])
    subtitle_url = None

    logger.info(f"[scene_locator] Find subtitle parts for the media item {media_id}")

    for media_part in parts:
        for part in media_part.get("Part", []):
            for stream in part.get("Stream", []):
                if stream.get("streamType") == 3:  # subtitle
                    logger.info(f"[scene_locator] Found subtitles for  {media_id}")
                    subtitle_url = stream.get("key")
                    break

    logger.info(f"[scene_locator] parts loop completed for the media item {media_id}")

    if not subtitle_url:
        logger.info(f"[scene_locator] Could not find subtitle url for media {media_id}")
        return {"scenes": []}

    logger.info(f"[scene_locator] Using subtitle URL: {subtitle_url}")

    # 3. Download subtitle file
    subtitle_text = _plex_download(subtitle_url)

    # 4. Parse SRT
    entries = _parse_srt(subtitle_text)
    if not entries:
        return {"scenes": []}

    # 5. Chunk subtitles (combine every ~10 seconds)
    chunks = []
    current = {"start": entries[0]["start"], "end": entries[0]["end"], "text": entries[0]["text"]}

    for entry in entries[1:]:
        if entry["start"] - current["end"] <= 10:
            current["end"] = entry["end"]
            current["text"] += " " + entry["text"]
        else:
            chunks.append(current)
            current = {"start": entry["start"], "end": entry["end"], "text": entry["text"]}

    chunks.append(current)

    # 6. Build TF-IDF vectors
    idf, vectors = _build_tfidf_vectors(chunks)

    # 7. Vectorize query
    query_vec = _vectorize_query(query, idf)

    # 8. Score chunks
    scored = []
    for chunk, vec in zip(chunks, vectors):
        score = _cosine(query_vec, vec)
        if score > 0:
            scored.append({
                "start": chunk["start"],
                "end": chunk["end"],
                "text": chunk["text"],
                "score": score
            })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return {"scenes": scored[:limit]}