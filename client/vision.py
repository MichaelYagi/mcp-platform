"""
Shared Ollama vision-model helper.
All vision inference in client.py and langgraph.py goes through call_vision_model().
"""
import os
import httpx


def _dedup_sentences(text: str) -> str:
    """Trim repetition loops — if a sentence appears 3+ times, cut before it."""
    if not text:
        return text
    sents = text.split(". ")
    seen: dict = {}
    cut = len(sents)
    for i, s in enumerate(sents):
        key = s.strip().lower()[:60]
        if not key:
            continue
        seen[key] = seen.get(key, 0) + 1
        if seen[key] >= 3:
            cut = i
            break
    if cut < len(sents):
        return ". ".join(sents[:cut]).rstrip(".") + "."
    return text


async def call_vision_model(b64: str, prompt: str, num_predict: int = 300) -> str:
    """
    POST b64-encoded image + prompt to the Ollama vision model.

    Returns the response text with repetition loops trimmed.
    Raises httpx.HTTPError on network/HTTP failure.
    """
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    model = os.getenv("OLLAMA_VISION_MODEL", "qwen3-vl:8b-instruct")

    async with httpx.AsyncClient(timeout=300.0) as hc:
        resp = await hc.post(
            f"{ollama_url}/api/chat",
            json={
                "model": model,
                "stream": False,
                "options": {"num_predict": num_predict, "repeat_penalty": 1.3, "temperature": 0.3},
                "messages": [{"role": "user", "content": prompt, "images": [b64]}],
            },
        )
        resp.raise_for_status()

    text = resp.json().get("message", {}).get("content", "").strip()
    return _dedup_sentences(text)
