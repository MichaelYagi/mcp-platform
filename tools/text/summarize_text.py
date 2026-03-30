import os
import httpx
import logging
from pathlib import Path
from .utils import load_text
from .split_text import split_text

logger = logging.getLogger("mcp_server")

# How much text to feed the LLM in a single summarization call.
# 8000 chars ~ 2000 tokens — comfortably fits qwen2.5 context while
# leaving room for the summary itself.
MAX_CHARS = 8000

STYLE_INSTRUCTIONS = {
    "brief":    "Write a single sentence.",
    "short":    "Write 2-3 sentences covering the key points.",
    "medium": (
        "Write a structured summary of 2-3 paragraphs. "
        "Cover the main topic, key details, and any notable facts or outcomes."
    ),
    "detailed": (
        "Write a comprehensive summary with multiple paragraphs. "
        "Cover all major sections, key people or events, important details, "
        "and relevant context."
    ),
}

def _get_model() -> str:
    try:
        model_file = Path(__file__).resolve().parents[2] / "client" / "last_model.txt"
        return model_file.read_text(encoding="utf-8").strip()
    except Exception as e:
        logger.warning(f"Could not read last_model.txt: {e}")
        return "qwen2.5:latest"

def _call_ollama(prompt: str, timeout: float = 120.0) -> str:
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    try:
        resp = httpx.post(
            f"{ollama_url}/api/generate",
            json={"model": _get_model(), "prompt": prompt, "stream": False},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except Exception as e:
        logger.warning(f"_call_ollama failed: {e}")
        return ""

def summarize_text(text: str | None = None,
                   file_path: str | None = None,
                   style: str = "medium") -> dict:

    full_text = load_text(text, file_path)
    chunks = split_text(full_text)["chunks"]
    instruction = STYLE_INSTRUCTIONS.get(style, STYLE_INSTRUCTIONS["medium"])

    # Concatenate chunks up to MAX_CHARS for a single-pass summary.
    # This avoids per-chunk LLM calls that compete with the client model.
    combined = ""
    for chunk in chunks:
        if len(combined) + len(chunk) > MAX_CHARS:
            break
        combined += chunk

    truncated = len(combined) < len(full_text)

    prompt = (
        f"Summarize the following text. {instruction} "
        f"Return only the summary, no preamble.\n\n{combined}"
    )

    logger.info(
        f"summarize_text: {len(full_text)} chars, "
        f"{len(chunks)} chunks, feeding {len(combined)} chars to LLM "
        f"({'truncated' if truncated else 'full'}), style={style!r}"
    )

    summary = _call_ollama(prompt, timeout=120.0)

    if not summary:
        # Fallback: return the beginning of the text
        summary = full_text[:500]

    return {
        "summary": summary,
        "source": file_path or "text",
        "original_length": len(full_text),
        "chunks_processed": len(chunks),
        "truncated": truncated,
        "style": style,
    }