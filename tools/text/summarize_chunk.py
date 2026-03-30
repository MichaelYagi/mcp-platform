import httpx
import logging
from pathlib import Path

logger = logging.getLogger("mcp_server")

def _get_model() -> str:
    try:
        model_file = Path(__file__).resolve().parents[2] / "client" / "last_model.txt"
        return model_file.read_text(encoding="utf-8").strip()
    except Exception as e:
        logger.warning(f"Could not read last_model.txt: {e}")
        return "qwen2.5:latest"

def summarize_chunk(chunk: str, style: str = "short") -> dict:
    style_instructions = {
        "brief":    "Write a single sentence.",
        "short":    "Write 1-2 sentences.",
        "medium":   "Write 3-5 sentences.",
        "detailed": "Write a thorough paragraph.",
    }
    instruction = style_instructions.get(style, "Write 1-2 sentences.")
    prompt = (
        f"Summarize this text segment. {instruction} "
        f"Return only the summary, no preamble.\n\n{chunk[:4000]}"
    )

    import os
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")

    try:
        resp = httpx.post(
            f"{ollama_url}/api/generate",
            json={"model": _get_model(), "prompt": prompt, "stream": False},
            timeout=120.0,
        )
        resp.raise_for_status()
        summary = resp.json().get("response", "").strip()
        return {
            "summary":          summary,
            "style_used":       style,
            "original_length":  len(chunk),
            "summary_length":   len(summary),
            "compression_ratio": round(len(summary) / max(len(chunk), 1), 2),
        }
    except Exception as e:
        logger.warning(f"summarize_chunk failed: {e}")
        return {
            "summary":          chunk[:300],
            "style_used":       style,
            "original_length":  len(chunk),
            "error":            str(e),
        }