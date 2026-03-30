import httpx
import logging
from pathlib import Path

logger = logging.getLogger("mcp_server")

_model_cache: str | None = None

def _get_model() -> str:
    global _model_cache
    if _model_cache is None:
        try:
            model_file = Path(__file__).resolve().parents[2] / "client" / "last_model.txt"
            _model_cache = model_file.read_text(encoding="utf-8").strip()
        except Exception as e:
            logger.warning(f"Could not read last_model.txt: {e}")
            _model_cache = "qwen2.5:latest"
    return _model_cache

def summarize_direct(text: str, style: str = "medium") -> dict:
    import os
    from pathlib import Path

    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    lock_file = Path(__file__).resolve().parents[2] / "client" / ".ollama_busy"

    style_instructions = {
        "brief":    "Write a single sentence.",
        "short":    "Write 1-2 sentences.",
        "medium":   "Write 3-5 sentences.",
        "detailed": "Write a thorough paragraph.",
    }
    instruction = style_instructions.get(style, "Write 3-5 sentences.")
    prompt = (
        f"Summarize the following text concisely. {instruction} "
        f"Return only the summary, no preamble.\n\n{text[:4000]}"
    )

    try:
        lock_file.touch()   # signal: Ollama is in use
        resp = httpx.post(
            f"{ollama_url}/api/generate",
            json={"model": _get_model(), "prompt": prompt, "stream": False},
            timeout=120.0,
        )
        resp.raise_for_status()
        summary = resp.json().get("response", "").strip()
        return {
            "summary":         summary,
            "style_used":      style,
            "original_length": len(text),
        }
    except Exception as e:
        logger.warning(f"summarize_direct failed: {e}")
        fallback = text[:500]
        last_period = fallback.rfind(".")
        if last_period > 100:
            fallback = fallback[:last_period + 1]
        return {
            "summary":         fallback,
            "style_used":      style,
            "original_length": len(text),
            "error":           str(e),
        }
    finally:
        lock_file.unlink(missing_ok=True)  # always release