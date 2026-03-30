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

def explain_simplified(concept: str) -> dict:
    prompt = (
        f"Explain the concept '{concept}' at three levels. "
        f"Respond in JSON with exactly these keys: "
        f"\"analogy\", \"simple_explanation\", \"technical_definition\". "
        f"No extra keys, no markdown, no preamble — raw JSON only."
    )

    import os, json
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")

    try:
        resp = httpx.post(
            f"{ollama_url}/api/generate",
            json={"model": _get_model(), "prompt": prompt, "stream": False},
            timeout=120.0,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "").strip()
        parsed = json.loads(raw)
        return {
            "concept":              concept,
            "analogy":              parsed.get("analogy", ""),
            "simple_explanation":   parsed.get("simple_explanation", ""),
            "technical_definition": parsed.get("technical_definition", ""),
        }
    except Exception as e:
        logger.warning(f"explain_simplified failed: {e}")
        return {
            "concept":              concept,
            "analogy":              "",
            "simple_explanation":   "",
            "technical_definition": "",
            "error":                str(e),
        }