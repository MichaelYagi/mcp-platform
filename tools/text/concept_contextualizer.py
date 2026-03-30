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

def concept_contextualizer(concept: str) -> dict:
    prompt = (
        f"Provide big-picture context for the concept '{concept}'. "
        f"Respond in JSON with exactly these keys: "
        f"\"why_it_matters\", \"problem_it_solves\", \"how_it_fits_in\", \"related_concepts\". "
        f"\"related_concepts\" should be a list of 3-5 strings. "
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
            "concept":          concept,
            "why_it_matters":   parsed.get("why_it_matters", ""),
            "problem_it_solves": parsed.get("problem_it_solves", ""),
            "how_it_fits_in":   parsed.get("how_it_fits_in", ""),
            "related_concepts": parsed.get("related_concepts", []),
        }
    except Exception as e:
        logger.warning(f"concept_contextualizer failed: {e}")
        return {
            "concept":           concept,
            "why_it_matters":    "",
            "problem_it_solves": "",
            "how_it_fits_in":    "",
            "related_concepts":  [],
            "error":             str(e),
        }