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

def merge_summaries(summaries: list[str], style: str = "medium") -> dict:
    style_instructions = {
        "short":    "Write 2-3 sentences.",
        "medium":   "Write a coherent paragraph.",
        "detailed": "Write a thorough multi-paragraph summary.",
    }
    instruction = style_instructions.get(style, "Write a coherent paragraph.")
    combined = "\n\n".join(f"- {s}" for s in summaries)
    prompt = (
        f"Combine these summaries into a single cohesive summary. {instruction} "
        f"Return only the merged summary, no preamble.\n\n{combined[:4000]}"
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
        merged = resp.json().get("response", "").strip()
        return {
            "merged_summary":      merged,
            "input_count":         len(summaries),
            "total_input_length":  sum(len(s) for s in summaries),
            "output_length":       len(merged),
        }
    except Exception as e:
        logger.warning(f"merge_summaries failed: {e}")
        return {
            "merged_summary":     combined[:300],
            "input_count":        len(summaries),
            "total_input_length": sum(len(s) for s in summaries),
            "error":              str(e),
        }