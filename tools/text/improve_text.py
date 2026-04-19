"""
Text improvement module — expand, improve, fix, shorten, rewrite with tone.
Calls Ollama directly for local inference.
"""
import os
import requests

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")

_SYSTEM = "You are an expert editor. Follow the instruction precisely. Return only the improved text — no preamble, no explanation, no quotes around the result."

_MODE_PROMPTS = {
    "expand": (
        "Expand the following text. Add detail, context, and depth while preserving "
        "the original meaning and tone. Make it richer and more thorough."
    ),
    "improve": (
        "Improve the following text for clarity, flow, and readability. Fix awkward "
        "phrasing, improve word choice, and ensure sentences read naturally. "
        "Preserve the original meaning and voice."
    ),
    "fix": (
        "Fix all grammar, spelling, punctuation, and capitalization errors in the "
        "following text. Make no other changes — do not rephrase or alter the content."
    ),
    "shorten": (
        "Shorten the following text. Remove redundancy, filler, and unnecessary detail "
        "while preserving all key information and the original meaning."
    ),
    "formal": (
        "Rewrite the following text in a formal, professional tone. Use precise language, "
        "avoid contractions and colloquialisms, and ensure it is appropriate for a "
        "business or academic context."
    ),
    "casual": (
        "Rewrite the following text in a casual, conversational tone. Make it friendly, "
        "relaxed, and easy to read — like you're talking to a friend."
    ),
}


def improve_text(text: str, mode: str, instruction: str = None) -> dict:
    """
    Improve text using Ollama.

    Args:
        text: The text to improve
        mode: One of expand, improve, fix, shorten, formal, casual, or custom
        instruction: Custom instruction (used when mode is 'custom')

    Returns:
        dict with result, mode, original_length, result_length
    """
    text = text.strip()
    if not text:
        return {"error": "text must not be empty"}

    mode = (mode or "improve").lower().strip()

    if mode == "custom":
        if not instruction:
            return {"error": "instruction is required when mode is 'custom'"}
        user_prompt = f"{instruction.strip()}\n\n{text}"
    elif mode in _MODE_PROMPTS:
        user_prompt = f"{_MODE_PROMPTS[mode]}\n\n{text}"
    else:
        valid = ", ".join(_MODE_PROMPTS.keys()) + ", custom"
        return {"error": f"Unknown mode '{mode}'. Valid modes: {valid}"}

    model = os.getenv("OLLAMA_MODEL", "qwen2.5:14b-instruct-q4_K_M")

    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": model,
                "stream": False,
                "options": {"temperature": 0.4},
                "messages": [
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user",   "content": user_prompt},
                ],
            },
            timeout=120,
        )
        resp.raise_for_status()
        result_text = resp.json().get("message", {}).get("content", "").strip()
    except requests.exceptions.ConnectionError:
        return {"error": "Could not connect to Ollama — is it running?"}
    except requests.exceptions.Timeout:
        return {"error": "Ollama timed out"}
    except Exception as e:
        return {"error": f"Ollama error: {e}"}

    return {
        "result":          result_text,
        "mode":            mode,
        "original_length": len(text),
        "result_length":   len(result_text),
    }