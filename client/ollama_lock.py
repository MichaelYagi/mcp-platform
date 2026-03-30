"""
Global asyncio lock for serializing Ollama requests.
Import this wherever you need exclusive access to Ollama —
summarize_direct, conversation_rag, etc.
"""
import asyncio

# One lock for the entire process. Any code that calls Ollama
# should acquire this before making a request.
_ollama_lock: asyncio.Lock | None = None

def get_ollama_lock() -> asyncio.Lock:
    """Return the process-wide Ollama lock, creating it if needed."""
    global _ollama_lock
    if _ollama_lock is None:
        _ollama_lock = asyncio.Lock()
    return _ollama_lock