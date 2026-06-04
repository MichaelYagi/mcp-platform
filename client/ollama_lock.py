"""
Global asyncio locks for serializing Ollama requests.

Two levels:
  get_background_lock()  — serializes background (non-user-facing) LLM calls so
                           memory consolidation, scheduler post-processing, etc.
                           do not pile up on the GPU simultaneously.
  background_ollama_call() — convenience async context manager wrapping the lock.

Foreground (user-facing) queries do NOT acquire either lock; they hit Ollama
directly.  The background lock only prevents multiple background tasks from
running concurrently — it does not block foreground work.
"""
import asyncio
from contextlib import asynccontextmanager

# Legacy name — kept for any code that still imports it.
_ollama_lock: asyncio.Lock | None = None

def get_ollama_lock() -> asyncio.Lock:
    """Return the legacy process-wide Ollama lock (kept for back-compat)."""
    global _ollama_lock
    if _ollama_lock is None:
        _ollama_lock = asyncio.Lock()
    return _ollama_lock


# Background-task lock — serialises memory consolidation, scheduler LLM calls, etc.
_background_lock: asyncio.Lock | None = None

def get_background_lock() -> asyncio.Lock:
    """Return the background-task lock."""
    global _background_lock
    if _background_lock is None:
        _background_lock = asyncio.Lock()
    return _background_lock


@asynccontextmanager
async def background_ollama_call():
    """Async context manager for background LLM calls.

    Acquire before any non-user-facing Ollama request to prevent multiple
    background tasks from contending for the GPU at the same time.
    """
    async with get_background_lock():
        yield
