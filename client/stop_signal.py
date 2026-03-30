"""
Global Stop Signal Module
Provides a simple global flag to stop long-running operations
Thread-safe implementation using threading.Event
"""

import logging
import threading
import time

logger = logging.getLogger("mcp_client")

# Thread-safe stop signal using Event
_STOP_EVENT = threading.Event()
_STOP_TIME = None  # Track when stop was requested


def request_stop():
    """Request that all operations stop at their next checkpoint"""
    global _STOP_TIME

    if not _STOP_EVENT.is_set():
        _STOP_EVENT.set()
        _STOP_TIME = time.time()
        logger.warning("🛑 STOP SIGNAL ACTIVATED - Operations will halt at next checkpoint")
    else:
        # Already requested - show how long ago
        elapsed = time.time() - _STOP_TIME if _STOP_TIME else 0
        logger.info(f"🛑 Stop already requested {elapsed:.1f}s ago - waiting for operation to complete...")


def clear_stop():
    """Clear the stop signal (call at start of operations)"""
    global _STOP_TIME

    if _STOP_EVENT.is_set():
        logger.info("✅ Stop signal cleared - ready for new operations")

    _STOP_EVENT.clear()
    _STOP_TIME = None


def is_stop_requested() -> bool:
    """
    Check if stop has been requested (thread-safe)

    Returns:
        bool: True if stop was requested, False otherwise
    """
    is_set = _STOP_EVENT.is_set()

    # Log every check at debug level for diagnostics
    if is_set:
        elapsed = time.time() - _STOP_TIME if _STOP_TIME else 0
        logger.debug(f"🛑 Stop check: TRUE (requested {elapsed:.1f}s ago)")

    return is_set


def check_stop_and_raise():
    """
    Check stop signal and raise an exception if requested.
    Use this in operations that can't return early gracefully.

    Raises:
        StopRequestedException: If stop was requested
    """
    if _STOP_EVENT.is_set():
        elapsed = time.time() - _STOP_TIME if _STOP_TIME else 0
        logger.warning(f"🛑 Stop detected after {elapsed:.1f}s - raising exception")
        raise StopRequestedException("Operation stopped by user")


def get_stop_status() -> dict:
    """
    Get detailed stop signal status (for debugging)

    Returns:
        dict: Status information including whether stop is active and timing
    """
    return {
        "is_stopped": _STOP_EVENT.is_set(),
        "stop_time": _STOP_TIME,
        "elapsed": time.time() - _STOP_TIME if _STOP_TIME else None
    }


class StopRequestedException(Exception):
    """Exception raised when a stop is requested"""
    pass