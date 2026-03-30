"""
Session State Module
====================
Provides per-session, scoped state that persists for the lifetime of a
WebSocket session without using any global mutable variables.

Design constraints:
- No shared state between sessions — each session_id gets its own isolated bag
- No hidden globals — the registry is explicitly passed through the call stack
- Typed domain slots for common cross-tool context (calendar, email, file)
- Generic _extras dict for one-off tool state that doesn't deserve a slot
- Automatic eviction of stale sessions to prevent unbounded memory growth

Usage
-----
In client.py (startup):
    from client.session_state import SessionStateRegistry
    session_state_registry = SessionStateRegistry()

In run_agent_wrapper (client.py):
    state = session_state_registry.get(current_session_id)
    result = await langgraph.run_agent(..., session_state=state)

In call_tools_with_stop_check (langgraph.py):
    # After a successful tool call, update session state from result
    session_state.update_from_tool_result(tool_name, result_dict)

    # Before a tool call that needs prior context, inject it
    tool_args = session_state.inject_into_args(tool_name, tool_args)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


# ─── Domain slots ─────────────────────────────────────────────────────────────
# Add new typed slots here as new tool categories are integrated.
# Keep them Optional so missing data is explicit, never a stale default.

@dataclass
class SessionState:
    """
    Isolated, session-scoped state bag.

    Typed slots cover the most common cross-tool context scenarios.
    _extras handles everything else without requiring a code change.
    """
    session_id: str

    # Timestamps
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    # ── Calendar ──────────────────────────────────────────────────────────────
    # Last calendar event fetched — so follow-up questions ("when does it end?")
    # don't require re-fetching.
    last_calendar_event: dict | None = None

    # ── Email ─────────────────────────────────────────────────────────────────
    # Last email thread referenced — enables "reply to that" without re-lookup.
    last_email_thread: dict | None = None

    # ── Files ─────────────────────────────────────────────────────────────────
    # Last file path opened — enables "summarize it" after "open X".
    last_file_path: str | None = None

    # ── Workflow preferences ──────────────────────────────────────────────────
    # Per-session user preferences set during a workflow, e.g.
    # {"response_language": "fr", "summary_style": "short"}
    workflow_prefs: dict = field(default_factory=dict)

    # ── Generic escape hatch ──────────────────────────────────────────────────
    # One-off tool state that doesn't deserve a typed slot yet.
    # Key convention: "<tool_name>.<field>" e.g. "rag_search.last_query"
    _extras: dict = field(default_factory=dict)

    # ─── Write helpers ────────────────────────────────────────────────────────

    def touch(self) -> None:
        """Update last_active timestamp."""
        self.last_active = time.time()

    def set(self, key: str, value: Any) -> None:
        """Write to the generic extras dict and touch the session."""
        self._extras[key] = value
        self.touch()

    def get(self, key: str, default: Any = None) -> Any:
        """Read from the generic extras dict."""
        return self._extras.get(key, default)

    def set_pref(self, key: str, value: Any) -> None:
        """Write a workflow preference."""
        self.workflow_prefs[key] = value
        self.touch()

    def get_pref(self, key: str, default: Any = None) -> Any:
        """Read a workflow preference."""
        return self.workflow_prefs.get(key, default)

    # ─── Automatic result ingestion ───────────────────────────────────────────

    def update_from_tool_result(self, tool_name: str, result: dict) -> None:
        """
        Inspect a successful tool result dict and populate the relevant typed
        slot automatically. Pattern-matched on result fields rather than
        tool_name so external MCP tools (Google Calendar, Gmail, etc.) are
        handled without hardcoding their names.

        Called by call_tools_with_stop_check in langgraph.py after every
        successful tool invocation.
        """
        if not isinstance(result, dict):
            return

        self.touch()

        # ── Calendar event ────────────────────────────────────────────────────
        # GCal tools return: {id, summary, start, end, attendees, ...}
        if any(k in result for k in ("eventId", "summary", "start", "end")):
            if result.get("eventId") or (result.get("start") and result.get("summary")):
                self.last_calendar_event = result

        # ── Email thread ──────────────────────────────────────────────────────
        # Gmail tools return: {threadId, messages, subject, ...}
        if any(k in result for k in ("threadId", "messageId")):
            self.last_email_thread = result

        # ── File path ─────────────────────────────────────────────────────────
        # read_file_tool returns: {success, file_name, content, ...}
        # Also catch any result with a "file_path" or "path" field.
        if result.get("success") and result.get("file_name"):
            # read_file_tool result — store inferred path
            self.last_file_path = result.get("file_path") or result.get("file_name")
        elif result.get("file_path"):
            self.last_file_path = result["file_path"]
        elif result.get("path") and isinstance(result.get("path"), str):
            self.last_file_path = result["path"]

        # ── Generic extras via tool_name prefix ───────────────────────────────
        # Always store the raw result under "<tool_name>.last_result" so
        # any tool's output is reachable via session_state.get("tool.last_result")
        # without requiring a typed slot.
        self._extras[f"{tool_name}.last_result"] = result

    def inject_into_args(self, tool_name: str, tool_args: dict) -> dict:
        """
        Optionally enrich tool_args with session context before a tool call.
        Only injects when the field is missing from args and a slot is populated.

        Called by call_tools_with_stop_check before every tool invocation.
        Returns the (potentially modified) args dict.
        """
        if not isinstance(tool_args, dict):
            return tool_args

        # Inject last_file_path into file-reading tools when no path given
        _FILE_TOOLS = {"read_file_tool_handler", "read_file_tool", "summarize_text_tool"}
        if tool_name in _FILE_TOOLS and self.last_file_path:
            if not tool_args.get("file_path") and not tool_args.get("text"):
                tool_args = dict(tool_args)
                tool_args["file_path"] = self.last_file_path

        return tool_args

    def summary(self) -> dict:
        """Return a lightweight summary of what's currently in session state."""
        return {
            "session_id":           self.session_id,
            "last_active":          round(self.last_active, 1),
            "has_calendar_event":   self.last_calendar_event is not None,
            "has_email_thread":     self.last_email_thread is not None,
            "last_file_path":       self.last_file_path,
            "workflow_prefs":       self.workflow_prefs,
            "extras_keys":          list(self._extras.keys()),
        }


# ─── Registry ─────────────────────────────────────────────────────────────────

class SessionStateRegistry:
    """
    Global index of per-session state objects.

    This is NOT shared global state — it is an index. The mutable state
    lives inside individual SessionState instances, each keyed by session_id.
    The registry is instantiated once in client.py and passed explicitly
    through the call stack.
    """

    def __init__(self, max_age_seconds: int = 3600):
        self._states: dict[str, SessionState] = {}
        self._max_age = max_age_seconds

    def get(self, session_id: str) -> SessionState:
        """
        Return the SessionState for this session, creating it if needed.
        Also evicts stale sessions opportunistically on each access.
        """
        self._evict_stale()
        if session_id not in self._states:
            self._states[session_id] = SessionState(session_id=session_id)
        return self._states[session_id]

    def evict(self, session_id: str) -> None:
        """Explicitly remove state for a session (on delete or new_session)."""
        self._states.pop(session_id, None)

    def _evict_stale(self) -> None:
        """Remove sessions idle longer than max_age_seconds."""
        cutoff = time.time() - self._max_age
        stale = [sid for sid, s in self._states.items() if s.last_active < cutoff]
        for sid in stale:
            del self._states[sid]

    def active_count(self) -> int:
        return len(self._states)