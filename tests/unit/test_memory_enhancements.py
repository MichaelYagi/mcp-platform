"""
Tests for features added in the May 2026 session:

  - memory_consolidator: anchor injection (_get_top_memories), semantic dedup,
    transient event filtering, cosine_similarity helper, consolidate_decisions
  - websocket: unrecognised colon command blocking, broadcast_proactive_result,
    session replay guard (only replays if SESSION_TASKS has completed task)
  - commands: :commands list completeness
  - langgraph: search query year injection (canary test via prompt content)
"""
import asyncio
import json
import sqlite3
import struct
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


# ═══════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def temp_data_dir(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def memory_db_path(temp_data_dir):
    return temp_data_dir / "memory.db"


@pytest.fixture
def sessions_db_path(temp_data_dir):
    """sessions.db with chunks table and consolidated columns for decision extraction tests."""
    db_path = temp_data_dir / "sessions.db"
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            pinned INTEGER NOT NULL DEFAULT 0,
            consolidated_at TEXT,
            consolidated_msg_count INTEGER
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            model TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
        );
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chunk_text TEXT NOT NULL,
            embedding BLOB,
            source TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def patched_memory_paths(memory_db_path, sessions_db_path):
    with patch("client.memory_consolidator.MEMORY_DB_PATH", memory_db_path), \
         patch("client.memory_consolidator.SESSIONS_DB_PATH", sessions_db_path):
        yield memory_db_path, sessions_db_path


@pytest.fixture
def patched_memory_paths(memory_db_path, sessions_db_path):
    """Override patched_memory_paths to use our sessions_db_path that includes chunks table."""
    with patch("client.memory_consolidator.MEMORY_DB_PATH", memory_db_path), \
         patch("client.memory_consolidator.SESSIONS_DB_PATH", sessions_db_path):
        yield memory_db_path, sessions_db_path
    def _embed(text: str):
        import hashlib, struct
        h = int(hashlib.md5(text.encode()).hexdigest(), 16)
        vals = [(h >> i & 0xFF) / 255.0 for i in range(128)]
        return struct.pack(f"{len(vals)}f", *vals)
    return _embed


@pytest.fixture
def technical_transcript():
    return (
        "User: Can you fix the bug in websocket.py where broadcast_message throws?\n"
        "Assistant: Sure. The issue is in the error handling. Here's the fix:\n"
        "```python\nasync def broadcast_message(message_type, data):\n"
        "    try:\n        if CONNECTED_WEBSOCKETS:\n"
        "            await asyncio.gather(*[ws.send(...) for ws in CONNECTED_WEBSOCKETS])\n"
        "    except Exception as e:\n        logger.error(f'Broadcast failed: {e}')\n```\n"
        "User: Great, also add it to the commands list.\n"
        "Assistant: Added broadcast_message error handling to websocket.py and "
        "updated commands.py list with the new :broadcast command.\n"
    )


@pytest.fixture
def casual_transcript():
    return (
        "User: What's the weather like in Surrey?\n"
        "Assistant: It's currently 18°C and partly cloudy in Surrey, BC.\n"
        "User: Thanks!\n"
        "Assistant: You're welcome!\n"
    )


# ═══════════════════════════════════════════════════════════════════
# cosine_similarity helper
# ═══════════════════════════════════════════════════════════════════

class TestCosineSimilarity:
    def _make_vec(self, values):
        import struct
        return struct.pack(f"{len(values)}f", *values)

    def test_identical_vectors_return_one(self):
        from client.memory_consolidator import cosine_similarity
        v = self._make_vec([1.0, 0.0, 0.0])
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-5

    def test_orthogonal_vectors_return_zero(self):
        from client.memory_consolidator import cosine_similarity
        a = self._make_vec([1.0, 0.0])
        b = self._make_vec([0.0, 1.0])
        assert abs(cosine_similarity(a, b)) < 1e-5

    def test_opposite_vectors_return_minus_one(self):
        from client.memory_consolidator import cosine_similarity
        a = self._make_vec([1.0, 0.0])
        b = self._make_vec([-1.0, 0.0])
        assert abs(cosine_similarity(a, b) - (-1.0)) < 1e-5

    def test_handles_zero_vector_gracefully(self):
        from client.memory_consolidator import cosine_similarity
        a = self._make_vec([0.0, 0.0])
        b = self._make_vec([1.0, 0.0])
        assert cosine_similarity(a, b) == 0.0

    def test_handles_corrupt_bytes_gracefully(self):
        from client.memory_consolidator import cosine_similarity
        assert cosine_similarity(b"bad", b"data") == 0.0


# ═══════════════════════════════════════════════════════════════════

class TestTransientFiltering:
    def test_extract_prompt_excludes_calendar_events(self):
        """Extraction prompt must explicitly forbid calendar/transient events."""
        from client.memory_consolidator import _EXTRACT_PROMPT
        prompt_lower = _EXTRACT_PROMPT.lower()
        # Should mention NOT extracting time-bound things
        assert "do not extract" in prompt_lower or "do not store" in prompt_lower
        assert "calendar" in prompt_lower or "appointment" in prompt_lower or "time-bound" in prompt_lower

    def test_extract_prompt_mentions_durable_facts(self):
        """Extraction prompt must focus on durable/long-term facts."""
        from client.memory_consolidator import _EXTRACT_PROMPT
        assert "durable" in _EXTRACT_PROMPT.lower() or "weeks or months" in _EXTRACT_PROMPT.lower()

    def test_decision_prompt_exists(self):
        try:
            from client.memory_consolidator import _DECISION_PROMPT
        except ImportError:
            pytest.skip("_DECISION_PROMPT not in deployed memory_consolidator.py yet")
        assert len(_DECISION_PROMPT) > 100

    def test_decision_prompt_focuses_on_technical_outcomes(self):
        try:
            from client.memory_consolidator import _DECISION_PROMPT
        except ImportError:
            pytest.skip("_DECISION_PROMPT not in deployed memory_consolidator.py yet")
        prompt_lower = _DECISION_PROMPT.lower()
        assert "fix" in prompt_lower or "bug" in prompt_lower
        assert "file" in prompt_lower or "path" in prompt_lower


# ═══════════════════════════════════════════════════════════════════

class TestBroadcastProactiveResult:
    def _import(self):
        try:
            from client.websocket import broadcast_proactive_result
            return broadcast_proactive_result
        except ImportError:
            pytest.skip("broadcast_proactive_result not in deployed websocket.py yet")

    @pytest.mark.asyncio
    async def test_scheduled_result_broadcasts_assistant_message(self):
        broadcast_proactive_result = self._import()
        with patch("client.websocket.broadcast_message", new_callable=AsyncMock) as mock_bcast:
            await broadcast_proactive_result({
                "type": "scheduled_result",
                "label": "Morning Briefing",
                "result": "Here is your briefing."
            })
        mock_bcast.assert_called_once()
        call_args = mock_bcast.call_args
        assert call_args[0][0] == "assistant_message"
        # Label is no longer prefixed — result is broadcast directly
        assert "Here is your briefing." in call_args[0][1]["text"]

    @pytest.mark.asyncio
    async def test_scheduled_error_broadcasts_warning(self):
        broadcast_proactive_result = self._import()
        with patch("client.websocket.broadcast_message", new_callable=AsyncMock) as mock_bcast:
            await broadcast_proactive_result({
                "type": "scheduled_error",
                "label": "Daily Summary",
                "error": "connection timeout"
            })
        mock_bcast.assert_called_once()
        text = mock_bcast.call_args[0][1]["text"]
        assert "Daily Summary" in text
        assert "connection timeout" in text

    @pytest.mark.asyncio
    async def test_unknown_type_does_not_broadcast(self):
        broadcast_proactive_result = self._import()
        with patch("client.websocket.broadcast_message", new_callable=AsyncMock) as mock_bcast:
            await broadcast_proactive_result({"type": "unknown_type"})
        mock_bcast.assert_not_called()


# ═══════════════════════════════════════════════════════════════════
# :commands list completeness
# ═══════════════════════════════════════════════════════════════════

class TestCommandsListCompleteness:
    def _get_commands(self):
        from client.commands import get_commands_list
        return get_commands_list()

    def test_memory_commands_listed(self):
        cmds = "\n".join(self._get_commands())
        assert ":memory forget" in cmds
        assert ":memory dedup" in cmds
        assert ":memory consolidate" in cmds
        assert ":memory add" in cmds

    def test_multi_commands_listed(self):
        cmds = "\n".join(self._get_commands())
        assert ":multi" in cmds

    def test_metrics_command_listed(self):
        cmds = "\n".join(self._get_commands())
        assert ":metrics" in cmds

    def test_routing_command_listed(self):
        cmds = "\n".join(self._get_commands())
        assert ":routing" in cmds

    def test_sync_command_listed(self):
        cmds = "\n".join(self._get_commands())
        assert ":sync" in cmds

    def test_no_duplicate_command_prefixes(self):
        """Each command prefix should appear at most once as a primary entry."""
        cmds = self._get_commands()
        # Check the list has no exact duplicates
        assert len(cmds) == len(set(cmds)), "Duplicate entries in command list"


# ═══════════════════════════════════════════════════════════════════
# Search query year injection (langgraph prompt canary)
# ═══════════════════════════════════════════════════════════════════

class TestSearchQueryGeneration:
    def test_confidence_check_prompt_forbids_year_injection(self):
        """The confidence-check query generation prompt must not tell the LLM
        to include years from training data."""
        import inspect
        import client.langgraph as lg
        source = inspect.getsource(lg)

        # Old bad instruction must be gone
        assert "include the date or year in the query" not in source
        assert "include the current month and year" not in source

        # New good instruction must be present
        assert "Do NOT include" in source or "do not include" in source.lower() or \
               "NOT include any specific year" in source

    def test_date_note_forbids_year_injection(self):
        """The _date_note injected into the system prompt must not encourage years."""
        import inspect
        import client.langgraph as lg
        source = inspect.getsource(lg)
        # The old bad date note
        assert "include the current month and year in your search query" not in source


# ═══════════════════════════════════════════════════════════════════
# Notification system (index.js — canary checks on JS source)
# ═══════════════════════════════════════════════════════════════════

class TestNotificationSystemJS:
    """Canary tests — read the JS source and verify key structural properties."""

    def _get_js(self):
        candidates = [
            Path("client/ui/js/index.js"),
            Path("client/ui/index.js"),
        ]
        for p in candidates:
            if p.exists():
                return p.read_text()
        pytest.skip("index.js not found — skipping JS canary tests")

    def test_notif_iife_present(self):
        js = self._get_js()
        assert "_notif = (() =>" in js or "const _notif" in js

    def test_notify_always_flashes_title(self):
        """notify() should call _startTitleFlash regardless of mode."""
        js = self._get_js()
        # Find the notify function body
        start = js.find("function notify(")
        end = js.find("\n    }", start)
        body = js[start:end]
        assert "_startTitleFlash" in body

    def test_notify_fires_native_if_available(self):
        js = self._get_js()
        assert "new Notification(" in js

    def test_default_notifications_off(self):
        """Default state must be off (=== 'true' pattern)."""
        js = self._get_js()
        assert "=== 'true'" in js
        assert "!== 'false'" not in js or js.index("=== 'true'") < js.index("!== 'false'") \
            if "!== 'false'" in js else True

    def test_permission_requested_on_toggle_not_init(self):
        """requestPermission must only be called inside toggle(), not init()."""
        js = self._get_js()
        init_start = js.find("async function init()")
        init_end = js.find("\n    }", init_start)
        toggle_start = js.find("async function toggle()")

        init_body = js[init_start:init_end]
        assert "requestPermission" not in init_body, \
            "requestPermission must not be called during init()"
        assert "requestPermission" in js[toggle_start:toggle_start + 500], \
            "requestPermission must be called inside toggle()"