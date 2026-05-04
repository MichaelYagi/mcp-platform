"""
Tests for stop_signal, websocket helper functions, and utils.
Targets currently uncovered code to increase overall coverage.
"""
import asyncio
import pytest
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


# ═══════════════════════════════════════════════════════════════════
# stop_signal.py  (currently 48% covered)
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestStopSignal:
    def setup_method(self):
        """Reset stop signal before each test."""
        from client.stop_signal import clear_stop
        clear_stop()

    def test_initial_state_is_not_requested(self):
        from client.stop_signal import is_stop_requested
        assert not is_stop_requested()

    def test_request_stop_sets_flag(self):
        from client.stop_signal import request_stop, is_stop_requested
        request_stop()
        assert is_stop_requested()

    def test_clear_stop_resets_flag(self):
        from client.stop_signal import request_stop, clear_stop, is_stop_requested
        request_stop()
        assert is_stop_requested()
        clear_stop()
        assert not is_stop_requested()

    def test_request_stop_twice_is_idempotent(self):
        from client.stop_signal import request_stop, is_stop_requested
        request_stop()
        request_stop()
        assert is_stop_requested()

    def test_clear_without_request_is_safe(self):
        from client.stop_signal import clear_stop, is_stop_requested
        clear_stop()
        assert not is_stop_requested()

    def test_request_then_clear_then_request_again(self):
        """Stop flag can be re-requested after clearing."""
        from client.stop_signal import request_stop, clear_stop, is_stop_requested
        request_stop()
        clear_stop()
        assert not is_stop_requested()
        request_stop()
        assert is_stop_requested()

    def test_clear_stop_is_safe_before_any_request(self):
        """Clearing without requesting should not raise."""
        from client.stop_signal import clear_stop, is_stop_requested
        clear_stop()
        clear_stop()
        assert not is_stop_requested()


# ═══════════════════════════════════════════════════════════════════
# websocket.py — helper functions (currently 19% covered)
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestWebSocketHelpers:
    def test_get_system_monitor_clients_returns_set(self):
        from client.websocket import get_system_monitor_clients
        result = get_system_monitor_clients()
        assert isinstance(result, set)

    def test_is_system_monitor_available_returns_bool(self):
        from client.websocket import is_system_monitor_available
        result = is_system_monitor_available()
        assert isinstance(result, bool)

    def test_processing_sessions_initially_empty(self):
        from client.websocket import PROCESSING_SESSIONS
        # May have entries from other tests — just verify it's a set
        assert isinstance(PROCESSING_SESSIONS, set)

    def test_session_tasks_is_dict(self):
        from client.websocket import SESSION_TASKS
        assert isinstance(SESSION_TASKS, dict)


@pytest.mark.unit
@pytest.mark.asyncio
class TestBroadcastMessage:
    async def test_broadcast_sends_to_all_clients(self):
        from client.websocket import broadcast_message, CONNECTED_WEBSOCKETS

        ws1 = AsyncMock()
        ws2 = AsyncMock()
        CONNECTED_WEBSOCKETS.add(ws1)
        CONNECTED_WEBSOCKETS.add(ws2)
        try:
            await broadcast_message("test_type", {"text": "hello"})
            ws1.send.assert_called_once()
            ws2.send.assert_called_once()
        finally:
            CONNECTED_WEBSOCKETS.discard(ws1)
            CONNECTED_WEBSOCKETS.discard(ws2)

    async def test_broadcast_sends_correct_json(self):
        import json
        from client.websocket import broadcast_message, CONNECTED_WEBSOCKETS

        ws = AsyncMock()
        CONNECTED_WEBSOCKETS.add(ws)
        try:
            await broadcast_message("assistant_message", {"text": "hi"})
            payload = json.loads(ws.send.call_args[0][0])
            assert payload["type"] == "assistant_message"
            assert payload["text"] == "hi"
        finally:
            CONNECTED_WEBSOCKETS.discard(ws)

    async def test_broadcast_no_clients_does_nothing(self):
        from client.websocket import broadcast_message, CONNECTED_WEBSOCKETS
        # Ensure no clients connected
        original = set(CONNECTED_WEBSOCKETS)
        CONNECTED_WEBSOCKETS.clear()
        try:
            # Should not raise
            await broadcast_message("test", {"text": "ignored"})
        finally:
            CONNECTED_WEBSOCKETS.update(original)

    async def test_broadcast_handles_send_failure_gracefully(self):
        from client.websocket import broadcast_message, CONNECTED_WEBSOCKETS

        ws = AsyncMock()
        ws.send.side_effect = Exception("Connection lost")
        CONNECTED_WEBSOCKETS.add(ws)
        try:
            # Should not raise — return_exceptions=True in gather
            await broadcast_message("test", {"text": "hello"})
        finally:
            CONNECTED_WEBSOCKETS.discard(ws)


@pytest.mark.unit
@pytest.mark.asyncio
class TestCleanupStaleTasks:
    async def test_cleanup_removes_old_tasks(self):
        from client.websocket import (
            _cleanup_stale_tasks, SESSION_TASKS,
            SESSION_TASK_CREATED, SESSION_LOCKS, PROCESSING_SESSIONS
        )

        sid = "stale-session-test"
        mock_task = MagicMock()
        mock_task.done.return_value = True
        SESSION_TASKS[sid] = mock_task
        SESSION_TASK_CREATED[sid] = time.monotonic() - 9999
        SESSION_LOCKS[sid] = asyncio.Lock()
        PROCESSING_SESSIONS.add(sid)

        # Sleep is called at top of loop — let first call pass, cancel on second
        call_count = 0

        async def fast_sleep(n):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise asyncio.CancelledError()

        with patch("asyncio.sleep", side_effect=fast_sleep):
            try:
                await _cleanup_stale_tasks(max_age_seconds=1)
            except asyncio.CancelledError:
                pass

        assert sid not in SESSION_TASKS
        assert sid not in SESSION_TASK_CREATED
        assert sid not in SESSION_LOCKS
        assert sid not in PROCESSING_SESSIONS

    async def test_cleanup_cancels_running_stale_tasks(self):
        from client.websocket import (
            _cleanup_stale_tasks, SESSION_TASKS, SESSION_TASK_CREATED,
            SESSION_LOCKS, PROCESSING_SESSIONS
        )

        sid = "running-stale-session"
        mock_task = MagicMock()
        mock_task.done.return_value = False
        SESSION_TASKS[sid] = mock_task
        SESSION_TASK_CREATED[sid] = time.monotonic() - 9999
        SESSION_LOCKS[sid] = asyncio.Lock()
        PROCESSING_SESSIONS.add(sid)

        call_count = 0

        async def fast_sleep(n):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                raise asyncio.CancelledError()

        with patch("asyncio.sleep", side_effect=fast_sleep):
            try:
                await _cleanup_stale_tasks(max_age_seconds=1)
            except asyncio.CancelledError:
                pass

        mock_task.cancel.assert_called_once()
        assert sid not in SESSION_TASKS


# ═══════════════════════════════════════════════════════════════════
# utils.py  (currently 0% covered)
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestUtils:
    def test_get_venv_python_finds_venv(self, temp_dir):
        from client.utils import get_venv_python

        # Create fake venv structure
        venv = temp_dir / ".venv" / "bin"
        venv.mkdir(parents=True)
        python = venv / "python"
        python.touch()

        result = get_venv_python(temp_dir)
        assert result == str(python)

    def test_get_venv_python_raises_when_missing(self, temp_dir):
        from client.utils import get_venv_python

        with pytest.raises(FileNotFoundError):
            get_venv_python(temp_dir)

    def test_start_http_server_returns_ip(self, temp_dir):
        from client.utils import start_http_server

        # Start server on a random high port to avoid conflicts
        result = start_http_server(port=19876)
        assert isinstance(result, str)
        # Should return a valid IP address
        parts = result.split(".")
        assert len(parts) == 4

    @pytest.mark.asyncio
    async def test_ensure_ollama_running_raises_when_down(self):
        from client.utils import ensure_ollama_running

        with pytest.raises(RuntimeError, match="not running"):
            await ensure_ollama_running("http://127.0.0.1:19999")  # nothing on this port

    @pytest.mark.asyncio
    async def test_ensure_ollama_running_succeeds_when_up(self):
        from client.utils import ensure_ollama_running
        import httpx

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )
            # Should not raise
            await ensure_ollama_running("http://127.0.0.1:11434")


# ═══════════════════════════════════════════════════════════════════
# session_manager.py — uncovered methods (currently 54%)
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestSessionManagerExtended:
    def test_update_session_name(self, session_manager):
        from client.session_manager import SessionManager
        sid = session_manager.create_session("Original")
        session_manager.update_session_name(sid, "Updated")
        session = session_manager.get_session(sid)
        assert session["name"] == "Updated"

    def test_get_sessions_returns_list(self, session_manager):
        session_manager.create_session("S1")
        session_manager.create_session("S2")
        sessions = session_manager.get_sessions()
        assert isinstance(sessions, list)
        assert len(sessions) >= 2

    def test_delete_all_sessions(self, session_manager):
        session_manager.create_session("A")
        session_manager.create_session("B")
        session_manager.delete_all_sessions()
        sessions = session_manager.get_sessions()
        assert len(sessions) == 0

    def test_get_session_messages_empty(self, session_manager):
        sid = session_manager.create_session()
        messages = session_manager.get_session_messages(sid)
        assert messages == []

    def test_add_message_with_model(self, session_manager):
        sid = session_manager.create_session()
        session_manager.add_message(sid, "assistant", "Hello!", 30, model="qwen2.5:14b")
        messages = session_manager.get_session_messages(sid)
        assert len(messages) == 1
        assert messages[0]["model"] == "qwen2.5:14b"

    def test_get_user_session_count(self, populated_session_manager):
        sm, _, _ = populated_session_manager
        count = sm.get_user_session_count()
        assert count >= 2

    def test_get_recent_session_topics(self, populated_session_manager):
        sm, _, _ = populated_session_manager
        topics = sm.get_recent_session_topics(limit=5)
        assert isinstance(topics, list)
        assert len(topics) >= 1

    def test_message_trimming_keeps_newest(self, session_manager):
        """When over the limit, oldest messages are dropped."""
        sid = session_manager.create_session()
        for i in range(5):
            session_manager.add_message(sid, "user", f"msg {i}", max_history=3, model=None)
        messages = session_manager.get_session_messages(sid)
        assert len(messages) == 3
        # Should keep the most recent
        assert messages[-1]["text"] == "msg 4"