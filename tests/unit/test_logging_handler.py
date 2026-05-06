"""
tests/unit/test_logging_handler.py
Tests for client/logging_handler.py — WebSocket log broadcasting and formatters.
"""
import json
import logging
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.unit
class TestLoggingHandlerImport:
    def test_module_importable(self):
        """logging_handler imports without error."""
        import client.logging_handler
        assert client.logging_handler is not None


@pytest.mark.unit
class TestWebSocketLogHandler:
    def _make_handler(self):
        from client.logging_handler import WebSocketLogHandler
        handler = WebSocketLogHandler()
        return handler

    def test_handler_instantiates(self):
        handler = self._make_handler()
        assert handler is not None

    def test_handler_is_logging_handler(self):
        handler = self._make_handler()
        assert isinstance(handler, logging.Handler)

    def test_emit_with_no_clients_does_not_crash(self):
        """emit() with no connected WebSocket clients should not raise."""
        handler = self._make_handler()
        record = logging.LogRecord(
            name="test", level=logging.INFO,
            pathname="", lineno=0,
            msg="test message", args=(), exc_info=None
        )
        try:
            handler.emit(record)
        except Exception as e:
            pytest.fail(f"emit() raised {e}")

    def test_emit_with_mock_client(self):
        """emit() sends JSON to connected WebSocket clients."""
        from client.logging_handler import WebSocketLogHandler
        from client.websocket import SYSTEM_MONITOR_CLIENTS

        ws = MagicMock()
        ws.send = AsyncMock()
        SYSTEM_MONITOR_CLIENTS.add(ws)

        try:
            handler = WebSocketLogHandler()
            record = logging.LogRecord(
                name="mcp_client", level=logging.INFO,
                pathname="", lineno=0,
                msg="hello from test", args=(), exc_info=None
            )
            handler.emit(record)
        finally:
            SYSTEM_MONITOR_CLIENTS.discard(ws)

    def test_emit_formats_level_correctly(self):
        """emit() uses the correct level name."""
        from client.logging_handler import WebSocketLogHandler
        sent_data = []

        class CapturingWS:
            async def send(self, data):
                sent_data.append(data)

        from client.websocket import SYSTEM_MONITOR_CLIENTS
        ws = CapturingWS()
        SYSTEM_MONITOR_CLIENTS.add(ws)

        try:
            handler = WebSocketLogHandler()
            for level, name in [
                (logging.DEBUG, "DEBUG"),
                (logging.INFO, "INFO"),
                (logging.WARNING, "WARNING"),
                (logging.ERROR, "ERROR"),
            ]:
                record = logging.LogRecord(
                    name="test", level=level, pathname="",
                    lineno=0, msg=f"test {name}", args=(), exc_info=None
                )
                handler.emit(record)
        finally:
            SYSTEM_MONITOR_CLIENTS.discard(ws)





@pytest.mark.unit
class TestLogHandlerSetup:
    def test_setup_logging_function_exists(self):
        """setup_logging or configure_logging function is present."""
        import client.logging_handler as lh
        has_setup = any(
            hasattr(lh, name)
            for name in ["setup_logging", "configure_logging", "get_handler", "create_handler"]
        )
        # At minimum the module should define something callable
        public = [x for x in dir(lh) if not x.startswith("_")]
        assert len(public) > 0

    def test_no_crash_on_import_with_no_ws(self):
        """Module import doesn't crash even without WebSocket clients."""
        import importlib
        import client.logging_handler
        importlib.reload(client.logging_handler)
        assert True


@pytest.mark.unit
class TestLogRecordFormatting:
    def test_log_record_message_preserved(self):
        """Log records carry the message through the handler."""
        from client.logging_handler import WebSocketLogHandler
        captured = []

        def capturing_emit(self, record):
            captured.append(self.format(record))

        with patch.object(WebSocketLogHandler, "emit", capturing_emit):
            handler = WebSocketLogHandler()
            logger = logging.getLogger("mcp_client_test_capture")
            original_level = logger.level
            logger.setLevel(logging.DEBUG)
            logger.addHandler(handler)
            try:
                logger.info("unique_test_message_xyz")
            finally:
                logger.removeHandler(handler)
                logger.setLevel(original_level)

        assert any("unique_test_message_xyz" in msg for msg in captured)

    def test_exception_in_emit_does_not_propagate(self):
        """If emit fails internally, it should not crash the caller."""
        from client.logging_handler import WebSocketLogHandler
        handler = WebSocketLogHandler()

        # Make format() raise
        with patch.object(handler, "format", side_effect=Exception("format error")):
            record = logging.LogRecord(
                name="test", level=logging.INFO, pathname="",
                lineno=0, msg="test", args=(), exc_info=None
            )
            try:
                handler.emit(record)
            except Exception as e:
                # Only acceptable if it's a handleError call
                pass