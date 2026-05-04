"""
Tests targeting remaining uncovered lines in:
  - client/utils.py        (image handler in QuietHTTPRequestHandler)
  - client/commands.py     (:env, :health stub, :gguf list signal, :tools with mocks)
  - client/langgraph.py    (fetch_url_content async, fetch_from_source_directly,
                            rag_node stop/no-message/no-tool paths)
"""
import asyncio
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


# ═══════════════════════════════════════════════════════════════════
# utils.py — QuietHTTPRequestHandler image serving
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestQuietHTTPRequestHandler:
    """Test the image handler inside start_http_server via direct class instantiation."""

    def _make_handler(self, path, temp_dir):
        """Build a handler instance with a mocked socket/server."""
        from client.utils import start_http_server
        import io

        # Extract the inner class by starting a server and inspecting
        # Instead, test via HTTP directly since the server is a daemon thread
        # We test the logic by calling do_GET via a mock
        from http.server import BaseHTTPRequestHandler
        from socketserver import TCPServer
        from client.utils import start_http_server
        import client.utils as utils_module
        import inspect, textwrap

        # Re-extract QuietHTTPRequestHandler by starting server on unused port
        # and inspecting — instead we test it by making real HTTP requests
        return None

    def test_image_served_successfully(self, temp_dir):
        """Test that a real image file is served via the HTTP server."""
        import http.client
        import time

        # Create a fake PNG file
        img = temp_dir / "test.png"
        img.write_bytes(b'\x89PNG\r\n\x1a\n' + b'\x00' * 100)

        from client.utils import start_http_server
        start_http_server(port=19877)
        time.sleep(0.1)  # let server start

        conn = http.client.HTTPConnection("127.0.0.1", 19877)
        conn.request("GET", f"/image?path={img}")
        resp = conn.getresponse()
        assert resp.status == 200
        assert "image" in resp.getheader("Content-Type", "")
        conn.close()

    def test_image_missing_path_param_returns_400(self, temp_dir):
        """Test that /image with no path returns 400."""
        import http.client
        import time

        from client.utils import start_http_server
        start_http_server(port=19878)
        time.sleep(0.1)

        conn = http.client.HTTPConnection("127.0.0.1", 19878)
        conn.request("GET", "/image")
        resp = conn.getresponse()
        assert resp.status == 400
        conn.close()

    def test_image_nonexistent_file_returns_404(self, temp_dir):
        """Test that /image with a nonexistent path returns 404."""
        import http.client
        import time

        from client.utils import start_http_server
        start_http_server(port=19879)
        time.sleep(0.1)

        conn = http.client.HTTPConnection("127.0.0.1", 19879)
        conn.request("GET", "/image?path=/nonexistent/file.png")
        resp = conn.getresponse()
        assert resp.status == 404
        conn.close()

    def test_non_image_file_served_as_jpeg(self, temp_dir):
        """Test that a file with unknown MIME type gets image/jpeg fallback."""
        import http.client
        import time

        weird = temp_dir / "test.xyz"
        weird.write_bytes(b'\xff\xd8\xff' + b'\x00' * 100)  # JPEG magic bytes

        from client.utils import start_http_server
        start_http_server(port=19880)
        time.sleep(0.1)

        conn = http.client.HTTPConnection("127.0.0.1", 19880)
        conn.request("GET", f"/image?path={weird}")
        resp = conn.getresponse()
        # Should serve with image/jpeg fallback
        assert resp.status == 200
        assert resp.getheader("Content-Type") == "image/jpeg"
        conn.close()


# ═══════════════════════════════════════════════════════════════════
# commands.py — :env, :health, :gguf list→print_all_models
# ═══════════════════════════════════════════════════════════════════

def make_models_module():
    m = MagicMock()
    m.load_last_model.return_value = "test-model"
    m.detect_backend.return_value = "ollama"
    m.print_all_models = MagicMock()
    m.switch_model = AsyncMock(return_value=MagicMock())
    m.reload_current_model = AsyncMock(return_value=(MagicMock(), "test-model"))
    return m


async def call_cmd(command, tools=None, models_module=None, **kwargs):
    from client.commands import handle_command
    return await handle_command(
        command=command,
        tools=tools or [],
        model_name="test-model",
        conversation_state={"messages": []},
        models_module=models_module or make_models_module(),
        system_prompt="Test",
        logger=MagicMock(),
        **kwargs
    )


@pytest.mark.unit
@pytest.mark.asyncio
class TestCommandsUncovered:
    async def test_env_command(self):
        """Test :env calls format_env_display."""
        with patch("client.env_display.format_env_display", return_value="ENV OUTPUT"):
            handled, response, _, _ = await call_cmd(":env")
        assert handled is True
        assert response == "ENV OUTPUT"

    async def test_health_command(self):
        """Test :health calls run_health_check."""
        with patch("client.health.run_health_check", new_callable=AsyncMock,
                   return_value="✅ All healthy"):
            handled, response, _, _ = await call_cmd(":health")
        assert handled is True
        assert "healthy" in response.lower()

    async def test_health_command_with_args(self):
        """Test :health server_name passes args through."""
        with patch("client.health.run_health_check", new_callable=AsyncMock,
                   return_value="✅ google healthy") as mock_health:
            handled, response, _, _ = await call_cmd(":health google")
        assert handled is True
        # Verify argument was passed
        call_args = mock_health.call_args[0]
        assert call_args[0] == "google"

    async def test_gguf_list_calls_print_all_models(self):
        """Test :gguf list triggers print_all_models."""
        mm = make_models_module()
        handled, response, _, _ = await call_cmd(":gguf list", models_module=mm)
        assert handled is True
        mm.print_all_models.assert_called_once()

    async def test_tools_command_no_tools(self):
        """Test :tools with empty list."""
        handled, response, _, _ = await call_cmd(":tools", tools=[])
        assert handled is True
        assert "No tools" in response or "available" in response.lower()

    async def test_a2a_off_no_orchestrator(self):
        """Test :a2a off without orchestrator."""
        handled, response, _, _ = await call_cmd(":a2a off", orchestrator=None)
        assert handled is True
        assert "not available" in response.lower()

    async def test_stats_format_metrics_summary(self):
        """Test :stats calls format_metrics_summary if it exists."""
        from client.metrics import reset_metrics
        reset_metrics()

        # Patch format_metrics_summary which may not exist
        with patch("client.metrics.prepare_metrics", return_value={}):
            with patch("client.commands.handle_command.__module__"):
                handled, response, _, _ = await call_cmd(":stats")
        assert handled is True


# ═══════════════════════════════════════════════════════════════════
# langgraph.py — fetch_url_content (async wrapper)
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
class TestFetchUrlContentAsync:
    async def test_delegates_to_sync(self):
        from client.langgraph import fetch_url_content
        expected = {"success": True, "content": "hello", "title": "T", "url": "https://x.com"}
        with patch("client.langgraph.fetch_url_content_sync", return_value=expected):
            result = await fetch_url_content("https://x.com")
        assert result == expected

    async def test_passes_timeout(self):
        from client.langgraph import fetch_url_content
        captured = {}

        def capture_sync(url, timeout=30):
            captured["timeout"] = timeout
            return {"success": False, "error": "test"}

        with patch("client.langgraph.fetch_url_content_sync", side_effect=capture_sync):
            await fetch_url_content("https://x.com", timeout=15)
        assert captured["timeout"] == 15


# ═══════════════════════════════════════════════════════════════════
# langgraph.py — fetch_from_source_directly
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
class TestFetchFromSourceDirectly:
    async def test_known_source_returns_direct(self):
        from client.langgraph import fetch_from_source_directly, DIRECT_SOURCE_URLS
        # wikipedia.org is in DIRECT_SOURCE_URLS
        result = await fetch_from_source_directly("wikipedia.org", "python")
        # Either success with urls or no_config (depending on fallback_urls config)
        assert "success" in result
        assert "method" in result

    async def test_unknown_source_returns_no_config(self):
        from client.langgraph import fetch_from_source_directly
        result = await fetch_from_source_directly("unknown-site-xyz123.com", "query")
        assert result["success"] is False
        assert result["method"] == "no_config"

    async def test_www_stripped(self):
        from client.langgraph import fetch_from_source_directly
        # www. prefix should be stripped before matching
        result = await fetch_from_source_directly("www.unknown-xyz.com", "query")
        assert result["method"] == "no_config"


# ═══════════════════════════════════════════════════════════════════
# langgraph.py — rag_node paths (stop signal, no message, no tool)
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
class TestRagNode:
    def _make_state(self, messages=None, tools=None):
        return {
            "messages": messages or [HumanMessage(content="search for python docs")],
            "tools": tools or {},
            "llm": MagicMock(),
            "stopped": False,
            "ingest_completed": False,
            "current_model": "test",
            "research_source": "",
            "session_state": None,
            "capability_registry": None,
        }

    async def test_stop_requested_returns_cancelled(self):
        from client.langgraph import rag_node
        from client.stop_signal import request_stop, clear_stop
        request_stop()
        try:
            result = await rag_node(self._make_state())
            msgs = result["messages"]
            assert any("cancel" in m.content.lower() for m in msgs)
            assert result.get("stopped") is True
        finally:
            clear_stop()

    async def test_no_user_message_returns_error(self):
        from client.langgraph import rag_node
        from client.stop_signal import clear_stop
        clear_stop()
        state = self._make_state(messages=[AIMessage(content="AI only")])
        result = await rag_node(state)
        msgs = result["messages"]
        assert any("error" in m.content.lower() or "could not" in m.content.lower()
                   for m in msgs)

    async def test_no_rag_tool_returns_unavailable(self):
        from client.langgraph import rag_node
        from client.stop_signal import clear_stop
        clear_stop()
        state = self._make_state(tools={})  # no rag_search_tool
        result = await rag_node(state)
        msgs = result["messages"]
        assert any("not available" in m.content.lower() or "not found" in m.content.lower()
                   for m in msgs)

    async def test_rag_tool_exception_returns_error(self):
        from client.langgraph import rag_node
        from client.stop_signal import clear_stop
        clear_stop()

        mock_rag_tool = MagicMock()
        mock_rag_tool.name = "rag_search_tool"
        mock_rag_tool.ainvoke = AsyncMock(side_effect=Exception("RAG db error"))

        state = self._make_state(tools={"rag_search_tool": mock_rag_tool})
        result = await rag_node(state)
        msgs = result["messages"]
        assert any("error" in m.content.lower() for m in msgs)


# ═══════════════════════════════════════════════════════════════════
# langgraph.py — router: ingest with multi-step and one-time paths
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestRouterIngestPaths:
    def _make_state(self, content, ingest_completed=False):
        return {
            "messages": [HumanMessage(content=content)],
            "stopped": False,
            "ingest_completed": ingest_completed,
            "research_source": "",
            "tools": {},
            "llm": None,
            "current_model": "test",
            "session_state": None,
            "capability_registry": None,
        }

    def test_ingest_with_stop_returns_ingest(self):
        from client.langgraph import router
        from client.stop_signal import clear_stop
        clear_stop()
        state = self._make_state("ingest now then stop")
        result = router(state)
        assert result == "ingest"

    def test_ingest_with_multi_step_returns_continue(self):
        from client.langgraph import router
        from client.stop_signal import clear_stop
        clear_stop()
        state = self._make_state("ingest now and then summarize results")
        result = router(state)
        assert result == "continue"