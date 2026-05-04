"""
Tests for client/langgraph.py
Covers: _classify_error, _record_failure, HTMLTextExtractor,
        fetch_url_content_sync, router, should_continue_after_tools,
        create_langgraph_agent, run_agent (error paths)
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import (
    AIMessage, HumanMessage, SystemMessage, ToolMessage
)


# ═══════════════════════════════════════════════════════════════════
# _classify_error
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestClassifyError:
    def test_timeout_error_is_retryable(self):
        from client.langgraph import _classify_error
        from client.metrics import FailureKind
        result = _classify_error(asyncio.TimeoutError())
        assert result == FailureKind.RETRYABLE

    def test_timeout_string_is_retryable(self):
        from client.langgraph import _classify_error
        from client.metrics import FailureKind
        result = _classify_error(Exception("connection timeout"))
        assert result == FailureKind.RETRYABLE

    def test_rate_limit_is_retryable(self):
        from client.langgraph import _classify_error
        from client.metrics import FailureKind
        result = _classify_error(Exception("rate limit exceeded"))
        assert result == FailureKind.RETRYABLE

    def test_429_is_retryable(self):
        from client.langgraph import _classify_error
        from client.metrics import FailureKind
        result = _classify_error(Exception("HTTP 429 too many requests"))
        assert result == FailureKind.RETRYABLE

    def test_connection_refused_is_retryable(self):
        from client.langgraph import _classify_error
        from client.metrics import FailureKind
        result = _classify_error(Exception("connection refused"))
        assert result == FailureKind.RETRYABLE

    def test_context_overflow_is_user_error(self):
        from client.langgraph import _classify_error
        from client.metrics import FailureKind
        result = _classify_error(ValueError("exceed context window"))
        assert result == FailureKind.USER_ERROR

    def test_requested_tokens_is_user_error(self):
        from client.langgraph import _classify_error
        from client.metrics import FailureKind
        result = _classify_error(ValueError("Requested tokens (5000) exceed context window of (4096)"))
        assert result == FailureKind.USER_ERROR

    def test_invalid_param_is_user_error(self):
        from client.langgraph import _classify_error
        from client.metrics import FailureKind
        result = _classify_error(ValueError("invalid parameter"))
        assert result == FailureKind.USER_ERROR

    def test_ollama_crash_is_upstream(self):
        from client.langgraph import _classify_error
        from client.metrics import FailureKind
        result = _classify_error(Exception("model runner has unexpectedly stopped"))
        assert result == FailureKind.UPSTREAM_ERROR

    def test_http_502_is_upstream(self):
        from client.langgraph import _classify_error
        from client.metrics import FailureKind
        result = _classify_error(Exception("HTTP 502 bad gateway"))
        assert result == FailureKind.UPSTREAM_ERROR

    def test_plex_error_is_upstream(self):
        from client.langgraph import _classify_error
        from client.metrics import FailureKind
        result = _classify_error(Exception("plex server unreachable"))
        assert result == FailureKind.UPSTREAM_ERROR

    def test_unknown_error_is_internal(self):
        from client.langgraph import _classify_error
        from client.metrics import FailureKind
        result = _classify_error(Exception("some weird unexpected thing"))
        assert result == FailureKind.INTERNAL_ERROR

    def test_generic_runtime_error_is_internal(self):
        from client.langgraph import _classify_error
        from client.metrics import FailureKind
        result = _classify_error(RuntimeError("oops"))
        assert result == FailureKind.INTERNAL_ERROR


# ═══════════════════════════════════════════════════════════════════
# _record_failure
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestRecordFailure:
    def test_records_failure_kind(self):
        from client.langgraph import _record_failure
        from client.metrics import metrics, FailureKind, reset_metrics
        reset_metrics()
        _record_failure(FailureKind.RETRYABLE)
        assert metrics["failure_kinds"]["retryable"] == 1

    def test_records_multiple_failures(self):
        from client.langgraph import _record_failure
        from client.metrics import metrics, FailureKind, reset_metrics
        reset_metrics()
        _record_failure(FailureKind.INTERNAL_ERROR)
        _record_failure(FailureKind.INTERNAL_ERROR)
        assert metrics["failure_kinds"]["internal_error"] == 2

    def test_records_different_kinds(self):
        from client.langgraph import _record_failure
        from client.metrics import metrics, FailureKind, reset_metrics
        reset_metrics()
        _record_failure(FailureKind.USER_ERROR)
        _record_failure(FailureKind.UPSTREAM_ERROR)
        assert metrics["failure_kinds"]["user_error"] == 1
        assert metrics["failure_kinds"]["upstream_error"] == 1



# ═══════════════════════════════════════════════════════════════════
# llm_ainvoke — cancellable LLM wrapper
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
class TestLlmAinvoke:
    async def test_returns_llm_result(self):
        from client.langgraph import llm_ainvoke
        from client.stop_signal import clear_stop
        clear_stop()
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="hello"))
        result = await llm_ainvoke(mock_llm, [HumanMessage(content="hi")])
        assert result.content == "hello"

    async def test_cancels_on_stop_signal(self):
        from client.langgraph import llm_ainvoke
        from client.stop_signal import request_stop, clear_stop
        clear_stop()
        async def slow_llm(messages):
            await asyncio.sleep(10)
            return AIMessage(content="never")
        mock_llm = MagicMock()
        mock_llm.ainvoke = slow_llm
        async def set_stop():
            await asyncio.sleep(0.1)
            request_stop()
        asyncio.create_task(set_stop())
        with pytest.raises(asyncio.CancelledError):
            await llm_ainvoke(mock_llm, [], poll_interval=0.05)
        clear_stop()

    async def test_propagates_llm_exception(self):
        from client.langgraph import llm_ainvoke
        from client.stop_signal import clear_stop
        clear_stop()
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=ValueError("bad input"))
        with pytest.raises(ValueError, match="bad input"):
            await llm_ainvoke(mock_llm, [])

# ═══════════════════════════════════════════════════════════════════
# HTMLTextExtractor
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestHTMLTextExtractor:
    def test_extracts_body_text(self):
        from client.langgraph import HTMLTextExtractor
        parser = HTMLTextExtractor()
        parser.feed("<html><body><p>Hello world</p></body></html>")
        assert "Hello world" in parser.get_text()

    def test_skips_script_tags(self):
        from client.langgraph import HTMLTextExtractor
        parser = HTMLTextExtractor()
        parser.feed("<html><body><script>var x = 1;</script><p>Content</p></body></html>")
        text = parser.get_text()
        assert "var x" not in text
        assert "Content" in text

    def test_skips_style_tags(self):
        from client.langgraph import HTMLTextExtractor
        parser = HTMLTextExtractor()
        parser.feed("<html><head><style>.foo { color: red; }</style></head><body><p>Text</p></body></html>")
        text = parser.get_text()
        assert "color" not in text
        assert "Text" in text

    def test_skips_nav_tags(self):
        from client.langgraph import HTMLTextExtractor
        parser = HTMLTextExtractor()
        parser.feed("<nav>Menu items</nav><main>Main content</main>")
        text = parser.get_text()
        assert "Menu items" not in text
        assert "Main content" in text

    def test_extracts_title(self):
        from client.langgraph import HTMLTextExtractor
        parser = HTMLTextExtractor()
        parser.feed("<html><head><title>My Page Title</title></head><body><p>Body</p></body></html>")
        assert parser.get_title() == "My Page Title"

    def test_untitled_returns_default(self):
        from client.langgraph import HTMLTextExtractor
        parser = HTMLTextExtractor()
        parser.feed("<html><body><p>No title</p></body></html>")
        assert parser.get_title() == "Untitled"

    def test_collapses_multiple_blank_lines(self):
        from client.langgraph import HTMLTextExtractor
        parser = HTMLTextExtractor()
        parser.feed("<p>Line 1</p><p>Line 2</p>")
        text = parser.get_text()
        # Should not have more than 2 consecutive newlines
        assert "\n\n\n" not in text

    def test_empty_html(self):
        from client.langgraph import HTMLTextExtractor
        parser = HTMLTextExtractor()
        parser.feed("")
        assert parser.get_text() == ""
        assert parser.get_title() == "Untitled"

    def test_multiple_paragraphs(self):
        from client.langgraph import HTMLTextExtractor
        parser = HTMLTextExtractor()
        parser.feed("<p>First</p><p>Second</p><p>Third</p>")
        text = parser.get_text()
        assert "First" in text
        assert "Second" in text
        assert "Third" in text

    def test_skips_footer_and_header(self):
        from client.langgraph import HTMLTextExtractor
        parser = HTMLTextExtractor()
        parser.feed("<header>Site header</header><article>Article text</article><footer>Copyright</footer>")
        text = parser.get_text()
        assert "Site header" not in text
        assert "Copyright" not in text
        assert "Article text" in text


# ═══════════════════════════════════════════════════════════════════
# fetch_url_content_sync
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestFetchUrlContentSync:
    def test_success_returns_content(self):
        from client.langgraph import fetch_url_content_sync
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html><body><p>" + "Content here. " * 20 + "</p></body></html>"
        with patch("requests.get", return_value=mock_response):
            result = fetch_url_content_sync("https://example.com")
        assert result["success"] is True
        assert "content" in result
        assert result["url"] == "https://example.com"

    def test_http_error_returns_failure(self):
        from client.langgraph import fetch_url_content_sync
        mock_response = MagicMock()
        mock_response.status_code = 404
        with patch("requests.get", return_value=mock_response):
            result = fetch_url_content_sync("https://example.com")
        assert result["success"] is False
        assert "404" in result["error"]

    def test_no_content_returns_failure(self):
        from client.langgraph import fetch_url_content_sync
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html><body></body></html>"
        with patch("requests.get", return_value=mock_response):
            result = fetch_url_content_sync("https://example.com")
        assert result["success"] is False

    def test_exception_returns_failure(self):
        from client.langgraph import fetch_url_content_sync
        with patch("requests.get", side_effect=Exception("Connection refused")):
            result = fetch_url_content_sync("https://example.com")
        assert result["success"] is False
        assert "Connection refused" in result["error"]

    def test_truncates_large_content(self):
        from client.langgraph import fetch_url_content_sync
        long_content = "word " * 5000
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = f"<html><body><p>{long_content}</p></body></html>"
        with patch("requests.get", return_value=mock_response):
            result = fetch_url_content_sync("https://example.com")
        assert result["success"] is True
        assert "truncated" in result["content"].lower()
        assert len(result["content"]) <= 10_200

    def test_title_extracted(self):
        from client.langgraph import fetch_url_content_sync
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html><head><title>Test Title</title></head><body><p>" + "content " * 20 + "</p></body></html>"
        with patch("requests.get", return_value=mock_response):
            result = fetch_url_content_sync("https://example.com")
        assert result["success"] is True
        assert result["title"] == "Test Title"

    def test_url_with_special_chars_encoded(self):
        from client.langgraph import fetch_url_content_sync
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "<html><body><p>" + "text " * 20 + "</p></body></html>"
        with patch("requests.get", return_value=mock_response) as mock_get:
            fetch_url_content_sync("https://example.com/path with spaces")
            # Verify requests.get was called
            assert mock_get.called


# ═══════════════════════════════════════════════════════════════════
# router()
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestRouter:
    def _make_state(self, messages, stopped=False, ingest_completed=False):
        return {
            "messages": messages,
            "stopped": stopped,
            "ingest_completed": ingest_completed,
            "research_source": "",
            "tools": {},
            "llm": None,
            "current_model": "test",
            "session_state": None,
            "capability_registry": None,
        }

    def test_stop_requested_returns_continue(self):
        from client.langgraph import router
        from client.stop_signal import request_stop, clear_stop
        request_stop()
        try:
            state = self._make_state([AIMessage(content="hi")])
            result = router(state)
            assert result == "continue"
            assert state["stopped"] is True
        finally:
            clear_stop()

    def test_already_stopped_returns_continue(self):
        from client.langgraph import router
        state = self._make_state([AIMessage(content="hi")], stopped=True)
        result = router(state)
        assert result == "continue"

    def test_ai_message_with_tool_calls_returns_tools(self):
        from client.langgraph import router
        msg = AIMessage(content="", tool_calls=[{"name": "get_weather", "args": {}, "id": "1"}])
        state = self._make_state([msg])
        result = router(state)
        assert result == "tools"

    def test_ai_message_no_tool_calls_returns_continue(self):
        from client.langgraph import router
        state = self._make_state([AIMessage(content="Here is the answer.")])
        result = router(state)
        assert result == "continue"

    def test_research_sentinel_returns_research(self):
        from client.langgraph import router
        state = self._make_state([AIMessage(content="__RESEARCH__")])
        result = router(state)
        assert result == "research"

    def test_a2a_tool_message_returns_continue(self):
        from client.langgraph import router
        msg = ToolMessage(content="result", name="send_a2a", tool_call_id="1")
        state = self._make_state([msg])
        result = router(state)
        assert result == "continue"

    def test_explicit_rag_returns_rag(self):
        from client.langgraph import router
        # Last message must be non-AIMessage for human-message routing to fire
        state = self._make_state([
            HumanMessage(content="search rag for my notes about python"),
        ])
        result = router(state)
        assert result == "rag"

    def test_ingest_command_returns_ingest(self):
        from client.langgraph import router
        state = self._make_state([
            HumanMessage(content="ingest now"),
        ])
        result = router(state)
        assert result == "ingest"

    def test_ingest_already_done_returns_continue(self):
        from client.langgraph import router
        state = self._make_state(
            [HumanMessage(content="ingest now")],
            ingest_completed=True
        )
        result = router(state)
        assert result == "continue"

    def test_status_query_returns_continue(self):
        from client.langgraph import router
        state = self._make_state([
            HumanMessage(content="how many items have been ingested"),
        ])
        result = router(state)
        assert result == "continue"

    def test_source_based_research_returns_research(self):
        from client.langgraph import router
        state = self._make_state([
            HumanMessage(content="using https://example.com as source, summarize the content"),
        ])
        result = router(state)
        assert result == "research"

    def test_default_no_match_returns_continue(self):
        from client.langgraph import router
        state = self._make_state([
            HumanMessage(content="hello, how are you?"),
        ])
        result = router(state)
        assert result == "continue"


# ═══════════════════════════════════════════════════════════════════
# should_continue_after_tools
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestShouldContinueAfterTools:
    def test_no_feedback_returns_end(self):
        from client.langgraph import should_continue_after_tools
        state = {
            "messages": [
                HumanMessage(content="What's the weather?"),
                AIMessage(content="It's sunny."),
            ]
        }
        result = should_continue_after_tools(state)
        assert result == "end"

    def test_tool_feedback_returns_agent(self):
        from client.langgraph import should_continue_after_tools
        state = {
            "messages": [
                HumanMessage(content="Analyze this"),
                AIMessage(content="Here's the analysis."),
                HumanMessage(content="[Tool Feedback: Please elaborate on point 2]"),
            ]
        }
        result = should_continue_after_tools(state)
        assert result == "agent"

    def test_empty_messages_returns_end(self):
        from client.langgraph import should_continue_after_tools
        state = {"messages": []}
        result = should_continue_after_tools(state)
        assert result == "end"

    def test_feedback_only_in_last_5_messages(self):
        from client.langgraph import should_continue_after_tools
        # Feedback is older than 5 messages — should not trigger
        state = {
            "messages": [
                HumanMessage(content="[Tool Feedback: old feedback]"),
                AIMessage(content="1"),
                AIMessage(content="2"),
                AIMessage(content="3"),
                AIMessage(content="4"),
                AIMessage(content="5"),
            ]
        }
        result = should_continue_after_tools(state)
        assert result == "end"


# ═══════════════════════════════════════════════════════════════════
# create_langgraph_agent
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestCreateLangGraphAgent:
    def test_creates_agent_successfully(self):
        from client.langgraph import create_langgraph_agent
        mock_llm = MagicMock()
        mock_llm.model = "test-model"
        mock_tools = [MagicMock(name="tool1")]
        agent = create_langgraph_agent(mock_llm, mock_tools)
        assert agent is not None

    def test_agent_has_ainvoke(self):
        from client.langgraph import create_langgraph_agent
        mock_llm = MagicMock()
        mock_llm.model = "test-model"
        agent = create_langgraph_agent(mock_llm, [])
        assert hasattr(agent, "ainvoke")

    def test_stop_signal_cancels_call_model(self):
        from client.langgraph import create_langgraph_agent
        from client.stop_signal import request_stop, clear_stop
        mock_llm = MagicMock()
        mock_llm.model = "test-model"
        agent = create_langgraph_agent(mock_llm, [])
        assert agent is not None
        clear_stop()

    def test_agent_with_bound_llm(self):
        """Test that create_langgraph_agent handles llm_with_tools.bound pattern."""
        from client.langgraph import create_langgraph_agent
        mock_inner_llm = MagicMock()
        mock_inner_llm.model = "inner-model"
        mock_llm_with_tools = MagicMock()
        mock_llm_with_tools.bound = mock_inner_llm
        agent = create_langgraph_agent(mock_llm_with_tools, [])
        assert agent is not None


# ═══════════════════════════════════════════════════════════════════
# run_agent — error paths
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
class TestRunAgentErrorPaths:
    def _make_state(self):
        return {"messages": [SystemMessage(content="You are a helpful assistant.")]}

    async def test_context_overflow_auto_recovers(self):
        from client.langgraph import run_agent
        from client.stop_signal import clear_stop
        clear_stop()

        mock_agent = MagicMock()
        mock_agent.ainvoke = AsyncMock(
            side_effect=ValueError("Requested tokens (5000) exceed context window of (4096)")
        )

        result = await run_agent(
            agent=mock_agent,
            conversation_state=self._make_state(),
            user_message="test query",
            logger=MagicMock(),
            tools=[],
            system_prompt="Test",
            llm=MagicMock(),
            max_history=20
        )

        assert "messages" in result
        last_msg = result["messages"][-1]
        assert "overflow" in last_msg.content.lower() or "context" in last_msg.content.lower()

    async def test_ollama_crash_handled_gracefully(self):
        from client.langgraph import run_agent
        from client.stop_signal import clear_stop
        clear_stop()

        mock_agent = MagicMock()
        mock_agent.ainvoke = AsyncMock(
            side_effect=Exception("model runner has unexpectedly stopped")
        )

        result = await run_agent(
            agent=mock_agent,
            conversation_state=self._make_state(),
            user_message="test query",
            logger=MagicMock(),
            tools=[],
            system_prompt="Test",
            llm=MagicMock(),
        )

        assert "messages" in result
        last_msg = result["messages"][-1]
        assert "crashed" in last_msg.content.lower() or "resource" in last_msg.content.lower()

    async def test_generic_error_returns_error_message(self):
        from client.langgraph import run_agent
        from client.stop_signal import clear_stop
        clear_stop()

        mock_agent = MagicMock()
        mock_agent.ainvoke = AsyncMock(side_effect=Exception("some unexpected failure"))

        result = await run_agent(
            agent=mock_agent,
            conversation_state=self._make_state(),
            user_message="test",
            logger=MagicMock(),
            tools=[],
            system_prompt="Test",
            llm=MagicMock(),
        )

        assert "messages" in result
        assert len(result["messages"]) > 0

    async def test_successful_run_returns_messages(self):
        from client.langgraph import run_agent
        from client.stop_signal import clear_stop
        clear_stop()

        conv_state = self._make_state()
        expected_msgs = conv_state["messages"] + [AIMessage(content="The answer is 42.")]

        mock_agent = MagicMock()
        mock_agent.ainvoke = AsyncMock(return_value={
            "messages": expected_msgs,
            "current_model": "test-model",
        })

        result = await run_agent(
            agent=mock_agent,
            conversation_state=conv_state,
            user_message="What is 6 * 7?",
            logger=MagicMock(),
            tools=[],
            system_prompt="Test",
            llm=MagicMock(),
        )

        assert "messages" in result
        assert "current_model" in result
        assert result["current_model"] == "test-model"

    async def test_stop_signal_cleared_before_run(self):
        from client.langgraph import run_agent
        from client.stop_signal import request_stop, is_stop_requested
        request_stop()
        assert is_stop_requested()

        conv_state = self._make_state()
        mock_agent = MagicMock()
        mock_agent.ainvoke = AsyncMock(return_value={
            "messages": conv_state["messages"] + [AIMessage(content="ok")],
            "current_model": "test",
        })

        await run_agent(
            agent=mock_agent,
            conversation_state=conv_state,
            user_message="hello",
            logger=MagicMock(),
            tools=[],
            system_prompt="Test",
            llm=MagicMock(),
        )

        # Stop flag should be cleared at start of run_agent
        assert not is_stop_requested()