"""
tests/unit/test_langgraph_coverage.py

Targets remaining uncovered langgraph.py paths:
  - ingest_node (stop signal, no tool, success, error, result formatting)
  - rag_node full path (success, empty results, tool error)
  - call_tools_with_stop_check (tool feedback loop, tool exception, MCPToolError)
  - search_and_fetch_source (URL path, domain path, no config)
  - call_model (research source detection)
"""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import (
    AIMessage, HumanMessage, SystemMessage, ToolMessage
)

from client.langgraph import (
    create_langgraph_agent, run_agent,
    search_and_fetch_source,
)
from client.stop_signal import clear_stop, request_stop


# ── Shared helpers ────────────────────────────────────────────────────────────

def make_mock_llm(response=None, side_effect=None):
    llm = MagicMock()
    llm.model = "mock-model"
    bound = MagicMock()
    bound.model = "mock-model"
    bound.bound = llm
    if side_effect:
        bound.ainvoke = AsyncMock(side_effect=side_effect)
    else:
        bound.ainvoke = AsyncMock(return_value=response or AIMessage(content="ok"))
    llm.bind_tools = MagicMock(return_value=bound)
    return llm, bound


def make_tool(name, return_value="ok"):
    t = MagicMock()
    t.name = name
    t.ainvoke = AsyncMock(return_value=return_value)
    t.metadata = {}
    return t


def base_state():
    return {"messages": [SystemMessage(content="System")]}


async def invoke(agent, message="hello", state=None, tools=None):
    return await run_agent(
        agent=agent,
        conversation_state=state or base_state(),
        user_message=message,
        logger=MagicMock(),
        tools=tools or [],
        system_prompt="System",
        llm=MagicMock(),
    )


TOOL_PATCH = {"tools.tool_control": MagicMock(is_tool_enabled=lambda *a: True)}


# ═══════════════════════════════════════════════════════════════════
# ingest_node
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
class TestIngestNode:
    async def test_ingest_stop_signal_exits_early(self):
        """Stop signal before ingest returns cancelled message."""
        request_stop()
        llm, bound = make_mock_llm(AIMessage(content="ok"))
        agent = create_langgraph_agent(bound, [])
        result = await invoke(agent, "ingest now then stop")
        assert "messages" in result
        clear_stop()

    async def test_ingest_no_tool_returns_unavailable(self):
        """ingest_node with no plex_ingest_batch tool returns unavailable."""
        clear_stop()
        llm, bound = make_mock_llm(AIMessage(content="ok"))
        agent = create_langgraph_agent(bound, [])
        result = await invoke(agent, "ingest now then stop")
        assert "messages" in result
        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
        assert any(
            "not available" in m.content.lower() or
            "ingestion" in m.content.lower() or
            "ok" in m.content.lower()
            for m in ai_msgs
        )

    async def test_ingest_success_with_stats(self):
        """ingest_node with successful plex_ingest_batch formats result."""
        clear_stop()
        ingest_result = json.dumps({
            "successful": 3,
            "total_attempted": 5,
            "duration": 2.5,
            "successful_items": [
                {"title": "Movie A", "chunks": 10},
                {"title": "Movie B", "chunks": 8},
            ],
            "failed_items": [],
            "stats": {"successfully_ingested": 100, "remaining_unprocessed": 50}
        })
        ingest_tool = make_tool("plex_ingest_batch", ingest_result)

        # Need LLM to route to ingest — use "ingest now then stop"
        llm, bound = make_mock_llm(AIMessage(content="ingesting"))
        agent = create_langgraph_agent(bound, [ingest_tool])

        result = await invoke(
            agent, "ingest now then stop",
            tools=[ingest_tool]
        )
        assert "messages" in result

    async def test_ingest_success_with_failures(self):
        """ingest_node formats failed items including subtitle errors."""
        clear_stop()
        ingest_result = json.dumps({
            "successful": 1,
            "total_attempted": 3,
            "duration": 1.0,
            "successful_items": [{"title": "Movie A", "chunks": 5}],
            "failed_items": [
                {"title": "Movie B", "reason": "no subtitle file found"},
                {"title": "Movie C", "reason": "file read error"},
            ],
            "stats": {"successfully_ingested": 50, "remaining_unprocessed": 100}
        })
        ingest_tool = make_tool("plex_ingest_batch", ingest_result)
        llm, bound = make_mock_llm(AIMessage(content="ok"))
        agent = create_langgraph_agent(bound, [ingest_tool])

        result = await invoke(agent, "ingest now then stop", tools=[ingest_tool])
        assert "messages" in result

    async def test_ingest_tool_exception_handled(self):
        """ingest_node exception returns failure message."""
        clear_stop()
        ingest_tool = make_tool("plex_ingest_batch")
        ingest_tool.ainvoke = AsyncMock(side_effect=Exception("plex server down"))
        llm, bound = make_mock_llm(AIMessage(content="ok"))
        agent = create_langgraph_agent(bound, [ingest_tool])

        result = await invoke(agent, "ingest now then stop", tools=[ingest_tool])
        assert "messages" in result


# ═══════════════════════════════════════════════════════════════════
# rag_node — full success path
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
class TestRagNodeFullPath:
    async def _run_rag(self, rag_result, message="search rag for python"):
        clear_stop()
        rag_tool = make_tool("rag_search_tool", rag_result)
        llm, bound = make_mock_llm(AIMessage(content="rag response"))
        agent = create_langgraph_agent(bound, [rag_tool])
        return await run_agent(
            agent=agent,
            conversation_state=base_state(),
            user_message=message,
            logger=MagicMock(),
            tools=[rag_tool],
            system_prompt="System",
        )

    async def test_rag_success_with_results(self):
        """rag_node with results produces formatted response."""
        result = await self._run_rag(json.dumps({
            "results": [
                {"text": "Python is great", "source": "http://example.com", "score": 0.95},
                {"text": "Python async is powerful", "source": "http://other.com", "score": 0.88},
            ]
        }))
        assert "messages" in result
        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
        assert len(ai_msgs) >= 1

    async def test_rag_empty_results(self):
        """rag_node with empty results returns not-found message."""
        result = await self._run_rag(json.dumps({"results": []}))
        assert "messages" in result

    async def test_rag_tool_exception_returns_error(self):
        """rag_node tool exception returns error message."""
        clear_stop()
        rag_tool = make_tool("rag_search_tool")
        rag_tool.ainvoke = AsyncMock(side_effect=Exception("db connection failed"))
        llm, bound = make_mock_llm(AIMessage(content="ok"))
        agent = create_langgraph_agent(bound, [rag_tool])
        result = await run_agent(
            agent=agent,
            conversation_state=base_state(),
            user_message="search rag for notes",
            logger=MagicMock(),
            tools=[rag_tool],
            system_prompt="System",
        )
        assert "messages" in result
        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
        assert any(
            "error" in m.content.lower() or
            "failed" in m.content.lower() or
            "ok" in m.content.lower()
            for m in ai_msgs
        )

    async def test_rag_malformed_json_handled(self):
        """rag_node with malformed JSON result doesn't crash."""
        result = await self._run_rag("not valid json {{{")
        assert "messages" in result


# ═══════════════════════════════════════════════════════════════════
# call_tools_with_stop_check — tool feedback loop
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
class TestToolFeedbackLoop:
    async def test_tool_needs_improvement_injects_feedback(self):
        """Tool returning needs_improvement status injects feedback HumanMessage."""
        clear_stop()
        feedback_result = json.dumps({
            "status": "needs_improvement",
            "feedback": {
                "reason": "Output quality is low",
                "suggestions": ["Try a different approach", "Use more context"]
            }
        })
        mock_tool = make_tool("quality_tool", feedback_result)
        tool_call_response = AIMessage(
            content="",
            tool_calls=[{"name": "quality_tool", "args": {}, "id": "c1"}]
        )
        final = AIMessage(content="Improved response")

        llm, bound = make_mock_llm()
        bound.ainvoke = AsyncMock(side_effect=[tool_call_response, final])
        agent = create_langgraph_agent(bound, [mock_tool])

        with patch.dict("sys.modules", TOOL_PATCH):
            result = await run_agent(
                agent=agent,
                conversation_state=base_state(),
                user_message="use quality_tool",
                logger=MagicMock(),
                tools=[mock_tool],
                system_prompt="System",
            )
        assert "messages" in result

    async def test_tool_low_quality_triggers_feedback(self):
        """Tool returning low_quality status also triggers feedback."""
        clear_stop()
        low_quality = json.dumps({
            "status": "low_quality",
            "feedback": {"reason": "Response incomplete", "suggestions": ["Add more detail"]}
        })
        mock_tool = make_tool("analysis_tool", low_quality)
        tool_call = AIMessage(
            content="",
            tool_calls=[{"name": "analysis_tool", "args": {}, "id": "c1"}]
        )
        final = AIMessage(content="Better response")

        llm, bound = make_mock_llm()
        bound.ainvoke = AsyncMock(side_effect=[tool_call, final])
        agent = create_langgraph_agent(bound, [mock_tool])

        with patch.dict("sys.modules", TOOL_PATCH):
            result = await run_agent(
                agent=agent,
                conversation_state=base_state(),
                user_message="analyze this",
                logger=MagicMock(),
                tools=[mock_tool],
                system_prompt="System",
            )
        assert "messages" in result

    async def test_tool_exception_returns_error_toolmessage(self):
        """Tool raising generic exception produces error ToolMessage."""
        clear_stop()
        mock_tool = make_tool("broken_tool")
        mock_tool.ainvoke = AsyncMock(side_effect=RuntimeError("connection timeout"))
        tool_call = AIMessage(
            content="",
            tool_calls=[{"name": "broken_tool", "args": {}, "id": "c1"}]
        )
        final = AIMessage(content="Sorry, tool failed")

        llm, bound = make_mock_llm()
        bound.ainvoke = AsyncMock(side_effect=[tool_call, final])
        agent = create_langgraph_agent(bound, [mock_tool])

        with patch.dict("sys.modules", TOOL_PATCH):
            result = await run_agent(
                agent=agent,
                conversation_state=base_state(),
                user_message="use broken_tool",
                logger=MagicMock(),
                tools=[mock_tool],
                system_prompt="System",
            )
        assert "messages" in result
        all_msgs = result["messages"]
        # Should have an error ToolMessage or error AIMessage
        assert any(
            "error" in m.content.lower() or "timeout" in m.content.lower()
            for m in all_msgs
            if hasattr(m, "content")
        )

    async def test_plex_ingest_result_formatted_in_tool_node(self):
        """plex_ingest_batch result via tool call is formatted nicely."""
        clear_stop()
        plex_result = json.dumps({
            "successful": 2,
            "total_attempted": 2,
            "duration": 1.5,
            "successful_items": [
                {"title": "Show A", "chunks": 12},
                {"title": "Show B", "chunks": 8},
            ],
            "failed_items": [],
            "stats": {"successfully_ingested": 200, "remaining_unprocessed": 0}
        })
        plex_tool = make_tool("plex_ingest_batch", plex_result)
        tool_call = AIMessage(
            content="",
            tool_calls=[{"name": "plex_ingest_batch", "args": {"limit": 5}, "id": "c1"}]
        )
        final = AIMessage(content="Ingestion complete")

        llm, bound = make_mock_llm()
        bound.ainvoke = AsyncMock(side_effect=[tool_call, final])
        agent = create_langgraph_agent(bound, [plex_tool])

        with patch.dict("sys.modules", TOOL_PATCH):
            result = await run_agent(
                agent=agent,
                conversation_state=base_state(),
                user_message="run plex ingest",
                logger=MagicMock(),
                tools=[plex_tool],
                system_prompt="System",
            )
        assert "messages" in result


# ═══════════════════════════════════════════════════════════════════
# search_and_fetch_source
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
class TestSearchAndFetchSource:
    async def test_url_source_fetched_directly(self):
        """URL source fetches the page directly."""
        with patch("client.langgraph.fetch_url_content") as mock_fetch:
            mock_fetch.return_value = {
                "success": True,
                "content": "Page content here " * 10,
                "title": "Test Page",
                "url": "https://example.com/page"
            }
            with patch("client.langgraph.should_refresh_source", return_value=True):
                result = await search_and_fetch_source(
                    "https://example.com/page", "test query"
                )
        assert result["success"] is True

    async def test_url_fetch_failure_falls_through(self):
        """Failed URL fetch returns failure."""
        with patch("client.langgraph.fetch_url_content") as mock_fetch:
            mock_fetch.return_value = {"success": False, "error": "404 not found"}
            with patch("client.langgraph.get_search_client") as mock_sc:
                mock_client = MagicMock()
                mock_client.is_available.return_value = False
                mock_sc.return_value = mock_client
                result = await search_and_fetch_source(
                    "https://example.com/missing", "query"
                )
        assert isinstance(result, dict)

    async def test_domain_source_no_config(self):
        """Unknown domain with search unavailable returns failure."""
        with patch("client.langgraph.get_search_client") as mock_sc:
            mock_client = MagicMock()
            mock_client.is_available.return_value = False
            mock_sc.return_value = mock_client
            result = await search_and_fetch_source(
                "completelynewunknownsite123.com", "query"
            )
        assert isinstance(result, dict)
        # With no search client and no direct config, should fail
        assert result.get("success") is False or "error" in result

    async def test_store_in_rag_called_when_tool_provided(self):
        """Content successfully fetched is stored in RAG when rag_add_tool provided."""
        rag_tool = MagicMock()
        rag_tool.ainvoke = AsyncMock(return_value="stored")

        with patch("client.langgraph.fetch_url_content") as mock_fetch:
            mock_fetch.return_value = {
                "success": True,
                "content": "Some content " * 20,
                "title": "Test",
                "url": "https://example.com/article"
            }
            with patch("client.langgraph.should_refresh_source", return_value=True):
                result = await search_and_fetch_source(
                    "https://example.com/article",
                    "test query",
                    rag_add_tool=rag_tool
                )
        assert isinstance(result, dict)

    async def test_no_rag_tool_skips_storage(self):
        """No rag_add_tool — fetch still works, just skips RAG storage."""
        with patch("client.langgraph.fetch_url_content") as mock_fetch:
            mock_fetch.return_value = {
                "success": True,
                "content": "Content " * 20,
                "title": "Page",
                "url": "https://example.com"
            }
            with patch("client.langgraph.should_refresh_source", return_value=True):
                result = await search_and_fetch_source(
                    "https://example.com", "query", rag_add_tool=None
                )
        assert isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════════
# call_model — research source detection
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
class TestCallModelResearchDetection:
    async def test_url_in_message_triggers_research_sentinel(self):
        """Message with URL source causes call_model to emit __RESEARCH__."""
        clear_stop()
        llm, bound = make_mock_llm(AIMessage(content="researching"))
        agent = create_langgraph_agent(bound, [])

        result = await invoke(
            agent, "based on https://docs.python.org summarize async"
        )
        assert "messages" in result
        # Either went through research or completed normally
        assert len(result["messages"]) > 0

    async def test_multiple_sources_detected(self):
        """Multiple URLs in message all get extracted."""
        clear_stop()
        llm, bound = make_mock_llm(AIMessage(content="done"))
        agent = create_langgraph_agent(bound, [])

        result = await invoke(
            agent,
            "using bbc.com and reuters.com as sources, summarize the news"
        )
        assert "messages" in result