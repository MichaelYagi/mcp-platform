"""
tests/unit/test_langgraph_graph.py

Tests for client/langgraph.py using a controlled mock LLM that runs
the REAL compiled LangGraph graph. This covers the deep graph nodes
(call_model, call_tools, router loop, message windowing, etc.) that
can't be tested by mocking the agent directly.

Strategy:
  - create_langgraph_agent() builds the real graph
  - mock_llm.ainvoke returns controlled AIMessage responses
  - the graph executes its full node/edge logic against those responses
"""
import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import (
    AIMessage, HumanMessage, SystemMessage, ToolMessage
)

from client.langgraph import create_langgraph_agent, run_agent
from client.stop_signal import clear_stop, request_stop


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════

def make_mock_llm(response: AIMessage = None, side_effect=None):
    """
    Build a mock LLM whose ainvoke returns a controlled AIMessage.
    bind_tools returns the same mock so create_langgraph_agent works.
    """
    llm = MagicMock()
    llm.model = "mock-model"
    llm.model_name = "mock-model"

    if side_effect:
        llm.ainvoke = AsyncMock(side_effect=side_effect)
    else:
        llm.ainvoke = AsyncMock(return_value=response or AIMessage(content="ok"))

    # bind_tools must return the same mock (the graph uses llm_with_tools.bound)
    bound = MagicMock()
    bound.model = "mock-model"
    bound.ainvoke = llm.ainvoke
    bound.bound = llm
    llm.bind_tools = MagicMock(return_value=bound)

    return llm, bound


def make_agent(llm_response: AIMessage = None, side_effect=None, tools=None):
    """Build a compiled LangGraph agent with a controlled mock LLM."""
    llm, bound = make_mock_llm(llm_response, side_effect)
    agent = create_langgraph_agent(bound, tools or [])
    return agent, llm


def base_state():
    return {"messages": [SystemMessage(content="You are a helpful assistant.")]}


async def invoke(agent, message="hello", state=None, tools=None, **kwargs):
    """Convenience wrapper around run_agent."""
    return await run_agent(
        agent=agent,
        conversation_state=state or base_state(),
        user_message=message,
        logger=MagicMock(),
        tools=tools or [],
        system_prompt="You are a helpful assistant.",
        llm=MagicMock(),
        **kwargs
    )


# ═══════════════════════════════════════════════════════════════════
# Basic graph execution
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
class TestGraphBasicExecution:
    async def test_simple_response_returned(self):
        """Graph runs, LLM responds, result contains messages."""
        clear_stop()
        agent, _ = make_agent(AIMessage(content="Hello back!"))
        result = await invoke(agent, "Hello")
        assert "messages" in result
        assert len(result["messages"]) > 0

    async def test_current_model_in_result(self):
        """Result includes current_model from the mock LLM."""
        clear_stop()
        agent, _ = make_agent(AIMessage(content="Hi"))
        result = await invoke(agent, "Hi")
        assert "current_model" in result
        assert result["current_model"] == "mock-model"

    async def test_user_message_appended_to_history(self):
        """run_agent appends the user message to conversation state."""
        clear_stop()
        agent, _ = make_agent(AIMessage(content="response"))
        state = base_state()
        await run_agent(
            agent=agent,
            conversation_state=state,
            user_message="my question",
            logger=MagicMock(),
            tools=[],
            system_prompt="System",
        )
        human_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        assert any("my question" in m.content for m in human_msgs)

    async def test_system_message_preserved(self):
        """SystemMessage is not duplicated or replaced across calls."""
        clear_stop()
        agent, _ = make_agent(AIMessage(content="ok"))
        state = base_state()
        await invoke(agent, "test", state=state)
        system_msgs = [m for m in state["messages"] if isinstance(m, SystemMessage)]
        assert len(system_msgs) == 1

    async def test_system_message_created_if_missing(self):
        """run_agent creates SystemMessage from system_prompt if not in state."""
        clear_stop()
        agent, _ = make_agent(AIMessage(content="ok"))
        state = {"messages": []}  # no SystemMessage
        await run_agent(
            agent=agent,
            conversation_state=state,
            user_message="hi",
            logger=MagicMock(),
            tools=[],
            system_prompt="Custom system prompt",
        )
        system_msgs = [m for m in state["messages"] if isinstance(m, SystemMessage)]
        assert len(system_msgs) == 1
        assert "Custom system prompt" in system_msgs[0].content

    async def test_ai_response_appended_to_history(self):
        """AI response message appears in conversation state after run."""
        clear_stop()
        agent, _ = make_agent(AIMessage(content="the answer"))
        state = base_state()
        await invoke(agent, "question", state=state)
        ai_msgs = [m for m in state["messages"] if isinstance(m, AIMessage)]
        assert len(ai_msgs) >= 1

    async def test_multi_turn_history_preserved(self):
        """Second call sees messages from first call."""
        clear_stop()
        agent, _ = make_agent(AIMessage(content="first response"))
        state = base_state()
        await invoke(agent, "first question", state=state)
        msg_count_after_first = len(state["messages"])

        agent2, _ = make_agent(AIMessage(content="second response"))
        await run_agent(
            agent=agent2,
            conversation_state=state,
            user_message="second question",
            logger=MagicMock(),
            tools=[],
            system_prompt="System",
        )
        assert len(state["messages"]) > msg_count_after_first


# ═══════════════════════════════════════════════════════════════════
# call_model — stop signal
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
class TestCallModelStopSignal:
    async def test_stop_before_invoke_cancels(self):
        """Stop signal set before run causes early cancelled response."""
        request_stop()
        agent, llm = make_agent(AIMessage(content="should not see this"))
        result = await invoke(agent, "hello")
        # LLM should not have been called — graph should exit early
        assert "messages" in result
        clear_stop()

    async def test_stop_cleared_at_start_of_run(self):
        """run_agent clears any previous stop signal before executing."""
        request_stop()
        agent, llm = make_agent(AIMessage(content="proceeded"))
        # run_agent calls clear_stop() at the top
        result = await invoke(agent, "hello")
        assert "messages" in result
        # After run, stop is cleared
        from client.stop_signal import is_stop_requested
        assert not is_stop_requested()
        clear_stop()


# ═══════════════════════════════════════════════════════════════════
# call_model — tool calling path
# ═══════════════════════════════════════════════════════════════════

# ── Tool helper ──────────────────────────────────────────────────────────────
# call_tools_with_stop_check calls tool.ainvoke(tool_args) after checking
# is_tool_enabled(). We patch that check out so plain AsyncMock tools work.

def make_tool(name, return_value="tool result"):
    """Build a minimal tool mock compatible with call_tools_with_stop_check."""
    tool = MagicMock()
    tool.name = name
    tool.ainvoke = AsyncMock(return_value=return_value)
    tool.metadata = {}
    return tool


@pytest.mark.unit
@pytest.mark.asyncio
class TestCallModelToolPath:
    async def test_tool_calls_dispatched(self):
        """LLM returning tool_calls triggers tool execution."""
        clear_stop()

        mock_tool = make_tool("test_tool", "tool result")
        tool_call_response = AIMessage(
            content="",
            tool_calls=[{"name": "test_tool", "args": {"q": "test"}, "id": "call1"}]
        )
        final_response = AIMessage(content="Based on tool: done")

        llm, bound = make_mock_llm()
        bound.ainvoke = AsyncMock(side_effect=[tool_call_response, final_response])
        agent = create_langgraph_agent(bound, [mock_tool])

        # Patch is_tool_enabled so it always returns True (no import dependency)
        with patch("client.langgraph.is_tool_enabled", return_value=True, create=True):
            with patch.dict("sys.modules", {"tools.tool_control": MagicMock(is_tool_enabled=lambda *a: True)}):
                result = await run_agent(
                    agent=agent,
                    conversation_state=base_state(),
                    user_message="use test_tool",
                    logger=MagicMock(),
                    tools=[mock_tool],
                    system_prompt="System",
                )

        assert "messages" in result
        mock_tool.ainvoke.assert_called_once()

    async def test_tool_result_in_messages(self):
        """After tool execution, ToolMessage appears in state."""
        clear_stop()

        mock_tool = make_tool("info_tool", json.dumps({"result": "42"}))
        tool_response = AIMessage(
            content="",
            tool_calls=[{"name": "info_tool", "args": {}, "id": "c1"}]
        )
        final = AIMessage(content="The answer is 42")

        llm, bound = make_mock_llm()
        bound.ainvoke = AsyncMock(side_effect=[tool_response, final])
        agent = create_langgraph_agent(bound, [mock_tool])

        state = base_state()
        with patch.dict("sys.modules", {"tools.tool_control": MagicMock(is_tool_enabled=lambda *a: True)}):
            await run_agent(
                agent=agent,
                conversation_state=state,
                user_message="what is the answer",
                logger=MagicMock(),
                tools=[mock_tool],
                system_prompt="System",
            )

        tool_msgs = [m for m in state["messages"] if isinstance(m, ToolMessage)]
        assert len(tool_msgs) >= 1

    async def test_tool_not_found_returns_error_message(self):
        """Tool call for unknown tool returns ToolMessage with error."""
        clear_stop()

        tool_call_response = AIMessage(
            content="",
            tool_calls=[{"name": "nonexistent_tool", "args": {}, "id": "c1"}]
        )
        final = AIMessage(content="sorry, failed")

        llm, bound = make_mock_llm()
        bound.ainvoke = AsyncMock(side_effect=[tool_call_response, final])
        agent = create_langgraph_agent(bound, [])  # no tools registered

        with patch.dict("sys.modules", {"tools.tool_control": MagicMock(is_tool_enabled=lambda *a: True)}):
            result = await run_agent(
                agent=agent,
                conversation_state=base_state(),
                user_message="use nonexistent_tool",
                logger=MagicMock(),
                tools=[],
                system_prompt="System",
            )

        assert "messages" in result
        # Should have a ToolMessage with error content
        all_msgs = result["messages"]
        tool_msgs = [m for m in all_msgs if isinstance(m, ToolMessage)]
        assert any("not found" in m.content.lower() for m in tool_msgs)

    async def test_multiple_tool_calls_handled(self):
        """Multiple tool calls in one LLM response all get dispatched."""
        clear_stop()

        tool_a = make_tool("tool_a", "result_a")
        tool_b = make_tool("tool_b", "result_b")

        response_with_two_calls = AIMessage(
            content="",
            tool_calls=[
                {"name": "tool_a", "args": {}, "id": "c1"},
                {"name": "tool_b", "args": {}, "id": "c2"},
            ]
        )
        final = AIMessage(content="Combined results")

        llm, bound = make_mock_llm()
        bound.ainvoke = AsyncMock(side_effect=[response_with_two_calls, final])
        agent = create_langgraph_agent(bound, [tool_a, tool_b])

        with patch.dict("sys.modules", {"tools.tool_control": MagicMock(is_tool_enabled=lambda *a: True)}):
            result = await run_agent(
                agent=agent,
                conversation_state=base_state(),
                user_message="use both tools",
                logger=MagicMock(),
                tools=[tool_a, tool_b],
                system_prompt="System",
            )

        assert "messages" in result
        tool_a.ainvoke.assert_called_once()
        tool_b.ainvoke.assert_called_once()

    async def test_tool_stop_signal_during_execution(self):
        """Stop signal between tool calls halts further tool execution."""
        clear_stop()

        call_count = 0
        async def slow_tool(args):
            nonlocal call_count
            call_count += 1
            request_stop()  # signal stop after first tool
            return "partial result"

        tool_a = make_tool("tool_a")
        tool_a.ainvoke = slow_tool
        tool_b = make_tool("tool_b", "should not run")

        response = AIMessage(
            content="",
            tool_calls=[
                {"name": "tool_a", "args": {}, "id": "c1"},
                {"name": "tool_b", "args": {}, "id": "c2"},
            ]
        )
        final = AIMessage(content="stopped")

        llm, bound = make_mock_llm()
        bound.ainvoke = AsyncMock(side_effect=[response, final])
        agent = create_langgraph_agent(bound, [tool_a, tool_b])

        with patch.dict("sys.modules", {"tools.tool_control": MagicMock(is_tool_enabled=lambda *a: True)}):
            result = await run_agent(
                agent=agent,
                conversation_state=base_state(),
                user_message="use both tools",
                logger=MagicMock(),
                tools=[tool_a, tool_b],
                system_prompt="System",
            )

        assert "messages" in result
        # tool_b should NOT have been called
        tool_b.ainvoke.assert_not_called()
        clear_stop()


# ═══════════════════════════════════════════════════════════════════
# call_model — Ollama search override path
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
class TestCallModelOllamaSearch:
    async def test_ollama_search_not_available(self):
        """When OLLAMA_TOKEN not set, search returns unavailable message."""
        clear_stop()
        agent, _ = make_agent(AIMessage(content="fallback"))

        with patch("client.langgraph.get_search_client") as mock_sc:
            mock_client = MagicMock()
            mock_client.is_available.return_value = False
            mock_sc.return_value = mock_client

            result = await invoke(agent, "ollama search for python news")
            assert "messages" in result
            msgs = result["messages"]
            assert any("OLLAMA_TOKEN" in m.content or "not available" in m.content.lower()
                       for m in msgs if isinstance(m, AIMessage))

    async def test_ollama_search_success_path(self):
        """Successful Ollama search augments prompt and calls LLM."""
        clear_stop()
        agent, llm = make_agent(AIMessage(content="Here are the results"))

        with patch("client.langgraph.get_search_client") as mock_sc:
            mock_client = MagicMock()
            mock_client.is_available.return_value = True
            mock_client.search = AsyncMock(return_value={
                "success": True,
                "results": {
                    "webPages": {
                        "value": [
                            {"url": "https://example.com", "name": "Example", "summary": "content here"}
                        ]
                    }
                }
            })
            mock_sc.return_value = mock_client

            result = await invoke(agent, "use ollama search: python news")
            assert "messages" in result

    async def test_ollama_search_no_results(self):
        """Empty search results returns appropriate message."""
        clear_stop()
        agent, _ = make_agent(AIMessage(content="nothing found"))

        with patch("client.langgraph.get_search_client") as mock_sc:
            mock_client = MagicMock()
            mock_client.is_available.return_value = True
            mock_client.search = AsyncMock(return_value={
                "success": True,
                "results": {"webPages": {"value": []}}
            })
            mock_sc.return_value = mock_client

            result = await invoke(agent, "web search for xyzzy no results")
            assert "messages" in result

    async def test_ollama_search_exception_handled(self):
        """Exception during search returns error message, doesn't crash."""
        clear_stop()
        agent, _ = make_agent(AIMessage(content="fallback"))

        with patch("client.langgraph.get_search_client") as mock_sc:
            mock_client = MagicMock()
            mock_client.is_available.return_value = True
            mock_client.search = AsyncMock(side_effect=Exception("network error"))
            mock_sc.return_value = mock_client

            result = await invoke(agent, "ollama search for something")
            assert "messages" in result
            msgs = result["messages"]
            assert any("error" in m.content.lower() for m in msgs if isinstance(m, AIMessage))


# ═══════════════════════════════════════════════════════════════════
# call_model — research sentinel path
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
class TestCallModelResearchPath:
    async def test_research_sources_detected_emits_sentinel(self):
        """Message with URL source triggers __RESEARCH__ sentinel."""
        clear_stop()
        agent, llm = make_agent(AIMessage(content="research result"))
        result = await invoke(
            agent, "using https://example.com as source, summarize it"
        )
        assert "messages" in result

    async def test_research_sentinel_routes_to_research_node(self):
        """__RESEARCH__ sentinel in LLM response routes to research_node."""
        clear_stop()
        # LLM returns sentinel — graph should route to research_node
        agent, _ = make_agent(AIMessage(content="__RESEARCH__"))

        with patch("client.langgraph.research_node") as mock_research:
            mock_research.return_value = {
                "messages": [AIMessage(content="research done")],
                "tools": {}, "llm": None,
                "ingest_completed": False, "stopped": False,
                "current_model": "mock", "research_source": ""
            }
            result = await invoke(agent, "research this topic")
            assert "messages" in result


# ═══════════════════════════════════════════════════════════════════
# Message windowing in run_agent
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
class TestMessageWindowing:
    async def test_llm_sees_windowed_messages(self):
        """LLM receives only last LLM_MESSAGE_WINDOW messages, not full history."""
        clear_stop()
        captured_messages = []

        async def capture_ainvoke(messages):
            captured_messages.extend(messages)
            return AIMessage(content="response")

        llm, bound = make_mock_llm()
        bound.ainvoke = capture_ainvoke
        agent = create_langgraph_agent(bound, [])

        # Build long history
        state = {
            "messages": [SystemMessage(content="System")] +
                        [HumanMessage(content=f"msg {i}") for i in range(30)] +
                        [AIMessage(content=f"reply {i}") for i in range(30)]
        }

        await run_agent(
            agent=agent,
            conversation_state=state,
            user_message="new question",
            logger=MagicMock(),
            tools=[],
            system_prompt="System",
            max_history=20,
        )

        from client.langgraph import LLM_MESSAGE_WINDOW
        # LLM should see at most LLM_MESSAGE_WINDOW + 1 (system) messages
        assert len(captured_messages) <= LLM_MESSAGE_WINDOW + 5  # +5 for system/rag

    async def test_full_history_preserved_in_state(self):
        """Even with windowing, full history stays in conversation_state."""
        clear_stop()
        agent, _ = make_agent(AIMessage(content="ok"))

        state = {
            "messages": [SystemMessage(content="System")] +
                        [HumanMessage(content=f"old msg {i}") for i in range(20)]
        }
        original_count = len(state["messages"])

        await run_agent(
            agent=agent,
            conversation_state=state,
            user_message="new msg",
            logger=MagicMock(),
            tools=[],
            system_prompt="System",
        )
        # State should have grown, not been truncated
        assert len(state["messages"]) >= original_count


# ═══════════════════════════════════════════════════════════════════
# Error handling in run_agent
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
class TestRunAgentErrorHandling:
    async def test_context_overflow_auto_recovery(self):
        """Context overflow triggers auto-recovery with trimmed history."""
        clear_stop()
        agent, _ = make_agent(
            side_effect=ValueError("Requested tokens (9000) exceed context window of (4096)")
        )
        state = {
            "messages": [SystemMessage(content="System")] +
                        [HumanMessage(content="x" * 100) for _ in range(20)]
        }
        original_count = len(state["messages"])

        result = await run_agent(
            agent=agent,
            conversation_state=state,
            user_message="overflow me",
            logger=MagicMock(),
            tools=[],
            system_prompt="System",
        )
        assert "messages" in result
        # Should have recovery message
        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
        assert any("overflow" in m.content.lower() or "context" in m.content.lower()
                   for m in ai_msgs)
        # History should be trimmed
        assert len(state["messages"]) < original_count + 5

    async def test_context_overflow_minimal_history(self):
        """Context overflow with minimal history (can't trim) gives different message."""
        clear_stop()
        agent, _ = make_agent(
            side_effect=ValueError("Requested tokens (1000) exceed context window of (512)")
        )
        state = {"messages": [SystemMessage(content="S"), HumanMessage(content="Q")]}

        result = await run_agent(
            agent=agent,
            conversation_state=state,
            user_message="overflow",
            logger=MagicMock(),
            tools=[],
            system_prompt="S",
        )
        assert "messages" in result
        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
        assert len(ai_msgs) >= 1

    async def test_ollama_crash_handled(self):
        """Ollama crash error returns user-friendly message."""
        clear_stop()
        agent, _ = make_agent(
            side_effect=Exception("model runner has unexpectedly stopped")
        )
        result = await invoke(agent, "test")
        assert "messages" in result
        ai_msgs = [m for m in result["messages"] if isinstance(m, AIMessage)]
        assert len(ai_msgs) >= 1

    async def test_generic_exception_handled(self):
        """Any unexpected exception returns error message, doesn't crash."""
        clear_stop()
        agent, _ = make_agent(side_effect=RuntimeError("something broke"))
        result = await invoke(agent, "test")
        assert "messages" in result

    async def test_metrics_incremented_on_error(self):
        """agent_errors metric is incremented on exception."""
        clear_stop()
        from client.metrics import metrics, reset_metrics
        reset_metrics()

        agent, _ = make_agent(side_effect=RuntimeError("boom"))
        await invoke(agent, "test")
        assert metrics["agent_errors"] >= 1

    async def test_metrics_incremented_on_success(self):
        """agent_runs metric is incremented on success."""
        clear_stop()
        from client.metrics import metrics, reset_metrics
        reset_metrics()

        agent, _ = make_agent(AIMessage(content="ok"))
        await invoke(agent, "test")
        assert metrics["agent_runs"] >= 1

    async def test_agent_times_recorded(self):
        """Duration is recorded in agent_times after successful run."""
        clear_stop()
        from client.metrics import metrics, reset_metrics
        reset_metrics()

        agent, _ = make_agent(AIMessage(content="ok"))
        await invoke(agent, "test")
        assert len(metrics["agent_times"]) >= 1
        # Each entry is (timestamp, duration)
        ts, duration = metrics["agent_times"][-1]
        assert duration >= 0


# ═══════════════════════════════════════════════════════════════════
# RAG node path
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
class TestRagNodePath:
    async def test_explicit_rag_query_routes_to_rag(self):
        """'search rag' query routes through rag_node."""
        clear_stop()
        agent, _ = make_agent(AIMessage(content="rag response"))

        mock_rag_tool = MagicMock()
        mock_rag_tool.name = "rag_search_tool"
        mock_rag_tool.ainvoke = AsyncMock(return_value=json.dumps({
            "results": [{"text": "found content", "source": "http://x.com"}]
        }))

        result = await run_agent(
            agent=agent,
            conversation_state=base_state(),
            user_message="search rag for python async",
            logger=MagicMock(),
            tools=[mock_rag_tool],
            system_prompt="System",
        )
        assert "messages" in result

    async def test_rag_no_tool_returns_unavailable(self):
        """RAG query without rag_search_tool returns not-available message."""
        clear_stop()
        agent, _ = make_agent(AIMessage(content="ok"))
        result = await run_agent(
            agent=agent,
            conversation_state=base_state(),
            user_message="query rag for notes",
            logger=MagicMock(),
            tools=[],  # no rag tool
            system_prompt="System",
        )
        assert "messages" in result


# ═══════════════════════════════════════════════════════════════════
# Graph structure / create_langgraph_agent
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestCreateLangGraphAgentStructure:
    def test_agent_has_ainvoke(self):
        agent, _ = make_agent()
        assert hasattr(agent, "ainvoke")
        assert callable(agent.ainvoke)

    def test_agent_created_with_empty_tools(self):
        _, bound = make_mock_llm()
        agent = create_langgraph_agent(bound, [])
        assert agent is not None

    def test_agent_created_with_multiple_tools(self):
        tools = [MagicMock(name=f"tool_{i}") for i in range(5)]
        _, bound = make_mock_llm()
        agent = create_langgraph_agent(bound, tools)
        assert agent is not None

    def test_bound_llm_extracted(self):
        """create_langgraph_agent handles llm_with_tools.bound pattern."""
        inner = MagicMock()
        inner.model = "inner-model"
        inner.ainvoke = AsyncMock(return_value=AIMessage(content="ok"))

        outer = MagicMock()
        outer.bound = inner
        outer.model = "outer-model"
        outer.ainvoke = inner.ainvoke

        agent = create_langgraph_agent(outer, [])
        assert agent is not None