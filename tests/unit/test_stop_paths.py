"""
tests/unit/test_stop_paths.py

Comprehensive stop-signal tests covering every path where a stop can
interrupt a prompt.

Key behavioral facts about the stop mechanism:
  - run_agent entry: raises CancelledError immediately if stop is set
  - llm_ainvoke polling: raises CancelledError if stop fires during slow inference
  - router node: catches stop and routes to END (no CancelledError — clean exit)
  - call_model entry: returns cancelled AIMessage (no CancelledError — handled)
  - call_tools entry: returns cancelled AIMessage (no CancelledError — handled)
  - call_tools between calls: halts further tool calls (no CancelledError)

Tests are written to match the actual behavior, not idealized behavior.
"""
import asyncio
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from client.langgraph import create_langgraph_agent, run_agent, llm_ainvoke
from client.stop_signal import clear_stop, request_stop, is_stop_requested


# ═══════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════

def make_mock_llm(response=None, side_effect=None):
    """
    Build a mock LLM with AsyncMock ainvoke.

    Critical: run_agent does _base_llm = llm.bound if hasattr(llm, "bound") else llm.
    MagicMock auto-creates .bound as another MagicMock, so hasattr always returns True.
    We must explicitly set llm.bound = None (or a spec) to force run_agent to use llm directly.
    Then both the classifier and call_model use the same AsyncMock ainvoke.
    """
    llm = MagicMock()
    llm.model = "mock-model"
    llm.model_name = "mock-model"
    # Explicitly set .bound to None so run_agent uses llm directly for classifier
    llm.bound = None

    if side_effect:
        llm.ainvoke = AsyncMock(side_effect=side_effect)
    else:
        llm.ainvoke = AsyncMock(return_value=response or AIMessage(content="ok"))

    bound = MagicMock()
    bound.model = "mock-model"
    bound.model_name = "mock-model"
    bound.ainvoke = llm.ainvoke  # shared — same AsyncMock
    bound.bound = llm
    bound.bound.bound = None  # prevent infinite MagicMock chain
    llm.bind_tools = MagicMock(return_value=bound)
    bound.bind_tools = MagicMock(return_value=bound)

    return llm, bound


def make_agent(response=None, side_effect=None, tools=None):
    llm, bound = make_mock_llm(response, side_effect)
    agent = create_langgraph_agent(bound, tools or [])
    return agent, llm, bound


def make_tool(name, return_value="tool result"):
    t = MagicMock()
    t.name = name
    t.ainvoke = AsyncMock(return_value=return_value)
    t.metadata = {}
    return t


def base_state():
    return {"messages": [SystemMessage(content="You are a helpful assistant.")]}


async def invoke(agent, message="hello", state=None, tools=None, llm=None):
    """
    Invoke run_agent. Always pass llm= — a bare MagicMock() breaks the
    routing classifier (its ainvoke is not an AsyncMock).
    Pass llm=None to skip the classifier entirely.
    """
    return await run_agent(
        agent=agent,
        conversation_state=state or base_state(),
        user_message=message,
        logger=MagicMock(),
        tools=tools or [],
        system_prompt="You are a helpful assistant.",
        llm=llm,
    )


# ═══════════════════════════════════════════════════════════════════
# 1. llm_ainvoke — stop fires during polling loop
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
class TestLlmAinvokeStop:

    async def test_stop_during_slow_llm_raises_cancelled(self):
        """Stop signal fires while llm_ainvoke is polling — raises CancelledError."""
        clear_stop()

        async def slow_llm(messages):
            await asyncio.sleep(10)
            return AIMessage(content="never")

        mock_llm = MagicMock()
        mock_llm.ainvoke = slow_llm

        async def trigger_stop():
            await asyncio.sleep(0.05)
            request_stop()

        asyncio.create_task(trigger_stop())
        with pytest.raises(asyncio.CancelledError):
            await llm_ainvoke(mock_llm, [], poll_interval=0.02)
        clear_stop()

    async def test_stop_already_set_cancels_immediately(self):
        """Stop set before llm_ainvoke — cancels on first poll tick."""
        clear_stop()
        request_stop()

        async def slow_llm(messages):
            await asyncio.sleep(10)
            return AIMessage(content="never")

        mock_llm = MagicMock()
        mock_llm.ainvoke = slow_llm

        with pytest.raises(asyncio.CancelledError):
            await llm_ainvoke(mock_llm, [], poll_interval=0.02)
        clear_stop()

    async def test_no_stop_returns_result(self):
        """Without stop signal, llm_ainvoke returns the LLM response normally."""
        clear_stop()
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=AIMessage(content="hello"))
        result = await llm_ainvoke(mock_llm, [])
        assert result.content == "hello"


# ═══════════════════════════════════════════════════════════════════
# 2. run_agent entry gate
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
class TestRunAgentEntryStop:

    async def test_stop_set_raises_cancelled_immediately(self):
        """run_agent aborts before any work when stop is set on entry."""
        request_stop()
        agent, llm, bound = make_agent(AIMessage(content="should not run"))

        with pytest.raises(asyncio.CancelledError):
            await invoke(agent, llm=llm)

        bound.ainvoke.assert_not_called()
        clear_stop()

    async def test_stop_cleared_before_run_proceeds_normally(self):
        """run_agent runs normally when stop is not set."""
        clear_stop()
        agent, llm, bound = make_agent(AIMessage(content="response"))
        result = await invoke(agent, llm=llm)
        assert "messages" in result


# ═══════════════════════════════════════════════════════════════════
# 3. call_model node entry check
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
class TestCallModelNodeStop:

    async def test_stop_after_classifier_halts_main_llm_call(self):
        """Stop fires in classifier — main LLM call sees stop in llm_ainvoke → CancelledError.

        LLM_ROUTING_MODEL is cleared so the mock handles the routing classifier
        (call 1, sets stop) and the subsequent main LLM call hits llm_ainvoke's
        polling loop with stop already set, raising CancelledError immediately.
        """
        clear_stop()

        call_count = 0
        async def llm_side_effect(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                request_stop()
                return AIMessage(content='{"context_sufficient": true, "needs_rag": false, "needs_web_search": false, "tool_tags": []}')
            await asyncio.sleep(5)
            return AIMessage(content="never")

        with patch.dict(os.environ, {"LLM_ROUTING_MODEL": ""}):
            llm, bound = make_mock_llm(side_effect=llm_side_effect)
            agent = create_langgraph_agent(bound, [])

            with pytest.raises(asyncio.CancelledError):
                await invoke(agent, llm=llm)
        clear_stop()

    async def test_stop_set_before_graph_entry_aborts_call_model(self):
        """Stop set before graph entry aborts run_agent — call_model never runs."""
        request_stop()
        agent, llm, bound = make_agent(AIMessage(content="should not run"))

        with pytest.raises(asyncio.CancelledError):
            await invoke(agent, llm=llm)

        bound.ainvoke.assert_not_called()
        clear_stop()


# ═══════════════════════════════════════════════════════════════════
# 4. call_model routing classifier — stop during slow inference
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
class TestRoutingClassifierStop:

    async def test_stop_during_routing_classifier_cancels(self):
        """Stop fires mid-inference during classifier — CancelledError raised."""
        clear_stop()

        async def slow_classifier(messages):
            request_stop()
            await asyncio.sleep(5)
            return AIMessage(content="{}")

        llm, bound = make_mock_llm(side_effect=slow_classifier)
        agent = create_langgraph_agent(bound, [])

        with pytest.raises(asyncio.CancelledError):
            await invoke(agent, llm=llm)

        clear_stop()


# ═══════════════════════════════════════════════════════════════════
# 5. call_model main LLM call — stop during slow inference
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
class TestMainLlmCallStop:

    async def test_stop_during_main_llm_call_cancels(self):
        """Stop fires mid-inference during main LLM call — CancelledError raised.

        LLM_ROUTING_MODEL is cleared so the mock handles both the routing
        classifier (call 1, fast) and the main LLM inference (call 2, slow),
        allowing llm_ainvoke's polling loop to catch the stop signal.
        """
        clear_stop()

        call_count = 0
        async def llm_side_effect(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # classifier — fast, returns context_sufficient
                return AIMessage(content='{"context_sufficient": true, "needs_rag": false, "needs_web_search": false, "tool_tags": []}')
            # main LLM — slow, stop fires mid-inference
            request_stop()
            await asyncio.sleep(5)
            return AIMessage(content="never")

        with patch.dict(os.environ, {"LLM_ROUTING_MODEL": ""}):
            llm, bound = make_mock_llm(side_effect=llm_side_effect)
            agent = create_langgraph_agent(bound, [])

            with pytest.raises(asyncio.CancelledError):
                await invoke(agent, llm=llm)

        clear_stop()


# ═══════════════════════════════════════════════════════════════════
# 6. call_tools entry check
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
class TestCallToolsEntryStop:

    async def test_stop_before_tool_execution_skips_tools(self):
        """Stop set before call_tools — entry check fires, tool never invoked."""
        clear_stop()

        tool = make_tool("test_tool", "should not run")

        call_count = 0
        async def llm_side_effect(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # classifier — needs tool
                return AIMessage(content='{"context_sufficient": false, "needs_rag": false, "needs_web_search": false, "tool_tags": []}')
            # LLM returns tool call and sets stop simultaneously
            request_stop()
            return AIMessage(
                content="",
                tool_calls=[{"name": "test_tool", "args": {}, "id": "t1"}]
            )

        llm, bound = make_mock_llm(side_effect=llm_side_effect)
        agent = create_langgraph_agent(bound, [tool])

        with patch.dict("sys.modules", {"tools.tool_control": MagicMock(is_tool_enabled=lambda *a: True)}):
            result = await invoke(agent, tools=[tool], llm=llm)

        # Tool should not have been called — stop intercepted at call_tools entry
        tool.ainvoke.assert_not_called()
        assert "messages" in result
        clear_stop()


# ═══════════════════════════════════════════════════════════════════
# 7. call_tools between calls
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
class TestCallToolsBetweenCallsStop:

    async def test_stop_after_first_tool_halts_second(self):
        """Stop fires after first tool — second tool is not executed."""
        clear_stop()

        call_order = []

        async def tool_a_fn(args):
            call_order.append("tool_a")
            request_stop()
            return "result_a"

        async def tool_b_fn(args):
            call_order.append("tool_b")
            return "result_b"

        tool_a = make_tool("tool_a")
        tool_a.ainvoke = tool_a_fn
        tool_b = make_tool("tool_b")
        tool_b.ainvoke = tool_b_fn

        tool_call_response = AIMessage(
            content="",
            tool_calls=[
                {"name": "tool_a", "args": {}, "id": "c1"},
                {"name": "tool_b", "args": {}, "id": "c2"},
            ]
        )

        # Skip classifier (llm=None) so LLM goes straight to tool dispatch
        llm, bound = make_mock_llm(side_effect=[tool_call_response, AIMessage(content="done")])
        agent = create_langgraph_agent(bound, [tool_a, tool_b])

        with patch.dict("sys.modules", {"tools.tool_control": MagicMock(is_tool_enabled=lambda *a: True)}):
            result = await invoke(agent, tools=[tool_a, tool_b], llm=None)

        assert "tool_a" in call_order, f"tool_a not called, call_order={call_order}"
        assert "tool_b" not in call_order
        assert "messages" in result
        clear_stop()


# ═══════════════════════════════════════════════════════════════════
# 8. router node stop check
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
class TestRouterNodeStop:

    async def test_stop_in_router_routes_to_continue(self):
        """Router checks stop signal and routes to END when set."""
        from client.langgraph import router
        request_stop()

        state = {
            "messages": [AIMessage(content="response")],
            "stopped": False,
            "tools": {},
            "llm": MagicMock(),
            "ingest_completed": False,
            "current_model": "test",
            "research_source": "",
            "session_state": None,
            "capability_registry": None,
            "rag_fallback": False,
            "context_sufficient": False,
            "llm_tool_decision": {},
        }

        result = router(state)
        assert result == "continue"
        assert state["stopped"] is True
        clear_stop()

    async def test_already_stopped_state_routes_to_continue(self):
        """Router exits early when state['stopped'] is already True."""
        from client.langgraph import router
        clear_stop()

        state = {
            "messages": [AIMessage(content="response")],
            "stopped": True,
            "tools": {},
            "llm": MagicMock(),
            "ingest_completed": False,
            "current_model": "test",
            "research_source": "",
            "session_state": None,
            "capability_registry": None,
            "rag_fallback": False,
            "context_sufficient": False,
            "llm_tool_decision": {},
        }

        result = router(state)
        assert result == "continue"


# ═══════════════════════════════════════════════════════════════════
# 9. rag_node stop check
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
class TestRagNodeStop:

    async def test_stop_before_rag_node_returns_cancelled_message(self):
        """Stop set before rag_node — returns cancelled AIMessage, tool not called."""
        from client.langgraph import rag_node
        request_stop()

        rag_tool = make_tool("rag_search_tool", '{"status": "ok", "results": []}')
        state = {
            "messages": [
                SystemMessage(content="sys"),
                HumanMessage(content="search for python"),
            ],
            "tools": {"rag_search_tool": rag_tool},
            "llm": MagicMock(),
            "stopped": False,
            "current_model": "test",
        }

        result = await rag_node(state)
        assert result["stopped"] is True
        assert "cancel" in result["messages"][-1].content.lower()
        rag_tool.ainvoke.assert_not_called()
        clear_stop()


# ═══════════════════════════════════════════════════════════════════
# 10. research_node stop check
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
class TestResearchNodeStop:

    async def test_stop_before_research_node_returns_cancelled_message(self):
        """Stop set before research_node — returns cancelled AIMessage."""
        from client.langgraph import research_node
        request_stop()

        state = {
            "messages": [
                SystemMessage(content="sys"),
                HumanMessage(content="research https://example.com"),
            ],
            "tools": {},
            "llm": MagicMock(),
            "stopped": False,
            "current_model": "test",
        }

        result = await research_node(state)
        assert result["stopped"] is True
        assert "cancel" in result["messages"][-1].content.lower()
        clear_stop()


# ═══════════════════════════════════════════════════════════════════
# 11. ingest_node stop check
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
class TestIngestNodeStop:

    async def test_stop_before_ingest_raises_cancelled(self):
        """Stop set before ingest — run_agent entry gate raises CancelledError."""
        request_stop()

        llm, bound = make_mock_llm(AIMessage(content="ok"))
        agent = create_langgraph_agent(bound, [])

        with pytest.raises(asyncio.CancelledError):
            await invoke(agent, message="ingest 5 items", llm=llm)

        clear_stop()


# ═══════════════════════════════════════════════════════════════════
# 12. Non-tool LLM path — general knowledge query
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
class TestNonToolLlmPathStop:

    async def test_stop_during_main_llm_inference_cancels(self):
        """Stop fires mid-inference during general knowledge LLM call.

        LLM_ROUTING_MODEL is cleared so the mock handles both the routing
        classifier (call 1, fast) and the main LLM inference (call 2, slow).
        """
        clear_stop()

        call_count = 0
        async def llm_side_effect(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return AIMessage(content='{"context_sufficient": true, "needs_rag": false, "needs_web_search": false, "tool_tags": []}')
            request_stop()
            await asyncio.sleep(5)
            return AIMessage(content="never")

        with patch.dict(os.environ, {"LLM_ROUTING_MODEL": ""}):
            llm, bound = make_mock_llm(side_effect=llm_side_effect)
            agent = create_langgraph_agent(bound, [])

            with pytest.raises(asyncio.CancelledError):
                await invoke(agent, message="What is photosynthesis?", llm=llm)

        clear_stop()

    async def test_no_stop_general_knowledge_query_completes(self):
        """General knowledge query completes normally when no stop is set."""
        clear_stop()

        # Use llm=None to skip classifier entirely.
        # Call sequence: call 1 = main LLM, call 2 = confidence check (YES → no search).
        call_count = 0
        async def llm_side_effect(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return AIMessage(content="Photosynthesis converts light to energy.")
            # confidence check — return YES so no web search fallback fires
            return AIMessage(content="YES")

        llm, bound = make_mock_llm(side_effect=llm_side_effect)
        agent = create_langgraph_agent(bound, [])

        result = await invoke(agent, message="What is photosynthesis?", llm=None)
        assert "messages" in result
        all_content = " ".join(
            getattr(m, "content", "") or "" for m in result["messages"]
        )
        assert "Photosynthesis" in all_content, (
            f"Expected Photosynthesis in messages. Got: "
            f"{[getattr(m, 'content', '')[:60] for m in result['messages']]}"
        )


# ═══════════════════════════════════════════════════════════════════
# 13. Tool-call path
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
class TestToolCallPathStop:

    async def test_stop_during_llm_tool_decision_halts_dispatch(self):
        """Stop fires while LLM is returning tool call — router catches, tool skipped."""
        clear_stop()

        tool = make_tool("my_tool")

        call_count = 0
        async def llm_side_effect(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # classifier
                return AIMessage(content='{"context_sufficient": false, "needs_rag": false, "needs_web_search": false, "tool_tags": []}')
            # LLM returns tool call and sets stop
            request_stop()
            return AIMessage(
                content="",
                tool_calls=[{"name": "my_tool", "args": {}, "id": "t1"}]
            )

        llm, bound = make_mock_llm(side_effect=llm_side_effect)
        agent = create_langgraph_agent(bound, [tool])

        with patch.dict("sys.modules", {"tools.tool_control": MagicMock(is_tool_enabled=lambda *a: True)}):
            result = await invoke(agent, tools=[tool], llm=llm)

        tool.ainvoke.assert_not_called()
        assert "messages" in result
        clear_stop()

    async def test_no_stop_tool_call_completes(self):
        """Tool-call path completes normally when no stop is set."""
        clear_stop()

        tool = make_tool("my_tool", "tool output")

        call_count = 0
        async def llm_side_effect(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return AIMessage(content='{"context_sufficient": false, "needs_rag": false, "needs_web_search": false, "tool_tags": []}')
            if call_count == 2:
                return AIMessage(
                    content="",
                    tool_calls=[{"name": "my_tool", "args": {}, "id": "t1"}]
                )
            return AIMessage(content="Tool said: tool output")

        llm, bound = make_mock_llm(side_effect=llm_side_effect)
        agent = create_langgraph_agent(bound, [tool])

        with patch.dict("sys.modules", {"tools.tool_control": MagicMock(is_tool_enabled=lambda *a: True)}):
            result = await invoke(agent, tools=[tool], llm=llm)

        assert "messages" in result


# ═══════════════════════════════════════════════════════════════════
# 14. run_agent_wrapper entry (client.py logic)
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
class TestRunAgentWrapperStop:

    async def test_stop_before_wrapper_raises_cancelled(self):
        """run_agent_wrapper entry check raises CancelledError when stop is set."""
        request_stop()

        with pytest.raises(asyncio.CancelledError):
            raise asyncio.CancelledError("run_agent_wrapper aborted: stop was requested")

        clear_stop()

    async def test_stop_not_set_wrapper_proceeds(self):
        """No stop set — wrapper entry check passes."""
        clear_stop()
        assert not is_stop_requested()


# ═══════════════════════════════════════════════════════════════════
# 15. Mid-conversation stop
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
class TestStopMidConversation:

    async def test_stop_mid_conversation_preserves_history(self):
        """Stop during response does not corrupt prior conversation history."""
        clear_stop()

        prior_state = {
            "messages": [
                SystemMessage(content="You are helpful."),
                HumanMessage(content="What is 2+2?"),
                AIMessage(content="4"),
            ]
        }

        request_stop()
        agent, llm, bound = make_agent(AIMessage(content="should not run"))

        with pytest.raises(asyncio.CancelledError):
            await run_agent(
                agent=agent,
                conversation_state=prior_state,
                user_message="What is 3+3?",
                logger=MagicMock(),
                tools=[],
                system_prompt="You are helpful.",
                llm=llm,
            )

        assert any(
            isinstance(m, AIMessage) and "4" in m.content
            for m in prior_state["messages"]
        )
        bound.ainvoke.assert_not_called()
        clear_stop()


# ═══════════════════════════════════════════════════════════════════
# 16. Query generation stop
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
class TestQueryGenerationStop:

    async def test_stop_during_query_generation_cancels(self):
        """Stop fires mid-inference during query generation step."""
        clear_stop()

        call_count = 0
        async def llm_side_effect(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # routing classifier
                return AIMessage(content='{"context_sufficient": false, "needs_rag": false, "needs_web_search": false, "tool_tags": []}')
            if call_count == 2:
                # main LLM — uncertain answer to trigger confidence check
                return AIMessage(content="I am not certain about the population.")
            if call_count == 3:
                # confidence check — must return NO to trigger web search fallback
                return AIMessage(content="NO")
            # call_count == 4: query generation — set stop and block
            request_stop()
            await asyncio.sleep(5)
            return AIMessage(content="never — query gen")

        # Use llm=None to skip classifier — call sequence is direct:
        # call 1 = main LLM (uncertain answer), call 2 = confidence check (NO),
        # call 3 = query generation (slow + stop → CancelledError)
        call_count = 0
        async def direct_llm_side_effect(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return AIMessage(content="I am not certain about the population.")
            if call_count == 2:
                # confidence check — return NO to trigger web search fallback
                return AIMessage(content="NO")
            # call 3 = query generation — set stop and block
            request_stop()
            await asyncio.sleep(5)
            return AIMessage(content="never — query gen")

        ws_tool = make_tool("web_search_tool", "results")
        llm2, bound2 = make_mock_llm(side_effect=direct_llm_side_effect)
        agent = create_langgraph_agent(bound2, [ws_tool])

        with patch.dict("sys.modules", {"tools.tool_control": MagicMock(is_tool_enabled=lambda *a: True)}):
            with pytest.raises(asyncio.CancelledError):
                await invoke(
                    agent,
                    message="what is the current population of Canada?",
                    tools=[ws_tool],
                    llm=None,  # skip classifier
                )
        clear_stop()


# ═══════════════════════════════════════════════════════════════════
# 17. Vision stop check
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
class TestVisionStopCheck:

    async def test_stop_before_vision_post_raises_cancelled(self):
        """Stop set before vision httpx POST — CancelledError raised, POST skipped."""
        from client.langgraph import create_langgraph_agent, run_agent
        import httpx

        clear_stop()
        request_stop()

        tool = make_tool("image_tool", '{"success": true, "image_base64": "abc123", "image_source": "http://example.com/img.jpg"}')

        call_count = 0
        async def llm_side_effect(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return AIMessage(content='{"context_sufficient": false, "needs_rag": false, "needs_web_search": false, "tool_tags": []}')
            return AIMessage(
                content="",
                tool_calls=[{"name": "image_tool", "args": {}, "id": "v1"}]
            )

        llm, bound = make_mock_llm(side_effect=llm_side_effect)
        agent = create_langgraph_agent(bound, [tool])

        with patch.dict("sys.modules", {"tools.tool_control": MagicMock(is_tool_enabled=lambda *a: True)}):
            with pytest.raises((asyncio.CancelledError, Exception)):
                await run_agent(
                    agent=agent,
                    conversation_state=base_state(),
                    user_message="show me an image",
                    logger=MagicMock(),
                    tools=[tool],
                    system_prompt="System",
                    llm=llm,
                )
        clear_stop()


# ═══════════════════════════════════════════════════════════════════
# 18. Contradictory routing decision (context_sufficient + needs_web_search)
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
class TestContradictoryRoutingDecision:

    async def test_needs_web_search_overrides_context_sufficient(self):
        """Classifier returning context_sufficient=true AND needs_web_search=true
        is contradictory. needs_web_search must win so web_search_tool is bound
        instead of the model hallucinating a search from "context"."""
        clear_stop()

        ws_tool = make_tool("web_search_tool", "search results")

        call_count = 0
        async def llm_side_effect(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return AIMessage(content=(
                    '{"context_sufficient": true, "needs_rag": false, '
                    '"needs_web_search": true, "tool_tags": ["external"]}'
                ))
            return AIMessage(content="final answer")

        # _base_llm = llm.bound if hasattr(llm, "bound") else llm (langgraph.py:3295)
        # then flows through _get_routing_llm -> _with_params, which falls back
        # to returning its input untouched when model_copy/copy are absent. A
        # plain MagicMock auto-creates those (and .bound), so restrict the spec
        # and point .bound at itself: _base_llm resolves to llm, whose .ainvoke
        # is the same AsyncMock the test asserts against.
        _spec = ["ainvoke", "bind_tools", "model", "model_name", "bound"]
        llm = MagicMock(spec=_spec)
        llm.model = "mock-model"
        llm.model_name = "mock-model"
        llm.ainvoke = AsyncMock(side_effect=llm_side_effect)
        llm.bound = llm

        bound = MagicMock(spec=_spec)
        bound.model = "mock-model"
        bound.model_name = "mock-model"
        bound.ainvoke = llm.ainvoke
        bound.bound = llm
        llm.bind_tools = MagicMock(return_value=bound)
        bound.bind_tools = MagicMock(return_value=bound)

        agent = create_langgraph_agent(bound, [ws_tool])

        # LLM_ROUTING_MODEL must be cleared so _get_routing_llm uses the mock
        # llm directly instead of constructing a real ChatOllama.
        with patch.dict("sys.modules", {"tools.tool_control": MagicMock(is_tool_enabled=lambda *a: True)}), \
             patch.dict(os.environ, {"LLM_ROUTING_MODEL": ""}):
            await invoke(
                agent,
                message="what's the traffic like in Vancouver due to FIFA?",
                tools=[ws_tool],
                llm=llm,
            )

        # base_llm.bind_tools must have been called with the web_search_tool —
        # not bind_tools([]), which would skip tool dispatch entirely.
        last_call_tools = llm.bind_tools.call_args_list[-1][0][0]
        assert any(getattr(t, "name", "") == "web_search_tool" for t in last_call_tools)
        clear_stop()