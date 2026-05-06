"""
tests/unit/test_langgraph_rag_memory.py

Tests for the RAG memory and session_id features added to run_agent:
  - session_id extracted from conversation_state
  - session_id injected into SystemMessage
  - overflow turns ingested into RAG
  - auto-RAG retrieval on every message
"""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, call
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from client.langgraph import create_langgraph_agent, run_agent, LLM_MESSAGE_WINDOW
from client.stop_signal import clear_stop


# ── Helpers ───────────────────────────────────────────────────────────────────

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


def make_rag_search_tool(results=None):
    """rag_search_tool returning controlled results."""
    tool = make_tool("rag_search_tool")
    tool.ainvoke = AsyncMock(return_value=json.dumps({
        "results": results or []
    }))
    return tool


def make_rag_add_tool():
    """rag_add_tool that tracks calls."""
    return make_tool("rag_add_tool", json.dumps({"chunks_added": 1}))


def base_state(session_id=None):
    state = {"messages": [SystemMessage(content="You are a helpful assistant.")]}
    if session_id is not None:
        state["session_id"] = session_id
    return state


async def invoke(agent, message="hello", state=None, tools=None):
    return await run_agent(
        agent=agent,
        conversation_state=state or base_state(),
        user_message=message,
        logger=MagicMock(),
        tools=tools or [],
        system_prompt="You are a helpful assistant.",
    )


# ═══════════════════════════════════════════════════════════════════
# session_id extraction and injection
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
class TestSessionIdInjection:
    async def test_session_id_injected_into_new_system_message(self):
        """session_id from conversation_state appears in new SystemMessage."""
        clear_stop()
        llm, bound = make_mock_llm(AIMessage(content="ok"))
        agent = create_langgraph_agent(bound, [])

        state = {"messages": [], "session_id": 42}
        await run_agent(
            agent=agent,
            conversation_state=state,
            user_message="hello",
            logger=MagicMock(),
            tools=[],
            system_prompt="System prompt here.",
        )
        system_msg = state["messages"][0]
        assert isinstance(system_msg, SystemMessage)
        assert "Current session ID: 42" in system_msg.content

    async def test_session_id_injected_into_existing_system_message(self):
        """session_id appended to pre-existing SystemMessage."""
        clear_stop()
        llm, bound = make_mock_llm(AIMessage(content="ok"))
        agent = create_langgraph_agent(bound, [])

        state = {
            "messages": [SystemMessage(content="Existing system prompt.")],
            "session_id": 99,
        }
        await run_agent(
            agent=agent,
            conversation_state=state,
            user_message="hello",
            logger=MagicMock(),
            tools=[],
            system_prompt="Existing system prompt.",
        )
        system_msg = state["messages"][0]
        assert "Current session ID: 99" in system_msg.content
        assert "Existing system prompt." in system_msg.content

    async def test_session_id_not_duplicated_on_second_call(self):
        """session_id injection is idempotent — not appended twice."""
        clear_stop()
        llm, bound = make_mock_llm(AIMessage(content="ok"))
        agent = create_langgraph_agent(bound, [])

        state = {
            "messages": [SystemMessage(content="System.")],
            "session_id": 7,
        }
        # First call
        await run_agent(
            agent=agent, conversation_state=state,
            user_message="first", logger=MagicMock(),
            tools=[], system_prompt="System.",
        )
        # Second call — must not double-append
        llm2, bound2 = make_mock_llm(AIMessage(content="ok2"))
        agent2 = create_langgraph_agent(bound2, [])
        await run_agent(
            agent=agent2, conversation_state=state,
            user_message="second", logger=MagicMock(),
            tools=[], system_prompt="System.",
        )
        system_content = state["messages"][0].content
        assert system_content.count("Current session ID: 7") == 1

    async def test_no_session_id_no_injection(self):
        """Without session_id in conversation_state nothing is injected."""
        clear_stop()
        llm, bound = make_mock_llm(AIMessage(content="ok"))
        agent = create_langgraph_agent(bound, [])

        state = {"messages": [SystemMessage(content="Clean system.")]}
        # No session_id key
        await run_agent(
            agent=agent, conversation_state=state,
            user_message="hello", logger=MagicMock(),
            tools=[], system_prompt="Clean system.",
        )
        assert "Current session ID" not in state["messages"][0].content

    async def test_session_id_in_llm_messages(self):
        """LLM receives the system message containing the session_id."""
        clear_stop()
        received = []

        async def capture(messages):
            received.extend(messages)
            return AIMessage(content="ok")

        llm, bound = make_mock_llm()
        bound.ainvoke = capture
        agent = create_langgraph_agent(bound, [])

        state = {
            "messages": [SystemMessage(content="System.")],
            "session_id": 123,
        }
        await run_agent(
            agent=agent, conversation_state=state,
            user_message="hi", logger=MagicMock(),
            tools=[], system_prompt="System.",
        )
        system_msgs = [m for m in received if isinstance(m, SystemMessage)]
        assert any("Current session ID: 123" in m.content for m in system_msgs)


# ═══════════════════════════════════════════════════════════════════
# Overflow ingestion into RAG
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
class TestOverflowIngestion:
    def _long_state(self, n_pairs=20, session_id=None):
        """State with n_pairs Human+AI turns — well above LLM_MESSAGE_WINDOW."""
        msgs = [SystemMessage(content="System.")]
        for i in range(n_pairs):
            msgs.append(HumanMessage(content=f"question {i}"))
            msgs.append(AIMessage(content=f"answer {i}"))
        state = {"messages": msgs}
        if session_id:
            state["session_id"] = session_id
        return state

    async def test_overflow_turns_ingested(self):
        """Turns beyond LLM_MESSAGE_WINDOW are ingested into rag_add_tool."""
        clear_stop()
        rag_add = make_rag_add_tool()
        rag_search = make_rag_search_tool([])
        llm, bound = make_mock_llm(AIMessage(content="ok"))
        agent = create_langgraph_agent(bound, [rag_add, rag_search])

        state = self._long_state(n_pairs=20, session_id=5)
        await run_agent(
            agent=agent, conversation_state=state,
            user_message="new question", logger=MagicMock(),
            tools=[rag_add, rag_search], system_prompt="System.",
        )
        assert rag_add.ainvoke.call_count >= 1

    async def test_overflow_ingested_as_paired_chunks(self):
        """Each ingested chunk contains both User and Assistant text."""
        clear_stop()
        rag_add = make_rag_add_tool()
        rag_search = make_rag_search_tool([])
        llm, bound = make_mock_llm(AIMessage(content="ok"))
        agent = create_langgraph_agent(bound, [rag_add, rag_search])

        state = self._long_state(n_pairs=20, session_id=5)
        await run_agent(
            agent=agent, conversation_state=state,
            user_message="hi", logger=MagicMock(),
            tools=[rag_add, rag_search], system_prompt="System.",
        )
        calls = rag_add.ainvoke.call_args_list
        assert len(calls) >= 1
        # Most chunks should be paired — allow the last one to be unpaired
        # if it's a trailing HumanMessage with no AI response yet
        paired = [c for c in calls
                  if "Assistant:" in (c[0][0] if c[0] else c[1]).get("text","")]
        unpaired = [c for c in calls
                    if "Assistant:" not in (c[0][0] if c[0] else c[1]).get("text","")]
        assert len(paired) >= 1, "Expected at least some paired Human+AI chunks"
        assert len(unpaired) <= 1, "At most one unpaired trailing chunk expected"
        for c in calls:
            args = c[0][0] if c[0] else c[1]
            assert args.get("text","").startswith("User:")

    async def test_overflow_source_contains_session_id(self):
        """Ingested chunks use conversation_history_{session_id} as source."""
        clear_stop()
        rag_add = make_rag_add_tool()
        rag_search = make_rag_search_tool([])
        llm, bound = make_mock_llm(AIMessage(content="ok"))
        agent = create_langgraph_agent(bound, [rag_add, rag_search])

        state = self._long_state(n_pairs=20, session_id=42)
        await run_agent(
            agent=agent, conversation_state=state,
            user_message="hi", logger=MagicMock(),
            tools=[rag_add, rag_search], system_prompt="System.",
        )
        for c in rag_add.ainvoke.call_args_list:
            args = c[0][0] if c[0] else c[1]
            assert "42" in args.get("source", "")

    async def test_no_overflow_no_rag_add(self):
        """Short history within window — rag_add_tool never called."""
        clear_stop()
        rag_add = make_rag_add_tool()
        rag_search = make_rag_search_tool([])
        llm, bound = make_mock_llm(AIMessage(content="ok"))
        agent = create_langgraph_agent(bound, [rag_add, rag_search])

        # Only 2 turns — well within window
        state = {
            "messages": [
                SystemMessage(content="System."),
                HumanMessage(content="hi"),
                AIMessage(content="hello"),
            ],
            "session_id": 1,
        }
        await run_agent(
            agent=agent, conversation_state=state,
            user_message="new", logger=MagicMock(),
            tools=[rag_add, rag_search], system_prompt="System.",
        )
        rag_add.ainvoke.assert_not_called()

    async def test_no_rag_add_tool_overflow_silently_skipped(self):
        """Overflow turns silently skipped when rag_add_tool not in tools."""
        clear_stop()
        llm, bound = make_mock_llm(AIMessage(content="ok"))
        agent = create_langgraph_agent(bound, [])

        state = self._long_state(n_pairs=20)
        # Should not raise even with no rag tool
        result = await run_agent(
            agent=agent, conversation_state=state,
            user_message="hi", logger=MagicMock(),
            tools=[], system_prompt="System.",
        )
        assert "messages" in result

    async def test_overflow_rag_add_failure_does_not_crash(self):
        """rag_add_tool raising during overflow ingestion is swallowed."""
        clear_stop()
        rag_add = make_rag_add_tool()
        rag_add.ainvoke = AsyncMock(side_effect=Exception("db locked"))
        rag_search = make_rag_search_tool([])
        llm, bound = make_mock_llm(AIMessage(content="ok"))
        agent = create_langgraph_agent(bound, [rag_add, rag_search])

        state = self._long_state(n_pairs=20, session_id=1)
        result = await run_agent(
            agent=agent, conversation_state=state,
            user_message="hi", logger=MagicMock(),
            tools=[rag_add, rag_search], system_prompt="System.",
        )
        assert "messages" in result


# ═══════════════════════════════════════════════════════════════════
# Auto-RAG retrieval
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
class TestAutoRagRetrieval:
    async def test_rag_results_injected_as_system_message(self):
        """When RAG returns results, a SystemMessage is prepended to llm_messages."""
        clear_stop()
        received = []

        async def capture(messages):
            received.extend(messages)
            return AIMessage(content="ok")

        rag_search = make_rag_search_tool([
            {"text": "Python was created in 1991", "source": "wiki", "score": 0.9},
        ])
        llm, bound = make_mock_llm()
        bound.ainvoke = capture
        agent = create_langgraph_agent(bound, [rag_search])

        await run_agent(
            agent=agent,
            conversation_state=base_state(),
            user_message="tell me about python",
            logger=MagicMock(),
            tools=[rag_search],
            system_prompt="System.",
        )
        system_msgs = [m for m in received if isinstance(m, SystemMessage)]
        rag_msgs = [m for m in system_msgs if "Relevant context" in m.content]
        assert len(rag_msgs) >= 1
        assert "Python was created in 1991" in rag_msgs[0].content

    async def test_rag_source_shown_in_injection(self):
        """RAG injection includes source attribution."""
        clear_stop()
        received = []

        async def capture(messages):
            received.extend(messages)
            return AIMessage(content="ok")

        rag_search = make_rag_search_tool([
            {"text": "Important fact", "source": "conversation_history_42", "score": 0.85},
        ])
        llm, bound = make_mock_llm()
        bound.ainvoke = capture
        agent = create_langgraph_agent(bound, [rag_search])

        await run_agent(
            agent=agent, conversation_state=base_state(),
            user_message="what fact?", logger=MagicMock(),
            tools=[rag_search], system_prompt="System.",
        )
        system_msgs = [m for m in received if isinstance(m, SystemMessage)]
        rag_msgs = [m for m in system_msgs if "Relevant context" in m.content]
        assert any("conversation_history_42" in m.content for m in rag_msgs)

    async def test_empty_rag_results_no_injection(self):
        """Empty RAG results produce no extra SystemMessage."""
        clear_stop()
        received = []

        async def capture(messages):
            received.extend(messages)
            return AIMessage(content="ok")

        rag_search = make_rag_search_tool([])  # no results
        llm, bound = make_mock_llm()
        bound.ainvoke = capture
        agent = create_langgraph_agent(bound, [rag_search])

        await run_agent(
            agent=agent, conversation_state=base_state(),
            user_message="hi", logger=MagicMock(),
            tools=[rag_search], system_prompt="System.",
        )
        system_msgs = [m for m in received if isinstance(m, SystemMessage)]
        rag_msgs = [m for m in system_msgs if "Relevant context" in m.content]
        assert len(rag_msgs) == 0

    async def test_rag_search_called_with_user_message(self):
        """rag_search_tool is called with the current user message as query."""
        clear_stop()
        rag_search = make_rag_search_tool([])
        llm, bound = make_mock_llm(AIMessage(content="ok"))
        agent = create_langgraph_agent(bound, [rag_search])

        await run_agent(
            agent=agent, conversation_state=base_state(),
            user_message="unique query string xyz",
            logger=MagicMock(), tools=[rag_search], system_prompt="System.",
        )
        rag_search.ainvoke.assert_called_once()
        call_args = rag_search.ainvoke.call_args[0][0]
        assert call_args.get("query") == "unique query string xyz"

    async def test_rag_search_failure_does_not_crash(self):
        """rag_search_tool exception during auto-RAG is swallowed."""
        clear_stop()
        rag_search = make_tool("rag_search_tool")
        rag_search.ainvoke = AsyncMock(side_effect=Exception("timeout"))
        llm, bound = make_mock_llm(AIMessage(content="ok"))
        agent = create_langgraph_agent(bound, [rag_search])

        result = await run_agent(
            agent=agent, conversation_state=base_state(),
            user_message="hi", logger=MagicMock(),
            tools=[rag_search], system_prompt="System.",
        )
        assert "messages" in result

    async def test_no_rag_search_tool_skips_retrieval(self):
        """No rag_search_tool in tools — auto-RAG step is silently skipped."""
        clear_stop()
        llm, bound = make_mock_llm(AIMessage(content="ok"))
        agent = create_langgraph_agent(bound, [])

        result = await run_agent(
            agent=agent, conversation_state=base_state(),
            user_message="hi", logger=MagicMock(),
            tools=[], system_prompt="System.",
        )
        assert "messages" in result

    async def test_rag_capped_at_five_results(self):
        """Auto-RAG injects at most 5 chunks even if more are returned."""
        clear_stop()
        received = []

        async def capture(messages):
            received.extend(messages)
            return AIMessage(content="ok")

        many_results = [
            {"text": f"Fact {i}", "source": f"src{i}", "score": 0.9}
            for i in range(10)
        ]
        rag_search = make_rag_search_tool(many_results)
        llm, bound = make_mock_llm()
        bound.ainvoke = capture
        agent = create_langgraph_agent(bound, [rag_search])

        await run_agent(
            agent=agent, conversation_state=base_state(),
            user_message="hi", logger=MagicMock(),
            tools=[rag_search], system_prompt="System.",
        )
        system_msgs = [m for m in received if isinstance(m, SystemMessage)]
        rag_msgs = [m for m in system_msgs if "Relevant context" in m.content]
        if rag_msgs:
            # Count bullet points — should be at most 5
            bullets = rag_msgs[0].content.count("•")
            assert bullets <= 5

    async def test_rag_malformed_json_response_handled(self):
        """rag_search_tool returning invalid JSON doesn't crash."""
        clear_stop()
        rag_search = make_tool("rag_search_tool")
        rag_search.ainvoke = AsyncMock(return_value="{{not json}}")
        llm, bound = make_mock_llm(AIMessage(content="ok"))
        agent = create_langgraph_agent(bound, [rag_search])

        result = await run_agent(
            agent=agent, conversation_state=base_state(),
            user_message="hi", logger=MagicMock(),
            tools=[rag_search], system_prompt="System.",
        )
        assert "messages" in result

    async def test_old_conversation_turn_retrieved_via_rag(self):
        """Simulates retrieving an old turn: RAG returns a conversation_history chunk."""
        clear_stop()
        received = []

        async def capture(messages):
            received.extend(messages)
            return AIMessage(content="The first prompt was about Python.")

        # RAG returns a chunk from an old conversation turn
        rag_search = make_rag_search_tool([{
            "text": "User: What is Python?\nAssistant: Python is a programming language.",
            "source": "conversation_history_5",
            "score": 0.92,
        }])
        llm, bound = make_mock_llm()
        bound.ainvoke = capture
        agent = create_langgraph_agent(bound, [rag_search])

        result = await run_agent(
            agent=agent, conversation_state=base_state(session_id=5),
            user_message="what was my first question?",
            logger=MagicMock(), tools=[rag_search], system_prompt="System.",
        )
        assert "messages" in result
        system_msgs = [m for m in received if isinstance(m, SystemMessage)]
        rag_content = " ".join(m.content for m in system_msgs)
        assert "conversation_history_5" in rag_content or "Python" in rag_content