"""
Extended tests for client/websocket.py
Covers: stop signal during process_query, history question patterns,
        session task management, PROCESSING_SESSIONS tracking
"""
import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import AIMessage, HumanMessage


# ═══════════════════════════════════════════════════════════════════
# History question patterns
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
class TestHistoryQuestionPatterns:
    async def _run(self, prompt, messages):
        from client.websocket import process_query, CONNECTED_WEBSOCKETS
        ws = AsyncMock()
        conv = {"messages": list(messages)}
        mock_run = AsyncMock()
        CONNECTED_WEBSOCKETS.add(ws)
        try:
            await process_query(
                websocket=ws, prompt=prompt, original_prompt=prompt,
                agent_ref=[MagicMock()], conversation_state=conv,
                run_agent_fn=mock_run, logger=MagicMock(), tools=[],
                session_manager=None, session_id=None, system_prompt="Test"
            )
        finally:
            CONNECTED_WEBSOCKETS.discard(ws)
        return ws, mock_run, conv

    async def test_last_prompt_answered_directly(self):
        ws, mock_run, _ = await self._run(
            "what did i just ask",
            [HumanMessage(content="What is the weather?")]
        )
        mock_run.assert_not_called()
        sent = [json.loads(c.args[0]) for c in ws.send.call_args_list if c.args]
        texts = [p.get("text", "") for p in sent if p.get("type") == "assistant_message"]
        assert any("What is the weather?" in t for t in texts)

    async def test_previous_prompt_answered(self):
        ws, mock_run, _ = await self._run(
            "what was my previous prompt",
            [HumanMessage(content="Tell me about Python")]
        )
        mock_run.assert_not_called()
        sent = [json.loads(c.args[0]) for c in ws.send.call_args_list if c.args]
        texts = [p.get("text", "") for p in sent if p.get("type") == "assistant_message"]
        assert any("Python" in t for t in texts)

    async def test_your_response_answered_directly(self):
        ws, mock_run, _ = await self._run(
            "what did you say",
            [AIMessage(content="The weather is sunny.")]
        )
        mock_run.assert_not_called()
        sent = [json.loads(c.args[0]) for c in ws.send.call_args_list if c.args]
        texts = [p.get("text", "") for p in sent if p.get("type") == "assistant_message"]
        assert any("sunny" in t for t in texts)

    async def test_your_last_response_answered(self):
        ws, mock_run, _ = await self._run(
            "what was your last response",
            [AIMessage(content="I found 3 movies.")]
        )
        mock_run.assert_not_called()

    async def test_conversation_summary_answered(self):
        ws, mock_run, _ = await self._run(
            "what did we discuss",
            [
                HumanMessage(content="What's the weather?"),
                AIMessage(content="It's sunny."),
            ]
        )
        mock_run.assert_not_called()
        sent = [json.loads(c.args[0]) for c in ws.send.call_args_list if c.args]
        texts = [p.get("text", "") for p in sent if p.get("type") == "assistant_message"]
        assert any("summary" in t.lower() or "weather" in t.lower() for t in texts)

    async def test_history_question_no_prior_message_falls_through(self):
        """If history question but no prior message, should fall through to LLM."""
        ws, mock_run, _ = await self._run(
            "what did i just ask",
            []  # no messages
        )
        # With no prior message, should call the agent
        mock_run.assert_called_once()

    async def test_normal_query_calls_llm(self):
        ws, mock_run, _ = await self._run(
            "What is the weather in Vancouver?",
            [HumanMessage(content="Hello")]
        )
        mock_run.assert_called_once()


# ═══════════════════════════════════════════════════════════════════
# PROCESSING_SESSIONS tracking
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
class TestProcessingSessionsTracking:
    async def test_session_added_to_processing_during_query(self):
        from client.websocket import process_query, CONNECTED_WEBSOCKETS, PROCESSING_SESSIONS
        ws = AsyncMock()
        session_id = "track-session-123"
        processing_during = []

        async def mock_run(agent, conv, msg, logger, tools, system_prompt):
            processing_during.append(session_id in PROCESSING_SESSIONS)
            return {
                "messages": conv.get("messages", []) + [AIMessage(content="ok")],
                "current_model": "test"
            }

        CONNECTED_WEBSOCKETS.add(ws)
        try:
            await process_query(
                websocket=ws, prompt="test", original_prompt="test",
                agent_ref=[MagicMock()], conversation_state={"messages": []},
                run_agent_fn=mock_run, logger=MagicMock(), tools=[],
                session_manager=None, session_id=session_id, system_prompt="Test"
            )
        finally:
            CONNECTED_WEBSOCKETS.discard(ws)

        assert any(processing_during), "Session should be in PROCESSING_SESSIONS during query"
        assert session_id not in PROCESSING_SESSIONS, "Session should be removed after query"

    async def test_session_removed_from_processing_after_error(self):
        from client.websocket import process_query, CONNECTED_WEBSOCKETS, PROCESSING_SESSIONS
        ws = AsyncMock()
        session_id = "error-session-456"

        async def failing_run(*args, **kwargs):
            raise RuntimeError("test failure")

        CONNECTED_WEBSOCKETS.add(ws)
        try:
            await process_query(
                websocket=ws, prompt="test", original_prompt="test",
                agent_ref=[MagicMock()], conversation_state={"messages": []},
                run_agent_fn=failing_run, logger=MagicMock(), tools=[],
                session_manager=None, session_id=session_id, system_prompt="Test"
            )
        finally:
            CONNECTED_WEBSOCKETS.discard(ws)

        assert session_id not in PROCESSING_SESSIONS


# ═══════════════════════════════════════════════════════════════════
# complete message sent after query
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
class TestCompleteMessageSent:
    async def test_complete_sent_after_successful_query(self):
        from client.websocket import process_query, CONNECTED_WEBSOCKETS
        ws = AsyncMock()

        async def mock_run(agent, conv, msg, logger, tools, system_prompt):
            return {
                "messages": conv.get("messages", []) + [AIMessage(content="done")],
                "current_model": "test"
            }

        CONNECTED_WEBSOCKETS.add(ws)
        try:
            await process_query(
                websocket=ws, prompt="hello", original_prompt="hello",
                agent_ref=[MagicMock()], conversation_state={"messages": []},
                run_agent_fn=mock_run, logger=MagicMock(), tools=[],
                session_manager=None, session_id=None, system_prompt="Test"
            )
        finally:
            CONNECTED_WEBSOCKETS.discard(ws)

        sent = [json.loads(c.args[0]) for c in ws.send.call_args_list if c.args]
        types = [p.get("type") for p in sent]
        assert "complete" in types

    async def test_assistant_message_sent_with_model(self):
        from client.websocket import process_query, CONNECTED_WEBSOCKETS
        ws = AsyncMock()

        async def mock_run(agent, conv, msg, logger, tools, system_prompt):
            return {
                "messages": conv.get("messages", []) + [AIMessage(content="response text")],
                "current_model": "qwen2.5:14b"
            }

        CONNECTED_WEBSOCKETS.add(ws)
        try:
            await process_query(
                websocket=ws, prompt="hello", original_prompt="hello",
                agent_ref=[MagicMock()], conversation_state={"messages": []},
                run_agent_fn=mock_run, logger=MagicMock(), tools=[],
                session_manager=None, session_id=None, system_prompt="Test"
            )
        finally:
            CONNECTED_WEBSOCKETS.discard(ws)

        sent = [json.loads(c.args[0]) for c in ws.send.call_args_list if c.args]
        assistant_msgs = [p for p in sent if p.get("type") == "assistant_message"]
        assert len(assistant_msgs) >= 1
        assert assistant_msgs[0].get("text") == "response text"
        assert assistant_msgs[0].get("model") == "qwen2.5:14b"


# ═══════════════════════════════════════════════════════════════════
# Stop signal during query
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
class TestStopDuringQuery:
    async def test_stop_flag_cleared_before_new_query(self):
        """Stop signal should be cleared at the start of each new query."""
        from client.websocket import process_query, CONNECTED_WEBSOCKETS
        from client.stop_signal import request_stop, is_stop_requested, clear_stop

        clear_stop()
        request_stop()
        assert is_stop_requested()

        ws = AsyncMock()
        agent_ran = []

        async def mock_run(agent, conv, msg, logger, tools, system_prompt):
            agent_ran.append(True)
            return {
                "messages": conv.get("messages", []) + [AIMessage(content="ok")],
                "current_model": "test"
            }

        CONNECTED_WEBSOCKETS.add(ws)
        try:
            await process_query(
                websocket=ws, prompt="hello", original_prompt="hello",
                agent_ref=[MagicMock()], conversation_state={"messages": []},
                run_agent_fn=mock_run, logger=MagicMock(), tools=[],
                session_manager=None, session_id=None, system_prompt="Test"
            )
        finally:
            CONNECTED_WEBSOCKETS.discard(ws)
            clear_stop()

        # Agent should have run — stop flag cleared before query
        assert agent_ran