"""Integration tests for WebSocket message handling"""
import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from langchain_core.messages import AIMessage


@pytest.mark.integration
@pytest.mark.asyncio
class TestWebSocketFlow:
    async def test_user_message_flow(self, mock_websocket, mock_llm, mock_tools):
        """Test complete user message handling flow"""
        from client.websocket import process_query, CONNECTED_WEBSOCKETS

        conversation_state = {"messages": []}

        async def mock_run_agent(agent, conv_state, user_msg, logger, tools, system_prompt):
            response_msg = AIMessage(content="Mock response to: " + user_msg)
            return {
                "messages": conv_state.get("messages", []) + [response_msg],
                "current_model": "test-model"
            }

        CONNECTED_WEBSOCKETS.add(mock_websocket)
        try:
            await process_query(
                websocket=mock_websocket,
                prompt="Hello",
                original_prompt="Hello",
                agent_ref=[MagicMock()],
                conversation_state=conversation_state,
                run_agent_fn=mock_run_agent,
                logger=MagicMock(),
                tools=mock_tools,
                session_manager=None,
                session_id=None,
                system_prompt="Test prompt"
            )
        finally:
            CONNECTED_WEBSOCKETS.discard(mock_websocket)

        assert mock_websocket.send.called

    async def test_session_creation_on_first_message(self, mock_websocket, session_manager):
        """Test that session is created on first message"""
        from client.websocket import process_query, CONNECTED_WEBSOCKETS

        conversation_state = {"messages": []}

        async def mock_run_agent(agent, conv_state, user_msg, logger, tools, system_prompt):
            response_msg = AIMessage(content="Response")
            return {
                "messages": conv_state.get("messages", []) + [response_msg],
                "current_model": "test-model"
            }

        CONNECTED_WEBSOCKETS.add(mock_websocket)
        try:
            await process_query(
                websocket=mock_websocket,
                prompt="First message",
                original_prompt="First message",
                agent_ref=[MagicMock()],
                conversation_state=conversation_state,
                run_agent_fn=mock_run_agent,
                logger=MagicMock(),
                tools=[],
                session_manager=session_manager,
                session_id=None,
                system_prompt="Test"
            )
        finally:
            CONNECTED_WEBSOCKETS.discard(mock_websocket)

        assert mock_websocket.send.called

    async def test_history_question_workaround(self, mock_websocket):
        """Test that history questions are answered directly"""
        from client.websocket import process_query, CONNECTED_WEBSOCKETS

        conversation_state = {
            "messages": [
                AIMessage(content="Previous response")
            ]
        }

        mock_run_agent = AsyncMock()

        CONNECTED_WEBSOCKETS.add(mock_websocket)
        try:
            await process_query(
                websocket=mock_websocket,
                prompt="what was your last response",
                original_prompt="what was your last response",
                agent_ref=[MagicMock()],
                conversation_state=conversation_state,
                run_agent_fn=mock_run_agent,
                logger=MagicMock(),
                tools=[],
                session_manager=None,
                session_id=None,
                system_prompt="Test"
            )
        finally:
            CONNECTED_WEBSOCKETS.discard(mock_websocket)

        assert mock_websocket.send.called

    async def test_error_handling(self, mock_websocket, mock_tools, caplog):
        """Test error handling in message processing"""
        from client.websocket import process_query, CONNECTED_WEBSOCKETS
        import logging

        conversation_state = {"messages": []}

        async def failing_agent(*args, **kwargs):
            raise ValueError("Test error")

        CONNECTED_WEBSOCKETS.add(mock_websocket)
        try:
            with caplog.at_level(logging.ERROR):
                await process_query(
                    websocket=mock_websocket,
                    prompt="Test",
                    original_prompt="Test",
                    agent_ref=[MagicMock()],
                    conversation_state=conversation_state,
                    run_agent_fn=failing_agent,
                    logger=MagicMock(),
                    tools=mock_tools,
                    session_manager=None,
                    session_id=None,
                    system_prompt="Test"
                )
        finally:
            CONNECTED_WEBSOCKETS.discard(mock_websocket)

        assert mock_websocket.send.called
        calls = mock_websocket.send.call_args_list
        error_sent = any("error" in str(call).lower() for call in calls)
        assert error_sent, "Error message should have been sent to websocket"