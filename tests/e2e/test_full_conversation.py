import pytest
import json
from unittest.mock import AsyncMock, MagicMock
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


@pytest.mark.e2e
@pytest.mark.asyncio
class TestFullConversation:
    async def test_multi_turn_conversation_with_context(
        self,
        session_manager,
        mock_llm,
        mock_tools
    ):
        """Test complete multi-turn conversation with context preservation"""
        from client.websocket import process_query, CONNECTED_WEBSOCKETS

        conversation_state = {"messages": []}
        mock_websocket = AsyncMock()
        mock_websocket.send = AsyncMock()

        # Track responses across turns
        responses = []

        async def mock_run_agent(agent, conv_state, user_msg, logger, tools, system_prompt):
            response_text = f"Response to: {user_msg}"
            responses.append(response_text)
            msg = AIMessage(content=response_text)
            return {
                "messages": conv_state.get("messages", []) + [msg],
                "current_model": "test-model"
            }

        CONNECTED_WEBSOCKETS.add(mock_websocket)
        try:
            # Turn 1
            await process_query(
                websocket=mock_websocket,
                prompt="What is the weather like?",
                original_prompt="What is the weather like?",
                agent_ref=[MagicMock()],
                conversation_state=conversation_state,
                run_agent_fn=mock_run_agent,
                logger=MagicMock(),
                tools=mock_tools,
                session_manager=session_manager,
                session_id=None,
                system_prompt="You are a helpful assistant."
            )

            # Turn 2
            await process_query(
                websocket=mock_websocket,
                prompt="What about tomorrow?",
                original_prompt="What about tomorrow?",
                agent_ref=[MagicMock()],
                conversation_state=conversation_state,
                run_agent_fn=mock_run_agent,
                logger=MagicMock(),
                tools=mock_tools,
                session_manager=session_manager,
                session_id=None,
                system_prompt="You are a helpful assistant."
            )

            # Turn 3
            await process_query(
                websocket=mock_websocket,
                prompt="Thanks for the info!",
                original_prompt="Thanks for the info!",
                agent_ref=[MagicMock()],
                conversation_state=conversation_state,
                run_agent_fn=mock_run_agent,
                logger=MagicMock(),
                tools=mock_tools,
                session_manager=session_manager,
                session_id=None,
                system_prompt="You are a helpful assistant."
            )
        finally:
            CONNECTED_WEBSOCKETS.discard(mock_websocket)

        # WebSocket received messages across all turns
        assert mock_websocket.send.called
        assert mock_websocket.send.call_count >= 3

        # All three turns were processed by the agent
        assert len(responses) == 3
        assert "What is the weather like?" in responses[0]
        assert "What about tomorrow?" in responses[1]
        assert "Thanks for the info!" in responses[2]

        # Messages were broadcast with correct types
        sent_payloads = [
            json.loads(call.args[0])
            for call in mock_websocket.send.call_args_list
            if call.args
        ]
        message_types = [p.get("type") for p in sent_payloads]
        assert "user_message" in message_types
        assert "assistant_message" in message_types

    async def test_history_question_does_not_call_llm(
        self,
        session_manager,
        mock_tools
    ):
        """Test that history questions are answered directly without calling the LLM"""
        from client.websocket import process_query, CONNECTED_WEBSOCKETS

        conversation_state = {
            "messages": [
                AIMessage(content="The weather is sunny and 22 degrees.")
            ]
        }
        mock_websocket = AsyncMock()
        mock_websocket.send = AsyncMock()
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
                tools=mock_tools,
                session_manager=None,
                session_id=None,
                system_prompt="You are a helpful assistant."
            )
        finally:
            CONNECTED_WEBSOCKETS.discard(mock_websocket)

        # LLM was never called — answered directly from history
        mock_run_agent.assert_not_called()

        # Response was still sent to websocket
        assert mock_websocket.send.called
        sent_payloads = [
            json.loads(call.args[0])
            for call in mock_websocket.send.call_args_list
            if call.args
        ]
        assistant_msgs = [p for p in sent_payloads if p.get("type") == "assistant_message"]
        assert len(assistant_msgs) > 0
        assert "sunny" in assistant_msgs[0].get("text", "")

    async def test_session_persists_across_turns(
        self,
        session_manager,
        mock_tools
    ):
        """Test that messages are stored in session_manager across turns.

        Note: session_created is sent from websocket_handler, not process_query.
        process_query receives a session_id and stores messages into it.
        So we pre-create the session and verify messages are stored correctly.
        """
        from client.websocket import process_query, CONNECTED_WEBSOCKETS

        conversation_state = {"messages": []}
        mock_websocket = AsyncMock()
        mock_websocket.send = AsyncMock()

        # Pre-create the session as websocket_handler would
        session_id = session_manager.create_session()

        async def mock_run_agent(agent, conv_state, user_msg, logger, tools, system_prompt):
            return {
                "messages": conv_state.get("messages", []) + [AIMessage(content="OK")],
                "current_model": "test-model"
            }

        CONNECTED_WEBSOCKETS.add(mock_websocket)
        try:
            for prompt in ["Hello", "How are you?"]:
                await process_query(
                    websocket=mock_websocket,
                    prompt=prompt,
                    original_prompt=prompt,
                    agent_ref=[MagicMock()],
                    conversation_state=conversation_state,
                    run_agent_fn=mock_run_agent,
                    logger=MagicMock(),
                    tools=mock_tools,
                    session_manager=session_manager,
                    session_id=session_id,
                    system_prompt="You are a helpful assistant."
                )
        finally:
            CONNECTED_WEBSOCKETS.discard(mock_websocket)

        # Both assistant responses were stored in the session
        # (user messages are stored in websocket_handler, not process_query)
        messages = session_manager.get_session_messages(session_id)
        assistant_messages = [m for m in messages if m["role"] == "assistant"]
        assert len(assistant_messages) == 2
        assert assistant_messages[0]["text"] == "OK"
        assert assistant_messages[1]["text"] == "OK"

    async def test_error_in_agent_sends_error_message(
        self,
        session_manager,
        mock_tools
    ):
        """Test that agent errors are caught and an error message is sent to the websocket"""
        from client.websocket import process_query, CONNECTED_WEBSOCKETS

        conversation_state = {"messages": []}
        mock_websocket = AsyncMock()
        mock_websocket.send = AsyncMock()

        async def failing_agent(*args, **kwargs):
            raise RuntimeError("Simulated agent failure")

        CONNECTED_WEBSOCKETS.add(mock_websocket)
        try:
            await process_query(
                websocket=mock_websocket,
                prompt="Do something",
                original_prompt="Do something",
                agent_ref=[MagicMock()],
                conversation_state=conversation_state,
                run_agent_fn=failing_agent,
                logger=MagicMock(),
                tools=mock_tools,
                session_manager=None,
                session_id=None,
                system_prompt="You are a helpful assistant."
            )
        finally:
            CONNECTED_WEBSOCKETS.discard(mock_websocket)

        # Should have sent something — either an error message or a complete signal
        assert mock_websocket.send.called
        sent_payloads = [
            json.loads(call.args[0])
            for call in mock_websocket.send.call_args_list
            if call.args
        ]
        types_sent = {p.get("type") for p in sent_payloads}
        # Must send either an error or a complete to unblock the UI
        assert types_sent & {"error", "assistant_message", "complete"}