import pytest
from unittest.mock import AsyncMock, MagicMock
from client.langgraph import run_agent

@pytest.mark.integration
@pytest.mark.asyncio
class TestLangGraphAgent:
    async def test_agent_execution_with_tools(self, mock_llm, mock_tools, conversation_state_with_history):
        """Test agent execution with tool calls"""
        # Mock agent
        mock_agent = AsyncMock()
        mock_agent.ainvoke = AsyncMock(return_value={
            "messages": conversation_state_with_history["messages"] + [MagicMock(content="Response")],
            "current_model": "test-model"
        })
        
        result = await run_agent(
            agent=mock_agent,
            conversation_state=conversation_state_with_history,
            user_message="Test message",
            logger=MagicMock(),
            tools=mock_tools,
            system_prompt="Test",
            llm=mock_llm,
            max_history=20
        )
        
        assert "messages" in result
        assert "current_model" in result
    
    async def test_context_window_overflow_recovery(self, mock_llm, mock_tools):
        """Test that context overflow is handled gracefully"""
        # Create state with many messages
        large_state = {
            "messages": [MagicMock(content="x" * 1000) for _ in range(100)]
        }
        
        mock_agent = AsyncMock()
        mock_agent.ainvoke = AsyncMock(side_effect=ValueError(
            "Requested tokens (10000) exceed context window of (4096)"
        ))
        
        result = await run_agent(
            agent=mock_agent,
            conversation_state=large_state,
            user_message="Test",
            logger=MagicMock(),
            tools=mock_tools,
            system_prompt="Test",
            llm=mock_llm,
            max_history=20
        )
        
        # Should have error message
        assert len(result["messages"]) > 0