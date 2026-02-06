from unittest.mock import MagicMock

import pytest
from client.context_tracker import ContextTracker, integrate_context_tracking
from langchain_core.messages import SystemMessage

@pytest.mark.unit
class TestContextTracker:
    def test_extract_project_path(self, populated_session_manager):
        """Test extracting project path from session"""
        sm, session_id, _ = populated_session_manager
        
        tracker = ContextTracker(sm)
        context = tracker.extract_context_from_session(session_id, "what dependencies?")
        
        assert "project_path" in context
        assert context["project_path"] == "/mnt/c/Users/Michael/project"
    
    def test_no_context_in_empty_session(self, session_manager):
        """Test no context extracted from empty session"""
        session_id = session_manager.create_session()
        
        tracker = ContextTracker(session_manager)
        context = tracker.extract_context_from_session(session_id, "hello")
        
        assert context == {}
    
    def test_create_context_message(self, session_manager):
        """Test creating SystemMessage with context"""
        tracker = ContextTracker(session_manager)
        context = {"project_path": "/mnt/c/test/project"}
        
        msg = tracker.create_context_message(context)
        
        assert isinstance(msg, SystemMessage)
        assert "/mnt/c/test/project" in msg.content
        assert "Active Project" in msg.content
    
    def test_should_inject_context(self, session_manager):
        """Test deciding when to inject context"""
        tracker = ContextTracker(session_manager)
        
        # Should inject if context exists
        assert tracker.should_inject_context("test", {"project_path": "/test"})
        
        # Should not inject if no context
        assert not tracker.should_inject_context("test", {})


@pytest.mark.unit
class TestContextIntegration:
    def test_integrate_context_tracking(self, populated_session_manager, empty_conversation_state):
        """Test full context integration"""
        sm, session_id, _ = populated_session_manager
        
        result = integrate_context_tracking(
            session_manager=sm,
            session_id=session_id,
            prompt="what are the dependencies?",
            conversation_state=empty_conversation_state,
            logger=MagicMock()
        )
        
        assert result is True
        assert len(empty_conversation_state["messages"]) > 0
        
        # Check SystemMessage was added
        system_msg = empty_conversation_state["messages"][0]
        assert isinstance(system_msg, SystemMessage)
        assert "CONVERSATION CONTEXT" in system_msg.content