"""
Tests for client/context_tracker.py
Uses the actual ContextTracker API.
"""
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import SystemMessage


# ═══════════════════════════════════════════════════════════════════
# ContextTracker — structured context extraction
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestContextTracker:
    def test_extract_project_path(self, populated_session_manager):
        """Test extracting project path from session messages via regex"""
        from client.context_tracker import ContextTracker

        sm, session_id, _ = populated_session_manager

        tracker = ContextTracker(sm)
        context = tracker.extract_structured_context(session_id, "what dependencies?")

        assert "project_path" in context
        assert context["project_path"] == "/mnt/c/Users/Michael/project"

    def test_no_context_in_empty_session(self, session_manager):
        """Test no context extracted from session with no path-containing messages"""
        from client.context_tracker import ContextTracker

        session_id = session_manager.create_session()
        session_manager.add_message(session_id, "user", "hello", 30, None)

        tracker = ContextTracker(session_manager)
        context = tracker.extract_structured_context(session_id, "hello")

        assert "project_path" not in context

    def test_returns_empty_dict_for_missing_session(self, session_manager):
        """Test returns empty dict when session_id is None"""
        from client.context_tracker import ContextTracker

        tracker = ContextTracker(session_manager)
        context = tracker.extract_structured_context(None, "hello")

        assert context == {}

    def test_returns_empty_dict_for_no_messages(self, session_manager):
        """Test returns empty dict for session with no messages"""
        from client.context_tracker import ContextTracker

        session_id = session_manager.create_session()
        tracker = ContextTracker(session_manager)
        context = tracker.extract_structured_context(session_id, "hello")

        assert context == {}

    def test_create_context_message(self, session_manager):
        """Test creating SystemMessage with project path context"""
        from client.context_tracker import ContextTracker

        tracker = ContextTracker(session_manager)
        context = {"project_path": "/mnt/c/test/project"}

        msg = tracker.build_structured_context_message(context)

        assert isinstance(msg, SystemMessage)
        assert "/mnt/c/test/project" in msg.content
        assert "Active Project" in msg.content
        assert "CONVERSATION CONTEXT" in msg.content

    def test_structured_message_includes_tool_hint(self, session_manager):
        """Test that structured message includes project_path tool hint"""
        from client.context_tracker import ContextTracker

        tracker = ContextTracker(session_manager)
        context = {"project_path": "/mnt/c/Users/Michael/myapp"}

        msg = tracker.build_structured_context_message(context)

        assert 'project_path="/mnt/c/Users/Michael/myapp"' in msg.content

    def test_should_inject_context_with_data(self, session_manager):
        """Test that non-empty context produces a message"""
        from client.context_tracker import ContextTracker

        tracker = ContextTracker(session_manager)
        context = {"project_path": "/mnt/c/test"}
        msg = tracker.build_structured_context_message(context)

        assert msg is not None

    def test_should_not_inject_context_when_empty(self, session_manager):
        """Test that empty context returns None"""
        from client.context_tracker import ContextTracker

        tracker = ContextTracker(session_manager)
        msg = tracker.build_structured_context_message({})

        assert msg is None


# ═══════════════════════════════════════════════════════════════════
# ContextTracker — RAG context
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestContextTrackerRAG:
    def test_build_rag_context_message_with_results(self, session_manager):
        """Test formatting RAG results into a SystemMessage"""
        from client.context_tracker import ContextTracker

        tracker = ContextTracker(session_manager)
        results = [
            {"score": 0.92, "text": "User asked about the weather in Vancouver."},
            {"score": 0.75, "text": "User asked about Plex recommendations."},
        ]

        msg = tracker.build_rag_context_message(results)

        assert isinstance(msg, SystemMessage)
        assert "RELEVANT CONTEXT FROM THIS SESSION" in msg.content
        assert "92%" in msg.content
        assert "Vancouver" in msg.content
        assert "75%" in msg.content

    def test_build_rag_context_message_empty(self, session_manager):
        """Test that empty results returns None"""
        from client.context_tracker import ContextTracker

        tracker = ContextTracker(session_manager)
        msg = tracker.build_rag_context_message([])

        assert msg is None

    def test_build_rag_context_truncates_long_text(self, session_manager):
        """Test that long text is truncated at 400 chars"""
        from client.context_tracker import ContextTracker

        tracker = ContextTracker(session_manager)
        long_text = "x" * 600
        results = [{"score": 0.8, "text": long_text}]

        msg = tracker.build_rag_context_message(results)

        assert "..." in msg.content
        assert "x" * 400 in msg.content
        assert "x" * 401 not in msg.content

    def test_retrieve_semantic_context_returns_empty_on_failure(self, session_manager):
        """Test graceful fallback when RAG module is unavailable"""
        from client.context_tracker import ContextTracker

        tracker = ContextTracker(session_manager)

        with patch("client.context_tracker.ContextTracker.retrieve_semantic_context",
                   return_value=[]):
            results = tracker.retrieve_semantic_context(1, "test query")
            assert results == []


# ═══════════════════════════════════════════════════════════════════
# integrate_context_tracking
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestContextIntegration:
    def test_integrate_injects_structured_context(self, populated_session_manager, empty_conversation_state):
        """Test that integrate_context_tracking injects structured context"""
        from client.context_tracker import integrate_context_tracking

        sm, session_id, _ = populated_session_manager

        with patch("client.context_tracker.ContextTracker.retrieve_semantic_context", return_value=[]):
            result = integrate_context_tracking(
                session_manager=sm,
                session_id=session_id,
                prompt="what are the dependencies?",
                conversation_state=empty_conversation_state,
                logger=MagicMock()
            )

        assert result is True
        assert len(empty_conversation_state["messages"]) > 0
        system_msg = empty_conversation_state["messages"][0]
        assert isinstance(system_msg, SystemMessage)
        assert "CONVERSATION CONTEXT" in system_msg.content

    def test_integrate_injects_rag_context(self, session_manager, empty_conversation_state):
        """Test that integrate_context_tracking injects RAG context when available"""
        from client.context_tracker import integrate_context_tracking

        session_id = session_manager.create_session()
        rag_results = [{"score": 0.85, "text": "Previous discussion about weather."}]

        with patch("client.context_tracker.ContextTracker.retrieve_semantic_context",
                   return_value=rag_results):
            result = integrate_context_tracking(
                session_manager=session_manager,
                session_id=session_id,
                prompt="what was the forecast?",
                conversation_state=empty_conversation_state,
                logger=MagicMock()
            )

        assert result is True
        contents = " ".join(
            m.content for m in empty_conversation_state["messages"]
            if isinstance(m, SystemMessage)
        )
        assert "RELEVANT CONTEXT" in contents

    def test_integrate_returns_false_with_no_session_id(self, empty_conversation_state):
        """Test returns False when no session_id provided"""
        from client.context_tracker import integrate_context_tracking

        result = integrate_context_tracking(
            session_manager=MagicMock(),
            session_id=None,
            prompt="hello",
            conversation_state=empty_conversation_state,
            logger=MagicMock()
        )

        assert result is False
        assert len(empty_conversation_state["messages"]) == 0

    def test_integrate_returns_false_with_no_session_manager(self, empty_conversation_state):
        """Test returns False when session_manager is None"""
        from client.context_tracker import integrate_context_tracking

        result = integrate_context_tracking(
            session_manager=None,
            session_id=1,
            prompt="hello",
            conversation_state=empty_conversation_state,
            logger=MagicMock()
        )

        assert result is False

    def test_integrate_returns_false_when_no_context_found(self, session_manager, empty_conversation_state):
        """Test returns False when neither RAG nor structured context finds anything"""
        from client.context_tracker import integrate_context_tracking

        session_id = session_manager.create_session()
        session_manager.add_message(session_id, "user", "hello world", 30, None)

        with patch("client.context_tracker.ContextTracker.retrieve_semantic_context", return_value=[]):
            result = integrate_context_tracking(
                session_manager=session_manager,
                session_id=session_id,
                prompt="hello",
                conversation_state=empty_conversation_state,
                logger=MagicMock()
            )

        assert result is False
        assert len(empty_conversation_state["messages"]) == 0