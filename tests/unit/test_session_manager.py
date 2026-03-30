import pytest
from client.session_manager import SessionManager

class TestSessionManager:
    def test_create_session(self, session_manager):
        """Test creating a new session"""
        session_id = session_manager.create_session("Test Session")
        assert session_id > 0
        
        session = session_manager.get_session(session_id)
        assert session["name"] == "Test Session"
    
    def test_add_message(self, session_manager):
        """Test adding messages to session"""
        session_id = session_manager.create_session()
        
        session_manager.add_message(session_id, "user", "Hello", 30, None)
        session_manager.add_message(session_id, "assistant", "Hi!", 30, "llama3.1:8b")
        
        messages = session_manager.get_session_messages(session_id)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["model"] == "llama3.1:8b"
    
    def test_message_history_limit(self, session_manager):
        """Test that message history is limited"""
        session_id = session_manager.create_session()
        
        # Add 15 messages with limit of 10
        for i in range(15):
            session_manager.add_message(session_id, "user", f"Message {i}", 10, None)
        
        messages = session_manager.get_session_messages(session_id)
        assert len(messages) == 10
        assert messages[0]["text"] == "Message 5"  # Oldest kept
    
    def test_delete_session(self, session_manager):
        """Test deleting a session"""
        session_id = session_manager.create_session()
        session_manager.add_message(session_id, "user", "Test", 30, None)
        
        session_manager.delete_session(session_id)
        
        session = session_manager.get_session(session_id)
        assert session is None
    
    def test_get_all_sessions(self, populated_session_manager):
        """Test retrieving all sessions"""
        sm, sid1, sid2 = populated_session_manager
        
        sessions = sm.get_all_sessions()
        assert len(sessions) >= 2
        assert all("id" in s for s in sessions)
    
    def test_cross_session_context(self, populated_session_manager):
        """Test getting context from multiple sessions"""
        sm, sid1, sid2 = populated_session_manager
        
        count = sm.get_user_session_count()
        assert count >= 2
        
        topics = sm.get_recent_session_topics(limit=5)
        assert len(topics) >= 2

@pytest.mark.unit
class TestSessionManagerEdgeCases:
    def test_nonexistent_session(self, session_manager):
        """Test accessing nonexistent session"""
        messages = session_manager.get_session_messages(9999)
        assert messages == []
    
    def test_empty_session(self, session_manager):
        """Test empty session has no messages"""
        session_id = session_manager.create_session()
        messages = session_manager.get_session_messages(session_id)
        assert len(messages) == 0