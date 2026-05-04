"""
Extended tests for client/session_manager.py
Covers: settings, chunk storage, search_messages, pin_session,
        set_message_image_source, get_message_by_id, is_first_session,
        get_all_sessions (pinned ordering), base64 stripping
"""
import pytest
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════
# Settings
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestSettings:
    def test_get_setting_default(self, session_manager):
        result = session_manager.get_setting("nonexistent_key", default="fallback")
        assert result == "fallback"

    def test_get_setting_default_none(self, session_manager):
        result = session_manager.get_setting("nonexistent_key")
        assert result is None

    def test_set_and_get_setting(self, session_manager):
        session_manager.set_setting("theme", "matrix")
        result = session_manager.get_setting("theme")
        assert result == "matrix"

    def test_set_setting_upsert(self, session_manager):
        session_manager.set_setting("key", "value1")
        session_manager.set_setting("key", "value2")
        result = session_manager.get_setting("key")
        assert result == "value2"

    def test_multiple_settings_independent(self, session_manager):
        session_manager.set_setting("a", "1")
        session_manager.set_setting("b", "2")
        assert session_manager.get_setting("a") == "1"
        assert session_manager.get_setting("b") == "2"


# ═══════════════════════════════════════════════════════════════════
# Chunk storage
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestChunkStorage:
    def test_store_chunk_returns_id(self, session_manager):
        chunk_id = session_manager.store_chunk("http://example.com", "chunk text")
        assert isinstance(chunk_id, int)
        assert chunk_id > 0

    def test_get_chunk_by_id(self, session_manager):
        chunk_id = session_manager.store_chunk("http://example.com", "hello world")
        text = session_manager.get_chunk(chunk_id)
        assert text == "hello world"

    def test_get_chunk_missing_returns_none(self, session_manager):
        result = session_manager.get_chunk(99999)
        assert result is None

    def test_get_chunks_by_source(self, session_manager):
        session_manager.store_chunk("http://source.com", "chunk 1")
        session_manager.store_chunk("http://source.com", "chunk 2")
        session_manager.store_chunk("http://other.com", "other chunk")
        chunks = session_manager.get_chunks_by_source("http://source.com")
        assert len(chunks) == 2
        texts = [c["text"] for c in chunks]
        assert "chunk 1" in texts
        assert "chunk 2" in texts

    def test_get_chunks_by_source_empty(self, session_manager):
        chunks = session_manager.get_chunks_by_source("http://nonexistent.com")
        assert chunks == []

    def test_delete_chunks_by_source(self, session_manager):
        session_manager.store_chunk("http://del.com", "to delete")
        session_manager.store_chunk("http://keep.com", "to keep")
        deleted = session_manager.delete_chunks_by_source("http://del.com")
        assert deleted == 1
        assert session_manager.get_chunks_by_source("http://del.com") == []
        assert len(session_manager.get_chunks_by_source("http://keep.com")) == 1

    def test_delete_all_chunks(self, session_manager):
        session_manager.store_chunk("http://a.com", "a")
        session_manager.store_chunk("http://b.com", "b")
        deleted = session_manager.delete_all_chunks()
        assert deleted == 2
        assert session_manager.get_chunks_by_source("http://a.com") == []

    def test_chunks_ordered_by_id(self, session_manager):
        session_manager.store_chunk("http://order.com", "first")
        session_manager.store_chunk("http://order.com", "second")
        session_manager.store_chunk("http://order.com", "third")
        chunks = session_manager.get_chunks_by_source("http://order.com")
        assert chunks[0]["text"] == "first"
        assert chunks[2]["text"] == "third"


# ═══════════════════════════════════════════════════════════════════
# search_messages
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestSearchMessages:
    def test_search_finds_matching_message(self, session_manager):
        sid = session_manager.create_session("Search Test")
        session_manager.add_message(sid, "user", "The quick brown fox", 30, None)
        results = session_manager.search_messages("quick brown")
        assert len(results) >= 1
        assert any("quick brown" in r["content"] for r in results)

    def test_search_no_match(self, session_manager):
        sid = session_manager.create_session()
        session_manager.add_message(sid, "user", "hello world", 30, None)
        results = session_manager.search_messages("xyzzy_no_match_xyz")
        assert results == []

    def test_search_returns_session_name(self, session_manager):
        sid = session_manager.create_session("My Named Session")
        session_manager.add_message(sid, "user", "unique search term xyz123", 30, None)
        results = session_manager.search_messages("xyz123")
        assert results[0]["session_name"] == "My Named Session"

    def test_search_respects_limit(self, session_manager):
        sid = session_manager.create_session()
        for i in range(10):
            session_manager.add_message(sid, "user", f"repeated term {i}", 30, None)
        results = session_manager.search_messages("repeated term", limit=3)
        assert len(results) <= 3


# ═══════════════════════════════════════════════════════════════════
# pin_session
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestPinSession:
    def test_pin_session(self, session_manager):
        sid = session_manager.create_session("Pinnable")
        session_manager.pin_session(sid, True)
        session = session_manager.get_session(sid)
        assert session["pinned"] is True

    def test_unpin_session(self, session_manager):
        sid = session_manager.create_session("Pinnable")
        session_manager.pin_session(sid, True)
        session_manager.pin_session(sid, False)
        session = session_manager.get_session(sid)
        assert session["pinned"] is False

    def test_pinned_sessions_first_in_get_all(self, session_manager):
        sid1 = session_manager.create_session("Normal")
        sid2 = session_manager.create_session("Pinned")
        session_manager.pin_session(sid2, True)
        sessions = session_manager.get_all_sessions()
        assert sessions[0]["id"] == sid2  # pinned first


# ═══════════════════════════════════════════════════════════════════
# get_message_by_id
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestGetMessageById:
    def test_get_message_by_id_found(self, session_manager):
        sid = session_manager.create_session()
        msg_id = session_manager.add_message(sid, "user", "hello", 30, None)
        msg = session_manager.get_message_by_id(msg_id)
        assert msg is not None
        assert msg["text"] == "hello"
        assert msg["role"] == "user"
        assert msg["session_id"] == sid

    def test_get_message_by_id_not_found(self, session_manager):
        msg = session_manager.get_message_by_id(99999)
        assert msg is None

    def test_get_message_by_id_includes_model(self, session_manager):
        sid = session_manager.create_session()
        msg_id = session_manager.add_message(sid, "assistant", "hi", 30, model="qwen2.5:14b")
        msg = session_manager.get_message_by_id(msg_id)
        assert msg["model"] == "qwen2.5:14b"


# ═══════════════════════════════════════════════════════════════════
# set_message_image_source
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestMessageImageSource:
    def test_set_image_url_retrieved_on_load(self, session_manager):
        sid = session_manager.create_session()
        msg_id = session_manager.add_message(sid, "user", "see image", 30, None)
        session_manager.set_message_image_source(msg_id, "https://example.com/img.jpg")
        messages = session_manager.get_session_messages(sid)
        assert messages[0].get("image_url") == "https://example.com/img.jpg"

    def test_set_local_image_encoded_on_load(self, session_manager, temp_dir):
        # Create a real tiny image file
        img_path = temp_dir / "test.jpg"
        img_path.write_bytes(b"fake image data")
        sid = session_manager.create_session()
        msg_id = session_manager.add_message(sid, "user", "local image", 30, None)
        session_manager.set_message_image_source(msg_id, str(img_path))
        messages = session_manager.get_session_messages(sid)
        # Local file gets base64 encoded
        assert "image" in messages[0]


# ═══════════════════════════════════════════════════════════════════
# is_first_session & base64 stripping
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestMiscSessionManager:
    def test_is_first_session_true(self, session_manager):
        assert session_manager.is_first_session() is True

    def test_is_first_session_false_after_two(self, session_manager):
        session_manager.create_session("First")
        session_manager.create_session("Second")
        assert session_manager.is_first_session() is False

    def test_base64_stripped_from_message(self, session_manager):
        """Large base64 blobs should be replaced with [image data]."""
        sid = session_manager.create_session()
        blob = "A" * 200  # large enough to trigger stripping
        session_manager.add_message(sid, "user", f"image data: {blob}", 30, None)
        messages = session_manager.get_session_messages(sid)
        assert "[image data]" in messages[0]["text"]
        assert "A" * 200 not in messages[0]["text"]

    def test_get_all_sessions_includes_pinned_field(self, session_manager):
        session_manager.create_session("Test")
        sessions = session_manager.get_all_sessions()
        assert "pinned" in sessions[0]

    def test_add_message_auto_creates_session(self, session_manager):
        """add_message should create a session if the session_id doesn't exist."""
        fake_sid = 9999
        session_manager.add_message(fake_sid, "user", "auto create", 30, None)
        session = session_manager.get_session(fake_sid)
        assert session is not None