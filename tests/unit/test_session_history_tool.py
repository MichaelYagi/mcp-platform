"""
tests/unit/test_session_history_tool.py

Tests for session_history_tool added to servers/rag/server.py.
Uses a real SQLite test DB and patches sqlite3.connect to redirect
the tool to our test database instead of the production one.
"""
import json
import sqlite3
import pytest
from pathlib import Path
from unittest.mock import patch


# ── DB fixture ────────────────────────────────────────────────────────────────

def make_sessions_db(tmp_path: Path) -> Path:
    """Create a minimal sessions.db with known data."""
    db_path = tmp_path / "sessions.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            model TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            image_source TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );
    """)
    conn.execute("INSERT INTO sessions (id, name) VALUES (1, 'Test Session')")
    conn.executemany(
        "INSERT INTO messages (session_id, role, content, model, created_at) VALUES (?,?,?,?,?)",
        [
            (1, "user",      "What is Python?",           None,         "2024-01-01 10:00:00"),
            (1, "assistant", "Python is a language.",     "qwen2.5:7b", "2024-01-01 10:00:05"),
            (1, "user",      "How does async work?",      None,         "2024-01-01 10:01:00"),
            (1, "assistant", "Async allows concurrency.", "qwen2.5:7b", "2024-01-01 10:01:10"),
        ]
    )
    conn.execute("INSERT INTO sessions (id, name) VALUES (2, 'Second Session')")
    conn.executemany(
        "INSERT INTO messages (session_id, role, content, model, created_at) VALUES (?,?,?,?,?)",
        [
            (2, "user",      "Tell me a joke.", None, "2024-01-02 09:00:00"),
            (2, "assistant", "Why did the...", None,  "2024-01-02 09:00:10"),
        ]
    )
    conn.commit()
    conn.close()
    return db_path


# ── Call helper ───────────────────────────────────────────────────────────────

def _call_tool(db_path: Path, session_id: str, limit: int = 20, order: str = "asc") -> str:
    """
    Call session_history_tool with sqlite3.connect redirected to our test DB.
    PROJECT_ROOT is captured at import time so we patch sqlite3.connect instead.
    """
    import servers.rag.server as srv

    original_connect = sqlite3.connect

    def patched_connect(path, *args, **kwargs):
        # Redirect any sessions.db path to our test DB
        if "sessions.db" in str(path):
            return original_connect(str(db_path), *args, **kwargs)
        return original_connect(path, *args, **kwargs)

    fn = srv.session_history_tool
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__

    with patch("sqlite3.connect", side_effect=patched_connect):
        return fn(session_id=session_id, limit=limit, order=order)


def _call_parsed(db_path: Path, session_id: str,
                 limit: int = 20, order: str = "asc") -> dict:
    return json.loads(_call_tool(db_path, session_id, limit, order))


# ═══════════════════════════════════════════════════════════════════
# Basic behaviour
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestSessionHistoryToolBasic:
    def test_returns_json_string(self, tmp_path):
        db = make_sessions_db(tmp_path)
        result = _call_tool(db, "1")
        assert isinstance(result, str)
        assert isinstance(json.loads(result), dict)

    def test_returns_correct_session_id(self, tmp_path):
        db = make_sessions_db(tmp_path)
        data = _call_parsed(db, "1")
        assert data["session_id"] == 1

    def test_returns_session_name(self, tmp_path):
        db = make_sessions_db(tmp_path)
        data = _call_parsed(db, "1")
        assert data["session_name"] == "Test Session"

    def test_returns_messages_list(self, tmp_path):
        db = make_sessions_db(tmp_path)
        data = _call_parsed(db, "1")
        assert isinstance(data["messages"], list)
        assert len(data["messages"]) == 4

    def test_messages_have_required_fields(self, tmp_path):
        db = make_sessions_db(tmp_path)
        for msg in _call_parsed(db, "1")["messages"]:
            assert "role" in msg
            assert "text" in msg
            assert "timestamp" in msg
            assert "index" in msg

    def test_first_user_prompt_convenience_field(self, tmp_path):
        db = make_sessions_db(tmp_path)
        data = _call_parsed(db, "1")
        assert "Python" in data["first_user_prompt"]

    def test_last_user_prompt_convenience_field(self, tmp_path):
        db = make_sessions_db(tmp_path)
        data = _call_parsed(db, "1")
        assert "async" in data["last_user_prompt"].lower()


# ═══════════════════════════════════════════════════════════════════
# Ordering and limit
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestSessionHistoryOrdering:
    def test_asc_order_oldest_first(self, tmp_path):
        db = make_sessions_db(tmp_path)
        data = _call_parsed(db, "1", order="asc")
        assert data["messages"][0]["text"] == "What is Python?"

    def test_desc_order_newest_first(self, tmp_path):
        db = make_sessions_db(tmp_path)
        data = _call_parsed(db, "1", order="desc")
        assert "async" in data["messages"][0]["text"].lower()

    def test_limit_respected(self, tmp_path):
        db = make_sessions_db(tmp_path)
        data = _call_parsed(db, "1", limit=2)
        assert data["total_returned"] == 2
        assert len(data["messages"]) == 2

    def test_limit_larger_than_history(self, tmp_path):
        db = make_sessions_db(tmp_path)
        data = _call_parsed(db, "1", limit=100)
        assert data["total_returned"] == 4

    def test_different_session_returns_own_messages(self, tmp_path):
        db = make_sessions_db(tmp_path)
        data = _call_parsed(db, "2")
        assert data["total_returned"] == 2
        assert data["messages"][0]["text"] == "Tell me a joke."


# ═══════════════════════════════════════════════════════════════════
# Error handling
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestSessionHistoryToolErrors:
    def test_missing_session_id_raises(self, tmp_path):
        db = make_sessions_db(tmp_path)
        with pytest.raises(Exception, match="session_id"):
            _call_tool(db, "")

    def test_nonexistent_session_raises(self, tmp_path):
        db = make_sessions_db(tmp_path)
        with pytest.raises(Exception):
            _call_parsed(db, "9999")

    def test_non_integer_session_id_raises(self, tmp_path):
        db = make_sessions_db(tmp_path)
        with pytest.raises(Exception):
            _call_parsed(db, "abc")

    def test_invalid_order_raises(self, tmp_path):
        db = make_sessions_db(tmp_path)
        with pytest.raises(Exception):
            _call_parsed(db, "1", order="random")

    def test_limit_zero_raises(self, tmp_path):
        db = make_sessions_db(tmp_path)
        with pytest.raises(Exception):
            _call_parsed(db, "1", limit=0)

    def test_invalid_limit_raises(self, tmp_path):
        db = make_sessions_db(tmp_path)
        with pytest.raises(Exception):
            _call_tool(db, "1", limit="notanumber")


# ═══════════════════════════════════════════════════════════════════
# Content correctness
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestSessionHistoryToolContent:
    def test_roles_are_valid(self, tmp_path):
        db = make_sessions_db(tmp_path)
        roles = {m["role"] for m in _call_parsed(db, "1")["messages"]}
        assert roles.issubset({"user", "assistant", "system"})

    def test_index_is_sequential(self, tmp_path):
        db = make_sessions_db(tmp_path)
        msgs = _call_parsed(db, "1")["messages"]
        assert [m["index"] for m in msgs] == list(range(1, len(msgs) + 1))

    def test_model_field_present_on_assistant(self, tmp_path):
        db = make_sessions_db(tmp_path)
        assistant_msgs = [m for m in _call_parsed(db, "1")["messages"]
                         if m["role"] == "assistant"]
        assert all("model" in m for m in assistant_msgs)
        assert assistant_msgs[0]["model"] == "qwen2.5:7b"

    def test_model_unknown_for_user_messages(self, tmp_path):
        db = make_sessions_db(tmp_path)
        user_msgs = [m for m in _call_parsed(db, "1")["messages"]
                    if m["role"] == "user"]
        assert all(m["model"] == "unknown" for m in user_msgs)

    def test_sessions_are_isolated(self, tmp_path):
        db = make_sessions_db(tmp_path)
        s1_texts = {m["text"] for m in _call_parsed(db, "1")["messages"]}
        s2_texts = {m["text"] for m in _call_parsed(db, "2")["messages"]}
        assert s1_texts.isdisjoint(s2_texts)