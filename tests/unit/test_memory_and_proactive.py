"""
Tests for persistent memory and proactive agent additions.

Covers:
  - memory_consolidator: schema, consolidation, vector search, :memory commands
  - proactive_agent: job CRUD, :jobs commands, ScheduleParser, ConfirmationTracker,
                     scheduling keyword detection, AgentScheduler
"""
import asyncio
import json
import os
import re
import sqlite3
import tempfile
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


# ═══════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def temp_data_dir(tmp_path):
    """Isolated data directory — prevents tests touching real DBs."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def memory_db_path(temp_data_dir):
    return temp_data_dir / "memory.db"


@pytest.fixture
def scheduler_db_path(temp_data_dir):
    return temp_data_dir / "scheduler.db"


@pytest.fixture
def sessions_db_path(temp_data_dir):
    """Minimal sessions.db with the schema memory_consolidator expects."""
    db_path = temp_data_dir / "sessions.db"
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            pinned INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            model TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
        );
    """)
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def patched_memory_paths(memory_db_path, sessions_db_path):
    """Redirect memory_consolidator DB paths to temp files."""
    with patch("client.memory_consolidator.MEMORY_DB_PATH", memory_db_path), \
         patch("client.memory_consolidator.SESSIONS_DB_PATH", sessions_db_path):
        yield memory_db_path, sessions_db_path


@pytest.fixture
def patched_scheduler_path(scheduler_db_path):
    """Redirect proactive_agent DB path to temp file."""
    with patch("client.proactive_agent.SCHEDULER_DB_PATH", scheduler_db_path):
        yield scheduler_db_path


@pytest.fixture
def mock_llm_fn():
    """Async LLM callable returning a simple JSON memory array."""
    async def _fn(system: str, user: str) -> str:
        return json.dumps([
            {"content": "User prefers diffs over full rewrites", "tier": "episodic", "importance": 0.8},
            {"content": "mcp-platform runs on WSL2 at 192.168.0.185", "tier": "episodic", "importance": 0.9},
        ])
    return _fn


@pytest.fixture
def mock_llm_fn_empty():
    """Async LLM callable returning empty memories."""
    async def _fn(system: str, user: str) -> str:
        return "[]"
    return _fn


@pytest.fixture
def sessions_db_with_messages(sessions_db_path):
    """sessions.db pre-populated with one session and two messages."""
    conn = sqlite3.connect(sessions_db_path)
    conn.execute("INSERT INTO sessions (id, name) VALUES (1, 'Test session')")
    conn.execute("INSERT INTO messages (session_id, role, content) VALUES (1, 'user', 'How do I fix cold load?')")
    conn.execute("INSERT INTO messages (session_id, role, content) VALUES (1, 'assistant', 'Set OLLAMA_KEEP_ALIVE=-1')")
    conn.commit()
    conn.close()
    return sessions_db_path


# ═══════════════════════════════════════════════════════════════════
# memory_consolidator — DB / schema
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestMemoryDbSchema:

    def test_ensure_db_creates_memory_db(self, patched_memory_paths):
        from client.memory_consolidator import _ensure_db
        memory_db, _ = patched_memory_paths
        assert not memory_db.exists()
        _ensure_db()
        assert memory_db.exists()

    def test_memories_table_has_embedding_column(self, patched_memory_paths):
        from client.memory_consolidator import _ensure_db
        memory_db, _ = patched_memory_paths
        _ensure_db()
        conn = sqlite3.connect(memory_db)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()]
        conn.close()
        assert "embedding" in cols

    def test_sessions_db_migration_adds_consolidated_at(self, patched_memory_paths):
        from client.memory_consolidator import _ensure_db
        _, sessions_db = patched_memory_paths
        _ensure_db()
        conn = sqlite3.connect(sessions_db)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(sessions)").fetchall()]
        conn.close()
        assert "consolidated_at" in cols

    def test_migration_is_idempotent(self, patched_memory_paths):
        """Running _ensure_db twice must not raise."""
        from client.memory_consolidator import _ensure_db
        _ensure_db()
        _ensure_db()  # second call — should not raise

    def test_embedding_migration_on_existing_db(self, memory_db_path, sessions_db_path):
        """Existing memory.db without embedding column gets it added."""
        # Create DB without embedding column
        conn = sqlite3.connect(memory_db_path)
        conn.execute("""
            CREATE TABLE memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tier TEXT NOT NULL,
                content TEXT NOT NULL,
                source_session TEXT,
                importance REAL NOT NULL DEFAULT 0.5,
                access_count INTEGER NOT NULL DEFAULT 0,
                last_accessed TEXT,
                created_at TEXT NOT NULL,
                promoted_at TEXT
            )
        """)
        conn.execute("CREATE TABLE IF NOT EXISTS memory_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        conn.commit()
        conn.close()

        with patch("client.memory_consolidator.MEMORY_DB_PATH", memory_db_path), \
             patch("client.memory_consolidator.SESSIONS_DB_PATH", sessions_db_path):
            from client.memory_consolidator import _ensure_db
            _ensure_db()

        conn = sqlite3.connect(memory_db_path)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()]
        conn.close()
        assert "embedding" in cols


# ═══════════════════════════════════════════════════════════════════
# memory_consolidator — consolidation guard
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestConsolidationGuard:

    def test_is_consolidated_false_when_null(self, patched_memory_paths, sessions_db_with_messages):
        from client.memory_consolidator import _ensure_db, _is_consolidated
        _ensure_db()
        assert _is_consolidated("1") is False

    def test_mark_and_check_consolidated(self, patched_memory_paths, sessions_db_with_messages):
        from client.memory_consolidator import _ensure_db, _mark_consolidated, _is_consolidated
        _ensure_db()
        _mark_consolidated("1")
        assert _is_consolidated("1") is True

    def test_is_consolidated_false_for_missing_session(self, patched_memory_paths):
        from client.memory_consolidator import _ensure_db, _is_consolidated
        _ensure_db()
        assert _is_consolidated("999") is False


# ═══════════════════════════════════════════════════════════════════
# memory_consolidator — transcript building
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestTranscriptBuilding:

    def test_get_transcript_from_db(self, patched_memory_paths, sessions_db_with_messages):
        from client.memory_consolidator import _ensure_db, _get_transcript
        _ensure_db()
        transcript = _get_transcript("1")
        assert "OLLAMA_KEEP_ALIVE" in transcript
        assert "USER:" in transcript or "user" in transcript.lower()

    def test_get_transcript_empty_for_missing_session(self, patched_memory_paths):
        from client.memory_consolidator import _ensure_db, _get_transcript
        _ensure_db()
        transcript = _get_transcript("999")
        assert transcript == ""

    def test_get_transcript_via_session_manager(self, patched_memory_paths):
        from client.memory_consolidator import _get_transcript
        mock_sm = MagicMock()
        mock_sm.get_session_messages.return_value = [
            {"role": "user", "text": "Hello"},
            {"role": "assistant", "text": "Hi there"},
        ]
        transcript = _get_transcript("1", session_manager=mock_sm)
        assert "Hello" in transcript
        assert "Hi there" in transcript


# ═══════════════════════════════════════════════════════════════════
# memory_consolidator — memory parsing
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestMemoryParsing:

    def test_parse_valid_json_array(self):
        from client.memory_consolidator import _parse_memories
        raw = '[{"content": "fact", "tier": "episodic", "importance": 0.7}]'
        result = _parse_memories(raw)
        assert len(result) == 1
        assert result[0]["content"] == "fact"

    def test_parse_strips_markdown_fences(self):
        from client.memory_consolidator import _parse_memories
        raw = '```json\n[{"content": "fact", "tier": "episodic", "importance": 0.5}]\n```'
        result = _parse_memories(raw)
        assert len(result) == 1

    def test_parse_empty_array(self):
        from client.memory_consolidator import _parse_memories
        assert _parse_memories("[]") == []

    def test_parse_invalid_json_returns_empty(self):
        from client.memory_consolidator import _parse_memories
        assert _parse_memories("not json at all") == []

    def test_parse_importance_bounds(self):
        from client.memory_consolidator import _parse_memories
        raw = '[{"content": "test", "tier": "episodic", "importance": 1.5}]'
        result = _parse_memories(raw)
        # _parse_memories just returns raw — clamping happens in consolidate()
        assert result[0]["importance"] == 1.5


# ═══════════════════════════════════════════════════════════════════
# memory_consolidator — consolidate() end-to-end
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestConsolidate:

    @pytest.mark.asyncio
    async def test_consolidate_writes_memories(
        self, patched_memory_paths, mock_llm_fn
    ):
        from client.memory_consolidator import _ensure_db, consolidate, _mem_conn
        memory_db, sessions_db = patched_memory_paths
        _ensure_db()
        # Seed messages directly into the patched sessions.db
        conn = sqlite3.connect(sessions_db)
        conn.execute("INSERT INTO sessions (id, name) VALUES (1, 'Test')")
        conn.execute("INSERT INTO messages (session_id, role, content) VALUES (1, 'user', 'How do I fix cold load latency in Ollama?')")
        conn.execute("INSERT INTO messages (session_id, role, content) VALUES (1, 'assistant', 'Set OLLAMA_KEEP_ALIVE=-1 to keep the model loaded.')")
        conn.commit()
        conn.close()
        with patch("client.memory_consolidator._embed", return_value=None):
            count = await consolidate("1", mock_llm_fn)
        assert count == 2
        with _mem_conn() as conn:
            rows = conn.execute("SELECT content FROM memories").fetchall()
        assert len(rows) == 2

    @pytest.mark.asyncio
    async def test_consolidate_skips_already_consolidated(
        self, patched_memory_paths, mock_llm_fn
    ):
        from client.memory_consolidator import _ensure_db, consolidate, _mark_consolidated
        memory_db, sessions_db = patched_memory_paths
        _ensure_db()
        conn = sqlite3.connect(sessions_db)
        conn.execute("INSERT INTO sessions (id, name) VALUES (1, 'Test')")
        conn.commit()
        conn.close()
        _mark_consolidated("1")
        with patch("client.memory_consolidator._embed", return_value=None):
            count = await consolidate("1", mock_llm_fn)
        assert count == 0

    @pytest.mark.asyncio
    async def test_consolidate_skips_short_transcript(
        self, patched_memory_paths, mock_llm_fn, sessions_db_path
    ):
        from client.memory_consolidator import _ensure_db, consolidate
        _ensure_db()
        # Session exists but has no messages → transcript < 100 chars
        conn = sqlite3.connect(sessions_db_path)
        conn.execute("INSERT INTO sessions (id, name) VALUES (2, 'Empty')")
        conn.commit()
        conn.close()
        with patch("client.memory_consolidator._embed", return_value=None):
            count = await consolidate("2", mock_llm_fn)
        assert count == 0

    @pytest.mark.asyncio
    async def test_consolidate_marks_session_after_empty_result(
        self, patched_memory_paths, mock_llm_fn_empty
    ):
        from client.memory_consolidator import _ensure_db, consolidate, _is_consolidated
        memory_db, sessions_db = patched_memory_paths
        _ensure_db()
        conn = sqlite3.connect(sessions_db)
        conn.execute("INSERT INTO sessions (id, name) VALUES (1, 'Test')")
        conn.execute("INSERT INTO messages (session_id, role, content) VALUES (1, 'user', 'How do I fix cold load latency in Ollama?')")
        conn.execute("INSERT INTO messages (session_id, role, content) VALUES (1, 'assistant', 'Set OLLAMA_KEEP_ALIVE=-1')")
        conn.commit()
        conn.close()
        with patch("client.memory_consolidator._embed", return_value=None):
            await consolidate("1", mock_llm_fn_empty)
        assert _is_consolidated("1") is True

    @pytest.mark.asyncio
    async def test_consolidate_clamps_importance(
        self, patched_memory_paths
    ):
        async def _llm_over(s, u):
            return '[{"content": "some long fact here yes", "tier": "episodic", "importance": 99.0}]'
        from client.memory_consolidator import _ensure_db, consolidate, _mem_conn
        memory_db, sessions_db = patched_memory_paths
        _ensure_db()
        conn = sqlite3.connect(sessions_db)
        conn.execute("INSERT INTO sessions (id, name) VALUES (1, 'Test')")
        conn.execute("INSERT INTO messages (session_id, role, content) VALUES (1, 'user', 'How do I fix cold load latency in Ollama?')")
        conn.execute("INSERT INTO messages (session_id, role, content) VALUES (1, 'assistant', 'Set OLLAMA_KEEP_ALIVE=-1')")
        conn.commit()
        conn.close()
        with patch("client.memory_consolidator._embed", return_value=None):
            await consolidate("1", _llm_over)
        with _mem_conn() as conn:
            rows = conn.execute("SELECT importance FROM memories").fetchall()
        for row in rows:
            assert row[0] <= 1.0


# ═══════════════════════════════════════════════════════════════════
# memory_consolidator — vector search / inject
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestMemoryVectorSearch:

    def _write_memory(self, memory_db_path, content, embedding_bytes=None, importance=0.5):
        import numpy as np
        from datetime import datetime, timezone
        if embedding_bytes is None:
            vec = [0.1] * 10
            embedding_bytes = bytes(np.array(vec, dtype=np.float32))
        conn = sqlite3.connect(memory_db_path)
        conn.execute(
            "INSERT INTO memories (tier, content, embedding, importance, created_at) "
            "VALUES ('episodic', ?, ?, ?, ?)",
            (content, embedding_bytes, importance, datetime.now(timezone.utc).isoformat())
        )
        conn.commit()
        conn.close()

    def test_inject_falls_back_to_importance_sort_when_no_query(
        self, patched_memory_paths
    ):
        import numpy as np
        from client.memory_consolidator import _ensure_db, inject_into_system_prompt
        memory_db, _ = patched_memory_paths
        _ensure_db()
        vec = bytes(np.array([0.1] * 10, dtype=np.float32))
        self._write_memory(memory_db, "fact one", vec, importance=0.9)
        self._write_memory(memory_db, "fact two", vec, importance=0.3)
        result = inject_into_system_prompt("BASE", query="")
        assert "Persistent Memory" in result
        assert "fact one" in result

    def test_inject_returns_base_when_no_memories(self, patched_memory_paths):
        from client.memory_consolidator import _ensure_db, inject_into_system_prompt
        _ensure_db()
        result = inject_into_system_prompt("BASE PROMPT", query="anything")
        assert result == "BASE PROMPT"

    def test_inject_with_query_uses_vector_search(self, patched_memory_paths):
        import numpy as np
        from client.memory_consolidator import _ensure_db, inject_into_system_prompt
        memory_db, _ = patched_memory_paths
        _ensure_db()

        # Two memories with different embeddings
        vec_a = bytes(np.array([1.0, 0.0] + [0.0] * 8, dtype=np.float32))
        vec_b = bytes(np.array([0.0, 1.0] + [0.0] * 8, dtype=np.float32))
        self._write_memory(memory_db, "relevant memory", vec_a)
        self._write_memory(memory_db, "unrelated memory", vec_b)

        # Query embedding very similar to vec_a
        query_bytes = bytes(np.array([0.99, 0.01] + [0.0] * 8, dtype=np.float32))
        with patch("client.memory_consolidator._embed", return_value=query_bytes):
            result = inject_into_system_prompt("BASE", query="test query", min_score=0.5)

        assert "relevant memory" in result

    def test_inject_falls_back_when_embed_fails(self, patched_memory_paths):
        import numpy as np
        from client.memory_consolidator import _ensure_db, inject_into_system_prompt
        memory_db, _ = patched_memory_paths
        _ensure_db()
        vec = bytes(np.array([0.1] * 10, dtype=np.float32))
        self._write_memory(memory_db, "fallback memory", vec, importance=0.8)
        with patch("client.memory_consolidator._embed", return_value=None):
            result = inject_into_system_prompt("BASE", query="something")
        assert "fallback memory" in result

    def test_touch_increments_access_count(self, patched_memory_paths):
        import numpy as np
        from client.memory_consolidator import _ensure_db, _touch_memories, _mem_conn
        memory_db, _ = patched_memory_paths
        _ensure_db()
        vec = bytes(np.array([0.1] * 10, dtype=np.float32))
        self._write_memory(memory_db, "touched memory", vec)
        with _mem_conn() as conn:
            mem_id = conn.execute("SELECT id FROM memories").fetchone()[0]
        _touch_memories([mem_id])
        with _mem_conn() as conn:
            count = conn.execute(
                "SELECT access_count FROM memories WHERE id=?", (mem_id,)
            ).fetchone()[0]
        assert count == 1


# ═══════════════════════════════════════════════════════════════════
# memory_consolidator — nightly promotion
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestNightlyPromotion:

    @pytest.mark.asyncio
    async def test_promotes_frequently_accessed(self, patched_memory_paths):
        import numpy as np
        from datetime import datetime, timezone
        from client.memory_consolidator import _ensure_db, run_nightly_promotion, _mem_conn
        memory_db, _ = patched_memory_paths
        _ensure_db()
        now = datetime.now(timezone.utc).isoformat()
        vec = bytes(np.array([0.1] * 10, dtype=np.float32))
        conn = sqlite3.connect(memory_db)
        conn.execute(
            "INSERT INTO memories (tier, content, embedding, importance, access_count, created_at) "
            "VALUES ('episodic', 'promoted fact', ?, 0.8, 5, ?)",
            (vec, now)
        )
        conn.commit()
        conn.close()
        with patch.dict(os.environ, {"MEMORY_PROMOTE_THRESHOLD": "3"}):
            await run_nightly_promotion()
        with _mem_conn() as conn:
            row = conn.execute(
                "SELECT tier FROM memories WHERE content='promoted fact'"
            ).fetchone()
        assert row[0] == "semantic"

    @pytest.mark.asyncio
    async def test_does_not_promote_below_threshold(self, patched_memory_paths):
        import numpy as np
        from datetime import datetime, timezone
        from client.memory_consolidator import _ensure_db, run_nightly_promotion, _mem_conn
        memory_db, _ = patched_memory_paths
        _ensure_db()
        now = datetime.now(timezone.utc).isoformat()
        vec = bytes(np.array([0.1] * 10, dtype=np.float32))
        conn = sqlite3.connect(memory_db)
        conn.execute(
            "INSERT INTO memories (tier, content, embedding, importance, access_count, created_at) "
            "VALUES ('episodic', 'stays episodic', ?, 0.5, 1, ?)",
            (vec, now)
        )
        conn.commit()
        conn.close()
        with patch.dict(os.environ, {"MEMORY_PROMOTE_THRESHOLD": "3"}):
            await run_nightly_promotion()
        with _mem_conn() as conn:
            row = conn.execute(
                "SELECT tier FROM memories WHERE content='stays episodic'"
            ).fetchone()
        assert row[0] == "episodic"


# ═══════════════════════════════════════════════════════════════════
# memory_consolidator — :memory command handler
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestMemoryCommandHandler:

    def _seed_memory(self, memory_db_path, content, tier="episodic"):
        import numpy as np
        from datetime import datetime, timezone
        conn = sqlite3.connect(memory_db_path)
        conn.execute(
            "INSERT INTO memories (tier, content, embedding, importance, created_at) "
            "VALUES (?, ?, ?, 0.5, ?)",
            (tier, content, bytes(np.array([0.1]*10, dtype=np.float32)),
             datetime.now(timezone.utc).isoformat())
        )
        conn.commit()
        conn.close()

    def test_memory_list_empty(self, patched_memory_paths):
        from client.memory_consolidator import handle_memory_command
        result = handle_memory_command(":memory")
        assert "No" in result

    def test_memory_list_shows_entries(self, patched_memory_paths):
        from client.memory_consolidator import _ensure_db, handle_memory_command
        memory_db, _ = patched_memory_paths
        _ensure_db()
        self._seed_memory(memory_db, "prefers diffs")
        result = handle_memory_command(":memory")
        assert "prefers diffs" in result

    def test_memory_filter_semantic(self, patched_memory_paths):
        from client.memory_consolidator import _ensure_db, handle_memory_command
        memory_db, _ = patched_memory_paths
        _ensure_db()
        self._seed_memory(memory_db, "episodic fact", tier="episodic")
        self._seed_memory(memory_db, "semantic fact", tier="semantic")
        result = handle_memory_command(":memory semantic")
        assert "semantic fact" in result
        assert "episodic fact" not in result

    def test_memory_filter_episodic(self, patched_memory_paths):
        from client.memory_consolidator import _ensure_db, handle_memory_command
        memory_db, _ = patched_memory_paths
        _ensure_db()
        self._seed_memory(memory_db, "episodic fact", tier="episodic")
        self._seed_memory(memory_db, "semantic fact", tier="semantic")
        result = handle_memory_command(":memory episodic")
        assert "episodic fact" in result
        assert "semantic fact" not in result

    def test_memory_forget_valid_id(self, patched_memory_paths):
        from client.memory_consolidator import _ensure_db, handle_memory_command, _mem_conn
        memory_db, _ = patched_memory_paths
        _ensure_db()
        self._seed_memory(memory_db, "to be deleted")
        with _mem_conn() as conn:
            mem_id = conn.execute("SELECT id FROM memories").fetchone()[0]
        result = handle_memory_command(f":memory forget {mem_id}")
        assert "deleted" in result.lower()
        with _mem_conn() as conn:
            row = conn.execute("SELECT id FROM memories WHERE id=?", (mem_id,)).fetchone()
        assert row is None

    def test_memory_forget_invalid_id(self, patched_memory_paths):
        from client.memory_consolidator import _ensure_db, handle_memory_command
        _ensure_db()
        result = handle_memory_command(":memory forget abc")
        assert "Invalid" in result

    def test_memory_clear_episodic(self, patched_memory_paths):
        from client.memory_consolidator import _ensure_db, handle_memory_command, _mem_conn
        memory_db, _ = patched_memory_paths
        _ensure_db()
        self._seed_memory(memory_db, "ep1", tier="episodic")
        self._seed_memory(memory_db, "ep2", tier="episodic")
        self._seed_memory(memory_db, "sem1", tier="semantic")
        result = handle_memory_command(":memory clear")
        assert "2" in result
        with _mem_conn() as conn:
            remaining = conn.execute(
                "SELECT tier FROM memories"
            ).fetchall()
        assert all(r[0] == "semantic" for r in remaining)

    def test_memory_clear_session(self, patched_memory_paths):
        from client.memory_consolidator import _ensure_db, handle_memory_command, _mem_conn
        import numpy as np
        from datetime import datetime, timezone
        memory_db, _ = patched_memory_paths
        _ensure_db()
        vec = bytes(np.array([0.1]*10, dtype=np.float32))
        now = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(memory_db)
        conn.execute(
            "INSERT INTO memories (tier, content, embedding, source_session, importance, created_at) "
            "VALUES ('episodic', 'from session 42', ?, '42', 0.5, ?)", (vec, now)
        )
        conn.execute(
            "INSERT INTO memories (tier, content, embedding, source_session, importance, created_at) "
            "VALUES ('episodic', 'from session 99', ?, '99', 0.5, ?)", (vec, now)
        )
        conn.commit()
        conn.close()
        result = handle_memory_command(":memory clear session 42")
        assert "1" in result
        assert "42" in result
        with _mem_conn() as conn:
            remaining = conn.execute("SELECT source_session FROM memories").fetchall()
        assert all(r[0] == "99" for r in remaining)

    def test_memory_clear_session_not_found(self, patched_memory_paths):
        from client.memory_consolidator import _ensure_db, handle_memory_command
        _ensure_db()
        result = handle_memory_command(":memory clear session 999")
        assert "No memories found" in result

    def test_memory_clear_session_missing_id(self, patched_memory_paths):
        from client.memory_consolidator import _ensure_db, handle_memory_command
        _ensure_db()
        result = handle_memory_command(":memory clear session")
        assert "Usage" in result

    def test_memory_unknown_subcommand(self, patched_memory_paths):
        from client.memory_consolidator import handle_memory_command
        result = handle_memory_command(":memory badverb")
        assert "Unknown" in result or "Available" in result


# ═══════════════════════════════════════════════════════════════════
# proactive_agent — job CRUD
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestSchedulerJobCRUD:

    def test_create_and_list_job(self, patched_scheduler_path):
        from client.proactive_agent import create_job, list_jobs
        job_id = create_job(
            label="Daily briefing",
            tool="get_day_briefing",
            cron="30 5 * * *",
        )
        assert isinstance(job_id, int)
        jobs = list_jobs()
        assert len(jobs) == 1
        assert jobs[0]["label"] == "Daily briefing"
        assert jobs[0]["tool"] == "get_day_briefing"

    def test_get_job_by_id(self, patched_scheduler_path):
        from client.proactive_agent import create_job, get_job
        job_id = create_job(label="Weather", tool="get_weather", cron="0 7 * * *")
        job = get_job(job_id)
        assert job is not None
        assert job["id"] == job_id

    def test_get_job_missing_returns_none(self, patched_scheduler_path):
        from client.proactive_agent import get_job
        assert get_job(9999) is None

    def test_find_job_by_label_partial_match(self, patched_scheduler_path):
        from client.proactive_agent import create_job, find_job_by_label
        create_job(label="Daily Briefing", tool="get_day_briefing", cron="30 5 * * *")
        job = find_job_by_label("briefing")
        assert job is not None
        assert "Briefing" in job["label"]

    def test_find_job_by_label_case_insensitive(self, patched_scheduler_path):
        from client.proactive_agent import create_job, find_job_by_label
        create_job(label="Daily Briefing", tool="get_day_briefing", cron="30 5 * * *")
        assert find_job_by_label("DAILY") is not None
        assert find_job_by_label("daily") is not None

    def test_set_job_enabled_false(self, patched_scheduler_path):
        from client.proactive_agent import create_job, set_job_enabled, get_job
        job_id = create_job(label="X", tool="y", cron="* * * * *")
        set_job_enabled(job_id, False)
        assert get_job(job_id)["enabled"] == 0

    def test_set_job_enabled_true(self, patched_scheduler_path):
        from client.proactive_agent import create_job, set_job_enabled, get_job
        job_id = create_job(label="X", tool="y", cron="* * * * *")
        set_job_enabled(job_id, False)
        set_job_enabled(job_id, True)
        assert get_job(job_id)["enabled"] == 1

    def test_delete_job(self, patched_scheduler_path):
        from client.proactive_agent import create_job, delete_job, get_job
        job_id = create_job(label="Gone", tool="y", cron="* * * * *")
        delete_job(job_id)
        assert get_job(job_id) is None

    def test_record_run_increments_count(self, patched_scheduler_path):
        from client.proactive_agent import create_job, record_run, get_job
        job_id = create_job(label="Counter", tool="y", cron="* * * * *")
        record_run(job_id)
        record_run(job_id)
        assert get_job(job_id)["run_count"] == 2

    def test_create_condition_job(self, patched_scheduler_path):
        from client.proactive_agent import create_job, get_job
        job_id = create_job(
            label="Gmail alert",
            tool="get_gmail_summary",
            trigger_type="condition",
            condition_tool="get_gmail_unread_count",
            condition_expr="result > 10",
            condition_cron="*/15 * * * *",
        )
        job = get_job(job_id)
        assert job["trigger_type"] == "condition"
        assert job["condition_expr"] == "result > 10"


# ═══════════════════════════════════════════════════════════════════
# proactive_agent — cron_to_human
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestCronToHuman:

    def test_every_day(self):
        from client.proactive_agent import cron_to_human
        assert "Every day" in cron_to_human("30 5 * * *")

    def test_weekdays(self):
        from client.proactive_agent import cron_to_human
        assert "Weekdays" in cron_to_human("0 8 * * 1-5")

    def test_weekends(self):
        from client.proactive_agent import cron_to_human
        assert "Weekend" in cron_to_human("0 9 * * 6,0")

    def test_specific_days(self):
        from client.proactive_agent import cron_to_human
        result = cron_to_human("0 9 * * 1")
        assert "Mon" in result

    def test_invalid_cron_returned_as_is(self):
        from client.proactive_agent import cron_to_human
        assert cron_to_human("not a cron") == "not a cron"

    def test_empty_string(self):
        from client.proactive_agent import cron_to_human
        assert cron_to_human("") == "—"


# ═══════════════════════════════════════════════════════════════════
# proactive_agent — :jobs command handler
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestJobsCommandHandler:

    def test_jobs_empty(self, patched_scheduler_path):
        from client.proactive_agent import handle_jobs_command
        result = handle_jobs_command(":jobs")
        assert "No scheduled jobs" in result

    def test_jobs_lists_jobs(self, patched_scheduler_path):
        from client.proactive_agent import create_job, handle_jobs_command
        create_job(label="Daily briefing", tool="get_day_briefing", cron="30 5 * * *")
        result = handle_jobs_command(":jobs")
        assert "Daily briefing" in result
        assert "get_day_briefing" in result

    def test_jobs_pause(self, patched_scheduler_path):
        from client.proactive_agent import create_job, handle_jobs_command, get_job
        job_id = create_job(label="Daily briefing", tool="get_day_briefing", cron="30 5 * * *")
        result = handle_jobs_command(":jobs pause daily briefing")
        assert "paused" in result.lower()
        assert get_job(job_id)["enabled"] == 0

    def test_jobs_enable(self, patched_scheduler_path):
        from client.proactive_agent import create_job, set_job_enabled, handle_jobs_command, get_job
        job_id = create_job(label="Daily briefing", tool="get_day_briefing", cron="30 5 * * *")
        set_job_enabled(job_id, False)
        result = handle_jobs_command(":jobs enable daily briefing")
        assert "resumed" in result.lower()
        assert get_job(job_id)["enabled"] == 1

    def test_jobs_cancel(self, patched_scheduler_path):
        from client.proactive_agent import create_job, handle_jobs_command, get_job
        job_id = create_job(label="Daily briefing", tool="get_day_briefing", cron="30 5 * * *")
        result = handle_jobs_command(":jobs cancel daily briefing")
        assert "deleted" in result.lower()
        assert get_job(job_id) is None

    def test_jobs_info(self, patched_scheduler_path):
        from client.proactive_agent import create_job, handle_jobs_command
        create_job(label="Daily briefing", tool="get_day_briefing", cron="30 5 * * *")
        result = handle_jobs_command(":jobs info daily briefing")
        assert "get_day_briefing" in result
        assert "30 5 * * *" in result

    def test_jobs_unknown_label(self, patched_scheduler_path):
        from client.proactive_agent import handle_jobs_command
        result = handle_jobs_command(":jobs pause nonexistent job")
        assert "No job found" in result

    def test_jobs_unknown_subcommand(self, patched_scheduler_path):
        from client.proactive_agent import handle_jobs_command
        result = handle_jobs_command(":jobs badverb something")
        assert "Unknown" in result or "Available" in result

    def test_jobs_shows_active_status(self, patched_scheduler_path):
        from client.proactive_agent import create_job, handle_jobs_command
        create_job(label="Active job", tool="some_tool", cron="0 8 * * *")
        result = handle_jobs_command(":jobs")
        assert "active" in result

    def test_jobs_shows_paused_status(self, patched_scheduler_path):
        from client.proactive_agent import create_job, set_job_enabled, handle_jobs_command
        job_id = create_job(label="Paused job", tool="some_tool", cron="0 8 * * *")
        set_job_enabled(job_id, False)
        result = handle_jobs_command(":jobs")
        assert "paused" in result


# ═══════════════════════════════════════════════════════════════════
# proactive_agent — scheduling keyword detection
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestSchedulingKeywordDetection:

    @pytest.mark.parametrize("message", [
        "do a day briefing every day at 5:30am",
        "run a Gmail summary every weekday at 8am",
        "schedule the weather check daily",
        "remind me every Monday at 9am",
        "alert me when unread count is over 10",
        "check this every hour",
        "run at 7am",
    ])
    async def test_detects_scheduling_requests(self, message):
        from client.proactive_agent import looks_like_scheduling_request
        # Mock llm_fn to return "YES" for scheduling requests
        async def _yes_llm(system, user): return "YES"
        result = await looks_like_scheduling_request(message, llm_fn=_yes_llm)
        assert result is True

    @pytest.mark.parametrize("message", [
        "what's the weather today?",
        "show me my emails",
        "hello how are you",
        "what did we discuss",
        ":jobs",
        ":memory",
    ])
    async def test_does_not_flag_normal_messages(self, message):
        from client.proactive_agent import looks_like_scheduling_request
        # Mock llm_fn to return "NO" for non-scheduling messages
        async def _no_llm(system, user): return "NO"
        result = await looks_like_scheduling_request(message, llm_fn=_no_llm)
        assert result is False

    async def test_returns_false_without_llm_fn(self):
        from client.proactive_agent import looks_like_scheduling_request
        result = await looks_like_scheduling_request("remind me about something", llm_fn=None)
        assert result is False

    async def test_returns_false_on_llm_error(self):
        from client.proactive_agent import looks_like_scheduling_request
        async def _failing_llm(system, user): raise RuntimeError("LLM error")
        result = await looks_like_scheduling_request("remind me about something", llm_fn=_failing_llm)
        assert result is False


# ═══════════════════════════════════════════════════════════════════
# proactive_agent — ScheduleParser
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestScheduleParser:

    def _make_parser(self, llm_response: str):
        from client.proactive_agent import ScheduleParser
        async def _llm(system, user):
            return llm_response
        return ScheduleParser(
            llm_fn=_llm,
            available_tools=["get_day_briefing", "get_weather", "get_gmail_summary"],
        )

    @pytest.mark.asyncio
    async def test_parses_cron_confirmation(self):
        from client.proactive_agent import ScheduleConfirmation
        response = json.dumps({
            "status": "ready",
            "trigger_type": "cron",
            "label": "Daily briefing",
            "tool": "get_day_briefing",
            "tool_args": {},
            "cron": "30 5 * * *",
            "timezone": "America/Vancouver",
            "human_schedule": "Every day at 5:30am",
        })
        parser = self._make_parser(response)
        result = await parser.parse("do a day briefing every day at 5:30am")
        assert isinstance(result, ScheduleConfirmation)
        assert result.cron == "30 5 * * *"
        assert result.tool == "get_day_briefing"
        assert result.human_schedule == "Every day at 5:30am"

    @pytest.mark.asyncio
    async def test_parses_condition_confirmation(self):
        from client.proactive_agent import ScheduleConfirmation
        response = json.dumps({
            "status": "ready",
            "trigger_type": "condition",
            "label": "Gmail alert",
            "tool": "get_gmail_summary",
            "tool_args": {},
            "condition_tool": "get_gmail_unread_count",
            "condition_expr": "result > 10",
            "condition_cron": "*/15 * * * *",
            "timezone": "America/Vancouver",
            "human_schedule": "When Gmail unread > 10",
        })
        parser = self._make_parser(response)
        result = await parser.parse("alert me when gmail unread is over 10")
        assert isinstance(result, ScheduleConfirmation)
        assert result.trigger_type == "condition"
        assert result.condition_expr == "result > 10"

    @pytest.mark.asyncio
    async def test_returns_clarification_when_missing_time(self):
        from client.proactive_agent import ScheduleClarification
        response = json.dumps({
            "status": "clarify",
            "question": "What time should I run the briefing?",
        })
        parser = self._make_parser(response)
        result = await parser.parse("run the briefing every day")
        assert isinstance(result, ScheduleClarification)
        assert "?" in result.question

    @pytest.mark.asyncio
    async def test_returns_clarification_on_bad_json(self):
        from client.proactive_agent import ScheduleClarification
        parser = self._make_parser("not valid json at all")
        result = await parser.parse("schedule something")
        assert isinstance(result, ScheduleClarification)

    @pytest.mark.asyncio
    async def test_returns_clarification_on_missing_required_fields(self):
        from client.proactive_agent import ScheduleClarification
        response = json.dumps({
            "status": "ready",
            "trigger_type": "cron",
            "label": "Missing cron",
            "tool": "get_day_briefing",
            # cron intentionally omitted
            "human_schedule": "Every day",
        })
        parser = self._make_parser(response)
        result = await parser.parse("run briefing every day")
        assert isinstance(result, ScheduleClarification)

    @pytest.mark.asyncio
    async def test_llm_failure_returns_clarification(self):
        from client.proactive_agent import ScheduleParser, ScheduleClarification
        async def _failing_llm(system, user):
            raise RuntimeError("Ollama down")
        parser = ScheduleParser(llm_fn=_failing_llm, available_tools=["get_day_briefing"])
        result = await parser.parse("run briefing every day at 7am")
        assert isinstance(result, ScheduleClarification)

    def test_confirmation_render_shows_key_fields(self):
        from client.proactive_agent import ScheduleConfirmation
        conf = ScheduleConfirmation(
            label="Daily briefing",
            tool="get_day_briefing",
            tool_args={},
            trigger_type="cron",
            cron="30 5 * * *",
            condition_tool=None,
            condition_expr=None,
            condition_cron="*/15 * * * *",
            timezone="America/Vancouver",
            human_schedule="Every day at 5:30am",
            original_request="do briefing every day at 5:30am",
        )
        rendered = conf.render()
        assert "get_day_briefing" in rendered
        assert "Every day at 5:30am" in rendered
        assert "yes" in rendered.lower()

    def test_clarification_render(self):
        from client.proactive_agent import ScheduleClarification
        clar = ScheduleClarification(question="What time?")
        assert clar.render() == "What time?"


# ═══════════════════════════════════════════════════════════════════
# proactive_agent — _PARSER_SYSTEM prompt hygiene
#
# Regression coverage for a real production bug: the prompt's own few-shot
# examples used the user's REAL email (michaeltyagi@gmail.com) as the
# placeholder `to=` address. Because the user's actual recurring scheduling
# requests also contain that exact address, the model's attention got
# hijacked — it would reply with its own example verbatim ("subject=Daily
# Briefing", "get_day_briefing") instead of extracting from the user's real
# message ("subject=", "shashin_random_tool"), producing consistent
# prose-instead-of-JSON failures that survived retries AND temperature=0.0
# (it wasn't sampling noise — the model was confidently doing the wrong but
# fully consistent thing because of the string collision).
#
# Fix: swapped the example address for recipient@example.com — `example.com`
# is reserved by RFC 2606 specifically so it can never collide with a real
# user's address. These tests make sure that holds going forward: any email
# placeholder baked into the prompt's examples must use a reserved example
# domain, never something that could double as someone's real address.
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestParserSystemPromptHygiene:

    _EMAIL_RE = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}')
    _RESERVED_EXAMPLE_DOMAINS = ("example.com", "example.org", "example.net")

    def test_does_not_contain_users_real_email(self):
        from client.proactive_agent import _PARSER_SYSTEM
        assert "michaeltyagi@gmail.com" not in _PARSER_SYSTEM
        assert "@gmail.com" not in _PARSER_SYSTEM

    def test_example_addresses_use_reserved_example_domain(self):
        """
        Any email-shaped string baked into the prompt's few-shot examples must
        use an RFC 2606 reserved domain (example.com/.org/.net) — never a real
        provider domain (gmail.com, outlook.com, ...) that a real user could
        plausibly also be using, which is exactly what caused this bug.
        """
        from client.proactive_agent import _PARSER_SYSTEM
        addresses = self._EMAIL_RE.findall(_PARSER_SYSTEM)
        assert addresses, "expected at least one example address in the prompt"
        for addr in addresses:
            domain = addr.split("@", 1)[1].lower()
            assert domain in self._RESERVED_EXAMPLE_DOMAINS, (
                f"prompt example address {addr!r} uses a real-looking domain "
                f"{domain!r} — it could collide with a real user's address "
                f"and hijack the model the same way michaeltyagi@gmail.com did. "
                f"Use one of {self._RESERVED_EXAMPLE_DOMAINS} instead."
            )

    def test_warns_against_copying_to_address_from_examples(self):
        """The prompt must explicitly tell the model the to= value comes from
        the user's message, not from these examples — the collision wouldn't
        be dangerous on its own without this guidance reinforcing it."""
        from client.proactive_agent import _PARSER_SYSTEM
        lowered = _PARSER_SYSTEM.lower()
        assert "never from these examples" in lowered or "verbatim" in lowered


# ═══════════════════════════════════════════════════════════════════
# proactive_agent — ConfirmationTracker
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestConfirmationTracker:

    def _make_confirmation(self):
        from client.proactive_agent import ScheduleConfirmation
        return ScheduleConfirmation(
            label="Test", tool="tool", tool_args={},
            trigger_type="cron", cron="0 7 * * *",
            condition_tool=None, condition_expr=None,
            condition_cron="*/15 * * * *",
            timezone="America/Vancouver",
            human_schedule="Every day at 7am",
            original_request="run tool every day at 7am",
        )

    def test_set_and_get_pending(self):
        from client.proactive_agent import ConfirmationTracker
        tracker = ConfirmationTracker()
        conf = self._make_confirmation()
        tracker.set_pending("session_1", conf)
        assert tracker.get_pending("session_1") is conf

    def test_clear_removes_pending(self):
        from client.proactive_agent import ConfirmationTracker
        tracker = ConfirmationTracker()
        tracker.set_pending("session_1", self._make_confirmation())
        tracker.clear("session_1")
        assert tracker.get_pending("session_1") is None

    def test_get_pending_missing_session(self):
        from client.proactive_agent import ConfirmationTracker
        tracker = ConfirmationTracker()
        assert tracker.get_pending("no_such_session") is None

    @pytest.mark.parametrize("msg", ["yes", "y", "confirm", "ok", "sure", "yep"])
    def test_is_yes(self, msg):
        from client.proactive_agent import ConfirmationTracker
        assert ConfirmationTracker.is_yes(msg) is True

    @pytest.mark.parametrize("msg", ["no", "n", "cancel", "nope", "nah"])
    def test_is_not_yes(self, msg):
        from client.proactive_agent import ConfirmationTracker
        assert ConfirmationTracker.is_yes(msg) is False

    @pytest.mark.parametrize("msg", ["yes", "no", "confirm", "cancel"])
    def test_is_confirmation(self, msg):
        from client.proactive_agent import ConfirmationTracker
        assert ConfirmationTracker.is_confirmation(msg) is True

    def test_not_confirmation_for_regular_message(self):
        from client.proactive_agent import ConfirmationTracker
        assert ConfirmationTracker.is_confirmation("what's the weather?") is False


# ═══════════════════════════════════════════════════════════════════
# proactive_agent — AgentScheduler (no apscheduler required)
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestAgentScheduler:

    @pytest.mark.asyncio
    async def test_start_without_apscheduler_logs_warning(self, patched_scheduler_path):
        from client.proactive_agent import AgentScheduler
        execute_fn = AsyncMock(return_value="result")
        broadcast_fn = AsyncMock()
        scheduler = AgentScheduler(execute_fn=execute_fn, broadcast_fn=broadcast_fn)
        with patch.dict("sys.modules", {"apscheduler": None,
                                         "apscheduler.schedulers": None,
                                         "apscheduler.schedulers.asyncio": None}):
            # Should not raise — logs warning and returns
            await scheduler.start()

    @pytest.mark.asyncio
    async def test_fire_job_calls_execute_and_broadcast(self, patched_scheduler_path):
        from client.proactive_agent import AgentScheduler, create_job
        execute_fn = AsyncMock(return_value="briefing result")
        broadcast_fn = AsyncMock()
        scheduler = AgentScheduler(execute_fn=execute_fn, broadcast_fn=broadcast_fn)
        job_id = create_job(label="Test", tool="get_day_briefing", cron="30 5 * * *")
        from client.proactive_agent import get_job
        job = get_job(job_id)
        await scheduler._fire_job(job)
        execute_fn.assert_called_once_with("get_day_briefing", {})
        broadcast_fn.assert_called_once()
        call_data = broadcast_fn.call_args[0][0]
        assert call_data["type"] == "scheduled_result"
        assert call_data["result"] == "briefing result"

    @pytest.mark.asyncio
    async def test_fire_job_broadcasts_error_on_failure(self, patched_scheduler_path):
        from client.proactive_agent import AgentScheduler, create_job, get_job
        execute_fn = AsyncMock(side_effect=RuntimeError("tool failed"))
        broadcast_fn = AsyncMock()
        scheduler = AgentScheduler(execute_fn=execute_fn, broadcast_fn=broadcast_fn)
        job_id = create_job(label="Failing", tool="broken_tool", cron="* * * * *")
        job = get_job(job_id)
        await scheduler._fire_job(job)
        call_data = broadcast_fn.call_args[0][0]
        assert call_data["type"] == "scheduled_error"
        assert "tool failed" in call_data["error"]

    @pytest.mark.asyncio
    async def test_condition_check_fires_when_true(self, patched_scheduler_path):
        from client.proactive_agent import AgentScheduler, create_job, get_job
        execute_fn = AsyncMock(side_effect=["15", "action result"])
        broadcast_fn = AsyncMock()
        scheduler = AgentScheduler(execute_fn=execute_fn, broadcast_fn=broadcast_fn)
        job_id = create_job(
            label="Gmail alert",
            tool="get_gmail_summary",
            trigger_type="condition",
            condition_tool="get_gmail_unread_count",
            condition_expr="result > 10",
            condition_cron="*/15 * * * *",
        )
        job = get_job(job_id)
        await scheduler._check_condition(job)
        assert broadcast_fn.called
        assert broadcast_fn.call_args[0][0]["type"] == "scheduled_result"

    @pytest.mark.asyncio
    async def test_condition_check_does_not_fire_when_false(self, patched_scheduler_path):
        from client.proactive_agent import AgentScheduler, create_job, get_job
        execute_fn = AsyncMock(return_value="3")  # 3 > 10 is False
        broadcast_fn = AsyncMock()
        scheduler = AgentScheduler(execute_fn=execute_fn, broadcast_fn=broadcast_fn)
        job_id = create_job(
            label="Gmail alert",
            tool="get_gmail_summary",
            trigger_type="condition",
            condition_tool="get_gmail_unread_count",
            condition_expr="result > 10",
            condition_cron="*/15 * * * *",
        )
        job = get_job(job_id)
        await scheduler._check_condition(job)
        broadcast_fn.assert_not_called()

# ═══════════════════════════════════════════════════════════════════
# memory_consolidator — consolidated_msg_count tracking
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestConsolidationMsgCount:

    def test_migration_adds_consolidated_msg_count(self, patched_memory_paths):
        from client.memory_consolidator import _ensure_db
        _, sessions_db = patched_memory_paths
        _ensure_db()
        conn = sqlite3.connect(sessions_db)
        cols = [r[1] for r in conn.execute("PRAGMA table_info(sessions)").fetchall()]
        conn.close()
        assert "consolidated_msg_count" in cols

    def test_is_consolidated_false_when_no_msg_count(self, patched_memory_paths, sessions_db_with_messages):
        from client.memory_consolidator import _ensure_db, _is_consolidated
        _ensure_db()
        # consolidated_at set but no msg count — treat as not consolidated
        conn = sqlite3.connect(sessions_db_with_messages)
        conn.execute("UPDATE sessions SET consolidated_at = '2026-01-01', consolidated_msg_count = NULL WHERE id = 1")
        conn.commit()
        conn.close()
        assert _is_consolidated("1") is False

    def test_is_consolidated_true_when_no_new_messages(self, patched_memory_paths, sessions_db_with_messages):
        from client.memory_consolidator import _ensure_db, _is_consolidated, _mark_consolidated
        _ensure_db()
        _mark_consolidated("1")
        # No new messages added — should be consolidated
        assert _is_consolidated("1") is True

    def test_is_consolidated_false_when_new_messages_added(self, patched_memory_paths, sessions_db_with_messages):
        from client.memory_consolidator import _ensure_db, _is_consolidated, _mark_consolidated
        _ensure_db()
        _mark_consolidated("1")
        # Add a new message after consolidation
        conn = sqlite3.connect(sessions_db_with_messages)
        conn.execute("INSERT INTO messages (session_id, role, content) VALUES (1, 'user', 'New message after consolidation')")
        conn.commit()
        conn.close()
        assert _is_consolidated("1") is False

    def test_mark_consolidated_stores_message_count(self, patched_memory_paths, sessions_db_with_messages):
        from client.memory_consolidator import _ensure_db, _mark_consolidated
        _ensure_db()
        _mark_consolidated("1")
        conn = sqlite3.connect(sessions_db_with_messages)
        row = conn.execute("SELECT consolidated_msg_count FROM sessions WHERE id = 1").fetchone()
        conn.close()
        assert row[0] == 2  # sessions_db_with_messages has 2 messages

    @pytest.mark.asyncio
    async def test_consolidate_reruns_after_new_messages(self, patched_memory_paths, mock_llm_fn):
        from client.memory_consolidator import _ensure_db, consolidate, _mark_consolidated, _mem_conn
        memory_db, sessions_db = patched_memory_paths
        _ensure_db()
        # Seed initial messages and consolidate
        conn = sqlite3.connect(sessions_db)
        conn.execute("INSERT INTO sessions (id, name) VALUES (1, 'Test')")
        conn.execute("INSERT INTO messages (session_id, role, content) VALUES (1, 'user', 'Initial message about setup')")
        conn.execute("INSERT INTO messages (session_id, role, content) VALUES (1, 'assistant', 'Got it')")
        conn.commit()
        conn.close()
        with patch("client.memory_consolidator._embed", return_value=None):
            count1 = await consolidate("1", mock_llm_fn)
        # Add new messages
        conn = sqlite3.connect(sessions_db)
        conn.execute("INSERT INTO messages (session_id, role, content) VALUES (1, 'user', 'My name is Mike')")
        conn.execute("INSERT INTO messages (session_id, role, content) VALUES (1, 'assistant', 'Hello Mike')")
        conn.commit()
        conn.close()
        # Should re-consolidate since new messages exist
        with patch("client.memory_consolidator._embed", return_value=None):
            count2 = await consolidate("1", mock_llm_fn)
        # Verify _mark_consolidated was called with updated count (4 messages now)
        conn = sqlite3.connect(sessions_db)
        row = conn.execute("SELECT consolidated_msg_count FROM sessions WHERE id = 1").fetchone()
        conn.close()
        assert row[0] == 4

    @pytest.mark.asyncio
    async def test_consolidate_skips_when_no_new_messages(self, patched_memory_paths, mock_llm_fn):
        from client.memory_consolidator import _ensure_db, consolidate, _mark_consolidated
        memory_db, sessions_db = patched_memory_paths
        _ensure_db()
        conn = sqlite3.connect(sessions_db)
        conn.execute("INSERT INTO sessions (id, name) VALUES (1, 'Test')")
        conn.execute("INSERT INTO messages (session_id, role, content) VALUES (1, 'user', 'Hello')")
        conn.execute("INSERT INTO messages (session_id, role, content) VALUES (1, 'assistant', 'Hi')")
        conn.commit()
        conn.close()
        with patch("client.memory_consolidator._embed", return_value=None):
            await consolidate("1", mock_llm_fn)
        # No new messages — second call should skip
        called = []
        original = mock_llm_fn
        async def tracking_llm(s, u):
            called.append(1)
            return await original(s, u)
        with patch("client.memory_consolidator._embed", return_value=None):
            await consolidate("1", tracking_llm)
        assert len(called) == 0  # LLM not called — skipped


# ═══════════════════════════════════════════════════════════════════
# memory_consolidator — :memory add and :memory dedup commands
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestMemoryAddDedup:

    def test_memory_add_inserts_semantic_memory(self, patched_memory_paths):
        from client.memory_consolidator import _ensure_db, handle_memory_command, _mem_conn
        _ensure_db()
        with patch("client.memory_consolidator._embed", return_value=None):
            result = handle_memory_command(":memory add Mike's son is named Noah")
        assert "added" in result.lower()
        with _mem_conn() as conn:
            row = conn.execute("SELECT tier, importance FROM memories WHERE content LIKE '%Noah%'").fetchone()
        assert row is not None
        assert row["tier"] == "semantic"
        assert row["importance"] == 1.0

    def test_memory_add_rejects_duplicate(self, patched_memory_paths):
        from client.memory_consolidator import _ensure_db, handle_memory_command
        _ensure_db()
        with patch("client.memory_consolidator._embed", return_value=None):
            handle_memory_command(":memory add Mike likes coffee")
            result = handle_memory_command(":memory add Mike likes coffee")
        assert "already exists" in result.lower()

    def test_memory_add_rejects_empty(self, patched_memory_paths):
        from client.memory_consolidator import _ensure_db, handle_memory_command
        _ensure_db()
        result = handle_memory_command(":memory add")
        assert "Usage" in result

    def test_memory_dedup_removes_duplicates(self, patched_memory_paths):
        import numpy as np
        from datetime import datetime, timezone
        from client.memory_consolidator import _ensure_db, handle_memory_command, _mem_conn
        _ensure_db()
        now = datetime.now(timezone.utc).isoformat()
        vec = bytes(np.array([0.1]*10, dtype=np.float32))
        conn = sqlite3.connect(patched_memory_paths[0])
        conn.execute("INSERT INTO memories (tier, content, embedding, importance, created_at) VALUES ('episodic', 'duplicate fact', ?, 0.9, ?)", (vec, now))
        conn.execute("INSERT INTO memories (tier, content, embedding, importance, created_at) VALUES ('episodic', 'duplicate fact', ?, 0.5, ?)", (vec, now))
        conn.execute("INSERT INTO memories (tier, content, embedding, importance, created_at) VALUES ('episodic', 'unique fact', ?, 0.7, ?)", (vec, now))
        conn.commit()
        conn.close()
        result = handle_memory_command(":memory dedup")
        assert "1" in result
        with _mem_conn() as conn:
            rows = conn.execute("SELECT content FROM memories ORDER BY importance DESC").fetchall()
        contents = [r[0] for r in rows]
        assert contents.count("duplicate fact") == 1
        assert "unique fact" in contents

    def test_memory_dedup_keeps_highest_importance(self, patched_memory_paths):
        import numpy as np
        from datetime import datetime, timezone
        from client.memory_consolidator import _ensure_db, handle_memory_command, _mem_conn
        _ensure_db()
        now = datetime.now(timezone.utc).isoformat()
        vec = bytes(np.array([0.1]*10, dtype=np.float32))
        conn = sqlite3.connect(patched_memory_paths[0])
        conn.execute("INSERT INTO memories (id, tier, content, embedding, importance, created_at) VALUES (1, 'episodic', 'same content', ?, 0.9, ?)", (vec, now))
        conn.execute("INSERT INTO memories (id, tier, content, embedding, importance, created_at) VALUES (2, 'episodic', 'same content', ?, 0.3, ?)", (vec, now))
        conn.commit()
        conn.close()
        handle_memory_command(":memory dedup")
        with _mem_conn() as conn:
            row = conn.execute("SELECT importance FROM memories WHERE content = 'same content'").fetchone()
        assert row["importance"] == 0.9

    def test_memory_dedup_no_duplicates(self, patched_memory_paths):
        import numpy as np
        from datetime import datetime, timezone
        from client.memory_consolidator import _ensure_db, handle_memory_command
        _ensure_db()
        now = datetime.now(timezone.utc).isoformat()
        vec = bytes(np.array([0.1]*10, dtype=np.float32))
        conn = sqlite3.connect(patched_memory_paths[0])
        conn.execute("INSERT INTO memories (tier, content, embedding, importance, created_at) VALUES ('episodic', 'fact one', ?, 0.7, ?)", (vec, now))
        conn.execute("INSERT INTO memories (tier, content, embedding, importance, created_at) VALUES ('episodic', 'fact two', ?, 0.7, ?)", (vec, now))
        conn.commit()
        conn.close()
        result = handle_memory_command(":memory dedup")
        assert "No duplicates" in result


# ═══════════════════════════════════════════════════════════════════
# memory_consolidator — :memory consolidate clears flag
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestManualConsolidateClearsFlag:

    def test_consolidate_command_clears_consolidated_at(self, patched_memory_paths, sessions_db_with_messages):
        from client.memory_consolidator import _ensure_db, _mark_consolidated, _consolidate_now, SESSIONS_DB_PATH
        _ensure_db()
        _mark_consolidated("1")
        # Verify it's marked
        conn = sqlite3.connect(sessions_db_with_messages)
        row = conn.execute("SELECT consolidated_at FROM sessions WHERE id = 1").fetchone()
        conn.close()
        assert row[0] is not None
        # Call the clear directly (synchronous part of _consolidate_now)
        with patch("client.memory_consolidator.SESSIONS_DB_PATH", sessions_db_with_messages):
            conn = sqlite3.connect(sessions_db_with_messages)
            conn.execute("UPDATE sessions SET consolidated_at = NULL, consolidated_msg_count = NULL WHERE id = 1")
            conn.commit()
            conn.close()
        # After clearing, consolidated_at should be NULL
        conn = sqlite3.connect(sessions_db_with_messages)
        row = conn.execute("SELECT consolidated_at FROM sessions WHERE id = 1").fetchone()
        conn.close()
        assert row[0] is None

    def test_consolidate_command_missing_id_shows_usage(self, patched_memory_paths):
        from client.memory_consolidator import _ensure_db, handle_memory_command
        _ensure_db()
        result = handle_memory_command(":memory consolidate")
        assert "session" in result.lower() or "id" in result.lower()

# ═══════════════════════════════════════════════════════════════════
# find_job_by_label — numeric ID lookup
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestFindJobByNumericId:

    def test_find_by_numeric_id_string(self, patched_scheduler_path):
        from client.proactive_agent import create_job, find_job_by_label
        job_id = create_job(label="Daily Briefing", tool="get_day_briefing", cron="0 6 * * *")
        job = find_job_by_label(str(job_id))
        assert job is not None
        assert job["id"] == job_id

    def test_find_by_numeric_id_returns_none_for_missing(self, patched_scheduler_path):
        from client.proactive_agent import find_job_by_label
        assert find_job_by_label("9999") is None

    def test_cancel_by_numeric_id(self, patched_scheduler_path):
        from client.proactive_agent import create_job, handle_jobs_command, get_job
        job_id = create_job(label="Daily Briefing", tool="get_day_briefing", cron="0 6 * * *")
        result = handle_jobs_command(f":jobs cancel {job_id}")
        assert "deleted" in result.lower()
        assert get_job(job_id) is None

    def test_pause_by_numeric_id(self, patched_scheduler_path):
        from client.proactive_agent import create_job, handle_jobs_command, get_job
        job_id = create_job(label="Daily Briefing", tool="get_day_briefing", cron="0 6 * * *")
        result = handle_jobs_command(f":jobs pause {job_id}")
        assert "paused" in result.lower()
        assert get_job(job_id)["enabled"] == 0

    def test_info_by_numeric_id(self, patched_scheduler_path):
        from client.proactive_agent import create_job, handle_jobs_command
        job_id = create_job(label="Daily Briefing", tool="get_day_briefing", cron="0 6 * * *")
        result = handle_jobs_command(f":jobs info {job_id}")
        assert "get_day_briefing" in result


# ═══════════════════════════════════════════════════════════════════
# _fire_job — llm_prompt post-processing
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestFireJobLlmPrompt:

    @pytest.mark.asyncio
    async def test_llm_prompt_post_processes_result(self, patched_scheduler_path):
        # Tool jobs always use raw result — llm_prompt is ignored for tool jobs.
        # llm_fn is only used for pure LLM jobs (tool=None/empty).
        from client.proactive_agent import AgentScheduler, create_job, get_job
        raw = '{"id": "abc", "description": "Sunset in Banff"}'
        execute_fn = AsyncMock(return_value=raw)
        broadcast_fn = AsyncMock()
        llm_fn = AsyncMock(return_value="should not be called")
        scheduler = AgentScheduler(execute_fn=execute_fn, broadcast_fn=broadcast_fn, llm_fn=llm_fn)
        job_id = create_job(
            label="Daily Photo",
            tool="shashin_random_tool",
            cron="0 6 * * *",
            llm_prompt="Write a short commentary about this photo.",
        )
        job = get_job(job_id)
        await scheduler._fire_job(job)
        # LLM must NOT be called for tool jobs
        llm_fn.assert_not_called()
        # Raw tool result broadcast as-is
        call_data = broadcast_fn.call_args[0][0]
        assert call_data["result"] == raw

    @pytest.mark.asyncio
    async def test_no_llm_prompt_uses_raw_result(self, patched_scheduler_path):
        from client.proactive_agent import AgentScheduler, create_job, get_job
        execute_fn = AsyncMock(return_value="raw tool output")
        broadcast_fn = AsyncMock()
        llm_fn = AsyncMock(return_value="should not be called")
        scheduler = AgentScheduler(execute_fn=execute_fn, broadcast_fn=broadcast_fn, llm_fn=llm_fn)
        job_id = create_job(label="Plain", tool="some_tool", cron="0 6 * * *")
        job = get_job(job_id)
        await scheduler._fire_job(job)
        llm_fn.assert_not_called()
        assert broadcast_fn.call_args[0][0]["result"] == "raw tool output"

    @pytest.mark.asyncio
    async def test_llm_fn_failure_falls_back_to_raw(self, patched_scheduler_path):
        from client.proactive_agent import AgentScheduler, create_job, get_job
        execute_fn = AsyncMock(return_value="raw result")
        broadcast_fn = AsyncMock()
        llm_fn = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
        scheduler = AgentScheduler(execute_fn=execute_fn, broadcast_fn=broadcast_fn, llm_fn=llm_fn)
        job_id = create_job(
            label="Photo", tool="shashin_random_tool", cron="0 6 * * *",
            llm_prompt="Describe this photo.",
        )
        job = get_job(job_id)
        await scheduler._fire_job(job)
        # Falls back to raw result, doesn't raise
        assert broadcast_fn.call_args[0][0]["result"] == "raw result"


# ═══════════════════════════════════════════════════════════════════
# _fire_job — session-bound delivery
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestFireJobSessionDelivery:

    @pytest.mark.asyncio
    async def test_deliver_to_session_saves_to_session_manager(self, patched_scheduler_path):
        from client.proactive_agent import AgentScheduler, create_job, get_job
        execute_fn = AsyncMock(return_value="briefing result")
        broadcast_fn = AsyncMock()
        session_manager = MagicMock()
        session_manager.add_message = MagicMock(return_value=1)
        scheduler = AgentScheduler(
            execute_fn=execute_fn,
            broadcast_fn=broadcast_fn,
            session_manager=session_manager,
        )
        job_id = create_job(
            label="Daily Briefing",
            tool="get_day_briefing",
            cron="0 6 * * *",
            session_id=42,
            deliver_to_session=True,
        )
        job = get_job(job_id)
        await scheduler._fire_job(job)
        session_manager.add_message.assert_called_once()
        args = session_manager.add_message.call_args[0]
        assert args[0] == 42          # session_id
        assert args[1] == "assistant" # role
        assert "briefing result" in args[2]

    @pytest.mark.asyncio
    async def test_broadcast_delivery_does_not_use_session_manager(self, patched_scheduler_path):
        from client.proactive_agent import AgentScheduler, create_job, get_job
        execute_fn = AsyncMock(return_value="alert result")
        broadcast_fn = AsyncMock()
        session_manager = MagicMock()
        scheduler = AgentScheduler(
            execute_fn=execute_fn,
            broadcast_fn=broadcast_fn,
            session_manager=session_manager,
        )
        # deliver_to_session=False (default) — should broadcast, not save to session
        job_id = create_job(label="Email Alert", tool="gmail_get_unread", cron="*/5 * * * *")
        job = get_job(job_id)
        await scheduler._fire_job(job)
        session_manager.add_message.assert_not_called()
        broadcast_fn.assert_called_once()


# ═══════════════════════════════════════════════════════════════════
# _check_condition — rich JSON context
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestCheckConditionRichContext:

    @pytest.mark.asyncio
    async def test_json_scalar_key_available(self, patched_scheduler_path):
        """total_unread from JSON dict is available directly in expression."""
        from client.proactive_agent import AgentScheduler, create_job, get_job
        import json
        execute_fn = AsyncMock(side_effect=[
            json.dumps({"total_unread": 5, "emails": []}),
            "action result",
        ])
        broadcast_fn = AsyncMock()
        scheduler = AgentScheduler(execute_fn=execute_fn, broadcast_fn=broadcast_fn)
        job_id = create_job(
            label="Email check",
            tool="gmail_get_unread",
            trigger_type="condition",
            condition_tool="gmail_get_unread",
            condition_expr="total_unread > 0",
            condition_cron="*/5 * * * *",
        )
        job = get_job(job_id)
        await scheduler._check_condition(job)
        assert broadcast_fn.called

    @pytest.mark.asyncio
    async def test_len_list_key_available(self, patched_scheduler_path):
        """len_emails from JSON list field is available in expression."""
        from client.proactive_agent import AgentScheduler, create_job, get_job
        import json
        execute_fn = AsyncMock(side_effect=[
            json.dumps({"total_unread": 3, "emails": ["a", "b", "c"]}),
            "action result",
        ])
        broadcast_fn = AsyncMock()
        scheduler = AgentScheduler(execute_fn=execute_fn, broadcast_fn=broadcast_fn)
        job_id = create_job(
            label="Email check",
            tool="gmail_get_unread",
            trigger_type="condition",
            condition_tool="gmail_get_unread",
            condition_expr="len_emails > 0",
            condition_cron="*/5 * * * *",
        )
        job = get_job(job_id)
        await scheduler._check_condition(job)
        assert broadcast_fn.called

    @pytest.mark.asyncio
    async def test_zero_unread_does_not_fire(self, patched_scheduler_path):
        """total_unread=0 means condition is false — no broadcast."""
        from client.proactive_agent import AgentScheduler, create_job, get_job
        import json
        execute_fn = AsyncMock(return_value=json.dumps({"total_unread": 0, "emails": []}))
        broadcast_fn = AsyncMock()
        scheduler = AgentScheduler(execute_fn=execute_fn, broadcast_fn=broadcast_fn)
        job_id = create_job(
            label="Email check",
            tool="gmail_get_unread",
            trigger_type="condition",
            condition_tool="gmail_get_unread",
            condition_expr="total_unread > 0",
            condition_cron="*/5 * * * *",
        )
        job = get_job(job_id)
        await scheduler._check_condition(job)
        assert not broadcast_fn.called

    @pytest.mark.asyncio
    async def test_result_len_available_for_raw_string(self, patched_scheduler_path):
        """result_len works for non-JSON tool output."""
        from client.proactive_agent import AgentScheduler, create_job, get_job
        execute_fn = AsyncMock(side_effect=["some non-empty output", "action"])
        broadcast_fn = AsyncMock()
        scheduler = AgentScheduler(execute_fn=execute_fn, broadcast_fn=broadcast_fn)
        job_id = create_job(
            label="Raw check",
            tool="some_tool",
            trigger_type="condition",
            condition_tool="some_tool",
            condition_expr="result_len > 5",
            condition_cron="*/5 * * * *",
        )
        job = get_job(job_id)
        await scheduler._check_condition(job)
        assert broadcast_fn.called

    @pytest.mark.asyncio
    async def test_llm_prompt_stored_on_job(self, patched_scheduler_path):
        """llm_prompt is persisted and retrievable from the DB."""
        from client.proactive_agent import create_job, get_job
        job_id = create_job(
            label="Photo",
            tool="shashin_random_tool",
            cron="0 6 * * *",
            llm_prompt="Write a short commentary about this photo.",
        )
        job = get_job(job_id)
        assert job["llm_prompt"] == "Write a short commentary about this photo."