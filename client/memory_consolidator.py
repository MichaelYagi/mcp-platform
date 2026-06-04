"""
client/memory_consolidator.py
==============================
Persistent memory layer for mcp-platform.

Three memory tiers:
  - episodic : significant events/decisions/outcomes from sessions
  - semantic  : distilled facts/preferences/patterns (promoted from episodic)
  - (working  : current session context — already handled by conversation_state)

Flow:
  1. Session ends (or 15min inactivity) → consolidate(session_id)
  2. LLM extracts structured memories from the transcript
  3. Memories written to memory.db with embeddings (optional)
  4. On new session → inject_into_system_prompt() prepends relevant memories

DB lives at data/memory.db alongside sessions.db.
Schema migration (add consolidated_at to sessions) runs automatically.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sqlite3
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("mcp_client")


def cosine_similarity(a: bytes, b: bytes) -> float:
    """Cosine similarity between two float32 byte blobs."""
    try:
        n = len(a) // 4
        va = struct.unpack(f"{n}f", a)
        vb = struct.unpack(f"{n}f", b)
        dot = sum(x * y for x, y in zip(va, vb))
        na  = sum(x * x for x in va) ** 0.5
        nb  = sum(x * x for x in vb) ** 0.5
        return dot / (na * nb) if na and nb else 0.0
    except Exception:
        return 0.0

# ── Paths ────────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEMORY_DB_PATH = _PROJECT_ROOT / "data" / "memory.db"
SESSIONS_DB_PATH = _PROJECT_ROOT / "data" / "sessions.db"

# Inactivity threshold before auto-consolidation fires
INACTIVITY_SECONDS = 15 * 60  # 15 minutes

# ── Schema ───────────────────────────────────────────────────────────────────

_MEMORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    tier         TEXT    NOT NULL CHECK(tier IN ('episodic','semantic')),
    content      TEXT    NOT NULL,
    embedding    BLOB,
    source_session TEXT,
    importance   REAL    NOT NULL DEFAULT 0.5,
    access_count INTEGER NOT NULL DEFAULT 0,
    last_accessed TEXT,
    created_at   TEXT    NOT NULL,
    promoted_at  TEXT
);

CREATE TABLE IF NOT EXISTS memory_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_SESSIONS_MIGRATION = """
ALTER TABLE sessions ADD COLUMN consolidated_at TEXT;
ALTER TABLE sessions ADD COLUMN consolidated_msg_count INTEGER;
"""


# ── DB helpers ───────────────────────────────────────────────────────────────

def _ensure_db():
    """Create memory.db and run sessions.db migration if needed."""
    MEMORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Memory DB
    with sqlite3.connect(MEMORY_DB_PATH) as conn:
        conn.executescript(_MEMORY_SCHEMA)
        # Migration: add embedding column if upgrading from schema without it
        cols = [r[1] for r in conn.execute("PRAGMA table_info(memories)").fetchall()]
        if "embedding" not in cols:
            conn.execute("ALTER TABLE memories ADD COLUMN embedding BLOB")
            logger.info("💾 memory.db migrated: added embedding column")

    # Sessions DB migration — add consolidated_at if missing
    if SESSIONS_DB_PATH.exists():
        try:
            with sqlite3.connect(SESSIONS_DB_PATH) as conn:
                cols = [r[1] for r in conn.execute("PRAGMA table_info(sessions)").fetchall()]
                if "consolidated_at" not in cols:
                    conn.execute("ALTER TABLE sessions ADD COLUMN consolidated_at TEXT")
                    logger.info("💾 sessions.db migrated: added consolidated_at column")
                if "consolidated_msg_count" not in cols:
                    conn.execute("ALTER TABLE sessions ADD COLUMN consolidated_msg_count INTEGER")
                    logger.info("💾 sessions.db migrated: added consolidated_msg_count column")
        except Exception as e:
            logger.warning(f"⚠️ sessions.db migration skipped: {e}")


def _mem_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(MEMORY_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _sess_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(SESSIONS_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ── Embedding helpers (same bge-large pipeline as conversation_rag) ───────────

_embeddings_model = None
_SBERT_MODEL = "all-MiniLM-L6-v2"  # 22 MB, 384-dim, CPU-native via sentence-transformers


def _maybe_clear_stale_embeddings(current_model: str) -> None:
    """If the stored embedding model name differs from *current_model*, null out
    all stored embeddings to avoid comparing vectors from incompatible spaces."""
    if not MEMORY_DB_PATH.exists():
        return
    try:
        with _mem_conn() as conn:
            row = conn.execute(
                "SELECT value FROM memory_meta WHERE key='embedding_model'"
            ).fetchone()
            stored = row[0] if row else None
            if stored and stored != current_model:
                conn.execute("UPDATE memories SET embedding = NULL")
                logger.info(
                    f"Embedding model changed ({stored} → {current_model}), "
                    "cleared stored embeddings — will re-embed on next consolidation"
                )
            conn.execute(
                "INSERT OR REPLACE INTO memory_meta(key, value) VALUES('embedding_model', ?)",
                (current_model,),
            )
    except Exception as e:
        logger.warning(f"🧠 Embedding compat check failed: {e}")


def _get_embeddings_model():
    """Return the embedding model, preferring CPU-native sentence-transformers.

    Falls back to Ollama bge-large if sentence-transformers is unavailable.
    Clears stored embeddings when the model changes to prevent dimension mismatch.
    """
    global _embeddings_model
    if _embeddings_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _embeddings_model = SentenceTransformer(_SBERT_MODEL)
            _maybe_clear_stale_embeddings(f"sentence-transformers/{_SBERT_MODEL}")
            logger.info(f"Memory embeddings: sentence-transformers/{_SBERT_MODEL} (CPU)")
        except Exception as _e:
            logger.info(f"sentence-transformers unavailable ({_e}), using bge-large via Ollama")
            from langchain_ollama import OllamaEmbeddings
            base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
            _embeddings_model = OllamaEmbeddings(model="bge-large", base_url=base_url)
            _maybe_clear_stale_embeddings("ollama/bge-large")
    return _embeddings_model


def _embed(text: str) -> Optional[bytes]:
    """Embed *text* and return raw float32 bytes, or None on failure."""
    try:
        import numpy as np
        model = _get_embeddings_model()
        if hasattr(model, "encode"):
            # sentence-transformers API
            vec = model.encode(text, normalize_embeddings=True)
        else:
            # langchain OllamaEmbeddings API
            vec = model.embed_query(text)
        return np.array(vec, dtype=np.float32).tobytes()
    except Exception as e:
        logger.warning(f"🧠 Memory embedding failed: {e}")
        return None


def _cosine(a_bytes: bytes, b_bytes: bytes) -> float:
    """Cosine similarity between two float32 byte blobs."""
    import numpy as np
    a = np.frombuffer(a_bytes, dtype=np.float32)
    b = np.frombuffer(b_bytes, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ── Consolidation guard ──────────────────────────────────────────────────────

def _get_session_msg_count(session_id: str) -> int:
    """Return current message count for a session."""
    if not SESSIONS_DB_PATH.exists():
        return 0
    try:
        with _sess_conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
            ).fetchone()
            return row[0] if row else 0
    except Exception:
        return 0


def _is_consolidated(session_id: str) -> bool:
    """True if session has been consolidated and no new messages since last run."""
    if not SESSIONS_DB_PATH.exists():
        return False
    try:
        with _sess_conn() as conn:
            row = conn.execute(
                "SELECT consolidated_at, consolidated_msg_count FROM sessions WHERE id = ?",
                (session_id,)
            ).fetchone()
            if row is None or row["consolidated_at"] is None:
                return False
            last_count = row["consolidated_msg_count"] or 0
            current_count = _get_session_msg_count(session_id)
            # Re-consolidate if new messages have been added
            return current_count <= last_count
    except Exception:
        return False


def _mark_consolidated(session_id: str):
    if not SESSIONS_DB_PATH.exists():
        return
    try:
        now = datetime.now(timezone.utc).isoformat()
        msg_count = _get_session_msg_count(session_id)
        with _sess_conn() as conn:
            conn.execute(
                "UPDATE sessions SET consolidated_at = ?, consolidated_msg_count = ? WHERE id = ?",
                (now, msg_count, session_id)
            )
    except Exception as e:
        logger.warning(f"⚠️ Could not mark session {session_id} consolidated: {e}")


# ── LLM extraction prompt ─────────────────────────────────────────────────────

_EXTRACT_PROMPT = """You are a memory extraction system. Your job is to extract durable facts about the user from the conversation below.

Extract facts that will still be true weeks or months from now. Focus on who the user is, not what they did today.

ALWAYS extract these categories if present:
- Full name, nicknames
- Family members and their details (spouse, children, parents, siblings — names, ages, jobs, hobbies, where they're from)
- Where the user was born, grew up, lives now
- Education (schools, degrees, fields of study)
- Career (job title, employer, past jobs, career changes)
- Hobbies and interests (past and present)
- Technical environment (hardware, software, tools, config)
- Projects they work on
- Preferences and working style
- Health conditions and physical limitations
- Solved problems and outcomes

EXAMPLES of good extractions:
- "Mike's wife is Ryuko, a dental hygienist who plays piano"
- "Mike's son Noah is 11 years old"
- "Noah plays cello but doesn't enjoy it"
- "Noah excels at swimming and enjoys manga and video games"
- "Mike was born in Richmond BC"
- "Ryuko is from Chigasaki Japan"
- "Noah was born in Ottawa Ontario"
- "Mike studied music (violin) at University of Ottawa"
- "Mike and Ryuko met at University of Ottawa"
- "Mike switched to computer science at Carleton University"
- "Mike's family moved to Surrey BC in 2015"

DO NOT skip personal facts just because they seem unrelated to tech. Family, origins, education are all important.

DO NOT extract time-bound or transient events — these will become stale and are not worth storing:
- Calendar events, meetings, appointments (e.g. "Mike has PTO from May 14-16")
- One-time tasks or reminders
- Current news or recent events discussed in the session
- Anything with a specific date that will have passed

DO NOT extract facts already covered by existing memories. Check carefully — if an existing memory says "Ryuko is a dental hygienist who plays piano", do not add "Ryuko is Mike's wife, a dental hygienist who plays piano" as a separate entry. Prefer to update the existing memory than create a near-duplicate.

Existing memories (do not duplicate):
{existing}

Transcript:
{transcript}

Return ONLY a JSON array. Each object:
{{
  "content": "concise fact in plain English (max 120 chars)",
  "tier": "episodic",
  "importance": 0.0-1.0
}}

Importance guide: identity/family = 0.9, location/education = 0.8, technical facts = 0.7, preferences = 0.8, health = 0.9, outcomes = 0.6

If nothing new to extract, return [].
Return ONLY the JSON array. No preamble, no markdown fences."""


# ── Technical decision extraction prompt ──────────────────────────────────────

_DECISION_PROMPT = """You are a technical session summarizer. Extract the key technical decisions, fixes, and outcomes from this conversation.

Focus on things that would be useful to know in a future session working on the same codebase:
- Bugs that were fixed (what was wrong, what file, what the fix was)
- Features that were added (what was added, where, how it works)
- Architectural decisions (why X was chosen over Y)
- Key file paths and what they do
- Configuration changes
- Patterns or approaches established

Each entry should be a self-contained fact that makes sense without the full conversation context.

EXAMPLES of good extractions:
- "Fixed stale year in search queries: langgraph.py confidence-check prompt now says do not include years"
- "Added broadcast_proactive_result() to client/websocket.py for scheduled job delivery to UI"
- "Per-query memory injection added in websocket.py before run_agent_fn call using inject_into_system_prompt()"
- "Cross-encoder now loads eagerly at startup in tools/rag/rag_search.py to avoid per-query load overhead"
- "websocket.py unrecognised colon commands now return error immediately instead of falling through to LLM"

DO NOT extract:
- General conversation or questions
- Things that were discussed but not actually implemented
- Opinions or recommendations that weren't acted on
- Personal facts about the user (those go in memory)

Transcript:
{transcript}

Return ONLY a JSON array of strings. Each string is one decision/outcome (max 200 chars each).
If there are no significant technical decisions, return [].
Return ONLY the JSON array. No preamble, no markdown fences."""


async def consolidate_decisions(session_id: str, llm_fn, session_manager=None) -> int:
    """
    Extract technical decisions and project context from a session and store
    them in the RAG chunks table so future sessions can search for them.

    Returns number of decisions written.
    """
    _ensure_db()

    # Reuse the same transcript fetch as memory consolidation
    transcript = _get_transcript(session_id, session_manager)
    if not transcript or len(transcript) < 200:
        return 0

    # Only extract if transcript looks technical (has code/file references)
    technical_signals = [".py", ".js", ".css", ".html", "def ", "class ", "import ",
                         "fix", "added", "update", "error", "bug", "function"]
    if not any(sig in transcript.lower() for sig in technical_signals):
        logger.debug(f"🧠 Session {session_id} has no technical content — skipping decision extraction")
        return 0

    prompt = _DECISION_PROMPT.format(transcript=transcript[:6000])

    try:
        raw = await llm_fn(
            "You are a technical session summarizer. Return only valid JSON.",
            prompt
        )
    except Exception as e:
        logger.warning(f"🧠 Decision extraction LLM call failed: {e}")
        return 0

    # Parse decisions
    try:
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
        decisions = json.loads(text)
        if not isinstance(decisions, list):
            return 0
        decisions = [d.strip() for d in decisions if isinstance(d, str) and len(d.strip()) > 10]
    except Exception as e:
        logger.warning(f"🧠 Decision extraction parse failed: {e}")
        return 0

    if not decisions:
        return 0

    # Store decisions in sessions.db chunks table as a single RAG document
    # Source format: session_decisions_{session_id} for easy identification/deletion
    source = f"session_decisions_{session_id}"
    from datetime import date as _date
    date_str = _date.today().isoformat()
    content = f"[Session {session_id} — {date_str}]\n" + "\n".join(f"- {d}" for d in decisions)

    try:
        embedding = _embed(content)
        with sqlite3.connect(SESSIONS_DB_PATH) as conn:
            conn.row_factory = sqlite3.Row
            # Delete any previous decision entry for this session
            conn.execute("DELETE FROM chunks WHERE source = ?", (source,))
            conn.execute(
                """INSERT INTO chunks (chunk_text, embedding, source, created_at)
                   VALUES (?, ?, ?, ?)""",
                (content, embedding, source, datetime.now(timezone.utc).isoformat())
            )
            conn.commit()
        logger.info(f"🧠 Stored {len(decisions)} technical decision(s) for session {session_id}")
        return len(decisions)
    except Exception as e:
        logger.warning(f"🧠 Failed to store decisions for session {session_id}: {e}")
        return 0


# ── Core consolidation ────────────────────────────────────────────────────────

async def consolidate(session_id: str, llm_fn, session_manager=None) -> int:
    """
    Extract memories from a session transcript and persist them.

    llm_fn: async callable(system: str, user: str) -> str
    Returns number of memories written.
    """
    logger.info(f"🧠 consolidate() called for session {session_id}")
    _ensure_db()

    if _is_consolidated(session_id):
        logger.debug(f"🧠 Session {session_id} already consolidated and no new messages — skipping")
        return 0

    # Fetch transcript from sessions.db via session_manager or direct query
    transcript = _get_transcript(session_id, session_manager)
    logger.info(f"🧠 Transcript length for session {session_id}: {len(transcript)} chars")
    if not transcript or len(transcript) < 100:
        logger.debug(f"🧠 Session {session_id} too short to consolidate")
        _mark_consolidated(session_id)
        return 0

    # Load existing semantic memories to avoid duplication
    existing = _get_recent_memories(limit=30)
    existing_text = "\n".join(f"- {m['content']}" for m in existing) or "None yet."

    system = "You are a memory extraction system. Return only valid JSON."
    user = _EXTRACT_PROMPT.format(existing=existing_text, transcript=transcript)

    try:
        raw = await llm_fn(system, user)
    except Exception as e:
        logger.error(f"🧠 Memory LLM call failed for session {session_id}: {e}")
        return 0

    memories = _parse_memories(raw)
    if not memories:
        _mark_consolidated(session_id)
        return 0

    now = datetime.now(timezone.utc).isoformat()
    written = 0
    with _mem_conn() as conn:
        existing_contents = {
            r[0].lower().strip()
            for r in conn.execute("SELECT content FROM memories").fetchall()
        }
        # Load embeddings for semantic near-duplicate detection
        existing_embeddings = conn.execute(
            "SELECT content, embedding FROM memories WHERE embedding IS NOT NULL"
        ).fetchall()

        for m in memories:
            content = m.get("content", "").strip()
            if not content or len(content) < 10:
                continue
            # Exact match check
            if content.lower().strip() in existing_contents:
                logger.debug(f"🧠 Skipping duplicate memory: {content[:60]}")
                continue
            importance = float(m.get("importance", 0.5))
            importance = max(0.0, min(1.0, importance))
            embedding = _embed(content)
            # Semantic near-duplicate check (catches typo variants and rephrasing)
            if embedding:
                is_near_dup = any(
                    existing_emb and cosine_similarity(embedding, existing_emb) > 0.92
                    for _, existing_emb in existing_embeddings
                )
                if is_near_dup:
                    logger.debug(f"🧠 Skipping near-duplicate memory (semantic): {content[:60]}")
                    continue
            conn.execute(
                """INSERT INTO memories
                   (tier, content, embedding, source_session, importance, created_at)
                   VALUES ('episodic', ?, ?, ?, ?, ?)""",
                (content, embedding, str(session_id), importance, now)
            )
            existing_contents.add(content.lower().strip())
            if embedding:
                existing_embeddings.append((content, embedding))
            written += 1

    _mark_consolidated(session_id)
    logger.info(f"🧠 Consolidated session {session_id}: {written} memories written")
    return written


def _get_transcript(session_id: str, session_manager=None) -> str:
    """Build a plain-text transcript from session messages."""
    messages = []

    if session_manager:
        try:
            messages = session_manager.get_session_messages(str(session_id))
            logger.info(f"🧠 Got {len(messages)} messages from session_manager for session {session_id}")
        except Exception as e:
            logger.warning(f"🧠 session_manager.get_session_messages failed: {e}")

    if not messages and SESSIONS_DB_PATH.exists():
        try:
            with _sess_conn() as conn:
                rows = conn.execute(
                    "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id",
                    (session_id,)
                ).fetchall()
                messages = [{"role": r["role"], "text": r["content"]} for r in rows]
        except Exception as e:
            logger.warning(f"🧠 Could not fetch transcript for {session_id}: {e}")

    lines = []
    for m in messages:
        role = m.get("role", "unknown").upper()
        text = (m.get("text") or m.get("content") or "").strip()
        if text and role in ("USER", "ASSISTANT"):
            lines.append(f"{role}: {text[:500]}")  # cap per-message length

    return "\n\n".join(lines)


def _parse_memories(raw: str) -> list[dict]:
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()
    # Find JSON array
    m = re.search(r"\[.*\]", clean, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        logger.warning(f"🧠 Memory parse failed: {clean[:200]!r}")
        return []


# ── Memory retrieval ──────────────────────────────────────────────────────────

def _get_recent_memories(limit: int = 50) -> list[dict]:
    if not MEMORY_DB_PATH.exists():
        return []
    try:
        with _mem_conn() as conn:
            rows = conn.execute(
                """SELECT id, tier, content, importance, access_count
                   FROM memories
                   ORDER BY importance DESC, created_at DESC
                   LIMIT ?""",
                (limit,)
            ).fetchall()
            return [dict(r) for r in rows]
    except Exception:
        return []


def _touch_memories(ids: list[int]):
    """Update access_count and last_accessed for retrieved memories."""
    if not ids or not MEMORY_DB_PATH.exists():
        return
    now = datetime.now(timezone.utc).isoformat()
    try:
        with _mem_conn() as conn:
            conn.execute(
                f"""UPDATE memories
                    SET access_count = access_count + 1, last_accessed = ?
                    WHERE id IN ({','.join('?' * len(ids))})""",
                [now] + ids
            )
    except Exception:
        pass


def inject_into_system_prompt(system_prompt: str, query: str = "",
                              max_memories: int = 20,
                              min_score: float = 0.2) -> str:
    """
    Prepend relevant persistent memories to the system prompt.
    Uses vector similarity when a query is provided; falls back to
    importance-sorted top-N when no query is given (e.g. cold startup).
    Called on every new session load.
    """
    if not MEMORY_DB_PATH.exists():
        return system_prompt

    if query:
        memories = _search_memories(query, top_k=max_memories, min_score=min_score)
    else:
        memories = _get_recent_memories(limit=max_memories)

    if not memories:
        return system_prompt

    _touch_memories([m["id"] for m in memories])

    lines = [
        "## Persistent Memory (from past sessions)",
        "The following facts are KNOWN and TRUE. Use them to answer questions directly without searching or asking.",
        ""
    ]
    for m in memories:
        tier_tag = "◆" if m["tier"] == "semantic" else "○"
        lines.append(f"{tier_tag} {m['content']}")

    memory_block = "\n".join(lines) + "\n\n---\n\n"
    return memory_block + system_prompt


def _search_memories(query: str, top_k: int = 20,
                     min_score: float = 0.2) -> list[dict]:
    """Vector similarity search over memory embeddings, with optional reranking."""
    if not MEMORY_DB_PATH.exists():
        return []

    query_bytes = _embed(query)
    if query_bytes is None:
        return _get_recent_memories(limit=top_k)

    try:
        with _mem_conn() as conn:
            rows = conn.execute(
                "SELECT id, tier, content, embedding, importance, access_count "
                "FROM memories WHERE embedding IS NOT NULL"
            ).fetchall()
    except Exception:
        return _get_recent_memories(limit=top_k)

    if not rows:
        return _get_recent_memories(limit=top_k)

    scored = []
    for r in rows:
        score = _cosine(query_bytes, r["embedding"])
        if score >= min_score:
            scored.append({
                "id":           r["id"],
                "tier":         r["tier"],
                "content":      r["content"],
                "importance":   r["importance"],
                "access_count": r["access_count"],
                "score":        score,
                "text":         r["content"],  # _rerank expects a 'text' key
            })

    scored.sort(key=lambda x: (x["score"], x["importance"]), reverse=True)
    top_score = scored[0]["score"] if scored else 0.0
    logger.info(f"🧠 Memory search '{query[:40]}': {len(scored)} above threshold, top score={top_score:.3f}")

    # Memory candidates are already sorted by cosine similarity + importance.
    # We skip cross-encoder reranking here intentionally — the reranker runs
    # in the RAG MCP server subprocess and loading it in the client process too
    # doubles GPU/CPU overhead on every query. Cosine order is sufficient for
    # the small number of memory candidates (typically < 20).

    return scored[:top_k]


# ── Nightly consolidation / promotion ────────────────────────────────────────

async def run_nightly_promotion():
    """
    Promote frequently-accessed episodic memories to semantic.
    Runs once daily. Schedule via AgentScheduler or asyncio.create_task loop.
    """
    if not MEMORY_DB_PATH.exists():
        return
    threshold = int(os.getenv("MEMORY_PROMOTE_THRESHOLD", "3"))
    now = datetime.now(timezone.utc).isoformat()
    try:
        with _mem_conn() as conn:
            promoted = conn.execute(
                """UPDATE memories
                   SET tier = 'semantic', promoted_at = ?
                   WHERE tier = 'episodic'
                     AND access_count >= ?
                   RETURNING id, content""",
                (now, threshold)
            ).fetchall()
            if promoted:
                logger.info(f"🧠 Promoted {len(promoted)} episodic → semantic memories")
    except Exception as e:
        # RETURNING not available in SQLite < 3.35 — fallback
        try:
            with _mem_conn() as conn:
                rows = conn.execute(
                    """SELECT id FROM memories
                       WHERE tier = 'episodic' AND access_count >= ?""",
                    (threshold,)
                ).fetchall()
                ids = [r["id"] for r in rows]
                if ids:
                    conn.execute(
                        f"""UPDATE memories SET tier = 'semantic', promoted_at = ?
                            WHERE id IN ({','.join('?' * len(ids))})""",
                        [now] + ids
                    )
                    logger.info(f"🧠 Promoted {len(ids)} episodic → semantic memories")
        except Exception as e2:
            logger.warning(f"🧠 Nightly promotion failed: {e2}")


# ── Inactivity watcher ────────────────────────────────────────────────────────

# Module-level reference set by InactivityWatcher.__init__ so
# handle_memory_command can access llm_fn and session_manager
_active_watcher: "InactivityWatcher | None" = None


class InactivityWatcher:
    """
    Watches per-session last-activity timestamps.
    After INACTIVITY_SECONDS of silence, fires consolidation.

    Usage:
        watcher = InactivityWatcher(llm_fn, session_manager)
        asyncio.create_task(watcher.run())

        # Call on every user/assistant message:
        watcher.touch(session_id)
    """

    def __init__(self, llm_fn, session_manager=None):
        global _active_watcher
        self._llm_fn = llm_fn
        self._session_manager = session_manager
        self._last_activity: dict[str, float] = {}
        self._consolidating: set[str] = set()
        _active_watcher = self

    def touch(self, session_id: str):
        import time
        self._last_activity[str(session_id)] = time.monotonic()

    def forget(self, session_id: str):
        """Call when a session is deleted."""
        self._last_activity.pop(str(session_id), None)
        self._consolidating.discard(str(session_id))

    async def consolidate_now(self, session_id: str):
        """Explicitly trigger consolidation (e.g. on session close)."""
        sid = str(session_id)
        if sid in self._consolidating:
            return
        self._consolidating.add(sid)
        try:
            await consolidate(sid, self._llm_fn, self._session_manager)
            await consolidate_decisions(sid, self._llm_fn, self._session_manager)
        finally:
            self._consolidating.discard(sid)

    async def run(self):
        import time
        logger.info("🧠 InactivityWatcher started")
        while True:
            await asyncio.sleep(60)  # check every minute
            now = time.monotonic()
            stale = [
                sid for sid, last in list(self._last_activity.items())
                if (now - last) >= INACTIVITY_SECONDS
                and sid not in self._consolidating
            ]
            for sid in stale:
                logger.info(f"🧠 Inactivity consolidation triggered for session {sid}")
                self._last_activity.pop(sid, None)
                asyncio.create_task(self.consolidate_now(sid))


# ── :memory colon command ─────────────────────────────────────────────────────

def handle_memory_command(raw: str, llm_fn=None, session_manager=None) -> str:
    """
    :memory                       — list all memories
    :memory semantic              — list only semantic (permanent) memories
    :memory episodic              — list only episodic memories
    :memory forget <id>           — delete a memory by ID
    :memory clear                 — delete all episodic memories
    :memory clear session <id>    — delete all memories from a specific session
    :memory consolidate           — extract memories from current session now
    :memory consolidate <id>      — extract memories from a specific session now
    """
    _ensure_db()
    tokens = raw.strip().split(None, 3)
    verb = tokens[1].lower() if len(tokens) > 1 else "list"

    if verb in ("list", "all"):
        return _format_memory_list()
    elif verb == "semantic":
        return _format_memory_list(tier="semantic")
    elif verb == "episodic":
        return _format_memory_list(tier="episodic")
    elif verb == "forget" and len(tokens) > 2:
        return _forget_memory(tokens[2])
    elif verb == "clear":
        if len(tokens) > 2 and tokens[2].lower() == "session":
            session_id = tokens[3].strip() if len(tokens) > 3 else ""
            if not session_id:
                return "Usage: :memory clear session <session_id>"
            return _clear_session_memories(session_id)
        return _clear_episodic()
    elif verb == "consolidate":
        session_id = tokens[2].strip() if len(tokens) > 2 else None
        return _consolidate_now(session_id, llm_fn, session_manager)
    elif verb == "dedup":
        return _dedup_memories()
    elif verb == "add":
        content = raw.strip().split(None, 2)[2] if len(tokens) > 2 else ""
        return _add_memory(content)
    else:
        return (
            "Unknown :memory subcommand. Available:\n"
            "  :memory                     — list all\n"
            "  :memory semantic            — permanent memories only\n"
            "  :memory episodic            — session-derived memories\n"
            "  :memory forget <id>         — delete one memory\n"
            "  :memory clear               — delete all episodic memories\n"
            "  :memory clear session <id>  — delete memories from one session\n"
            "  :memory consolidate <id>    — extract memories from a session now\n"
            "  :memory dedup               — remove duplicate memories\n"
            "  :memory add <fact>          — manually add a memory\n"
        )


def _format_memory_list(tier: Optional[str] = None) -> str:
    if not MEMORY_DB_PATH.exists():
        return "No memory database yet. Memories are created after sessions end."

    with _mem_conn() as conn:
        query = "SELECT * FROM memories"
        params = []
        if tier:
            query += " WHERE tier = ?"
            params.append(tier)
        query += " ORDER BY importance DESC, created_at DESC"
        rows = conn.execute(query, params).fetchall()

    if not rows:
        label = f"{tier} " if tier else ""
        return f"No {label}memories stored yet."

    tier_label = f" ({tier})" if tier else ""
    lines = [f"PERSISTENT MEMORY{tier_label}\n" + "─" * 48]
    for r in rows:
        tier_tag = "◆ semantic" if r["tier"] == "semantic" else "○ episodic"
        accessed = f"accessed {r['access_count']}x" if r["access_count"] else "never accessed"
        lines.append(
            f"[{r['id']}] {r['content']}\n"
            f"    {tier_tag}  |  importance: {r['importance']:.1f}  |  {accessed}"
        )

    lines.append("─" * 48)
    lines.append(f"{len(rows)} memory/memories total")
    return "\n".join(lines)


def _forget_memory(id_str: str) -> str:
    try:
        mem_id = int(id_str.strip())
    except ValueError:
        return f"Invalid memory ID: {id_str!r}"

    if not MEMORY_DB_PATH.exists():
        return "No memory database found."

    with _mem_conn() as conn:
        row = conn.execute("SELECT content FROM memories WHERE id = ?", (mem_id,)).fetchone()
        if not row:
            return f"No memory with ID {mem_id}."
        conn.execute("DELETE FROM memories WHERE id = ?", (mem_id,))
        return f"Memory [{mem_id}] deleted: \"{row['content'][:80]}\""


def _clear_episodic() -> str:
    if not MEMORY_DB_PATH.exists():
        return "No memory database found."
    with _mem_conn() as conn:
        n = conn.execute("SELECT COUNT(*) FROM memories WHERE tier = 'episodic'").fetchone()[0]
        conn.execute("DELETE FROM memories WHERE tier = 'episodic'")
    return f"Cleared {n} episodic memory/memories."


def _clear_session_memories(session_id: str) -> str:
    """Delete all memories extracted from a specific session."""
    if not MEMORY_DB_PATH.exists():
        return "No memory database found."
    with _mem_conn() as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM memories WHERE source_session = ?", (session_id,)
        ).fetchone()[0]
        if n == 0:
            return f"No memories found for session {session_id}."
        conn.execute("DELETE FROM memories WHERE source_session = ?", (session_id,))
    return f"Cleared {n} memory/memories from session {session_id}."


def _dedup_memories() -> str:
    """Delete duplicate memories, keeping the highest importance copy of each."""
    if not MEMORY_DB_PATH.exists():
        return "No memory database found."
    with _mem_conn() as conn:
        rows = conn.execute(
            "SELECT id, content, importance FROM memories ORDER BY importance DESC, id ASC"
        ).fetchall()
        seen = {}
        to_delete = []
        for r in rows:
            key = r["content"].lower().strip()
            if key in seen:
                to_delete.append(r["id"])
            else:
                seen[key] = r["id"]
        if not to_delete:
            return "No duplicates found."
        conn.execute(
            f"DELETE FROM memories WHERE id IN ({','.join('?' * len(to_delete))})",
            to_delete
        )
    return f"Removed {len(to_delete)} duplicate memory/memories."


def _add_memory(content: str) -> str:
    """Manually insert a single memory as semantic (permanent) tier."""
    if not content or len(content.strip()) < 3:
        return "Usage: :memory add <fact to remember>"
    _ensure_db()
    content = content.strip()
    now = datetime.now(timezone.utc).isoformat()
    embedding = _embed(content)
    with _mem_conn() as conn:
        existing = conn.execute(
            "SELECT id FROM memories WHERE LOWER(content) = ?",
            (content.lower(),)
        ).fetchone()
        if existing:
            return f"Memory already exists: \"{content}\""
        conn.execute(
            """INSERT INTO memories
               (tier, content, embedding, source_session, importance, created_at)
               VALUES ('semantic', ?, ?, 'manual', 1.0, ?)""",
            (content, embedding, now)
        )
    return f"Memory added: \"{content}\""


def _consolidate_now(session_id: Optional[str], llm_fn, session_manager=None) -> str:
    """
    Synchronous wrapper around consolidate() for use in the command handler.
    Falls back to _active_watcher's llm_fn if none provided.
    """
    if llm_fn is None and _active_watcher is not None:
        llm_fn = _active_watcher._llm_fn
        if session_manager is None:
            session_manager = _active_watcher._session_manager

    if llm_fn is None:
        return (
            "Memory consolidation requires the LLM to be available.\n"
            "Try again after startup completes."
        )

    if not session_id:
        return (
            "Please provide a session ID: :memory consolidate <id>\n"
            "Use :sessions to list available sessions."
        )

    # Always clear consolidated_at for manual runs — user explicitly wants a re-run
    if SESSIONS_DB_PATH.exists():
        try:
            with _sess_conn() as conn:
                conn.execute(
                    "UPDATE sessions SET consolidated_at = NULL WHERE id = ?",
                    (session_id,)
                )
        except Exception as e:
            logger.warning(f"🧠 Could not clear consolidated_at for session {session_id}: {e}")

    # Run async consolidate in the running event loop
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            async def _run_and_log():
                try:
                    count = await consolidate(session_id, llm_fn, session_manager)
                    logger.info(f"🧠 Manual consolidation complete: {count} memories written for session {session_id}")
                except Exception as e:
                    logger.error(f"🧠 Manual consolidation failed for session {session_id}: {e}", exc_info=True)
            asyncio.ensure_future(_run_and_log())
            return (
                f"Consolidation started for session {session_id}.\n"
                f"Run :memory in a few seconds to see the results."
            )
        else:
            count = loop.run_until_complete(consolidate(session_id, llm_fn, session_manager))
            if count == 0:
                return (
                    f"No new memories extracted from session {session_id}.\n"
                    "The session may already be consolidated, too short, or contain no memorable content."
                )
            return f"Extracted {count} memory/memories from session {session_id}. Run :memory to see them."
    except Exception as e:
        return f"Consolidation failed: {e}"