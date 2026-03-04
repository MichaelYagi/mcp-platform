"""
RAG Utilities - SQLite Backend
High-performance database with proper indexing and transactions
"""

import json
import logging
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

logger = logging.getLogger("mcp_server")

# Database file location
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAG_DB_FILE = PROJECT_ROOT / "data" / "rag_database.db"

import threading
_init_lock = threading.Lock()
_db_initialized = False


def ensure_data_dir():
    """Ensure the data directory exists"""
    RAG_DB_FILE.parent.mkdir(parents=True, exist_ok=True)


def get_connection() -> sqlite3.Connection:
    """
    Open a new SQLite connection for the current call.
    Callers are responsible for calling conn.close() when done.
    """
    global _db_initialized
    ensure_data_dir()

    conn = sqlite3.connect(str(RAG_DB_FILE))
    conn.row_factory = sqlite3.Row

    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")
    conn.execute("PRAGMA temp_store=MEMORY")

    if not _db_initialized:
        with _init_lock:
            if not _db_initialized:
                _initialize_database(conn)
                _db_initialized = True

    return conn


def _initialize_database(conn: sqlite3.Connection):
    """Create tables and indexes if they don't exist, migrate schema if needed."""
    cursor = conn.cursor()

    # Main documents table.
    # chunk_id   — FK into sessions.db chunks.id for document RAG rows (session_id IS NULL)
    # message_id — FK into sessions.db messages.id for conversation turn rows (session_id IS NOT NULL)
    # Text is never stored here; always retrieved from sessions.db via chunk_id or message_id.
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id         TEXT PRIMARY KEY,
            embedding  BLOB NOT NULL,
            source     TEXT,
            chunk_id   INTEGER,
            message_id INTEGER,
            session_id TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Migrate existing rows: add chunk_id / message_id columns if absent
    cursor.execute("PRAGMA table_info(documents)")
    existing_cols = {row[1] for row in cursor.fetchall()}
    for col, typedef in [("chunk_id", "INTEGER"), ("message_id", "INTEGER")]:
        if col not in existing_cols:
            cursor.execute(f"ALTER TABLE documents ADD COLUMN {col} {typedef}")

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_documents_source 
        ON documents(source)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_documents_created 
        ON documents(created_at)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_documents_session_created
        ON documents(session_id, created_at)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_documents_chunk_id
        ON documents(chunk_id)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_documents_message_id
        ON documents(message_id)
    """)

    conn.commit()
    logger.debug("✅ RAG database initialized")


def load_rag_db() -> List[Dict[str, Any]]:
    """
    Load all non-conversation documents from the RAG database.
    Conversation turns (session_id IS NOT NULL) are excluded — accessed
    via conversation_rag.retrieve_context() instead.

    Returns dicts with: id, embedding, source, chunk_id
    Text is NOT stored here — callers retrieve it from sessions.db via chunk_id.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, embedding, source, chunk_id
            FROM documents
            WHERE session_id IS NULL
            ORDER BY created_at
        """)

        documents = []
        for row in cursor.fetchall():
            embedding_data = row['embedding']
            if isinstance(embedding_data, bytes):
                embedding = np.frombuffer(embedding_data, dtype=np.float32).tolist()
            else:
                embedding = json.loads(embedding_data)

            documents.append({
                "id":       row['id'],
                "embedding": embedding,
                "chunk_id": row['chunk_id'],
                "metadata": {"source": row['source']},
            })

        logger.debug(f"📂 Loaded {len(documents)} documents from RAG database")
        return documents

    except Exception as e:
        logger.error(f"❌ Error loading RAG database: {e}")
        return []
    finally:
        conn.close()


def save_rag_db(db: List[Dict[str, Any]]):
    """
    Save documents to the RAG database in a single transaction.
    Only saves non-conversation documents (session_id stays NULL).
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("BEGIN TRANSACTION")

        for doc in db:
            embedding_blob = json.dumps(doc['embedding'])
            metadata = doc.get('metadata', {})
            cursor.execute("""
                INSERT OR REPLACE INTO documents (id, embedding, source, chunk_id, session_id)
                VALUES (?, ?, ?, ?, NULL)
            """, (
                doc['id'],
                embedding_blob,
                metadata.get('source'),
                doc.get('chunk_id'),
            ))

        cursor.execute("COMMIT")
        logger.debug(f"💾 Saved {len(db)} documents to RAG database")

    except Exception as e:
        logger.error(f"❌ Error saving RAG database: {e}")
        try:
            cursor.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        conn.close()


def save_rag_db_batch(documents: List[Dict[str, Any]]):
    """Efficiently save a batch of documents using bulk insert."""
    try:
        conn = get_connection()
        cursor = conn.cursor()

        data = []
        for doc in documents:
            embedding_blob = json.dumps(doc['embedding'])
            metadata = doc.get('metadata', {})
            data.append((
                doc['id'],
                embedding_blob,
                metadata.get('source'),
                doc.get('chunk_id'),
                None,  # session_id — external docs never have one
            ))

        cursor.execute("BEGIN TRANSACTION")
        cursor.executemany("""
            INSERT OR REPLACE INTO documents (id, embedding, source, chunk_id, session_id)
            VALUES (?, ?, ?, ?, ?)
        """, data)
        cursor.execute("COMMIT")
        logger.debug(f"💾 Batch saved {len(documents)} documents")

    except Exception as e:
        logger.error(f"❌ Error in batch save: {e}")
        try:
            cursor.execute("ROLLBACK")
        except Exception:
            pass
        raise
    finally:
        conn.close()


def get_document_count() -> int:
    """Get total number of non-conversation documents in database"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM documents WHERE session_id IS NULL")
        return cursor.fetchone()[0]
    except Exception as e:
        logger.error(f"❌ Error getting document count: {e}")
        return 0
    finally:
        conn.close()


def get_documents_by_source(source: str) -> List[Dict[str, Any]]:
    """Get all documents from a specific source."""
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, embedding, source, chunk_id
            FROM documents
            WHERE source = ? AND session_id IS NULL
            ORDER BY created_at
        """, (source,))

        documents = []
        for row in cursor.fetchall():
            embedding_data = row['embedding']
            if isinstance(embedding_data, bytes):
                embedding = np.frombuffer(embedding_data, dtype=np.float32).tolist()
            else:
                embedding = json.loads(embedding_data)

            documents.append({
                "id":        row['id'],
                "embedding": embedding,
                "chunk_id":  row['chunk_id'],
                "metadata":  {"source": row['source']},
            })

        return documents

    except Exception as e:
        logger.error(f"❌ Error getting documents by source: {e}")
        return []
    finally:
        conn.close()


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    try:
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        dot = np.dot(v1, v2)
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(max(0.0, min(1.0, dot / (n1 * n2))))
    except Exception as e:
        logger.error(f"❌ Error calculating cosine similarity: {e}")
        return 0.0


def clear_rag_db():
    """
    Clear only external/document RAG entries (preserves conversation turns).
    Also clears the corresponding chunk text rows from sessions.db.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM documents WHERE session_id IS NULL")
        conn.commit()
        cursor.execute("VACUUM")
        logger.info("🗑️  Cleared document RAG database (conversation turns preserved)")
    except Exception as e:
        logger.error(f"❌ Error clearing database: {e}")
    finally:
        conn.close()

    # Purge chunk text from sessions.db
    try:
        from client.session_manager import SessionManager
        deleted = SessionManager().delete_all_chunks()
        logger.info(f"🗑️  Cleared {deleted} chunk text rows from sessions.db")
    except Exception as e:
        logger.warning(f"⚠️ Could not clear chunk text from sessions.db: {e}")


def migrate_from_json():
    """Migrate from old JSON database to SQLite."""
    old_json_file = PROJECT_ROOT / "data" / "rag_database.json"

    if not old_json_file.exists():
        logger.info("📂 No JSON database to migrate")
        return

    logger.info("🔄 Starting migration from JSON to SQLite...")

    try:
        with open(old_json_file, 'r', encoding='utf-8') as f:
            old_db = json.load(f)

        logger.info(f"📂 Loaded {len(old_db)} documents from JSON")
        save_rag_db_batch(old_db)
        logger.info(f"✅ Migration complete: {len(old_db)} documents")

        backup_file = old_json_file.with_suffix('.json.backup')
        old_json_file.rename(backup_file)
        logger.info(f"📦 Old JSON backed up to: {backup_file}")

    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise


def get_database_stats() -> Dict[str, Any]:
    """Get statistics about the database"""
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM documents WHERE session_id IS NULL")
        total_docs = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM documents WHERE session_id IS NOT NULL")
        conversation_turns = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT source) FROM documents WHERE session_id IS NULL")
        unique_sources = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT session_id) FROM documents WHERE session_id IS NOT NULL")
        unique_sessions = cursor.fetchone()[0]

        db_size_bytes = RAG_DB_FILE.stat().st_size if RAG_DB_FILE.exists() else 0
        db_size_mb = db_size_bytes / (1024 * 1024)

        return {
            "total_documents": total_docs,
            "conversation_turns_indexed": conversation_turns,
            "unique_sessions_indexed": unique_sessions,
            "unique_sources": unique_sources,
            "database_size_mb": round(db_size_mb, 2),
            "database_file": str(RAG_DB_FILE)
        }

    except Exception as e:
        logger.error(f"❌ Error getting database stats: {e}")
        return {}
    finally:
        conn.close()


def delete_conversation_session(session_id: int):
    """
    Delete all RAG conversation turns for a specific session.
    Called by :clear session <id> to keep rag_database.db in sync with sessions.db.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM documents WHERE session_id = ?", (str(session_id),))
        deleted = cursor.rowcount
        conn.commit()
        logger.info(f"🗑️  Deleted {deleted} RAG turns for session {session_id}")
    except Exception as e:
        logger.error(f"❌ Error deleting RAG turns for session {session_id}: {e}")
    finally:
        conn.close()


def clear_all_conversation_turns():
    """
    Delete ALL conversation turns from the RAG database.
    Called by :clear sessions to keep rag_database.db in sync with sessions.db.
    Ingested document rows (session_id IS NULL) are preserved.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM documents WHERE session_id IS NOT NULL")
        deleted = cursor.rowcount
        conn.commit()
        logger.info(f"🗑️  Cleared {deleted} conversation turns from RAG database")
    except Exception as e:
        logger.error(f"❌ Error clearing conversation turns from RAG database: {e}")
    finally:
        conn.close()