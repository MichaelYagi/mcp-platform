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

# Connection pool
_db_connection = None


def ensure_data_dir():
    """Ensure the data directory exists"""
    RAG_DB_FILE.parent.mkdir(parents=True, exist_ok=True)


def get_connection() -> sqlite3.Connection:
    """Get or create database connection with optimizations"""
    global _db_connection

    if _db_connection is None:
        ensure_data_dir()
        _db_connection = sqlite3.connect(str(RAG_DB_FILE), check_same_thread=False)
        _db_connection.row_factory = sqlite3.Row

        # Performance optimizations
        _db_connection.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
        _db_connection.execute("PRAGMA synchronous=NORMAL")  # Faster writes
        _db_connection.execute("PRAGMA cache_size=-64000")  # 64MB cache
        _db_connection.execute("PRAGMA temp_store=MEMORY")  # In-memory temp tables

        # Create / migrate tables
        _initialize_database(_db_connection)

    return _db_connection


def _initialize_database(conn: sqlite3.Connection):
    """Create tables and indexes if they don't exist, migrate schema if needed."""
    cursor = conn.cursor()

    # Main documents table (original schema)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            embedding BLOB NOT NULL,
            source TEXT,
            length INTEGER,
            word_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ── Schema migration: add session_id column if it doesn't exist ──────────
    cursor.execute("PRAGMA table_info(documents)")
    existing_columns = {row[1] for row in cursor.fetchall()}

    if "session_id" not in existing_columns:
        logger.info("🔄 Migrating RAG schema: adding session_id column")
        cursor.execute("ALTER TABLE documents ADD COLUMN session_id TEXT")
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_documents_session_id
            ON documents(session_id)
        """)
        logger.info("✅ session_id column added to documents table")

    # ── Standard indexes ──────────────────────────────────────────────────────
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_documents_source 
        ON documents(source)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_documents_created 
        ON documents(created_at)
    """)

    # Composite index: session queries filter on session_id then order by created_at
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_documents_session_created
        ON documents(session_id, created_at)
    """)

    conn.commit()
    logger.debug("✅ Database initialized")


def load_rag_db() -> List[Dict[str, Any]]:
    """
    Load all NON-conversation documents from the RAG database.
    Conversation turns (session_id IS NOT NULL) are excluded — they are
    accessed via conversation_rag.retrieve_context() instead.
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, text, embedding, source, length, word_count
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

            doc = {
                "id": row['id'],
                "text": row['text'],
                "embedding": embedding,
                "metadata": {
                    "source": row['source'],
                    "length": row['length'],
                    "word_count": row['word_count']
                }
            }
            documents.append(doc)

        logger.debug(f"📂 Loaded {len(documents)} documents from database")
        return documents

    except Exception as e:
        logger.error(f"❌ Error loading RAG database: {e}")
        return []


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
                INSERT OR REPLACE INTO documents 
                (id, text, embedding, source, session_id, length, word_count)
                VALUES (?, ?, ?, ?, NULL, ?, ?)
            """, (
                doc['id'],
                doc['text'],
                embedding_blob,
                metadata.get('source'),
                metadata.get('length'),
                metadata.get('word_count')
            ))

        cursor.execute("COMMIT")
        logger.debug(f"💾 Saved {len(db)} documents to database")

    except Exception as e:
        logger.error(f"❌ Error saving RAG database: {e}")
        try:
            cursor.execute("ROLLBACK")
        except:
            pass
        raise


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
                doc['text'],
                embedding_blob,
                metadata.get('source'),
                None,  # session_id — external docs never have one
                metadata.get('length'),
                metadata.get('word_count')
            ))

        cursor.execute("BEGIN TRANSACTION")
        cursor.executemany("""
            INSERT OR REPLACE INTO documents 
            (id, text, embedding, source, session_id, length, word_count)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, data)
        cursor.execute("COMMIT")

        logger.debug(f"💾 Batch saved {len(documents)} documents")

    except Exception as e:
        logger.error(f"❌ Error in batch save: {e}")
        try:
            cursor.execute("ROLLBACK")
        except:
            pass
        raise


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


def get_documents_by_source(source: str) -> List[Dict[str, Any]]:
    """Get all documents from a specific source."""
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, text, embedding, source, length, word_count
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

            doc = {
                "id": row['id'],
                "text": row['text'],
                "embedding": embedding,
                "metadata": {
                    "source": row['source'],
                    "length": row['length'],
                    "word_count": row['word_count']
                }
            }
            documents.append(doc)

        return documents

    except Exception as e:
        logger.error(f"❌ Error getting documents by source: {e}")
        return []


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    try:
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)

        dot_product = np.dot(vec1_np, vec2_np)
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(max(0.0, min(1.0, dot_product / (norm1 * norm2))))

    except Exception as e:
        logger.error(f"❌ Error calculating cosine similarity: {e}")
        return 0.0


def clear_rag_db():
    """Clear only external/document RAG entries (preserves conversation turns)"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM documents WHERE session_id IS NULL")
        conn.commit()
        cursor.execute("VACUUM")
        logger.info("🗑️  Cleared document RAG database (conversation turns preserved)")
    except Exception as e:
        logger.error(f"❌ Error clearing database: {e}")


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

        cursor.execute("SELECT SUM(word_count) FROM documents WHERE session_id IS NULL")
        total_words = cursor.fetchone()[0] or 0

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
            "total_words": total_words,
            "unique_sources": unique_sources,
            "database_size_mb": round(db_size_mb, 2),
            "database_file": str(RAG_DB_FILE)
        }

    except Exception as e:
        logger.error(f"❌ Error getting database stats: {e}")
        return {}