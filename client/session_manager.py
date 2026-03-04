import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any


class SessionManager:
    def __init__(self, db_path: str = None):
        if db_path is None:
            # Default to mcp_a2a/data/sessions.db
            db_path = os.path.join(os.path.dirname(__file__), "..", "data", "sessions.db")
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_db()

    def init_db(self):
        """Initialize the database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Messages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                model TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
            )
        ''')

        # Chunks table — stores ingested document text chunks for RAG lookup.
        # The RAG DB (rag_database.db) stores embeddings + chunk_id FK pointing here.
        # Text never lives in the RAG DB; this is the single source of truth for chunk text.
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                text TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_chunks_source
            ON chunks(source)
        ''')

        # Check if model column exists (for migration from old schema)
        cursor.execute("PRAGMA table_info(messages)")
        columns = [col[1] for col in cursor.fetchall()]
        if 'model' not in columns:
            cursor.execute('ALTER TABLE messages ADD COLUMN model TEXT')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_messages_session 
            ON messages(session_id)
        ''')

        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_messages_created 
            ON messages(created_at)
        ''')

        conn.commit()
        conn.close()

    def create_session(self, name: str = None) -> int:
        """Create a new session and return its ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('INSERT INTO sessions (name) VALUES (?)', (name,))
        session_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return session_id

    def update_session_name(self, session_id: int, name: str):
        """Update the session name"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE sessions SET name = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?
        ''', (name, session_id))
        conn.commit()
        conn.close()

    def add_message(self, session_id: int, role: str, content: str, max_history: int = 30, model: str = None) -> int:
        """
        Add a message to a session and enforce the message limit.

        Returns:
            The inserted message ID (messages.id). The caller should pass this
            to conversation_rag.store_turn_async so the RAG embedding can be
            correlated back to this exact row for reliable text lookup.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Ensure session exists before adding message
        cursor.execute('SELECT id FROM sessions WHERE id = ?', (session_id,))
        if not cursor.fetchone():
            default_name = content[:50] if role == 'user' else 'New Chat'
            cursor.execute('''
                INSERT INTO sessions (id, name, created_at, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ''', (session_id, default_name))
            conn.commit()

        cursor.execute('''
            INSERT INTO messages (session_id, role, content, model) VALUES (?, ?, ?, ?)
        ''', (session_id, role, content, model))

        message_id = cursor.lastrowid

        cursor.execute('''
            UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?
        ''', (session_id,))

        # Enforce message limit
        cursor.execute('SELECT COUNT(*) FROM messages WHERE session_id = ?', (session_id,))
        count = cursor.fetchone()[0]

        if count > max_history:
            to_delete = count - max_history
            cursor.execute('''
                DELETE FROM messages 
                WHERE id IN (
                    SELECT id FROM messages WHERE session_id = ?
                    ORDER BY created_at ASC LIMIT ?
                )
            ''', (session_id, to_delete))

        conn.commit()
        conn.close()
        return message_id

    def get_session_messages(self, session_id: int) -> List[Dict]:
        """Get all messages for a session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT role, content, model, created_at 
            FROM messages WHERE session_id = ? ORDER BY created_at ASC
        ''', (session_id,))
        messages = [
            {'role': r[0], 'text': r[1], 'model': r[2], 'timestamp': r[3]}
            for r in cursor.fetchall()
        ]
        conn.close()
        return messages

    def get_message_by_id(self, message_id: int) -> Optional[Dict]:
        """
        Look up a single message by its primary key.
        Used by conversation_rag.retrieve_context to fetch turn text
        directly by ID, replacing the old fragile positional index approach.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, session_id, role, content, model, created_at
            FROM messages WHERE id = ?
        ''', (message_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return {
                'id': row[0], 'session_id': row[1], 'role': row[2],
                'text': row[3], 'model': row[4], 'timestamp': row[5],
            }
        return None

    # ------------------------------------------------------------------ #
    # Chunk storage — ingested document text for RAG                       #
    # ------------------------------------------------------------------ #

    def store_chunk(self, source: str, text: str) -> int:
        """
        Persist a document chunk and return its ID.
        Called by rag_add before embedding so the chunk_id can be stored
        alongside the embedding in rag_database.db.

        Args:
            source: Source identifier (URL, file path, etc.)
            text:   Raw chunk text

        Returns:
            Integer chunk ID (chunks.id)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('INSERT INTO chunks (source, text) VALUES (?, ?)', (source, text))
        chunk_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return chunk_id

    def get_chunk(self, chunk_id: int) -> Optional[str]:
        """
        Retrieve chunk text by primary key.
        Used by rag_search to assemble (query, passage) pairs for reranking
        and to populate result text in search responses.

        Args:
            chunk_id: chunks.id value stored in the RAG DB

        Returns:
            Chunk text, or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT text FROM chunks WHERE id = ?', (chunk_id,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None

    def get_chunks_by_source(self, source: str) -> List[Dict]:
        """Get all chunks for a given source."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, text, created_at FROM chunks WHERE source = ? ORDER BY id ASC
        ''', (source,))
        chunks = [{'id': r[0], 'text': r[1], 'created_at': r[2]} for r in cursor.fetchall()]
        conn.close()
        return chunks

    def delete_chunks_by_source(self, source: str) -> int:
        """
        Delete all chunks for a given source.
        Call when re-ingesting a URL to avoid duplicate chunks.

        Returns:
            Number of rows deleted
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM chunks WHERE source = ?', (source,))
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        return deleted

    def delete_all_chunks(self) -> int:
        """Delete all document chunks. Called by clear_rag_db flows."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM chunks')
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        return deleted

    # ------------------------------------------------------------------ #
    # Session management                                                    #
    # ------------------------------------------------------------------ #

    def get_all_sessions(self) -> List[Dict]:
        """Get all sessions ordered by most recent"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, name, created_at, updated_at FROM sessions ORDER BY updated_at DESC
        ''')
        sessions = [
            {'id': r[0], 'name': r[1] or 'Untitled Session', 'created_at': r[2], 'updated_at': r[3]}
            for r in cursor.fetchall()
        ]
        conn.close()
        return sessions

    def delete_session(self, session_id: int):
        """Delete a session, all its messages, and its RAG conversation turns"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM sessions WHERE id = ?', (session_id,))
        cursor.execute('DELETE FROM messages WHERE session_id = ?', (session_id,))
        conn.commit()
        conn.close()

        try:
            from tools.rag.rag_utils import delete_conversation_session
            delete_conversation_session(session_id)
        except Exception as e:
            import logging
            logging.getLogger("session_manager").warning(
                f"⚠️ Could not clear RAG turns for session {session_id}: {e}"
            )

    def delete_all_sessions(self):
        """Delete all sessions, all messages, and all RAG conversation turns"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM sessions')
        cursor.execute('DELETE FROM messages')
        conn.commit()
        conn.close()

        try:
            from tools.rag.rag_utils import clear_all_conversation_turns
            clear_all_conversation_turns()
        except Exception as e:
            import logging
            logging.getLogger("session_manager").warning(
                f"⚠️ Could not clear RAG conversation turns: {e}"
            )

    def get_sessions(self) -> list[Dict]:
        """Get all session details"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT id, name, created_at, updated_at FROM sessions')
        sessions = [
            {'id': r[0], 'name': r[1] or 'Untitled Session', 'created_at': r[2], 'updated_at': r[3]}
            for r in cursor.fetchall()
        ]
        conn.close()
        return sessions

    def get_session(self, session_id: int) -> Optional[Dict]:
        """Get session details"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, name, created_at, updated_at FROM sessions WHERE id = ?
        ''', (session_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return {'id': row[0], 'name': row[1], 'created_at': row[2], 'updated_at': row[3]}
        return None

    def get_user_session_count(self) -> int:
        """Get total number of sessions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM sessions')
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def get_recent_session_topics(self, limit: int = 5) -> List[Dict]:
        """Get topics from recent sessions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT s.id, s.name, s.created_at,
                   (SELECT content FROM messages
                    WHERE session_id = s.id AND role = 'user'
                    ORDER BY created_at ASC LIMIT 1) as first_message
            FROM sessions s
            ORDER BY s.updated_at DESC LIMIT ?
        ''', (limit,))
        topics = [
            {'session_id': r[0], 'name': r[1], 'created_at': r[2], 'first_message': r[3]}
            for r in cursor.fetchall()
        ]
        conn.close()
        return topics

    def is_first_session(self) -> bool:
        """Check if this is the very first session"""
        return self.get_user_session_count() <= 1