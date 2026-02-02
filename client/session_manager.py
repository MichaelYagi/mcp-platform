import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

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

        # Check if model column exists (for migration from old schema)
        cursor.execute("PRAGMA table_info(messages)")
        columns = [col[1] for col in cursor.fetchall()]
        if 'model' not in columns:
            cursor.execute('ALTER TABLE messages ADD COLUMN model TEXT')

        # Create indexes
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

        cursor.execute('''
            INSERT INTO sessions (name) VALUES (?)
        ''', (name,))

        session_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return session_id

    def update_session_name(self, session_id: int, name: str):
        """Update the session name"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            UPDATE sessions 
            SET name = ?, updated_at = CURRENT_TIMESTAMP 
            WHERE id = ?
        ''', (name, session_id))

        conn.commit()
        conn.close()

    def add_message(self, session_id: int, role: str, content: str, max_history: int = 30, model: str = None):
        """Add a message to a session and enforce message limit"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # CRITICAL FIX: Ensure session exists before adding message
        cursor.execute('SELECT id FROM sessions WHERE id = ?', (session_id,))
        session_exists = cursor.fetchone()

        if not session_exists:
            # Session doesn't exist - create it with a name based on first message
            default_name = content[:50] if role == 'user' else 'New Chat'
            cursor.execute('''
                           INSERT INTO sessions (id, name, created_at, updated_at)
                           VALUES (?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                           ''', (session_id, default_name))
            conn.commit()

        # Insert new message with model name
        cursor.execute('''
            INSERT INTO messages (session_id, role, content, model) 
            VALUES (?, ?, ?, ?)
        ''', (session_id, role, content, model))

        # Update session timestamp
        cursor.execute('''
            UPDATE sessions 
            SET updated_at = CURRENT_TIMESTAMP 
            WHERE id = ?
        ''', (session_id,))

        # Enforce message limit
        cursor.execute('''
            SELECT COUNT(*) FROM messages WHERE session_id = ?
        ''', (session_id,))

        count = cursor.fetchone()[0]

        if count > max_history:
            # Delete oldest messages
            to_delete = count - max_history
            cursor.execute('''
                DELETE FROM messages 
                WHERE id IN (
                    SELECT id FROM messages
                    WHERE session_id = ?
                    ORDER BY created_at ASC 
                    LIMIT ?
                )
            ''', (session_id, to_delete))

        conn.commit()
        conn.close()

    def get_session_messages(self, session_id: int) -> List[Dict]:
        """Get all messages for a session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT role, content, model, created_at 
            FROM messages 
            WHERE session_id = ? 
            ORDER BY created_at ASC
        ''', (session_id,))

        messages = []
        for row in cursor.fetchall():
            messages.append({
                'role': row[0],
                'text': row[1],
                'model': row[2],  # ← ADD THIS
                'timestamp': row[3]
            })

        conn.close()
        return messages

    def get_all_sessions(self) -> List[Dict]:
        """Get all sessions ordered by most recent"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, name, created_at, updated_at 
            FROM sessions 
            ORDER BY updated_at DESC
        ''')

        sessions = []
        for row in cursor.fetchall():
            sessions.append({
                'id': row[0],
                'name': row[1] or 'Untitled Session',
                'created_at': row[2],
                'updated_at': row[3]
            })

        conn.close()
        return sessions

    def delete_session(self, session_id: int):
        """Delete a session and all its messages"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('DELETE FROM sessions WHERE id = ?', (session_id,))

        cursor.execute('DELETE FROM messages WHERE session_id = ?', (session_id,))

        conn.commit()
        conn.close()

    def get_session(self, session_id: int) -> Optional[Dict]:
        """Get session details"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, name, created_at, updated_at 
            FROM sessions 
            WHERE id = ?
        ''', (session_id,))

        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                'id': row[0],
                'name': row[1],
                'created_at': row[2],
                'updated_at': row[3]
            }
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
                       SELECT s.id,
                              s.name,
                              s.created_at,
                              (SELECT content
                               FROM messages
                               WHERE session_id = s.id
                                 AND role = 'user'
                               ORDER BY created_at ASC LIMIT 1) as first_message
                       FROM sessions s
                       ORDER BY s.updated_at DESC
                           LIMIT ?
                       ''', (limit,))

        topics = []
        for row in cursor.fetchall():
            topics.append({
                'session_id': row[0],
                'name': row[1],
                'created_at': row[2],
                'first_message': row[3]
            })
        conn.close()
        return topics


    def is_first_session(self) -> bool:
        """Check if this is the very first session"""
        return self.get_user_session_count() <= 1