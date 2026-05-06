"""
RAG MCP Server - WITH FEEDBACK/IMPROVEMENT SUPPORT
Runs over stdio transport
"""
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env", override=True)

from servers.skills.skill_loader import SkillLoader

import inspect
import json
import logging
from pathlib import Path
from tools.tool_control import check_tool_enabled
try:
    from client.tool_meta import tool_meta
except Exception:
    # Fallback stub — metadata is attached but not used in server subprocess
    def tool_meta(**kwargs):
        def decorator(fn): return fn
        return decorator

from mcp.server.fastmcp import FastMCP
from client.stop_signal import is_stop_requested
from tools.rag.rag_add import rag_add
from tools.rag.rag_search import rag_search

# ── Failure taxonomy ──────────────────────────────────────────────────────────
try:
    from metrics import FailureKind, MCPToolError, JsonFormatter
except ImportError:
    try:
        from client.metrics import FailureKind, MCPToolError, JsonFormatter
    except ImportError:
        from enum import Enum
        class FailureKind(Enum):
            RETRYABLE      = "retryable"
            USER_ERROR     = "user_error"
            UPSTREAM_ERROR = "upstream_error"
            INTERNAL_ERROR = "internal_error"
        class MCPToolError(Exception):
            def __init__(self, kind, message, detail=None):
                self.kind = kind; self.message = message; self.detail = detail or {}
                super().__init__(message)
        JsonFormatter = None

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.handlers.clear()

formatter = JsonFormatter() if JsonFormatter is not None else logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

file_handler = logging.FileHandler(LOG_DIR / "mcp-server.log", encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

logging.getLogger("mcp").setLevel(logging.DEBUG)
logging.getLogger("mcp_rag_server").setLevel(logging.INFO)

logger = logging.getLogger("mcp_rag_server")
logger.info("🚀 Server logging initialized - writing to logs/mcp-server.log")

mcp = FastMCP("rag-server")

@mcp.tool()
@check_tool_enabled(category="rag")
@tool_meta(
    tags=["read", "search", "rag", "session"],
    triggers=[
        "first prompt", "first message", "what did i ask",
        "earlier in this session", "start of conversation",
        "what did we discuss", "session history", "my previous messages",
        "what did i say", "beginning of this chat", "recap this session",
        "summarise this session", "summarize this session",
    ],
    idempotent=True,
    example='use session_history_tool: session_id="" [limit="20"] [order="asc"]',
)
def session_history_tool(
    session_id: str,
    limit: int = 20,
    order: str = "asc",
) -> str:
    """
    Retrieve ordered message history for a session from the database.

    Use this for temporal or structural questions about the conversation:
      - 'What was my first prompt?'
      - 'Summarise this session'
      - 'What did I ask earlier?'
      - 'What was discussed at the start of this chat?'

    Use rag_search_tool instead for semantic/content questions like
    'What did we discuss about X?' where meaning matters more than order.

    The current session ID is always included in the system prompt — pass
    it through directly when answering questions about the current session.

    Args:
        session_id (str): Session ID to retrieve history for.
                          Use the session ID from the system prompt for the current session.
        limit (int, optional): Maximum number of messages to return (default: 20).
        order (str, optional): 'asc' for oldest-first, 'desc' for newest-first (default: 'asc').

    Returns:
        JSON string with ordered messages containing role, text, timestamp, and model.
    """
    if not session_id or not str(session_id).strip():
        raise MCPToolError(
            FailureKind.USER_ERROR,
            "session_id is required. It is provided in the system prompt.",
            {"tool": "session_history_tool"},
        )

    try:
        session_id_int = int(session_id)
    except (ValueError, TypeError):
        raise MCPToolError(
            FailureKind.USER_ERROR,
            f"session_id must be an integer, got: {session_id!r}",
            {"tool": "session_history_tool"},
        )

    try:
        limit = int(limit) if limit is not None else 20
        if limit < 1:
            raise MCPToolError(
                FailureKind.USER_ERROR,
                f"limit must be >= 1, got {limit}",
                {"tool": "session_history_tool"},
            )
    except MCPToolError:
        raise
    except (TypeError, ValueError):
        raise MCPToolError(
            FailureKind.USER_ERROR,
            f"Invalid limit: {limit}",
            {"tool": "session_history_tool"},
        )

    if order not in ("asc", "desc"):
        raise MCPToolError(
            FailureKind.USER_ERROR,
            f"order must be 'asc' or 'desc', got: {order!r}",
            {"tool": "session_history_tool"},
        )

    direction = "ASC" if order == "asc" else "DESC"

    logger.info(
        f"🛠 [server] session_history_tool called: session_id={session_id_int}, "
        f"limit={limit}, order={order}"
    )

    try:
        import sqlite3 as _sqlite3

        sessions_db = str(PROJECT_ROOT / "data" / "sessions.db")
        conn = _sqlite3.connect(sessions_db)
        cursor = conn.cursor()

        # Verify session exists
        cursor.execute("SELECT id, name FROM sessions WHERE id = ?", (session_id_int,))
        session_row = cursor.fetchone()
        if not session_row:
            conn.close()
            raise MCPToolError(
                FailureKind.USER_ERROR,
                f"Session {session_id_int} not found.",
                {"tool": "session_history_tool", "session_id": session_id_int},
            )

        session_name = session_row[1] or f"Session {session_id_int}"

        # Fetch messages
        cursor.execute(
            f"""
            SELECT id, role, content, model, created_at
            FROM messages
            WHERE session_id = ?
            ORDER BY created_at {direction}
            LIMIT ?
            """,
            (session_id_int, limit),
        )
        rows = cursor.fetchall()
        conn.close()

        messages = []
        for i, row in enumerate(rows, 1):
            messages.append({
                "index":     i,
                "id":        row[0],
                "role":      row[1],
                "text":      row[2],
                "model":     row[3] or "unknown",
                "timestamp": row[4],
            })

        # Build a brief readable summary for quick orientation
        user_msgs = [m for m in messages if m["role"] == "user"]
        first_prompt = user_msgs[0]["text"][:120] if user_msgs else None
        last_prompt  = user_msgs[-1]["text"][:120] if user_msgs else None

        return json.dumps({
            "session_id":    session_id_int,
            "session_name":  session_name,
            "total_returned": len(messages),
            "order":         order,
            "first_user_prompt": first_prompt,
            "last_user_prompt":  last_prompt,
            "messages":      messages,
        }, indent=2)

    except MCPToolError:
        raise
    except Exception as e:
        logger.error(f"❌ session_history_tool failed: {e}", exc_info=True)
        raise MCPToolError(
            FailureKind.INTERNAL_ERROR,
            f"Failed to retrieve session history: {e}",
            {"tool": "session_history_tool", "session_id": session_id},
        )

@mcp.tool()
@check_tool_enabled(category="rag")
@tool_meta(tags=["write","rag"],triggers=["add to rag","ingest url","add document"],idempotent=False,example='use rag_add_tool: text="" [source=""] [chunk_size=""]')
def rag_add_tool(text: str, source: str | None = None, chunk_size: int = 500) -> str:
    """
    Add text to the RAG vector database.

    Args:
        text (str, required): Content to add
        source (str, optional): Source identifier (default: "manual")
        chunk_size (int, optional): Words per chunk (default: 500)

    Returns:
        JSON string with chunks_added, source, total_text_length, embeddings_generated.
    """
    if not text or not text.strip():
        raise MCPToolError(FailureKind.USER_ERROR, "text must not be empty",
                           {"tool": "rag_add_tool"})

    try:
        chunk_size = int(chunk_size) if chunk_size is not None else 500
        if chunk_size < 1:
            raise MCPToolError(FailureKind.USER_ERROR, f"chunk_size must be >= 1, got {chunk_size}",
                               {"tool": "rag_add_tool", "param": "chunk_size"})
    except MCPToolError:
        raise
    except (TypeError, ValueError):
        raise MCPToolError(FailureKind.USER_ERROR, f"Invalid chunk_size: {chunk_size}",
                           {"tool": "rag_add_tool", "param": "chunk_size"})

    if not source:
        source = "manual"

    logger.info(f"🛠 [server] rag_add_tool called with text length: {len(text)}, source: {source}")
    try:
        result = rag_add(text, source, chunk_size)
        return json.dumps(result, indent=2)
    except MCPToolError:
        raise
    except Exception as e:
        logger.error(f"❌ rag_add_tool failed: {e}", exc_info=True)
        raise MCPToolError(FailureKind.INTERNAL_ERROR, f"RAG add failed: {e}",
                           {"tool": "rag_add_tool", "source": source})


@mcp.tool()
@check_tool_enabled(category="rag")
@tool_meta(tags=["read","search","rag"],triggers=["search rag","find in rag","what do you know about","search my knowledge"],idempotent=True,example='use rag_search_tool: [query=""] [top_k=""] [min_score=""]',text_fields=["preview"])
def rag_search_tool(query: str = "", text: str = "", top_k: int = 5, min_score: float = 0.3) -> str:
    """
    Search the RAG database using semantic similarity.

    Args:
        query (str, optional): Search query text (str, optional. Primary search parameter)
        text (str, optional): Alternative parameter for search query (str, optional. Fallback)
        top_k (int, optional): Number of results to return (default: 5)
        min_score (float, optional): Minimum similarity score (default: 0.3)

    Returns:
        JSON string with search results and optional feedback for improvement.
    """
    search_query = query or text

    if not search_query or not search_query.strip():
        raise MCPToolError(FailureKind.USER_ERROR, "No search query provided. Use the 'query' parameter.",
                           {"tool": "rag_search_tool", "example": {"query": "search term", "top_k": 5}})

    try:
        top_k = int(top_k) if top_k is not None else 5
        if top_k < 1:
            raise MCPToolError(FailureKind.USER_ERROR, f"top_k must be >= 1, got {top_k}",
                               {"tool": "rag_search_tool", "param": "top_k"})
    except MCPToolError:
        raise
    except (TypeError, ValueError):
        raise MCPToolError(FailureKind.USER_ERROR, f"Invalid top_k: {top_k}",
                           {"tool": "rag_search_tool", "param": "top_k"})

    logger.info(f"🛠 [server] rag_search_tool called with query: {search_query}, top_k: {top_k}")

    if is_stop_requested():
        logger.warning("🛑 rag_search_tool: Stop requested - skipping search")
        return json.dumps({
            "results": [],
            "query": search_query,
            "total_results": 0,
            "stopped": True,
            "message": "Search cancelled by user"
        }, indent=2)

    try:
        result = rag_search(search_query, top_k, min_score)
    except Exception as e:
        logger.error(f"❌ rag_search_tool failed: {e}", exc_info=True)
        raise MCPToolError(FailureKind.INTERNAL_ERROR, f"RAG search failed: {e}",
                           {"tool": "rag_search_tool", "query": search_query})

    # ── Quality check and feedback logic (unchanged) ──────────────────────────
    results = result.get("results", [])

    if not results:
        result["status"] = "needs_improvement"
        result["feedback"] = {
            "reason": "No results found. The query may be too specific or use terminology not in the database.",
            "suggestions": [
                "Try more general terms (e.g., 'time travel' instead of 'temporal displacement paradox')",
                "Use simpler language",
                "Try related concepts or synonyms",
                "Search for broader topics first"
            ],
            "auto_retry": False
        }
        logger.warning(f"⚠️ No results for query: {search_query}")

    elif len(results) < 3 and top_k >= 5:
        max_score = max((r.get("score", 0) for r in results), default=0)
        if max_score < 0.5:
            result["status"] = "needs_improvement"
            result["feedback"] = {
                "reason": f"Only {len(results)} weak results found (max score: {max_score:.2f}). Query may be too narrow.",
                "suggestions": [
                    "Broaden your search terms",
                    "Remove very specific details",
                    "Try searching for the main topic only"
                ],
                "auto_retry": False
            }
            logger.warning(f"⚠️ Weak results for query: {search_query} (max score: {max_score:.2f})")

    elif results:
        avg_score = sum(r.get("score", 0) for r in results) / len(results)
        max_score = max((r.get("score", 0) for r in results), default=0)

        if avg_score < 0.4:
            result["status"] = "low_quality"
            result["feedback"] = {
                "reason": f"Results found but average relevance is low ({avg_score:.2f}). Consider refining.",
                "suggestions": [
                    "Try different phrasing",
                    "Add more context to your query",
                    "Use more specific terms if you want precise results",
                    "Use broader terms if you want more results"
                ],
                "auto_retry": False
            }
            logger.info(f"ℹ️ Low quality results for query: {search_query} (avg: {avg_score:.2f})")
        else:
            result["status"] = "success"
            logger.info(f"✅ Good results for query: {search_query} (avg: {avg_score:.2f}, max: {max_score:.2f})")

    return json.dumps(result, indent=2)


@mcp.tool()
@check_tool_enabled(category="rag")
@tool_meta(tags=["read","rag"],triggers=["rag status","rag stats","how many ingested","rag info"],idempotent=False,example="use rag_status_tool")
def rag_status_tool() -> str:
    """
    Get quick status of RAG database.

    Returns:
        JSON string with rag_documents, total_words, unique_sources, ingestion_stats.
    """
    logger.info(f"🛠 [server] rag_status_tool called")
    from tools.rag.rag_vector_db import get_rag_stats
    from tools.rag.rag_storage import get_ingestion_stats

    result = {}

    try:
        rag_stats = get_rag_stats()
        result["rag_database"] = {
            "total_documents": rag_stats.get("total_documents", 0),
            "total_words": rag_stats.get("total_words", 0),
            "unique_sources": rag_stats.get("unique_sources", 0)
        }
    except Exception as e:
        logger.error(f"❌ Error getting RAG stats: {e}")
        raise MCPToolError(FailureKind.INTERNAL_ERROR, f"RAG stats unavailable: {e}",
                           {"tool": "rag_status_tool"})

    try:
        ingestion_stats = get_ingestion_stats()
        total = ingestion_stats["total_items"]
        ingested = ingestion_stats["successfully_ingested"]
        pct = round(ingested / total * 100, 1) if total else 0
        result["ingestion_tracking"] = {
            "total_plex_items": total,
            "successfully_ingested": ingested,
            "marked_no_subtitles": ingestion_stats["missing_subtitles"],
            "not_yet_processed": ingestion_stats["remaining"]
        }
        result["summary"] = f"{ingested} items ingested out of {total} total ({pct}% complete)"
    except Exception as e:
        logger.warning(f"⚠️ Plex ingestion stats unavailable: {e}")
        result["ingestion_tracking"] = {"error": "Plex unavailable"}
        result["summary"] = (
            f"RAG database: {result.get('rag_database', {}).get('total_documents', 0)} documents. "
            f"Plex ingestion stats unavailable."
        )

    return json.dumps(result, indent=2)


@mcp.tool()
@check_tool_enabled(category="rag")
@tool_meta(tags=["read","rag"],triggers=["browse rag","show rag","list rag","rag contents","whats in rag"],idempotent=False,example="use rag_browse_tool",text_fields=["preview"])
def rag_browse_tool(limit: int = 10) -> str:
    """
    Browse recent documents in the RAG database with previews.

    Args:
        limit (int, optional): Number of documents to show (default: 10, max: 50)

    Returns:
        JSON string with documents, total_shown, total_in_database.
    """
    logger.info(f"🛠 [server] rag_browse_tool called with limit: {limit}")

    try:
        limit = int(limit) if limit is not None else 10
        limit = min(max(1, limit), 50)
    except (TypeError, ValueError):
        raise MCPToolError(FailureKind.USER_ERROR, f"Invalid limit: {limit}",
                           {"tool": "rag_browse_tool", "param": "limit"})

    try:
        import sqlite3 as _sqlite3
        from tools.rag.rag_vector_db import get_connection

        sessions_db = str(PROJECT_ROOT / "data" / "sessions.db")
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("ATTACH DATABASE ? AS sessions", (sessions_db,))

        cursor.execute("SELECT COUNT(*) FROM documents")
        total_count = cursor.fetchone()[0]

        cursor.execute("""
            SELECT d.id,
                   SUBSTR(c.text, 1, 200) AS preview,
                   d.source,
                   c.created_at,
                   LENGTH(c.text) AS text_len
            FROM documents d
            JOIN sessions.chunks c ON d.chunk_id = c.id
            ORDER BY c.created_at DESC
            LIMIT ?
        """, (limit,))

        documents = []
        for row in cursor.fetchall():
            source = row[2] or "Unknown source"
            title = source.split("/")[-1].replace("_", " ") if source.startswith("http") else source
            preview = row[1] or ""
            documents.append({
                "id": row[0],
                "preview": preview + ("..." if len(preview) >= 200 else ""),
                "source": source,
                "title": title,
                "chars": row[4],
                "created": row[3]
            })

        return json.dumps({
            "documents": documents,
            "total_shown": len(documents),
            "total_in_database": total_count,
            "summary": f"Showing {len(documents)} of {total_count} documents"
        }, indent=2)

    except MCPToolError:
        raise
    except Exception as e:
        logger.error(f"❌ Error browsing RAG: {e}")
        raise MCPToolError(FailureKind.INTERNAL_ERROR, f"RAG browse failed: {e}",
                           {"tool": "rag_browse_tool"})
    finally:
        try:
            conn.close()
        except Exception:
            pass


@mcp.tool()
@check_tool_enabled(category="rag")
@tool_meta(tags=["read","rag"],triggers=["rag sources","list sources","what sources in rag"],idempotent=True,example="use rag_list_sources_tool")
def rag_list_sources_tool() -> str:
    """
    List all unique sources stored in the RAG database with document counts.

    Returns:
        JSON string with sources, total_sources, total_documents.
    """
    logger.info(f"🛠 [server] rag_list_sources_tool called")

    try:
        from tools.rag.rag_vector_db import get_connection

        conn = get_connection()
        cursor = conn.cursor()

        sessions_db = str(PROJECT_ROOT / "data" / "sessions.db")
        cursor.execute("ATTACH DATABASE ? AS sessions", (sessions_db,))

        cursor.execute("""
            SELECT d.source,
                   COUNT(*)                        AS doc_count,
                   SUM(LENGTH(c.text))             AS total_chars,
                   MIN(SUBSTR(c.text, 1, 100))     AS sample_text
            FROM documents d
            JOIN sessions.chunks c ON d.chunk_id = c.id
            WHERE d.source IS NOT NULL
              AND d.source != ''
            GROUP BY d.source
            ORDER BY doc_count DESC
        """)

        sources = []
        total_docs = 0

        for row in cursor.fetchall():
            source = row[0]
            doc_count = row[1]
            total_chars = row[2]
            sample_text = row[3] or ""
            title = source.split("/")[-1].replace("_", " ") if source.startswith("http") else source
            sources.append({
                "source": source,
                "title": title,
                "documents": doc_count,
                "chars": total_chars,
                "sample": sample_text[:100] + "..." if len(sample_text) > 100 else sample_text
            })
            total_docs += doc_count

        return json.dumps({
            "sources": sources,
            "total_sources": len(sources),
            "total_documents": total_docs,
            "summary": f"{len(sources)} unique sources with {total_docs} total documents"
        }, indent=2)

    except MCPToolError:
        raise
    except Exception as e:
        logger.error(f"❌ Error listing sources: {e}")
        raise MCPToolError(FailureKind.INTERNAL_ERROR, f"RAG source listing failed: {e}",
                           {"tool": "rag_list_sources_tool"})
    finally:
        try:
            conn.close()
        except Exception:
            pass


@mcp.tool()
@check_tool_enabled(category="rag")
@tool_meta(tags=["destructive","rag"],triggers=["delete from rag","remove from rag","delete rag source","remove rag source"],idempotent=False,example='use rag_delete_source_tool: source=""')
def rag_delete_source_tool(source: str) -> str:
    """
    Delete all documents from a specific source from the RAG database.

    Use this to remove stale, outdated, or incorrect content so it can be
    re-ingested fresh. The source string must match exactly what is shown
    by rag_list_sources_tool.

    Args:
        source (str, required): Source identifier to delete (URL or label).
                                 Use rag_list_sources_tool to find exact values.

    Returns:
        JSON string with deleted_documents, source, and confirmation message.
    """
    if not source or not source.strip():
        raise MCPToolError(FailureKind.USER_ERROR, "source must not be empty",
                           {"tool": "rag_delete_source_tool"})

    logger.info(f"🛠 [server] rag_delete_source_tool called with source: {source}")

    try:
        from tools.rag.rag_vector_db import get_connection

        conn = get_connection()
        cursor = conn.cursor()

        # Count first so we can report how many were deleted
        cursor.execute("SELECT COUNT(*) FROM documents WHERE source = ?", (source,))
        count = cursor.fetchone()[0]

        if count == 0:
            conn.close()
            raise MCPToolError(
                FailureKind.USER_ERROR,
                f"No documents found for source: '{source}'. "
                f"Use rag_list_sources_tool to see exact source values.",
                {"tool": "rag_delete_source_tool", "source": source}
            )

        # Delete the documents (chunk text in sessions.db is left in place —
        # it is referenced by other tables and cleaned up by the sessions layer)
        cursor.execute("DELETE FROM documents WHERE source = ?", (source,))
        conn.commit()
        conn.close()

        logger.info(f"✅ Deleted {count} documents from source: {source}")
        return json.dumps({
            "deleted_documents": count,
            "source": source,
            "message": f"Deleted {count} document(s) from '{source}'. "
                       f"Re-ingest the source to restore it."
        }, indent=2)

    except MCPToolError:
        raise
    except Exception as e:
        logger.error(f"❌ rag_delete_source_tool failed: {e}", exc_info=True)
        raise MCPToolError(FailureKind.INTERNAL_ERROR, f"RAG delete failed: {e}",
                           {"tool": "rag_delete_source_tool", "source": source})


@mcp.tool()
@check_tool_enabled(category="rag")
@tool_meta(tags=["destructive","rag"],triggers=["delete rag document","remove rag document","delete document by id"],idempotent=False,example='use rag_delete_document_tool: document_id=""')
def rag_delete_document_tool(document_id: str) -> str:
    """
    Delete a single document from the RAG database by its ID.

    Use rag_browse_tool to find document IDs. Useful for removing a specific
    chunk without deleting the entire source.

    Args:
        document_id (str, required): Document ID from rag_browse_tool results.

    Returns:
        JSON string with deleted (bool), document_id, and source of deleted doc.
    """
    if not document_id or not document_id.strip():
        raise MCPToolError(FailureKind.USER_ERROR, "document_id must not be empty",
                           {"tool": "rag_delete_document_tool"})

    logger.info(f"🛠 [server] rag_delete_document_tool called with id: {document_id}")

    try:
        from tools.rag.rag_vector_db import get_connection

        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT source FROM documents WHERE id = ?", (document_id,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            raise MCPToolError(
                FailureKind.USER_ERROR,
                f"Document ID '{document_id}' not found. "
                f"Use rag_browse_tool to find valid IDs.",
                {"tool": "rag_delete_document_tool", "document_id": document_id}
            )

        source = row[0]
        cursor.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        conn.commit()
        conn.close()

        logger.info(f"✅ Deleted document {document_id} from source: {source}")
        return json.dumps({
            "deleted": True,
            "document_id": document_id,
            "source": source,
            "message": f"Document '{document_id}' deleted from source '{source}'."
        }, indent=2)

    except MCPToolError:
        raise
    except Exception as e:
        logger.error(f"❌ rag_delete_document_tool failed: {e}", exc_info=True)
        raise MCPToolError(FailureKind.INTERNAL_ERROR, f"RAG document delete failed: {e}",
                           {"tool": "rag_delete_document_tool", "document_id": document_id})



@mcp.tool()
@check_tool_enabled(category="rag")
def list_capabilities(filter_tags: str | None = None) -> str:
    """
    Return the full capability schema for every tool on this server.

    Agents call this to discover what this server can do, what parameters
    each tool accepts, and what constraints apply — without needing the
    client-side CapabilityRegistry.

    Args:
        filter_tags (str, optional): Comma-separated tags to filter by
                                     e.g. "read,search" or "write"

    Returns:
        JSON string with:
        - server: Server name
        - tools: Array of tool capability objects, each with:
            - name, description, input_schema, tags, rate_limit,
              idempotent, example
    """
    logger.info(f"🛠  list_capabilities called (filter_tags={filter_tags})")

    try:
        from client.capability_registry import (
            CapabilityRegistry, _TOOL_TAGS, _TOOL_RATE_LIMITS, _TOOL_IDEMPOTENT,
            _extract_schema, _INTERNAL_TOOLS
        )
    except ImportError:
        return json.dumps({"error": "CapabilityRegistry not available"}, indent=2)

    import sys as _sys
    _current = _sys.modules[__name__]

    wanted_tags = set(t.strip() for t in filter_tags.split(",") if t.strip()) if filter_tags else None

    tools_out = []
    seen = set()
    for _name, _obj in vars(_current).items():
        if not callable(_obj) or _name.startswith("_") or _name in _INTERNAL_TOOLS:
            continue
        if not hasattr(_obj, "__tool_meta__") and not hasattr(_obj, "name"):
            # Only include decorated MCP tools — they have __wrapped__ or .name
            # Fall back to checking if the function is in the mcp tool registry
            _tool_fn = getattr(_current, _name, None)
            if not (hasattr(_tool_fn, "__tool_meta__") or hasattr(_tool_fn, "_mcp_tool")):
                continue
        if _name in seen:
            continue
        seen.add(_name)

        tags = _TOOL_TAGS.get(_name, [])
        if wanted_tags and not (wanted_tags & set(tags)):
            continue

        # Build minimal ParamSchema list inline (no full tool object available)
        import inspect as _inspect
        sig = _inspect.signature(_obj)
        params = []
        for pname, param in sig.parameters.items():
            if pname in ("self",):
                continue
            has_default = param.default is not _inspect.Parameter.empty
            ann = param.annotation
            type_str = (
                ann.__name__ if hasattr(ann, "__name__")
                else str(ann).replace("typing.", "").replace("Optional[", "").rstrip("]")
                if ann is not _inspect.Parameter.empty else "string"
            )
            params.append({
                "name":     pname,
                "type":     type_str,
                "required": not has_default,
                "default":  None if not has_default else str(param.default),
            })

        tools_out.append({
            "name":         _name,
            "description":  (_obj.__doc__ or "").strip().split("\n")[0],
            "input_schema": params,
            "tags":         tags,
            "rate_limit":   _TOOL_RATE_LIMITS.get(_name),
            "idempotent":   _TOOL_IDEMPOTENT.get(_name, True),
        })

    return json.dumps({
        "server": mcp.name,
        "tools":  tools_out,
        "total":  len(tools_out),
    }, indent=2)


@mcp.tool()
@check_tool_enabled(category="rag")
def list_skills() -> str:
    """List all available skills for this server."""
    logger.info(f"🛠 list_skills called")
    if skill_registry is None:
        return json.dumps({"server": "rag-server", "skills": [], "message": "Skills not loaded"}, indent=2)
    return json.dumps({"server": "rag-server", "skills": skill_registry.list()}, indent=2)


@mcp.tool()
@check_tool_enabled(category="rag")
def read_skill(skill_name: str) -> str:
    """Read the full content of a skill."""
    logger.info(f"🛠 read_skill called")
    if skill_registry is None:
        return json.dumps({"error": "Skills not loaded"}, indent=2)
    content = skill_registry.get_skill_content(skill_name)
    if content:
        return content
    available = [s.name for s in skill_registry.skills.values()]
    return json.dumps({"error": f"Skill '{skill_name}' not found", "available_skills": available}, indent=2)


def get_tool_names_from_module():
    current_module = sys.modules[__name__]
    tool_names = []
    for name, obj in inspect.getmembers(current_module):
        if inspect.isfunction(obj) and obj.__module__ == __name__:
            if not name.startswith('_') and name != 'get_tool_names_from_module':
                tool_names.append(name)
    return tool_names


if __name__ == "__main__":
    server_tools = get_tool_names_from_module()
    skills_dir = Path(__file__).parent / "skills"
    loader = SkillLoader(server_tools, category="rag")
    skill_registry = loader.load_all(skills_dir)
    logger.info(f"🛠 {len(server_tools)} tools: {', '.join(server_tools)}")
    logger.info(f"🛠 {len(skill_registry.skills)} skills loaded")
    mcp.run(transport="stdio")