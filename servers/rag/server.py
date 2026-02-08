"""
RAG MCP Server
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
from tools.tool_control import check_tool_enabled, is_tool_enabled, disabled_tool_response

from mcp.server.fastmcp import FastMCP
from client.stop_signal import is_stop_requested
from tools.rag.rag_add import rag_add
from tools.rag.rag_search import rag_search
from tools.rag.rag_diagnose import diagnose_rag

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Create the root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Remove any existing handlers (in case something already configured it)
root_logger.handlers.clear()

# Create formatter
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Create file handler
file_handler = logging.FileHandler(LOG_DIR / "mcp-server.log", encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Add handlers to root logger
root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

# Disable propagation to avoid duplicate logs
logging.getLogger("mcp").setLevel(logging.DEBUG)
logging.getLogger("mcp_rag_server").setLevel(logging.INFO)

logger = logging.getLogger("mcp_rag_server")
logger.info("🚀 Server logging initialized - writing to logs/mcp-server.log")

mcp = FastMCP("rag-server")

@mcp.tool()
@check_tool_enabled(category="rag")
def rag_add_tool(text: str, source: str | None = None, chunk_size: int = 500) -> str:
    """
    Add text to the RAG (Retrieval-Augmented Generation) vector database.

    Args:
        text (str, required): Content to add (subtitles, articles, notes, etc.)
        source (str, optional): Source identifier (e.g., "movie:12345", "article:tech-news")
        chunk_size (int, optional): Words per chunk for embedding (default: 500)

    Returns:
        JSON string with:
        - chunks_added: Number of chunks created and stored
        - source: Source identifier used
        - total_text_length: Length of input text
        - embeddings_generated: Number of embeddings created

    Automatically chunks text, generates embeddings using bge-large model, and stores in vector database.

    Use when ingesting movie/TV subtitles, knowledge base articles, or any text for later semantic retrieval.
    """
    logger.info(f"🛠 [server] rag_add_tool called with text length: {len(text)}, source: {source}")
    result = rag_add(text, source, chunk_size)
    return json.dumps(result, indent=2)


@mcp.tool()
@check_tool_enabled(category="rag")
def rag_search_tool(query: str = "", text: str = "", top_k: int = 5, min_score: float = 0.0) -> str:
    """
    Search the RAG database using semantic similarity with STOP SIGNAL support.

    Args:
        query (str): Search query text (primary parameter)
        text (str): Alternative parameter for search query (fallback)
        top_k (int): Number of results to return (default: 5)
        min_score (float): Minimum similarity score (default: 0.0)

    Returns:
        JSON string with search results
    """
    # Accept either parameter
    search_query = query or text

    if not search_query or not search_query.strip():
        return json.dumps({
            "error": "No search query provided",
            "message": "Please provide a search term using 'query' parameter",
            "example": {"query": "search term", "top_k": 5}
        }, indent=2)

    logger.info(f"🛠 [server] rag_search_tool called with query: {search_query}, top_k: {top_k}")

    # Check stop BEFORE expensive search
    if is_stop_requested():
        logger.warning("🛑 rag_search_tool: Stop requested - skipping search")
        return json.dumps({
            "results": [],
            "query": search_query,
            "total_results": 0,
            "stopped": True,
            "message": "Search cancelled by user"
        }, indent=2)

    result = rag_search(search_query, top_k, min_score)
    return json.dumps(result, indent=2)

@mcp.tool()
@check_tool_enabled(category="rag")
def rag_diagnose_tool() -> str:
    """
    Diagnose RAG database for incomplete or problematic entries.

    Args:
        None

    Returns:
        JSON string with:
        - total_items: Total Plex items available
        - ingested_count: Number of items successfully ingested
        - missing_subtitles: Array of items with no subtitle data:
          - title: Movie/episode title
          - id: Plex ratingKey
          - type: "movie" or "episode"
        - not_yet_ingested: Array of items not yet processed:
          - title: Movie/episode title
          - id: Plex ratingKey
          - type: "movie" or "episode"
        - statistics: Overall ingestion statistics

    Use to find which Plex items are missing subtitle data or haven't been ingested yet.
    Helps identify gaps in the RAG database.
    """
    logger.info(f"🛠 [server] rag_diagnose_tool called")
    result = diagnose_rag()
    return json.dumps(result, indent=2)

@mcp.tool()
@check_tool_enabled(category="rag")
def rag_status_tool() -> str:
    """
    Get quick status of RAG database without full diagnostics.

    Returns:
        JSON string with:
        - rag_documents: Number of documents in RAG database
        - total_words: Total words stored
        - unique_sources: Number of unique media items
        - ingestion_stats: Summary from storage tracking

    Use for quick checks of RAG database health.
    """
    logger.info(f"🛠 [server] rag_status_tool called")
    from tools.rag.rag_vector_db import get_rag_stats
    from tools.rag.rag_storage import get_ingestion_stats

    try:
        rag_stats = get_rag_stats()
        ingestion_stats = get_ingestion_stats()

        result = {
            "rag_database": {
                "total_documents": rag_stats.get("total_documents", 0),
                "total_words": rag_stats.get("total_words", 0),
                "unique_sources": rag_stats.get("unique_sources", 0)
            },
            "ingestion_tracking": {
                "total_plex_items": ingestion_stats["total_items"],
                "successfully_ingested": ingestion_stats["successfully_ingested"],
                "marked_no_subtitles": ingestion_stats["missing_subtitles"],
                "not_yet_processed": ingestion_stats["remaining"]
            },
            "summary": f"{ingestion_stats['successfully_ingested']} items ingested out of {ingestion_stats['total_items']} total ({round(ingestion_stats['successfully_ingested'] / ingestion_stats['total_items'] * 100, 1)}% complete)"
        }

        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"❌ Error getting RAG status: {e}")
        return json.dumps({"error": str(e)}, indent=2)


@mcp.tool()
@check_tool_enabled(category="rag")
def rag_browse_tool(limit: int = 10) -> str:
    """
    Browse recent documents in the RAG database with previews.

    Args:
        limit (int, optional): Number of documents to show (default: 10, max: 50)

    Returns:
        JSON string with:
        - documents: Array of recent documents with previews
        - total_shown: Number of documents shown
        - total_in_database: Total documents available

    Use to see what content has been stored recently and what topics are available.
    """
    logger.info(f"🛠 [server] rag_browse_tool called with limit: {limit}")

    try:
        from tools.rag.rag_vector_db import get_connection

        # Clamp limit
        limit = min(max(1, limit), 50)

        conn = get_connection()
        cursor = conn.cursor()

        # Get total count
        cursor.execute("SELECT COUNT(*) FROM documents")
        total_count = cursor.fetchone()[0]

        # Get recent documents with previews
        cursor.execute("""
                       SELECT id,
                              SUBSTR(text, 1, 200) as preview,
                              source,
                              word_count,
                              created_at
                       FROM documents
                       ORDER BY created_at DESC LIMIT ?
                       """, (limit,))

        documents = []
        for row in cursor.fetchall():
            # Extract readable title from source
            source = row[2] or "Unknown source"
            if source.startswith('http'):
                title = source.split('/')[-1].replace('_', ' ')
            else:
                title = source

            documents.append({
                "id": row[0],
                "preview": row[1] + "...",
                "source": source,
                "title": title,
                "words": row[3],
                "created": row[4]
            })

        result = {
            "documents": documents,
            "total_shown": len(documents),
            "total_in_database": total_count,
            "summary": f"Showing {len(documents)} of {total_count} documents"
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"❌ Error browsing RAG: {e}")
        return json.dumps({"error": str(e)}, indent=2)

@mcp.tool()
@check_tool_enabled(category="rag")
def rag_list_sources_tool() -> str:
    """
    List all unique sources stored in the RAG database with document counts.

    Returns:
        JSON string with:
        - sources: Array of sources with their document counts and sample titles
        - total_sources: Number of unique sources
        - total_documents: Total documents across all sources

    Use to see what content is available in RAG for searching.
    """
    logger.info(f"🛠 [server] rag_list_sources_tool called")

    try:
        from tools.rag.rag_vector_db import get_connection

        conn = get_connection()
        cursor = conn.cursor()

        # Get sources with counts and sample text
        cursor.execute("""
                       SELECT source,
                              COUNT(*)                  as doc_count,
                              SUM(word_count)           as total_words,
                              MIN(SUBSTR(text, 1, 100)) as sample_text
                       FROM documents
                       WHERE source IS NOT NULL
                         AND source != ''
                       GROUP BY source
                       ORDER BY doc_count DESC
                       """)

        sources = []
        total_docs = 0

        for row in cursor.fetchall():
            source = row[0]
            doc_count = row[1]
            total_words = row[2]
            sample_text = row[3]

            # Extract title from URL or use source as-is
            if source.startswith('http'):
                # Extract page title from URL
                title = source.split('/')[-1].replace('_', ' ')
            else:
                title = source

            sources.append({
                "source": source,
                "title": title,
                "documents": doc_count,
                "words": total_words,
                "sample": sample_text[:100] + "..." if len(sample_text) > 100 else sample_text
            })

            total_docs += doc_count

        result = {
            "sources": sources,
            "total_sources": len(sources),
            "total_documents": total_docs,
            "summary": f"{len(sources)} unique sources with {total_docs} total documents"
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"❌ Error listing sources: {e}")
        return json.dumps({"error": str(e)}, indent=2)

skill_registry = None

@mcp.tool()
@check_tool_enabled(category="rag")
def list_skills() -> str:
    """List all available skills for this server."""
    logger.info(f"🛠  list_skills called")
    if skill_registry is None:
        return json.dumps({
            "server": "rag-server",
            "skills": [],
            "message": "Skills not loaded"
        }, indent=2)

    return json.dumps({
        "server": "rag-server",
        "skills": skill_registry.list()
    }, indent=2)


@mcp.tool()
@check_tool_enabled(category="rag")
def read_skill(skill_name: str) -> str:
    """Read the full content of a skill."""
    logger.info(f"🛠  read_skill called")

    if skill_registry is None:
        return json.dumps({"error": "Skills not loaded"}, indent=2)

    content = skill_registry.get_skill_content(skill_name)
    if content:
        return content

    available = [s.name for s in skill_registry.skills.values()]
    return json.dumps({
        "error": f"Skill '{skill_name}' not found",
        "available_skills": available
    }, indent=2)

def get_tool_names_from_module():
    """Extract all function names from current module (auto-discovers tools)"""
    current_module = sys.modules[__name__]
    tool_names = []

    for name, obj in inspect.getmembers(current_module):
        if inspect.isfunction(obj) and obj.__module__ == __name__:
            if not name.startswith('_') and name != 'get_tool_names_from_module':
                tool_names.append(name)

    return tool_names

if __name__ == "__main__":
    # Auto-extract tool names - NO manual list needed!
    server_tools = get_tool_names_from_module()

    # Load skills
    skills_dir = Path(__file__).parent / "skills"
    loader = SkillLoader(server_tools)
    skill_registry = loader.load_all(skills_dir)

    logger.info(f"🛠  {len(server_tools)} tools: {', '.join(server_tools)}")
    logger.info(f"🛠  {len(skill_registry.skills)} skills loaded")
    mcp.run(transport="stdio")