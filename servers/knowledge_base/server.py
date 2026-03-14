"""
Knowledge Base MCP Server
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
from tools.knowledge_base.kb_add import kb_add
from tools.knowledge_base.kb_get import kb_get
from tools.knowledge_base.kb_search import kb_search
from tools.knowledge_base.kb_update import kb_update
from tools.knowledge_base.kb_delete import kb_delete
from tools.knowledge_base.kb_delete_many import kb_delete_many
from tools.knowledge_base.kb_list import kb_list
from tools.knowledge_base.kb_search_tags import kb_search_tags
from tools.knowledge_base.kb_search_semantic import kb_search_semantic
from tools.knowledge_base.kb_update_versioned import kb_update_versioned

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
logging.getLogger("mcp_kb_server").setLevel(logging.INFO)

logger = logging.getLogger("mcp_kb_server")
logger.info("🚀 Server logging initialized - writing to logs/mcp-server.log")

mcp = FastMCP("kb-server")

@mcp.tool()
@check_tool_enabled(category="knowledge_base")
def add_entry(title: str, content: str, tags: List[str] | None = None) -> str:
    """
    Create a new KNOWLEDGE BASE entry (for storing information, notes, URLs).

    NOT for todo items - use add_todo_item for tasks.

    Args:
        title (str, required): Title or heading for the entry
        content (str, required): The actual content/text to store
        tags (List[str], optional): List of category tags for organization

    Returns:
        JSON string containing the created entry with:
        - id: Generated unique identifier
        - title: Entry title
        - content: Stored content
        - tags: Associated tags
        - created_at: Timestamp

    Use when user wants to save information, notes, URLs, or structured knowledge.
    """
    logger.info(f"🛠 [server] add_entry called with title: {title}, content: {content}, tags: {tags}")
    tags = tags or []
    result = kb_add(title, content, tags)
    return json.dumps(result)


@mcp.tool()
@check_tool_enabled(category="knowledge_base")
def search_entries(query: str) -> str:
    """
    Perform full-text search across all knowledge base entries.

    Args:
        query (str, required): Keywords, phrases, or text to search for

    Returns:
        JSON string with array of matching entries, each containing:
        - id: Entry identifier
        - title: Entry title
        - content: Entry content
        - tags: Associated tags
        - relevance_score: Match quality

    Use when user wants to find, look up, or retrieve entries by content.
    """
    logger.info(f"🛠 [server] search_entries called with query: {query}")
    results = kb_search(query)
    return json.dumps(results, indent=2)


@mcp.tool()
@check_tool_enabled(category="knowledge_base")
def search_by_tag(tag: str) -> str:
    """
    Retrieve all knowledge base entries with a specific tag.

    Args:
        tag (str, required): The tag name to filter by

    Returns:
        JSON string with array of entries having that tag, each containing:
        - id: Entry identifier
        - title: Entry title
        - content: Entry content
        - tags: All associated tags

    Use when user asks for entries grouped by topic, category, or label.
    """
    logger.info(f"🛠 [server] search_by_tag called with tag: {tag}")
    result = kb_search_tags(tag)
    return json.dumps(result, indent=2)


@mcp.tool()
@check_tool_enabled(category="knowledge_base")
def search_semantic(query: str, top_k: Optional[int] = 5) -> str:
    """
    Perform semantic (embedding-based) search across the knowledge base.

    Args:
        query (str, required): Concept or question to find related content
        top_k (int, optional): Number of results to return (default: 5)

    Returns:
        JSON string with array of most relevant entries, each containing:
        - id: Entry identifier
        - title: Entry title
        - content: Entry content
        - tags: Associated tags
        - similarity_score: Semantic relevance (0-1)

    Use for finding conceptually related ideas, similar content, or concept-level matches.
    """
    top_k = int(top_k) if top_k is not None else 5
    top_k = int(top_k) if top_k is not None else 5
    logger.info(f"🛠 [server] search_semantic called with query: {query}")
    result = kb_search_semantic(query, top_k)
    return json.dumps(result, indent=2)


@mcp.tool()
@check_tool_enabled(category="knowledge_base")
def get_entry(entry_id: str) -> str:
    """
    Retrieve a single knowledge base entry by its ID.

    Args:
        entry_id (str, required): The unique identifier of the entry

    Returns:
        JSON string containing:
        - id: Entry identifier
        - title: Entry title
        - content: Full entry content
        - tags: Associated tags
        - created_at: Creation timestamp
        - updated_at: Last modification timestamp

    Use when user wants to view, inspect, or reference a specific saved entry.
    """
    logger.info(f"🛠 [server] get_entry called with entry_id: {entry_id}")
    result = kb_get(entry_id)
    return json.dumps(result)


@mcp.tool()
@check_tool_enabled(category="knowledge_base")
def delete_entry(entry_id: str) -> str:
    """
    Delete a single knowledge base entry by ID.

    Args:
        entry_id (str, required): The unique identifier of the entry to delete

    Returns:
        JSON string with:
        - success: Boolean indicating if deletion succeeded
        - deleted_id: ID of the deleted entry
        - message: Confirmation or error message

    Use when user wants to remove or clean up a specific entry.
    """
    logger.info(f"🛠 [server] delete_entry called with entry_id: {entry_id}")
    result = kb_delete(entry_id)
    return json.dumps(result)


@mcp.tool()
@check_tool_enabled(category="knowledge_base")
def delete_entries(entry_ids: List[str]) -> str:
    """
    Delete multiple knowledge base entries at once.

    Args:
        entry_ids (List[str], required): List of entry IDs to delete

    Returns:
        JSON string with:
        - deleted_count: Number of entries successfully deleted
        - deleted_ids: List of IDs that were removed
        - failed_ids: List of IDs that couldn't be deleted (if any)

    Use for bulk cleanup or batch deletion operations.
    """
    logger.info(f"🛠 [server] delete_entries called with entry_ids: {entry_ids}")
    result = kb_delete_many(entry_ids)
    return json.dumps(result, indent=2)


@mcp.tool()
@check_tool_enabled(category="knowledge_base")
def update_entry(entry_id: str,
                 title: str | None = None,
                 content: str | None = None,
                 tags: List[str] | None = None) -> str:
    """
    Update an existing knowledge base entry.

    NOT for todo items - use update_todo_item for tasks.

    Args:
        entry_id (str, required): ID of the entry to update
        title (str, optional): New title (omit to keep current)
        content (str, optional): New content (omit to keep current)
        tags (List[str], optional): New tags list (omit to keep current)

    Returns:
        JSON string with the updated entry containing:
        - id: Entry identifier
        - title: Updated title
        - content: Updated content
        - tags: Updated tags
        - updated_at: New modification timestamp

    Use when user wants to revise, correct, or expand an entry.
    """
    logger.info(
        f"🛠 [server] update_entry called with entry_id: {entry_id}, title: {title}, content: {content}, tags: {tags}")
    result = kb_update(entry_id, title, content, tags)
    return json.dumps(result, indent=2)


@mcp.tool()
@check_tool_enabled(category="knowledge_base")
def list_entries() -> str:
    """
    List all entries in the knowledge base.

    Args:
        None

    Returns:
        JSON string with array of all entries, each containing:
        - id: Entry identifier
        - title: Entry title
        - content: Entry content (may be truncated)
        - tags: Associated tags
        - created_at: Creation timestamp

    Use when user wants an overview, index, or inventory of stored information.
    """
    logger.info(f"🛠 [server] list_entries called")
    result = kb_list()
    return json.dumps(result, indent=2)


@mcp.tool()
@check_tool_enabled(category="knowledge_base")
def update_entry_versioned(entry_id: str,
                           title: str | None = None,
                           content: str | None = None,
                           tags: List[str] | None = None) -> str:
    """
    Update an entry while preserving version history.

    Args:
        entry_id (str, required): ID of the entry to update
        title (str, optional): New title
        content (str, optional): New content
        tags (List[str], optional): New tags list

    Returns:
        JSON string with:
        - current: The new version with all fields
        - version_number: Version identifier
        - previous_version: Reference to the old version
        - change_summary: What was modified

    Use when user wants safe, versioned edits or audit-friendly changes.
    """
    logger.info(
        f"🛠 [server] update_entry_versioned called with entry_id: {entry_id}, title: {title}, content: {content}, tags: {tags}")
    result = kb_update_versioned(entry_id, title, content, tags)
    return json.dumps(result, indent=2)

skill_registry = None

@mcp.tool()
@check_tool_enabled(category="knowledge_base")
def list_skills() -> str:
    """List all available skills for this server."""
    logger.info(f"🛠  list_skills called")
    if skill_registry is None:
        return json.dumps({
            "server": "knowledge-base-server",
            "skills": [],
            "message": "Skills not loaded"
        }, indent=2)

    return json.dumps({
        "server": "knowledge-base--server",
        "skills": skill_registry.list()
    }, indent=2)


@mcp.tool()
@check_tool_enabled(category="knowledge_base")
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
    loader = SkillLoader(server_tools, category="knowledge_base")
    skill_registry = loader.load_all(skills_dir)

    logger.info(f"🛠  {len(server_tools)} tools: {', '.join(server_tools)}")
    logger.info(f"🛠  {len(skill_registry.skills)} skills loaded")
    mcp.run(transport="stdio")