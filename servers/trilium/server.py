"""
Trilium Notes MCP Server
Provides tools to search and manage Trilium notes via ETAPI
Runs over stdio transport
"""
import sys
import os
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
from datetime import datetime
import requests
from pathlib import Path
from tools.tool_control import check_tool_enabled, is_tool_enabled, disabled_tool_response

from mcp.server.fastmcp import FastMCP

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Create the root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Remove any existing handlers
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
logging.getLogger("trilium_server").setLevel(logging.INFO)

logger = logging.getLogger("trilium_server")
logger.info("🚀 Server logging initialized - writing to logs/trilium-server.log")

mcp = FastMCP("trilium-server")

# ═══════════════════════════════════════════════════════════════════
# ENVIRONMENT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

TRILIUM_URL = os.getenv("TRILIUM_URL")
TRILIUM_TOKEN = os.getenv("TRILIUM_TOKEN")

if not TRILIUM_URL or not TRILIUM_TOKEN:
    logger.warning("=" * 70)
    logger.warning("⚠️  TRILIUM_URL and TRILIUM_TOKEN not configured")
    logger.warning("   Trilium server will start but tools will be unavailable")
    logger.warning("   Set these in .env to enable Trilium functionality:")
    logger.warning("   TRILIUM_URL=http://localhost:8080")
    logger.warning("   TRILIUM_TOKEN=your_etapi_token_here")
    logger.warning("=" * 70)
    TRILIUM_AVAILABLE = False
else:
    TRILIUM_AVAILABLE = True
    logger.info("✅ Trilium configuration found")

# ═══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

def trilium_unavailable_error():
    """Return error message when Trilium is not configured"""
    return {
        "error": "Trilium is not configured",
        "message": "Set TRILIUM_URL and TRILIUM_TOKEN in .env to enable Trilium functionality",
        "help": {
            "TRILIUM_URL": "http://localhost:8080",
            "TRILIUM_TOKEN": "Get from Trilium: Options > ETAPI > Create token"
        }
    }


def make_request(method: str, endpoint: str, data: dict = None, params: dict = None, expect_json: bool = True) -> dict | str:
    """
    Make authenticated request to Trilium ETAPI

    Args:
        method: HTTP method (GET, POST, PUT, DELETE, PATCH)
        endpoint: API endpoint (e.g., '/notes/search')
        data: JSON body for POST/PUT/PATCH
        params: Query parameters for GET
        expect_json: Whether to parse response as JSON (default: True)

    Returns:
        Response JSON dict, raw text string, or error dict
    """
    if not TRILIUM_AVAILABLE:
        return trilium_unavailable_error()

    url = f"{TRILIUM_URL}/etapi{endpoint}"
    headers = {
        "Authorization": TRILIUM_TOKEN,
        "Content-Type": "application/json"
    }

    try:
        response = requests.request(
            method=method,
            url=url,
            headers=headers,
            json=data,
            params=params,
            timeout=30
        )

        if response.status_code == 404:
            return {"error": "Not found", "status": 404}

        if response.status_code >= 400:
            return {
                "error": f"HTTP {response.status_code}",
                "message": response.text,
                "status": response.status_code
            }

        # Return empty dict for 204 No Content
        if response.status_code == 204:
            return {"success": True, "status": 204}

        # Return raw text if not expecting JSON (for content endpoints)
        if not expect_json:
            return response.text

        # Try to parse as JSON
        return response.json()

    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Request failed: {e}")
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════
# SEARCH TOOLS
# ═══════════════════════════════════════════════════════════════════

@mcp.tool()
@check_tool_enabled(category="trilium")
def search_notes(query: str, limit: int = 50, include_content: bool = True) -> str:
    """
    Search Trilium notes using full-text search.

    Searches note titles and content. Supports Trilium search syntax:
    - Simple words: "meeting notes"
    - Phrases: "project planning"
    - Wildcards: "meet*" matches "meeting", "meetings"
    - Labels: "#todo" finds notes with todo label
    - Operators: AND, OR, NOT

    Args:
        query: Search query (e.g., "meeting notes", "#project", "todo AND urgent")
        limit: Maximum number of results (default: 50)
        include_content: Whether to fetch content preview (slower but more detailed, default: True)

    Returns:
        JSON with matching notes including noteId, title, type, content preview (if enabled)
    """
    if not TRILIUM_AVAILABLE:
        return json.dumps(trilium_unavailable_error(), indent=2)

    logger.info(f"🔍 [server] search_notes called with query: '{query}' (limit: {limit}, content: {include_content})")

    result = make_request(
        method="GET",
        endpoint="/notes",
        params={
            "search": query,
            "limit": limit
        }
    )

    if "error" in result:
        return json.dumps(result, indent=2)

    # Get full note details for each result
    notes = []
    for note_summary in result.get("results", [])[:limit]:
        note_id = note_summary.get("noteId")

        # Get note metadata
        note_detail = make_request("GET", f"/notes/{note_id}")

        if "error" not in note_detail:
            note_data = {
                "noteId": note_id,
                "title": note_detail.get("title", "Untitled"),
                "type": note_detail.get("type", "text"),
                "dateCreated": note_detail.get("dateCreated"),
                "dateModified": note_detail.get("dateModified")
            }

            # Optionally fetch content (slower)
            if include_content:
                # Get note content separately (returns raw text)
                content = make_request("GET", f"/notes/{note_id}/content", expect_json=False)

                # Handle content
                if isinstance(content, str):
                    preview = content[:200] + "..." if len(content) > 200 else content
                else:
                    content = ""
                    preview = ""

                note_data["contentPreview"] = preview
                note_data["contentLength"] = len(content)

            notes.append(note_data)

    response = {
        "query": query,
        "total_results": len(notes),
        "results": notes,
        "content_included": include_content
    }

    logger.info(f"✅ [server] Found {len(notes)} notes")
    return json.dumps(response, indent=2)


@mcp.tool()
@check_tool_enabled(category="trilium")
def search_by_label(label: str, value: str = None, limit: int = 50) -> str:
    """
    Search notes by label (attribute).

    Labels in Trilium are key-value pairs attached to notes.
    Examples: #project, #status=active, #priority=high

    Args:
        label: Label name (e.g., "project", "todo", "priority")
        value: Optional label value (e.g., "active", "high")
        limit: Maximum number of results (default: 50)

    Returns:
        JSON with matching notes

    Examples:
        - search_by_label("todo") - finds all notes with #todo label
        - search_by_label("priority", "high") - finds #priority=high notes
        - search_by_label("project", "website") - finds #project=website notes
    """
    if not TRILIUM_AVAILABLE:
        return json.dumps(trilium_unavailable_error(), indent=2)

    # Build search query
    if value:
        query = f"#{label}={value}"
    else:
        query = f"#{label}"

    logger.info(f"🏷️  [server] search_by_label called with label: {query}")

    # Use the search endpoint with label query
    return search_notes(query, limit)


@mcp.tool()
@check_tool_enabled(category="trilium")
def get_note_by_id(note_id: str) -> str:
    """
    Get full details of a specific note by ID.

    Args:
        note_id: Trilium note ID (e.g., "xnEwGlHiQZhN")

    Returns:
        JSON with complete note details including:
        - noteId, title, type (text, code, image, etc.)
        - content (full note content)
        - dateCreated, dateModified
        - parentNoteIds, childNoteIds
        - attributes array (labels and relations)
    """
    if not TRILIUM_AVAILABLE:
        return json.dumps(trilium_unavailable_error(), indent=2)

    logger.info(f"📄 [server] get_note_by_id called with noteId: {note_id}")

    # Get note metadata
    note = make_request("GET", f"/notes/{note_id}")

    if "error" in note:
        return json.dumps(note, indent=2)

    # Get note content (returns raw HTML/text, not JSON)
    content = make_request("GET", f"/notes/{note_id}/content", expect_json=False)

    # Add content to note dict
    if isinstance(content, str):
        note["content"] = content
    elif isinstance(content, dict) and "error" in content:
        note["content_error"] = content.get("error")
        note["content"] = ""

    logger.info(f"✅ [server] Retrieved note: {note.get('title', 'Untitled')}")
    return json.dumps(note, indent=2)


# ═══════════════════════════════════════════════════════════════════
# NOTE MANAGEMENT TOOLS
# ═══════════════════════════════════════════════════════════════════

@mcp.tool()
@check_tool_enabled(category="trilium")
def create_note(
    parent_note_id: str,
    title: str,
    content: str = "",
    note_type: str = "text"
) -> str:
    """
    Create a new note in Trilium.

    Args:
        parent_note_id: ID of parent note (use "root" for top level)
        title: Note title
        content: Note content (default: empty)
        note_type: Note type - "text", "code", "book", "render" (default: "text")

    Returns:
        JSON with created note details including noteId

    Examples:
        - create_note("root", "Meeting Notes", "Discussion points...")
        - create_note("xnEwGl", "TODO List", "- Task 1\n- Task 2")
        - create_note("abc123", "Code Snippet", "def hello()...", "code")
    """
    if not TRILIUM_AVAILABLE:
        return json.dumps(trilium_unavailable_error(), indent=2)

    logger.info(f"📝 [server] create_note called: '{title}' under {parent_note_id}")

    # Create note with POST /notes
    result = make_request(
        method="POST",
        endpoint="/notes",
        data={
            "parentNoteId": parent_note_id,
            "title": title,
            "type": note_type,
            "content": content
        }
    )

    if "error" in result:
        return json.dumps(result, indent=2)

    note_id = result.get("note", {}).get("noteId") if "note" in result else result.get("noteId")
    logger.info(f"✅ [server] Created note: {note_id}")
    return json.dumps(result, indent=2)


@mcp.tool()
@check_tool_enabled(category="trilium")
def update_note_content(note_id: str, content: str) -> str:
    """
    Update the content of an existing note.

    Args:
        note_id: ID of note to update
        content: New content for the note

    Returns:
        JSON with success status
    """
    if not TRILIUM_AVAILABLE:
        return json.dumps(trilium_unavailable_error(), indent=2)

    logger.info(f"✏️  [server] update_note_content called for noteId: {note_id}")

    # Trilium content endpoint expects raw content in body, not JSON
    url = f"{TRILIUM_URL}/etapi/notes/{note_id}/content"
    headers = {
        "Authorization": TRILIUM_TOKEN,
        "Content-Type": "text/html"  # or text/plain depending on note type
    }

    try:
        response = requests.put(
            url=url,
            headers=headers,
            data=content.encode('utf-8'),
            timeout=30
        )

        if response.status_code >= 400:
            return json.dumps({
                "error": f"HTTP {response.status_code}",
                "message": response.text
            }, indent=2)

        logger.info(f"✅ [server] Updated note content")
        return json.dumps({
            "success": True,
            "noteId": note_id,
            "message": "Content updated"
        }, indent=2)

    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Failed to update content: {e}")
        return json.dumps({"error": str(e)}, indent=2)


@mcp.tool()
@check_tool_enabled(category="trilium")
def update_note_title(note_id: str, title: str) -> str:
    """
    Update the title of an existing note.

    Args:
        note_id: ID of note to update
        title: New title for the note

    Returns:
        JSON with success status
    """
    if not TRILIUM_AVAILABLE:
        return json.dumps(trilium_unavailable_error(), indent=2)

    logger.info(f"✏️  [server] update_note_title called: {note_id} → '{title}'")

    result = make_request(
        method="PATCH",
        endpoint=f"/notes/{note_id}",
        data={"title": title}
    )

    if "error" in result:
        return json.dumps(result, indent=2)

    logger.info(f"✅ [server] Updated note title")
    return json.dumps({"success": True, "noteId": note_id, "title": title}, indent=2)


@mcp.tool()
@check_tool_enabled(category="trilium")
def delete_note(note_id: str) -> str:
    """
    Delete a note (move to trash).

    Args:
        note_id: ID of note to delete

    Returns:
        JSON with success status

    Note: This moves the note to trash, not permanent deletion.
    """
    if not TRILIUM_AVAILABLE:
        return json.dumps(trilium_unavailable_error(), indent=2)

    logger.info(f"🗑️  [server] delete_note called for noteId: {note_id}")

    result = make_request(
        method="DELETE",
        endpoint=f"/notes/{note_id}"
    )

    if "error" in result:
        return json.dumps(result, indent=2)

    logger.info(f"✅ [server] Deleted note")
    return json.dumps({"success": True, "noteId": note_id, "message": "Note deleted (moved to trash)"}, indent=2)


# ═══════════════════════════════════════════════════════════════════
# LABEL/ATTRIBUTE TOOLS
# ═══════════════════════════════════════════════════════════════════

@mcp.tool()
@check_tool_enabled(category="trilium")
def add_label_to_note(note_id: str, label: str, value: str = None) -> str:
    """
    Add a label (attribute) to a note.

    Labels help organize and categorize notes in Trilium.

    Args:
        note_id: ID of note to add label to
        label: Label name (e.g., "priority", "project", "status")
        value: Optional label value (e.g., "high", "website", "active")

    Returns:
        JSON with created attribute details

    Examples:
        - add_label_to_note("abc123", "todo") → #todo
        - add_label_to_note("abc123", "priority", "high") → #priority=high
        - add_label_to_note("abc123", "project", "website") → #project=website
    """
    if not TRILIUM_AVAILABLE:
        return json.dumps(trilium_unavailable_error(), indent=2)

    label_display = f"#{label}={value}" if value else f"#{label}"
    logger.info(f"🏷️  [server] add_label_to_note called: {note_id} → {label_display}")

    result = make_request(
        method="POST",
        endpoint="/attributes",
        data={
            "noteId": note_id,
            "type": "label",
            "name": label,
            "value": value or ""
        }
    )

    if "error" in result:
        return json.dumps(result, indent=2)

    logger.info(f"✅ [server] Added label: {label_display}")
    return json.dumps(result, indent=2)


@mcp.tool()
@check_tool_enabled(category="trilium")
def get_note_labels(note_id: str) -> str:
    """
    Get all labels (attributes) for a note.

    Note: This extracts labels from the note's attributes array.
    Trilium ETAPI includes attributes in the note response.

    Args:
        note_id: ID of note

    Returns:
        JSON with list of labels/attributes
    """
    if not TRILIUM_AVAILABLE:
        return json.dumps(trilium_unavailable_error(), indent=2)

    logger.info(f"🏷️  [server] get_note_labels called for noteId: {note_id}")

    # Get note details which includes attributes
    note = make_request("GET", f"/notes/{note_id}")

    if "error" in note:
        return json.dumps(note, indent=2)

    # Extract labels from attributes array
    attributes = note.get("attributes", [])
    labels = [attr for attr in attributes if attr.get("type") == "label"]

    logger.info(f"✅ [server] Found {len(labels)} labels")
    return json.dumps({
        "noteId": note_id,
        "labels": labels,
        "total": len(labels)
    }, indent=2)


# ═══════════════════════════════════════════════════════════════════
# NAVIGATION TOOLS
# ═══════════════════════════════════════════════════════════════════

@mcp.tool()
@check_tool_enabled(category="trilium")
def get_note_children(note_id: str) -> str:
    """
    Get child notes of a parent note.

    Args:
        note_id: ID of parent note (use "root" for top level)

    Returns:
        JSON with list of child notes
    """
    if not TRILIUM_AVAILABLE:
        return json.dumps(trilium_unavailable_error(), indent=2)

    logger.info(f"📂 [server] get_note_children called for noteId: {note_id}")

    # Get the parent note first
    parent = make_request("GET", f"/notes/{note_id}")

    if "error" in parent:
        return json.dumps(parent, indent=2)

    child_note_ids = parent.get("childNoteIds", [])

    # Get details for each child
    children = []
    for child_id in child_note_ids:
        child = make_request("GET", f"/notes/{child_id}")
        if "error" not in child:
            children.append({
                "noteId": child_id,
                "title": child.get("title", "Untitled"),
                "type": child.get("type", "text"),
                "dateModified": child.get("dateModified")
            })

    logger.info(f"✅ [server] Found {len(children)} children")
    return json.dumps({
        "parentNoteId": note_id,
        "parentTitle": parent.get("title", "Untitled"),
        "children": children,
        "total": len(children)
    }, indent=2)


@mcp.tool()
@check_tool_enabled(category="trilium")
def get_recent_notes(limit: int = 20) -> str:
    """
    Get recently modified notes.

    Args:
        limit: Maximum number of notes to return (default: 20)

    Returns:
        JSON with list of recently modified notes
    """
    if not TRILIUM_AVAILABLE:
        return json.dumps(trilium_unavailable_error(), indent=2)

    logger.info(f"📅 [server] get_recent_notes called (limit: {limit})")

    # Search with empty query returns all notes, sorted by modification date
    result = make_request(
        method="GET",
        endpoint="/notes",
        params={
            "orderBy": "dateModified",
            "orderDirection": "desc",
            "limit": limit
        }
    )

    if "error" in result:
        return json.dumps(result, indent=2)

    notes = []
    for note_summary in result.get("results", [])[:limit]:
        note_id = note_summary.get("noteId")
        note = make_request("GET", f"/notes/{note_id}")

        if "error" not in note:
            notes.append({
                "noteId": note_id,
                "title": note.get("title", "Untitled"),
                "type": note.get("type", "text"),
                "dateModified": note.get("dateModified")
            })

    logger.info(f"✅ [server] Found {len(notes)} recent notes")
    return json.dumps({
        "notes": notes,
        "total": len(notes)
    }, indent=2)


# ═══════════════════════════════════════════════════════════════════
# SKILLS AND SERVER STARTUP
# ═══════════════════════════════════════════════════════════════════

skill_registry = None

@mcp.tool()
@check_tool_enabled(category="trilium")
def list_skills() -> str:
    """List all available skills for this server."""
    logger.info(f"🛠  list_skills called")
    if skill_registry is None:
        return json.dumps({
            "server": "trilium-server",
            "skills": [],
            "message": "Skills not loaded"
        }, indent=2)

    return json.dumps({
        "server": "trilium-server",
        "skills": skill_registry.list()
    }, indent=2)


@mcp.tool()
@check_tool_enabled(category="trilium")
def read_skill(skill_name: str) -> str:
    """Read the full content of a skill."""
    logger.info(f"🛠  read_skill called with skill_name: {skill_name}")

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
            if not name.startswith('_') and name not in [
                'get_tool_names_from_module',
                'trilium_unavailable_error',
                'make_request'
            ]:
                tool_names.append(name)

    return tool_names


if __name__ == "__main__":
    # Auto-extract tool names - NO manual list needed!
    server_tools = get_tool_names_from_module()

    # Load skills
    skills_dir = Path(__file__).parent / "skills"
    loader = SkillLoader(server_tools, category="trilium")
    skill_registry = loader.load_all(skills_dir)

    logger.info(f"🛠  {len(server_tools)} tools: {', '.join(server_tools)}")
    logger.info(f"🛠  {len(skill_registry.skills)} skills loaded")

    if TRILIUM_AVAILABLE:
        logger.info("✅ Trilium server starting with full functionality")
    else:
        logger.warning("⚠️  Trilium server starting with limited functionality (set TRILIUM_URL and TRILIUM_TOKEN)")

    mcp.run(transport="stdio")