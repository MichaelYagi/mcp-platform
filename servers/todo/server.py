"""
To-do Tools MCP Server
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
from tools.todo.add_todo import add_todo
from tools.todo.list_todos import list_todos
from tools.todo.update_todo import update_todo
from tools.todo.delete_todo import delete_todo
from tools.todo.search_todos import search_todos
from tools.todo.delete_all_todos import delete_all_todos

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
logging.getLogger("mcp_todo_tools_server").setLevel(logging.INFO)

logger = logging.getLogger("mcp_todo_tools_server")
logger.info("🚀 Server logging initialized - writing to logs/mcp-server.log")

mcp = FastMCP("todo-tools-server")

@mcp.tool()
@check_tool_enabled(category="todo")
def add_todo_item(title: str,
                  description: Optional[str] = None,
                  due_by: Optional[str] = None) -> str:
    """
    Create a new TODO/TASK item.

    NOT for knowledge base entries - use add_entry for notes.

    Args:
        title (str, required): Task title or name
        description (str, optional): Additional details about the task
        due_by (str, optional): Due date in YYYY-MM-DD format

    Returns:
        JSON string with the created todo containing:
        - id: Unique task identifier (use this for updates)
        - title: Task title
        - description: Task description
        - status: Current status ("open" by default)
        - due_by: Due date
        - created_at: Creation timestamp

    Use when user wants to track tasks, reminders, or deadlines.
    """
    logger.info(f"🛠 [server] add_todo_item called with title: {title}, description: {description}, due_date: {due_by}")
    result = add_todo(title, description, due_by)
    return json.dumps(result, indent=2)


@mcp.tool()
@check_tool_enabled(category="todo")
def list_todo_items() -> str:
    """
    List all todo items.

    Args:
        None

    Returns:
        JSON string with array of all todos, each containing:
        - id: Task identifier (CRITICAL: use this ID for update_todo_item)
        - title: Task title
        - description: Task description
        - status: Current status (open/complete)
        - due_by: Due date
        - created_at: When it was created

    IMPORTANT: Extract the 'id' field to use with update_todo_item or delete_todo_item.

    Use when user wants an overview of their tasks or reminders.
    """
    logger.info(f"🛠 [server] list_todo_items called")
    result = list_todos()
    return json.dumps(result, indent=2)


@mcp.tool()
@check_tool_enabled(category="todo")
def search_todo_items(text: Optional[str] = None,
                      status: Optional[str] = None,
                      due_before: Optional[str] = None,
                      due_after: Optional[str] = None,
                      order_by: Optional[str] = None,
                      ascending: bool = True) -> str:
    """
    Search and filter todo items with multiple criteria.

    Args:
        text (str, optional): Search in title/description
        status (str, optional): Filter by status ("open" or "complete")
        due_before (str, optional): Due before this date (YYYY-MM-DD)
        due_after (str, optional): Due after this date (YYYY-MM-DD)
        order_by (str, optional): Sort field (due_by, created_at, title)
        ascending (bool, optional): Sort direction (default: True)

    Returns:
        JSON string with array of matching todos, each containing:
        - id: Task identifier
        - title: Task title
        - description: Task description
        - status: Current status
        - due_by: Due date
        - created_at: Creation timestamp

    Use when user wants to find, filter, or organize tasks by specific criteria.
    """
    logger.info(
        f"🛠 [server] search_todo_items called with text: {text}, status: {status}, due_before: {due_before}, due_after: {due_after}")
    result = search_todos(
        text=text,
        status=status,
        due_before=due_before,
        due_after=due_after,
        order_by=order_by or "due_by",
        ascending=ascending
    )
    return json.dumps(result, indent=2)


@mcp.tool()
@check_tool_enabled(category="todo")
def update_todo_item(todo_id: str,
                     title: Optional[str] = None,
                     description: Optional[str] = None,
                     status: Optional[str] = None,
                     due_by: Optional[str] = None) -> str:
    """
    Update a TODO/TASK item (e.g., mark complete).

    FOR TODO LISTS ONLY. Use update_entry for knowledge base notes.

    Args:
        todo_id (str, required): The 'id' field from list_todo_items
        title (str, optional): New title
        description (str, optional): New description
        status (str, optional): New status ("open" or "complete")
        due_by (str, optional): New due date (YYYY-MM-DD)

    Returns:
        JSON string with the updated todo containing:
        - id: Task identifier
        - title: Updated title
        - description: Updated description
        - status: Updated status
        - due_by: Updated due date
        - updated_at: Modification timestamp

    IMPORTANT: Use the exact 'id' from list_todo_items, not the task title.

    Use when user wants to modify, mark complete, or correct a task.
    """
    logger.info(
        f"🛠 [server] update_todo_item called with todo_id: {todo_id}, title: {title}, description: {description}, status: {status}, due_date: {due_by}")
    result = update_todo(todo_id, title, description, status, due_by)
    return json.dumps(result, indent=2)


@mcp.tool()
@check_tool_enabled(category="todo")
def delete_todo_item(todo_id: str) -> str:
    """
    Delete a single todo item by its ID.

    Args:
        todo_id (str, required): The 'id' field from list_todo_items

    Returns:
        JSON string with:
        - success: Boolean indicating if deletion succeeded
        - deleted_id: ID of the deleted task
        - message: Confirmation or error message

    Use when user wants to remove a specific task.
    """
    logger.info(f"🛠 [server] delete_todo_item called with todo_id: {todo_id}")
    result = delete_todo(todo_id)
    return json.dumps(result, indent=2)


@mcp.tool()
@check_tool_enabled(category="todo")
def delete_all_todo_items() -> str:
    """
    Delete ALL todo items. Use with caution!

    Args:
        None

    Returns:
        JSON string with:
        - deleted_count: Number of tasks deleted
        - deleted_ids: List of all deleted task IDs

    Use when user explicitly wants to clear their entire task list.
    """
    logger.info(f"🛠 [server] delete_all_todo_items called")
    deleted_ids = delete_all_todos()
    result = {
        "deleted_count": len(deleted_ids),
        "deleted_ids": deleted_ids
    }
    return json.dumps(result, indent=2)

skill_registry = None

@mcp.tool()
@check_tool_enabled(category="todo")
def list_skills() -> str:
    """List all available skills for this server."""
    logger.info(f"🛠  list_skills called")
    if skill_registry is None:
        return json.dumps({
            "server": "todo-server",
            "skills": [],
            "message": "Skills not loaded"
        }, indent=2)

    return json.dumps({
        "server": "todo-server",
        "skills": skill_registry.list()
    }, indent=2)


@mcp.tool()
@check_tool_enabled(category="todo")
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
    loader = SkillLoader(server_tools, category="todo")
    skill_registry = loader.load_all(skills_dir)

    logger.info(f"🛠  {len(server_tools)} tools: {', '.join(server_tools)}")
    logger.info(f"🛠  {len(skill_registry.skills)} skills loaded")
    mcp.run(transport="stdio")