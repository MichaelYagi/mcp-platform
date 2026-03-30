"""
Tool Disabling Helper for MCP Servers
=====================================

Allows disabling specific tools via environment variable.

Setup in .env:
    # Disable individual tools
    DISABLED_TOOLS=add_todo_item,delete_all_todo_items,terminate_process

    # Or disable all tools from a category
    DISABLED_TOOLS=todo:*,system:terminate_process

Usage in server:
    from tools.tool_control import is_tool_enabled, disabled_tool_response

    @mcp.tool()
    def my_tool(param: str) -> str:
        if not is_tool_enabled("my_tool"):
            return disabled_tool_response("my_tool")

        # Normal tool logic
        return do_work(param)

Or use decorator:
    from tools.tool_control import check_tool_enabled

    @mcp.tool()
    @check_tool_enabled
    def my_tool(param: str) -> str:
        # Normal tool logic
        return do_work(param)
"""

import os
import json
import logging
from functools import wraps
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Load disabled tools from environment variable
_DISABLED_TOOLS_RAW = os.getenv("DISABLED_TOOLS", "")
_DISABLED_TOOLS = set()
_DISABLED_CATEGORIES = {}

def _parse_disabled_tools():
    """Parse DISABLED_TOOLS environment variable"""
    global _DISABLED_TOOLS, _DISABLED_CATEGORIES

    if not _DISABLED_TOOLS_RAW:
        return

    items = [item.strip() for item in _DISABLED_TOOLS_RAW.split(",") if item.strip()]

    for item in items:
        if ":" in item:
            # Category pattern: "todo:*" or "system:terminate_process"
            category, tool = item.split(":", 1)
            if tool == "*":
                _DISABLED_CATEGORIES[category] = "*"
            else:
                if category not in _DISABLED_CATEGORIES:
                    _DISABLED_CATEGORIES[category] = []
                if isinstance(_DISABLED_CATEGORIES[category], list):
                    _DISABLED_CATEGORIES[category].append(tool)
        else:
            # Simple tool name
            _DISABLED_TOOLS.add(item)

    if _DISABLED_TOOLS or _DISABLED_CATEGORIES:
        logger.info(f"🚫 Tool disabling active:")
        if _DISABLED_TOOLS:
            logger.info(f"   Disabled tools: {', '.join(_DISABLED_TOOLS)}")
        if _DISABLED_CATEGORIES:
            for cat, tools in _DISABLED_CATEGORIES.items():
                if tools == "*":
                    logger.info(f"   Disabled category: {cat}:* (all tools)")
                else:
                    logger.info(f"   Disabled {cat}: {', '.join(tools)}")

# Parse on module load
_parse_disabled_tools()


def is_tool_enabled(tool_name: str, category: Optional[str] = None) -> bool:
    """
    Check if a tool is enabled.

    Args:
        tool_name: Name of the tool to check
        category: Optional category (e.g., "todo", "system")

    Returns:
        True if tool is enabled, False if disabled

    Example:
        if not is_tool_enabled("add_todo_item", "todo"):
            return disabled_tool_response("add_todo_item")
    """
    # Check simple tool name
    if tool_name in _DISABLED_TOOLS:
        return False

    # Check category patterns
    if category:
        if category in _DISABLED_CATEGORIES:
            cat_rules = _DISABLED_CATEGORIES[category]
            if cat_rules == "*":
                # All tools in category disabled
                return False
            if tool_name in cat_rules:
                # Specific tool in category disabled
                return False

    return True


def disabled_tool_response(tool_name: str, reason: Optional[str] = None) -> str:
    """
    Return a standard response for disabled tools.

    Args:
        tool_name: Name of the disabled tool
        reason: Optional reason for disabling

    Returns:
        JSON string with error message

    Example:
        return disabled_tool_response("delete_all_todo_items", "Disabled for safety")
    """
    default_reason = f"Tool '{tool_name}' is currently disabled via DISABLED_TOOLS configuration"

    return json.dumps({
        "error": reason or default_reason,
        "tool": tool_name,
        "disabled": True,
        "message": "This tool has been disabled by the administrator. Check DISABLED_TOOLS environment variable."
    }, indent=2)


def check_tool_enabled(func: Callable = None, *, category: Optional[str] = None):
    """
    Decorator to automatically check if a tool is enabled.

    Args:
        func: Function to wrap (when used without arguments)
        category: Optional category for the tool

    Returns:
        Wrapped function that checks if tool is enabled

    Usage:
        @mcp.tool()
        @check_tool_enabled
        def my_tool(param: str) -> str:
            return do_work(param)

        # With category:
        @mcp.tool()
        @check_tool_enabled(category="todo")
        def add_todo_item(title: str) -> str:
            return add_todo(title)
    """
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            tool_name = f.__name__

            if not is_tool_enabled(tool_name, category):
                logger.warning(f"🚫 Tool '{tool_name}' called but is disabled")
                return disabled_tool_response(tool_name)

            return f(*args, **kwargs)

        @wraps(f)
        async def async_wrapper(*args, **kwargs):
            tool_name = f.__name__

            if not is_tool_enabled(tool_name, category):
                logger.warning(f"🚫 Tool '{tool_name}' called but is disabled")
                return disabled_tool_response(tool_name)

            return await f(*args, **kwargs)

        # Return appropriate wrapper based on whether function is async
        import asyncio
        if asyncio.iscoroutinefunction(f):
            return async_wrapper
        else:
            return wrapper

    # Handle both @check_tool_enabled and @check_tool_enabled(category="...")
    if func is None:
        return decorator
    else:
        return decorator(func)


def get_disabled_tools() -> dict:
    """
    Get information about disabled tools.

    Returns:
        Dict with disabled tools and categories

    Example:
        info = get_disabled_tools()
        print(f"Disabled: {info['tools']}")
    """
    return {
        "tools": list(_DISABLED_TOOLS),
        "categories": dict(_DISABLED_CATEGORIES),
        "raw": _DISABLED_TOOLS_RAW
    }


def filter_enabled_tools(tools: list, category: Optional[str] = None) -> list:
    """
    Filter a list of tools to only include enabled ones.

    Args:
        tools: List of tool names or tool objects
        category: Optional category for the tools

    Returns:
        Filtered list of enabled tools

    Example:
        all_tools = ["add_todo", "delete_all_todos", "list_todos"]
        enabled = filter_enabled_tools(all_tools, "todo")
        # Returns: ["add_todo", "list_todos"] if delete_all_todos is disabled
    """
    filtered = []

    for tool in tools:
        # Handle both tool names (str) and tool objects
        if isinstance(tool, str):
            tool_name = tool
        elif hasattr(tool, 'name'):
            tool_name = tool.name
        elif hasattr(tool, '__name__'):
            tool_name = tool.__name__
        else:
            # Unknown format, include it
            filtered.append(tool)
            continue

        # Check if enabled
        if is_tool_enabled(tool_name, category):
            filtered.append(tool)

    return filtered


__all__ = [
    'is_tool_enabled',
    'disabled_tool_response',
    'check_tool_enabled',
    'get_disabled_tools',
    'filter_enabled_tools'
]