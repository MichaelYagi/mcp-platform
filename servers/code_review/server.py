"""
Code Review MCP Server with comprehensive code review
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
from tools.tool_control import check_tool_enabled
try:
    from client.tool_meta import tool_meta
except Exception:
    def tool_meta(**kwargs):
        def decorator(fn): return fn
        return decorator

from mcp.server.fastmcp import FastMCP
from tools.code_review.fix_bug import fix_bug
from tools.code_review.search_code import search_code
from tools.code_review.review_code import review_python_file

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
logging.getLogger("mcp_code_review_server").setLevel(logging.INFO)

logger = logging.getLogger("mcp_code_review_server")
logger.info("🚀 Server logging initialized - writing to logs/mcp-server.log")

mcp = FastMCP("code-review-server")


@mcp.tool()
@check_tool_enabled(category="code_reviewer")
@tool_meta(tags=["read","code","ai"],triggers=["review code","check code","audit code","code quality"],idempotent=True,example='use review_code: path="" [max_bytes=""]',intent_category="code")
def review_code(path: str, max_bytes: Optional[int] = 200_000) -> str:
    """
    Perform comprehensive code review and static analysis on a Python file or directory.

    This tool analyzes Python code for:
    - Security vulnerabilities (hardcoded credentials, eval/exec, SQL injection)
    - Code quality issues (missing docstrings, long functions, complexity)
    - Exception handling problems (bare except, silent failures)
    - Performance concerns (nested comprehensions, global variables)
    - Syntax errors and type issues

    Can review:
    - Single Python file: Detailed analysis of one file
    - Directory: Analyzes all .py files in directory (non-recursive)

    Args:
        path (str, required): Absolute or relative path to Python file or directory
        max_bytes (int, optional): Maximum file size to process per file (default: 200,000)

    Returns:
        JSON string with detailed analysis including metrics, issues by severity,
        and recommendations.

    Use when user wants to review code for quality, security, or get improvement suggestions.
    """
    try:
        max_bytes = int(max_bytes) if max_bytes is not None else 200_000
    except (TypeError, ValueError):
        raise MCPToolError(FailureKind.USER_ERROR, f"Invalid max_bytes value: {max_bytes}",
                           {"tool": "review_code", "param": "max_bytes"})

    logger.info(f"🛠 [server] review_code called with path: {path}, max_bytes: {max_bytes}")

    target = Path(path)
    if not target.exists():
        raise MCPToolError(FailureKind.USER_ERROR, f"Path does not exist: {path}",
                           {"tool": "review_code", "path": path})

    try:
        result = review_python_file(path, max_bytes)
        return json.dumps(result, indent=2)
    except MCPToolError:
        raise
    except Exception as e:
        logger.error(f"❌ review_code failed: {str(e)}", exc_info=True)
        raise MCPToolError(FailureKind.INTERNAL_ERROR, f"Review failed: {e}",
                           {"tool": "review_code", "path": path})


@mcp.tool()
@check_tool_enabled(category="code_reviewer")
@tool_meta(tags=["read","search","code"],triggers=["search code","find in code","grep code","search codebase"],idempotent=True,example='use search_code_in_directory: query="" [extension=""] [directory=""]',intent_category="code")
def search_code_in_directory(
        query: str,
        extension: Optional[str] = None,
        directory: Optional[str] = "."
) -> str:
    """
    Search source code for text or regex patterns across multiple files.

    Args:
        query (str, required): Text or regex pattern to find (e.g., 'class Weather', 'to-do')
        extension (str, optional): Filter by file type (e.g., 'py', 'js', 'java')
        directory (str, optional): Starting folder path (default: current directory)

    Returns:
        JSON string with matches, total_matches, and files_searched.

    Use when user wants to locate code, patterns, class definitions, function calls,
    or text references across a codebase.
    """
    if not query or not query.strip():
        raise MCPToolError(FailureKind.USER_ERROR, "query must not be empty",
                           {"tool": "search_code_in_directory"})

    logger.info(f"🛠 [server] search_code_in_directory called with query: {query}, "
                f"extension: {extension}, directory: {directory}")
    try:
        result = search_code(query, extension, directory)
        return json.dumps(result, indent=2)
    except MCPToolError:
        raise
    except Exception as e:
        logger.error(f"❌ search_code_in_directory failed: {e}", exc_info=True)
        raise MCPToolError(FailureKind.INTERNAL_ERROR, f"Code search failed: {e}",
                           {"tool": "search_code_in_directory", "query": query})


@mcp.tool()
@check_tool_enabled(category="code_reviewer")
@tool_meta(tags=["read","code","ai"],triggers=["debug","fix bug","fix error","analyze error"],idempotent=True,example='use debug_fix: error_message="" [stack_trace=""] [code_snippet=""] [environment=""]',intent_category="code")
def debug_fix(error_message: str,
              stack_trace: Optional[str] = None,
              code_snippet: Optional[str] = None,
              environment: Optional[str] = None) -> str:
    """
    Analyze a bug and propose fixes with root cause analysis.

    Args:
        error_message (str, required): The error message or exception text
        stack_trace (str, optional): Full stack trace if available
        code_snippet (str, optional): Relevant code that caused the error
        environment (str, optional): Environment details (OS, language version, etc.)

    Returns:
        JSON string with error_type, likely_causes, suggested_fixes, references,
        and severity.

    Use when user wants help diagnosing, debugging, or fixing code issues.
    """
    if not error_message or not error_message.strip():
        raise MCPToolError(FailureKind.USER_ERROR, "error_message must not be empty",
                           {"tool": "debug_fix"})

    logger.info(f"🛠 [server] debug_fix called with error_message: {error_message}, "
                f"stack_trace: {stack_trace}, code_snippet: {code_snippet}, "
                f"environment: {environment}")
    try:
        result = fix_bug(
            error_message=error_message,
            stack_trace=stack_trace,
            code_snippet=code_snippet,
            environment=environment
        )
        return json.dumps(result, indent=2)
    except MCPToolError:
        raise
    except Exception as e:
        logger.error(f"❌ debug_fix failed: {e}", exc_info=True)
        raise MCPToolError(FailureKind.INTERNAL_ERROR, f"Bug analysis failed: {e}",
                           {"tool": "debug_fix", "error_message": error_message})


skill_registry = None


@mcp.tool()
@check_tool_enabled(category="code_reviewer")
@tool_meta(tags=["read"],triggers=["list capabilities","what can you do"],idempotent=True,example="use list_capabilities",intent_category="code")
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
        JSON string with server name, tools array, and total count.
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
        _tool_fn = getattr(_current, _name, None)
        if not (hasattr(_tool_fn, "__tool_meta__") or hasattr(_tool_fn, "_mcp_tool")):
            continue
        if _name in seen:
            continue
        seen.add(_name)

        tags = _TOOL_TAGS.get(_name, [])
        if wanted_tags and not (wanted_tags & set(tags)):
            continue

        import inspect as _inspect
        sig = _inspect.signature(_obj)
        params = []
        for pname, param in sig.parameters.items():
            if pname == "self":
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
@check_tool_enabled(category="code_reviewer")
def list_skills() -> str:
    """List all available skills for this server."""
    logger.info(f"🛠  list_skills called")
    if skill_registry is None:
        return json.dumps({"server": "code-review-server", "skills": [], "message": "Skills not loaded"}, indent=2)
    return json.dumps({"server": "code-review-server", "skills": skill_registry.list()}, indent=2)


@mcp.tool()
@check_tool_enabled(category="code_reviewer")
def read_skill(skill_name: str) -> str:
    """Read the full content of a skill."""
    logger.info(f"🛠  read_skill called")
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
    loader = SkillLoader(server_tools, category="code_reviewer")
    skill_registry = loader.load_all(skills_dir)
    logger.info(f"🛠  {len(server_tools)} tools: {', '.join(server_tools)}")
    logger.info(f"🛠  {len(skill_registry.skills)} skills loaded")
    mcp.run(transport="stdio")