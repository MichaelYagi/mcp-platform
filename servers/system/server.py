"""
System Tools MCP Server
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
from tools.system import get_hardware_specs
from tools.system.system_info import get_system_stats
from tools.system.processes import list_processes, kill_process

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
logging.getLogger("mcp_system_server").setLevel(logging.INFO)

logger = logging.getLogger("mcp_system_server")
logger.info("🚀 Server logging initialized - writing to logs/mcp-server.log")

mcp = FastMCP("system-server")

@mcp.tool()
@check_tool_enabled(category="system")
@tool_meta(tags=["read","system"],triggers=["hardware","specs","cpu","gpu","ram","system specs"],idempotent=True,example="use get_hardware_specs_tool")
def get_hardware_specs_tool() -> str:
    """
    Get detailed hardware specifications including CPU, GPU, and RAM.

    Args:
        None

    Returns:
        JSON string with:
        - cpu: {model, cores, threads, frequency}
        - gpu: [{name, vram, driver_version}] (array of GPUs)
        - ram: {total_gb, type, speed_mhz}
        - platform: Operating system name

    Works across Windows, Linux, and macOS.

    Use when user asks about hardware specs, system specs, CPU, GPU, graphics card, or RAM.
    """
    logger.info(f"🛠 [server] get_hardware_specs_tool called")
    try:
        result = get_hardware_specs()
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"❌ get_hardware_specs_tool failed: {e}", exc_info=True)
        raise MCPToolError(FailureKind.UPSTREAM_ERROR, f"Hardware specs unavailable: {e}", {"tool": "get_hardware_specs_tool"})


@mcp.tool()
@check_tool_enabled(category="system")
@tool_meta(tags=["read","system"],triggers=["system info","os info","hostname","uptime"],idempotent=False,example="use get_system_info")
def get_system_info() -> str:
    """
    Retrieve current system health and resource usage.

    Args:
        None

    Returns:
        JSON string with:
        - os: {name, version, architecture}
        - cpu: {usage_percent, load_average}
        - memory: {total_gb, used_gb, available_gb, percent_used}
        - disk: {total_gb, used_gb, free_gb, percent_used}
        - uptime: System uptime in seconds

    Use when user asks about system performance, diagnostics, or machine status.
    """
    logger.info(f"🛠 [server] get_system_info called")
    try:
        return get_system_stats()
    except Exception as e:
        logger.error(f"❌ get_system_info failed: {e}", exc_info=True)
        raise MCPToolError(FailureKind.UPSTREAM_ERROR, f"System stats unavailable: {e}", {"tool": "get_system_info"})


@mcp.tool()
@check_tool_enabled(category="system")
@tool_meta(tags=["read","system"],triggers=["processes","running processes","what is running","top processes"],idempotent=False,example='use list_system_processes [top_n=""]')
def list_system_processes(top_n: Optional[int] = 10) -> str:
    """
    List active system processes sorted by resource usage.

    Args:
        top_n (int, optional): Number of top processes to return (default: 10)

    Returns:
        JSON string with array of processes, each containing:
        - pid: Process ID
        - name: Process name
        - cpu_percent: CPU usage percentage
        - memory_percent: RAM usage percentage
        - status: Process status (running, sleeping, etc.)

    Use when user asks what is running or wants to inspect system activity.
    """
    try:
        top_n = int(top_n) if top_n is not None else 10
    except (TypeError, ValueError) as e:
        raise MCPToolError(FailureKind.USER_ERROR, f"Invalid top_n value: {top_n}", {"tool": "list_system_processes", "param": "top_n"})
    logger.info(f"🛠 [server] list_system_processes called with top_n: {top_n}")
    try:
        return list_processes(top_n)
    except Exception as e:
        logger.error(f"❌ list_system_processes failed: {e}", exc_info=True)
        raise MCPToolError(FailureKind.UPSTREAM_ERROR, f"Process listing failed: {e}", {"tool": "list_system_processes"})


@mcp.tool()
@check_tool_enabled(category="system")
@tool_meta(tags=["write","destructive","system"],triggers=["kill process","terminate","stop process"],idempotent=False,example='use terminate_process: pid=""')
def terminate_process(pid: int) -> str:
    """
    Terminate a process by its process ID (PID).

    Args:
        pid (int, required): The process ID to terminate

    Returns:
        JSON string with:
        - success: Boolean indicating if termination succeeded
        - pid: The process ID that was terminated
        - message: Confirmation or error message

    Use when user explicitly requests to stop or kill a specific process.
    """
    logger.info(f"🛠 [server] terminate_process called with pid: {pid}")
    try:
        return kill_process(pid)
    except ProcessLookupError as e:
        raise MCPToolError(FailureKind.USER_ERROR, f"Process {pid} not found", {"tool": "terminate_process", "pid": pid})
    except PermissionError as e:
        raise MCPToolError(FailureKind.USER_ERROR, f"Permission denied to terminate process {pid}", {"tool": "terminate_process", "pid": pid})
    except Exception as e:
        logger.error(f"❌ terminate_process failed: {e}", exc_info=True)
        raise MCPToolError(FailureKind.INTERNAL_ERROR, f"Failed to terminate process {pid}: {e}", {"tool": "terminate_process", "pid": pid})

skill_registry = None

@mcp.tool()
@check_tool_enabled(category="system")
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
@check_tool_enabled(category="system")
def list_skills() -> str:
    """List all available skills for this server."""
    logger.info(f"🛠  list_skills called")
    if skill_registry is None:
        return json.dumps({
            "server": "system-server",
            "skills": [],
            "message": "Skills not loaded"
        }, indent=2)

    return json.dumps({
        "server": "system-server",
        "skills": skill_registry.list()
    }, indent=2)


@mcp.tool()
@check_tool_enabled(category="system")
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
    server_tools = get_tool_names_from_module()

    skills_dir = Path(__file__).parent / "skills"
    loader = SkillLoader(server_tools, category="system")
    skill_registry = loader.load_all(skills_dir)

    logger.info(f"🛠  {len(server_tools)} tools: {', '.join(server_tools)}")
    logger.info(f"🛠  {len(skill_registry.skills)} skills loaded")
    mcp.run(transport="stdio")