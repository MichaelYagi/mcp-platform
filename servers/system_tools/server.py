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
from tools.tool_control import check_tool_enabled, is_tool_enabled, disabled_tool_response

from mcp.server.fastmcp import FastMCP
from tools.system import get_hardware_specs
from tools.system.system_info import get_system_stats
from tools.system.processes import list_processes, kill_process

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
logging.getLogger("mcp_system_tools_server").setLevel(logging.INFO)

logger = logging.getLogger("mcp_system_tools_server")
logger.info("🚀 Server logging initialized - writing to logs/mcp-server.log")

mcp = FastMCP("system-tools-server")

@mcp.tool()
@check_tool_enabled(category="system_tools")
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
    result = get_hardware_specs()
    return json.dumps(result, indent=2)


@mcp.tool()
@check_tool_enabled(category="system_tools")
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
    return get_system_stats()


@mcp.tool()
@check_tool_enabled(category="system_tools")
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
    top_n = int(top_n) if top_n is not None else 10
    logger.info(f"🛠 [server] list_system_processes called with top_n: {top_n}")
    return list_processes(top_n)


@mcp.tool()
@check_tool_enabled(category="system_tools")
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
    return kill_process(pid)

skill_registry = None

@mcp.tool()
@check_tool_enabled(category="system_tools")
def list_skills() -> str:
    """List all available skills for this server."""
    logger.info(f"🛠  list_skills called")
    if skill_registry is None:
        return json.dumps({
            "server": "system-tools-server",
            "skills": [],
            "message": "Skills not loaded"
        }, indent=2)

    return json.dumps({
        "server": "system-tools-server",
        "skills": skill_registry.list()
    }, indent=2)


@mcp.tool()
@check_tool_enabled(category="system_tools")
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
    loader = SkillLoader(server_tools, category="system_tools")
    skill_registry = loader.load_all(skills_dir)

    logger.info(f"🛠  {len(server_tools)} tools: {', '.join(server_tools)}")
    logger.info(f"🛠  {len(skill_registry.skills)} skills loaded")
    mcp.run(transport="stdio")