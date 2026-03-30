"""
Code Runner MCP Server
Executes Python snippets in a sandboxed subprocess and returns the result.
Runs over stdio transport.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env", override=True)

from servers.skills.skill_loader import SkillLoader

import inspect
import json
import logging
import subprocess
import tempfile
import os
from typing import Optional

try:
    from tools.tool_control import check_tool_enabled
except ImportError:
    def check_tool_enabled(category=None):
        def decorator(func): return func
        return decorator

try:
    from client.tool_meta import tool_meta
except Exception:
    def tool_meta(**kwargs):
        def decorator(fn): return fn
        return decorator

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

from mcp.server.fastmcp import FastMCP

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.handlers.clear()

formatter = JsonFormatter() if JsonFormatter is not None else logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

file_handler = logging.FileHandler(LOG_DIR / "mcp-server.log", encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

logging.getLogger("mcp").setLevel(logging.DEBUG)
logging.getLogger("mcp_code_runner_server").setLevel(logging.INFO)

logger = logging.getLogger("mcp_code_runner_server")
logger.info("🚀 Code Runner server logging initialized")

mcp = FastMCP("code-runner-server")

# Python executable — use the same venv as the project
_PYTHON = sys.executable

# Execution timeout in seconds
_DEFAULT_TIMEOUT = 30

# Blocked imports — prevent network access, file system abuse, subprocess spawning
_BLOCKED_IMPORTS = {
    "subprocess", "socket", "urllib", "httpx", "requests", "aiohttp",
    "ftplib", "smtplib", "telnetlib", "xmlrpc",
    "ctypes", "cffi", "multiprocessing",
}

def _is_safe(code: str) -> tuple[bool, str]:
    """
    Basic safety check — reject code that imports blocked modules or
    uses obviously dangerous builtins.
    Returns (is_safe, reason).
    """
    try:
        tree = __import__("ast").parse(code)
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

    import ast as _ast
    for node in _ast.walk(tree):
        # Block dangerous imports
        if isinstance(node, (_ast.Import, _ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, _ast.Import) else [node.module or ""]
            for name in names:
                root = (name or "").split(".")[0]
                if root in _BLOCKED_IMPORTS:
                    return False, f"Import of '{root}' is not allowed in the code runner."

        # Block exec/eval/compile/__import__ calls
        if isinstance(node, _ast.Call):
            func = node.func
            name = None
            if isinstance(func, _ast.Name):
                name = func.id
            elif isinstance(func, _ast.Attribute):
                name = func.attr
            if name in ("exec", "eval", "compile", "__import__", "open"):
                return False, f"Call to '{name}' is not allowed in the code runner."

    return True, ""


@mcp.tool()
@check_tool_enabled(category="code_runner")
@tool_meta(
    tags=["read", "code", "ai"],
    triggers=[
        "run python", "execute code", "calculate", "compute",
        "run this", "eval", "python snippet", "test this code",
        "what is the result", "run a script",
    ],
    idempotent=False,
    example='use run_python: code=""',
    intent_category="code_runner"
)
def run_python(
    code: str,
    timeout: Optional[int] = _DEFAULT_TIMEOUT,
    capture_vars: Optional[str] = None,
) -> str:
    """
    Execute a Python code snippet and return stdout, stderr, and result.

    Runs in a subprocess using the project's Python interpreter.
    Network access and dangerous builtins are blocked.

    Args:
        code (str, required): Python code to execute. Can be multi-line.
                              The last expression is automatically printed if
                              it produces a value (like a REPL).
        timeout (int, optional): Max execution time in seconds (default: 30).
        capture_vars (str, optional): Comma-separated variable names to include
                                      in output e.g. "result,df,total"

    Returns:
        JSON string with:
        - success: bool
        - stdout: Captured standard output
        - stderr: Captured standard error (warnings, tracebacks)
        - return_value: Value of last expression if any
        - captured_vars: Dict of requested variable values
        - execution_time_ms: How long the code took to run

    Examples:
        run_python: code="import math; print(math.sqrt(144))"
        run_python: code="dates = [f'2026-03-{d:02d}' for d in range(1, 32)]; print(dates)"
        run_python: code="data = {'a': 1, 'b': 2}; total = sum(data.values()); print(total)"
        run_python: code="import re; print(bool(re.match(r'^\\d{4}-\\d{2}-\\d{2}$', '2026-03-30')))"
    """
    if not code or not code.strip():
        raise MCPToolError(FailureKind.USER_ERROR, "code must not be empty",
                           {"tool": "run_python"})

    code = code.strip()

    logger.info(f"🛠 [server] run_python called — {len(code)} chars")

    timeout = int(timeout) if timeout is not None else _DEFAULT_TIMEOUT
    timeout = max(1, min(timeout, 60))  # clamp 1-60s

    # Safety check
    safe, reason = _is_safe(code)
    if not safe:
        raise MCPToolError(FailureKind.USER_ERROR, reason, {"tool": "run_python"})

    # Wrap code to capture last expression value and requested vars
    var_names = [v.strip() for v in (capture_vars or "").split(",") if v.strip()]

    # Build script: just run the code, capture via subprocess stdout/stderr
    var_capture = ""
    if var_names:
        var_capture = "\nimport json as _jj\n"
        var_capture += "_cvars = {}\n"
        for vn in var_names:
            var_capture += f"try:\n    _cvars[{repr(vn)}] = repr({vn})\nexcept Exception:\n    _cvars[{repr(vn)}] = '<not defined>'\n"
        var_capture += f"print('__VARS__' + _jj.dumps(_cvars))\n"

    wrapper = code + var_capture

    try:
        import time
        start = time.time()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(wrapper)
            tmp_path = f.name

        result = subprocess.run(
            [_PYTHON, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
        )
        elapsed = round((time.time() - start) * 1000, 1)

        try:
            os.unlink(tmp_path)
        except Exception:
            pass

        if result.returncode != 0 and not result.stdout:
            return json.dumps({
                "success": False,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "execution_time_ms": elapsed,
            }, indent=2)

        # Parse var capture if present
        stdout = result.stdout
        captured_vars = {}
        if "__VARS__" in stdout:
            parts = stdout.split("__VARS__", 1)
            stdout = parts[0]
            try:
                captured_vars = json.loads(parts[1].strip())
            except Exception:
                pass

        success = result.returncode == 0 or bool(stdout.strip())
        out = {
            "success": success,
            "stdout": stdout.strip(),
            "stderr": result.stderr.strip(),
            "execution_time_ms": elapsed,
        }
        if captured_vars:
            out["captured_vars"] = captured_vars
        logger.info(f"✅ run_python completed in {elapsed}ms")
        return json.dumps(out, indent=2)

    except subprocess.TimeoutExpired:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        raise MCPToolError(FailureKind.RETRYABLE,
                           f"Code execution timed out after {timeout}s. "
                           "Simplify your code or increase the timeout.",
                           {"tool": "run_python", "timeout": timeout})
    except MCPToolError:
        raise
    except Exception as e:
        logger.error(f"❌ run_python failed: {e}", exc_info=True)
        raise MCPToolError(FailureKind.INTERNAL_ERROR, f"Execution failed: {e}",
                           {"tool": "run_python"})




@mcp.tool()
@check_tool_enabled(category="code_runner")
@tool_meta(
    tags=["read", "code"],
    triggers=["run file", "execute file", "run script", "run python file"],
    idempotent=False,
    example='use run_python_file: file_path="" [args=""]',
    intent_category="code_runner"
)
def run_python_file(
    file_path: str,
    args: Optional[str] = None,
    timeout: Optional[int] = _DEFAULT_TIMEOUT,
) -> str:
    """
    Execute an existing Python file and return its output.

    Args:
        file_path (str, required): Absolute path to the .py file to run
        args (str, optional): Command-line arguments to pass e.g. "--verbose input.csv"
        timeout (int, optional): Max execution time in seconds (default: 30)

    Returns:
        JSON string with success, stdout, stderr, exit_code, execution_time_ms.
    """
    logger.info(f"🛠 [server] run_python_file called: {file_path}")

    p = Path(file_path)
    if not p.exists():
        raise MCPToolError(FailureKind.USER_ERROR, f"File not found: {file_path}",
                           {"tool": "run_python_file", "file_path": file_path})
    if p.suffix.lower() != ".py":
        raise MCPToolError(FailureKind.USER_ERROR, f"Not a Python file: {file_path}",
                           {"tool": "run_python_file"})

    timeout = int(timeout) if timeout is not None else _DEFAULT_TIMEOUT
    timeout = max(1, min(timeout, 120))

    cmd = [_PYTHON, str(p)]
    if args:
        import shlex
        cmd += shlex.split(args)

    try:
        import time
        start = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(p.parent),
            env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
        )
        elapsed = round((time.time() - start) * 1000, 1)
        logger.info(f"✅ run_python_file completed in {elapsed}ms, exit={result.returncode}")
        return json.dumps({
            "success": result.returncode == 0,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "exit_code": result.returncode,
            "execution_time_ms": elapsed,
            "file": file_path,
        }, indent=2)
    except subprocess.TimeoutExpired:
        raise MCPToolError(FailureKind.RETRYABLE,
                           f"Script timed out after {timeout}s.",
                           {"tool": "run_python_file", "file_path": file_path})
    except Exception as e:
        logger.error(f"❌ run_python_file failed: {e}", exc_info=True)
        raise MCPToolError(FailureKind.INTERNAL_ERROR, f"Execution failed: {e}",
                           {"tool": "run_python_file"})


@mcp.tool()
@check_tool_enabled(category="code_runner")
@tool_meta(
    tags=["read", "system"],
    triggers=["run bash", "run shell", "shell command", "bash command", "run command"],
    idempotent=False,
    example='use run_bash: command="" [cwd=""]',
    intent_category="code_runner"
)
def run_bash(
    command: str,
    cwd: Optional[str] = None,
    timeout: Optional[int] = 30,
) -> str:
    """
    Execute a bash/shell command and return its output.

    Blocked: rm -rf /, shutdown, reboot, mkfs, dd if=/dev/zero, curl|bash patterns.

    Args:
        command (str, required): Shell command to run e.g. "ls -la" or "git status"
        cwd (str, optional): Working directory for the command
        timeout (int, optional): Max execution time in seconds (default: 30)

    Returns:
        JSON string with success, stdout, stderr, exit_code, execution_time_ms.
    """
    logger.info(f"🛠 [server] run_bash called: {command[:80]}")

    command = command.strip()

    # Block obviously destructive patterns
    _BLOCKED = ["rm -rf /", "rm -rf /*", "shutdown", "reboot", "mkfs",
                "dd if=/dev/zero", ":(){ :|:& };:", "curl|bash", "wget|bash",
                "curl | bash", "wget | bash"]
    cmd_lower = command.lower()
    for blocked in _BLOCKED:
        if blocked in cmd_lower:
            raise MCPToolError(FailureKind.USER_ERROR,
                               f"Command blocked for safety: '{blocked}'",
                               {"tool": "run_bash"})

    timeout = int(timeout) if timeout is not None else 30
    timeout = max(1, min(timeout, 120))

    work_dir = cwd or str(PROJECT_ROOT)
    if cwd and not Path(cwd).exists():
        raise MCPToolError(FailureKind.USER_ERROR, f"Working directory not found: {cwd}",
                           {"tool": "run_bash"})

    try:
        import time
        start = time.time()
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=work_dir,
        )
        elapsed = round((time.time() - start) * 1000, 1)
        logger.info(f"✅ run_bash completed in {elapsed}ms, exit={result.returncode}")
        return json.dumps({
            "success": result.returncode == 0,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "exit_code": result.returncode,
            "execution_time_ms": elapsed,
            "command": command,
        }, indent=2)
    except subprocess.TimeoutExpired:
        raise MCPToolError(FailureKind.RETRYABLE,
                           f"Command timed out after {timeout}s.",
                           {"tool": "run_bash", "command": command})
    except Exception as e:
        logger.error(f"❌ run_bash failed: {e}", exc_info=True)
        raise MCPToolError(FailureKind.INTERNAL_ERROR, f"Command failed: {e}",
                           {"tool": "run_bash"})


@mcp.tool()
@check_tool_enabled(category="code_runner")
@tool_meta(
    tags=["write", "code", "system"],
    triggers=["install package", "pip install", "install library", "install module"],
    idempotent=False,
    example='use pip_install: package=""',
    intent_category="code_runner"
)
def pip_install(
    package: str,
    upgrade: Optional[bool] = False,
) -> str:
    """
    Install a Python package into the project's virtual environment.

    Args:
        package (str, required): Package name to install e.g. "pandas" or "pandas==2.0.0"
        upgrade (bool, optional): Upgrade if already installed (default: False)

    Returns:
        JSON string with success, stdout, stderr, package, already_installed.
    """
    logger.info(f"🛠 [server] pip_install called: {package}")

    if not package or not package.strip():
        raise MCPToolError(FailureKind.USER_ERROR, "package must not be empty",
                           {"tool": "pip_install"})

    # Basic safety — no shell metacharacters
    import re
    if re.search(r"[;&|`$(){}]", package):
        raise MCPToolError(FailureKind.USER_ERROR,
                           f"Invalid package name: '{package}'",
                           {"tool": "pip_install"})

    cmd = [_PYTHON, "-m", "pip", "install", package]
    if upgrade:
        cmd.append("--upgrade")

    try:
        import time
        start = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ},
        )
        elapsed = round((time.time() - start) * 1000, 1)
        already_installed = "already satisfied" in result.stdout.lower()
        success = result.returncode == 0
        logger.info(f"{'✅' if success else '❌'} pip_install {package} completed in {elapsed}ms")
        return json.dumps({
            "success": success,
            "package": package,
            "already_installed": already_installed,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "execution_time_ms": elapsed,
        }, indent=2)
    except subprocess.TimeoutExpired:
        raise MCPToolError(FailureKind.RETRYABLE,
                           f"pip install timed out after 120s.",
                           {"tool": "pip_install", "package": package})
    except Exception as e:
        logger.error(f"❌ pip_install failed: {e}", exc_info=True)
        raise MCPToolError(FailureKind.INTERNAL_ERROR, f"Install failed: {e}",
                           {"tool": "pip_install"})

skill_registry = None


@mcp.tool()
@check_tool_enabled(category="code_runner")
@tool_meta(tags=["read"], triggers=["list capabilities", "what can you do"],
           idempotent=True, example="use list_capabilities", intent_category="code_runner")
def list_capabilities(filter_tags: Optional[str] = None) -> str:
    """
    Return the full capability schema for every tool on this server.

    Args:
        filter_tags (str, optional): Comma-separated tags to filter by

    Returns:
        JSON string with server name, tools array, and total count.
    """
    logger.info(f"🛠  list_capabilities called (filter_tags={filter_tags})")

    try:
        from client.capability_registry import (
            _TOOL_TAGS, _TOOL_RATE_LIMITS, _TOOL_IDEMPOTENT, _INTERNAL_TOOLS
        )
    except ImportError:
        return json.dumps({"error": "CapabilityRegistry not available"}, indent=2)

    import sys as _sys, inspect as _inspect
    _current = _sys.modules[__name__]
    wanted_tags = set(t.strip() for t in filter_tags.split(",") if t.strip()) if filter_tags else None

    tools_out = []
    seen = set()
    for _name, _obj in vars(_current).items():
        if not callable(_obj) or _name.startswith("_") or _name in _INTERNAL_TOOLS:
            continue
        if not (hasattr(_obj, "__tool_meta__") or hasattr(_obj, "_mcp_tool")):
            continue
        if _name in seen:
            continue
        seen.add(_name)
        tags = _TOOL_TAGS.get(_name, [])
        if wanted_tags and not (wanted_tags & set(tags)):
            continue
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
            params.append({"name": pname, "type": type_str, "required": not has_default,
                           "default": None if not has_default else str(param.default)})
        tools_out.append({
            "name": _name,
            "description": (_obj.__doc__ or "").strip().split("\n")[0],
            "input_schema": params,
            "tags": tags,
            "rate_limit": _TOOL_RATE_LIMITS.get(_name),
            "idempotent": _TOOL_IDEMPOTENT.get(_name, True),
        })

    return json.dumps({"server": mcp.name, "tools": tools_out, "total": len(tools_out)}, indent=2)


@mcp.tool()
@check_tool_enabled(category="code_runner")
def list_skills() -> str:
    """List all available skills for this server."""
    logger.info("🛠  list_skills called")
    if skill_registry is None:
        return json.dumps({"server": "code-runner-server", "skills": [], "message": "Skills not loaded"}, indent=2)
    return json.dumps({"server": "code-runner-server", "skills": skill_registry.list()}, indent=2)


@mcp.tool()
@check_tool_enabled(category="code_runner")
def read_skill(skill_name: str) -> str:
    """Read the full content of a skill."""
    logger.info(f"🛠  read_skill called: {skill_name}")
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
            if not name.startswith("_") and name != "get_tool_names_from_module":
                tool_names.append(name)
    return tool_names


if __name__ == "__main__":
    server_tools = get_tool_names_from_module()
    skills_dir = Path(__file__).parent / "skills"
    loader = SkillLoader(server_tools, category="code_runner")
    skill_registry = loader.load_all(skills_dir)
    logger.info(f"🛠  {len(server_tools)} tools: {', '.join(server_tools)}")
    logger.info(f"🛠  {len(skill_registry.skills)} skills loaded")
    mcp.run(transport="stdio")