"""
GitHub MCP Server
Clones and reviews GitHub repositories using existing code tools
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
import os
import tempfile
import shutil
from typing import Optional, List, Dict, Any

from mcp.server.fastmcp import FastMCP
from tools.tool_control import check_tool_enabled

try:
    from client.tool_meta import tool_meta
except Exception:
    def tool_meta(**kwargs):
        def decorator(fn): return fn
        return decorator

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

try:
    import git
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False

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
logging.getLogger("mcp_github_server").setLevel(logging.INFO)

logger = logging.getLogger("mcp_github_server")
logger.info("🚀 GitHub server logging initialized")

mcp = FastMCP("github-server")

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")


def parse_github_url(url: str) -> Dict[str, str]:
    """
    Parse GitHub URL into components.

    Examples:
        https://github.com/user/repo → {owner: user, repo: repo, branch: None, path: None}
        https://github.com/user/repo/tree/main → {owner: user, repo: repo, branch: main, path: None}
        https://github.com/user/repo/blob/main/src/file.py → {owner: user, repo: repo, branch: main, path: src/file.py}
    """
    import re
    url = url.rstrip("/")
    pattern = r"github\.com/([^/]+)/([^/]+)"
    match = re.search(pattern, url)
    if not match:
        return {"error": f"Invalid GitHub URL: {url}"}

    owner = match.group(1)
    repo = match.group(2).replace(".git", "")
    branch = None
    path = None

    tree_pattern = r"/tree/([^/]+)(?:/(.+))?"
    blob_pattern = r"/blob/([^/]+)/(.+)"
    tree_match = re.search(tree_pattern, url)
    blob_match = re.search(blob_pattern, url)

    if blob_match:
        branch = blob_match.group(1)
        path = blob_match.group(2)
    elif tree_match:
        branch = tree_match.group(1)
        path = tree_match.group(2) if tree_match.group(2) else None

    return {
        "owner": owner,
        "repo": repo,
        "branch": branch or "main",
        "path": path,
        "full_name": f"{owner}/{repo}"
    }


@mcp.tool()
@check_tool_enabled(category="github")
@tool_meta(tags=["write","external","code"],triggers=["clone repo","clone github","download repo","get repo"],idempotent=False,example='use github_clone_repo: url="" [branch=""] [depth=""]',intent_category="github_review")
def github_clone_repo(
        url: str,
        branch: Optional[str] = None,
        depth: int = 1
) -> str:
    """
    Clone a GitHub repository to a temporary directory.

    Supports full repo URLs, branch URLs, and shallow clones.

    Args:
        url (str, required): GitHub repository URL
        branch (str, optional): Specific branch to clone (overrides URL branch)
        depth (int, optional): Clone depth (1=shallow, 0=full). Default: 1

    Returns:
        JSON with local_path, owner, repo, branch, files_count, size_mb.
    """
    logger.info(f"🛠 [server] github_clone_repo called: {url}")

    if not GIT_AVAILABLE:
        raise MCPToolError(FailureKind.USER_ERROR,
                           "GitPython not installed. Run: pip install gitpython",
                           {"tool": "github_clone_repo"})

    parsed = parse_github_url(url)
    if "error" in parsed:
        raise MCPToolError(FailureKind.USER_ERROR, parsed["error"],
                           {"tool": "github_clone_repo", "url": url})

    target_branch = branch or parsed["branch"]
    temp_dir = tempfile.mkdtemp(prefix=f"github_{parsed['repo']}_")
    clone_url = f"https://github.com/{parsed['full_name']}.git"
    if GITHUB_TOKEN:
        clone_url = f"https://{GITHUB_TOKEN}@github.com/{parsed['full_name']}.git"

    try:
        logger.info(f"📥 Cloning {parsed['full_name']} (branch: {target_branch}, depth: {depth})")
        if depth > 0:
            repo = git.Repo.clone_from(clone_url, temp_dir, branch=target_branch, depth=depth)
        else:
            repo = git.Repo.clone_from(clone_url, temp_dir, branch=target_branch)

        file_count = sum(1 for _ in Path(temp_dir).rglob("*") if _.is_file())
        size_bytes = sum(f.stat().st_size for f in Path(temp_dir).rglob("*") if f.is_file())
        size_mb = size_bytes / (1024 * 1024)

        logger.info(f"✅ Cloned: {file_count} files, {size_mb:.2f} MB")
        return json.dumps({
            "status": "success",
            "local_path": temp_dir,
            "owner": parsed["owner"],
            "repo": parsed["repo"],
            "branch": target_branch,
            "full_name": parsed["full_name"],
            "files_count": file_count,
            "size_mb": round(size_mb, 2),
            "url": url
        }, indent=2)

    except git.exc.GitCommandError as e:
        if Path(temp_dir).exists():
            shutil.rmtree(temp_dir)
        raise MCPToolError(FailureKind.UPSTREAM_ERROR, f"Clone failed: {e}",
                           {"tool": "github_clone_repo", "url": url, "branch": target_branch})
    except Exception as e:
        if Path(temp_dir).exists():
            shutil.rmtree(temp_dir)
        raise MCPToolError(FailureKind.INTERNAL_ERROR, f"Unexpected error: {e}",
                           {"tool": "github_clone_repo", "url": url})


@mcp.tool()
@check_tool_enabled(category="github")
@tool_meta(tags=["read","external","code"],triggers=["list files in repo","repo files","show repo files"],idempotent=True,example='use github_list_files: local_path="" [extensions=""] [exclude_dirs=""]',intent_category="github_review")
def github_list_files(
        local_path: str,
        extensions: Optional[List[str]] = None,
        exclude_dirs: Optional[List[str]] = None
) -> str:
    """
    List files in a cloned repository with optional filtering.

    Args:
        local_path (str, required): Path from github_clone_repo
        extensions (list, optional): Filter by extensions e.g. ["py", "js"]
        exclude_dirs (list, optional): Directories to skip (default: .git, node_modules, etc.)

    Returns:
        JSON with files, total_files, by_extension, total_size_mb.
    """
    logger.info(f"🛠 [server] github_list_files called: {local_path}")

    if not Path(local_path).exists():
        raise MCPToolError(FailureKind.USER_ERROR, f"Path not found: {local_path}",
                           {"tool": "github_list_files", "local_path": local_path})

    if exclude_dirs is None:
        exclude_dirs = [".git", "node_modules", "__pycache__", ".next", "dist", "build", "target"]

    all_files = []
    by_extension = {}
    total_size = 0

    for file_path in Path(local_path).rglob("*"):
        if not file_path.is_file():
            continue
        if any(excl in file_path.parts for excl in exclude_dirs):
            continue
        if extensions and file_path.suffix.lstrip(".") not in extensions:
            continue

        rel_path = str(file_path.relative_to(local_path))
        all_files.append(rel_path)
        ext = file_path.suffix.lstrip(".") or "no_extension"
        by_extension[ext] = by_extension.get(ext, 0) + 1
        total_size += file_path.stat().st_size

    logger.info(f"✅ Found {len(all_files)} files")
    return json.dumps({
        "local_path": local_path,
        "files": sorted(all_files),
        "total_files": len(all_files),
        "by_extension": by_extension,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "filters": {"extensions": extensions, "excluded_dirs": exclude_dirs}
    }, indent=2)


@mcp.tool()
@check_tool_enabled(category="github")
@tool_meta(tags=["read","external","code"],triggers=["get file from repo","read repo file","show file in repo"],idempotent=True,example='use github_get_file_content: local_path="" file_path=""',intent_category="github_review")
def github_get_file_content(
        local_path: str,
        file_path: str
) -> str:
    """
    Read the content of a specific file from a cloned repository.

    Args:
        local_path (str, required): Repository path from github_clone_repo
        file_path (str, required): Relative path to file (e.g., "src/main.py")

    Returns:
        JSON with file_path, content, size_bytes, lines.
    """
    logger.info(f"🛠 [server] github_get_file_content called: {file_path}")

    full_path = Path(local_path) / file_path
    if not full_path.exists():
        raise MCPToolError(FailureKind.USER_ERROR, f"File not found: {file_path}",
                           {"tool": "github_get_file_content", "file_path": file_path})

    try:
        content = full_path.read_text(encoding="utf-8")
        lines = content.count("\n") + 1
        logger.info(f"✅ Read {file_path} ({lines} lines)")
        return json.dumps({
            "file_path": file_path,
            "content": content,
            "size_bytes": full_path.stat().st_size,
            "lines": lines
        }, indent=2)
    except Exception as e:
        raise MCPToolError(FailureKind.INTERNAL_ERROR, f"Failed to read file: {e}",
                           {"tool": "github_get_file_content", "file_path": file_path})


@mcp.tool()
@check_tool_enabled(category="github")
@tool_meta(tags=["destructive","external","code"],triggers=["cleanup repo","delete cloned repo","remove repo"],idempotent=False,example='use github_cleanup_repo: local_path=""',intent_category="github_review")
def github_cleanup_repo(local_path: str) -> str:
    """
    Delete a cloned repository to free up disk space.

    Args:
        local_path (str, required): Path from github_clone_repo

    Returns:
        JSON with cleanup status.
    """
    logger.info(f"🛠 [server] github_cleanup_repo called: {local_path}")

    if not Path(local_path).exists():
        raise MCPToolError(FailureKind.USER_ERROR, f"Path not found: {local_path}",
                           {"tool": "github_cleanup_repo", "local_path": local_path})

    try:
        shutil.rmtree(local_path)
        logger.info(f"✅ Cleaned up: {local_path}")
        return json.dumps({
            "status": "success",
            "message": "Repository cleaned up",
            "path": local_path
        }, indent=2)
    except Exception as e:
        raise MCPToolError(FailureKind.INTERNAL_ERROR, f"Cleanup failed: {e}",
                           {"tool": "github_cleanup_repo", "local_path": local_path})


skill_registry = None


@mcp.tool()
@check_tool_enabled(category="github")
@tool_meta(tags=["read"],triggers=["list capabilities","what can you do"],idempotent=True,example="use list_capabilities",intent_category="github_review")
def list_capabilities(filter_tags: str | None = None) -> str:
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
        _tool_fn = getattr(_current, _name, None)
        if not (hasattr(_tool_fn, "__tool_meta__") or hasattr(_tool_fn, "_mcp_tool")):
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
@check_tool_enabled(category="github")
def list_skills() -> str:
    """List all available skills for this server."""
    logger.info("🛠  list_skills called")
    if skill_registry is None:
        return json.dumps({"server": "github-server", "skills": [], "message": "Skills not loaded"}, indent=2)
    return json.dumps({"server": "github-server", "skills": skill_registry.list()}, indent=2)


@mcp.tool()
@check_tool_enabled(category="github")
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
            if not name.startswith('_') and name not in ['get_tool_names_from_module', 'parse_github_url']:
                tool_names.append(name)
    return tool_names


if __name__ == "__main__":
    server_tools = get_tool_names_from_module()

    skills_dir = Path(__file__).parent / "skills"
    loader = SkillLoader(server_tools, category="github")
    skill_registry = loader.load_all(skills_dir)

    logger.info(f"🛠  {len(server_tools)} tools: {', '.join(server_tools)}")
    logger.info(f"🛠  {len(skill_registry.skills)} skills loaded")

    if not GIT_AVAILABLE:
        logger.warning("⚠️  GitPython not installed — run: pip install gitpython")
    if not GITHUB_TOKEN:
        logger.warning("⚠️  GITHUB_TOKEN not set — public repos only")
    else:
        logger.info("✅ GITHUB_TOKEN configured")

    mcp.run(transport="stdio")