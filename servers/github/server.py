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

# Try to import git (install with: pip install gitpython)
try:
    import git

    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False
    print("⚠️  GitPython not installed. Run: pip install gitpython")

# Setup logging
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.handlers.clear()

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

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

# GitHub token from environment
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")


def parse_github_url(url: str) -> Dict[str, str]:
    """
    Parse GitHub URL into components

    Examples:
        https://github.com/user/repo → {owner: user, repo: repo, branch: None, path: None}
        https://github.com/user/repo/tree/main → {owner: user, repo: repo, branch: main, path: None}
        https://github.com/user/repo/blob/main/src/file.py → {owner: user, repo: repo, branch: main, path: src/file.py}
    """
    import re

    # Remove trailing slashes
    url = url.rstrip("/")

    # Basic pattern: github.com/owner/repo
    pattern = r"github\.com/([^/]+)/([^/]+)"
    match = re.search(pattern, url)

    if not match:
        return {"error": f"Invalid GitHub URL: {url}"}

    owner = match.group(1)
    repo = match.group(2).replace(".git", "")

    # Check for branch/path
    branch = None
    path = None

    # Pattern: /tree/branch or /blob/branch/path
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
def github_clone_repo(
        url: str,
        branch: Optional[str] = None,
        depth: int = 1
) -> str:
    """
    Clone a GitHub repository to temporary directory.

    Supports:
    - Full repo URLs: https://github.com/user/repo
    - Branch URLs: https://github.com/user/repo/tree/branch
    - Shallow clones (default depth=1 for speed)

    Args:
        url (str): GitHub repository URL
        branch (str, optional): Specific branch to clone (overrides URL branch)
        depth (int, optional): Clone depth (1=shallow, 0=full). Default: 1

    Returns:
        JSON with:
        - local_path: Path where repo was cloned
        - owner: Repository owner
        - repo: Repository name
        - branch: Checked out branch
        - files_count: Number of files cloned
        - size_mb: Approximate size in MB

    Use cases:
        - "Clone https://github.com/anthropics/anthropic-sdk-python"
        - "Download https://github.com/user/repo/tree/develop"
        - "Get https://github.com/user/repo for review"
    """
    logger.info(f"🛠  github_clone_repo called: {url}")

    if not GIT_AVAILABLE:
        return json.dumps({
            "error": "GitPython not installed",
            "install": "pip install gitpython"
        }, indent=2)

    # Parse URL
    parsed = parse_github_url(url)
    if "error" in parsed:
        return json.dumps(parsed, indent=2)

    # Use provided branch or parsed branch
    target_branch = branch or parsed["branch"]

    # Create temp directory
    temp_dir = tempfile.mkdtemp(prefix=f"github_{parsed['repo']}_")
    clone_url = f"https://github.com/{parsed['full_name']}.git"

    # Add token if available
    if GITHUB_TOKEN:
        clone_url = f"https://{GITHUB_TOKEN}@github.com/{parsed['full_name']}.git"

    try:
        logger.info(f"📥 Cloning {parsed['full_name']} (branch: {target_branch}, depth: {depth})")

        # Clone with options
        if depth > 0:
            repo = git.Repo.clone_from(
                clone_url,
                temp_dir,
                branch=target_branch,
                depth=depth
            )
        else:
            repo = git.Repo.clone_from(
                clone_url,
                temp_dir,
                branch=target_branch
            )

        # Count files
        file_count = sum(1 for _ in Path(temp_dir).rglob("*") if _.is_file())

        # Estimate size
        size_bytes = sum(f.stat().st_size for f in Path(temp_dir).rglob("*") if f.is_file())
        size_mb = size_bytes / (1024 * 1024)

        result = {
            "status": "success",
            "local_path": temp_dir,
            "owner": parsed["owner"],
            "repo": parsed["repo"],
            "branch": target_branch,
            "full_name": parsed["full_name"],
            "files_count": file_count,
            "size_mb": round(size_mb, 2),
            "url": url
        }

        logger.info(f"✅ Cloned successfully: {file_count} files, {size_mb:.2f} MB")
        return json.dumps(result, indent=2)

    except git.exc.GitCommandError as e:
        logger.error(f"❌ Git clone failed: {e}")
        # Cleanup on failure
        if Path(temp_dir).exists():
            shutil.rmtree(temp_dir)

        return json.dumps({
            "error": "Clone failed",
            "message": str(e),
            "url": url,
            "branch": target_branch,
            "suggestions": [
                "Check if repository exists",
                "Verify branch name",
                "Add GITHUB_TOKEN to .env for private repos"
            ]
        }, indent=2)


@mcp.tool()
def github_list_files(
        local_path: str,
        extensions: Optional[List[str]] = None,
        exclude_dirs: Optional[List[str]] = None
) -> str:
    """
    List files in cloned repository with filtering.

    Args:
        local_path (str): Path from github_clone_repo
        extensions (list, optional): Filter by extensions (e.g., ["py", "js", "ts"])
        exclude_dirs (list, optional): Directories to skip (default: [".git", "node_modules", "__pycache__"])

    Returns:
        JSON with:
        - files: List of file paths relative to repo root
        - total_files: Total count
        - by_extension: Count grouped by extension
        - total_size_mb: Total size

    Use cases:
        - "List Python files in cloned repo"
        - "Show all JavaScript files"
        - "What files are in this repo?"
    """
    logger.info(f"🛠  github_list_files called: {local_path}")

    if not Path(local_path).exists():
        return json.dumps({"error": f"Path not found: {local_path}"}, indent=2)

    # Default exclusions
    if exclude_dirs is None:
        exclude_dirs = [".git", "node_modules", "__pycache__", ".next", "dist", "build", "target"]

    # Find all files
    all_files = []
    by_extension = {}
    total_size = 0

    for file_path in Path(local_path).rglob("*"):
        if not file_path.is_file():
            continue

        # Check if in excluded directory
        if any(excl in file_path.parts for excl in exclude_dirs):
            continue

        # Check extension filter
        if extensions:
            if file_path.suffix.lstrip(".") not in extensions:
                continue

        # Get relative path
        rel_path = str(file_path.relative_to(local_path))
        all_files.append(rel_path)

        # Count by extension
        ext = file_path.suffix.lstrip(".") or "no_extension"
        by_extension[ext] = by_extension.get(ext, 0) + 1

        # Track size
        total_size += file_path.stat().st_size

    result = {
        "local_path": local_path,
        "files": sorted(all_files),
        "total_files": len(all_files),
        "by_extension": by_extension,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "filters": {
            "extensions": extensions,
            "excluded_dirs": exclude_dirs
        }
    }

    logger.info(f"✅ Found {len(all_files)} files")
    return json.dumps(result, indent=2)


@mcp.tool()
def github_get_file_content(
        local_path: str,
        file_path: str
) -> str:
    """
    Read content of a specific file from cloned repo.

    Args:
        local_path (str): Repository path from github_clone_repo
        file_path (str): Relative path to file (e.g., "src/main.py")

    Returns:
        JSON with:
        - file_path: Path to file
        - content: File contents
        - size_bytes: File size
        - lines: Line count

    Use when reviewing specific files in detail.
    """
    logger.info(f"🛠  github_get_file_content called: {file_path}")

    full_path = Path(local_path) / file_path

    if not full_path.exists():
        return json.dumps({"error": f"File not found: {file_path}"}, indent=2)

    try:
        content = full_path.read_text(encoding="utf-8")
        lines = content.count("\\n") + 1

        return json.dumps({
            "file_path": file_path,
            "content": content,
            "size_bytes": full_path.stat().st_size,
            "lines": lines
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "error": f"Failed to read file: {str(e)}",
            "file_path": file_path
        }, indent=2)


@mcp.tool()
def github_cleanup_repo(local_path: str) -> str:
    """
    Delete cloned repository to free up space.

    Args:
        local_path (str): Path from github_clone_repo

    Returns:
        JSON with cleanup status

    Use after review is complete to clean up temporary files.
    """
    logger.info(f"🛠  github_cleanup_repo called: {local_path}")

    if not Path(local_path).exists():
        return json.dumps({"error": "Path not found", "path": local_path}, indent=2)

    try:
        shutil.rmtree(local_path)
        logger.info(f"✅ Cleaned up: {local_path}")

        return json.dumps({
            "status": "success",
            "message": "Repository cleaned up",
            "path": local_path
        }, indent=2)

    except Exception as e:
        logger.error(f"❌ Cleanup failed: {e}")
        return json.dumps({
            "error": "Cleanup failed",
            "message": str(e),
            "path": local_path
        }, indent=2)


# Skill management
skill_registry = None


@mcp.tool()
def list_skills() -> str:
    """List all available skills for GitHub server."""
    logger.info("📚 list_skills called")
    if skill_registry is None:
        return json.dumps({
            "server": "github-server",
            "skills": [],
            "message": "Skills not loaded"
        }, indent=2)

    return json.dumps({
        "server": "github-server",
        "skills": skill_registry.list()
    }, indent=2)


@mcp.tool()
def read_skill(skill_name: str) -> str:
    """Read the full content of a skill."""
    logger.info(f"📖 read_skill called: {skill_name}")

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
    """Auto-discover tools from this module"""
    current_module = sys.modules[__name__]
    tool_names = []

    for name, obj in inspect.getmembers(current_module):
        if inspect.isfunction(obj) and obj.__module__ == __name__:
            if not name.startswith('_') and name != 'get_tool_names_from_module':
                tool_names.append(name)

    return tool_names


if __name__ == "__main__":
    # Auto-discover tools
    server_tools = get_tool_names_from_module()

    # Load skills
    skills_dir = Path(__file__).parent / "skills"
    loader = SkillLoader(server_tools, category="github_server")
    skill_registry = loader.load_all(skills_dir)

    logger.info(f"🛠️  {len(server_tools)} tools: {', '.join(server_tools)}")
    logger.info(f"📚 {len(skill_registry.skills)} skills loaded")

    if not GIT_AVAILABLE:
        logger.warning("⚠️  GitPython not installed - install with: pip install gitpython")

    if not GITHUB_TOKEN:
        logger.warning("⚠️  GITHUB_TOKEN not set - public repos only")
    else:
        logger.info("✅ GITHUB_TOKEN configured")

    mcp.run(transport="stdio")