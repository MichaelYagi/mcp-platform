"""
Image Review MCP Server
Runs over stdio transport
"""
import sys
import os
import re
import base64
import inspect
import json
import logging
import requests
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env", override=True)

from servers.skills.skill_loader import SkillLoader
from tools.tool_control import check_tool_enabled

from mcp.server.fastmcp import FastMCP

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
logging.getLogger("mcp_image_review_server").setLevel(logging.INFO)

logger = logging.getLogger("mcp_image_review_server")
logger.info("🚀 Image Review Server logging initialized - writing to logs/mcp-server.log")

mcp = FastMCP("image-review-server")

SHASHIN_BASE_URL = os.getenv("SHASHIN_BASE_URL", "http://192.168.0.199:6624")
SHASHIN_API_KEY  = os.getenv("SHASHIN_API_KEY", "")

_SHASHIN_HEADERS = {
    "Content-Type": "application/json",
    "x-api-key": SHASHIN_API_KEY,
}


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_file_path(file_path: str) -> Path:
    """
    Resolve a file path to an absolute Path, handling:
    - Linux/WSL paths  (/home/..., /mnt/c/...)
    - Windows paths    (C:\\Users\\... or C:/Users/...)
    - Home paths       (~/...)
    """
    win_match = re.match(r'^([A-Za-z]):[/\\](.*)', file_path)
    if win_match:
        drive = win_match.group(1).lower()
        rest  = win_match.group(2).replace('\\', '/')
        file_path = f"/mnt/{drive}/{rest}"
    return Path(file_path).expanduser().resolve()


def _fetch_image_as_base64(image_url: str) -> dict:
    """
    Fetch any image URL and return it base64-encoded.
    Automatically applies Shashin auth headers for local Shashin URLs.
    """
    try:
        headers = {}
        if SHASHIN_API_KEY and (
            "192.168." in image_url or "shashin" in image_url.lower()
        ):
            headers = _SHASHIN_HEADERS

        resp = requests.get(image_url, headers=headers, timeout=30)
        resp.raise_for_status()
        encoded = base64.b64encode(resp.content).decode("utf-8")
        logger.info(f"[fetch_image] {image_url} — {len(resp.content)} bytes encoded")
        return {"success": True, "image_base64": encoded}

    except Exception as exc:
        logger.error(f"[fetch_image] Failed to fetch {image_url}: {exc}")
        return {"success": False, "error": f"Could not fetch image: {exc}"}


def _read_image_file_as_base64(file_path: str) -> dict:
    """
    Read a local image file and return it base64-encoded.
    Handles Linux, WSL, Windows, and home (~) paths.
    """
    try:
        resolved = _resolve_file_path(file_path)
        if not resolved.exists():
            return {"success": False, "error": f"File not found: {resolved}"}
        if not resolved.is_file():
            return {"success": False, "error": f"Path is not a file: {resolved}"}

        with open(resolved, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")

        logger.info(f"[read_image_file] {resolved} encoded successfully")
        return {"success": True, "image_base64": encoded}

    except Exception as exc:
        logger.error(f"[read_image_file] Failed to read {file_path}: {exc}")
        return {"success": False, "error": f"Could not read image file: {exc}"}


def _shashin_get(path: str) -> dict:
    """Thin GET wrapper for the Shashin API."""
    url = f"{SHASHIN_BASE_URL}{path}"
    try:
        resp = requests.get(url, headers=_SHASHIN_HEADERS, timeout=15)
        resp.raise_for_status()
        return {"ok": True, "data": resp.json()}
    except requests.exceptions.HTTPError as exc:
        return {"ok": False, "error": f"HTTP {exc.response.status_code}: {exc.response.text[:200]}"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# analyze_image_tool
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
@check_tool_enabled(category="image_tools")
def analyze_image_tool(
    image_url: Optional[str] = None,
    image_file_path: Optional[str] = None,
    image_base64: Optional[str] = None,
) -> str:
    """
    Fetch and base64-encode any image for vision inference.

    Accepts an image from a URL, a local file path, or a pre-encoded base64
    string. Returns the base64 payload to the LangGraph agent, which passes
    it to Ollama for vision inference.

    IMPORTANT: The active Ollama model must support vision inputs.
    If Ollama rejects the image, relay the error to the user and ask them
    to switch to a vision-capable model such as "qwen2.5vl:7b" or "llava:7b".

    Args:
        image_url (str, optional):       Any HTTP(S) URL pointing to an image
        image_file_path (str, optional): Local file path to an image.
                                         Supports Linux (/home/..., /mnt/c/...),
                                         Windows (C:\\Users\\...), and home (~/) paths.
        image_base64 (str, optional):    Pre-encoded base64 image string (returned as-is)

    Provide exactly one of image_url, image_file_path, or image_base64.

    Returns:
        JSON string with:
        - success (bool)
        - image_base64 (str) — base64-encoded image, pass this to Ollama
        - error (str)        — human-readable message on failure
    """
    logger.info(
        f"🛠 [server] analyze_image_tool called — "
        f"url={image_url}, file_path={image_file_path}, has_base64={bool(image_base64)}"
    )

    if not image_url and not image_file_path and not image_base64:
        return json.dumps({
            "success": False,
            "error": "Provide one of: image_url, image_file_path, or image_base64.",
        })

    if image_base64:
        return json.dumps({"success": True, "image_base64": image_base64})

    if image_file_path:
        result = _read_image_file_as_base64(image_file_path)
        if result.get("success"):
            result["image_source"] = image_file_path
        return json.dumps(result, indent=2)

    # URL case — return image_source so langgraph/websocket fetch it
    # directly rather than embedding a large base64 blob in the tool result.
    return json.dumps({"success": True, "image_source": image_url}, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# shashin_search_tool
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
@check_tool_enabled(category="image_tools")
def shashin_search_tool(
    term: str,
    page: int = 0,
) -> str:
    """
    Search the Shashin self-hosted media gallery by keyword term.

    Calls the Shashin search endpoint and returns matching image metadata
    including thumbnail/original URLs, camera info, and detected keywords.
    Use the returned image IDs with shashin_analyze_tool for vision inference.

    Args:
        term (str, required): Search term e.g. "cat", "Noah", "beach", "sunset"
        page (int, optional): Pagination page index, 0-based (default: 0)

    Returns:
        JSON string with:
        - success (bool)
        - term (str)           — echoed search term
        - count (int)          — number of results on this page
        - page (int)           — current page
        - total_pages (int)    — total pages available
        - results (list)       — image metadata objects, each with:
            - id (str)           — UUID, pass to shashin_analyze_tool
            - fileName (str)
            - takenAt (str)
            - camera (str)
            - keywords (str)     — detected object tags from Shashin AI
            - thumbnailUrl (str) — full URL to 225px thumbnail
            - originalUrl (str)  — full URL to original image
        - error (str)          — present only on failure
    """
    logger.info(
        f"🛠 [server] shashin_search_tool called — term={term}, page={page}"
    )

    result = _shashin_get(f"/api/v1/search/{page}/term/{term}")

    if not result["ok"]:
        return json.dumps({"success": False, "error": result["error"]}, indent=2)

    data        = result["data"]
    metadata    = data.get("metadataSearchList", [])
    keyword_map = data.get("keywordMap", {})
    total_pages = data.get("totalPages", 1)

    results = []
    for img in metadata:
        img_id = img["id"]
        results.append({
            "id":           img_id,
            "fileName":     img.get("fileName"),
            "title":        img.get("title"),
            "description":  img.get("description"),
            "takenAt":      img.get("takenAt"),
            "camera":       img.get("camera"),
            "lens":         img.get("lens"),
            "placeName":    img.get("placeName"),
            "year":         img.get("year"),
            "month":        img.get("month"),
            "day":          img.get("day"),
            "keywords":     keyword_map.get(img_id),
            "thumbnailUrl": f"{SHASHIN_BASE_URL}{img.get('thumbnailUrlSmall', '')}",
            "originalUrl":  f"{SHASHIN_BASE_URL}{img.get('thumbnailUrlOriginal', '')}",
        })

    return json.dumps({
        "success":     True,
        "term":        term,
        "count":       len(results),
        "page":        page,
        "total_pages": total_pages,
        "results":     results,
    }, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# shashin_random_tool
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
@check_tool_enabled(category="image_tools")
def shashin_random_tool() -> str:
    """
    Fetch metadata for a random image from the Shashin gallery.

    Calls the Shashin random metadata endpoint and returns the image ID and
    basic metadata. Pass the returned image_id to shashin_analyze_tool to
    fetch and describe the image via vision inference.

    Returns:
        JSON string with:
        - success (bool)
        - image_id (str)   — UUID, pass to shashin_analyze_tool
        - fileName (str)
        - takenAt (str)
        - camera (str)
        - placeName (str)
        - keywords (str)
        - error (str)      — present only on failure
    """
    logger.info("🛠 [server] shashin_random_tool called")

    result = _shashin_get("/api/v1/random/metadata/type/image")

    if not result["ok"]:
        return json.dumps({"success": False, "error": result["error"]}, indent=2)

    body = result["data"]
    if body.get("status") != "success":
        return json.dumps({"success": False, "error": body.get("msg", "Unknown error")}, indent=2)

    data = body.get("metadata", {})
    image_id = data.get("id")

    if not image_id:
        return json.dumps({"success": False, "error": "No image ID in random response"}, indent=2)

    return json.dumps({
        "success":        True,
        "image_id":       image_id,
        "fileName":       data.get("fileName"),
        "takenAt":        data.get("takenAt"),
        "camera":         data.get("camera"),
        "placeName":      data.get("placeName")
    }, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# shashin_analyze_tool
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
@check_tool_enabled(category="image_tools")
def shashin_analyze_tool(
    image_id: str,
    use_thumbnail: bool = True,
) -> str:
    """
    Fetch a Shashin image by ID and return its base64-encoded data for vision inference.

    The base64 payload is returned to the LangGraph agent, which passes it to
    Ollama for vision inference. No Ollama calls happen inside this tool.

    Existing Shashin recognition labels and object keywords are included
    alongside the image data as context for the agent.

    IMPORTANT: The active Ollama model must support vision inputs. If Ollama
    rejects the image, relay the error to the user and ask them to switch to
    a vision-capable model such as "qwen2.5vl:7b" or "llava:7b".

    Args:
        image_id (str, required):       Shashin image UUID from shashin_search_tool
        use_thumbnail (bool, optional): Use 225px thumbnail for speed (default: True).
                                        Set False for full-resolution analysis.

    Returns:
        JSON string with:
        - success (bool)
        - image_base64 (str)      — base64-encoded image, pass this to Ollama
        - existing_label (str)    — Shashin face recognition label if any
        - existing_keywords (str) — Shashin object detection tags if any
        - image_id (str)
        - thumbnail_used (bool)
        - error (str)             — present only on failure

    Always call shashin_search_tool first to obtain the image_id.
    """
    logger.info(
        f"🛠 [server] shashin_analyze_tool called — "
        f"image_id={image_id}, use_thumbnail={use_thumbnail}"
    )

    # Keywords come from shashin_search_tool results; no extra metadata fetch needed.

    if use_thumbnail:
        image_url = f"{SHASHIN_BASE_URL}/api/v1/thumbnails/225/{image_id}"
    else:
        image_url = f"{SHASHIN_BASE_URL}/api/v1/image/{image_id}"

    fetch_result = _fetch_image_as_base64(image_url)

    if not fetch_result["success"]:
        return json.dumps({
            "success":  False,
            "error":    fetch_result["error"],
            "image_id": image_id,
        }, indent=2)

    return json.dumps({
        "success":        True,
        "image_base64":   fetch_result["image_base64"],
        "image_id":       image_id,
        "thumbnail_used": use_thumbnail,
    }, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Skills plumbing
# ─────────────────────────────────────────────────────────────────────────────

skill_registry = None

@mcp.tool()
@check_tool_enabled(category="image_tools")
def list_skills() -> str:
    """List all available skills for this server."""
    logger.info("🛠  list_skills called")
    if skill_registry is None:
        return json.dumps({
            "server": "image-review-server",
            "skills": [],
            "message": "Skills not loaded"
        }, indent=2)
    return json.dumps({
        "server": "image-review-server",
        "skills": skill_registry.list()
    }, indent=2)


@mcp.tool()
@check_tool_enabled(category="image_tools")
def read_skill(skill_name: str) -> str:
    """Read the full content of a skill."""
    logger.info("🛠  read_skill called")
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
    loader = SkillLoader(server_tools, category="image_tools")
    skill_registry = loader.load_all(skills_dir)

    logger.info(f"🛠  {len(server_tools)} tools: {', '.join(server_tools)}")
    logger.info(f"🛠  {len(skill_registry.skills)} skills loaded")
    mcp.run(transport="stdio")