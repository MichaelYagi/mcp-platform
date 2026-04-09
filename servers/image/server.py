"""
Image Tools MCP Server
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
try:
    from client.tool_meta import tool_meta
except Exception:
    # Fallback stub — metadata is attached but not used in server subprocess
    def tool_meta(**kwargs):
        def decorator(fn): return fn
        return decorator

from mcp.server.fastmcp import FastMCP

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
logging.getLogger("mcp_image_server").setLevel(logging.INFO)

logger = logging.getLogger("mcp_image_server")
logger.info("🚀 Image Tools Server logging initialized - writing to logs/mcp-server.log")

mcp = FastMCP("image-server")

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
    win_match = re.match(r'^([A-Za-z]):[/\\](.*)', file_path)
    if win_match:
        drive = win_match.group(1).lower()
        rest  = win_match.group(2).replace('\\', '/')
        file_path = f"/mnt/{drive}/{rest}"
    return Path(file_path).expanduser().resolve()


def _fetch_image_as_base64(image_url: str) -> dict:
    try:
        headers = {}
        if SHASHIN_API_KEY and (
            "192.168." in image_url or "shashin" in image_url.lower()
        ):
            headers = _SHASHIN_HEADERS

        resp = requests.get(image_url, headers=headers, timeout=30)
        resp.raise_for_status()

        # Reject non-image responses (e.g. HTML viewer pages, JSON error pages)
        content_type = resp.headers.get("content-type", "").lower()
        if content_type and not any(t in content_type for t in
                                    ("image/", "application/octet-stream")):
            raise MCPToolError(
                FailureKind.USER_ERROR,
                f"URL did not return an image (content-type: {content_type}). "
                f"Make sure the URL points directly to an image file, not a viewer page.",
                {"url": image_url, "content_type": content_type}
            )

        encoded = base64.b64encode(resp.content).decode("utf-8")
        logger.info(f"[fetch_image] {image_url} — {len(resp.content)} bytes encoded")
        return {"success": True, "image_base64": encoded}

    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response else 0
        kind = FailureKind.USER_ERROR if status in (400, 401, 403, 404) else FailureKind.UPSTREAM_ERROR
        raise MCPToolError(kind, f"HTTP {status} fetching image: {image_url}",
                           {"url": image_url, "status_code": status})
    except requests.exceptions.ConnectionError as exc:
        raise MCPToolError(FailureKind.UPSTREAM_ERROR, f"Connection failed fetching image: {image_url}",
                           {"url": image_url})
    except Exception as exc:
        logger.error(f"[fetch_image] Failed to fetch {image_url}: {exc}")
        raise MCPToolError(FailureKind.INTERNAL_ERROR, f"Could not fetch image: {exc}",
                           {"url": image_url})


def _read_image_file_as_base64(file_path: str) -> dict:
    try:
        resolved = _resolve_file_path(file_path)
        if not resolved.exists():
            raise MCPToolError(FailureKind.USER_ERROR, f"File not found: {resolved}",
                               {"file_path": file_path})
        if not resolved.is_file():
            raise MCPToolError(FailureKind.USER_ERROR, f"Path is not a file: {resolved}",
                               {"file_path": file_path})

        with open(resolved, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")

        logger.info(f"[read_image_file] {resolved} encoded successfully")
        return {"success": True, "image_base64": encoded}

    except MCPToolError:
        raise
    except Exception as exc:
        logger.error(f"[read_image_file] Failed to read {file_path}: {exc}")
        raise MCPToolError(FailureKind.INTERNAL_ERROR, f"Could not read image file: {exc}",
                           {"file_path": file_path})


def _shashin_get(path: str) -> dict:
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
@check_tool_enabled(category="image")
@tool_meta(tags=["read","vision","ai"],triggers=["analyze image","describe image","what is in this image","look at image"],idempotent=True,example='use analyze_image_tool: [image_url=""] [image_file_path=""] [image_base64=""] [query=""]')
def analyze_image_tool(
    image_url: Optional[str] = None,
    image_file_path: Optional[str] = None,
    image_base64: Optional[str] = None,
    query: Optional[str] = None,
) -> str:
    """
    Fetch and base64-encode any image for vision inference.

    Accepts an image from a URL, a local file path, or a pre-encoded base64 string.
    Provide exactly one of image_url, image_file_path, or image_base64.

    Args:
        image_url (str, optional):       Any HTTP(S) URL pointing to an image.
        image_file_path (str, optional): Local file path to an image.
        image_base64 (str, optional):    Already-encoded base64 image data.
        query (str, optional):           A specific question to answer about the image.
                                         If omitted, a general description is produced.

    Returns:
        JSON string with success (bool), image_base64 (str), error (str on failure).
    """
    logger.info(
        f"🛠 [server] analyze_image_tool called — "
        f"url={image_url}, file_path={image_file_path}, has_base64={bool(image_base64)}"
    )

    if not image_url and not image_file_path and not image_base64:
        raise MCPToolError(FailureKind.USER_ERROR,
                           "Provide one of: image_url, image_file_path, or image_base64.",
                           {"tool": "analyze_image_tool"})

    if image_base64:
        result = {"success": True, "image_base64": image_base64}
    elif image_file_path:
        result = _read_image_file_as_base64(image_file_path)
        if result.get("success"):
            result["image_source"] = image_file_path
    else:
        result = _fetch_image_as_base64(image_url)
        if result.get("success"):
            result["image_source"] = image_url

    if query:
        result["query"] = query

    return json.dumps(result, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# shashin_search_tool
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
@check_tool_enabled(category="image")
@tool_meta(tags=["read","search","media"],triggers=["find photos","search photos","my photos","show photos","photos of"],idempotent=True,example='use shashin_search_tool: term="" [page=""]',text_fields=["tags"],intent_category="shashin_search")
def shashin_search_tool(
    term: str,
    page: Optional[int] = 0,
) -> str:
    """
    Search the Shashin self-hosted media gallery by keyword term.

    Args:
        term (str, required): Search term e.g. "cat", "Noah", "beach", "sunset"
        page (int, optional): Pagination page index, 0-based (default: 0)

    Returns:
        Formatted numbered list of matching images with metadata.
    """
    if not term or not term.strip():
        raise MCPToolError(FailureKind.USER_ERROR, "term must not be empty",
                           {"tool": "shashin_search_tool"})

    try:
        page = int(page) if page else 0
    except (TypeError, ValueError):
        page = 0

    logger.info(f"🛠 [server] shashin_search_tool called — term={term}, page={page}")

    result = _shashin_get(f"/api/v1/search/{page}/term/{term}")

    if not result["ok"]:
        raise MCPToolError(FailureKind.UPSTREAM_ERROR, f"Shashin search failed: {result['error']}",
                           {"tool": "shashin_search_tool", "term": term})

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

    lines = [f'Found {len(results)} photo(s) matching "{term}" (page {page + 1} of {total_pages}):\n']
    shashin_base = os.getenv("SHASHIN_BASE_URL", "http://192.168.0.199:6624")

    for i, r in enumerate(results, 1):
        img_md = f"![{r['fileName']}]({shashin_base}/api/v1/thumbnails/225/{r['id']})" if r.get("thumbnailUrl") else ""
        lines.append(f"{i}. {img_md}")
        lines.append(f"   📕 {r['fileName']} — {r['takenAt']}")
        lines.append(f"   🆔 {r['id']}")
        if r.get("id"):
            lines.append(f"   🔗 {shashin_base}/search?term={r['id']}")
        if r.get("description"):
            lines.append(f"   📝 {r['description']}")
        if r.get("placeName"):
            placename = r['placeName'].split(";", 1)[0].strip()
            import urllib.parse as _up
            lines.append(f"   📍 [{placename}](https://maps.google.com/?q={_up.quote(placename)})")
        if r.get("keywords"):
            kw = r["keywords"] if isinstance(r["keywords"], str) else ", ".join(r["keywords"])
            lines.append(f"   🏷️ {kw}")
        lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# shashin_random_tool
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
@check_tool_enabled(category="image")
@tool_meta(tags=["read","media"],triggers=["random photo","surprise me","show me a random"],idempotent=False,example="use shashin_random_tool",intent_category="shashin_random")
def shashin_random_tool() -> str:
    """
    Fetch metadata for a random image from the Shashin gallery.

    Returns:
        JSON string with success, image_id, image_source, fileName, takenAt,
        camera, placeName, description.
    """
    logger.info("🛠 [server] shashin_random_tool called")

    result = _shashin_get("/api/v1/random/metadata/type/image")

    if not result["ok"]:
        raise MCPToolError(FailureKind.UPSTREAM_ERROR, f"Shashin random fetch failed: {result['error']}",
                           {"tool": "shashin_random_tool"})

    body = result["data"]
    if body.get("status") != "success":
        raise MCPToolError(FailureKind.UPSTREAM_ERROR, f"Shashin returned error: {body.get('msg', 'Unknown error')}",
                           {"tool": "shashin_random_tool"})

    data = body.get("metadata", {})
    image_id = data.get("id")

    if not image_id:
        raise MCPToolError(FailureKind.UPSTREAM_ERROR, "No image ID in random response",
                           {"tool": "shashin_random_tool"})

    image_source          = f"{SHASHIN_BASE_URL}/api/v1/thumbnails/225/{image_id}"
    image_source_original = f"{SHASHIN_BASE_URL}/api/v1/thumbnails/original/{image_id}"

    return json.dumps({
        "success":                True,
        "image_id":               image_id,
        "image_source":           image_source,
        "image_source_original":  image_source_original,
        "fileName":               data.get("fileName"),
        "takenAt":                data.get("takenAt"),
        "camera":                 data.get("camera"),
        "placeName":              data.get("placeName"),
        "description":            data.get("description"),
    }, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# shashin_analyze_tool
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
@check_tool_enabled(category="image")
@tool_meta(tags=["read","vision","media","ai"],triggers=["analyze photo","describe photo","what is in this photo"],idempotent=True,example='use shashin_analyze_tool: image_id="" [query=""] [use_thumbnail=""]',intent_category="shashin_analyze")
def shashin_analyze_tool(
    image_id: str,
    use_thumbnail: bool = True,
    query: Optional[str] = None,
) -> str:
    """
    Fetch a Shashin image by ID and return its metadata for vision inference.

    Args:
        image_id (str, required):        Shashin image UUID from shashin_search_tool
        use_thumbnail (bool, optional):  Use 225px thumbnail for speed (default: True)
        query (str, optional):           A specific question to answer about the image.
                                         If omitted, a general description is produced.

    Returns:
        JSON string with success, image_source, image_id, fileName, takenAt,
        camera, placeName, description.
    """
    if not image_id or not image_id.strip():
        raise MCPToolError(FailureKind.USER_ERROR, "image_id must not be empty",
                           {"tool": "shashin_analyze_tool"})

    logger.info(
        f"🛠 [server] shashin_analyze_tool called — "
        f"image_id={image_id}, use_thumbnail={use_thumbnail}"
    )

    meta_result = _shashin_get(f"/api/v1/media/metadata/{image_id}")
    if meta_result["ok"]:
        body = meta_result["data"]
        meta = body.get("metadata", {})
    else:
        # Non-fatal — return what we have without metadata
        logger.warning(f"[shashin_analyze_tool] Metadata fetch failed: {meta_result['error']}")
        meta = {}

    image_source          = f"{SHASHIN_BASE_URL}/api/v1/thumbnails/225/{image_id}"
    image_source_original = f"{SHASHIN_BASE_URL}/api/v1/thumbnails/original/{image_id}"

    result = {
        "success":                True,
        "image_source":           image_source,
        "image_source_original":  image_source_original,
        "image_id":               image_id,
        "fileName":               meta.get("fileName"),
        "takenAt":                meta.get("takenAt"),
        "camera":                 meta.get("camera"),
        "placeName":              meta.get("placeName"),
        "description":            meta.get("description"),
    }
    if query:
        result["query"] = query
    return json.dumps(result, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# web_image_search_tool
# ─────────────────────────────────────────────────────────────────────────────

@mcp.tool()
@check_tool_enabled(category="image")
@tool_meta(tags=["read","search","external"],triggers=["show me a picture of","what does it look like","find image of"],idempotent=True,example='use web_image_search_tool: query=""',intent_category="web_image_search")
def web_image_search_tool(query: str) -> str:
    """
    Search the web for images using Google Images (via Serper).

    Args:
        query (str, required): The entity to search for, e.g. "Jorma Tommila", "Eiffel Tower"

    Returns:
        Formatted numbered list of image results with title, source, link, image URL.
    """
    if not query or not query.strip():
        raise MCPToolError(FailureKind.USER_ERROR, "query must not be empty",
                           {"tool": "web_image_search_tool"})

    logger.info(f"🛠 [server] web_image_search_tool called — query={query!r}")

    serper_key = os.getenv("SERPER_API_KEY", "")
    if not serper_key:
        raise MCPToolError(FailureKind.USER_ERROR, "SERPER_API_KEY not configured in .env",
                           {"tool": "web_image_search_tool"})

    try:
        resp = requests.get(
            "https://google.serper.dev/images",
            params={"q": query, "apiKey": serper_key, "num": 10},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response else 0
        if status == 401:
            raise MCPToolError(FailureKind.USER_ERROR, "Invalid SERPER_API_KEY",
                               {"tool": "web_image_search_tool"})
        raise MCPToolError(FailureKind.UPSTREAM_ERROR, f"Serper API returned HTTP {status}",
                           {"tool": "web_image_search_tool", "query": query})
    except requests.exceptions.ConnectionError:
        raise MCPToolError(FailureKind.RETRYABLE, "Connection to Serper API failed",
                           {"tool": "web_image_search_tool"})
    except Exception as exc:
        logger.error(f"[web_image_search_tool] Request failed: {exc}")
        raise MCPToolError(FailureKind.UPSTREAM_ERROR, f"Image search failed: {exc}",
                           {"tool": "web_image_search_tool", "query": query})

    images = data.get("images", [])
    if not images:
        return f'No images found for "{query}".'

    MAX_W, MAX_H = 1200, 900

    results = []
    for img in images:
        w = img.get("imageWidth", 0)
        h = img.get("imageHeight", 0)
        image_url = (
            img.get("imageUrl", "")
            if w <= MAX_W and h <= MAX_H
            else img.get("thumbnailUrl", "") or img.get("imageUrl", "")
        )
        results.append({
            "title":     img.get("title", ""),
            "source":    img.get("source", "") or img.get("domain", ""),
            "link":      img.get("link", ""),
            "image_url": image_url,
            "width":     w,
            "height":    h,
        })

    lines = [f'Found {len(results)} image(s) for "{query}":\n']
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}")
        if r.get("image_url"):
            lines.append(f"   ![{r['title']}]({r['image_url']})")
        if r.get("link"):
            lines.append(f"   🔗 {r['link']}")
        if r.get("source"):
            lines.append(f"   📰 {r['source']}")
        lines.append("")

    logger.info(f"[web_image_search_tool] returning {len(results)} results for {query!r}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Skills plumbing
# ─────────────────────────────────────────────────────────────────────────────

skill_registry = None

@mcp.tool()
@check_tool_enabled(category="image")
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
@check_tool_enabled(category="image")
def list_skills() -> str:
    """List all available skills for this server."""
    logger.info("🛠  list_skills called")
    if skill_registry is None:
        return json.dumps({"server": "image-server", "skills": [], "message": "Skills not loaded"}, indent=2)
    return json.dumps({"server": "image-server", "skills": skill_registry.list()}, indent=2)


@mcp.tool()
@check_tool_enabled(category="image")
def read_skill(skill_name: str) -> str:
    """Read the full content of a skill."""
    logger.info("🛠  read_skill called")
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
    loader = SkillLoader(server_tools, category="image")
    skill_registry = loader.load_all(skills_dir)
    logger.info(f"🛠  {len(server_tools)} tools: {', '.join(server_tools)}")
    logger.info(f"🛠  {len(skill_registry.skills)} skills loaded")
    mcp.run(transport="stdio")