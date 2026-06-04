"""
Discord MCP server for mcp-platform.
Sends notifications to Discord channels via webhooks.
No bot token or server membership required.

Requirements:
    pip install httpx

Environment variables (.env):
    DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...

    Optional additional webhooks:
    DISCORD_WEBHOOK_ALERTS=https://discord.com/api/webhooks/...
    DISCORD_WEBHOOK_GENERAL=https://discord.com/api/webhooks/...
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
        from servers.error_fallback import FailureKind, MCPToolError, JsonFormatter

from mcp.server.fastmcp import FastMCP

from servers.logging_setup import setup_server_logging
logger = setup_server_logging("mcp_discord_server", PROJECT_ROOT, JsonFormatter)
logger.info("🚀 Discord server logging initialized")

# ── httpx ──────────────────────────────────────────────────────────────────────
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logger.warning("⚠️  httpx not installed — run: pip install httpx")

# ── Config ─────────────────────────────────────────────────────────────────────
# Default webhook — used when no webhook name is specified
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# Collect all configured webhooks (DISCORD_WEBHOOK_* env vars)
DISCORD_WEBHOOKS: dict[str, str] = {}
for _k, _v in os.environ.items():
    if _k.startswith("DISCORD_WEBHOOK_") and _v.startswith("https://discord.com/api/webhooks/"):
        _name = _k.replace("DISCORD_WEBHOOK_", "").lower() or "default"
        DISCORD_WEBHOOKS[_name] = _v

mcp = FastMCP("discord-server")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _resolve_webhook(webhook: Optional[str]) -> str:
    """Resolve a webhook name or URL to a full webhook URL."""
    if not webhook:
        if not DISCORD_WEBHOOK_URL:
            raise MCPToolError(
                FailureKind.USER_ERROR,
                "DISCORD_WEBHOOK_URL not set in .env — add a webhook URL to use Discord notifications",
                {"tool": "discord_notify"}
            )
        return DISCORD_WEBHOOK_URL

    # If it looks like a full URL, use it directly
    if webhook.startswith("https://"):
        return webhook

    # Look up by name (e.g. "alerts" → DISCORD_WEBHOOK_ALERTS)
    key = webhook.lower()
    if key in DISCORD_WEBHOOKS:
        return DISCORD_WEBHOOKS[key]

    available = list(DISCORD_WEBHOOKS.keys())
    raise MCPToolError(
        FailureKind.USER_ERROR,
        f"Webhook '{webhook}' not found. Available: {available}",
        {"tool": "discord_notify", "available_webhooks": available}
    )


# ── Tools ──────────────────────────────────────────────────────────────────────

@mcp.tool()
@check_tool_enabled(category="discord")
@tool_meta(
    tags=["write", "discord", "notifications", "external"],
    triggers=["notify discord", "discord notification", "send discord message",
              "post to discord", "discord alert", "notify me on discord",
              "send discord notification", "message discord"],
    idempotent=False,
    template='use discord_notify: message="" [webhook=""] [username=""] [title=""]',
    intent_category="discord",
output_type="none",pipe_targets={"message":"text"})
def discord_notify(
    message: str,
    webhook: Optional[str] = None,
    username: Optional[str] = None,
    title: Optional[str] = None,
) -> str:
    """
    Send a notification message to a Discord channel via webhook.

    No bot token or server membership required. Just a webhook URL configured
    in .env as DISCORD_WEBHOOK_URL (or DISCORD_WEBHOOK_<NAME> for multiple).

    Args:
        message (str, required): The notification text. Supports Discord markdown:
                                  **bold**, *italic*, `code`, > quote.
        webhook (str, optional): Webhook name (e.g. "alerts") or full URL.
                                  Defaults to DISCORD_WEBHOOK_URL from .env.
        username (str, optional): Override the bot display name for this message.
                                  Defaults to the name set when creating the webhook.
        title (str, optional): Optional bold title shown above the message.

    Returns:
        JSON with:
        - status: "sent"
        - channel: Webhook name used
        - content: The message sent

    Use cases:
        - Scheduler action: "notify me on Discord when condition fires"
        - "Send a Discord notification saying the deploy is done"
        - "Post an alert to Discord: invoice email received"
    """
    logger.info(f"🛠  discord_notify called: webhook={webhook or 'default'}")

    if not HTTPX_AVAILABLE:
        raise MCPToolError(FailureKind.USER_ERROR,
                           "httpx not installed — run: pip install httpx",
                           {"tool": "discord_notify"})
    if not message or not message.strip():
        raise MCPToolError(FailureKind.USER_ERROR, "message must not be empty",
                           {"tool": "discord_notify"})

    webhook_url = _resolve_webhook(webhook)
    webhook_name = webhook or "default"

    # Build content — prepend bold title if provided
    content = f"**{title}**\n{message}" if title else message

    # Extract local image URLs from markdown: ![](http://192.168.x.x/...)
    import re as _re
    import urllib.request as _urllib
    _LOCAL_IMG_RE = _re.compile(
        r'!\[[^\]]*\]\((http://(?:192\.168\.|10\.|172\.(?:1[6-9]|2\d|3[01])\.)[^\s)]+)\)'
    )
    local_images = _LOCAL_IMG_RE.findall(content)

    # Strip markdown image syntax from the text content
    clean_content = _LOCAL_IMG_RE.sub("", content).strip()

    payload: dict = {"content": clean_content}
    if username:
        payload["username"] = username

    try:
        if local_images:
            # Fetch the first local image and upload as a file attachment
            img_url = local_images[0]
            # Prefer original resolution over thumbnail
            if '/thumbnails/225/' in img_url:
                img_url = img_url.replace('/thumbnails/225/', '/thumbnails/original/')
            try:
                with _urllib.urlopen(img_url, timeout=10) as _resp:
                    img_data = _resp.read()
                    content_type = _resp.headers.get("Content-Type", "image/jpeg").split(";")[0]

                ext = {"image/jpeg": "jpg", "image/png": "png", "image/gif": "gif",
                       "image/webp": "webp"}.get(content_type, "jpg")
                filename = f"photo.{ext}"

                resp = httpx.post(
                    webhook_url,
                    data={"payload_json": json.dumps(payload)},
                    files={"file": (filename, img_data, content_type)},
                    timeout=30,
                )
            except Exception as _img_err:
                logger.warning(f"⚠️  discord_notify: image fetch failed ({_img_err}), sending text only")
                resp = httpx.post(webhook_url, json=payload, timeout=10)
        else:
            resp = httpx.post(webhook_url, json=payload, timeout=10)

        if resp.status_code not in (200, 204):
            raise MCPToolError(
                FailureKind.UPSTREAM_ERROR,
                f"Discord webhook returned {resp.status_code}: {resp.text}",
                {"tool": "discord_notify", "status_code": resp.status_code}
            )
        logger.info(f"✅ discord_notify: sent to webhook '{webhook_name}'")
        return json.dumps({
            "status":  "sent",
            "channel": webhook_name,
            "content": clean_content,
        }, indent=2)

    except MCPToolError:
        raise
    except Exception as e:
        logger.error(f"❌ discord_notify error: {e}")
        raise MCPToolError(FailureKind.UPSTREAM_ERROR, f"Discord error: {e}",
                           {"tool": "discord_notify"})


@mcp.tool()
@check_tool_enabled(category="discord")
@tool_meta(
    tags=["read", "discord", "external"],
    triggers=["list discord webhooks", "discord webhooks configured",
              "what discord channels", "discord setup"],
    idempotent=True,
    template="use discord_list_webhooks",
    intent_category="discord",
    text_fields=["text"],
)
def discord_list_webhooks() -> str:
    """
    List all configured Discord webhooks from .env.

    Shows which webhook names are available to use with discord_notify.

    Returns:
        JSON with:
        - count: Number of configured webhooks
        - webhooks: List of webhook names (URLs are masked for security)
        - text: Human-readable list
    """
    logger.info("🛠  discord_list_webhooks called")

    webhooks = []
    for name, url in DISCORD_WEBHOOKS.items():
        # Mask the token portion of the URL
        parts = url.rstrip("/").split("/")
        masked = "/".join(parts[:-1]) + "/***"
        webhooks.append({"name": name, "url_masked": masked})

    lines = []
    for w in webhooks:
        lines.append(f"  {w['name']}: {w['url_masked']}")

    text = "\n".join(lines) if lines else "No webhooks configured. Add DISCORD_WEBHOOK_URL to .env"
    logger.info(f"✅ discord_list_webhooks: {len(webhooks)} configured")
    return json.dumps({
        "count":    len(webhooks),
        "webhooks": webhooks,
        "text":     text,
    }, indent=2)


skill_registry = None


@mcp.tool()
@check_tool_enabled(category="discord")
@tool_meta(tags=["read"], triggers=["list capabilities", "what can you do"],
           idempotent=True, template="use list_capabilities", intent_category="discord")
def list_capabilities(filter_tags: Optional[str] = None) -> str:
    """Return the full capability schema for every tool on this server."""
    logger.info(f"🛠  list_capabilities called (filter_tags={filter_tags})")
    try:
        from client.capability_registry import (
            _TOOL_TAGS, _TOOL_RATE_LIMITS, _TOOL_IDEMPOTENT, _INTERNAL_TOOLS
        )
    except ImportError:
        return json.dumps({"error": "CapabilityRegistry not available"}, indent=2)

    _current = sys.modules[__name__]
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
        sig = inspect.signature(_obj)
        params = []
        for pname, param in sig.parameters.items():
            if pname == "self":
                continue
            has_default = param.default is not inspect.Parameter.empty
            ann = param.annotation
            type_str = (
                ann.__name__ if hasattr(ann, "__name__")
                else str(ann).replace("typing.", "").replace("Optional[", "").rstrip("]")
                if ann is not inspect.Parameter.empty else "string"
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
@check_tool_enabled(category="discord")
def list_skills() -> str:
    """List all available skills for this server."""
    logger.info("🛠  list_skills called")
    if skill_registry is None:
        return json.dumps({"server": "discord-server", "skills": [], "message": "Skills not loaded"}, indent=2)
    return json.dumps({"server": "discord-server", "skills": skill_registry.list()}, indent=2)


@mcp.tool()
@check_tool_enabled(category="discord")
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
    if not DISCORD_WEBHOOK_URL and not DISCORD_WEBHOOKS:
        logger.warning("⚠️  No DISCORD_WEBHOOK_URL configured in .env — discord_notify will fail")
    else:
        logger.info(f"🔔 {len(DISCORD_WEBHOOKS)} webhook(s) configured: {list(DISCORD_WEBHOOKS.keys())}")

    server_tools = get_tool_names_from_module()
    skills_dir = Path(__file__).parent / "skills"
    loader = SkillLoader(server_tools, category="discord")
    skill_registry = loader.load_all(skills_dir)
    logger.info(f"🛠️  {len(server_tools)} tools: {', '.join(server_tools)}")
    logger.info(f"📚 {len(skill_registry.skills)} skills loaded")
    mcp.run(transport="stdio")