"""
Location MCP Server
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
from tools.location.geolocate_util import geolocate_ip, CLIENT_IP
from tools.location.get_location import get_location as get_location_fn
from tools.location.get_time import get_time as get_time_fn
from tools.location.get_weather import get_weather as get_weather_fn

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
logging.getLogger("mcp_location_server").setLevel(logging.INFO)

logger = logging.getLogger("mcp_location_server")
logger.info("🚀 Server logging initialized - writing to logs/mcp-server.log")

mcp = FastMCP("location-server")

@mcp.tool()
@check_tool_enabled(category="location")
@tool_meta(tags=["read","external"],triggers=["my location","where am i","current location","where do i live"],idempotent=True,example='use get_location_tool [city=""] [state=""] [country=""]')
def get_location_tool(city: str | None = None, state: str | None = None, country: str | None = None) -> str:
    """
    Retrieve structured geographic information for any location.

    Args:
        city (str, optional): City name (e.g., "Surrey", "Tokyo")
        state (str, optional): State/province (e.g., "BC", "California", "Ontario")
        country (str, optional): Country name (e.g., "Canada", "Japan")

    All arguments are optional. If none provided, uses client's IP to determine location.
    Timezone is NEVER required - determined automatically.

    Returns:
        JSON string with:
        - city: City name
        - state: State/province/region
        - country: Country name
        - latitude: Geographic latitude
        - longitude: Geographic longitude
        - timezone: IANA timezone identifier
        - timezone_offset: UTC offset

    Use when user asks about where a place is, geographic context, or "my location".
    """
    logger.info(f"🛠 [server] get_location_tool called with city: {city}, state: {state}, country: {country}")
    try:
        if not city and CLIENT_IP:
            loc = geolocate_ip(CLIENT_IP)
            if loc:
                city = loc.get("city")
                state = loc.get("region")
                country = loc.get("country")
        result = get_location_fn(city, state, country)
        if not result:
            raise MCPToolError(FailureKind.USER_ERROR, f"Location not found for: {city}, {state}, {country}",
                               {"tool": "get_location_tool", "city": city, "state": state, "country": country})
        return result
    except MCPToolError:
        raise
    except Exception as e:
        logger.error(f"❌ get_location_tool failed: {e}", exc_info=True)
        raise MCPToolError(FailureKind.UPSTREAM_ERROR, f"Location lookup failed: {e}",
                           {"tool": "get_location_tool", "city": city})


@mcp.tool()
@check_tool_enabled(category="location")
@tool_meta(tags=["read","external"],triggers=["what time","current time","what date","time now"],idempotent=False,example='use get_time_tool [city=""] [state=""] [country=""]')
def get_time_tool(city: str | None = None, state: str | None = None, country: str | None = None) -> str:
    """
    Get the current local time for any city in the world.

    Args:
        city (str, optional): City name (e.g., "London", "New York")
        state (str, optional): State/province (e.g., "NY", "Queensland")
        country (str, optional): Country name (e.g., "United States", "Australia")

    All arguments are optional. If none provided, uses client's IP to determine location.
    Timezone is NEVER required - determined automatically from location.

    Returns:
        JSON string with:
        - city: City name
        - state: State/province
        - country: Country name
        - current_time: Current time in HH:MM:SS format
        - date: Current date in YYYY-MM-DD format
        - timezone: IANA timezone identifier
        - day_of_week: Day name (Monday, Tuesday, etc.)

    Use when user asks "What time is it in X" or "What time is it here".
    """
    logger.info(f"🛠 [server] get_time_tool called with city: {city}, state: {state}, country: {country}")
    try:
        if not city and CLIENT_IP:
            loc = geolocate_ip(CLIENT_IP)
            if loc:
                city = loc.get("city")
                state = loc.get("region")
                country = loc.get("country")
        result = get_time_fn(city, state, country)
        if not result:
            raise MCPToolError(FailureKind.USER_ERROR, f"Could not determine time for: {city}, {state}, {country}",
                               {"tool": "get_time_tool", "city": city})
        return result
    except MCPToolError:
        raise
    except Exception as e:
        logger.error(f"❌ get_time_tool failed: {e}", exc_info=True)
        raise MCPToolError(FailureKind.UPSTREAM_ERROR, f"Time lookup failed: {e}",
                           {"tool": "get_time_tool", "city": city})


@mcp.tool()
@check_tool_enabled(category="location")
@tool_meta(tags=["read","external"],triggers=["weather","temperature","forecast","rain","snow","wind"],idempotent=False,example='use get_weather_tool [city=""] [state=""] [country=""] [forecast_days=""]')
def get_weather_tool(
        city: str | None = None,
        state: str | None = None,
        country: str | None = None,
        forecast_days: Optional[int] = 7
) -> str:
    """
    Get weather forecast. Returns current conditions plus daily forecasts.

    CRITICAL: forecast_days means "total days starting from today"
    - forecast_days=1 returns 1 day (today only)
    - forecast_days=2 returns 2 days (today + tomorrow)
    - forecast_days=7 returns 7 days (today through 6 days out)

    The forecast array ALWAYS starts with today at index 0:
    - forecast[0] = today (relative_day: "today")
    - forecast[1] = tomorrow (relative_day: "tomorrow")
    - forecast[2] = day after tomorrow (relative_day: "day_after_tomorrow")

    FOR TOMORROW'S WEATHER: You MUST use forecast_days=2 (minimum)
    Then read forecast[1] or find entry where relative_day=="tomorrow"

    Args:
        city: City name (optional, defaults to IP location)
        state: State/province (optional)
        country: Country (optional)
        forecast_days: How many days total (1-16, default 7)

    Returns:
        JSON with current weather and forecast array. Each forecast has:
        - date: "2026-02-24"
        - day_label: "Tomorrow" (human readable)
        - relative_day: "tomorrow" (machine readable - use this!)
        - condition, temperatures, precipitation, etc.
    """
    try:
        forecast_days = int(forecast_days) if forecast_days is not None else 7
        if not (1 <= forecast_days <= 16):
            raise MCPToolError(FailureKind.USER_ERROR, f"forecast_days must be 1-16, got {forecast_days}",
                               {"tool": "get_weather_tool", "param": "forecast_days", "value": forecast_days})
    except MCPToolError:
        raise
    except (TypeError, ValueError) as e:
        raise MCPToolError(FailureKind.USER_ERROR, f"Invalid forecast_days value: {forecast_days}",
                           {"tool": "get_weather_tool", "param": "forecast_days"})

    logger.info(f"🛠 [server] get_weather_tool called with city: {city}, state: {state}, country: {country}, forecast_days: {forecast_days}")
    logger.info(f"🌤️  CLIENT_IP = {CLIENT_IP}")

    try:
        if not city and CLIENT_IP:
            logger.info(f"🌤️  No city provided, using IP geolocation...")
            loc = geolocate_ip(CLIENT_IP)
            logger.info(f"🌤️  Geolocation result: {loc}")
            if loc:
                city = loc.get("city")
                state = loc.get("region")
                country = loc.get("country")
                logger.info(f"🌤️  Resolved to: city={city}, state={state}, country={country}")

        result = get_weather_fn(city, state, country, forecast_days=forecast_days)
        if not result:
            raise MCPToolError(FailureKind.USER_ERROR, f"No weather data for: {city}, {state}, {country}",
                               {"tool": "get_weather_tool", "city": city})
        logger.info(f"🌤️  Returning weather result")
        return result
    except MCPToolError:
        raise
    except Exception as e:
        logger.error(f"❌ get_weather_tool failed: {e}", exc_info=True)
        raise MCPToolError(FailureKind.UPSTREAM_ERROR, f"Weather fetch failed: {e}",
                           {"tool": "get_weather_tool", "city": city})

skill_registry = None

@mcp.tool()
@check_tool_enabled(category="location")
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
@check_tool_enabled(category="location")
def list_skills() -> str:
    """List all available skills for this server."""
    logger.info(f"🛠  list_skills called")
    if skill_registry is None:
        return json.dumps({
            "server": "location-server",
            "skills": [],
            "message": "Skills not loaded"
        }, indent=2)

    return json.dumps({
        "server": "location-server",
        "skills": skill_registry.list()
    }, indent=2)


@mcp.tool()
@check_tool_enabled(category="location")
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
    loader = SkillLoader(server_tools, category="location")
    skill_registry = loader.load_all(skills_dir)

    logger.info(f"🛠  {len(server_tools)} tools: {', '.join(server_tools)}")
    logger.info(f"🛠  {len(skill_registry.skills)} skills loaded")
    mcp.run(transport="stdio")