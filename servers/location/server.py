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
from tools.tool_control import check_tool_enabled, is_tool_enabled, disabled_tool_response

from mcp.server.fastmcp import FastMCP
from tools.location.geolocate_util import geolocate_ip, CLIENT_IP
from tools.location.get_location import get_location as get_location_fn
from tools.location.get_time import get_time as get_time_fn
from tools.location.get_weather import get_weather as get_weather_fn

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
logging.getLogger("mcp_location_server").setLevel(logging.INFO)

logger = logging.getLogger("mcp_location_server")
logger.info("🚀 Server logging initialized - writing to logs/mcp-server.log")

mcp = FastMCP("location-server")

@mcp.tool()
@check_tool_enabled(category="location")
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
    if not city and CLIENT_IP:
        loc = geolocate_ip(CLIENT_IP)
        if loc:
            city = loc.get("city")
            state = loc.get("region")
            country = loc.get("country")

    return json.dumps(get_location_fn(city, state, country), indent=2)


@mcp.tool()
@check_tool_enabled(category="location")
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
    if not city and CLIENT_IP:
        loc = geolocate_ip(CLIENT_IP)
        if loc:
            city = loc.get("city")
            state = loc.get("region")
            country = loc.get("country")

    return json.dumps(get_time_fn(city, state, country), indent=2)


@mcp.tool()
@check_tool_enabled(category="location")
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
    forecast_days = int(forecast_days) if forecast_days is not None else 7
    logger.info(
        f"🛠 [server] get_weather_tool called with city: {city}, state: {state}, country: {country}, forecast_days: {forecast_days}")
    logger.info(f"🌤️  CLIENT_IP = {CLIENT_IP}")

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
    logger.info(f"🌤️  Returning weather result")
    return result

skill_registry = None

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
    # Auto-extract tool names - NO manual list needed!
    server_tools = get_tool_names_from_module()

    # Load skills
    skills_dir = Path(__file__).parent / "skills"
    loader = SkillLoader(server_tools, category="location")
    skill_registry = loader.load_all(skills_dir)

    logger.info(f"🛠  {len(server_tools)} tools: {', '.join(server_tools)}")
    logger.info(f"🛠  {len(skill_registry.skills)} skills loaded")
    mcp.run(transport="stdio")