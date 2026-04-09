import json
import urllib.parse
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from typing import Optional
from tools.location.resolve_location import resolve_location
from tools.location.resolve_timezone import resolve_timezone
from tools.location.get_time_data import DEFAULT_TZ


def _maps_link(city, state, country) -> str:
    parts = [p for p in [city, state, country] if p]
    query = ", ".join(parts)
    return f"[{query}](https://maps.google.com/?q={urllib.parse.quote(query)})"


def get_time(
        city: Optional[str] = None,
        state: Optional[str] = None,
        country: Optional[str] = None,
        timezone: Optional[str] = None
) -> str:
    loc = resolve_location(city, state, country)

    # Determine timezone
    tz_name = timezone or resolve_timezone(loc["city"], loc["state"], loc["country"])

    try:
        tz = ZoneInfo(tz_name)
    except ZoneInfoNotFoundError:
        tz = ZoneInfo(DEFAULT_TZ)
        tz_name = DEFAULT_TZ

    now = datetime.now(tz)

    result = {
        "city": loc["city"],
        "state": loc["state"],
        "country": loc["country"],
        "timezone": tz_name,
        "local_time": now.strftime("%-I:%M %p, %A %B %-d %Y"),
        "maps_link": _maps_link(loc["city"], loc["state"], loc["country"]),
    }

    return json.dumps(result, indent=2)