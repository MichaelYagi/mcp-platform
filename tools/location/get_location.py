import json
import urllib.parse
from typing import Optional
from tools.location.resolve_location import resolve_location

def _maps_link(city, state, country) -> str:
    parts = [p for p in [city, state, country] if p]
    query = ", ".join(parts)
    return f"[{query}](https://maps.google.com/?q={urllib.parse.quote(query)})"

def get_location(city: Optional[str] = None, state: Optional[str] = None, country: Optional[str] = None) -> str:
    """
    Returns the resolved location as JSON.
    """
    loc = resolve_location(city, state, country)
    loc["maps_link"] = _maps_link(loc.get("city"), loc.get("state"), loc.get("country"))
    return json.dumps(loc, indent=2)