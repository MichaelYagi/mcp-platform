import json
from typing import Optional
from tools.location.resolve_location import resolve_location

def get_location(city: Optional[str] = None, state: Optional[str] = None, country: Optional[str] = None) -> str:
    """
    Returns the resolved location as JSON.
    """
    loc = resolve_location(city, state, country)
    return json.dumps(loc, indent=2)
