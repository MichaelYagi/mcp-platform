import os
from datetime import datetime

from tools.location.geolocate_util import geolocate_ip, CLIENT_IP
from tools.location.get_time_data import TZ_TO_LOCATION, DEFAULT_FALLBACK

def detect_default_location():
    """
    Detects the user's location based on IP, and falls back on system timezone.
    Falls back to Surrey, Canada if unknown.

    Priority:
    1. DEFAULT_CITY / DEFAULT_STATE / DEFAULT_COUNTRY env vars (explicit override)
    2. IP geolocation
    3. System timezone lookup
    4. Hardcoded DEFAULT_FALLBACK
    """
    # Explicit env var override — highest priority
    env_city    = os.environ.get("DEFAULT_CITY")
    env_state   = os.environ.get("DEFAULT_STATE")
    env_country = os.environ.get("DEFAULT_COUNTRY")
    if env_city or env_state or env_country:
        return {"city": env_city, "state": env_state, "country": env_country}

    loc = geolocate_ip(CLIENT_IP)
    if loc:
        city = loc.get("city")
        state = loc.get("region")
        country = loc.get("country")
        return {"city": city, "state": state, "country": country}

    local_tz = datetime.now().astimezone().tzinfo
    tz_name = getattr(local_tz, "key", None)

    if tz_name and tz_name in TZ_TO_LOCATION:
        loc_data = TZ_TO_LOCATION[tz_name]
        return {"city": loc_data["city"], "state": loc_data["state"], "country": loc_data["country"]}

    return dict(DEFAULT_FALLBACK)