from datetime import datetime

from tools.location.geolocate_util import geolocate_ip, CLIENT_IP
from tools.location.get_time_data import TZ_TO_LOCATION, DEFAULT_FALLBACK

def detect_default_location():
    """
    Detects the user's location based on IP, and falls back on system timezone.
    Falls back to Surrey, Canada if unknown.
    """
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
        return {"city": loc_data[0], "state": loc_data[1], "country": loc_data[2]}

    fallback = DEFAULT_FALLBACK
    return {"city": fallback[0], "state": fallback[1], "country": fallback[2]}
