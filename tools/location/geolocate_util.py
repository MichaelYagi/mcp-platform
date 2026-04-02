"""
Utility for IP-based geolocation.
Separated to avoid circular imports.
"""
import os
import requests

def geolocate_ip(ip: str):
    """
    Get location information from an IP address using ip-api.com.

    Args:
        ip: IP address to geolocate

    Returns:
        dict with normalized keys {city, region, country} or None if failed.
        Keys are normalized to match what detect_location.py expects.
    """
    # Read CLIENT_IP at call time, not import time, so load_dotenv has run
    _ip = ip or os.environ.get("CLIENT_IP")
    if not _ip:
        return None

    try:
        resp = requests.get(f"http://ip-api.com/json/{_ip}", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "fail":
            return None
        # ip-api.com uses "regionName" and "country" (full name) — normalize
        # to the keys detect_location.py expects: city, region, country
        return {
            "city":    data.get("city"),
            "region":  data.get("regionName"),
            "country": data.get("country"),
        }
    except Exception:
        return None


# CLIENT_IP kept for callers that import it directly (e.g. location server.py)
CLIENT_IP = os.environ.get("CLIENT_IP")