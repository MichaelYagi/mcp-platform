"""
Utility for IP-based geolocation.
Separated to avoid circular imports.
"""
import os
import requests

CLIENT_IP = os.environ.get("CLIENT_IP")


def geolocate_ip(ip: str):
    """
    Get location information from an IP address using ipapi.co

    Args:
        ip: IP address to geolocate

    Returns:
        dict with location info or None if failed
    """
    if not ip:
        return None

    try:
        # resp = requests.get(f"https://ipapi.co/{ip}/json/", timeout=5)
        resp = requests.get(f"http://ip-api.com/json/{ip}", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None