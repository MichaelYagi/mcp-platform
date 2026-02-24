import json
import requests
from typing import Optional
from datetime import datetime, date
from tools.location.resolve_location import resolve_location

# WMO Weather interpretation codes -> human readable description + chance label
WMO_CODES = {
    0: ("☀️ Clear sky", 0),
    1: ("🌤️ Mainly clear", 5),
    2: ("⛅ Partly cloudy", 20),
    3: ("☁️ Overcast", 0),
    45: ("🌫️ Fog", 0),
    48: ("🌫️ Depositing rime fog", 0),
    51: ("🌦️ Light drizzle", 40),
    53: ("🌦️ Moderate drizzle", 60),
    55: ("🌧️ Dense drizzle", 80),
    56: ("🌨️ Light freezing drizzle", 40),
    57: ("🌨️ Dense freezing drizzle", 80),
    61: ("🌧️ Slight rain", 50),
    63: ("🌧️ Moderate rain", 70),
    65: ("🌧️ Heavy rain", 90),
    66: ("🌨️ Light freezing rain", 50),
    67: ("🌨️ Heavy freezing rain", 85),
    71: ("🌨️ Slight snowfall", 50),
    73: ("❄️ Moderate snowfall", 70),
    75: ("❄️ Heavy snowfall", 90),
    77: ("🌨️ Snow grains", 60),
    80: ("🌦️ Slight rain showers", 50),
    81: ("🌧️ Moderate rain showers", 65),
    82: ("⛈️ Violent rain showers", 90),
    85: ("🌨️ Slight snow showers", 50),
    86: ("❄️ Heavy snow showers", 80),
    95: ("⛈️ Thunderstorm", 75),
    96: ("⛈️ Thunderstorm with slight hail", 80),
    99: ("⛈️ Thunderstorm with heavy hail", 90),
}


def _wmo_description(code: int) -> str:
    return WMO_CODES.get(code, ("Unknown", 0))[0]


def _celsius_to_fahrenheit(c: float) -> float:
    return round(c * 9 / 5 + 32, 1)


def _get_date_label(date_str: str) -> str:
    """
    Convert date string to friendly label like 'Today', 'Tomorrow', or day name

    Args:
        date_str: Date in format "2026-02-23"

    Returns:
        Label like "Today", "Tomorrow", "Wednesday", etc.
    """
    try:
        forecast_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        today = date.today()

        days_diff = (forecast_date - today).days

        if days_diff == 0:
            return "Today"
        elif days_diff == 1:
            return "Tomorrow"
        elif days_diff == -1:
            return "Yesterday"
        elif 2 <= days_diff <= 6:
            return forecast_date.strftime("%A")  # "Monday", "Tuesday", etc.
        else:
            return forecast_date.strftime("%A, %B %d")  # "Monday, February 23"
    except:
        return date_str


def _geocode(city: str, state: Optional[str] = None, country: Optional[str] = None):
    """
    Use Open-Meteo's geocoding API to resolve a city name to lat/lon.
    Returns dict with lat, lon, resolved_city, resolved_state, resolved_country
    or None on failure.
    """
    query = city
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={requests.utils.quote(query)}&count=10&language=en&format=json"
    resp = requests.get(url, timeout=5)
    data = resp.json()

    results = data.get("results", [])
    if not results:
        return None

    # Try to match state/country if provided
    best = None
    for r in results:
        r_country = r.get("country", "").lower()
        r_state = r.get("admin1", "").lower()
        r_cc = r.get("country_code", "").lower()

        country_match = (
                not country or
                country.lower() in r_country or
                r_country in country.lower() or
                country.lower() == r_cc
        )
        state_match = (
                not state or
                state.lower() in r_state or
                r_state in state.lower() or
                state.lower() == r.get("admin1_code", "").lower()
        )

        if country_match and state_match:
            best = r
            break

    if not best:
        best = results[0]  # Fall back to top result

    return {
        "lat": best["latitude"],
        "lon": best["longitude"],
        "city": best.get("name", city),
        "state": best.get("admin1", state or ""),
        "country": best.get("country", country or ""),
        "timezone": best.get("timezone", "auto"),
    }


def get_weather(
        city: Optional[str] = None,
        state: Optional[str] = None,
        country: Optional[str] = None,
        forecast_days: int = 7
) -> str:
    """
    Fetches current weather and a multi-day forecast using the free Open-Meteo API.
    No API key required.

    Location resolution priority:
      1. Use provided city/state/country arguments
      2. If none provided, defaults to Vancouver, BC, Canada

    When parsing locations:
    • City  = city name (e.g., Surrey)
    • State = province, prefecture, or state (e.g., BC, Ontario, Kanagawa, California)
    • Country = full country name (e.g., Canada, Japan, United States)

    Never put a province or state into the country field.

    Args:
        city (str, optional): City name
        state (str, optional): State / province / prefecture
        country (str, optional): Full country name
        forecast_days (int): Number of forecast days to return (1-16, default 7)
    """
    # --- Resolve location ---
    # If nothing is provided, default to Vancouver BC
    if not city and not state and not country:
        city = "Vancouver"
        state = "BC"
        country = "Canada"

    # Use existing resolve_location for any normalization it provides,
    # then geocode to get lat/lon
    loc = resolve_location(city, state, country)

    geo = _geocode(loc.get("city") or city, loc.get("state") or state, loc.get("country") or country)
    if not geo:
        return json.dumps({
            "error": "geocode_failed",
            "message": f"Could not resolve location: city={city}, state={state}, country={country}",
            "city": city,
            "state": state,
            "country": country,
        }, indent=2)

    lat = geo["lat"]
    lon = geo["lon"]
    timezone = geo.get("timezone") or "auto"
    forecast_days = max(1, min(16, forecast_days))

    # --- Fetch weather from Open-Meteo ---
    current_vars = [
        "temperature_2m",
        "apparent_temperature",
        "relative_humidity_2m",
        "weather_code",
        "precipitation_probability",
        "wind_speed_10m",
        "is_day",
    ]
    daily_vars = [
        "weather_code",
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_probability_max",
        "wind_speed_10m_max",
        "sunrise",
        "sunset",
    ]

    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current={','.join(current_vars)}"
        f"&daily={','.join(daily_vars)}"
        f"&forecast_days={forecast_days}"
        f"&timezone={requests.utils.quote(timezone)}"
        f"&wind_speed_unit=kmh"
    )

    try:
        response = requests.get(url, timeout=10)
        data = response.json()
    except Exception as e:
        return json.dumps({
            "error": "request_failed",
            "message": str(e),
            "city": geo["city"],
            "state": geo["state"],
            "country": geo["country"],
        }, indent=2)

    if "error" in data:
        return json.dumps({
            "error": "api_error",
            "message": data.get("reason", "Unknown error from Open-Meteo"),
            "city": geo["city"],
            "state": geo["state"],
            "country": geo["country"],
        }, indent=2)

    current = data.get("current", {})
    daily = data.get("daily", {})

    # --- Build current weather ---
    cur_temp_c = current.get("temperature_2m")
    cur_feels_c = current.get("apparent_temperature")
    cur_code = current.get("weather_code", 0)

    current_weather = {
        "condition": _wmo_description(cur_code),
        "precipitation_chance": f"{current.get('precipitation_probability', 0)}%",
        "temperature_c": cur_temp_c,
        "temperature_f": _celsius_to_fahrenheit(cur_temp_c) if cur_temp_c is not None else None,
        "feelslike_c": cur_feels_c,
        "feelslike_f": _celsius_to_fahrenheit(cur_feels_c) if cur_feels_c is not None else None,
        "humidity": f"{current.get('relative_humidity_2m', 0)}%",
        "wind_speed_kph": current.get("wind_speed_10m"),
        "is_day": bool(current.get("is_day", 1)),
    }

    # --- Build daily forecast ---
    dates = daily.get("time", [])
    codes = daily.get("weather_code", [])
    max_temps = daily.get("temperature_2m_max", [])
    min_temps = daily.get("temperature_2m_min", [])
    precip_probs = daily.get("precipitation_probability_max", [])
    wind_maxes = daily.get("wind_speed_10m_max", [])
    sunrises = daily.get("sunrise", [])
    sunsets = daily.get("sunset", [])

    forecast = []
    today = date.today()

    for i, date_str in enumerate(dates):
        code = codes[i] if i < len(codes) else 0
        max_c = max_temps[i] if i < len(max_temps) else None
        min_c = min_temps[i] if i < len(min_temps) else None

        # Calculate relative day
        try:
            forecast_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            days_from_today = (forecast_date - today).days

            if days_from_today == 0:
                relative_day = "today"
            elif days_from_today == 1:
                relative_day = "tomorrow"
            elif days_from_today == 2:
                relative_day = "day_after_tomorrow"
            else:
                relative_day = f"{days_from_today}_days_from_now"
        except:
            relative_day = "unknown"

        forecast.append({
            "date": date_str,
            "day_label": _get_date_label(date_str),
            "relative_day": relative_day,
            "condition": _wmo_description(code),
            "precipitation_chance": f"{precip_probs[i]}%" if i < len(precip_probs) and precip_probs[
                i] is not None else "N/A",
            "max_temp_c": max_c,
            "max_temp_f": _celsius_to_fahrenheit(max_c) if max_c is not None else None,
            "min_temp_c": min_c,
            "min_temp_f": _celsius_to_fahrenheit(min_c) if min_c is not None else None,
            "max_wind_kph": wind_maxes[i] if i < len(wind_maxes) else None,
            "sunrise": sunrises[i] if i < len(sunrises) else None,
            "sunset": sunsets[i] if i < len(sunsets) else None,
        })

    result = {
        "city": geo["city"],
        "state": geo["state"],
        "country": geo["country"],
        "latitude": lat,
        "longitude": lon,
        "timezone": timezone,
        "current": current_weather,
        "forecast": forecast,
    }

    return json.dumps(result, indent=2)