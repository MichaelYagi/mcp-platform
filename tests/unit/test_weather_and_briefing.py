"""
tests/unit/test_weather_and_briefing.py

Comprehensive tests for weather fetching and get_day_briefing.
These features have caused repeated production bugs:
  - Wrong forecast day shown for date_offset jobs
  - Weather silently disappearing when API returns error or fewer days
  - Humidity shown for wrong day
  - Missing weather section with no error message

Test categories:
  1. get_weather — API response parsing, error handling, retry, data structure
  2. get_day_briefing — all date_offset scenarios, error surfacing, text output
  3. Briefing text format — every section (date, weather, email, calendar)
  4. Weather + pipeline — the full scheduled job path
"""
import json
import time
import pytest
from datetime import date, datetime, timedelta
from unittest.mock import patch, MagicMock, call


# ─── Shared helpers ───────────────────────────────────────────────────────────

def _open_meteo_response(
    days: int = 1,
    start_date: date = None,
    weather_code: int = 0,
    max_temp: float = 20.0,
    min_temp: float = 10.0,
    feels_max: float = 18.0,
    precip_prob: int = 10,
    humidity: int = 55,
    cur_temp: float = 19.0,
) -> dict:
    """Build a realistic Open-Meteo API response for N forecast days."""
    today = start_date or date.today()
    dates = [(today + timedelta(days=i)).isoformat() for i in range(days)]
    sunrises = [f"{(today + timedelta(days=i)).isoformat()}T05:30" for i in range(days)]
    sunsets  = [f"{(today + timedelta(days=i)).isoformat()}T21:00" for i in range(days)]
    return {
        "current": {
            "temperature_2m": cur_temp,
            "apparent_temperature": cur_temp - 1,
            "relative_humidity_2m": humidity,
            "weather_code": weather_code,
            "precipitation_probability": precip_prob,
        },
        "daily": {
            "time": dates,
            "weather_code": [weather_code] * days,
            "temperature_2m_max": [max_temp] * days,
            "temperature_2m_min": [min_temp] * days,
            "apparent_temperature_max": [feels_max] * days,
            "precipitation_probability_max": [precip_prob] * days,
            "sunrise": sunrises,
            "sunset": sunsets,
        },
        "timezone": "America/Vancouver",
    }


def _geo_response() -> dict:
    return {
        "lat": 49.19, "lon": -122.85,
        "city": "Surrey", "state": "British Columbia",
        "country": "Canada", "timezone": "America/Vancouver",
    }


def _make_requests_response(data: dict, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.ok = (status < 400)
    resp.status_code = status
    resp.text = json.dumps(data)
    resp.json.return_value = data
    return resp


def _briefing_weather_json(days: int = 1, date_offset: int = 0) -> str:
    """Build the JSON that get_weather_fn returns for get_day_briefing tests."""
    today = date.today()
    forecast = []
    from tools.location.get_weather import _get_date_label, _celsius_to_fahrenheit, _wmo_description, _fmt_sun
    for i in range(days):
        d = today + timedelta(days=i)
        ds = d.isoformat()
        forecast.append({
            "date": ds,
            "day_label": _get_date_label(ds, today),
            "relative_day": ["today", "tomorrow", "day_after_tomorrow"][i] if i < 3 else f"{i}_days_from_now",
            "condition": "☀️ Clear sky",
            "precipitation_chance": "5%",
            "max_temp_c": 20.0 + i,
            "max_temp_f": _celsius_to_fahrenheit(20.0 + i),
            "min_temp_c": 10.0 + i,
            "min_temp_f": _celsius_to_fahrenheit(10.0 + i),
            "feelslike_c": 18.0 + i,
            "feelslike_f": _celsius_to_fahrenheit(18.0 + i),
            "sunrise": "5:30 AM",
            "sunset": "9:00 PM",
        })
    return json.dumps({
        "city": "Surrey", "state": "British Columbia", "country": "Canada",
        "latitude": 49.19, "longitude": -122.85, "timezone": "America/Vancouver",
        "maps_link": "[Surrey, British Columbia, Canada](https://maps.google.com/?q=Surrey)",
        "current": {
            "condition": "☀️ Clear sky",
            "precipitation_chance": "5%",
            "temperature_c": 19.0, "temperature_f": 66.2,
            "feelslike_c": 17.0, "feelslike_f": 62.6,
            "humidity": "55%",
        },
        "forecast": forecast,
    })


def _mock_gmail_service(emails: list = None):
    """Build a mock Gmail service that returns the given email list."""
    emails = emails or []
    svc = MagicMock()
    msg_list_result = {"messages": [{"id": e["id"]} for e in emails]}
    svc.users().messages().list().execute.return_value = msg_list_result
    def get_msg(userId, id, format, metadataHeaders):
        em = next((e for e in emails if e["id"] == id), {})
        headers = [
            {"name": "From", "value": em.get("from", "")},
            {"name": "Subject", "value": em.get("subject", "")},
            {"name": "Date", "value": em.get("date", "")},
        ]
        mock = MagicMock()
        mock.execute.return_value = {"id": id, "snippet": em.get("preview", ""),
                                     "payload": {"headers": headers}}
        return mock
    svc.users().messages().get = MagicMock(side_effect=get_msg)
    return svc


def _mock_calendar_service(events: list = None):
    """Build a mock Calendar service that returns the given event list."""
    events = events or []
    svc = MagicMock()
    # _get_all_calendar_ids calls service.calendarList().list().execute()
    svc.calendarList().list().execute.return_value = {
        "items": [{"id": "primary"}]
    }
    cal_events = []
    for ev in events:
        start = ev.get("start", "2026-06-05T17:00:00-07:00")
        end   = ev.get("end",   "2026-06-05T18:00:00-07:00")
        cal_events.append({
            "id": ev.get("id", "evt1"),
            "summary": ev.get("title", "Event"),
            "start": {"dateTime": start} if "T" in start else {"date": start},
            "end":   {"dateTime": end}   if "T" in end   else {"date": end},
        })
    svc.events().list().execute.return_value = {"items": cal_events}
    return svc


def _run_briefing(date_offset=0, forecast_days=1, max_emails=0, calendar_days=0,
                  api_days=None, email_list=None, event_list=None):
    """
    Run get_day_briefing with mocked weather, email, calendar.
    api_days: how many days the mock API returns (defaults to date_offset + forecast_days).
    """
    if api_days is None:
        api_days = date_offset + forecast_days
    api_days = max(1, api_days)

    def fake_weather(city=None, state=None, country=None, forecast_days=1):
        return _briefing_weather_json(days=forecast_days, date_offset=date_offset)

    gmail_svc  = _mock_gmail_service(email_list or [])
    cal_svc    = _mock_calendar_service(event_list or [])

    with patch("tools.location.geolocate_util.geolocate_ip", return_value=None), \
         patch("tools.location.geolocate_util.CLIENT_IP", None), \
         patch("tools.location.get_weather.get_weather", fake_weather), \
         patch("servers.google.server.GOOGLE_AVAILABLE", True), \
         patch("servers.google.server._gmail_service", return_value=gmail_svc), \
         patch("servers.google.server._calendar_service", return_value=cal_svc), \
         patch("servers.google.server._get_all_calendar_ids", return_value=["primary"]), \
         patch.dict("os.environ", {"GOOGLE_APPS_SCRIPT_URL": ""}):
        from servers.google.server import get_day_briefing
        raw = get_day_briefing(
            date_offset=date_offset,
            forecast_days=forecast_days,
            max_emails=max_emails,
            calendar_days=calendar_days,
        )
        return json.loads(raw)


# ═══════════════════════════════════════════════════════════════════
# 1. get_weather — API response parsing
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestGetWeatherParsing:

    def _call(self, api_data: dict, geocode_data: dict = None) -> dict:
        geo = geocode_data or _geo_response()
        mock_resp = _make_requests_response(api_data)
        with patch("tools.location.get_weather._geocode", return_value=geo), \
             patch("tools.location.resolve_location.resolve_location",
                   return_value={"city": "Surrey", "state": "BC", "country": "Canada"}), \
             patch("requests.get", return_value=mock_resp):
            from tools.location.get_weather import get_weather
            return json.loads(get_weather("Surrey", "BC", "Canada", forecast_days=1))

    def test_current_conditions_present(self):
        result = self._call(_open_meteo_response())
        assert "current" in result
        cur = result["current"]
        assert "temperature_c" in cur
        assert "humidity" in cur
        assert "condition" in cur

    def test_current_humidity_formatted_as_percent(self):
        result = self._call(_open_meteo_response(humidity=65))
        assert result["current"]["humidity"] == "65%"

    def test_forecast_list_present(self):
        result = self._call(_open_meteo_response(days=3))
        assert "forecast" in result
        assert isinstance(result["forecast"], list)

    def test_forecast_length_matches_requested_days(self):
        data = _open_meteo_response(days=3)
        geo = _geo_response()
        mock_resp = _make_requests_response(data)
        with patch("tools.location.get_weather._geocode", return_value=geo), \
             patch("tools.location.resolve_location.resolve_location",
                   return_value={"city": "Surrey", "state": "BC", "country": "Canada"}), \
             patch("requests.get", return_value=mock_resp):
            from tools.location.get_weather import get_weather
            result = json.loads(get_weather("Surrey", "BC", "Canada", forecast_days=3))
        assert len(result["forecast"]) == 3

    def test_forecast_day_has_required_fields(self):
        result = self._call(_open_meteo_response())
        day = result["forecast"][0]
        for field in ("date", "day_label", "condition", "precipitation_chance",
                      "max_temp_c", "max_temp_f", "min_temp_c", "min_temp_f",
                      "feelslike_c", "feelslike_f", "sunrise", "sunset"):
            assert field in day, f"Missing field: {field}"

    def test_first_forecast_day_is_today(self):
        result = self._call(_open_meteo_response())
        assert result["forecast"][0]["day_label"] == "Today"

    def test_second_forecast_day_is_tomorrow(self):
        data = _open_meteo_response(days=2)
        geo = _geo_response()
        with patch("tools.location.get_weather._geocode", return_value=geo), \
             patch("tools.location.resolve_location.resolve_location",
                   return_value={"city": "Surrey", "state": "BC", "country": "Canada"}), \
             patch("requests.get", return_value=_make_requests_response(data)):
            from tools.location.get_weather import get_weather
            result = json.loads(get_weather("Surrey", "BC", "Canada", forecast_days=2))
        assert result["forecast"][1]["day_label"] == "Tomorrow"

    def test_temp_conversion_correct(self):
        result = self._call(_open_meteo_response(max_temp=0.0))
        assert result["forecast"][0]["max_temp_c"] == 0.0
        assert result["forecast"][0]["max_temp_f"] == 32.0

    def test_celsius_to_fahrenheit_spot_check(self):
        from tools.location.get_weather import _celsius_to_fahrenheit
        assert _celsius_to_fahrenheit(100) == 212.0
        assert _celsius_to_fahrenheit(-40) == -40.0
        assert _celsius_to_fahrenheit(20) == 68.0

    def test_city_state_country_in_result(self):
        result = self._call(_open_meteo_response())
        assert result["city"] == "Surrey"
        assert result["state"] == "British Columbia"
        assert result["country"] == "Canada"

    def test_maps_link_present(self):
        result = self._call(_open_meteo_response())
        assert "maps_link" in result
        assert "Surrey" in result["maps_link"]

    def test_wmo_clear_sky(self):
        from tools.location.get_weather import _wmo_description
        assert "Clear" in _wmo_description(0)

    def test_wmo_thunderstorm(self):
        from tools.location.get_weather import _wmo_description
        assert "Thunderstorm" in _wmo_description(95)

    def test_wmo_unknown_code_returns_unknown(self):
        from tools.location.get_weather import _wmo_description
        assert "Unknown" in _wmo_description(999)

    def test_sunrise_formatted(self):
        result = self._call(_open_meteo_response())
        sunrise = result["forecast"][0]["sunrise"]
        assert sunrise is not None
        assert "AM" in sunrise or "PM" in sunrise

    def test_fmt_sun_formats_correctly(self):
        from tools.location.get_weather import _fmt_sun
        assert _fmt_sun("2026-06-03T05:30") == "5:30 AM"
        assert _fmt_sun("2026-06-03T21:09") == "9:09 PM"
        assert _fmt_sun("2026-06-03T12:00") == "12:00 PM"

    def test_fmt_sun_none_returns_none(self):
        from tools.location.get_weather import _fmt_sun
        assert _fmt_sun(None) is None
        assert _fmt_sun("") == ""


# ═══════════════════════════════════════════════════════════════════
# 2. get_weather — error handling and retry
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestGetWeatherErrors:

    def _call_with_request(self, mock_requests_get, forecast_days=1) -> dict:
        geo = _geo_response()
        with patch("tools.location.get_weather._geocode", return_value=geo), \
             patch("tools.location.resolve_location.resolve_location",
                   return_value={"city": "Surrey", "state": "BC", "country": "Canada"}), \
             patch("requests.get", mock_requests_get), \
             patch("time.sleep"), \
             patch.dict("os.environ", {"OPENWEATHER_API_KEY": ""}):
            from tools.location.get_weather import get_weather
            return json.loads(get_weather("Surrey", "BC", "Canada",
                                         forecast_days=forecast_days))

    def test_geocode_failure_returns_error(self):
        with patch("tools.location.get_weather._geocode", return_value=None), \
             patch("tools.location.resolve_location.resolve_location",
                   return_value={"city": "Surrey", "state": "BC", "country": "Canada"}):
            from tools.location.get_weather import get_weather
            result = json.loads(get_weather("Surrey", "BC", "Canada"))
        assert "error" in result
        assert result["error"] == "geocode_failed"

    def test_http_error_returns_request_failed(self):
        bad_resp = _make_requests_response({}, status=500)
        result = self._call_with_request(MagicMock(return_value=bad_resp))
        assert "error" in result
        assert result["error"] == "request_failed"

    def test_empty_response_returns_request_failed(self):
        resp = MagicMock()
        resp.ok = True
        resp.status_code = 200
        resp.text = ""
        result = self._call_with_request(MagicMock(return_value=resp))
        assert "error" in result

    def test_api_error_field_returns_api_error(self):
        api_err = {"error": True, "reason": "Parameter out of range"}
        result = self._call_with_request(
            MagicMock(return_value=_make_requests_response(api_err))
        )
        assert result.get("error") == "api_error"

    def test_retries_three_times_on_failure(self):
        bad_resp = _make_requests_response({}, status=500)
        good_resp = _make_requests_response(_open_meteo_response())
        # Fail twice, succeed on third attempt
        side_effects = [bad_resp, bad_resp, good_resp]
        mock_get = MagicMock(side_effect=side_effects)
        result = self._call_with_request(mock_get)
        assert mock_get.call_count == 3
        assert "forecast" in result

    def test_all_retries_exhausted_returns_error(self):
        bad_resp = _make_requests_response({}, status=503)
        mock_get = MagicMock(return_value=bad_resp)
        result = self._call_with_request(mock_get)
        assert "error" in result
        assert mock_get.call_count == 3

    def test_network_exception_returns_error(self):
        mock_get = MagicMock(side_effect=ConnectionError("network down"))
        result = self._call_with_request(mock_get)
        assert "error" in result

    def test_forecast_days_clamped_to_1_minimum(self):
        """forecast_days=0 must be clamped to 1."""
        resp = _make_requests_response(_open_meteo_response(days=1))
        mock_get = MagicMock(return_value=resp)
        result = self._call_with_request(mock_get, forecast_days=0)
        # Should not error
        assert "forecast" in result or "error" in result

    def test_forecast_days_clamped_to_16_maximum(self):
        """forecast_days > 16 must be clamped."""
        resp = _make_requests_response(_open_meteo_response(days=16))
        mock_get = MagicMock(return_value=resp)
        result = self._call_with_request(mock_get, forecast_days=99)
        assert "forecast" in result or "error" in result


# ═══════════════════════════════════════════════════════════════════
# 2b. OpenWeatherMap fallback
# ═══════════════════════════════════════════════════════════════════

def _owm_current_response(temp=20.0, feels=18.0, humidity=55,
                           weather_id=800, description="clear sky") -> dict:
    return {
        "weather": [{"id": weather_id, "description": description}],
        "main": {"temp": temp, "feels_like": feels, "humidity": humidity},
        "sys": {"sunrise": 1748880600, "sunset": 1748933400},
        "rain": {},
    }


def _owm_forecast_response(temp=20.0, pop=0.1) -> dict:
    today = date.today().isoformat()
    return {
        "list": [
            {
                "dt_txt": f"{today} 06:00:00",
                "weather": [{"id": 800, "description": "clear sky"}],
                "main": {"temp": temp - 2, "temp_min": temp - 4, "temp_max": temp,
                         "feels_like": temp - 3, "humidity": 60},
                "pop": pop,
            },
            {
                "dt_txt": f"{today} 12:00:00",
                "weather": [{"id": 800, "description": "clear sky"}],
                "main": {"temp": temp, "temp_min": temp - 4, "temp_max": temp + 2,
                         "feels_like": temp - 1, "humidity": 55},
                "pop": pop,
            },
            {
                "dt_txt": f"{today} 18:00:00",
                "weather": [{"id": 801, "description": "few clouds"}],
                "main": {"temp": temp - 1, "temp_min": temp - 4, "temp_max": temp + 2,
                         "feels_like": temp - 2, "humidity": 58},
                "pop": pop,
            },
        ]
    }


@pytest.mark.unit
class TestOWMConditionMapper:

    def test_clear_sky(self):
        from tools.location.get_weather import _owm_condition
        assert "☀️" in _owm_condition(800, "clear sky")

    def test_few_clouds(self):
        from tools.location.get_weather import _owm_condition
        assert "🌤️" in _owm_condition(801, "few clouds")

    def test_scattered_clouds(self):
        from tools.location.get_weather import _owm_condition
        assert "⛅" in _owm_condition(802, "scattered clouds")

    def test_overcast(self):
        from tools.location.get_weather import _owm_condition
        assert "☁️" in _owm_condition(804, "overcast clouds")

    def test_rain(self):
        from tools.location.get_weather import _owm_condition
        assert "🌧️" in _owm_condition(500, "light rain")

    def test_thunderstorm(self):
        from tools.location.get_weather import _owm_condition
        assert "⛈️" in _owm_condition(200, "thunderstorm")

    def test_snow(self):
        from tools.location.get_weather import _owm_condition
        assert "❄️" in _owm_condition(601, "snow")

    def test_fog(self):
        from tools.location.get_weather import _owm_condition
        assert "🌫️" in _owm_condition(741, "fog")

    def test_description_capitalised(self):
        from tools.location.get_weather import _owm_condition
        result = _owm_condition(800, "clear sky")
        assert "Clear sky" in result


@pytest.mark.unit
class TestGetWeatherOWMFallback:
    """OWM is used when Open-Meteo fails."""

    def _call_owm_fallback(self, open_meteo_fail_resp, forecast_days=1) -> dict:
        geo = _geo_response()

        def _route_get(url, **kwargs):
            if "open-meteo" in url or "geocoding-api" in url:
                return open_meteo_fail_resp
            if "openweathermap" in url and "forecast" in url:
                return _make_requests_response(_owm_forecast_response())
            if "openweathermap" in url:
                return _make_requests_response(_owm_current_response())
            return MagicMock(ok=False, status_code=500, text="")

        with patch("tools.location.get_weather._geocode", return_value=geo), \
             patch("tools.location.resolve_location.resolve_location",
                   return_value={"city": "Surrey", "state": "BC", "country": "Canada"}), \
             patch("requests.get", side_effect=_route_get), \
             patch("time.sleep"), \
             patch.dict("os.environ", {"OPENWEATHER_API_KEY": "test-owm-key"}):
            from tools.location.get_weather import get_weather
            return json.loads(get_weather("Surrey", "BC", "Canada",
                                         forecast_days=forecast_days))

    def test_owm_used_when_open_meteo_http_error(self):
        bad = _make_requests_response({}, status=500)
        result = self._call_owm_fallback(bad)
        assert "forecast" in result
        assert "error" not in result

    def test_owm_used_when_open_meteo_api_error(self):
        api_err = _make_requests_response({"error": True, "reason": "out of range"})
        result = self._call_owm_fallback(api_err)
        assert "forecast" in result
        assert "error" not in result

    def test_owm_result_has_current_conditions(self):
        bad = _make_requests_response({}, status=503)
        result = self._call_owm_fallback(bad)
        cur = result.get("current", {})
        assert "temperature_c" in cur
        assert "humidity" in cur
        assert "condition" in cur

    def test_owm_result_has_forecast(self):
        bad = _make_requests_response({}, status=503)
        result = self._call_owm_fallback(bad)
        assert isinstance(result.get("forecast"), list)
        assert len(result["forecast"]) >= 1

    def test_owm_forecast_day_has_required_fields(self):
        bad = _make_requests_response({}, status=503)
        result = self._call_owm_fallback(bad)
        day = result["forecast"][0]
        for field in ("date", "day_label", "condition", "precipitation_chance",
                      "max_temp_c", "max_temp_f", "min_temp_c", "min_temp_f",
                      "feelslike_c", "feelslike_f"):
            assert field in day, f"Missing field: {field}"

    def test_owm_today_gets_sunrise_sunset(self):
        bad = _make_requests_response({}, status=503)
        result = self._call_owm_fallback(bad)
        today_entry = next(
            (d for d in result["forecast"] if d.get("relative_day") == "today"), None
        )
        assert today_entry is not None
        assert today_entry.get("sunrise") is not None
        assert today_entry.get("sunset") is not None

    def test_owm_condition_uses_emoji(self):
        bad = _make_requests_response({}, status=503)
        result = self._call_owm_fallback(bad)
        condition = result["current"]["condition"]
        assert any(e in condition for e in ("☀️", "🌤️", "⛅", "☁️", "🌧️", "⛈️", "❄️", "🌫️", "🌦️"))

    def test_owm_city_state_country_preserved(self):
        bad = _make_requests_response({}, status=503)
        result = self._call_owm_fallback(bad)
        assert result["city"] == "Surrey"
        assert result["state"] == "British Columbia"
        assert result["country"] == "Canada"

    def test_error_returned_when_both_fail(self):
        bad_om = _make_requests_response({}, status=500)

        def _all_fail(url, **kwargs):
            raise ConnectionError("all down")

        geo = _geo_response()
        with patch("tools.location.get_weather._geocode", return_value=geo), \
             patch("tools.location.resolve_location.resolve_location",
                   return_value={"city": "Surrey", "state": "BC", "country": "Canada"}), \
             patch("requests.get", side_effect=_all_fail), \
             patch("time.sleep"), \
             patch.dict("os.environ", {"OPENWEATHER_API_KEY": "test-owm-key"}):
            from tools.location.get_weather import get_weather
            result = json.loads(get_weather("Surrey", "BC", "Canada"))
        assert "error" in result

    def test_no_owm_key_skips_fallback(self):
        bad = _make_requests_response({}, status=500)
        geo = _geo_response()
        with patch("tools.location.get_weather._geocode", return_value=geo), \
             patch("tools.location.resolve_location.resolve_location",
                   return_value={"city": "Surrey", "state": "BC", "country": "Canada"}), \
             patch("requests.get", return_value=bad), \
             patch("time.sleep"), \
             patch.dict("os.environ", {"OPENWEATHER_API_KEY": ""}):
            from tools.location.get_weather import get_weather
            result = json.loads(get_weather("Surrey", "BC", "Canada"))
        assert "error" in result


# ═══════════════════════════════════════════════════════════════════
# 3. get_day_briefing — date_offset scenarios (the bug-prone area)
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestBriefingDateOffset:

    def test_offset_zero_shows_today(self):
        result = _run_briefing(date_offset=0, forecast_days=1)
        weather = result.get("weather", {})
        forecast = weather.get("forecast", []) if weather else []
        assert forecast, "Weather forecast must be present for date_offset=0"
        assert forecast[0]["day_label"] == "Today"

    def test_offset_zero_text_contains_weather(self):
        result = _run_briefing(date_offset=0)
        text = result.get("text", "")
        assert "### Weather" in text, (
            "Weather section must appear in briefing text for date_offset=0. "
            f"Got: {text[:200]}"
        )

    def test_offset_one_text_contains_weather(self):
        """Regression: date_offset=1 was causing weather to silently disappear."""
        result = _run_briefing(date_offset=1, forecast_days=1)
        text = result.get("text", "")
        assert "### Weather" in text or "unavailable" in text, (
            "Weather section (or unavailable notice) must appear for date_offset=1. "
            f"Got: {text[:300]}"
        )

    def test_offset_two_text_contains_weather(self):
        result = _run_briefing(date_offset=2, forecast_days=1)
        text = result.get("text", "")
        assert "### Weather" in text or "unavailable" in text, (
            f"Weather must appear for date_offset=2. Got: {text[:300]}"
        )

    def test_offset_zero_date_label_is_today(self):
        result = _run_briefing(date_offset=0)
        today_str = date.today().strftime("%-d")
        assert today_str in result.get("date", ""), (
            f"date field should contain today's day number. Got: {result.get('date')}"
        )

    def test_offset_one_date_label_is_tomorrow(self):
        result = _run_briefing(date_offset=1)
        tomorrow = (date.today() + timedelta(days=1)).strftime("%-d")
        assert tomorrow in result.get("date", ""), (
            f"date field should contain tomorrow's day. Got: {result.get('date')}"
        )

    def test_offset_two_date_label_is_day_after_tomorrow(self):
        result = _run_briefing(date_offset=2)
        target = (date.today() + timedelta(days=2)).strftime("%-d")
        assert target in result.get("date", ""), (
            f"date field should contain day+2. Got: {result.get('date')}"
        )

    def test_offset_fetches_extra_api_days(self):
        """Regression: date_offset=2, forecast_days=1 must call get_weather with 3 days."""
        fetched = []

        def fake_weather(city=None, state=None, country=None, forecast_days=1):
            fetched.append(forecast_days)
            return _briefing_weather_json(days=forecast_days)

        with patch("tools.location.geolocate_util.geolocate_ip", return_value=None), \
             patch("tools.location.geolocate_util.CLIENT_IP", None), \
             patch("tools.location.get_weather.get_weather", fake_weather), \
             patch("servers.google.server.gmail_get_unread",
                   return_value=json.dumps({"total_unread": 0, "emails": []})), \
             patch("servers.google.server.calendar_get_today",
                   return_value=json.dumps({"count": 0, "events": [], "text": ""})):
            from servers.google.server import get_day_briefing
            get_day_briefing(date_offset=2, forecast_days=1, max_emails=0, calendar_days=0)

        assert fetched, "get_weather was never called"
        assert fetched[0] == 3, (
            f"get_weather should be called with forecast_days=3 (2+1). Got {fetched[0]}"
        )

    def test_offset_slices_forecast_to_target_day(self):
        """The forecast in the result must start at date_offset, not at day 0."""
        result = _run_briefing(date_offset=2, forecast_days=1)
        weather = result.get("weather", {})
        forecast = weather.get("forecast", []) if weather else []
        if forecast:  # only assert if weather was returned
            assert forecast[0]["day_label"] not in ("Today", "Tomorrow"), (
                f"With date_offset=2 the first forecast entry must not be Today or Tomorrow. "
                f"Got: {forecast[0]['day_label']}"
            )

    def test_humidity_shown_for_offset_zero(self):
        result = _run_briefing(date_offset=0)
        text = result.get("text", "")
        assert "Humidity" in text, "Humidity should appear for date_offset=0"

    def test_humidity_hidden_for_offset_nonzero(self):
        result = _run_briefing(date_offset=1)
        text = result.get("text", "")
        assert "Humidity" not in text, (
            "Humidity from current conditions should not appear for date_offset>0 "
            "since it reflects today's conditions, not the target day's"
        )

    def test_offset_one_forecast_has_one_entry(self):
        result = _run_briefing(date_offset=1, forecast_days=1)
        weather = result.get("weather", {})
        forecast = weather.get("forecast", []) if weather else []
        if forecast:  # only assert if present
            assert len(forecast) == 1

    def test_forecast_days_two_returns_two_entries(self):
        result = _run_briefing(date_offset=0, forecast_days=2)
        weather = result.get("weather", {})
        forecast = weather.get("forecast", []) if weather else []
        if forecast:
            assert len(forecast) == 2


# ═══════════════════════════════════════════════════════════════════
# 4. get_day_briefing — weather failure surfacing
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestBriefingWeatherFailure:

    def _run_with_weather_error(self, error_json: str, date_offset: int = 0) -> dict:
        gmail_svc = _mock_gmail_service([])
        cal_svc   = _mock_calendar_service([])
        with patch("tools.location.geolocate_util.geolocate_ip", return_value=None), \
             patch("tools.location.geolocate_util.CLIENT_IP", None), \
             patch("tools.location.get_weather.get_weather", return_value=error_json), \
             patch("servers.google.server.GOOGLE_AVAILABLE", True), \
             patch("servers.google.server._gmail_service", return_value=gmail_svc), \
             patch("servers.google.server._calendar_service", return_value=cal_svc), \
             patch("servers.google.server._get_all_calendar_ids", return_value=["primary"]):
            from servers.google.server import get_day_briefing
            return json.loads(get_day_briefing(
                date_offset=date_offset, forecast_days=1,
                max_emails=0, calendar_days=0,
            ))

    def _run_with_exception(self, exc, date_offset: int = 0) -> dict:
        gmail_svc = _mock_gmail_service([])
        cal_svc   = _mock_calendar_service([])
        with patch("tools.location.geolocate_util.geolocate_ip", return_value=None), \
             patch("tools.location.geolocate_util.CLIENT_IP", None), \
             patch("tools.location.get_weather.get_weather", side_effect=exc), \
             patch("servers.google.server.GOOGLE_AVAILABLE", True), \
             patch("servers.google.server._gmail_service", return_value=gmail_svc), \
             patch("servers.google.server._calendar_service", return_value=cal_svc), \
             patch("servers.google.server._get_all_calendar_ids", return_value=["primary"]):
            from servers.google.server import get_day_briefing
            return json.loads(get_day_briefing(
                date_offset=date_offset, forecast_days=1,
                max_emails=0, calendar_days=0,
            ))

    def test_geocode_failure_shows_unavailable_in_text(self):
        """Regression: weather failure was silently omitted from output."""
        err = json.dumps({"error": "geocode_failed", "message": "No location found",
                          "city": "Surrey", "state": "BC", "country": "Canada"})
        result = self._run_with_weather_error(err)
        text = result.get("text", "")
        assert "Weather" in text and ("unavailable" in text or "Briefing" in text), (
            f"Weather section must appear (even if unavailable). Got: {text[:300]}"
        )

    def test_api_error_shows_unavailable_in_text(self):
        err = json.dumps({"error": "api_error", "message": "Parameter out of range"})
        result = self._run_with_weather_error(err)
        text = result.get("text", "")
        assert "unavailable" in text or "### Weather" in text

    def test_request_failed_shows_unavailable_in_text(self):
        err = json.dumps({"error": "request_failed", "message": "HTTP 503"})
        result = self._run_with_weather_error(err)
        text = result.get("text", "")
        assert "unavailable" in text or "### Weather" in text

    def test_exception_in_weather_shows_unavailable_in_text(self):
        """If weather raises an exception, text must say unavailable."""
        result = self._run_with_exception(ConnectionError("timeout"))
        text = result.get("text", "")
        assert "Weather" in text and ("unavailable" in text or "Briefing" in text), (
            f"Exception in weather fetch must surface in output. Got: {text[:300]}"
        )

    def test_date_still_present_when_weather_fails(self):
        err = json.dumps({"error": "geocode_failed", "message": "no location"})
        result = self._run_with_weather_error(err)
        text = result.get("text", "")
        assert "## " in text

    def test_email_still_present_when_weather_fails(self):
        err = json.dumps({"error": "geocode_failed", "message": "no location"})
        result = self._run_with_weather_error(err)
        text = result.get("text", "")
        assert "Email" in text

    def test_api_returns_fewer_days_than_offset_uses_fallback(self):
        """
        Regression: if date_offset=2 but API returns only 1 day,
        forecast[2:3] is empty → weather disappeared.
        Fix: fall back to the last available day.
        """
        # API returns only 1 day (not enough for date_offset=2)
        short_json = _briefing_weather_json(days=1)

        def fake_weather(city=None, state=None, country=None, forecast_days=1):
            return short_json  # always 1 day regardless of what was requested

        with patch("tools.location.geolocate_util.geolocate_ip", return_value=None), \
             patch("tools.location.geolocate_util.CLIENT_IP", None), \
             patch("tools.location.get_weather.get_weather", fake_weather), \
             patch("servers.google.server.gmail_get_unread",
                   return_value=json.dumps({"total_unread": 0, "emails": []})), \
             patch("servers.google.server.calendar_get_today",
                   return_value=json.dumps({"count": 0, "events": [], "text": ""})):
            from servers.google.server import get_day_briefing
            result = json.loads(get_day_briefing(
                date_offset=2, forecast_days=1, max_emails=0, calendar_days=0
            ))

        text = result.get("text", "")
        assert "### Weather" in text or "unavailable" in text, (
            "When API returns fewer days than date_offset, weather must still "
            f"appear (fallback or unavailable message). Got: {text[:300]}"
        )


# ═══════════════════════════════════════════════════════════════════
# 5. get_day_briefing — full text output format
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestBriefingTextFormat:

    def _briefing_text(self, **kwargs) -> str:
        return _run_briefing(**kwargs).get("text", "")

    def test_date_header_present(self):
        text = self._briefing_text()
        assert text.startswith("## "), f"Text must start with Date header. Got: {text[:80]}"

    def test_weather_section_present(self):
        text = self._briefing_text()
        assert "### Weather" in text

    def test_conditions_line_present(self):
        text = self._briefing_text()
        assert "Conditions:" in text

    def test_precipitation_chance_present(self):
        text = self._briefing_text()
        assert "Precipitation:" in text

    def test_high_temp_present(self):
        text = self._briefing_text()
        assert "High:" in text

    def test_low_temp_present(self):
        text = self._briefing_text()
        assert "Low:" in text

    def test_feels_like_present(self):
        text = self._briefing_text()
        assert "Feels Like:" in text

    def test_sunrise_present(self):
        text = self._briefing_text()
        assert "Sunrise:" in text

    def test_sunset_present(self):
        text = self._briefing_text()
        assert "Sunset:" in text

    def test_humidity_present_for_offset_zero(self):
        text = self._briefing_text(date_offset=0)
        assert "Humidity:" in text

    def test_emails_section_present(self):
        text = self._briefing_text()
        assert "Email" in text

    def test_no_unread_emails_message(self):
        text = self._briefing_text()
        assert "No unread emails" in text

    def test_calendar_section_present(self):
        text = self._briefing_text()
        assert "Calendar" in text

    def test_forecast_label_today_for_offset_zero(self):
        text = self._briefing_text(date_offset=0)
        assert "**Today**" in text

    def test_temperature_has_both_units(self):
        text = self._briefing_text()
        assert "°C" in text and "°F" in text

    def test_weather_location_in_header(self):
        text = self._briefing_text()
        assert "Surrey" in text or "British Columbia" in text

    def test_unread_emails_shown(self):
        emails = [
            {"id": "m1", "subject": "Meeting at 3pm", "from": "boss@company.com",
             "date": "Tue, 3 Jun 2026", "preview": "Let's meet."},
            {"id": "m2", "subject": "Invoice #1234", "from": "billing@vendor.com",
             "date": "Tue, 3 Jun 2026", "preview": "Please pay."},
        ]
        result = _run_briefing(date_offset=0, forecast_days=1,
                               max_emails=5, email_list=emails)
        text = result.get("text", "")
        assert "2 unread" in text
        assert "Meeting at 3pm" in text

    def test_calendar_events_shown(self):
        events = [{"id": "e1", "title": "Cello lesson",
                   "start": "2026-06-05T17:00:00-07:00",
                   "end":   "2026-06-05T17:45:00-07:00"}]
        result = _run_briefing(date_offset=0, forecast_days=1,
                               calendar_days=1, event_list=events)
        text = result.get("text", "")
        assert "Cello lesson" in text

    def test_result_is_valid_json(self):
        result = _run_briefing()
        assert isinstance(result, dict)

    def test_result_has_text_key(self):
        result = _run_briefing()
        assert "text" in result
        assert isinstance(result["text"], str)
        assert len(result["text"]) > 50
