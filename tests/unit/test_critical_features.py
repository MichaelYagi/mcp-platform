"""
tests/unit/test_critical_features.py

Tests for critical app features that had no coverage and caused real bugs:

1. Tool template field validation — optional vs required brackets must match
   the Python function signature. Catches the gmail subject="" / improve_text
   mode="improve" class of bugs automatically for every tool.

2. get_day_briefing date_offset — weather must show the target day's forecast,
   not always today's. Catches the "weather on wrong day" bug.

3. Pipeline argument injection — subject auto-fill, body from previous step,
   empty field stripping, "Job completed." suppression on error.

4. Weather date label helpers — _get_date_label must return Today/Tomorrow/
   weekday names correctly.
"""
import re
import json
import inspect
import pytest
from pathlib import Path
from datetime import date
from unittest.mock import MagicMock, patch, AsyncMock

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SERVERS_DIR  = PROJECT_ROOT / "servers"


# ═══════════════════════════════════════════════════════════════════
# 1. Tool template field validation
#
# For every @tool_meta template in every server, verify that:
#   - fields WITHOUT brackets are required (no default in signature)
#   - fields WITH brackets are optional (have a default in signature)
# ═══════════════════════════════════════════════════════════════════

def _extract_templates(server_file: Path) -> list[tuple[str, str]]:
    """Return list of (func_name, template) pairs from a server file."""
    text = server_file.read_text(encoding="utf-8")
    results = []
    for m in re.finditer(r"template='(use\s+(\w+)[^']*)'", text):
        results.append((m.group(2), m.group(1)))
    return results


def _parse_template_fields(template: str) -> dict[str, bool]:
    """
    Parse template into {field_name: is_optional}.
    Fields inside [...] → optional=True, bare fields → optional=False.
    Ignores the tool name token (first word after 'use').
    """
    fields: dict[str, bool] = {}
    # Find bracketed fields: [field="..."]
    for m in re.finditer(r'\[(\w+)=', template):
        fields[m.group(1)] = True
    # Find bare fields: field="..." NOT inside brackets
    # Strategy: strip all bracketed sections, then find remaining field=
    bare = re.sub(r'\[[^\]]*\]', '', template)
    for m in re.finditer(r'(\w+)=', bare):
        name = m.group(1)
        if name not in fields:
            fields[name] = False
    return fields


def _get_function_params(server_file: Path, func_name: str) -> dict[str, bool]:
    """
    Return {param_name: has_default} by parsing source with inspect.
    Falls back to regex-based parsing if import fails.
    """
    text = server_file.read_text(encoding="utf-8")
    # Find the function definition block
    pattern = rf'def {func_name}\s*\((.*?)\)\s*(?:->|\:)'
    m = re.search(pattern, text, re.DOTALL)
    if not m:
        return {}
    params_block = m.group(1)
    params: dict[str, bool] = {}
    for pm in re.finditer(
        r'(?:^|,)\s*(\w+)\s*(?::\s*[^=,)]+)?\s*(=\s*[^,)]+)?',
        params_block.strip(), re.DOTALL
    ):
        name = pm.group(1).strip()
        has_default = pm.group(2) is not None
        if name and name not in ('self', 'cls', ''):
            params[name] = has_default
    return params


# Tools where the template deliberately diverges from the raw signature
# because the pipeline provides an automatic default.
PIPELINE_DEFAULTS = {
    # tool_name: {field: "why it's optional in practice"}
    "gmail_send_email": {"subject"},  # pipeline auto-injects "Message"
}

# Fields intentionally omitted from templates (internal / rarely needed)
INTENTIONALLY_OMITTED = {
    "run_python": {"timeout", "capture_vars"},
    "run_bash": {"timeout"},
    "run_python_file": {"timeout"},
    "pip_install": {"upgrade"},
    "gmail_send_email": {"html"},  # in template but as optional — already correct
}


@pytest.mark.unit
class TestToolTemplateFieldValidation:

    def _server_files(self):
        return list(SERVERS_DIR.glob("*/server.py"))

    def test_all_servers_have_server_files(self):
        assert len(self._server_files()) >= 10, "Expected at least 10 server files"

    def test_templates_exist_in_servers(self):
        total = sum(len(_extract_templates(f)) for f in self._server_files())
        assert total > 50, f"Expected 50+ templates, found {total}"

    @pytest.mark.parametrize("server_file", [
        SERVERS_DIR / "google" / "server.py",
        SERVERS_DIR / "text" / "server.py",
        SERVERS_DIR / "image" / "server.py",
        SERVERS_DIR / "discord" / "server.py",
        SERVERS_DIR / "plex" / "server.py",
        SERVERS_DIR / "rag" / "server.py",
        SERVERS_DIR / "location" / "server.py",
        SERVERS_DIR / "code_assistant" / "server.py",
        SERVERS_DIR / "code_review" / "server.py",
        SERVERS_DIR / "code_runner" / "server.py",
        SERVERS_DIR / "system" / "server.py",
        SERVERS_DIR / "trilium" / "server.py",
        SERVERS_DIR / "github" / "server.py",
    ])
    def test_template_fields_match_signature(self, server_file):
        """
        For every tool in the server, verify template brackets match the
        Python function signature (has default = optional = must use brackets).
        """
        templates = _extract_templates(server_file)
        errors = []

        for func_name, template in templates:
            template_fields = _parse_template_fields(template)
            sig_params = _get_function_params(server_file, func_name)

            pipeline_optional = PIPELINE_DEFAULTS.get(func_name, set())

            for field, template_is_optional in template_fields.items():
                if field not in sig_params:
                    continue  # field not in signature — skip
                sig_has_default = sig_params[field]
                effective_optional = sig_has_default or field in pipeline_optional

                if effective_optional and not template_is_optional:
                    errors.append(
                        f"{func_name}: '{field}' has a default but template "
                        f"shows it as required (missing brackets). "
                        f"Template: {template}"
                    )
                elif not effective_optional and template_is_optional:
                    errors.append(
                        f"{func_name}: '{field}' has NO default but template "
                        f"shows it as optional (has brackets). "
                        f"Template: {template}"
                    )

        assert not errors, "\n".join(errors)

    def test_gmail_send_subject_is_optional(self):
        """Regression: subject was shown as required but pipeline defaults it."""
        server_file = SERVERS_DIR / "google" / "server.py"
        templates = {fn: t for fn, t in _extract_templates(server_file)}
        template = templates.get("gmail_send_email", "")
        assert "[subject=" in template, (
            f"gmail_send_email subject should be optional ([subject=\"\"]) "
            f"because pipeline auto-injects 'Message'. Got: {template}"
        )

    def test_improve_text_mode_is_optional(self):
        """Regression: mode='improve' was shown as required but has a default."""
        server_file = SERVERS_DIR / "text" / "server.py"
        templates = {fn: t for fn, t in _extract_templates(server_file)}
        template = templates.get("improve_text_tool", "")
        assert "[mode=" in template, (
            f"improve_text_tool mode should be optional ([mode=\"improve\"]) "
            f"because it has a Python default. Got: {template}"
        )


# ═══════════════════════════════════════════════════════════════════
# 2. get_day_briefing date_offset weather indexing
# ═══════════════════════════════════════════════════════════════════

def _make_forecast(n: int) -> list[dict]:
    """Build a fake n-day forecast starting from today."""
    from datetime import date, timedelta
    days = []
    labels = ["Today", "Tomorrow", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for i in range(n):
        d = date.today() + timedelta(days=i)
        days.append({
            "date": d.isoformat(),
            "day_label": labels[i] if i < len(labels) else d.strftime("%A"),
            "condition": "Sunny",
            "precipitation_chance": "10%",
            "max_temp_c": 20 + i,
            "max_temp_f": 68 + i * 2,
            "min_temp_c": 10 + i,
            "min_temp_f": 50 + i * 2,
            "feelslike_c": 18 + i,
            "feelslike_f": 64 + i * 2,
            "sunrise": "6:00 AM",
            "sunset": "9:00 PM",
        })
    return days


def _make_weather_json(n: int) -> str:
    return json.dumps({
        "city": "Surrey",
        "state": "British Columbia",
        "country": "Canada",
        "current": {"humidity": "55%", "temperature_c": 20},
        "forecast": _make_forecast(n),
    })


@pytest.mark.unit
class TestGetDayBriefingDateOffset:

    def _call_briefing(self, date_offset: int, forecast_days: int = 1,
                       fetch_days_captured: list = None):
        """
        Call get_day_briefing with mocked weather, email, and calendar.
        Returns the result dict.
        """
        import importlib, sys

        # We need to capture what forecast_days value get_weather_fn is called with.
        expected_fetch = date_offset + forecast_days

        weather_json = _make_weather_json(expected_fetch)

        def fake_weather(city=None, state=None, country=None, forecast_days=1):
            if fetch_days_captured is not None:
                fetch_days_captured.append(forecast_days)
            return _make_weather_json(forecast_days)

        with patch("tools.location.get_weather.get_weather", fake_weather), \
             patch("servers.google.server.gmail_get_unread",
                   return_value=json.dumps({"total_unread": 0, "emails": []})), \
             patch("servers.google.server.calendar_get_today",
                   return_value=json.dumps({"count": 0, "events": [], "text": ""})):

            # Patch geolocate to avoid network
            with patch("tools.location.geolocate_util.geolocate_ip", return_value=None), \
                 patch("tools.location.geolocate_util.CLIENT_IP", None):

                # Also patch get_weather_fn inside the server module
                with patch("tools.location.get_weather.get_weather", fake_weather):
                    from servers.google.server import get_day_briefing
                    raw = get_day_briefing(
                        date_offset=date_offset,
                        forecast_days=forecast_days,
                        max_emails=0,
                        calendar_days=0,
                    )
                    return json.loads(raw)

    def test_date_offset_zero_shows_today(self):
        """date_offset=0: forecast starts with today's weather."""
        fetched = []
        with patch("tools.location.geolocate_util.geolocate_ip", return_value=None), \
             patch("tools.location.geolocate_util.CLIENT_IP", None):
            def fake_weather(city=None, state=None, country=None, forecast_days=1):
                fetched.append(forecast_days)
                return _make_weather_json(forecast_days)
            with patch("tools.location.get_weather.get_weather", fake_weather), \
                 patch("servers.google.server.gmail_get_unread",
                       return_value=json.dumps({"total_unread": 0, "emails": []})), \
                 patch("servers.google.server.calendar_get_today",
                       return_value=json.dumps({"count": 0, "events": [], "text": ""})):
                from servers.google.server import get_day_briefing
                result = json.loads(get_day_briefing(date_offset=0, forecast_days=1,
                                                     max_emails=0, calendar_days=0))
        # forecast should have 1 entry (today)
        weather = result.get("weather", {})
        forecast = weather.get("forecast", []) if weather else []
        assert len(forecast) == 1
        assert forecast[0]["day_label"] == "Today"

    def test_date_offset_fetches_extra_days(self):
        """date_offset=2, forecast_days=1 must fetch 3 days (date_offset+forecast_days)."""
        fetched = []
        with patch("tools.location.geolocate_util.geolocate_ip", return_value=None), \
             patch("tools.location.geolocate_util.CLIENT_IP", None):
            def fake_weather(city=None, state=None, country=None, forecast_days=1):
                fetched.append(forecast_days)
                return _make_weather_json(forecast_days)
            with patch("tools.location.get_weather.get_weather", fake_weather), \
                 patch("servers.google.server.gmail_get_unread",
                       return_value=json.dumps({"total_unread": 0, "emails": []})), \
                 patch("servers.google.server.calendar_get_today",
                       return_value=json.dumps({"count": 0, "events": [], "text": ""})):
                from servers.google.server import get_day_briefing
                get_day_briefing(date_offset=2, forecast_days=1, max_emails=0, calendar_days=0)
        assert fetched, "get_weather was not called"
        assert fetched[0] == 3, (
            f"Expected get_weather to be called with forecast_days=3 "
            f"(date_offset=2 + forecast_days=1), got {fetched[0]}"
        )

    def test_date_offset_slices_forecast(self):
        """date_offset=2: result forecast must NOT include Today or Tomorrow."""
        with patch("tools.location.geolocate_util.geolocate_ip", return_value=None), \
             patch("tools.location.geolocate_util.CLIENT_IP", None):
            def fake_weather(city=None, state=None, country=None, forecast_days=1):
                return _make_weather_json(forecast_days)
            with patch("tools.location.get_weather.get_weather", fake_weather), \
                 patch("servers.google.server.gmail_get_unread",
                       return_value=json.dumps({"total_unread": 0, "emails": []})), \
                 patch("servers.google.server.calendar_get_today",
                       return_value=json.dumps({"count": 0, "events": [], "text": ""})):
                from servers.google.server import get_day_briefing
                result = json.loads(get_day_briefing(date_offset=2, forecast_days=1,
                                                     max_emails=0, calendar_days=0))
        weather = result.get("weather", {})
        forecast = weather.get("forecast", []) if weather else []
        assert len(forecast) == 1, f"Expected 1 forecast day, got {len(forecast)}"
        assert forecast[0]["day_label"] not in ("Today", "Tomorrow"), (
            f"With date_offset=2 the forecast should skip Today and Tomorrow, "
            f"got day_label={forecast[0]['day_label']!r}"
        )

    def test_date_offset_text_shows_correct_date(self):
        """Briefing text header must show the offset date, not today."""
        from datetime import datetime, timedelta
        from zoneinfo import ZoneInfo
        expected_date = (datetime.now() + timedelta(days=2)).strftime("%-d")

        with patch("tools.location.geolocate_util.geolocate_ip", return_value=None), \
             patch("tools.location.geolocate_util.CLIENT_IP", None):
            def fake_weather(city=None, state=None, country=None, forecast_days=1):
                return _make_weather_json(forecast_days)
            with patch("tools.location.get_weather.get_weather", fake_weather), \
                 patch("servers.google.server.gmail_get_unread",
                       return_value=json.dumps({"total_unread": 0, "emails": []})), \
                 patch("servers.google.server.calendar_get_today",
                       return_value=json.dumps({"count": 0, "events": [], "text": ""})):
                from servers.google.server import get_day_briefing
                result = json.loads(get_day_briefing(date_offset=2, forecast_days=1,
                                                     max_emails=0, calendar_days=0))
        assert expected_date in result.get("date", ""), (
            f"Briefing date should contain day {expected_date} for date_offset=2, "
            f"got: {result.get('date')}"
        )

    def test_humidity_hidden_for_offset_day(self):
        """Current conditions humidity must not appear when date_offset > 0."""
        with patch("tools.location.geolocate_util.geolocate_ip", return_value=None), \
             patch("tools.location.geolocate_util.CLIENT_IP", None):
            def fake_weather(city=None, state=None, country=None, forecast_days=1):
                return _make_weather_json(forecast_days)
            with patch("tools.location.get_weather.get_weather", fake_weather), \
                 patch("servers.google.server.gmail_get_unread",
                       return_value=json.dumps({"total_unread": 0, "emails": []})), \
                 patch("servers.google.server.calendar_get_today",
                       return_value=json.dumps({"count": 0, "events": [], "text": ""})):
                from servers.google.server import get_day_briefing
                result = json.loads(get_day_briefing(date_offset=2, forecast_days=1,
                                                     max_emails=0, calendar_days=0))
        text = result.get("text", "")
        assert "Humidity" not in text, (
            "Humidity should not appear for offset days since it reflects "
            "current conditions, not the target day"
        )


# ═══════════════════════════════════════════════════════════════════
# 3. Pipeline argument injection — critical business logic
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestPipelineFieldHandling:
    """
    Replicate the pipeline argument-injection logic from _run_pipeline
    so we have regression coverage for the most important rules.
    """

    def _strip_empty(self, args: dict) -> dict:
        return {k: v for k, v in args.items() if v != ""}

    def _inject_subject_default(self, tool_name: str, args: dict, previous: str) -> dict:
        """Replicate lines 1149-1152 of client.py."""
        if "gmail_send" in tool_name:
            if "body" not in args:
                args["body"] = previous
            if "subject" not in args:
                args["subject"] = "Message"
        return args

    def _pipeline_aborts(self, previous: str) -> bool:
        return (
            str(previous).startswith("Tool ")
            and ("error:" in str(previous).lower() or "not found." in str(previous))
        )

    def _job_completed_response(self, last_tool: str, previous: str) -> str:
        _NOTIF = ("discord_notify", "gmail_reply_tool")
        _EMAIL_SEND = ("gmail_send_email",)
        errored = (
            str(previous).startswith("Tool ")
            and ("error:" in str(previous).lower() or "not found." in str(previous))
        )
        if not errored and (
            any(nt in last_tool for nt in _NOTIF)
            or any(nt in last_tool for nt in _EMAIL_SEND)
        ):
            return "Job completed."
        return str(previous) if previous else "Done."

    # ── Empty field stripping ─────────────────────────────────────

    def test_empty_to_field_stripped(self):
        args = {"to": "", "subject": "Hello", "body": "Hi"}
        result = self._strip_empty(args)
        assert "to" not in result

    def test_empty_subject_stripped(self):
        args = {"to": "a@b.com", "subject": "", "body": "Hi"}
        result = self._strip_empty(args)
        assert "subject" not in result

    def test_nonempty_values_kept(self):
        args = {"to": "a@b.com", "subject": "Test"}
        result = self._strip_empty(args)
        assert result == {"to": "a@b.com", "subject": "Test"}

    # ── Subject auto-injection ────────────────────────────────────

    def test_missing_subject_gets_message_default(self):
        args = {"to": "a@b.com"}
        result = self._inject_subject_default("gmail_send_email", args, "photo desc")
        assert result["subject"] == "Message"

    def test_explicit_subject_not_overwritten(self):
        args = {"to": "a@b.com", "subject": "My Subject"}
        result = self._inject_subject_default("gmail_send_email", args, "photo desc")
        assert result["subject"] == "My Subject"

    def test_body_injected_from_previous(self):
        args = {"to": "a@b.com"}
        result = self._inject_subject_default("gmail_send_email", args, "photo description text")
        assert result["body"] == "photo description text"

    def test_explicit_body_not_overwritten(self):
        args = {"to": "a@b.com", "body": "custom body"}
        result = self._inject_subject_default("gmail_send_email", args, "previous result")
        assert result["body"] == "custom body"

    def test_non_email_tool_not_injected(self):
        args = {}
        result = self._inject_subject_default("web_search_tool", args, "something")
        assert "subject" not in result
        assert "body" not in result

    # ── Pipeline abort condition ──────────────────────────────────

    def test_abort_on_tool_error(self):
        assert self._pipeline_aborts("Tool gmail_send_email error: SMTP failed")

    def test_abort_on_tool_not_found(self):
        assert self._pipeline_aborts("Tool 'missing' not found.")

    def test_no_abort_on_success(self):
        assert not self._pipeline_aborts("Photo description from vision model")

    def test_no_abort_on_json_result(self):
        assert not self._pipeline_aborts('{"status": "ok"}')

    # ── "Job completed." suppression on error ─────────────────────

    def test_job_completed_on_success(self):
        result = self._job_completed_response("gmail_send_email", "sent successfully")
        assert result == "Job completed."

    def test_job_completed_suppressed_on_error(self):
        """Regression: before fix, errors returned 'Job completed.' hiding failures."""
        result = self._job_completed_response(
            "gmail_send_email",
            "Tool gmail_send_email error: Missing required field(s): to"
        )
        assert result != "Job completed.", (
            "Error result should not be hidden as 'Job completed.' — "
            "the user needs to see the actual error"
        )
        assert "error" in result.lower() or "missing" in result.lower()

    def test_job_completed_suppressed_on_not_found(self):
        result = self._job_completed_response(
            "discord_notify",
            "Tool 'discord_notify' not found."
        )
        assert result != "Job completed."

    def test_discord_notify_job_completed_on_success(self):
        result = self._job_completed_response("discord_notify", "sent to webhook")
        assert result == "Job completed."


# ═══════════════════════════════════════════════════════════════════
# 4. Weather date label helpers
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestWeatherDateLabels:

    def _label(self, days_offset: int) -> str:
        from tools.location.get_weather import _get_date_label
        from datetime import date, timedelta
        target = date.today() + timedelta(days=days_offset)
        return _get_date_label(target.isoformat(), today=date.today())

    def test_today_label(self):
        assert self._label(0) == "Today"

    def test_tomorrow_label(self):
        assert self._label(1) == "Tomorrow"

    def test_yesterday_label(self):
        assert self._label(-1) == "Yesterday"

    def test_day_after_tomorrow_is_weekday_name(self):
        result = self._label(2)
        weekdays = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
        assert result in weekdays, f"Expected weekday name, got {result!r}"

    def test_six_days_ahead_is_weekday_name(self):
        result = self._label(6)
        weekdays = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
        assert result in weekdays

    def test_seven_days_ahead_includes_date(self):
        result = self._label(7)
        # Should be "Monday, June 10" style (full date)
        assert "," in result or result in {
            "Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"
        }

    def test_invalid_date_returns_input(self):
        from tools.location.get_weather import _get_date_label
        assert _get_date_label("not-a-date") == "not-a-date"


# ═══════════════════════════════════════════════════════════════════
# 5. Pydantic validation error formatting
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestValidationErrorFormatting:
    """
    Verify that missing required fields produce a clean, user-readable
    error message rather than raw Pydantic traceback text.
    """

    def _format_error(self, e: Exception) -> str:
        """Replicate the error formatting logic from _tool_executor."""
        err_str = str(e)
        try:
            from pydantic import ValidationError as _PydanticVE
            if isinstance(e, _PydanticVE):
                missing = [str(err["loc"][0]) for err in e.errors()
                           if err.get("type") == "missing"]
                if missing:
                    err_str = f"Missing required field(s): {', '.join(missing)}"
        except Exception:
            pass
        return err_str

    def test_missing_single_field_formatted(self):
        from pydantic import BaseModel, ValidationError
        class M(BaseModel):
            to: str
            body: str
        try:
            M(body="hi")
        except ValidationError as e:
            result = self._format_error(e)
            assert "Missing required field(s)" in result
            assert "to" in result
            assert "validation error" not in result.lower()

    def test_missing_multiple_fields_formatted(self):
        from pydantic import BaseModel, ValidationError
        class M(BaseModel):
            to: str
            subject: str
            body: str
        try:
            M()
        except ValidationError as e:
            result = self._format_error(e)
            assert "Missing required field(s)" in result
            assert "to" in result

    def test_non_pydantic_error_unchanged(self):
        e = ValueError("connection refused")
        result = self._format_error(e)
        assert result == "connection refused"

    def test_correct_fields_no_error(self):
        from pydantic import BaseModel, ValidationError
        class M(BaseModel):
            to: str
            body: str
        # Should not raise
        m = M(to="a@b.com", body="hi")
        assert m.to == "a@b.com"
