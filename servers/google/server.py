"""
Google MCP Server
Gmail and Google Calendar tools via OAuth2
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env", override=True)

from servers.skills.skill_loader import SkillLoader

import base64
import inspect
import json
import logging
import os
from datetime import datetime, timezone, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional, List

from mcp.server.fastmcp import FastMCP
from tools.tool_control import check_tool_enabled
try:
    from client.tool_meta import tool_meta
except Exception:
    # Fallback stub — metadata is attached but not used in server subprocess
    def tool_meta(**kwargs):
        def decorator(fn): return fn
        return decorator

# ── Failure taxonomy ──────────────────────────────────────────────────────────
try:
    from metrics import FailureKind, MCPToolError, JsonFormatter
except ImportError:
    try:
        from client.metrics import FailureKind, MCPToolError, JsonFormatter
    except ImportError:
        from enum import Enum
        class FailureKind(Enum):
            RETRYABLE      = "retryable"
            USER_ERROR     = "user_error"
            UPSTREAM_ERROR = "upstream_error"
            INTERNAL_ERROR = "internal_error"
        class MCPToolError(Exception):
            def __init__(self, kind, message, detail=None):
                self.kind = kind; self.message = message; self.detail = detail or {}
                super().__init__(message)
        JsonFormatter = None

# Google API imports
try:
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError

    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    print("⚠️  Google API libraries not installed.")
    print("    Run: pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client")

# ── Logging ────────────────────────────────────────────────────────────────────

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.handlers.clear()

formatter = JsonFormatter() if JsonFormatter is not None else logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

file_handler = logging.FileHandler(LOG_DIR / "mcp-server.log", encoding="utf-8")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

root_logger.addHandler(file_handler)
root_logger.addHandler(console_handler)

logging.getLogger("mcp").setLevel(logging.DEBUG)
logging.getLogger("mcp_google_server").setLevel(logging.INFO)

logger = logging.getLogger("mcp_google_server")
logger.info("🚀 Google server logging initialized")

mcp = FastMCP("google-server")

# ── OAuth2 Configuration ───────────────────────────────────────────────────────

# Scopes required for Gmail (read + send) and Calendar (read + write)
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/calendar.events",
]

# Paths — set GOOGLE_CREDENTIALS_FILE and GOOGLE_TOKEN_FILE in .env,
# or drop credentials.json / token.json next to this server.py.
CREDENTIALS_FILE = os.getenv(
    "GOOGLE_CREDENTIALS_FILE",
    str(Path(__file__).parent / "credentials.json")
)
TOKEN_FILE = os.getenv(
    "GOOGLE_TOKEN_FILE",
    str(Path(__file__).parent / "token.json")
)


def _get_google_creds() -> Optional["Credentials"]:
    """
    Load or refresh OAuth2 credentials.
    On first run with no token.json, an interactive browser flow is triggered.
    """
    if not GOOGLE_AVAILABLE:
        return None

    creds = None

    if Path(TOKEN_FILE).exists():
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                logger.warning(f"Token refresh failed: {e} — re-authorizing")
                creds = None

        if not creds:
            if not Path(CREDENTIALS_FILE).exists():
                logger.error(f"credentials.json not found at {CREDENTIALS_FILE}")
                return None
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_console()

        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())

    return creds


def _gmail_service():
    creds = _get_google_creds()
    if not creds:
        return None
    return build("gmail", "v1", credentials=creds)


def _calendar_service():
    creds = _get_google_creds()
    if not creds:
        return None
    return build("calendar", "v3", credentials=creds)


def _not_available(tool_name: str) -> str:
    return json.dumps({
        "error": "Google API libraries not installed",
        "tool": tool_name,
        "install": "pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client"
    }, indent=2)


def _parse_message_headers(headers: list) -> dict:
    """Extract common headers from a Gmail message header list."""
    result = {}
    for h in headers:
        name = h.get("name", "").lower()
        if name in ("from", "to", "subject", "date", "cc"):
            result[name] = h.get("value", "")
    return result


def _extract_body(payload: dict) -> str:
    """Recursively extract plain-text body from a Gmail message payload."""
    mime_type = payload.get("mimeType", "")
    body = payload.get("body", {})

    if mime_type == "text/plain" and body.get("data"):
        return base64.urlsafe_b64decode(body["data"]).decode("utf-8", errors="replace")

    if mime_type == "text/html" and body.get("data"):
        # Fallback HTML — only used if no text/plain part exists
        raw = base64.urlsafe_b64decode(body["data"]).decode("utf-8", errors="replace")
        # Strip tags very lightly for readability
        import re
        return re.sub(r"<[^>]+>", "", raw)

    for part in payload.get("parts", []):
        result = _extract_body(part)
        if result:
            return result

    return ""


# ══════════════════════════════════════════════════════════════════════════════
# GMAIL TOOLS
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
@check_tool_enabled(category="google")
@tool_meta(tags=["read","search","email","external"],triggers=["unread emails","new emails","check email","do i have mail"],idempotent=False,example="use gmail_get_unread",intent_category="google",text_fields=["preview"])
def gmail_get_unread(max_results: int = 25) -> str:
    """
    Fetch all unread emails from Gmail inbox.

    Args:
        max_results (int, optional): Maximum number of unread emails to return. Default: 25.

    Returns:
        JSON with:
        - emails: List of unread messages (sender, subject, date, snippet, id)
        - total_unread: Count of unread messages returned

    Use cases:
        - "Show my unread emails"
        - "Do I have any new mail?"
        - "What emails haven't I read?"
    """
    logger.info(f"🛠  gmail_get_unread called (max={max_results})")

    if not GOOGLE_AVAILABLE:
        return _not_available("gmail_get_unread")

    try:
        service = _gmail_service()
        if not service:
            raise MCPToolError(FailureKind.USER_ERROR, "Could not authenticate with Google — check credentials.json and token.json",
                               {"tool": "gmail_get_unread"})

        result = service.users().messages().list(
            userId="me",
            labelIds=["INBOX", "UNREAD"],
            maxResults=max_results
        ).execute()

        messages = result.get("messages", [])
        emails = []

        for msg_ref in messages:
            msg = service.users().messages().get(
                userId="me",
                id=msg_ref["id"],
                format="metadata",
                metadataHeaders=["From", "To", "Subject", "Date"]
            ).execute()

            headers = _parse_message_headers(msg.get("payload", {}).get("headers", []))
            _msg_id = msg["id"]
            emails.append({
                "from":    headers.get("from", ""),
                "subject": headers.get("subject", "(no subject)"),
                "date":    headers.get("date", ""),
                "preview": msg.get("snippet", ""),
                "link":    f"https://mail.google.com/mail/u/0/#inbox/{_msg_id}",
                "id":      _msg_id,
            })

        logger.info(f"✅ Fetched {len(emails)} unread emails")
        return json.dumps({
            "total_unread": len(emails),
            "emails": emails
        }, indent=2)

    except MCPToolError:
        raise
    except HttpError as e:
        logger.error(f"❌ Gmail API error: {e}")
        raise MCPToolError(FailureKind.UPSTREAM_ERROR, f"Gmail API error: {e}",
                           {"tool": "gmail_get_unread", "status": getattr(e, 'status_code', None)})


@mcp.tool()
@check_tool_enabled(category="google")
@tool_meta(tags=["read","search","email","external"],triggers=["recent emails","my inbox","show emails","latest emails"],idempotent=False,example="use gmail_get_recent",intent_category="google",text_fields=["preview"])
def gmail_get_recent(max_results: int = 10) -> str:
    """
    Fetch the most recent emails from Gmail inbox (read and unread).

    Args:
        max_results (int, optional): Number of recent emails to return. Default: 10.

    Returns:
        JSON with:
        - emails: List of recent messages (sender, subject, date, snippet, read status, id)
        - count: Number of emails returned

    Use cases:
        - "Show my last 10 emails"
        - "What are my most recent emails?"
        - "Show me my inbox"
    """
    logger.info(f"🛠  gmail_get_recent called (max={max_results})")

    if not GOOGLE_AVAILABLE:
        return _not_available("gmail_get_recent")

    try:
        service = _gmail_service()
        if not service:
            raise MCPToolError(FailureKind.USER_ERROR, "Could not authenticate with Google — check credentials.json and token.json",
                               {"tool": "gmail_get_recent"})

        result = service.users().messages().list(
            userId="me",
            labelIds=["INBOX"],
            maxResults=max_results
        ).execute()

        messages = result.get("messages", [])
        emails = []

        for msg_ref in messages:
            msg = service.users().messages().get(
                userId="me",
                id=msg_ref["id"],
                format="metadata",
                metadataHeaders=["From", "To", "Subject", "Date"]
            ).execute()

            headers = _parse_message_headers(msg.get("payload", {}).get("headers", []))
            label_ids = msg.get("labelIds", [])
            is_unread = "UNREAD" in label_ids
            _msg_id = msg["id"]
            emails.append({
                "from":    headers.get("from", ""),
                "subject": headers.get("subject", "(no subject)"),
                "date":    headers.get("date", ""),
                "preview": msg.get("snippet", ""),
                "unread":  is_unread,
                "link":    f"https://mail.google.com/mail/u/0/#inbox/{_msg_id}",
                "id":      _msg_id,
            })

        logger.info(f"✅ Fetched {len(emails)} recent emails")
        return json.dumps({
            "count": len(emails),
            "emails": emails
        }, indent=2)

    except MCPToolError:
        raise
    except HttpError as e:
        logger.error(f"❌ Gmail API error: {e}")
        raise MCPToolError(FailureKind.UPSTREAM_ERROR, f"Gmail API error: {e}",
                           {"tool": "gmail_get_recent", "status": getattr(e, 'status_code', None)})


@mcp.tool()
@check_tool_enabled(category="google")
@tool_meta(tags=["read","email","external"],triggers=["read email","open email","show email"],idempotent=True,example='use gmail_get_email: message_id=""',intent_category="google",text_fields=["body"])
def gmail_get_email(message_id: str) -> str:
    """
    Read the full content of a specific Gmail message.

    Args:
        message_id (str, required): Gmail message ID (from gmail_get_unread or gmail_get_recent)

    Returns:
        JSON with:
        - id: Message ID
        - from / to / cc: Addresses
        - subject: Email subject
        - date: Sent date
        - body: Full plain-text body
        - snippet: Short preview

    Use cases:
        - "Read that email from Sarah"
        - "Open the email about the meeting"
        - "Show the full content of message <id>"
    """
    logger.info(f"🛠  gmail_get_email called: {message_id}")

    if not GOOGLE_AVAILABLE:
        return _not_available("gmail_get_email")

    if not message_id or not message_id.strip():
        raise MCPToolError(FailureKind.USER_ERROR, "message_id must not be empty",
                           {"tool": "gmail_get_email"})

    try:
        service = _gmail_service()
        if not service:
            raise MCPToolError(FailureKind.USER_ERROR, "Could not authenticate with Google — check credentials.json and token.json",
                               {"tool": "gmail_get_email"})

        msg = service.users().messages().get(
            userId="me",
            id=message_id,
            format="full"
        ).execute()

        payload = msg.get("payload", {})
        headers = _parse_message_headers(payload.get("headers", []))
        body = _extract_body(payload)

        return json.dumps({
            "from":    headers.get("from", ""),
            "to":      headers.get("to", ""),
            "cc":      headers.get("cc", "") or None,
            "subject": headers.get("subject", "(no subject)"),
            "date":    headers.get("date", ""),
            "body":    body,
            "link":    f"https://mail.google.com/mail/u/0/#inbox/{msg['id']}",
            "id":      msg["id"],
        }, indent=2)

    except MCPToolError:
        raise
    except HttpError as e:
        logger.error(f"❌ Gmail API error: {e}")
        status = getattr(e, 'status_code', None)
        kind = FailureKind.USER_ERROR if status == 404 else FailureKind.UPSTREAM_ERROR
        raise MCPToolError(kind, f"Gmail API error: {e}",
                           {"tool": "gmail_get_email", "message_id": message_id, "status": status})


@mcp.tool()
@check_tool_enabled(category="google")
@tool_meta(tags=["write","email","external"],triggers=["send email","compose email","email someone","write email"],idempotent=False,example='use gmail_send_email: to="" subject="" body="" [cc=""] [html=""]',intent_category="google")
def gmail_send_email(
        to: str,
        subject: str,
        body: str,
        cc: Optional[str] = None,
        html: bool = False
) -> str:
    """
    Compose and send an email via Gmail.

    Args:
        to (str, required): Recipient email address (or comma-separated list)
        subject (str, required): Email subject line
        body (str, required): Email body — plain text, or HTML if html=True
        cc (str, optional): CC addresses (comma-separated)
        html (bool, optional): Send as HTML instead of plain text. Default: False

    Returns:
        JSON with:
        - status: "sent"
        - message_id: ID of the sent message
        - thread_id: Thread ID
        - to / subject: Confirmed recipients and subject

    Use cases:
        - "Send an email to john@example.com saying the meeting is at 3pm"
        - "Email sarah@example.com about the project update"
        - "Compose an email to the team"
    """
    logger.info(f"🛠  gmail_send_email called: to={to}, subject={subject}")

    if not GOOGLE_AVAILABLE:
        return _not_available("gmail_send_email")

    if not to or not to.strip():
        raise MCPToolError(FailureKind.USER_ERROR, "to must not be empty",
                           {"tool": "gmail_send_email"})
    if not subject or not subject.strip():
        raise MCPToolError(FailureKind.USER_ERROR, "subject must not be empty",
                           {"tool": "gmail_send_email"})

    try:
        service = _gmail_service()
        if not service:
            raise MCPToolError(FailureKind.USER_ERROR, "Could not authenticate with Google — check credentials.json and token.json",
                               {"tool": "gmail_send_email"})

        if html:
            mime_msg = MIMEMultipart("alternative")
            mime_msg.attach(MIMEText(body, "html"))
        else:
            mime_msg = MIMEText(body, "plain")

        mime_msg["To"] = to
        mime_msg["Subject"] = subject
        if cc:
            mime_msg["Cc"] = cc

        raw = base64.urlsafe_b64encode(mime_msg.as_bytes()).decode()
        sent = service.users().messages().send(
            userId="me",
            body={"raw": raw}
        ).execute()

        logger.info(f"✅ Email sent: id={sent['id']}")
        return json.dumps({
            "status":  "sent",
            "to":      to,
            "cc":      cc or None,
            "subject": subject,
            "summary": f"Email sent to {to}: '{subject}'"
        }, indent=2)

    except MCPToolError:
        raise
    except HttpError as e:
        logger.error(f"❌ Gmail send error: {e}")
        raise MCPToolError(FailureKind.UPSTREAM_ERROR, f"Gmail send error: {e}",
                           {"tool": "gmail_send_email", "to": to, "subject": subject})


# ══════════════════════════════════════════════════════════════════════════════
# GOOGLE CALENDAR TOOLS
# ══════════════════════════════════════════════════════════════════════════════

def _format_event(event: dict, include_id: bool = False) -> dict:
    """Normalize a Calendar API event into a clean, readable dict."""
    import re as _re
    start = event.get("start", {})
    end   = event.get("end",   {})

    start_str = start.get("dateTime") or start.get("date", "")
    end_str   = end.get("dateTime")   or end.get("date", "")
    all_day   = "date" in start and "dateTime" not in start

    def _fmt_time(s: str) -> str:
        if not s:
            return ""
        try:
            from datetime import datetime as _dt
            dt = _dt.fromisoformat(s)
            if all_day:
                return dt.strftime("%a %b %-d, %Y")
            return dt.strftime("%a %b %-d, %-I:%M %p")
        except Exception:
            return s

    start_fmt = _fmt_time(start_str)
    end_fmt   = _fmt_time(end_str)

    # Build "when" string — omit end if same as start or same date for all-day
    if end_fmt and end_fmt != start_fmt:
        when = f"{start_fmt} – {end_fmt}"
    else:
        when = start_fmt

    # Strip HTML tags from description
    raw_desc = event.get("description", "") or ""
    clean_desc = _re.sub(r"<[^>]+>", "", raw_desc).strip()

    # Attendees: exclude organizer and self (anyone whose responseStatus is 'accepted'
    # as organizer, or whose email matches the organizer)
    organizer_email = event.get("organizer", {}).get("email", "").lower()
    raw_attendees = []
    for a in event.get("attendees", []):
        email = a.get("email", "").lower()
        if email == organizer_email:
            continue
        if a.get("self"):
            continue
        raw_attendees.append(a.get("displayName") or a.get("email", ""))
    attendees_str = ", ".join(raw_attendees) if raw_attendees else ""

    result = {
        "title":   event.get("summary", "(no title)"),
        "when":    when,
        "all_day": all_day,
    }
    if event.get("location"):
        result["location"] = event["location"]
    if clean_desc:
        result["notes"] = clean_desc[:200]
    if organizer_email:
        result["organizer"] = organizer_email
    if attendees_str:
        result["attendees"] = attendees_str
    if event.get("hangoutLink"):
        result["meet_link"] = event["hangoutLink"]
    if event.get("htmlLink"):
        result["calendar_link"] = event["htmlLink"]
    if include_id:
        result["id"] = event.get("id", "")

    return result


@mcp.tool()
@check_tool_enabled(category="google")
@tool_meta(tags=["read","calendar","external"],triggers=["calendar today","schedule today","meetings today","whats on today"],idempotent=False,example="use calendar_get_today",intent_category="google",text_fields=["notes"])
def calendar_get_today() -> str:
    """
    Get all calendar events for today.

    Returns:
        JSON with:
        - date: Today's date (YYYY-MM-DD)
        - events: List of today's events (title, start, end, location, attendees)
        - count: Number of events

    Use cases:
        - "What's on my calendar today?"
        - "Do I have any meetings today?"
        - "Show today's appointments"
    """
    logger.info("🛠  calendar_get_today called")

    if not GOOGLE_AVAILABLE:
        return _not_available("calendar_get_today")

    try:
        service = _calendar_service()
        if not service:
            raise MCPToolError(FailureKind.USER_ERROR, "Could not authenticate with Google — check credentials.json and token.json",
                               {"tool": "calendar_get_today"})

        now = datetime.now(timezone.utc)
        start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)

        result = service.events().list(
            calendarId="primary",
            timeMin=start_of_day.isoformat(),
            timeMax=end_of_day.isoformat(),
            singleEvents=True,
            orderBy="startTime"
        ).execute()

        events = [_format_event(e) for e in result.get("items", [])]

        logger.info(f"✅ Found {len(events)} events today")
        return json.dumps({
            "date": start_of_day.strftime("%Y-%m-%d"),
            "count": len(events),
            "events": events
        }, indent=2)

    except MCPToolError:
        raise
    except HttpError as e:
        logger.error(f"❌ Calendar API error: {e}")
        raise MCPToolError(FailureKind.UPSTREAM_ERROR, f"Calendar API error: {e}",
                           {"tool": "calendar_get_today"})


@mcp.tool()
@check_tool_enabled(category="google")
@tool_meta(tags=["read","calendar","external"],triggers=["this week calendar","weekly schedule","meetings this week","whats on this week"],idempotent=False,example="use calendar_get_this_week",intent_category="google",text_fields=["notes"])
def calendar_get_this_week() -> str:
    """
    Get all calendar events for the current week (Monday–Sunday).

    Returns:
        JSON with:
        - week_start / week_end: Date range (YYYY-MM-DD)
        - events: List of events grouped by day
        - count: Total number of events

    Use cases:
        - "What's happening this week?"
        - "Show my schedule for the week"
        - "Any meetings this week?"
    """
    logger.info("🛠  calendar_get_this_week called")

    if not GOOGLE_AVAILABLE:
        return _not_available("calendar_get_this_week")

    try:
        service = _calendar_service()
        if not service:
            raise MCPToolError(FailureKind.USER_ERROR, "Could not authenticate with Google — check credentials.json and token.json",
                               {"tool": "calendar_get_this_week"})

        now = datetime.now(timezone.utc)
        days_since_monday = now.weekday()
        monday = (now - timedelta(days=days_since_monday)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        sunday = monday + timedelta(days=7)

        result = service.events().list(
            calendarId="primary",
            timeMin=monday.isoformat(),
            timeMax=sunday.isoformat(),
            singleEvents=True,
            orderBy="startTime"
        ).execute()

        raw_events = result.get("items", [])

        # Format events, tracking current day to insert day-break headers in titles
        formatted = []
        current_day = None
        for e in raw_events:
            start = e.get("start", {})
            day = (start.get("dateTime") or start.get("date", ""))[:10]
            ev = _format_event(e)
            # Embed the day label into the title so the list builder shows it
            if day and day != current_day:
                current_day = day
                try:
                    from datetime import datetime as _dt
                    day_label = _dt.fromisoformat(day).strftime("%A %b %-d")
                except Exception:
                    day_label = day
                ev["_day_header"] = day_label
            formatted.append(ev)

        logger.info(f"✅ Found {len(formatted)} events this week")
        return json.dumps({
            "week_start": monday.strftime("%Y-%m-%d"),
            "week_end":   (sunday - timedelta(days=1)).strftime("%Y-%m-%d"),
            "count":      len(formatted),
            "events":     formatted
        }, indent=2)

    except MCPToolError:
        raise
    except HttpError as e:
        logger.error(f"❌ Calendar API error: {e}")
        raise MCPToolError(FailureKind.UPSTREAM_ERROR, f"Calendar API error: {e}",
                           {"tool": "calendar_get_this_week"})


@mcp.tool()
@check_tool_enabled(category="google")
@tool_meta(tags=["write","calendar","external"],triggers=["create event","schedule meeting","add to calendar","book appointment"],idempotent=False,example='use calendar_create_event: summary="" start="" end="" [description=""] [location=""] [attendees=""] [all_day=""]',intent_category="google")
def calendar_create_event(
        summary: str,
        start: str,
        end: str,
        description: Optional[str] = None,
        location: Optional[str] = None,
        attendees: Optional[List[str]] = None,
        all_day: bool = False
) -> str:
    """
    Create a new event on Google Calendar.

    Args:
        summary (str, required): Event title / name
        start (str, required): Start time — ISO 8601 datetime ("2025-06-15T14:00:00") or date ("2025-06-15") for all-day
        end (str, required): End time — ISO 8601 datetime or date for all-day
        description (str, optional): Event description / notes
        location (str, optional): Location string or address
        attendees (list, optional): List of attendee email addresses
        all_day (bool, optional): True for an all-day event (uses date format). Default: False

    Returns:
        JSON with:
        - status: "created"
        - event_id: New event ID
        - summary / start / end: Confirmed event details
        - html_link: Link to open event in Google Calendar

    Use cases:
        - "Create a meeting tomorrow at 2pm called Project Sync"
        - "Add dentist appointment on June 20th all day"
        - "Schedule a call with bob@example.com at 3pm Friday"
    """
    logger.info(f"🛠  calendar_create_event called: {summary} @ {start}")

    if not GOOGLE_AVAILABLE:
        return _not_available("calendar_create_event")

    if not summary or not summary.strip():
        raise MCPToolError(FailureKind.USER_ERROR, "summary must not be empty",
                           {"tool": "calendar_create_event"})
    if not start or not end:
        raise MCPToolError(FailureKind.USER_ERROR, "start and end are required",
                           {"tool": "calendar_create_event"})

    try:
        service = _calendar_service()
        if not service:
            raise MCPToolError(FailureKind.USER_ERROR, "Could not authenticate with Google — check credentials.json and token.json",
                               {"tool": "calendar_create_event"})

        if all_day:
            event_body = {
                "summary": summary,
                "start": {"date": start[:10]},
                "end": {"date": end[:10]}
            }
        else:
            start_dt = start if "Z" in start or "+" in start else start
            end_dt = end if "Z" in end or "+" in end else end
            event_body = {
                "summary": summary,
                "start": {"dateTime": start_dt, "timeZone": "America/Vancouver"},
                "end": {"dateTime": end_dt, "timeZone": "America/Vancouver"}
            }

        if description:
            event_body["description"] = description
        if location:
            event_body["location"] = location
        if attendees:
            event_body["attendees"] = [{"email": addr} for addr in attendees]

        created = service.events().insert(
            calendarId="primary",
            body=event_body,
            sendUpdates="all" if attendees else "none"
        ).execute()

        logger.info(f"✅ Event created: {created['id']}")
        _start_raw = created.get("start", {}).get("dateTime") or created.get("start", {}).get("date", "")
        _end_raw   = created.get("end",   {}).get("dateTime") or created.get("end",   {}).get("date", "")
        def _fmt(s):
            try:
                from datetime import datetime as _dt
                dt = _dt.fromisoformat(s)
                return dt.strftime("%a %b %-d, %-I:%M %p") if "T" in s else dt.strftime("%a %b %-d, %Y")
            except Exception:
                return s
        return json.dumps({
            "status":        "created",
            "title":         created.get("summary"),
            "when":          _fmt(_start_raw) + (" – " + _fmt(_end_raw) if _end_raw else ""),
            "calendar_link": created.get("htmlLink"),
            "meet_link":     created.get("hangoutLink") or None,
            "summary":       f"✅ '{created.get('summary')}' added to your calendar for {_fmt(_start_raw)}"
        }, indent=2)

    except MCPToolError:
        raise
    except HttpError as e:
        logger.error(f"❌ Calendar create error: {e}")
        raise MCPToolError(FailureKind.UPSTREAM_ERROR, f"Calendar API error: {e}",
                           {"tool": "calendar_create_event", "summary": summary})


# ══════════════════════════════════════════════════════════════════════════════
# SKILL MANAGEMENT
@mcp.tool()
@check_tool_enabled(category="google")
@tool_meta(tags=["write","email","external"],triggers=["reply to email","reply to this email","respond to email"],idempotent=False,example='use gmail_reply_tool: message_id="" body="" [cc=""]',intent_category="google")
def gmail_reply_tool(message_id: str, body: str, cc: Optional[str] = None) -> str:
    """
    Reply to an existing Gmail message, threading it correctly.

    Fetches the original message to get subject, recipients and thread ID,
    then sends a reply with proper In-Reply-To and References headers so
    Gmail threads it with the original conversation.

    Args:
        message_id (str, required): Gmail message ID shown as `id: ...` in email list results
        body (str, required): Reply body text
        cc (str, optional): Additional CC addresses (comma-separated)

    Returns:
        JSON with status, to, subject, and confirmation summary.

    Use cases:
        - "Reply to that email"
        - "use gmail_reply_tool: message_id=\"abc123\" body=\"Thanks, see you then!\""
    """
    logger.info(f"🛠  gmail_reply_tool called: message_id={message_id}")

    if not GOOGLE_AVAILABLE:
        return _not_available("gmail_reply_tool")

    if not message_id or not message_id.strip():
        raise MCPToolError(FailureKind.USER_ERROR, "message_id must not be empty",
                           {"tool": "gmail_reply_tool"})
    if not body or not body.strip():
        raise MCPToolError(FailureKind.USER_ERROR, "body must not be empty",
                           {"tool": "gmail_reply_tool"})

    try:
        service = _gmail_service()
        if not service:
            raise MCPToolError(FailureKind.USER_ERROR,
                               "Could not authenticate with Google — check credentials.json and token.json",
                               {"tool": "gmail_reply_tool"})

        # Fetch the original message to get threading headers
        original = service.users().messages().get(
            userId="me",
            id=message_id,
            format="full"
        ).execute()

        payload = original.get("payload", {})
        headers = {h["name"].lower(): h["value"]
                   for h in payload.get("headers", [])}

        subject = headers.get("subject", "")
        if not subject.lower().startswith("re:"):
            subject = f"Re: {subject}"

        # Reply goes to whoever sent the original
        reply_to = headers.get("reply-to") or headers.get("from", "")
        thread_id = original.get("threadId", "")
        message_id_header = headers.get("message-id", "")

        mime_msg = MIMEText(body, "plain")
        mime_msg["To"]         = reply_to
        mime_msg["Subject"]    = subject
        if cc:
            mime_msg["Cc"]     = cc
        if message_id_header:
            mime_msg["In-Reply-To"] = message_id_header
            mime_msg["References"]  = message_id_header

        raw = base64.urlsafe_b64encode(mime_msg.as_bytes()).decode()
        sent = service.users().messages().send(
            userId="me",
            body={"raw": raw, "threadId": thread_id}
        ).execute()

        logger.info(f"✅ Reply sent: id={sent['id']} thread={thread_id}")
        return json.dumps({
            "status":  "sent",
            "to":      reply_to,
            "subject": subject,
            "summary": f"Reply sent to {reply_to}: '{subject}'"
        }, indent=2)

    except MCPToolError:
        raise
    except HttpError as e:
        logger.error(f"❌ Gmail reply error: {e}")
        raise MCPToolError(FailureKind.UPSTREAM_ERROR, f"Gmail reply error: {e}",
                           {"tool": "gmail_reply_tool", "message_id": message_id})

@mcp.tool()
@check_tool_enabled(category="google")
@tool_meta(tags=["read","email","calendar","external"],triggers=["my day","day briefing","morning briefing","what's on today","today's summary","how's my day"],idempotent=False,example='use get_day_briefing [max_emails=""] [forecast_days=""]',intent_category="google")
def get_day_briefing(max_emails: Optional[int] = 10, forecast_days: Optional[int] = 1) -> str:
    """
    Get a combined briefing for today: weather, unread emails, and calendar events.

    Calls get_weather_tool, gmail_get_unread, and calendar_get_today internally
    and returns all three in a single structured response.

    Args:
        max_emails (int, optional):    Max unread emails to include (default: 10)
        forecast_days (int, optional): Days of weather forecast (default: 1 = today only)

    Returns:
        JSON with:
        - weather:   Current conditions and forecast (same format as get_weather_tool)
        - email:     Unread emails (same format as gmail_get_unread)
        - calendar:  Today's events (same format as calendar_get_today)
        - errors:    Any per-section errors that occurred
    """
    max_emails    = int(max_emails)    if max_emails    is not None else 10
    forecast_days = int(forecast_days) if forecast_days is not None else 1
    logger.info(f"🛠  get_day_briefing called (max_emails={max_emails}, forecast_days={forecast_days})")

    result = {"weather": None, "email": None, "calendar": None, "errors": {}}

    # ── Weather ───────────────────────────────────────────────────────────────
    try:
        from tools.location.geolocate_util import geolocate_ip, CLIENT_IP
        from tools.location.get_weather import get_weather as get_weather_fn

        city    = os.getenv("DEFAULT_CITY")
        state   = os.getenv("DEFAULT_STATE")
        country = os.getenv("DEFAULT_COUNTRY")
        if not city and CLIENT_IP:
            loc = geolocate_ip(CLIENT_IP)
            if loc:
                city    = loc.get("city")
                state   = loc.get("region")
                country = loc.get("country")

        weather_raw = get_weather_fn(city, state, country, forecast_days=forecast_days)
        result["weather"] = json.loads(weather_raw) if isinstance(weather_raw, str) else weather_raw
        logger.info("✅ Weather fetched")
    except Exception as e:
        logger.warning(f"⚠️  Weather failed: {e}")
        result["errors"]["weather"] = str(e)

    # ── Unread email ──────────────────────────────────────────────────────────
    if GOOGLE_AVAILABLE:
        try:
            service = _gmail_service()
            if service:
                res = service.users().messages().list(
                    userId="me", labelIds=["INBOX", "UNREAD"], maxResults=max_emails
                ).execute()
                messages = res.get("messages", [])
                emails = []
                for msg_ref in messages:
                    msg = service.users().messages().get(
                        userId="me", id=msg_ref["id"], format="metadata",
                        metadataHeaders=["From", "Subject", "Date"]
                    ).execute()
                    headers = _parse_message_headers(msg.get("payload", {}).get("headers", []))
                    _msg_id = msg["id"]
                    emails.append({
                        "from":    headers.get("from", ""),
                        "subject": headers.get("subject", "(no subject)"),
                        "date":    headers.get("date", ""),
                        "preview": msg.get("snippet", ""),
                        "link":    f"https://mail.google.com/mail/u/0/#inbox/{_msg_id}",
                        "id":      _msg_id,
                    })
                result["email"] = {"total_unread": len(emails), "emails": emails}
                logger.info(f"✅ {len(emails)} unread emails fetched")
        except Exception as e:
            logger.warning(f"⚠️  Gmail failed: {e}")
            result["errors"]["email"] = str(e)

    # ── Calendar ──────────────────────────────────────────────────────────────
    if GOOGLE_AVAILABLE:
        try:
            service = _calendar_service()
            if service:
                now = datetime.now(timezone.utc)
                start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
                end_of_day = start_of_day + timedelta(days=1)
                res = service.events().list(
                    calendarId="primary",
                    timeMin=start_of_day.isoformat(),
                    timeMax=end_of_day.isoformat(),
                    singleEvents=True,
                    orderBy="startTime"
                ).execute()
                events = [_format_event(e) for e in res.get("items", [])]
                result["calendar"] = {
                    "date":   start_of_day.strftime("%Y-%m-%d"),
                    "count":  len(events),
                    "events": events,
                }
                logger.info(f"✅ {len(events)} calendar events fetched")
        except Exception as e:
            logger.warning(f"⚠️  Calendar failed: {e}")
            result["errors"]["calendar"] = str(e)

    if not result["errors"]:
        del result["errors"]

    return json.dumps(result, indent=2)


skill_registry = None


@mcp.tool()
def list_capabilities(filter_tags: str | None = None) -> str:
    """
    Return the full capability schema for every tool on this server.

    Agents call this to discover what this server can do, what parameters
    each tool accepts, and what constraints apply.

    Args:
        filter_tags (str, optional): Comma-separated tags to filter by
                                     e.g. "read,search" or "write"

    Returns:
        JSON string with server name, tools array, and total count.
    """
    logger.info(f"🛠  list_capabilities called (filter_tags={filter_tags})")

    try:
        from client.capability_registry import (
            _TOOL_TAGS, _TOOL_RATE_LIMITS, _TOOL_IDEMPOTENT, _INTERNAL_TOOLS
        )
    except ImportError:
        return json.dumps({"error": "CapabilityRegistry not available"}, indent=2)

    import sys as _sys, inspect as _inspect
    _current = _sys.modules[__name__]

    wanted_tags = set(t.strip() for t in filter_tags.split(",") if t.strip()) if filter_tags else None

    tools_out = []
    seen = set()
    for _name, _obj in vars(_current).items():
        if not callable(_obj) or _name.startswith("_") or _name in _INTERNAL_TOOLS:
            continue
        if not hasattr(_obj, "__tool_meta__") and not hasattr(_obj, "_mcp_tool"):
            continue
        if _name in seen:
            continue
        seen.add(_name)

        tags = _TOOL_TAGS.get(_name, [])
        if wanted_tags and not (wanted_tags & set(tags)):
            continue

        sig = _inspect.signature(_obj)
        params = []
        for pname, param in sig.parameters.items():
            if pname in ("self",):
                continue
            has_default = param.default is not _inspect.Parameter.empty
            ann = param.annotation
            type_str = (
                ann.__name__ if hasattr(ann, "__name__")
                else str(ann).replace("typing.", "").replace("Optional[", "").rstrip("]")
                if ann is not _inspect.Parameter.empty else "string"
            )
            params.append({
                "name":     pname,
                "type":     type_str,
                "required": not has_default,
                "default":  None if not has_default else str(param.default),
            })

        tools_out.append({
            "name":         _name,
            "description":  (_obj.__doc__ or "").strip().split("\n")[0],
            "input_schema": params,
            "tags":         tags,
            "rate_limit":   _TOOL_RATE_LIMITS.get(_name),
            "idempotent":   _TOOL_IDEMPOTENT.get(_name, True),
        })

    return json.dumps({
        "server": mcp.name,
        "tools":  tools_out,
        "total":  len(tools_out),
    }, indent=2)


@mcp.tool()
def list_skills() -> str:
    """List all available skills for the Google server."""
    logger.info("📚 list_skills called")
    if skill_registry is None:
        return json.dumps({
            "server": "google-server",
            "skills": [],
            "message": "Skills not loaded"
        }, indent=2)
    return json.dumps({
        "server": "google-server",
        "skills": skill_registry.list()
    }, indent=2)


@mcp.tool()
def read_skill(skill_name: str) -> str:
    """Read the full content of a skill."""
    logger.info(f"📖 read_skill called: {skill_name}")
    if skill_registry is None:
        return json.dumps({"error": "Skills not loaded"}, indent=2)

    content = skill_registry.get_skill_content(skill_name)
    if content:
        return content

    available = [s.name for s in skill_registry.skills.values()]
    return json.dumps({
        "error": f"Skill '{skill_name}' not found",
        "available_skills": available
    }, indent=2)


def get_tool_names_from_module():
    """Auto-discover tools from this module."""
    current_module = sys.modules[__name__]
    tool_names = []
    for name, obj in inspect.getmembers(current_module):
        if inspect.isfunction(obj) and obj.__module__ == __name__:
            if not name.startswith("_") and name != "get_tool_names_from_module":
                tool_names.append(name)
    return tool_names


if __name__ == "__main__":
    server_tools = get_tool_names_from_module()

    skills_dir = Path(__file__).parent / "skills"
    loader = SkillLoader(server_tools, category="google_server")
    skill_registry = loader.load_all(skills_dir)

    logger.info(f"🛠️  {len(server_tools)} tools: {', '.join(server_tools)}")
    logger.info(f"📚 {len(skill_registry.skills)} skills loaded")

    if not GOOGLE_AVAILABLE:
        logger.warning("⚠️  Google API libraries not installed — run: pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client")

    creds_exists = Path(CREDENTIALS_FILE).exists()
    token_exists = Path(TOKEN_FILE).exists()
    logger.info(f"🔑 credentials.json: {'✅ found' if creds_exists else '❌ missing — download from Google Cloud Console'}")
    logger.info(f"🔑 token.json: {'✅ found' if token_exists else '⚠️  not yet — will be created on first auth'}")

    mcp.run(transport="stdio")