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
import requests as _requests
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
        from servers.error_fallback import FailureKind, MCPToolError, JsonFormatter

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

from servers.logging_setup import setup_server_logging
logger = setup_server_logging("mcp_google_server", PROJECT_ROOT, JsonFormatter)
logger.info("🚀 Google server logging initialized")

mcp = FastMCP("google-server")

# Cached credentials — avoids hitting the network on every tool call.
# Invalidated when expired; google_reauth_complete also clears it.
_cached_creds = None

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
    Load or refresh OAuth2 credentials, with in-memory caching.
    Returns cached creds immediately if still valid; only hits the network
    when the token is expired or missing.
    """
    global _cached_creds

    if not GOOGLE_AVAILABLE:
        return None

    # Return cached creds if still valid
    if _cached_creds and _cached_creds.valid:
        return _cached_creds

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
            # No interactive flow here — use google_reauth_start / google_reauth_complete
            logger.error("No valid token — call google_reauth_start to re-authorise")
            return None

        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())

    _cached_creds = creds
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


def _get_all_calendar_ids(service) -> list[str]:
    """Return all calendar IDs the user has access to, including shared calendars."""
    try:
        result = service.calendarList().list().execute()
        return [cal["id"] for cal in result.get("items", [])]
    except Exception as e:
        logger.warning(f"⚠️  Could not fetch calendar list: {e} — falling back to primary")
        return ["primary"]


def _not_available(tool_name: str) -> str:
    return json.dumps({
        "error": "Google API libraries not installed",
        "tool": tool_name,
        "install": "pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client"
    }, indent=2)


def _script_post(payload: dict) -> dict:
    """POST to the configured Apps Script endpoint. Extracts the key from the URL and includes it in the body."""
    from urllib.parse import urlparse, parse_qs
    url = os.getenv("GOOGLE_APPS_SCRIPT_URL", "").rstrip("/")
    if not url:
        raise RuntimeError("GOOGLE_APPS_SCRIPT_URL not set")
    key = parse_qs(urlparse(url).query).get("key", [""])[0]
    resp = _requests.post(url, json={**payload, "key": key}, timeout=15)
    resp.raise_for_status()
    try:
        data = resp.json()
    except ValueError:
        snippet = resp.text.strip()[:200]
        raise RuntimeError(f"Apps Script returned a non-JSON response: {snippet or '(empty body)'}")
    if "error" in data:
        raise RuntimeError(data["error"])
    return data


def _summarise_emails(emails: list) -> dict:
    """Batch-summarise email previews via Ollama. Returns {1-based index: summary string}."""
    summaries = {}
    if not emails:
        return summaries
    try:
        import time as _time_mod
        import urllib.request as _urllib_req
        _time_mod.sleep(1)
        _ollama_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        _model      = os.getenv("OLLAMA_MODEL", "qwen2.5:14b-instruct-q4_K_M")
        _batch_lines = [
            f"{i}. From: {em['from']} | Subject: {em['subject']} | Snippet: {em['preview']}"
            for i, em in enumerate(emails, 1)
        ]
        _prompt = (
            "Summarise each email below in one short sentence (max 15 words). "
            "Reply ONLY with numbered lines matching the input numbers, nothing else.\n\n"
            + "\n".join(_batch_lines)
        )
        _payload = json.dumps({"model": _model, "prompt": _prompt, "stream": False}).encode()
        _req = _urllib_req.Request(
            f"{_ollama_url}/api/generate", data=_payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with _urllib_req.urlopen(_req, timeout=120) as _resp:
            _resp_text = json.loads(_resp.read().decode()).get("response", "")
        for _line in _resp_text.strip().splitlines():
            _line = _line.strip()
            if _line and _line[0].isdigit():
                _dot = _line.find(".")
                if _dot != -1:
                    summaries[int(_line[:_dot].strip())] = _line[_dot + 1:].strip()
        logger.info(f"✅ Summarised {len(summaries)} emails in one LLM call")
    except Exception as e:
        logger.warning(f"⚠️  Email summarisation failed, using snippets: {e}")
    return summaries


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
@tool_meta(tags=["read","search","email","external"],triggers=["unread emails","new emails","check email","do i have mail"],idempotent=False,template="use gmail_get_unread",intent_category="google",text_fields=["text"])
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

    # ── Data collection ───────────────────────────────────────────────────────
    emails = []
    total_unread = 0

    if os.getenv("GOOGLE_APPS_SCRIPT_URL"):
        try:
            _surl = os.getenv("GOOGLE_APPS_SCRIPT_URL", "").rstrip("/")
            _sep = "&" if "?" in _surl else "?"
            _resp = _requests.get(f"{_surl}{_sep}type=gmail&max_emails={max_results}", timeout=10)
            _resp.raise_for_status()
            _data = _resp.json()
            if "error" in _data:
                raise RuntimeError(_data["error"])
            _msgs = _data.get("messages", [])
            emails = [
                {
                    "from":    m.get("from", ""),
                    "subject": m.get("subject", "(no subject)"),
                    "date":    m.get("date", ""),
                    "preview": m.get("preview", ""),
                    "link":    m.get("link", ""),
                    "id":      m.get("id", ""),
                }
                for m in _msgs
            ]
            total_unread = _data.get("unreadCount", len(emails))
            logger.info(f"✅ Fetched {len(emails)} unread emails (Apps Script)")
        except Exception as e:
            logger.error(f"❌ gmail_get_unread (Apps Script) failed: {e}")
            raise MCPToolError(FailureKind.UPSTREAM_ERROR, f"Gmail fetch error: {e}",
                               {"tool": "gmail_get_unread"})
    else:
        if not GOOGLE_AVAILABLE:
            return _not_available("gmail_get_unread")
        try:
            service = _gmail_service()
            if not service:
                raise MCPToolError(FailureKind.USER_ERROR, "Could not authenticate with Google — check credentials.json and token.json",
                                   {"tool": "gmail_get_unread"})
            result = service.users().messages().list(
                userId="me", labelIds=["INBOX", "UNREAD"], maxResults=max_results
            ).execute()
            for msg_ref in result.get("messages", []):
                msg = service.users().messages().get(
                    userId="me", id=msg_ref["id"], format="metadata",
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
            total_unread = len(emails)
            logger.info(f"✅ Fetched {len(emails)} unread emails")
        except MCPToolError:
            raise
        except HttpError as e:
            logger.error(f"❌ Gmail API error: {e}")
            raise MCPToolError(FailureKind.UPSTREAM_ERROR, f"Gmail API error: {e}",
                               {"tool": "gmail_get_unread", "status": getattr(e, 'status_code', None)})

    # ── Shared output building ────────────────────────────────────────────────
    import html as _html
    for em in emails:
        em["from"]    = _html.unescape(em["from"])
        em["subject"] = _html.unescape(em["subject"])
        em["preview"] = _html.unescape(em["preview"])

    _summaries = _summarise_emails(emails)

    lines = []
    for _i, em in enumerate(emails, 1):
        _summary = _summaries.get(_i) or (em["preview"][:120] + "…" if len(em["preview"]) > 120 else em["preview"])
        lines.append(f"{_i}. **{em['subject']}**")
        lines.append(f"   **From:** {em['from']}")
        lines.append(f"   **Date:** {em['date']}")
        lines.append(f"   **Summary:** {_summary}")
        if em.get("id"):
            lines.append(f"   **ID:** {em['id']}")
        if em.get("link"):
            lines.append(f"   **Link:** {em['link']}")
        lines.append("")

    return json.dumps({
        "total_unread": total_unread,
        "text":         "\n".join(lines),
        "emails":       emails,
    }, indent=2)


@mcp.tool()
@check_tool_enabled(category="google")
@tool_meta(tags=["read","search","email","external"],triggers=["recent emails","my inbox","show emails","latest emails","show my inbox","last emails","recent messages","inbox"],idempotent=False,template="use gmail_get_recent",intent_category="google",text_fields=["text"])
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

    if not GOOGLE_AVAILABLE and not os.getenv("GOOGLE_APPS_SCRIPT_URL"):
        return _not_available("gmail_get_recent")

    try:
        # ── Data collection ──────────────────────────────────────────────────
        emails = []
        if os.getenv("GOOGLE_APPS_SCRIPT_URL"):
            _surl = os.getenv("GOOGLE_APPS_SCRIPT_URL", "").rstrip("/")
            _sep = "&" if "?" in _surl else "?"
            _resp = _requests.get(
                f"{_surl}{_sep}type=gmail_recent&max_emails={max_results}", timeout=15)
            _resp.raise_for_status()
            _data = _resp.json()
            if "error" in _data:
                raise RuntimeError(_data["error"])
            for _m in _data.get("messages", []):
                emails.append({
                    "from":    _m.get("from", ""),
                    "subject": _m.get("subject", "(no subject)"),
                    "date":    _m.get("date", ""),
                    "preview": _m.get("preview", ""),
                    "unread":  _m.get("unread", False),
                    "link":    _m.get("link", ""),
                    "id":      _m.get("id", ""),
                })
        else:
            service = _gmail_service()
            if not service:
                raise MCPToolError(FailureKind.USER_ERROR,
                                   "Could not authenticate with Google — check credentials.json and token.json",
                                   {"tool": "gmail_get_recent"})
            result = service.users().messages().list(
                userId="me", labelIds=["INBOX"], maxResults=max_results).execute()
            for msg_ref in result.get("messages", []):
                msg = service.users().messages().get(
                    userId="me", id=msg_ref["id"], format="metadata",
                    metadataHeaders=["From", "To", "Subject", "Date"]).execute()
                headers = _parse_message_headers(msg.get("payload", {}).get("headers", []))
                _msg_id = msg["id"]
                emails.append({
                    "from":    headers.get("from", ""),
                    "subject": headers.get("subject", "(no subject)"),
                    "date":    headers.get("date", ""),
                    "preview": msg.get("snippet", ""),
                    "unread":  "UNREAD" in msg.get("labelIds", []),
                    "link":    f"https://mail.google.com/mail/u/0/#inbox/{_msg_id}",
                    "id":      _msg_id,
                })

        # ── Shared output building ───────────────────────────────────────────
        import html as _html
        for em in emails:
            em["from"]    = _html.unescape(em.get("from", ""))
            em["subject"] = _html.unescape(em.get("subject", ""))
            em["preview"] = _html.unescape(em.get("preview", ""))

        _summaries = _summarise_emails(emails)
        lines = []
        for _i, em in enumerate(emails, 1):
            _summary = _summaries.get(_i) or (em["preview"][:120] + "…" if len(em["preview"]) > 120 else em["preview"])
            lines.append(f"{_i}. **{em['subject']}**")
            lines.append(f"   **From:** {em['from']}")
            lines.append(f"   **Date:** {em['date']}")
            lines.append(f"   **Summary:** {_summary}")
            if em.get("id"):
                lines.append(f"   **ID:** {em['id']}")
            if em.get("link"):
                lines.append(f"   **Link:** {em['link']}")
            lines.append("")

        logger.info(f"✅ Fetched {len(emails)} recent emails")
        return json.dumps({"count": len(emails), "text": "\n".join(lines), "emails": emails}, indent=2)

    except MCPToolError:
        raise
    except Exception as e:
        logger.error(f"❌ Gmail recent error: {e}")
        raise MCPToolError(FailureKind.UPSTREAM_ERROR, f"Gmail error: {e}",
                           {"tool": "gmail_get_recent"})


@mcp.tool()
@check_tool_enabled(category="google")
@tool_meta(tags=["read","email","external"],triggers=["read email","open email","show email","get email","view email","email details","email content"],idempotent=True,template='use gmail_get_email: message_id=""',intent_category="google",text_fields=["body"])
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

    if not GOOGLE_AVAILABLE and not os.getenv("GOOGLE_APPS_SCRIPT_URL"):
        return _not_available("gmail_get_email")

    if not message_id or not message_id.strip():
        raise MCPToolError(FailureKind.USER_ERROR, "message_id must not be empty",
                           {"tool": "gmail_get_email"})

    try:
        if os.getenv("GOOGLE_APPS_SCRIPT_URL"):
            _surl = os.getenv("GOOGLE_APPS_SCRIPT_URL", "").rstrip("/")
            _sep = "&" if "?" in _surl else "?"
            _resp = _requests.get(f"{_surl}{_sep}type=gmail_message&id={message_id}", timeout=15)
            _resp.raise_for_status()
            _data = _resp.json()
            if "error" in _data:
                raise MCPToolError(FailureKind.USER_ERROR, _data["error"],
                                   {"tool": "gmail_get_email", "message_id": message_id})
            return json.dumps({
                "from":    _data.get("from", ""),
                "to":      _data.get("to", ""),
                "cc":      _data.get("cc") or None,
                "subject": _data.get("subject", "(no subject)"),
                "date":    _data.get("date", ""),
                "body":    _data.get("body", ""),
                "link":    _data.get("link", f"https://mail.google.com/mail/u/0/#inbox/{message_id}"),
                "id":      _data.get("id", message_id),
            }, indent=2)

        service = _gmail_service()
        if not service:
            raise MCPToolError(FailureKind.USER_ERROR,
                               "Could not authenticate with Google — check credentials.json and token.json",
                               {"tool": "gmail_get_email"})

        msg = service.users().messages().get(
            userId="me", id=message_id, format="full").execute()

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
    except Exception as e:
        logger.error(f"❌ Gmail get email error: {e}")
        raise MCPToolError(FailureKind.UPSTREAM_ERROR, f"Gmail error: {e}",
                           {"tool": "gmail_get_email", "message_id": message_id})


@mcp.tool()
@check_tool_enabled(category="google")
@tool_meta(tags=["write","email","external"],triggers=["send email","compose email","email someone","write email","draft email","send a message","compose a message"],idempotent=False,template='use gmail_send_email: to="" [subject=""] body="" [cc=""] [html=""]',intent_category="google",
output_type="none",pipe_targets={"body":"text"})
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

    if not to or not to.strip():
        raise MCPToolError(FailureKind.USER_ERROR, "to must not be empty",
                           {"tool": "gmail_send_email"})
    if not subject or not subject.strip():
        raise MCPToolError(FailureKind.USER_ERROR, "subject must not be empty",
                           {"tool": "gmail_send_email"})

    if os.getenv("GOOGLE_APPS_SCRIPT_URL"):
        try:
            payload = {"type": "send_email", "to": to, "subject": subject, "body": body}
            if cc:
                payload["cc"] = cc
            if html:
                payload["htmlBody"] = body
            _script_post(payload)
            logger.info(f"✅ Email sent via Apps Script to {to}")
            return json.dumps({"status": "sent", "to": to, "cc": cc or None, "subject": subject,
                               "summary": f"Email sent to {to}: '{subject}'"}, indent=2)
        except Exception as e:
            logger.error(f"❌ Gmail send (Apps Script) failed: {e}")
            raise MCPToolError(FailureKind.UPSTREAM_ERROR, f"Gmail send error: {e}",
                               {"tool": "gmail_send_email", "to": to, "subject": subject})

    if not GOOGLE_AVAILABLE:
        return _not_available("gmail_send_email")

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

    title = event.get("summary", "(no title)")
    result = {
        "title":   title,
        "when":    when,
        "all_day": all_day,
    }
    if event.get("location"):
        raw_loc = event["location"]
        import urllib.parse as _up
        result["location"] = f"[{raw_loc}](https://maps.google.com/?q={_up.quote(raw_loc)})"
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

    # Pre-formatted summary for the list builder
    summary_parts = [f"📅 {when}"]
    if clean_desc:
        summary_parts.append(f"   {clean_desc[:100]}")
    if attendees_str:
        summary_parts.append(f"   👥 {attendees_str}")
    if event.get("hangoutLink"):
        summary_parts.append(f"   🎥 {event['hangoutLink']}")
    result["summary"] = "\n".join(summary_parts)

    return result


@mcp.tool()
@check_tool_enabled(category="google")
@tool_meta(tags=["read","calendar","external"],triggers=["calendar today","schedule today","meetings today","whats on today","what's on today","what do i have today","today's events","what's happening today","anything today"],idempotent=False,template="use calendar_get_today",intent_category="google",text_fields=["text"])
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

    # ── Data collection ───────────────────────────────────────────────────────
    events = []
    date_label = ""

    if os.getenv("GOOGLE_APPS_SCRIPT_URL"):
        try:
            _surl = os.getenv("GOOGLE_APPS_SCRIPT_URL", "").rstrip("/")
            _sep = "&" if "?" in _surl else "?"
            _resp = _requests.get(f"{_surl}{_sep}type=calendar&offset=0&days=1", timeout=10)
            _resp.raise_for_status()
            _data = _resp.json()
            if "error" in _data:
                raise RuntimeError(_data["error"])
            events = [
                {
                    "title":          e.get("title", ""),
                    "when":           e.get("time", ""),
                    "location":       e.get("location", ""),
                    "notes":          e.get("notes", ""),
                    "organizer":      e.get("organizer", ""),
                    "attendees":      e.get("attendees", ""),
                    "calendar_link":  e.get("calendarLink", ""),
                }
                for e in _data.get("events", [])
            ]
            date_label = _data.get("date", "")
            logger.info(f"✅ Found {len(events)} events today (Apps Script)")
        except Exception as e:
            logger.error(f"❌ calendar_get_today (Apps Script) failed: {e}")
            raise MCPToolError(FailureKind.UPSTREAM_ERROR, f"Calendar fetch error: {e}",
                               {"tool": "calendar_get_today"})
    else:
        if not GOOGLE_AVAILABLE:
            return _not_available("calendar_get_today")
        try:
            service = _calendar_service()
            if not service:
                raise MCPToolError(FailureKind.USER_ERROR, "Could not authenticate with Google — check credentials.json and token.json",
                                   {"tool": "calendar_get_today"})
            try:
                from zoneinfo import ZoneInfo
                from tools.location.resolve_timezone import resolve_timezone
                _tz = ZoneInfo(resolve_timezone(
                    os.getenv("DEFAULT_CITY", ""), os.getenv("DEFAULT_STATE", ""), os.getenv("DEFAULT_COUNTRY", "")
                ))
            except Exception:
                _tz = timezone.utc
            now = datetime.now(_tz)
            start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_day = start_of_day + timedelta(days=1)
            calendar_ids = _get_all_calendar_ids(service)
            seen_ids = set()
            for cal_id in calendar_ids:
                try:
                    result = service.events().list(
                        calendarId=cal_id,
                        timeMin=start_of_day.isoformat(),
                        timeMax=end_of_day.isoformat(),
                        singleEvents=True, orderBy="startTime"
                    ).execute()
                    for e in result.get("items", []):
                        if e["id"] not in seen_ids:
                            seen_ids.add(e["id"])
                            events.append(_format_event(e))
                except Exception as _ce:
                    logger.warning(f"⚠️  Could not fetch calendar {cal_id}: {_ce}")
            events.sort(key=lambda e: e.get("start", ""))
            date_label = start_of_day.strftime("%A, %B %-d %Y")
            logger.info(f"✅ Found {len(events)} events today across {len(calendar_ids)} calendar(s)")
        except MCPToolError:
            raise
        except HttpError as e:
            logger.error(f"❌ Calendar API error: {e}")
            raise MCPToolError(FailureKind.UPSTREAM_ERROR, f"Calendar API error: {e}",
                               {"tool": "calendar_get_today"})

    # ── Shared output building ────────────────────────────────────────────────
    if not events:
        text_out = f"No events today ({date_label})."
    else:
        lines = []
        for i, ev in enumerate(events, 1):
            lines.append(f"{i}. **{ev['title']}**")
            lines.append(f"   **When:** {ev['when']}")
            if ev.get("notes"):
                lines.append(f"   **Notes:** {ev['notes'][:200]}")
            if ev.get("location"):
                lines.append(f"   **Location:** {ev['location']}")
            if ev.get("attendees"):
                lines.append(f"   **Attendees:** {ev['attendees']}")
            if ev.get("organizer"):
                lines.append(f"   **Organizer:** {ev['organizer']}")
            if ev.get("meet_link"):
                lines.append(f"   **Meet:** {ev['meet_link']}")
            if ev.get("calendar_link"):
                lines.append(f"   **Calendar:** {ev['calendar_link']}")
        text_out = "\n".join(lines)

    return json.dumps({
        "date":   date_label,
        "count":  len(events),
        "text":   text_out,
        "events": events,
    }, indent=2)


@mcp.tool()
@check_tool_enabled(category="google")
@tool_meta(tags=["read","calendar","external"],triggers=["this week calendar","weekly schedule","meetings this week","whats on this week","week ahead","week's events","schedule this week","what's happening this week"],idempotent=False,template="use calendar_get_this_week",intent_category="google",text_fields=["text"])
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

    # ── Timezone (shared) ─────────────────────────────────────────────────────
    try:
        from zoneinfo import ZoneInfo
        from tools.location.resolve_timezone import resolve_timezone
        _tz = ZoneInfo(resolve_timezone(
            os.getenv("DEFAULT_CITY", ""), os.getenv("DEFAULT_STATE", ""), os.getenv("DEFAULT_COUNTRY", "")
        ))
    except Exception:
        _tz = timezone.utc
    now = datetime.now(_tz)
    monday = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    sunday = monday + timedelta(days=7)

    # ── Data collection ───────────────────────────────────────────────────────
    # Both paths produce a flat list of events; each event has a "start_date" (YYYY-MM-DD) for grouping
    flat_events = []

    if os.getenv("GOOGLE_APPS_SCRIPT_URL"):
        try:
            _surl = os.getenv("GOOGLE_APPS_SCRIPT_URL", "").rstrip("/")
            _sep = "&" if "?" in _surl else "?"
            _offset = -now.weekday()
            _resp = _requests.get(f"{_surl}{_sep}type=calendar&offset={_offset}&days=7", timeout=10)
            _resp.raise_for_status()
            _data = _resp.json()
            if "error" in _data:
                raise RuntimeError(_data["error"])
            for e in _data.get("events", []):
                flat_events.append({
                    "title":      e.get("title", ""),
                    "when":          e.get("time", ""),
                    "start_date":    e.get("startDate", ""),
                    "location":      e.get("location", ""),
                    "notes":         e.get("notes", ""),
                    "organizer":     e.get("organizer", ""),
                    "attendees":     e.get("attendees", ""),
                    "calendar_link": e.get("calendarLink", ""),
                })
            logger.info(f"✅ Fetched {len(flat_events)} events this week (Apps Script)")
        except Exception as e:
            logger.error(f"❌ calendar_get_this_week (Apps Script) failed: {e}")
            raise MCPToolError(FailureKind.UPSTREAM_ERROR, f"Calendar fetch error: {e}",
                               {"tool": "calendar_get_this_week"})
    else:
        if not GOOGLE_AVAILABLE:
            return _not_available("calendar_get_this_week")
        try:
            service = _calendar_service()
            if not service:
                raise MCPToolError(FailureKind.USER_ERROR, "Could not authenticate with Google — check credentials.json and token.json",
                                   {"tool": "calendar_get_this_week"})
            calendar_ids = _get_all_calendar_ids(service)
            seen_ids = set()
            raw = []
            for cal_id in calendar_ids:
                try:
                    result = service.events().list(
                        calendarId=cal_id,
                        timeMin=monday.isoformat(), timeMax=sunday.isoformat(),
                        singleEvents=True, orderBy="startTime"
                    ).execute()
                    for e in result.get("items", []):
                        if e["id"] not in seen_ids:
                            seen_ids.add(e["id"])
                            raw.append(e)
                except Exception as _ce:
                    logger.warning(f"⚠️  Could not fetch calendar {cal_id}: {_ce}")
            raw.sort(key=lambda e: (e.get("start") or {}).get("dateTime") or (e.get("start") or {}).get("date", ""))
            for e in raw:
                start = e.get("start", {})
                start_date = (start.get("dateTime") or start.get("date", ""))[:10]
                ev = _format_event(e)
                ev["start_date"] = start_date
                flat_events.append(ev)
            logger.info(f"✅ Fetched {len(flat_events)} events this week across {len(calendar_ids)} calendar(s)")
        except MCPToolError:
            raise
        except HttpError as e:
            logger.error(f"❌ Calendar API error: {e}")
            raise MCPToolError(FailureKind.UPSTREAM_ERROR, f"Calendar API error: {e}",
                               {"tool": "calendar_get_this_week"})

    # ── Shared output building ────────────────────────────────────────────────
    from datetime import datetime as _dt
    by_day = {}
    day_labels = {}
    for ev in flat_events:
        day = ev.get("start_date") or "unknown"
        if day not in by_day:
            by_day[day] = []
            try:
                day_labels[day] = _dt.fromisoformat(day).strftime("%A, %B %-d")
            except Exception:
                day_labels[day] = day
        by_day[day].append(ev)

    total = sum(len(v) for v in by_day.values())
    lines = []
    for day in sorted(by_day.keys()):
        lines.append(f"**{day_labels[day]}**")
        for i, ev in enumerate(by_day[day], 1):
            lines.append(f"  {i}. **{ev['title']}**")
            lines.append(f"     **When:** {ev['when']}")
            if ev.get("notes"):
                lines.append(f"     **Notes:** {ev['notes'][:200]}")
            if ev.get("location"):
                lines.append(f"     **Location:** {ev['location']}")
            if ev.get("attendees"):
                lines.append(f"     **Attendees:** {ev['attendees']}")
            if ev.get("organizer"):
                lines.append(f"     **Organizer:** {ev['organizer']}")
            if ev.get("meet_link"):
                lines.append(f"     **Meet:** {ev['meet_link']}")
            if ev.get("calendar_link"):
                lines.append(f"     **Calendar:** {ev['calendar_link']}")
        lines.append("")

    return json.dumps({
        "week_start": monday.strftime("%Y-%m-%d"),
        "week_end":   (sunday - timedelta(days=1)).strftime("%Y-%m-%d"),
        "count":      total,
        "text":       "\n".join(lines),
        "by_day":     {day: by_day[day] for day in sorted(by_day.keys())},
    }, indent=2)


@mcp.tool()
@check_tool_enabled(category="google")
@tool_meta(tags=["write","calendar","external"],triggers=["create event","schedule meeting","add to calendar","book appointment","new event","put on calendar","add meeting","schedule event"],idempotent=False,template='use calendar_create_event: summary="" start="" end="" [description=""] [location=""] [attendees=""] [all_day=""]',intent_category="google",
output_type="none")
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

    if not summary or not summary.strip():
        raise MCPToolError(FailureKind.USER_ERROR, "summary must not be empty",
                           {"tool": "calendar_create_event"})
    if not start or not end:
        raise MCPToolError(FailureKind.USER_ERROR, "start and end are required",
                           {"tool": "calendar_create_event"})

    if os.getenv("GOOGLE_APPS_SCRIPT_URL"):
        try:
            payload = {"type": "create_event", "title": summary, "start": start, "end": end, "allDay": all_day}
            if description:
                payload["description"] = description
            if location:
                payload["location"] = location
            if attendees:
                payload["guests"] = ",".join(attendees) if isinstance(attendees, list) else attendees
            _script_post(payload)
            logger.info(f"✅ Calendar event created via Apps Script: {summary}")
            return json.dumps({"status": "created", "title": summary, "start": start, "end": end,
                               "summary": f"'{summary}' added to your calendar for {start}"}, indent=2)
        except Exception as e:
            logger.error(f"❌ Calendar create (Apps Script) failed: {e}")
            raise MCPToolError(FailureKind.UPSTREAM_ERROR, f"Calendar create error: {e}",
                               {"tool": "calendar_create_event", "summary": summary})

    if not GOOGLE_AVAILABLE:
        return _not_available("calendar_create_event")

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
@tool_meta(tags=["write","email","external"],triggers=["reply to email","reply to this email","respond to email","write a reply","respond to this","reply back"],idempotent=False,template='use gmail_reply_tool: message_id="" body="" [cc=""]',intent_category="google",
output_type="none",pipe_targets={"body":"text"})
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

    if not GOOGLE_AVAILABLE and not os.getenv("GOOGLE_APPS_SCRIPT_URL"):
        return _not_available("gmail_reply_tool")

    if not message_id or not message_id.strip():
        raise MCPToolError(FailureKind.USER_ERROR, "message_id must not be empty",
                           {"tool": "gmail_reply_tool"})
    if not body or not body.strip():
        raise MCPToolError(FailureKind.USER_ERROR, "body must not be empty",
                           {"tool": "gmail_reply_tool"})

    try:
        if os.getenv("GOOGLE_APPS_SCRIPT_URL"):
            _result = _script_post({"type": "reply_email", "message_id": message_id,
                                    "body": body, **({"cc": cc} if cc else {})})
            logger.info(f"✅ Reply sent via Apps Script: message_id={message_id}")
            return json.dumps({
                "status":  "sent",
                "to":      _result.get("to", ""),
                "subject": _result.get("subject", ""),
                "summary": f"Reply sent to {_result.get('to', '')}: '{_result.get('subject', '')}'"
            }, indent=2)

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
@tool_meta(
    tags=["read","search","email","external"],
    triggers=["search email","find email","look for email","email from","emails about","find messages"],
    idempotent=True,
    template='use gmail_search: query="" [max_results=""]',
    intent_category="google",
    text_fields=["text"]
)
def gmail_search(query: str, max_results: int = 20) -> str:
    """
    Search Gmail messages using a Gmail query string.

    Supports all Gmail search operators:
      from:john@example.com   — sender filter
      to:me                   — recipient filter
      subject:invoice         — subject keyword
      is:unread               — unread only
      is:starred              — starred messages
      after:2024/01/01        — date range
      has:attachment          — has attachments
      label:work              — label filter
      in:inbox                — location filter
      thread:THREAD_ID        — all messages in a thread

    Multiple operators can be combined: "from:john is:unread subject:invoice"

    Args:
        query (str, required): Gmail search query string
        max_results (int, optional): Maximum messages to return. Default: 20.

    Returns:
        JSON with:
        - query: The search query used
        - count: Number of messages returned
        - messages: List of matching messages (id, thread_id, from, subject, date, snippet, link)
        - text: Human-readable formatted list

    Use cases:
        - Scheduler condition: check for new emails matching a pattern
        - "Find all unread emails from john@company.com"
        - "Search for emails with subject containing 'invoice'"
        - "Are there any unread emails from my boss?"
    """
    logger.info(f"🛠  gmail_search called: query={query!r} max={max_results}")

    if not GOOGLE_AVAILABLE and not os.getenv("GOOGLE_APPS_SCRIPT_URL"):
        return _not_available("gmail_search")

    if not query or not query.strip():
        raise MCPToolError(FailureKind.USER_ERROR, "query must not be empty",
                           {"tool": "gmail_search"})

    try:
        import html as _html
        import urllib.parse as _urlparse

        # ── Data collection ──────────────────────────────────────────────────
        messages = []
        if os.getenv("GOOGLE_APPS_SCRIPT_URL"):
            _surl = os.getenv("GOOGLE_APPS_SCRIPT_URL", "").rstrip("/")
            _sep = "&" if "?" in _surl else "?"
            _q = _urlparse.quote(query)
            _resp = _requests.get(
                f"{_surl}{_sep}type=gmail_search&query={_q}&max_results={max_results}", timeout=15)
            _resp.raise_for_status()
            _data = _resp.json()
            if "error" in _data:
                raise RuntimeError(_data["error"])
            for _m in _data.get("messages", []):
                messages.append({
                    "id":        _m.get("id", ""),
                    "thread_id": _m.get("thread_id", ""),
                    "from":      _m.get("from", ""),
                    "subject":   _m.get("subject", "(no subject)"),
                    "date":      _m.get("date", ""),
                    "snippet":   _m.get("snippet", ""),
                    "unread":    _m.get("unread", False),
                    "link":      _m.get("link", ""),
                })
        else:
            service = _gmail_service()
            if not service:
                raise MCPToolError(FailureKind.USER_ERROR,
                                   "Could not authenticate with Google — check credentials.json and token.json",
                                   {"tool": "gmail_search"})
            result = service.users().messages().list(
                userId="me", q=query, maxResults=max_results).execute()
            for ref in result.get("messages", []):
                msg = service.users().messages().get(
                    userId="me", id=ref["id"], format="metadata",
                    metadataHeaders=["From", "To", "Subject", "Date"]).execute()
                headers = _parse_message_headers(msg.get("payload", {}).get("headers", []))
                _msg_id = msg["id"]
                messages.append({
                    "id":        _msg_id,
                    "thread_id": msg.get("threadId", ""),
                    "from":      headers.get("from", ""),
                    "subject":   headers.get("subject", "(no subject)"),
                    "date":      headers.get("date", ""),
                    "snippet":   msg.get("snippet", ""),
                    "unread":    "UNREAD" in msg.get("labelIds", []),
                    "link":      f"https://mail.google.com/mail/u/0/#inbox/{_msg_id}",
                })

        # ── Shared output building ───────────────────────────────────────────
        for m in messages:
            m["from"]    = _html.unescape(m.get("from", ""))
            m["subject"] = _html.unescape(m.get("subject", ""))
            m["snippet"] = _html.unescape(m.get("snippet", ""))

        lines = []
        for i, m in enumerate(messages, 1):
            lines.append(f"{i}. **{m['subject']}**")
            lines.append(f"   **From:** {m['from']}")
            lines.append(f"   **Date:** {m['date']}")
            lines.append(f"   **Snippet:** {m['snippet'][:120]}{'…' if len(m['snippet']) > 120 else ''}")
            if m.get("id"):
                lines.append(f"   **ID:** {m['id']}")
            if m.get("thread_id"):
                lines.append(f"   **Thread:** {m['thread_id']}")
            if m.get("link"):
                lines.append(f"   **Link:** {m['link']}")
            lines.append("")

        logger.info(f"✅ gmail_search returned {len(messages)} messages for query={query!r}")
        return json.dumps({
            "query":        query,
            "count":        len(messages),
            "len_messages": len(messages),
            "messages":     messages,
            "text":         "\n".join(lines) if lines else f"No messages found for query: {query}",
        }, indent=2)

    except MCPToolError:
        raise
    except Exception as e:
        logger.error(f"❌ Gmail search error: {e}")
        raise MCPToolError(FailureKind.UPSTREAM_ERROR, f"Gmail error: {e}",
                           {"tool": "gmail_search", "query": query})


@mcp.tool()
@check_tool_enabled(category="google")
@tool_meta(
    tags=["read","email","external"],
    triggers=["have i replied","did i reply","check if replied","replied to thread","reply status"],
    idempotent=True,
    template='use gmail_check_replied: thread_id="" [since_hours=""]',
    intent_category="google",
    text_fields=["text"]
)
def gmail_check_replied(thread_id: str, since_hours: Optional[float] = None) -> str:
    """
    Check whether you have sent a reply in a given Gmail thread.

    Fetches all messages in the thread and checks whether any are from the
    authenticated user (i.e. in the SENT label). Optionally restricts the
    check to replies sent within the last N hours.

    Args:
        thread_id (str, required): Gmail thread ID (from gmail_search or gmail_get_recent)
        since_hours (float, optional): If provided, only count replies sent within
                                       this many hours. E.g. 2.0 = last 2 hours.
                                       Default: check entire thread history.

    Returns:
        JSON with:
        - thread_id: The thread checked
        - replied: True if you have sent a reply (within the time window if specified)
        - reply_count: Number of your sent messages in the thread (within window)
        - last_reply_at: ISO timestamp of your most recent reply, or null
        - since_hours: The time window used, or null
        - text: Human-readable summary

    Use cases:
        - Scheduler condition: "replied == False" triggers a follow-up action
        - "Have I replied to this thread?"
        - "Check if I've responded to John's email in the last 2 hours"
    """
    logger.info(f"🛠  gmail_check_replied called: thread_id={thread_id} since_hours={since_hours}")

    if not GOOGLE_AVAILABLE and not os.getenv("GOOGLE_APPS_SCRIPT_URL"):
        return _not_available("gmail_check_replied")

    if not thread_id or not thread_id.strip():
        raise MCPToolError(FailureKind.USER_ERROR, "thread_id must not be empty",
                           {"tool": "gmail_check_replied"})

    try:
        if os.getenv("GOOGLE_APPS_SCRIPT_URL"):
            _surl = os.getenv("GOOGLE_APPS_SCRIPT_URL", "").rstrip("/")
            _sep = "&" if "?" in _surl else "?"
            _url = f"{_surl}{_sep}type=gmail_check_replied&thread_id={thread_id}"
            if since_hours is not None:
                _url += f"&since_hours={since_hours}"
            _resp = _requests.get(_url, timeout=15)
            _resp.raise_for_status()
            _data = _resp.json()
            if "error" in _data:
                raise MCPToolError(FailureKind.USER_ERROR, _data["error"],
                                   {"tool": "gmail_check_replied", "thread_id": thread_id})
            replied      = _data.get("replied", False)
            reply_count  = _data.get("reply_count", 0)
            last_reply_at = _data.get("last_reply_at")
        else:
            service = _gmail_service()
            if not service:
                raise MCPToolError(FailureKind.USER_ERROR,
                                   "Could not authenticate with Google — check credentials.json and token.json",
                                   {"tool": "gmail_check_replied"})
            thread = service.users().threads().get(
                userId="me", id=thread_id, format="metadata",
                metadataFields="messages/id,messages/labelIds,messages/internalDate,messages/payload/headers"
            ).execute()
            cutoff_ms = None
            if since_hours is not None:
                cutoff_dt = datetime.now(timezone.utc) - timedelta(hours=since_hours)
                cutoff_ms = int(cutoff_dt.timestamp() * 1000)
            sent_messages = []
            for msg in thread.get("messages", []):
                if "SENT" not in msg.get("labelIds", []):
                    continue
                internal_date_ms = int(msg.get("internalDate", 0))
                if cutoff_ms is not None and internal_date_ms < cutoff_ms:
                    continue
                headers = {h["name"].lower(): h["value"]
                           for h in (msg.get("payload") or {}).get("headers", [])}
                sent_messages.append({"id": msg["id"], "date": headers.get("date", ""),
                                      "sent_at_ms": internal_date_ms})
            sent_messages.sort(key=lambda m: m["sent_at_ms"])
            replied      = len(sent_messages) > 0
            reply_count  = len(sent_messages)
            last_reply_at = None
            if sent_messages:
                from datetime import datetime as _dt
                last_ms = sent_messages[-1]["sent_at_ms"]
                last_reply_at = _dt.fromtimestamp(last_ms / 1000, tz=timezone.utc).isoformat()

        window_str = f" in the last {since_hours}h" if since_hours is not None else ""
        if replied:
            text = f"You have sent {reply_count} reply/replies in this thread{window_str}. Last reply: {last_reply_at}."
        else:
            text = f"No reply found from you in this thread{window_str}."

        logger.info(f"✅ gmail_check_replied: thread={thread_id} replied={replied} count={reply_count}")
        return json.dumps({
            "thread_id":    thread_id,
            "replied":      replied,
            "reply_count":  reply_count,
            "last_reply_at": last_reply_at,
            "since_hours":  since_hours,
            "text":         text,
        }, indent=2)

    except MCPToolError:
        raise
    except Exception as e:
        logger.error(f"❌ gmail_check_replied error: {e}")
        raise MCPToolError(FailureKind.UPSTREAM_ERROR, f"Gmail error: {e}",
                           {"tool": "gmail_check_replied", "thread_id": thread_id})

@mcp.tool()
@check_tool_enabled(category="google")
@tool_meta(
    tags=["read","email","calendar","external"],
    triggers=["my day","day briefing","morning briefing","today's summary",
              "how's my day","daily briefing","brief me","start of day",
              "tomorrow's briefing","what's on tomorrow","tomorrow's schedule",
              "day summary","get day briefing"],
    idempotent=False,
    template='use get_day_briefing: [date_offset=""] [max_emails=""] [forecast_days=""] [calendar_days=""]',
    intent_category="google",
    text_fields=["weather.text","email.text","calendar.text"]
)
def get_day_briefing(max_emails: Optional[int] = 10, forecast_days: Optional[int] = 1, calendar_days: Optional[int] = 1, date_offset: Optional[int] = 0) -> str:
    """
    Get a combined briefing for a given day: weather, unread emails, and calendar events.

    Calls get_weather_tool, gmail_get_unread, and calendar_get_today internally
    and returns all three in a single structured response.

    Args:
        max_emails (int, optional):    Max unread emails to include (default: 10)
        forecast_days (int, optional): Days of weather forecast (default: 1 = today only)
        calendar_days (int, optional): Days of calendar events to include (default: 1 = today only)
        date_offset (int, optional):   Day offset from today (default: 0 = today, 1 = tomorrow).
                                       Use 1 when the user asks about tomorrow, next day, or the day after today.

    Returns:
        JSON with:
        - weather:   Current conditions and forecast (same format as get_weather_tool)
        - email:     Unread emails (same format as gmail_get_unread)
        - calendar:  Events for the requested day (same format as calendar_get_today)
        - errors:    Any per-section errors that occurred
    """
    max_emails    = int(max_emails)    if max_emails    is not None else 10
    forecast_days = int(forecast_days) if forecast_days is not None else 1
    calendar_days = int(calendar_days) if calendar_days is not None else 1
    date_offset   = int(date_offset)   if date_offset   is not None else 0
    logger.info(f"🛠  get_day_briefing called (max_emails={max_emails}, forecast_days={forecast_days}, calendar_days={calendar_days}, date_offset={date_offset})")

    # Compute local date label up front for the briefing header
    try:
        from zoneinfo import ZoneInfo
        from tools.location.resolve_timezone import resolve_timezone
        _tz_name = resolve_timezone(
            os.getenv("DEFAULT_CITY", ""),
            os.getenv("DEFAULT_STATE", ""),
            os.getenv("DEFAULT_COUNTRY", ""),
        )
        _now = datetime.now(ZoneInfo(_tz_name))
    except Exception:
        _now = datetime.now(timezone.utc)
    _now = _now + timedelta(days=date_offset)
    _date_label = _now.strftime("%A, %B %-d %Y")

    result = {"date": _date_label, "weather": None, "email": None, "calendar": None, "errors": {}}

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

        # Fetch enough days so that forecast[date_offset] is the target date.
        # e.g. date_offset=2, forecast_days=1 → fetch 3 days, take day index 2.
        _fetch_days = max(1, date_offset + forecast_days)
        weather_raw = get_weather_fn(city, state, country, forecast_days=_fetch_days)
        weather_data = json.loads(weather_raw) if isinstance(weather_raw, str) else weather_raw
        # Trim forecast to start at date_offset so the first entry is the target day
        if date_offset > 0 and isinstance(weather_data, dict) and weather_data.get("forecast"):
            weather_data = dict(weather_data)
            _full_forecast = weather_data["forecast"]
            _sliced = _full_forecast[date_offset:date_offset + forecast_days]
            if not _sliced:
                # API returned fewer days than date_offset — use whatever is last available
                logger.warning(f"⚠️  Weather forecast slice empty: date_offset={date_offset}, forecast_days={forecast_days}, api_returned={len(_full_forecast)}. Using last available day.")
                _sliced = _full_forecast[-forecast_days:] if _full_forecast else []
            weather_data["forecast"] = _sliced
        # Forward pre-rendered text if the weather tool produced one; otherwise build a fallback
        if isinstance(weather_data, dict) and not weather_data.get("text"):
            _cur = weather_data.get("current_conditions") or weather_data.get("current") or {}
            _loc = weather_data.get("location", "")
            _desc = _cur.get("description") or _cur.get("condition", "")
            _temp = _cur.get("temperature") or _cur.get("temp", "")
            _feel = _cur.get("feels_like", "")
            _wind = _cur.get("wind", "")
            _lines = [f"📍 {_loc}" if _loc else None,
                      f"🌤 {_desc}" if _desc else None,
                      f"🌡 {_temp}" + (f" (feels like {_feel})" if _feel else "") if _temp else None,
                      f"💨 Wind: {_wind}" if _wind else None]
            _forecast = weather_data.get("forecast", [])
            for _day in _forecast[:3]:
                _day_label = _day.get("day") or _day.get("date", "")
                _day_desc  = _day.get("description") or _day.get("condition", "")
                _day_hi    = _day.get("high") or _day.get("temp_max", "")
                _day_lo    = _day.get("low") or _day.get("temp_min", "")
                if _day_label:
                    _lines.append(f"  {_day_label}: {_day_desc} {_day_hi}/{_day_lo}".strip())
            weather_data["text"] = "\n".join(l for l in _lines if l)
        result["weather"] = weather_data
        if isinstance(weather_data, dict) and weather_data.get("error"):
            logger.warning(f"⚠️  Weather API error: {weather_data.get('error')} — {weather_data.get('message', '')}")
        else:
            logger.info("✅ Weather fetched")
    except Exception as e:
        logger.warning(f"⚠️  Weather failed: {e}")
        result["errors"]["weather"] = str(e)

    # ── Email + Calendar ──────────────────────────────────────────────────────
    _script_url = os.getenv("GOOGLE_APPS_SCRIPT_URL", "").rstrip("/")

    if _script_url:
        # ── Apps Script path (no OAuth) ───────────────────────────────────────
        # URL already contains ?key=... — append remaining params with &
        _sep = "&" if "?" in _script_url else "?"
        try:
            _resp = _requests.get(
                f"{_script_url}{_sep}type=gmail&max_emails={max_emails}",
                timeout=10
            )
            _resp.raise_for_status()
            _gmail_data = _resp.json()
            if "error" in _gmail_data:
                raise RuntimeError(_gmail_data["error"])
            _msgs = _gmail_data.get("messages", [])
            emails = [
                {
                    "from":    m.get("from", ""),
                    "subject": m.get("subject", "(no subject)"),
                    "date":    m.get("date", ""),
                    "id":      m.get("id", ""),
                    "link":    m.get("link", ""),
                    "preview": m.get("preview", ""),
                }
                for m in _msgs
            ]
            result["email"] = {"total_unread": _gmail_data.get("unreadCount", 0), "emails": emails}
            logger.info(f"✅ {len(emails)} unread emails fetched (Apps Script)")
        except Exception as e:
            logger.warning(f"⚠️  Gmail (Apps Script) failed: {e}")
            result["errors"]["email"] = str(e)

        try:
            _cresp = _requests.get(
                f"{_script_url}{_sep}type=calendar&offset={date_offset}&days={calendar_days}",
                timeout=10
            )
            _cresp.raise_for_status()
            _cal_data = _cresp.json()
            if "error" in _cal_data:
                raise RuntimeError(_cal_data["error"])
            _evs = _cal_data.get("events", [])
            events = [
                {
                    "title":         e.get("title", ""),
                    "when":          e.get("time", ""),
                    "location":      e.get("location", ""),
                    "notes":         e.get("notes", ""),
                    "organizer":     e.get("organizer", ""),
                    "attendees":     e.get("attendees", ""),
                    "calendar_link": e.get("calendarLink", ""),
                }
                for e in _evs
            ]
            result["calendar"] = {
                "date":   _cal_data.get("date", ""),
                "count":  len(events),
                "events": events,
            }
            logger.info(f"✅ {len(events)} calendar events fetched (Apps Script)")
        except Exception as e:
            logger.warning(f"⚠️  Calendar (Apps Script) failed: {e}")
            result["errors"]["calendar"] = str(e)

    else:
        # ── OAuth path (fallback when Apps Script not configured) ─────────────
        if not GOOGLE_AVAILABLE:
            result["errors"]["email"] = "Google API libraries not installed"
            result["errors"]["calendar"] = "Google API libraries not installed"
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

            try:
                service = _calendar_service()
                if service:
                    try:
                        from zoneinfo import ZoneInfo
                        from tools.location.resolve_timezone import resolve_timezone
                        _city    = os.getenv("DEFAULT_CITY", "")
                        _state   = os.getenv("DEFAULT_STATE", "")
                        _country = os.getenv("DEFAULT_COUNTRY", "")
                        _tz_name = resolve_timezone(_city, _state, _country)
                        _tz = ZoneInfo(_tz_name)
                    except Exception:
                        _tz = timezone.utc
                    now = datetime.now(_tz)
                    start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=date_offset)
                    end_of_day = start_of_day + timedelta(days=calendar_days)
                    calendar_ids = _get_all_calendar_ids(service)
                    events = []
                    seen_ids = set()
                    for cal_id in calendar_ids:
                        try:
                            res = service.events().list(
                                calendarId=cal_id,
                                timeMin=start_of_day.isoformat(),
                                timeMax=end_of_day.isoformat(),
                                singleEvents=True,
                                orderBy="startTime"
                            ).execute()
                            for e in res.get("items", []):
                                if e["id"] not in seen_ids:
                                    seen_ids.add(e["id"])
                                    events.append(_format_event(e))
                        except Exception as _ce:
                            logger.warning(f"⚠️  Could not fetch calendar {cal_id}: {_ce}")
                    events.sort(key=lambda e: e.get("start", ""))
                    _cal_lines = []
                    for _i_ev, _ev in enumerate(events, 1):
                        _cal_lines.append(f"{_i_ev}. **{_ev['title']}**")
                        _cal_lines.append(f"   **When:** {_ev['when']}")
                        if _ev.get("notes"):
                            _cal_lines.append(f"   **Notes:** {_ev['notes'][:200]}")
                        if _ev.get("location"):
                            _cal_lines.append(f"   **Location:** {_ev['location']}")
                        if _ev.get("organizer"):
                            _cal_lines.append(f"   **Organizer:** {_ev['organizer']}")
                        if _ev.get("attendees"):
                            _cal_lines.append(f"   **Attendees:** {_ev['attendees']}")
                        if _ev.get("meet_link"):
                            _cal_lines.append(f"   **Meet:** {_ev['meet_link']}")
                        if _ev.get("calendar_link"):
                            _cal_lines.append(f"   **Calendar:** {_ev['calendar_link']}")
                        _cal_lines.append("")
                    result["calendar"] = {
                        "date":   start_of_day.strftime("%Y-%m-%d"),
                        "count":  len(events),
                        "text":   "\n".join(_cal_lines),
                        "events": events,
                    }
                    logger.info(f"✅ {len(events)} calendar events fetched")
            except Exception as e:
                logger.warning(f"⚠️  Calendar failed: {e}")
                result["errors"]["calendar"] = str(e)

    if not result["errors"]:
        del result["errors"]

    # Build a pre-formatted text so _process_tool_result returns it directly
    # without LLM involvement, giving a consistent layout every time.
    _text_parts = [f"## {result['date']}"]

    _wd = result.get("weather")
    _weather_err = (result.get("errors") or {}).get("weather")
    if _weather_err:
        _text_parts.append(f"\n### Weather\nunavailable ({_weather_err})")
    elif isinstance(_wd, dict) and not _wd.get("forecast"):
        _api_err = _wd.get("message") or _wd.get("error") or "no forecast data returned"
        _text_parts.append(f"\n### Weather\nunavailable ({_api_err})")
    if isinstance(_wd, dict) and _wd.get("forecast"):
        _city  = _wd.get("city", "")
        _state = _wd.get("state", "")
        _ctry  = _wd.get("country", "")
        _loc   = ", ".join(p for p in [_city, _state, _ctry] if p)
        _text_parts.append(f"\n### Weather — {_loc}")
        _cur = _wd.get("current", {})
        _cur_humidity = _cur.get("humidity")
        for _idx, _day in enumerate(_wd.get("forecast", [])):
            _label = _day.get("day_label") or _day.get("date", "")
            _text_parts.append(f"\n**{_label}**  ")
            _text_parts.append(f"**Conditions:** {_day.get('condition', 'N/A')}  ")
            _text_parts.append(f"**Precipitation:** {_day.get('precipitation_chance', 'N/A')}  ")
            _hi_c = _day.get("max_temp_c"); _hi_f = _day.get("max_temp_f")
            _lo_c = _day.get("min_temp_c"); _lo_f = _day.get("min_temp_f")
            _fl_c = _day.get("feelslike_c"); _fl_f = _day.get("feelslike_f")
            if _hi_c is not None:
                _text_parts.append(f"**High:** {_hi_c}°C ({_hi_f}°F)  ")
            if _lo_c is not None:
                _text_parts.append(f"**Low:** {_lo_c}°C ({_lo_f}°F)  ")
            if _fl_c is not None:
                _text_parts.append(f"**Feels Like:** {_fl_c}°C ({_fl_f}°F)  ")
            if _idx == 0 and _cur_humidity and date_offset == 0:
                _text_parts.append(f"**Humidity:** {_cur_humidity}  ")
            if _day.get("sunrise"):
                _text_parts.append(f"**Sunrise:** {_day['sunrise']}  ")
            if _day.get("sunset"):
                _text_parts.append(f"**Sunset:** {_day['sunset']}  ")

    _ed = result.get("email")
    _email_err = (result.get("errors") or {}).get("email")
    if _email_err:
        _text_parts.append(f"\n### Emails\nunavailable ({_email_err})")
    elif isinstance(_ed, dict):
        _total = _ed.get("total_unread", 0)
        if _total == 0:
            _text_parts.append("\n### Emails\nNo unread emails")
        else:
            _text_parts.append(f"\n### Emails — {_total} unread")
            _briefing_emails = _ed.get("emails", [])[:5]
            _briefing_summaries = _summarise_emails(_briefing_emails)
            for _bi, _em in enumerate(_briefing_emails, 1):
                _summary = _briefing_summaries.get(_bi) or (_em.get("preview", "")[:120])
                _text_parts.append(f"\n{_bi}. **{_em.get('subject', '(no subject)')}**  ")
                _text_parts.append(f"**From:** {_em.get('from', '')}  ")
                _text_parts.append(f"**Date:** {_em.get('date', '')}  ")
                if _summary:
                    _text_parts.append(f"**Summary:** {_summary}  ")
                if _em.get("link"):
                    _text_parts.append(f"**Link:** {_em['link']}  ")
    else:
        _text_parts.append("\n### Emails\nunavailable")

    _cd = result.get("calendar")
    _cal_err = (result.get("errors") or {}).get("calendar")
    if _cal_err:
        _text_parts.append(f"\n### Calendar\nunavailable ({_cal_err})")
    elif isinstance(_cd, dict):
        _count = _cd.get("count", 0)
        if _count == 0:
            _text_parts.append("\n### Calendar\nNo events scheduled")
        else:
            _text_parts.append(f"\n### Calendar — {_count} event(s)")
            for _ci, _ev in enumerate(_cd.get("events", [])[:5], 1):
                _text_parts.append(f"\n{_ci}. **{_ev.get('title', '')}**  ")
                _text_parts.append(f"**When:** {_ev.get('when', '')}  ")
                if _ev.get("location"):
                    _text_parts.append(f"**Location:** {_ev['location']}  ")
                if _ev.get("attendees"):
                    _text_parts.append(f"**Attendees:** {_ev['attendees']}  ")
                if _ev.get("meet_link"):
                    _text_parts.append(f"**Meet:** {_ev['meet_link']}  ")
                if _ev.get("calendar_link"):
                    _text_parts.append(f"**Calendar:** {_ev['calendar_link']}  ")

    result["text"] = "\n".join(_text_parts)

    return json.dumps(result, indent=2)


skill_registry = None

# Module-level store for the in-progress OAuth flow (step 1 → step 2)
_reauth_flow = None


@mcp.tool()
@check_tool_enabled(category="google")
@tool_meta(
    tags=["auth","google"],
    triggers=["reauth google","google auth","re-authorise google","google token","fix google auth","google login"],
    idempotent=False,
    template="use google_reauth_start | use google_reauth_complete: code=\"\"",
    intent_category="google",
output_type="none")
def google_reauth_start() -> str:
    """
    Step 1 of 2: Begin Google OAuth re-authorisation.

    Generates an authorisation URL and returns it. Open the URL in your browser,
    approve access, then copy the authorisation code shown by Google and pass it
    to google_reauth_complete.

    Call this when Google tools return auth errors or token.json is missing/invalid.

    Returns:
        JSON with:
        - status: "awaiting_code"
        - auth_url: URL to open in your browser
        - instructions: What to do next
    """
    global _reauth_flow
    logger.info("🛠  google_reauth_start called")

    if not GOOGLE_AVAILABLE:
        return _not_available("google_reauth_start")
    if not Path(CREDENTIALS_FILE).exists():
        raise MCPToolError(FailureKind.USER_ERROR,
                           f"credentials.json not found at {CREDENTIALS_FILE}",
                           {"tool": "google_reauth_start"})

    import urllib.parse as _up
    _reauth_flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
    _reauth_flow.redirect_uri = "urn:ietf:wg:oauth:2.0:oob"

    raw_url, _ = _reauth_flow.authorization_url(
        prompt="consent",
        access_type="offline",
        include_granted_scopes="true",
    )

    # Strip PKCE params — OOB does not support code_challenge
    parsed = _up.urlparse(raw_url)
    qs = _up.parse_qs(parsed.query, keep_blank_values=True)
    qs.pop("code_challenge", None)
    qs.pop("code_challenge_method", None)
    auth_url = parsed._replace(query=_up.urlencode(qs, doseq=True)).geturl()

    # Clear code_verifier from session so fetch_token won't send it
    sess = _reauth_flow.oauth2session
    for attr in ("_code_verifier", "code_verifier", "_pkce_verifier"):
        if hasattr(sess, attr):
            setattr(sess, attr, None)
    if hasattr(sess, "_client"):
        for attr in ("code_verifier", "_code_verifier"):
            if hasattr(sess._client, attr):
                setattr(sess._client, attr, None)

    logger.info("✅ google_reauth_start: auth URL generated")
    return json.dumps({
        "status":       "awaiting_code",
        "auth_url":     auth_url,
        "instructions": (
            "1. Open the auth_url in your browser\n"
            "2. Sign in and approve access\n"
            "3. If you see an 'unverified app' warning: click Advanced → Go to mcp-platform (unsafe)\n"
            "4. Copy the authorisation code shown by Google\n"
            "5. Call google_reauth_complete with the code"
        ),
    }, indent=2)


@mcp.tool()
@check_tool_enabled(category="google")
@tool_meta(
    tags=["auth","google"],
    triggers=["complete google auth","google auth code","finish google reauth","submit auth code"],
    idempotent=False,
    template='use google_reauth_complete: code=""',
    intent_category="google",
output_type="none")
def google_reauth_complete(code: str) -> str:
    """
    Step 2 of 2: Complete Google OAuth re-authorisation.

    Takes the authorisation code from google_reauth_start and exchanges it for
    a token, writing token.json. All Google tools will work immediately after.

    Args:
        code (str, required): The authorisation code shown by Google after approving access.

    Returns:
        JSON with:
        - status: "authorised"
        - token_file: Path where token.json was written
        - scopes: List of granted scopes
    """
    global _reauth_flow, _cached_creds
    logger.info("🛠  google_reauth_complete called")

    if not GOOGLE_AVAILABLE:
        return _not_available("google_reauth_complete")
    if not code or not code.strip():
        raise MCPToolError(FailureKind.USER_ERROR, "code must not be empty",
                           {"tool": "google_reauth_complete"})
    if _reauth_flow is None:
        raise MCPToolError(FailureKind.USER_ERROR,
                           "No auth flow in progress — call google_reauth_start first",
                           {"tool": "google_reauth_complete"})

    try:
        clean_code = "".join(code.split())  # strip any whitespace/newlines

        # Ensure no code_verifier is sent — OOB does not support PKCE.
        # The library may have stored a verifier internally; patch it out at every level.
        sess = _reauth_flow.oauth2session
        for attr in ("_code_verifier", "code_verifier", "_pkce_verifier"):
            if hasattr(sess, attr):
                setattr(sess, attr, None)
        if hasattr(sess, "_client"):
            cli = sess._client
            for attr in ("code_verifier", "_code_verifier"):
                if hasattr(cli, attr):
                    setattr(cli, attr, None)
            # Patch prepare_request_body to strip code_verifier from token POST
            if hasattr(cli, "prepare_request_body"):
                _orig_prb = cli.prepare_request_body
                def _patched_prb(*a, **kw):
                    kw.pop("code_verifier", None)
                    return _orig_prb(*a, **kw)
                cli.prepare_request_body = _patched_prb

        _reauth_flow.fetch_token(code=clean_code)
        creds = _reauth_flow.credentials
        _reauth_flow = None  # clear after use

        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())

        # Clear credential cache so next tool call loads the fresh token
        _cached_creds = None

        # Remove the pending auth flag so the UI stops showing the banner
        _auth_pending = PROJECT_ROOT / "auth_pending.json"
        if _auth_pending.exists():
            _auth_pending.unlink()

        scopes = list(creds.scopes) if creds.scopes else SCOPES
        logger.info(f"✅ google_reauth_complete: token written to {TOKEN_FILE}")
        return json.dumps({
            "status":     "authorised",
            "token_file": TOKEN_FILE,
            "scopes":     scopes,
            "summary":    "Google authorisation complete — all Google tools are now available",
        }, indent=2)

    except Exception as e:
        _reauth_flow = None
        logger.error(f"❌ google_reauth_complete failed: {e}")
        raise MCPToolError(FailureKind.UPSTREAM_ERROR,
                           f"Token exchange failed: {e}",
                           {"tool": "google_reauth_complete"})


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

    # ── OAuth startup validation ───────────────────────────────────────────────
    _creds_path = Path(CREDENTIALS_FILE)
    _token_path = Path(TOKEN_FILE)

    if not _creds_path.exists():
        logger.info("🔑 credentials.json not found — skipping OAuth validation (no Google integration configured)")
    else:
        # Validate credentials.json structure
        _creds_valid = False
        try:
            import json as _json
            with open(_creds_path) as _f:
                _creds_data = _json.load(_f)
            if "installed" not in _creds_data and "web" not in _creds_data:
                logger.error(f"❌ credentials.json is not a valid OAuth client file — download a fresh copy from Google Cloud Console")
            else:
                _creds_valid = True
                logger.info("🔑 credentials.json: ✅ valid")
        except Exception as _e:
            logger.error(f"❌ credentials.json could not be read: {_e} — download a fresh copy from Google Cloud Console")

        if _creds_valid:
            _AUTH_PENDING_FILE = PROJECT_ROOT / "auth_pending.json"

            def _trigger_reauth(reason: str):
                logger.warning(f"🔑 token.json: ❌ {reason}")
                try:
                    result = json.loads(google_reauth_start())
                    auth_url = result.get("auth_url", "")
                    with open(_AUTH_PENDING_FILE, "w") as _f:
                        json.dump({"auth_url": auth_url, "reason": reason}, _f)
                    logger.warning("🔑 Re-authorisation required. Open this URL in your browser:")
                    logger.warning(f"🔑 {auth_url}")
                    logger.warning("🔑 Then call google_reauth_complete with the code from the browser.")
                except Exception as _e:
                    logger.error(f"🔑 Could not generate auth URL: {_e}")

            if not _token_path.exists():
                _trigger_reauth("missing")
            else:
                try:
                    from google.oauth2.credentials import Credentials as _Creds
                    from google.auth.transport.requests import Request as _Request
                    _tok = _Creds.from_authorized_user_file(str(_token_path), SCOPES)
                    if _tok.valid:
                        logger.info("🔑 token.json: ✅ valid")
                    elif _tok.expired and _tok.refresh_token:
                        try:
                            _tok.refresh(_Request())
                            with open(_token_path, "w") as _f:
                                _f.write(_tok.to_json())
                            logger.info("🔑 token.json: ✅ refreshed successfully")
                        except Exception as _e:
                            _trigger_reauth(f"refresh failed ({_e})")
                    else:
                        _trigger_reauth("invalid and cannot be refreshed")
                except Exception as _e:
                    _trigger_reauth(f"could not be read ({_e})")

    mcp.run(transport="stdio")