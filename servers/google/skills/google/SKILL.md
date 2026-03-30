---
name: google
description: >
  Read and send Gmail emails, and manage Google Calendar events.
  Check unread mail, browse the inbox, send emails, view today's
  schedule, see the week ahead, and create new calendar events.
tags:
  - gmail
  - calendar
  - google
  - email
  - schedule
  - appointments
tools:
  - gmail_get_unread
  - gmail_get_recent
  - gmail_get_email
  - gmail_send_email
  - calendar_get_today
  - calendar_get_this_week
  - calendar_create_event
---

# Google (Gmail + Calendar) Skill

## 🎯 Overview

Interact with Gmail and Google Calendar through a single server.

**Gmail capabilities:**
1. **Unread emails** — fetch all unread messages in the inbox
2. **Recent emails** — top 10 inbox messages (read and unread)
3. **Read email** — open the full body of a specific message
4. **Send email** — compose and send a message

**Calendar capabilities:**
1. **Today's events** — everything on the calendar for today
2. **This week's events** — full week view, grouped by day
3. **Create event** — add a new timed or all-day event

**Authentication:** OAuth2 via `credentials.json` (downloaded from Google Cloud Console). A `token.json` is written on first authorization and silently refreshed afterwards. See ⚙️ Configuration below.

---

## 📋 Gmail Workflows

### Check unread mail

```python
gmail_get_unread()           # default: up to 25 unread
gmail_get_unread(max_results=50)
```

**Returns:**
```json
{
  "total_unread": 4,
  "emails": [
    {
      "id": "18f3a...",
      "from": "Alice <alice@example.com>",
      "subject": "Project update",
      "date": "Fri, 20 Jun 2025 09:14:22 +0000",
      "snippet": "Hi, just wanted to let you know..."
    }
  ]
}
```

### Browse recent inbox

```python
gmail_get_recent()           # default: 10 most recent
gmail_get_recent(max_results=20)
```

Each email includes an `"unread": true/false` flag.

### Read a full email

```python
gmail_get_email(message_id="18f3a...")
```

Returns the complete plain-text body plus all headers (`from`, `to`, `cc`, `subject`, `date`).

### Send an email

```python
gmail_send_email(
    to="bob@example.com",
    subject="Meeting rescheduled",
    body="Hi Bob, the meeting is now at 3 PM."
)

# With CC
gmail_send_email(
    to="bob@example.com",
    cc="carol@example.com",
    subject="Shared update",
    body="..."
)

# HTML email
gmail_send_email(
    to="bob@example.com",
    subject="Newsletter",
    body="<h1>Hello</h1><p>Body text here.</p>",
    html=True
)
```

---

## 📋 Calendar Workflows

### Today's schedule

```python
calendar_get_today()
```

**Returns:**
```json
{
  "date": "2025-06-20",
  "count": 2,
  "events": [
    {
      "summary": "Team standup",
      "start": "2025-06-20T09:00:00-07:00",
      "end":   "2025-06-20T09:30:00-07:00",
      "location": "",
      "attendees": ["alice@example.com", "bob@example.com"],
      "meet_link": "https://meet.google.com/abc-defg-hij"
    }
  ]
}
```

### This week's schedule

```python
calendar_get_this_week()
```

Returns events in two forms: a flat `events` list and a `by_day` dict keyed by `"YYYY-MM-DD"` for easy day-by-day presentation.

### Create an event

```python
# Timed event
calendar_create_event(
    summary="Project sync",
    start="2025-06-23T14:00:00",
    end="2025-06-23T15:00:00",
    description="Weekly check-in",
    location="Board room",
    attendees=["alice@example.com", "bob@example.com"]
)

# All-day event
calendar_create_event(
    summary="Conference",
    start="2025-06-25",
    end="2025-06-26",
    all_day=True
)
```

**Returns:**
```json
{
  "status": "created",
  "event_id": "abc123...",
  "summary": "Project sync",
  "start": "2025-06-23T14:00:00-07:00",
  "end": "2025-06-23T15:00:00-07:00",
  "html_link": "https://www.google.com/calendar/event?eid=..."
}
```

---

## 🚀 Complete Examples

### Example 1: Morning briefing

```
User: "Give me my morning briefing — unread emails and today's calendar"

Agent workflow:
1. gmail_get_unread()
   → 3 unread emails

2. calendar_get_today()
   → 2 events: standup at 9 AM, 1:1 at 2 PM

Agent response:
"Good morning! Here's your briefing:

📧 3 unread emails:
- Alice: Project update (9:14 AM)
- Noreply: Your invoice is ready (8:02 AM)
- Bob: Quick question (yesterday)

📅 Today's calendar:
- 9:00–9:30 AM: Team standup (Google Meet)
- 2:00–3:00 PM: 1:1 with Manager
```

### Example 2: Schedule a meeting and notify attendees

```
User: "Schedule a design review Friday at 10am with bob@example.com and send him an email about it"

Agent workflow:
1. calendar_create_event(
       summary="Design review",
       start="2025-06-27T10:00:00",
       end="2025-06-27T11:00:00",
       attendees=["bob@example.com"]
   )
   → Event created, invite sent via calendar

2. gmail_send_email(
       to="bob@example.com",
       subject="Design review — Friday 10 AM",
       body="Hi Bob, I've added a design review on Friday at 10 AM to the calendar. See you then!"
   )
   → Sent

Agent response:
"Done! I've created the design review on Friday June 27 at 10 AM and
added Bob as an attendee. I also sent him an email confirming the details."
```

### Example 3: Weekly planning

```
User: "What's my schedule this week, and do I have any unread emails from Alice?"

Agent workflow:
1. calendar_get_this_week()
   → 8 events across Mon–Fri

2. gmail_get_unread()
   → Filter for emails from Alice → 1 match

Agent response:
"This week you have 8 events:
Monday: Sprint planning (10 AM)
Tuesday: 1:1 (2 PM)
Wednesday: free
...

📧 Alice sent you an unread email:
Subject: 'Updated wireframes' (Tuesday 4:22 PM)"
```

---

## 🔧 Tool Reference

| Tool | Purpose | Key Args |
|------|---------|----------|
| `gmail_get_unread` | Fetch unread inbox messages | `max_results` |
| `gmail_get_recent` | Fetch recent inbox (read + unread) | `max_results` |
| `gmail_get_email` | Read full message body | `message_id` |
| `gmail_send_email` | Compose and send an email | `to`, `subject`, `body`, `cc`, `html` |
| `calendar_get_today` | Today's events | — |
| `calendar_get_this_week` | This week's events | — |
| `calendar_create_event` | Create a calendar event | `summary`, `start`, `end`, `attendees`, `all_day` |

---

## ⚙️ Configuration

### Step 1 — Create a Google Cloud project and enable APIs

1. Go to https://console.cloud.google.com/
2. Create a new project (or use an existing one)
3. Enable the **Gmail API** and **Google Calendar API** under *APIs & Services → Library*

### Step 2 — Create OAuth2 credentials

1. Go to *APIs & Services → Credentials*
2. Click *Create Credentials → OAuth client ID*
3. Application type: **Desktop app**
4. Download the JSON — rename it `credentials.json`
5. Place it next to `server.py`, or set the path in `.env`:

```bash
GOOGLE_CREDENTIALS_FILE=/path/to/credentials.json
GOOGLE_TOKEN_FILE=/path/to/token.json    # optional — defaults to server dir
```

### Step 3 — First-time authorization

On the very first run, a browser window opens asking you to approve access.
After you approve, `token.json` is written and all subsequent runs are silent.

### Required OAuth scopes (set automatically)

| Scope | Purpose |
|-------|---------|
| `gmail.readonly` | Read inbox and messages |
| `gmail.send` | Send emails |
| `calendar.readonly` | Read events |
| `calendar.events` | Create / modify events |

---

## ⚠️ Notes

**Timezone:** `calendar_create_event` defaults to `America/Vancouver`. Change the
`timeZone` value in `server.py` to match your local timezone.

**All-day events:** Pass `all_day=True` and use `"YYYY-MM-DD"` format for `start`/`end`.

**Rate limits:** Gmail API allows ~1 billion quota units/day; Calendar ~1 million.
Normal usage will never approach these limits.

**Token refresh:** `token.json` is refreshed automatically when it expires.
If refresh fails (revoked access), delete `token.json` and re-authorize.

**Install dependencies:**
```bash
pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client
```