"""
client/proactive_agent.py
==========================
Proactive Agent system for mcp-platform.

Two trigger types:
  cron      — time-based (every day at 5:30am)
  condition — poll-based (run when Gmail unread > 10)

Natural language scheduling flow:
  1. User says something scheduling-related
  2. _looks_like_scheduling_request() routes to ScheduleParser
  3. LLM extracts intent → ScheduleConfirmation or ScheduleClarification
  4. Confirmation shown to user; yes/no handled by ConfirmationTracker
  5. On yes → written to scheduler.db, hot-registered in AgentScheduler

:jobs command interface (bypasses LLM entirely):
  :jobs                        — list all jobs
  :jobs pause <label>          — disable
  :jobs enable <label>         — re-enable
  :jobs cancel <label>         — delete
  :jobs info <label>           — full detail

DB lives at data/scheduler.db.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("mcp_client")

# ── Paths ─────────────────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCHEDULER_DB_PATH = _PROJECT_ROOT / "data" / "scheduler.db"

# ── Schema ────────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scheduled_jobs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    label           TEXT    NOT NULL,
    trigger_type    TEXT    NOT NULL DEFAULT 'cron',  -- 'cron' | 'condition'
    tool            TEXT    NOT NULL,
    tool_args       TEXT    NOT NULL DEFAULT '{}',
    cron            TEXT,                              -- 5-field cron (cron jobs)
    condition_tool  TEXT,                              -- tool to call for check (condition jobs)
    condition_tool_args TEXT NOT NULL DEFAULT '{}',    -- args for the condition tool (condition jobs)
    condition_expr  TEXT,                              -- e.g. "result > 10"
    condition_cron  TEXT    NOT NULL DEFAULT '*/15 * * * *',  -- how often to poll
    timezone        TEXT    NOT NULL DEFAULT 'America/Vancouver',
    enabled         INTEGER NOT NULL DEFAULT 1,
    created_at      TEXT    NOT NULL,
    last_run        TEXT,
    last_check      TEXT,
    run_count       INTEGER NOT NULL DEFAULT 0,
    llm_prompt      TEXT,                          -- instruction passed to LLM after tool runs
    session_id      INTEGER,                       -- session to deliver result to (NULL = broadcast)
    deliver_to_session INTEGER NOT NULL DEFAULT 0  -- 1 = deliver to session_id, 0 = broadcast
);
"""

# ── DB helpers ────────────────────────────────────────────────────────────────

def _ensure_db():
    SCHEDULER_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(SCHEDULER_DB_PATH) as conn:
        conn.executescript(_SCHEMA)
        # Migrate: add llm_prompt column if it doesn't exist yet
        cols = [r[1] for r in conn.execute("PRAGMA table_info(scheduled_jobs)").fetchall()]
        if "llm_prompt" not in cols:
            conn.execute("ALTER TABLE scheduled_jobs ADD COLUMN llm_prompt TEXT")
            logger.info("⏰ scheduler.db migrated: added llm_prompt column")
        if "session_id" not in cols:
            conn.execute("ALTER TABLE scheduled_jobs ADD COLUMN session_id INTEGER")
            logger.info("⏰ scheduler.db migrated: added session_id column")
        if "deliver_to_session" not in cols:
            conn.execute("ALTER TABLE scheduled_jobs ADD COLUMN deliver_to_session INTEGER NOT NULL DEFAULT 0")
            logger.info("⏰ scheduler.db migrated: added deliver_to_session column")
        if "run_date" not in cols:
            conn.execute("ALTER TABLE scheduled_jobs ADD COLUMN run_date TEXT")
            logger.info("⏰ scheduler.db migrated: added run_date column")
        if "end_date" not in cols:
            conn.execute("ALTER TABLE scheduled_jobs ADD COLUMN end_date TEXT")
            logger.info("⏰ scheduler.db migrated: added end_date column")
        if "condition_tool_args" not in cols:
            conn.execute("ALTER TABLE scheduled_jobs ADD COLUMN condition_tool_args TEXT NOT NULL DEFAULT '{}'")
            logger.info("⏰ scheduler.db migrated: added condition_tool_args column")


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(SCHEDULER_DB_PATH)
    c.row_factory = sqlite3.Row
    return c


# ── CRUD ──────────────────────────────────────────────────────────────────────

def create_job(
    label: str,
    tool: str,
    trigger_type: str = "cron",
    cron: str = None,
    tool_args: dict = None,
    tz: str = "America/Vancouver",
    condition_tool: str = None,
    condition_tool_args: dict = None,
    condition_expr: str = None,
    condition_cron: str = "*/15 * * * *",
    llm_prompt: str = None,
    run_date: str = None,
    end_date: str = None,
    session_id: int = None,
    deliver_to_session: bool = False,
    # Keep timezone as alias for backwards compat
    timezone: str = None,
) -> int:
    _ensure_db()
    tz_val = timezone or tz  # support both param names
    now = datetime.now(__import__('datetime').timezone.utc).isoformat()
    with _conn() as conn:
        cur = conn.execute(
            """INSERT INTO scheduled_jobs
               (label, trigger_type, tool, tool_args, cron,
                condition_tool, condition_tool_args, condition_expr, condition_cron,
                timezone, enabled, created_at, llm_prompt,
                session_id, deliver_to_session, run_date, end_date)
               VALUES (?,?,?,?,?,?,?,?,?,?,1,?,?,?,?,?,?)""",
            (
                label, trigger_type, tool,
                json.dumps(tool_args or {}),
                cron, condition_tool,
                json.dumps(condition_tool_args or {}),
                condition_expr, condition_cron,
                tz_val, now, llm_prompt,
                session_id, 1 if deliver_to_session else 0,
                run_date, end_date
            )
        )
        return cur.lastrowid


def list_jobs() -> list[dict]:
    _ensure_db()
    with _conn() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM scheduled_jobs ORDER BY id"
        ).fetchall()]


def get_job(job_id: int) -> Optional[dict]:
    _ensure_db()
    with _conn() as conn:
        r = conn.execute("SELECT * FROM scheduled_jobs WHERE id=?", (job_id,)).fetchone()
        return dict(r) if r else None


def find_job_by_label(label: str) -> Optional[dict]:
    _ensure_db()
    with _conn() as conn:
        # Try numeric ID first
        try:
            job_id = int(label.strip())
            r = conn.execute(
                "SELECT * FROM scheduled_jobs WHERE id=?", (job_id,)
            ).fetchone()
            if r:
                return dict(r)
        except (ValueError, TypeError):
            pass
        # Fall back to label substring match
        r = conn.execute(
            "SELECT * FROM scheduled_jobs WHERE LOWER(label) LIKE ?",
            (f"%{label.lower()}%",)
        ).fetchone()
        return dict(r) if r else None


def set_job_enabled(job_id: int, enabled: bool):
    _ensure_db()
    with _conn() as conn:
        conn.execute("UPDATE scheduled_jobs SET enabled=? WHERE id=?",
                     (1 if enabled else 0, job_id))


def delete_job(job_id: int):
    _ensure_db()
    with _conn() as conn:
        conn.execute("DELETE FROM scheduled_jobs WHERE id=?", (job_id,))


def record_run(job_id: int, is_check: bool = False):
    now = datetime.now(__import__('datetime').timezone.utc).isoformat()
    col = "last_check" if is_check else "last_run"
    update = f"SET {col}=?, run_count=run_count+1 WHERE id=?" if not is_check \
        else f"SET {col}=? WHERE id=?"
    with _conn() as conn:
        conn.execute(f"UPDATE scheduled_jobs {update}", (now, job_id))


# ── Cron → human ─────────────────────────────────────────────────────────────

def cron_to_human(cron: str) -> str:
    if not cron:
        return "—"
    parts = cron.strip().split()
    if len(parts) != 5:
        return cron
    minute, hour, dom, month, dow = parts
    time_str = ""
    if minute.isdigit() and hour.isdigit():
        h, m = int(hour), int(minute)
        suffix = "am" if h < 12 else "pm"
        h12 = h if 1 <= h <= 12 else (12 if h == 0 else h - 12)
        time_str = f"{h12}:{m:02d}{suffix}"
    day_map = {"0":"Sun","1":"Mon","2":"Tue","3":"Wed","4":"Thu","5":"Fri","6":"Sat"}
    if dom == "*" and month == "*":
        if dow == "*":
            return f"Every day at {time_str}"
        elif dow == "1-5":
            return f"Weekdays at {time_str}"
        elif dow in ("6,0", "0,6"):
            return f"Weekends at {time_str}"
        else:
            days = ", ".join(day_map.get(d, d) for d in dow.split(","))
            return f"Every {days} at {time_str}"
    return cron


# ── :jobs command handler ─────────────────────────────────────────────────────

def handle_jobs_command(raw: str) -> str:
    """
    Fully deterministic — no LLM involved.
    :jobs                    — list all
    :jobs pause <label>      — pause
    :jobs enable <label>     — enable
    :jobs cancel <label>     — delete
    :jobs info <label>       — full detail
    """
    tokens = raw.strip().split(None, 2)
    if len(tokens) == 1:
        return _format_job_list()
    verb = tokens[1].lower()
    label = tokens[2] if len(tokens) > 2 else ""
    if verb == "pause":
        return _toggle_job(label, False)
    elif verb == "enable":
        return _toggle_job(label, True)
    elif verb in ("cancel", "delete", "remove"):
        if label.strip().lower() == "all":
            return _cancel_all_jobs()
        return _cancel_job(label)
    elif verb == "info":
        return _job_info(label)
    else:
        return (
            "Unknown :jobs subcommand. Available:\n"
            "  :jobs                  — list all\n"
            "  :jobs pause <label>    — pause a job\n"
            "  :jobs enable <label>   — resume a job\n"
            "  :jobs cancel <label>   — delete a job\n"
            "  :jobs info <label>     — full detail\n"
        )


def _format_job_list() -> str:
    jobs = list_jobs()
    if not jobs:
        return (
            "No scheduled jobs.\n"
            "Create one by describing what you want scheduled, e.g.:\n"
            "  \"do a day briefing every day at 5:30am\""
        )
    lines = [f"SCHEDULED JOBS\n{'─'*52}"]
    for j in jobs:
        status = "● active" if j["enabled"] else "⏸ paused"
        ttype = j["trigger_type"]
        if ttype == "once":
            _rd = (j.get("run_date") or "")[:16].replace("T", " ")
            schedule = f"Once at {_rd}"
            schedule_detail = f"run_date: {j.get('run_date', '?')}"
        elif ttype == "cron":
            schedule = cron_to_human(j["cron"])
            schedule_detail = f"cron: {j['cron']}"
        else:
            schedule = f"condition: {j['condition_expr'] or '?'}"
            schedule_detail = f"polls {cron_to_human(j['condition_cron'])}"
        last = j["last_run"][:16].replace("T", " ") if j["last_run"] else "never"
        lines.append(
            f"[{j['id']}] {j['label']}\n"
            f"    Tool:     {j['tool']}\n"
            f"    Schedule: {schedule}  ({schedule_detail})\n"
            f"    Status:   {status}  |  Last run: {last}  |  Runs: {j['run_count']}"
        )
    lines.append("─" * 52)
    lines.append(f"{len(jobs)} job(s) total")
    return "\n".join(lines)


def _toggle_job(label: str, enabled: bool) -> str:
    job = find_job_by_label(label)
    if not job:
        return f"No job found matching '{label}'."
    set_job_enabled(job["id"], enabled)
    return f"Job '{job['label']}' {'resumed' if enabled else 'paused'}."


def _cancel_all_jobs() -> str:
    jobs = list_jobs()
    if not jobs:
        return "No scheduled jobs to cancel."
    for job in jobs:
        delete_job(job["id"])
    return f"✅ Deleted {len(jobs)} job(s)."


def _cancel_job(label: str) -> str:
    job = find_job_by_label(label)
    if not job:
        return f"No job found matching '{label}'."
    delete_job(job["id"])
    return f"Job '{job['label']}' deleted."


def _job_info(label: str) -> str:
    job = find_job_by_label(label)
    if not job:
        return f"No job found matching '{label}'."
    args = json.loads(job["tool_args"] or "{}")
    if job["trigger_type"] == "once":
        sched_lines = f"Run date:  {job.get('run_date', '?')}\n"
    elif job["trigger_type"] == "cron":
        sched_lines = (
            f"Cron:      {job['cron']}\n"
            f"Schedule:  {cron_to_human(job['cron'])}\n"
        )
        if job.get("end_date"):
            sched_lines += f"Ends:      {job['end_date']}\n"
    else:
        sched_lines = (
            f"Condition: {job['condition_tool']} → {job['condition_expr']}\n"
            f"Polls:     {cron_to_human(job['condition_cron'])}\n"
        )
    return (
        f"JOB DETAIL — {job['label']}\n{'─'*44}\n"
        f"ID:        {job['id']}\n"
        f"Type:      {job['trigger_type']}\n"
        f"Tool:      {job['tool']}\n"
        f"Args:      {json.dumps(args) if args else 'none'}\n"
        + sched_lines +
        f"Timezone:  {job['timezone']}\n"
        + (f"LLM:       {job['llm_prompt']}\n" if job.get('llm_prompt') and "|" not in (job.get('llm_prompt') or '') else "") +
        (("Pipeline:\n" + "\n".join(f"  {i+1}. {s.strip()}" for i, s in enumerate((job.get('llm_prompt') or '').split('|'))) + "\n") if job.get('llm_prompt') and "|" in (job.get('llm_prompt') or '') else "") +
        f"Status:    {'active' if job['enabled'] else 'paused'}\n"
        f"Created:   {job['created_at'][:16].replace('T',' ')}\n"
        f"Last run:  {job['last_run'][:16].replace('T',' ') if job['last_run'] else 'never'}\n"
        f"Run count: {job['run_count']}\n"
    )


# ── Schedule confirmation types ───────────────────────────────────────────────

@dataclass
class ScheduleConfirmation:
    label: str
    tool: str
    tool_args: dict
    trigger_type: str           # 'cron' or 'condition'
    cron: Optional[str]
    condition_tool: Optional[str]
    condition_expr: Optional[str]
    condition_cron: str
    timezone: str
    human_schedule: str
    original_request: str
    condition_tool_args: dict = field(default_factory=dict)
    llm_prompt: Optional[str] = None  # instruction passed to LLM after tool runs
    run_date: Optional[str] = None    # ISO datetime for 'once' trigger
    end_date: Optional[str] = None    # ISO datetime to stop cron
    session_id: Optional[int] = None    # session to deliver to (None = broadcast)
    deliver_to_session: bool = False    # True = deliver to session_id, False = broadcast

    def render(self) -> str:
        tool_line = f"  Tool:      {self.tool}\n" if self.tool else ""
        args_str = json.dumps(self.tool_args) if self.tool_args else "none"
        args_line = f"  Args:      {args_str}\n" if self.tool_args else ""

        if self.trigger_type == "once":
            sched = f"  Schedule:  {self.human_schedule}\n"
        elif self.trigger_type == "cron":
            end_line = f"  Ends:      {self.end_date}\n" if self.end_date else ""
            sched = f"  Schedule:  {self.human_schedule}\n" + end_line
        else:
            cta_str = json.dumps(self.condition_tool_args) if self.condition_tool_args else "none"
            sched = (
                f"  Condition: {self.condition_tool}({cta_str}) → {self.condition_expr}\n"
                f"  Polls:     {cron_to_human(self.condition_cron)}\n"
            )

        # Always show condition details if present (even for once trigger)
        condition_line = ""
        if self.condition_tool and self.trigger_type == "once":
            cta_str = json.dumps(self.condition_tool_args) if self.condition_tool_args else "none"
            condition_line = f"  Condition: {self.condition_tool}({cta_str}) → {self.condition_expr}\n"

        llm_line = ""
        if self.llm_prompt:
            if "|" in self.llm_prompt:
                steps = [s.strip() for s in self.llm_prompt.split("|")]
                llm_line = "  Pipeline:\n" + "\n".join(f"    {i+1}. {s}" for i, s in enumerate(steps)) + "\n"
            else:
                llm_line = f"  LLM:       {self.llm_prompt}\n"

        return (
            f"Here's what I'll schedule — confirm to proceed:\n\n"
            f"  Label:     {self.label}\n"
            + tool_line
            + args_line
            + sched
            + condition_line
            + llm_line +
            f"  Timezone:  {self.timezone}\n\n"
            f"Reply **yes** to confirm, **no** to cancel, or describe any changes."
        )


@dataclass
class ScheduleClarification:
    question: str
    partial: dict = field(default_factory=dict)

    def render(self) -> str:
        return self.question


# ── ScheduleParser ────────────────────────────────────────────────────────────

_PARSER_SYSTEM = """You are a scheduling intent extractor for an AI assistant platform.

Extract scheduling parameters from the user's request and return ONLY a JSON object.

AVAILABLE TOOLS:
{tool_list}

OUTPUT — return exactly one of these shapes:

Shape A — single tool, cron trigger:
{{
  "status": "ready",
  "trigger_type": "cron",
  "label": "<3-5 word label>",
  "tool": "<exact tool name>",
  "tool_args": {{}},
  "cron": "<5-field cron>",
  "timezone": "<IANA timezone, default America/Vancouver>",
  "human_schedule": "<plain English>",
  "llm_prompt": "<optional: instruction to pass to the LLM after the tool runs, e.g. 'Summarize this and send it to Discord using discord_notify'>",
  "deliver_to_session": false
}}

Shape B — single tool, condition trigger:
{{
  "status": "ready",
  "trigger_type": "condition",
  "label": "<3-5 word label>",
  "tool": "<action tool to run when condition is true>",
  "tool_args": {{}},
  "condition_tool": "<tool to call for the check>",
  "condition_tool_args": {{}},
  "condition_expr": "<Python expression — see notes below>",
  "condition_cron": "<how often to poll, default */15 * * * *>",
  "run_date": "<ISO datetime if one-time check e.g. 2026-05-29T19:50:00, else null>",
  "timezone": "America/Vancouver",
  "human_schedule": "<plain English description>",
  "llm_prompt": "<optional: instruction for LLM to execute after condition fires>",
  "deliver_to_session": false
}}

Shape D — tool pipeline (multiple tools in sequence), cron or once trigger:
USE THIS when the user asks to call more than one tool, e.g. "get briefing AND send to Discord", "run X then do Y".
The llm_prompt must be a pipe-separated list of `use <tool>` steps.
Each step's output is automatically passed as input to the next step.
{{
  "status": "ready",
  "trigger_type": "cron",
  "label": "<3-5 word label>",
  "tool": "",
  "tool_args": {{}},
  "cron": "<5-field cron, or omit if run_date is set>",
  "run_date": "<ISO datetime if one-time, else null>",
  "timezone": "America/Vancouver",
  "human_schedule": "<plain English>",
  "llm_prompt": "use get_day_briefing | use discord_notify",
  "deliver_to_session": false
}}

Pipeline syntax rules:
- Separate steps with |
- Each step must start with `use <tool_name>`
- Optionally add args: `use discord_notify: message="..."` 
- If no args on a notification step, the previous result is passed automatically as message/body
- For gmail_send_email you MUST specify to and subject in the step args
- Examples:
  "use get_day_briefing | use discord_notify"
  "use gmail_get_unread | use discord_notify"
  "use shashin_random_tool | use gmail_send_email: to=\"michaeltyagi@gmail.com\" subject=\"Daily Photo\""
  "use get_day_briefing | use gmail_send_email: to=\"michaeltyagi@gmail.com\" subject=\"Daily Briefing\""

WHEN TO USE SHAPE D:
- User mentions two or more tool actions: "get X and send to Discord", "run X then notify me", "call X and then Y"
- Any time discord_notify is combined with another data-fetching tool
- Any time the action requires getting data first then sending it somewhere

CONDITION EXPRESSION NOTES:
The expression is evaluated with these variables available:
- result       : raw string output, or parsed number if numeric, or len(list) if JSON array
- data         : parsed JSON dict or list (if tool returns JSON)
- result_len   : len of raw string result
- Any top-level integer/float/bool/string keys from a JSON dict response
  e.g. if tool returns {{"total_unread": 3, "count": 5}} then total_unread and count are available
- len_<key>    : length of any list field e.g. len_emails, len_results

Examples:
  "total_unread > 0"       — Gmail: fires when there are unread emails
  "result > 10"            — numeric tool output exceeds 10
  "result_len > 100"       — raw output is non-trivial (has content)
  "len_results > 0"        — tool returned a non-empty results list
  "count > 5"              — any JSON field named count exceeds 5

Shape C — missing info:
{{
  "status": "clarify",
  "question": "<single specific question>"
}}

STRICT RULES:
1. NEVER guess or assume a time if not given. Use Shape C.
2. NEVER assume which tool if ambiguous. Use Shape C.
3. NEVER default silently. Missing time or frequency = Shape C.
4. cron must be valid 5-field cron.
5. Return ONLY JSON. No preamble, no markdown fences.
6. Set deliver_to_session=true when the user's request implies they want the result in the current conversation (e.g. 'show me', 'tell me', 'give me'). Set false for monitoring/alerting jobs.
7. When the request involves multiple tools or sending data somewhere after fetching it, ALWAYS use Shape D with pipe-separated `use <tool>` steps in llm_prompt.
8. If the request says "today", "tonight", "at X pm today", or any specific one-time time, set run_date to the ISO datetime and set condition_cron to null for Shape B. This means check ONCE at that time, not repeatedly."""


class ScheduleParser:
    def __init__(self, llm_fn, available_tools: list[str],
                 default_timezone: str = "America/Vancouver"):
        self._llm_fn = llm_fn
        self._tools = available_tools
        self._tz = default_timezone

    async def parse(self, user_message: str) -> ScheduleConfirmation | ScheduleClarification:
        # Filter tool list to relevant tools only — reduces context for the LLM.
        # Always include common scheduling tools; also include any tool mentioned by name.
        _ALWAYS_INCLUDE = {
            "get_day_briefing", "gmail_get_unread", "gmail_get_recent", "gmail_search",
            "gmail_check_replied", "gmail_send_email", "gmail_reply_tool",
            "calendar_get_today", "calendar_get_this_week", "calendar_create_event",
            "discord_notify", "discord_list_webhooks",
            "get_weather_tool", "get_location_tool", "get_time_tool",
            "shashin_random_tool", "shashin_search_tool",
            "web_search_tool", "rag_search_tool",
        }
        msg_lower = user_message.lower()
        filtered = [
            t for t in self._tools
            if t in _ALWAYS_INCLUDE or t.lower() in msg_lower
        ]
        # Always have at least the always-include set that's available
        if not filtered:
            filtered = [t for t in self._tools if t in _ALWAYS_INCLUDE]
        tool_list = "\n".join(f"  - {t}" for t in filtered)
        system = _PARSER_SYSTEM.format(tool_list=tool_list)
        try:
            raw = await self._llm_fn(system, user_message)
        except Exception as e:
            logger.error(f"ScheduleParser LLM call failed: {e}")
            return ScheduleClarification(
                question="I had trouble parsing that scheduling request. "
                         "Could you rephrase it? e.g. 'Run the day briefing every day at 7am'."
            )
        return await self._parse_response(raw, user_message)

    async def _parse_response(self, raw: str, original: str) -> ScheduleConfirmation | ScheduleClarification:
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            return ScheduleClarification(
                question="I couldn't parse that as a scheduling request. "
                         "Try: 'Schedule [tool] every day at [time]'."
            )

        status = data.get("status")

        if status == "ready":
            ttype = data.get("trigger_type", "cron")

            # If model returned cron but run_date is set, honour run_date as once
            if ttype == "cron" and data.get("run_date"):
                ttype = "once"
                data["trigger_type"] = "once"

            # Condition job with run_date = one-time check, not recurring poll
            if ttype == "condition" and data.get("run_date"):
                data["trigger_type"] = "once"
                ttype = "once"
                # Recalculate run_date using today's actual date (LLM often gets date wrong)
                _rd = data["run_date"]
                try:
                    from datetime import date as _date_cond
                    _time_part = _rd.split("T")[1][:5] if "T" in _rd else "00:00"
                    _h, _m = map(int, _time_part.split(":"))
                    data["run_date"] = f"{_date_cond.today().isoformat()}T{_h:02d}:{_m:02d}:00"
                    _suffix = "am" if _h < 12 else "pm"
                    _h12 = _h if 1 <= _h <= 12 else (12 if _h == 0 else _h - 12)
                    data["human_schedule"] = f"Once at {_h12}:{_m:02d}{_suffix} today"
                except Exception:
                    pass

            # Condition job with today signal but no run_date — extract from condition_cron
            if ttype == "condition" and not data.get("run_date"):
                import re as _re_cond
                _one_time = _re_cond.search(
                    r'\b(today|tonight|this\s+(morning|afternoon|evening|night)|just\s+once|one\s+time)\b',
                    original, _re_cond.IGNORECASE
                )
                if _one_time and data.get("condition_cron"):
                    _cron_parts = data["condition_cron"].strip().split()
                    if len(_cron_parts) == 5 and _cron_parts[0].isdigit() and _cron_parts[1].isdigit():
                        from datetime import date as _date2
                        _rd = f"{_date2.today().isoformat()}T{int(_cron_parts[1]):02d}:{int(_cron_parts[0]):02d}:00"
                        data["run_date"] = _rd
                        data["trigger_type"] = "once"
                        ttype = "once"
                        _h, _m = int(_cron_parts[1]), int(_cron_parts[0])
                        _suffix = "am" if _h < 12 else "pm"
                        _h12 = _h if 1 <= _h <= 12 else (12 if _h == 0 else _h - 12)
                        data["human_schedule"] = f"Once at {_h12}:{_m:02d}{_suffix} today"

            # Deterministic one-time detection — no LLM call needed
            if ttype == "cron" and data.get("cron"):
                import re as _re_once
                _one_time_signals = _re_once.search(
                    r'\b(today|tonight|this\s+(morning|afternoon|evening|night)|right\s+now|just\s+once|one\s+time|one-time)\b',
                    original, _re_once.IGNORECASE
                )
                _cron_parts = (data.get("cron") or "").strip().split()
                _is_specific_date = (
                    len(_cron_parts) == 5 and
                    _cron_parts[2] != "*" and _cron_parts[3] != "*"
                )
                if _one_time_signals or _is_specific_date:
                    ttype = "once"
                    data["trigger_type"] = "once"
                    _run_date = None
                    if len(_cron_parts) >= 2 and _cron_parts[0].isdigit() and _cron_parts[1].isdigit():
                        from datetime import date as _date
                        _run_date = f"{_date.today().isoformat()}T{int(_cron_parts[1]):02d}:{int(_cron_parts[0]):02d}:00"
                        _h, _m = int(_cron_parts[1]), int(_cron_parts[0])
                        _suffix = "am" if _h < 12 else "pm"
                        _h12 = _h if 1 <= _h <= 12 else (12 if _h == 0 else _h - 12)
                        data["human_schedule"] = f"Once at {_h12}:{_m:02d}{_suffix} today"
                    if _run_date:
                        data["run_date"] = _run_date

            # Validate cron — must be exactly 5 fields
            if ttype == "cron" and data.get("cron"):
                if len(data["cron"].strip().split()) != 5:
                    logger.warning(f"ScheduleParser: malformed cron '{data['cron']}'")
                    return ScheduleClarification(
                        question="I couldn't parse a valid schedule from that. "
                                 "Could you rephrase? e.g. 'Run get_day_briefing at 7:54am today'"
                    )

            required = ["label", "human_schedule"]
            # tool is only required when there's no llm_prompt (compound jobs use llm_prompt alone)
            if not data.get("llm_prompt"):
                required.append("tool")
            if ttype == "cron":
                required.append("cron")
            elif ttype == "once":
                required.append("run_date")
            else:
                required += ["condition_tool", "condition_expr"]
            missing = [k for k in required if not data.get(k)]
            if missing:
                return ScheduleClarification(
                    question=f"I'm missing: {', '.join(missing)}. Could you be more specific?"
                )
            return ScheduleConfirmation(
                label=data["label"],
                tool=data.get("tool") or "",
                tool_args=data.get("tool_args", {}),
                trigger_type=ttype,
                cron=data.get("cron"),
                condition_tool=data.get("condition_tool"),
                condition_tool_args=data.get("condition_tool_args", {}),
                condition_expr=data.get("condition_expr"),
                condition_cron=data.get("condition_cron") or "*/15 * * * *",
                timezone=data.get("timezone", self._tz),
                human_schedule=data["human_schedule"],
                original_request=original,
                llm_prompt=data.get("llm_prompt") or None,
                run_date=data.get("run_date") or None,
                end_date=data.get("end_date") or None,
                deliver_to_session=bool(data.get("deliver_to_session", False)),
            )

        elif status == "clarify":
            return ScheduleClarification(
                question=data.get("question",
                    "Could you give me more details about what to schedule?")
            )

        return ScheduleClarification(
            question="I wasn't sure how to interpret that. "
                     "Try: 'Schedule [tool name] every day at [time]'."
        )


# ── Confirmation tracker ──────────────────────────────────────────────────────

class ConfirmationTracker:
    """Holds a pending ScheduleConfirmation per session while waiting for yes/no."""

    def __init__(self):
        self._pending: dict[str, ScheduleConfirmation] = {}

    def set_pending(self, session_id: str, confirmation: ScheduleConfirmation):
        self._pending[str(session_id)] = confirmation

    def get_pending(self, session_id: str) -> Optional[ScheduleConfirmation]:
        return self._pending.get(str(session_id))

    def clear(self, session_id: str):
        self._pending.pop(str(session_id), None)

    @staticmethod
    def is_confirmation(message: str) -> bool:
        return message.strip().lower() in (
            "yes", "y", "confirm", "ok", "sure", "yep",
            "no", "n", "cancel", "nope", "nah"
        )

    @staticmethod
    def is_yes(message: str) -> bool:
        return message.strip().lower() in ("yes", "y", "confirm", "ok", "sure", "yep")


# ── Scheduling keyword detector ───────────────────────────────────────────────

async def looks_like_scheduling_request(message: str, llm_fn=None) -> bool:
    """
    Use the LLM to determine if a message is a scheduling/automation request.
    Falls back to False if llm_fn is not provided or the call fails.
    """
    import re as _re_sched

    # Direct pipeline dispatch — never a scheduling request
    # (only bypass if there are no time/schedule keywords in the message)
    # Strip quoted parameter values first so words like "Daily" inside
    # query="Daily facts" don't trip the time-keyword check.
    _msg_no_quotes = _re_sched.sub(r'["\'][^"\']*["\']', '', message)
    _has_time = _re_sched.search(
        r'\b(at\s+\d|every\b|daily\b|weekly\b|tomorrow\b|tonight\b|morning\b|minute\b|hours\b|seconds\b|pm\b|am\b|in\s+\d+\s+(minute|hour|day))',
        _msg_no_quotes, _re_sched.IGNORECASE
    )
    # Direct tool call — never a scheduling request unless it has time/schedule keywords
    if message.lstrip().lower().startswith("use ") and not _has_time:
        return False

    # Direct condition check — "check <tool> if <expr> then use <tool>" — never a scheduling request
    if message.lstrip().lower().startswith("check ") and " then " in message.lower() and not _has_time:
        return False

    _time_pattern = _re_sched.search(
        r'\b(at\s+\d{1,2}(:\d{2})?\s*(am|pm)|every\s+(day|morning|night|hour|week|monday|tuesday|wednesday|thursday|friday|saturday|sunday)|daily|weekly|each\s+(day|morning|night)|\d{1,2}(:\d{2})?\s*(am|pm)\s+today|tomorrow\s+at)',
        _msg_no_quotes, _re_sched.IGNORECASE
    )
    if _time_pattern:
        return True

    if not llm_fn:
        return False
    try:
        system = (
            "You are a classifier. Reply with only YES or NO.\n"
            "Does the following message ask to schedule, automate, repeat, "
            "poll, or trigger a recurring task or alert? "
            "Examples that are YES: 'show me weather every morning', "
            "'alert me when I have unread emails', 'run a briefing daily at 7am', "
            "'check my calendar every hour', 'remind me every Monday'. "
            "Examples that are NO: 'what is the weather', 'show my emails', "
            "'what time is it', 'summarize this'."
        )
        response = await llm_fn(system, message)
        return response.strip().upper().startswith("Y")
    except Exception as e:
        logger.warning(f"⏰ Scheduling classifier failed: {e}")
        return False


# ── AgentScheduler ────────────────────────────────────────────────────────────

class AgentScheduler:
    """
    Wraps APScheduler AsyncIOScheduler.

    execute_fn:  async (tool_name: str, args: dict) -> str
    broadcast_fn: async (data: dict) -> None
    llm_fn:      async (prompt: str) -> str   -- optional, used when job has llm_prompt
    """

    def __init__(self, execute_fn, broadcast_fn, llm_fn=None, session_manager=None, agent_fn=None):
        self._execute_fn = execute_fn
        self._broadcast_fn = broadcast_fn
        self._llm_fn = llm_fn
        self._session_manager = session_manager
        self._agent_fn = agent_fn  # full agent run with tool access (run_agent_wrapper)
        self._scheduler = None

    async def start(self):
        try:
            from apscheduler.schedulers.asyncio import AsyncIOScheduler
        except ImportError:
            logger.warning("⚠️ apscheduler not installed — AgentScheduler disabled. "
                           "Run: pip install apscheduler")
            return

        _ensure_db()
        self._scheduler = AsyncIOScheduler()
        self._load_all_jobs()
        self._scheduler.start()
        logger.info("⏰ AgentScheduler started")

    def _load_all_jobs(self):
        for job in list_jobs():
            if job["enabled"]:
                self._register_job(job)

    def _register_job(self, job: dict):
        if self._scheduler is None:
            return
        try:
            from apscheduler.triggers.cron import CronTrigger
            from apscheduler.triggers.date import DateTrigger
        except ImportError:
            return

        ttype = job["trigger_type"]
        fire_fn = self._fire_job if ttype in ("cron", "once") else self._check_condition

        if ttype == "once":
            run_date = job.get("run_date")
            if not run_date:
                logger.warning(f"⏰ Job {job['id']} is 'once' but has no run_date — skipping")
                return
            trigger = DateTrigger(run_date=run_date, timezone=job["timezone"])
        else:
            cron_str = job["cron"] if ttype == "cron" else job["condition_cron"]
            if not cron_str:
                return
            _cron = CronTrigger.from_crontab(cron_str, timezone=job["timezone"])
            _end = job.get("end_date") or None
            if _end:
                _cron.end_date = _end
            trigger = _cron

        self._scheduler.add_job(
            fire_fn,
            trigger=trigger,
            id=str(job["id"]),
            args=[job],
            replace_existing=True,
        )

    async def _fire_job(self, job: dict):
        logger.info(f"⏰ Firing scheduled job: {job['label']} (id={job['id']})")
        record_run(job["id"])
        # Auto-delete one-shot jobs after firing
        if job.get("trigger_type") == "once":
            try:
                delete_job(job["id"])
                logger.info(f"⏰ One-shot job {job['id']} deleted after firing")
            except Exception as _del_err:
                logger.warning(f"⏰ Failed to delete one-shot job {job['id']}: {_del_err}")
        try:
            tool = (job.get("tool") or "").strip()
            llm_prompt = (job.get("llm_prompt") or "").strip()
            condition_result = job.get("_condition_result", "")  # injected by _check_condition

            if not tool and not llm_prompt:
                logger.warning(f"⏰ Job {job['id']} has no tool and no llm_prompt — skipping")
                return

            # ── Pipeline detection ────────────────────────────────────────────
            # If llm_prompt contains pipe-separated `use <tool>` steps, execute
            # them in sequence without any LLM involvement.
            # Format: "use tool_a | use tool_b: arg='...' | use tool_c"
            # Each step's output is available to the next via {previous_result}.
            _PIPE_STEP_RE = re.compile(
                r'use\s+(\w+)(?:\s*:\s*(.*))?', re.IGNORECASE
            )
            _is_pipeline = (
                llm_prompt and
                "|" in llm_prompt and
                all(_PIPE_STEP_RE.match(s.strip()) for s in llm_prompt.split("|") if s.strip())
            )

            tool_result = None

            if _is_pipeline:
                # Execute each step in sequence
                steps = [s.strip() for s in llm_prompt.split("|") if s.strip()]
                previous_result = condition_result or None
                for step_idx, step in enumerate(steps):
                    m = _PIPE_STEP_RE.match(step)
                    if not m:
                        continue
                    step_tool = m.group(1)
                    step_args_str = (m.group(2) or "").strip()

                    # Parse key=value args from the step
                    step_args: dict = {}
                    if step_args_str:
                        try:
                            # Try JSON first
                            step_args = json.loads(step_args_str)
                        except (json.JSONDecodeError, ValueError):
                            # Parse key="value" or key='value' pairs
                            # Use a more robust regex that handles content with apostrophes
                            import re as _re2
                            # Match key="..." (double quotes, handles escaped quotes)
                            for _m in _re2.finditer(r'(\w+)\s*=\s*"((?:[^"\\]|\\.)*)"', step_args_str):
                                step_args[_m.group(1)] = _m.group(2).replace('\\"', '"')
                            # Match key='...' (single quotes) only if no double-quoted matches
                            if not step_args:
                                for _m in _re2.finditer(r"(\w+)\s*=\s*'((?:[^'\\]|\\.)*)'", step_args_str):
                                    step_args[_m.group(1)] = _m.group(2).replace("\\'", "'")

                    # If the step has no args and we have a previous result,
                    # inject it as the message/body for notification tools
                    if previous_result and not step_args:
                        if any(kw in step_tool.lower() for kw in ("notify", "send", "reply", "post")):
                            if "email" in step_tool.lower():
                                # gmail_send_email needs to/subject/body — can't proceed without to
                                logger.warning(f"⏰ Pipeline: {step_tool} requires 'to' and 'subject' args — specify them in the pipeline step")
                                step_args = {"body": str(previous_result)}
                            else:
                                step_args = {"message": str(previous_result)}

                    # If step has some args but missing content field, inject previous result
                    elif previous_result and step_args:
                        _is_email = "email" in step_tool.lower()
                        _content_key = "body" if _is_email else "message"
                        _content_keys = ("message", "body", "content", "text")
                        _has_content = any(k in step_args for k in _content_keys)

                        # Remove wrong content key if present (e.g. "message" on gmail_send_email)
                        if _is_email and "message" in step_args and "body" not in step_args:
                            step_args["body"] = step_args.pop("message")
                            _has_content = True

                        if not _has_content:
                            step_args[_content_key] = str(previous_result)
                        elif _is_email and not step_args.get("body"):
                            step_args["body"] = str(previous_result)

                    logger.info(f"⏰ Pipeline step {step_idx + 1}/{len(steps)}: {step_tool}({step_args})")
                    try:
                        _NOTIF = ("discord_notify", "gmail_send_email", "gmail_reply_tool", "gmail_send")
                        _is_notif = any(n in step_tool for n in _NOTIF)
                        # Use raw executor for data steps (no LLM reformatting)
                        # Use full executor for notification steps (handles HTML conversion etc.)
                        _exec = self._execute_fn if _is_notif else (getattr(self, '_raw_execute_fn', None) or self._execute_fn)
                        previous_result = await _exec(step_tool, step_args)
                        if previous_result and str(previous_result).startswith(f"Tool '{step_tool}' not found"):
                            logger.error(f"⏰ Pipeline: tool '{step_tool}' not found — check tool is registered and server started correctly")
                        logger.info(f"⏰ Pipeline step {step_idx + 1} completed: {str(previous_result)[:100]}")
                    except Exception as _step_err:
                        logger.error(f"⏰ Pipeline step {step_idx + 1} ({step_tool}) failed: {_step_err}")
                        previous_result = f"Error in step {step_idx + 1} ({step_tool}): {_step_err}"
                        break

                    # Extract best text from JSON results before passing to next step
                    if previous_result and str(previous_result).strip().startswith("{"):
                        try:
                            _pd = json.loads(str(previous_result))
                            if isinstance(_pd, dict):
                                _tv = (_pd.get("text") or _pd.get("summary") or "").strip()
                                if _tv:
                                    previous_result = _tv
                        except Exception:
                            pass

                _NOTIF_TOOLS = ("discord_notify", "gmail_send_email", "gmail_reply_tool", "gmail_send")
                _last_step = steps[-1] if steps else ""
                _last_tool_name = _last_step.split()[1].split(":")[0] if len(_last_step.split()) > 1 else ""
                if any(_nt in _last_tool_name for _nt in _NOTIF_TOOLS):
                    result = "Job completed."
                else:
                    result = previous_result or "Job completed."

            else:
                # Original single-tool or LLM-based execution

                # For once-trigger jobs that have a condition_tool, run the check first
                if tool and job.get("condition_tool") and not condition_result:
                    _cond_tool = job["condition_tool"]
                    _cond_args = json.loads(job.get("condition_tool_args") or "{}")
                    _cond_expr = job.get("condition_expr") or ""
                    logger.info(f"⏰ Running condition check: {_cond_tool}({_cond_args}) → {_cond_expr!r}")
                    try:
                        # Use raw execute (bypasses LLM formatting) to get parseable JSON
                        _exec = getattr(self, '_raw_execute_fn', None) or self._execute_fn
                        _cond_raw = await _exec(_cond_tool, _cond_args)
                        _cond_vars = {"result": _cond_raw, "result_len": len(str(_cond_raw)), "data": {}}
                        try:
                            import json as _cj
                            _cond_data = _cj.loads(_cond_raw) if isinstance(_cond_raw, str) else {}
                            _cond_vars["data"] = _cond_data
                            if isinstance(_cond_data, dict):
                                for _k, _v in _cond_data.items():
                                    if isinstance(_v, (int, float, bool, str)):
                                        _cond_vars[_k] = _v
                                    elif isinstance(_v, list):
                                        _cond_vars[f"len_{_k}"] = len(_v)
                            elif isinstance(_cond_data, list):
                                _cond_vars["result"] = len(_cond_data)
                            logger.info(f"⏰ Condition vars keys: {list(_cond_vars.keys())}")
                        except Exception as _parse_err:
                            logger.warning(f"⏰ Condition JSON parse failed: {_parse_err} — raw: {str(_cond_raw)[:100]}")
                        _cond_fired = bool(eval(_cond_expr, {"__builtins__": {}}, _cond_vars)) if _cond_expr else True
                    except Exception as _ce:
                        logger.warning(f"⏰ Condition check failed: {_ce}")
                        _cond_fired = False

                    if not _cond_fired:
                        logger.info(f"⏰ Condition FALSE — skipping action tool")
                        result = "Condition not met — no action taken."
                        # Skip to delivery
                        if job.get("deliver_to_session") and job.get("session_id") and self._session_manager:
                            try:
                                import os as _os2
                                _max2 = int(_os2.getenv("MAX_MESSAGE_HISTORY", 30))
                                self._session_manager.add_message(job["session_id"], "assistant", result, _max2, "proactive")
                            except Exception:
                                pass
                        await self._broadcast_fn({"type": "scheduled_result", "job_id": job["id"], "label": job["label"], "result": result, "session_id": job.get("session_id")})
                        return

                    condition_result = str(_cond_raw)

                if tool:
                    args = json.loads(job["tool_args"] or "{}")
                    # Inject condition result as message if no args specified
                    if condition_result and not args:
                        if any(k in tool for k in ("notify", "send", "email", "reply", "post")):
                            # Extract human-readable text from JSON if available
                            _msg = condition_result
                            try:
                                _cd = json.loads(condition_result)
                                if isinstance(_cd, dict):
                                    _msg = _cd.get("text") or _cd.get("summary") or _cd.get("message") or condition_result
                            except Exception:
                                pass
                            args = {"message": _msg}
                    tool_result = await self._execute_fn(tool, args)
                    result = tool_result

                if llm_prompt and not tool:
                    # Pure LLM job (no tool) — llm_prompt is the entire instruction
                    context_parts = []
                    if condition_result:
                        context_parts.append(f"Condition check result:\n{condition_result}")
                    context = "\n\n".join(context_parts)
                    full_prompt = f"{llm_prompt}\n\n{context}".strip() if context else llm_prompt

                    if self._agent_fn:
                        # Full agent run — LLM has tool access for compound actions
                        result = await self._agent_fn(full_prompt)
                    elif self._llm_fn:
                        # Fallback: bare LLM (no tool access)
                        result = await self._llm_fn(full_prompt)
                    else:
                        result = tool_result or "No result."
                elif not tool:
                    result = tool_result or "No result."

            # Deliver to specific session or broadcast to all clients
            if job.get("deliver_to_session") and job.get("session_id") and self._session_manager:
                try:
                    import os as _os
                    _max = int(_os.getenv("MAX_MESSAGE_HISTORY", 30))
                    self._session_manager.add_message(
                        job["session_id"], "assistant", result, _max, "proactive"
                    )
                    logger.info(f"⏰ Delivered job {job['id']} result to session {job['session_id']}")
                except Exception as _sm_err:
                    logger.warning(f"⏰ session_manager delivery failed: {_sm_err} — falling back to broadcast")
                # Also broadcast so the UI updates regardless of which session is open
                await self._broadcast_fn({
                    "type": "scheduled_result",
                    "job_id": job["id"],
                    "label": job["label"],
                    "result": result,
                    "session_id": job.get("session_id"),
                })
            else:
                await self._broadcast_fn({
                    "type": "scheduled_result",
                    "job_id": job["id"],
                    "label": job["label"],
                    "result": result,
                })
        except Exception as e:
            logger.error(f"⏰ Scheduled job {job['id']} failed: {e}")
            await self._broadcast_fn({
                "type": "scheduled_error",
                "job_id": job["id"],
                "label": job["label"],
                "error": str(e),
            })

    async def _check_condition(self, job: dict):
        """Poll the condition tool; fire the action tool only if condition is true."""
        record_run(job["id"], is_check=True)
        try:
            condition_args = json.loads(job.get("condition_tool_args") or "{}")
            # Use raw execute (bypasses LLM formatting) to get parseable JSON
            _exec = getattr(self, '_raw_execute_fn', None) or self._execute_fn
            result_raw = await _exec(job["condition_tool"], condition_args)

            # Build evaluation context — try to give the expression as much
            # to work with as possible regardless of what the tool returned.
            eval_ctx: dict = {"__builtins__": {}}

            # Always expose the raw string
            eval_ctx["result"] = result_raw

            # Try numeric parse
            try:
                eval_ctx["result"] = float(str(result_raw).strip())
            except (ValueError, TypeError):
                pass

            # Try JSON parse — expose parsed dict/list AND common convenience keys
            try:
                parsed = json.loads(result_raw)
                eval_ctx["data"] = parsed
                if isinstance(parsed, dict):
                    # Flat convenience keys: total_unread, count, total_results, etc.
                    for k, v in parsed.items():
                        if isinstance(v, (int, float, bool, str)):
                            eval_ctx[k] = v
                    # len(emails), len(results) etc.
                    for k, v in parsed.items():
                        if isinstance(v, list):
                            eval_ctx[f"len_{k}"] = len(v)
                elif isinstance(parsed, list):
                    eval_ctx["result"] = len(parsed)
                    eval_ctx["data"] = parsed
            except (json.JSONDecodeError, TypeError):
                pass

            # Also expose len(result) for raw string checks
            eval_ctx["result_len"] = len(str(result_raw))

            _expr = job["condition_expr"]
            # Reject expressions containing function calls, dunder access, or imports
            if any(tok in _expr for tok in ("(", ")", "__", "import", ";", "`")):
                raise ValueError(f"Unsafe condition expression rejected: {_expr!r}")
            triggered = bool(eval(_expr, eval_ctx))
            logger.debug(
                f"⏰ Condition '{job['condition_expr']}' evaluated "
                f"{'True' if triggered else 'False'} for job {job['id']}"
            )
        except Exception as e:
            logger.warning(f"⏰ Condition check failed for job {job['id']}: {e}")
            return

        if triggered:
            logger.info(f"⏰ Condition met for job '{job['label']}' — firing action")
            # Inject condition result as context for llm_prompt
            job_with_context = dict(job)
            job_with_context["_condition_result"] = result_raw
            await self._fire_job(job_with_context)

    def add_job(self, job_id: int):
        """Hot-register a newly created job without restart."""
        job = get_job(job_id)
        if job and job["enabled"]:
            self._register_job(job)

    def remove_job(self, job_id: int):
        if self._scheduler:
            try:
                self._scheduler.remove_job(str(job_id))
            except Exception:
                pass

    async def stop(self):
        if self._scheduler:
            self._scheduler.shutdown(wait=False)