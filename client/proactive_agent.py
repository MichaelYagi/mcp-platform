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
    condition_expr  TEXT,                              -- e.g. "result > 10"
    condition_cron  TEXT    NOT NULL DEFAULT '*/15 * * * *',  -- how often to poll
    timezone        TEXT    NOT NULL DEFAULT 'America/Vancouver',
    enabled         INTEGER NOT NULL DEFAULT 1,
    created_at      TEXT    NOT NULL,
    last_run        TEXT,
    last_check      TEXT,
    run_count       INTEGER NOT NULL DEFAULT 0
);
"""

# ── DB helpers ────────────────────────────────────────────────────────────────

def _ensure_db():
    SCHEDULER_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(SCHEDULER_DB_PATH) as conn:
        conn.executescript(_SCHEMA)


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
    condition_expr: str = None,
    condition_cron: str = "*/15 * * * *",
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
                condition_tool, condition_expr, condition_cron,
                timezone, enabled, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,1,?)""",
            (
                label, trigger_type, tool,
                json.dumps(tool_args or {}),
                cron, condition_tool, condition_expr, condition_cron,
                tz_val, now
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
        if ttype == "cron":
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
    if job["trigger_type"] == "cron":
        sched_lines = (
            f"Cron:      {job['cron']}\n"
            f"Schedule:  {cron_to_human(job['cron'])}\n"
        )
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

    def render(self) -> str:
        args_str = json.dumps(self.tool_args) if self.tool_args else "none"
        if self.trigger_type == "cron":
            sched = (
                f"  Schedule:  {self.human_schedule}\n"
                f"  Cron:      {self.cron}\n"
            )
        else:
            sched = (
                f"  Condition: {self.condition_tool} → {self.condition_expr}\n"
                f"  Polls:     {cron_to_human(self.condition_cron)}\n"
            )
        return (
            f"Here's what I'll schedule — confirm to proceed:\n\n"
            f"  Label:     {self.label}\n"
            f"  Tool:      {self.tool}\n"
            f"  Args:      {args_str}\n"
            + sched +
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

Shape A — all fields resolved (cron trigger):
{{
  "status": "ready",
  "trigger_type": "cron",
  "label": "<3-5 word label>",
  "tool": "<exact tool name>",
  "tool_args": {{}},
  "cron": "<5-field cron>",
  "timezone": "<IANA timezone, default America/Vancouver>",
  "human_schedule": "<plain English>"
}}

Shape B — all fields resolved (condition trigger):
{{
  "status": "ready",
  "trigger_type": "condition",
  "label": "<3-5 word label>",
  "tool": "<action tool to run when condition is true>",
  "tool_args": {{}},
  "condition_tool": "<tool to call for the check>",
  "condition_expr": "<expression e.g. result > 10>",
  "condition_cron": "<how often to poll, default */15 * * * *>",
  "timezone": "America/Vancouver",
  "human_schedule": "<plain English description>"
}}

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
5. Return ONLY JSON. No preamble, no markdown fences."""


class ScheduleParser:
    def __init__(self, llm_fn, available_tools: list[str],
                 default_timezone: str = "America/Vancouver"):
        self._llm_fn = llm_fn
        self._tools = available_tools
        self._tz = default_timezone

    async def parse(self, user_message: str) -> ScheduleConfirmation | ScheduleClarification:
        tool_list = "\n".join(f"  - {t}" for t in self._tools)
        system = _PARSER_SYSTEM.format(tool_list=tool_list)
        try:
            raw = await self._llm_fn(system, user_message)
        except Exception as e:
            logger.error(f"ScheduleParser LLM call failed: {e}")
            return ScheduleClarification(
                question="I had trouble parsing that scheduling request. "
                         "Could you rephrase it? e.g. 'Run the day briefing every day at 7am'."
            )
        return self._parse_response(raw, user_message)

    def _parse_response(self, raw: str, original: str) -> ScheduleConfirmation | ScheduleClarification:
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
            required = ["label", "tool", "human_schedule"]
            if ttype == "cron":
                required.append("cron")
            else:
                required += ["condition_tool", "condition_expr"]
            missing = [k for k in required if not data.get(k)]
            if missing:
                return ScheduleClarification(
                    question=f"I'm missing: {', '.join(missing)}. Could you be more specific?"
                )
            return ScheduleConfirmation(
                label=data["label"],
                tool=data["tool"],
                tool_args=data.get("tool_args", {}),
                trigger_type=ttype,
                cron=data.get("cron"),
                condition_tool=data.get("condition_tool"),
                condition_expr=data.get("condition_expr"),
                condition_cron=data.get("condition_cron", "*/15 * * * *"),
                timezone=data.get("timezone", self._tz),
                human_schedule=data["human_schedule"],
                original_request=original,
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

_SCHEDULE_KEYWORDS = [
    "every day", "every morning", "every night", "every week", "every hour",
    "every monday", "every tuesday", "every wednesday", "every thursday",
    "every friday", "every saturday", "every sunday",
    "weekdays", "weekends", "daily", "weekly", "hourly",
    "each morning", "each day", "each week",
    "schedule", "remind me", "run at", "do a briefing",
    "at 5am", "at 6am", "at 7am", "at 8am", "at 9am", "at 10am",
    "at 11am", "at 12pm", "at noon", "at 1pm", "at 2pm", "at 3pm",
    "at 4pm", "at 5pm", "at 6pm", "at 7pm", "at 8pm", "at 9pm",
    "alert me when", "notify me when", "tell me when", "check if",
    "watch for", "when unread", "if unread",
]


def looks_like_scheduling_request(message: str) -> bool:
    lower = message.lower()
    return any(kw in lower for kw in _SCHEDULE_KEYWORDS)


# ── AgentScheduler ────────────────────────────────────────────────────────────

class AgentScheduler:
    """
    Wraps APScheduler AsyncIOScheduler.

    execute_fn:  async (tool_name: str, args: dict) -> str
    broadcast_fn: async (data: dict) -> None
    """

    def __init__(self, execute_fn, broadcast_fn):
        self._execute_fn = execute_fn
        self._broadcast_fn = broadcast_fn
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
        except ImportError:
            return

        cron_str = job["cron"] if job["trigger_type"] == "cron" else job["condition_cron"]
        if not cron_str:
            return

        trigger = CronTrigger.from_crontab(cron_str, timezone=job["timezone"])
        fire_fn = self._fire_job if job["trigger_type"] == "cron" else self._check_condition

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
        try:
            args = json.loads(job["tool_args"] or "{}")
            result = await self._execute_fn(job["tool"], args)
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
            result_raw = await self._execute_fn(job["condition_tool"], {})
            # Evaluate expression: result is accessible as 'result'
            result = result_raw  # noqa — used in eval below
            try:
                # Parse numeric result if possible
                result_num = float(str(result_raw).strip())
                result = result_num
            except (ValueError, TypeError):
                pass

            triggered = bool(eval(job["condition_expr"], {"result": result, "__builtins__": {}}))
        except Exception as e:
            logger.warning(f"⏰ Condition check failed for job {job['id']}: {e}")
            return

        if triggered:
            logger.info(f"⏰ Condition met for job '{job['label']}' — firing action")
            await self._fire_job(job)

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