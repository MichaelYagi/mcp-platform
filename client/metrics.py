"""
Metrics Module
Tracks performance metrics for MCP components with timestamps
"""

import contextvars
import json
import logging
import time
import uuid
from collections import defaultdict
from enum import Enum

# ─── Correlation IDs ──────────────────────────────────────────────────────────
# One ContextVar per async task — safe across concurrent sessions.

_trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="")


def new_trace() -> str:
    """Generate a new trace ID and bind it to the current async context."""
    tid = str(uuid.uuid4())
    _trace_id_var.set(tid)
    return tid


def get_trace() -> str:
    """Return the trace ID for the current async context, or '' if unset."""
    return _trace_id_var.get()


# ─── Failure Taxonomy ─────────────────────────────────────────────────────────

class FailureKind(Enum):
    RETRYABLE      = "retryable"       # network blip, timeout, rate limit
    USER_ERROR     = "user_error"      # bad params, missing required field
    UPSTREAM_ERROR = "upstream_error"  # Plex/Ollama/external service down
    INTERNAL_ERROR = "internal_error"  # unhandled exception, logic bug


class MCPToolError(Exception):
    """Structured tool error carrying a FailureKind for retry/routing decisions."""
    def __init__(self, kind: FailureKind, message: str, detail: dict = None):
        self.kind = kind
        self.message = message
        self.detail = detail or {}
        super().__init__(message)


# ─── JSON Formatter ───────────────────────────────────────────────────────────

class JsonFormatter(logging.Formatter):
    """
    Replaces the plain-text log formatter across all components.

    Every record emits a JSON object on a single line:
        {"ts": 1234567890.123, "level": "INFO", "logger": "mcp_client",
         "trace_id": "abc-...", "msg": "Tool completed", "exc": null}

    trace_id is pulled from the ContextVar automatically — no call-site changes
    needed. Falls back to "" when no trace is active (e.g. startup messages).

    Install once per handler:
        handler.setFormatter(JsonFormatter())
    """

    def format(self, record: logging.LogRecord) -> str:
        exc_text = None
        if record.exc_info:
            exc_text = self.formatException(record.exc_info)
        elif record.exc_text:
            exc_text = record.exc_text

        entry = {
            "ts":       round(record.created, 3),
            "level":    record.levelname,
            "logger":   record.name,
            "trace_id": get_trace(),
            "msg":      record.getMessage(),
        }
        if exc_text:
            entry["exc"] = exc_text

        # Carry any extra fields attached via logger.info(..., extra={...})
        _SKIP = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)
        for k, v in record.__dict__.items():
            if k not in _SKIP and not k.startswith("_"):
                entry[k] = v

        try:
            return json.dumps(entry, default=str)
        except Exception:
            # Last-resort fallback — never let the formatter crash the process
            return json.dumps({"ts": entry["ts"], "level": "ERROR",
                               "logger": record.name, "trace_id": entry["trace_id"],
                               "msg": f"[JsonFormatter error] {record.getMessage()}"})


# ─── Structured Logger ────────────────────────────────────────────────────────

class StructuredLogger:
    """
    Emits JSON log records to stdout.
    Each record includes the current trace_id automatically.
    Pair with a log aggregator (Loki, jq, sqlite sink) as needed.
    """
    def __init__(self, server_name: str):
        self._server = server_name

    def log(self, event: str, level: str = "INFO", **kwargs):
        record = {
            "ts":       round(time.time(), 3),
            "level":    level,
            "server":   self._server,
            "trace_id": get_trace(),
            "event":    event,
            **kwargs,
        }
        print(json.dumps(record, default=str), flush=True)

    # Convenience wrappers
    def info(self,  event: str, **kw): self.log(event, "INFO",  **kw)
    def warn(self,  event: str, **kw): self.log(event, "WARN",  **kw)
    def error(self, event: str, **kw): self.log(event, "ERROR", **kw)


# ─── Existing metrics dict (unchanged) ───────────────────────────────────────

metrics = {
    "agent_runs": 0,
    "agent_errors": 0,
    "agent_times": [],  # list of (timestamp, duration) tuples
    "llm_calls": 0,
    "llm_errors": 0,
    "llm_times": [],  # list of (timestamp, duration) tuples
    "tool_calls": defaultdict(int),  # tool_name: count
    "tool_errors": defaultdict(int),  # tool_name: count
    "tool_times": defaultdict(list),  # tool_name: [(timestamp, duration), ...]
    # failure_kinds: FailureKind.value -> count  (populated by langgraph)
    "failure_kinds": defaultdict(int),
}

# ─── Latency histogram configuration ─────────────────────────────────────────

# Number of most-recent samples used for percentile/histogram computation.
# Older samples are retained in the raw lists for time-series graphs but
# excluded from p50/p95/p99 so stale spikes don't pollute current stats.
METRICS_WINDOW = 1000

# Histogram bucket boundaries in seconds.
# Label → upper bound (None = open-ended catch-all)
_BUCKETS = [
    ("<100ms",   0.1),
    ("100-500ms", 0.5),
    ("500ms-1s",  1.0),
    ("1-5s",      5.0),
    (">5s",       None),
]


# ─── Latency helpers ──────────────────────────────────────────────────────────

def _percentiles(durations: list[float]) -> dict:
    """
    Compute p50, p95, p99 from a list of duration values (seconds).
    Returns all three rounded to 2 decimal places, or 0 if empty.
    """
    if not durations:
        return {"p50": 0, "p95": 0, "p99": 0}
    s = sorted(durations)
    n = len(s)

    def _pct(p: float) -> float:
        idx = (p / 100) * (n - 1)
        lo, hi = int(idx), min(int(idx) + 1, n - 1)
        frac = idx - lo
        return round(s[lo] * (1 - frac) + s[hi] * frac, 2)

    return {"p50": _pct(50), "p95": _pct(95), "p99": _pct(99)}


def _histogram(durations: list[float]) -> dict:
    """
    Bucket durations into the _BUCKETS bands.
    Returns {label: count} for all bands (zero-filled so the frontend
    always receives a consistent schema).
    """
    counts = {label: 0 for label, _ in _BUCKETS}
    for d in durations:
        for label, upper in _BUCKETS:
            if upper is None or d < upper:
                counts[label] += 1
                break
    return counts


def _windowed(time_list: list, window: int = METRICS_WINDOW) -> list[float]:
    """Extract the last `window` durations from a [(ts, duration), ...] list."""
    return [dur for _, dur in time_list[-window:]]


# ─── Reset ────────────────────────────────────────────────────────────────────

def reset_metrics() -> None:
    """
    Zero all counters and clear all time series.
    Called by the reset_metrics WebSocket handler.
    Preserves the dict structure — never replaces the object itself
    so existing references in langgraph.py stay valid.
    """
    metrics["agent_runs"]   = 0
    metrics["agent_errors"] = 0
    metrics["agent_times"]  = []
    metrics["llm_calls"]    = 0
    metrics["llm_errors"]   = 0
    metrics["llm_times"]    = []
    metrics["tool_calls"].clear()
    metrics["tool_errors"].clear()
    metrics["tool_times"].clear()
    metrics["failure_kinds"].clear()


# ─── prepare_metrics ──────────────────────────────────────────────────────────

def prepare_metrics():
    """
    Prepare metrics data for broadcasting.

    For each component (agent, LLM, per-tool) returns:
      - counts and error rates (unchanged)
      - avg_time          — mean over all samples
      - p50 / p95 / p99   — percentiles over last METRICS_WINDOW samples
      - histogram         — bucket counts over last METRICS_WINDOW samples
      - times             — last 100 (timestamp, duration) pairs for sparklines
    """
    tool_total_calls  = sum(metrics["tool_calls"].values())
    tool_total_errors = sum(metrics["tool_errors"].values())
    total_errors      = metrics["agent_errors"] + metrics["llm_errors"] + tool_total_errors
    agent_error_rate  = (
        metrics["agent_errors"] / metrics["agent_runs"] * 100
        if metrics["agent_runs"] > 0 else 0
    )

    # ── Raw duration lists ────────────────────────────────────────────────────
    agent_all  = [d for _, d in metrics["agent_times"]]
    llm_all    = [d for _, d in metrics["llm_times"]]

    # ── Windowed duration lists (for percentiles / histograms) ────────────────
    agent_win  = _windowed(metrics["agent_times"])
    llm_win    = _windowed(metrics["llm_times"])

    # ── Averages (all-time) ───────────────────────────────────────────────────
    agent_avg  = round(sum(agent_all) / len(agent_all), 2) if agent_all else 0
    llm_avg    = round(sum(llm_all)   / len(llm_all),   2) if llm_all   else 0

    # ── Time series for sparklines (last 100) ─────────────────────────────────
    def format_time_series(time_list):
        """Convert [(timestamp, duration), ...] to {timestamps: [...], durations: [...]}"""
        if not time_list:
            return {"timestamps": [], "durations": []}
        timestamps, durations = zip(*time_list)
        return {"timestamps": list(timestamps), "durations": list(durations)}

    recent_agent = metrics["agent_times"][-100:]
    recent_llm   = metrics["llm_times"][-100:]

    # ── Per-tool stats ────────────────────────────────────────────────────────
    per_tool = {}
    for name in metrics["tool_calls"]:
        tl       = metrics["tool_times"].get(name, [])
        all_dur  = [d for _, d in tl]
        win_dur  = _windowed(tl)
        avg      = round(sum(all_dur) / len(all_dur), 2) if all_dur else 0
        per_tool[name] = {
            "calls":     metrics["tool_calls"][name],
            "errors":    metrics["tool_errors"][name],
            "avg_time":  avg,
            **_percentiles(win_dur),          # p50, p95, p99
            "histogram": _histogram(win_dur),
            "times":     format_time_series(tl[-100:]),
        }

    return {
        "agent": {
            "runs":       metrics["agent_runs"],
            "errors":     metrics["agent_errors"],
            "error_rate": round(agent_error_rate, 2),
            "avg_time":   agent_avg,
            **_percentiles(agent_win),
            "histogram":  _histogram(agent_win),
            "times":      format_time_series(recent_agent),
        },
        "llm": {
            "calls":     metrics["llm_calls"],
            "errors":    metrics["llm_errors"],
            "avg_time":  llm_avg,
            **_percentiles(llm_win),
            "histogram": _histogram(llm_win),
            "times":     format_time_series(recent_llm),
        },
        "tools": {
            "total_calls":  tool_total_calls,
            "total_errors": tool_total_errors,
            "per_tool":     per_tool,
        },
        "overall_errors": total_errors,
        "failure_kinds":  dict(metrics["failure_kinds"]),
        "window_size":    METRICS_WINDOW,
        "buckets":        [label for label, _ in _BUCKETS],
    }