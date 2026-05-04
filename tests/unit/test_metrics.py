"""
Tests for client/metrics.py
Covers: trace IDs, FailureKind, JsonFormatter, StructuredLogger,
        _percentiles, _histogram, _windowed, reset_metrics, prepare_metrics
"""
import json
import logging
import time
import pytest
from collections import defaultdict


# ═══════════════════════════════════════════════════════════════════
# Trace IDs
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestTraceIds:
    def test_new_trace_returns_uuid_string(self):
        from client.metrics import new_trace
        tid = new_trace()
        assert isinstance(tid, str)
        assert len(tid) == 36  # UUID format
        assert tid.count("-") == 4

    def test_get_trace_returns_current(self):
        from client.metrics import new_trace, get_trace
        tid = new_trace()
        assert get_trace() == tid

    def test_get_trace_default_is_empty(self):
        from client.metrics import get_trace, _trace_id_var
        # Reset to default
        token = _trace_id_var.set("")
        assert get_trace() == ""
        _trace_id_var.reset(token)

    def test_new_trace_each_call_unique(self):
        from client.metrics import new_trace
        ids = {new_trace() for _ in range(10)}
        assert len(ids) == 10


# ═══════════════════════════════════════════════════════════════════
# FailureKind & MCPToolError
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestFailureKind:
    def test_all_failure_kinds_exist(self):
        from client.metrics import FailureKind
        assert FailureKind.RETRYABLE.value == "retryable"
        assert FailureKind.USER_ERROR.value == "user_error"
        assert FailureKind.UPSTREAM_ERROR.value == "upstream_error"
        assert FailureKind.INTERNAL_ERROR.value == "internal_error"

    def test_mcp_tool_error_carries_kind(self):
        from client.metrics import MCPToolError, FailureKind
        err = MCPToolError(FailureKind.RETRYABLE, "timeout", {"url": "http://x"})
        assert err.kind == FailureKind.RETRYABLE
        assert err.message == "timeout"
        assert err.detail == {"url": "http://x"}

    def test_mcp_tool_error_default_detail(self):
        from client.metrics import MCPToolError, FailureKind
        err = MCPToolError(FailureKind.INTERNAL_ERROR, "oops")
        assert err.detail == {}

    def test_mcp_tool_error_is_exception(self):
        from client.metrics import MCPToolError, FailureKind
        with pytest.raises(MCPToolError):
            raise MCPToolError(FailureKind.USER_ERROR, "bad input")


# ═══════════════════════════════════════════════════════════════════
# JsonFormatter
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestJsonFormatter:
    def _make_record(self, msg="test message", level=logging.INFO, name="test_logger"):
        record = logging.LogRecord(
            name=name, level=level, pathname="", lineno=0,
            msg=msg, args=(), exc_info=None
        )
        return record

    def test_output_is_valid_json(self):
        from client.metrics import JsonFormatter
        formatter = JsonFormatter()
        record = self._make_record()
        output = formatter.format(record)
        parsed = json.loads(output)
        assert isinstance(parsed, dict)

    def test_output_has_required_fields(self):
        from client.metrics import JsonFormatter
        formatter = JsonFormatter()
        record = self._make_record("hello world")
        parsed = json.loads(formatter.format(record))
        assert "ts" in parsed
        assert "level" in parsed
        assert "logger" in parsed
        assert "trace_id" in parsed
        assert "msg" in parsed

    def test_message_content_preserved(self):
        from client.metrics import JsonFormatter
        formatter = JsonFormatter()
        record = self._make_record("my log message")
        parsed = json.loads(formatter.format(record))
        assert parsed["msg"] == "my log message"

    def test_level_name_correct(self):
        from client.metrics import JsonFormatter
        formatter = JsonFormatter()
        record = self._make_record(level=logging.WARNING)
        parsed = json.loads(formatter.format(record))
        assert parsed["level"] == "WARNING"

    def test_trace_id_included(self):
        from client.metrics import JsonFormatter, new_trace
        formatter = JsonFormatter()
        tid = new_trace()
        record = self._make_record()
        parsed = json.loads(formatter.format(record))
        assert parsed["trace_id"] == tid

    def test_exc_info_included(self):
        from client.metrics import JsonFormatter
        formatter = JsonFormatter()
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            record = logging.LogRecord(
                name="test", level=logging.ERROR, pathname="", lineno=0,
                msg="error occurred", args=(), exc_info=sys.exc_info()
            )
        parsed = json.loads(formatter.format(record))
        assert "exc" in parsed
        assert "ValueError" in parsed["exc"]


# ═══════════════════════════════════════════════════════════════════
# StructuredLogger
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestStructuredLogger:
    def test_log_outputs_valid_json(self, capsys):
        from client.metrics import StructuredLogger
        logger = StructuredLogger("test_server")
        logger.log("test_event", level="INFO", key="value")
        captured = capsys.readouterr()
        parsed = json.loads(captured.out.strip())
        assert parsed["event"] == "test_event"
        assert parsed["server"] == "test_server"
        assert parsed["level"] == "INFO"
        assert parsed["key"] == "value"

    def test_info_convenience(self, capsys):
        from client.metrics import StructuredLogger
        logger = StructuredLogger("srv")
        logger.info("info_event")
        captured = capsys.readouterr()
        parsed = json.loads(captured.out.strip())
        assert parsed["level"] == "INFO"

    def test_warn_convenience(self, capsys):
        from client.metrics import StructuredLogger
        logger = StructuredLogger("srv")
        logger.warn("warn_event")
        captured = capsys.readouterr()
        parsed = json.loads(captured.out.strip())
        assert parsed["level"] == "WARN"

    def test_error_convenience(self, capsys):
        from client.metrics import StructuredLogger
        logger = StructuredLogger("srv")
        logger.error("error_event")
        captured = capsys.readouterr()
        parsed = json.loads(captured.out.strip())
        assert parsed["level"] == "ERROR"


# ═══════════════════════════════════════════════════════════════════
# Latency helpers
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestPercentiles:
    def test_empty_returns_zeros(self):
        from client.metrics import _percentiles
        result = _percentiles([])
        assert result == {"p50": 0, "p95": 0, "p99": 0}

    def test_single_value(self):
        from client.metrics import _percentiles
        result = _percentiles([1.0])
        assert result["p50"] == 1.0
        assert result["p95"] == 1.0
        assert result["p99"] == 1.0

    def test_percentiles_ordered(self):
        from client.metrics import _percentiles
        durations = [float(i) for i in range(1, 101)]
        result = _percentiles(durations)
        assert result["p50"] <= result["p95"] <= result["p99"]

    def test_p50_is_median(self):
        from client.metrics import _percentiles
        result = _percentiles([1.0, 2.0, 3.0, 4.0, 5.0])
        assert result["p50"] == 3.0

    def test_returns_rounded_values(self):
        from client.metrics import _percentiles
        result = _percentiles([0.123456789])
        assert result["p50"] == round(0.123456789, 2)


@pytest.mark.unit
class TestHistogram:
    def test_empty_returns_zero_counts(self):
        from client.metrics import _histogram
        result = _histogram([])
        assert all(v == 0 for v in result.values())

    def test_all_buckets_present(self):
        from client.metrics import _histogram, _BUCKETS
        result = _histogram([])
        expected_labels = {label for label, _ in _BUCKETS}
        assert set(result.keys()) == expected_labels

    def test_fast_request_in_first_bucket(self):
        from client.metrics import _histogram
        result = _histogram([0.05])  # 50ms
        assert result["<100ms"] == 1
        assert result["100-500ms"] == 0

    def test_slow_request_in_last_bucket(self):
        from client.metrics import _histogram
        result = _histogram([10.0])  # 10 seconds
        assert result[">5s"] == 1

    def test_multiple_buckets(self):
        from client.metrics import _histogram
        result = _histogram([0.05, 0.3, 0.8, 3.0, 10.0])
        assert result["<100ms"] == 1
        assert result["100-500ms"] == 1
        assert result["500ms-1s"] == 1
        assert result["1-5s"] == 1
        assert result[">5s"] == 1


@pytest.mark.unit
class TestWindowed:
    def test_returns_durations_only(self):
        from client.metrics import _windowed
        time_list = [(1000.0, 0.5), (1001.0, 1.2), (1002.0, 0.3)]
        result = _windowed(time_list)
        assert result == [0.5, 1.2, 0.3]

    def test_respects_window_size(self):
        from client.metrics import _windowed
        time_list = [(float(i), float(i)) for i in range(100)]
        result = _windowed(time_list, window=10)
        assert len(result) == 10
        assert result[-1] == 99.0  # last item

    def test_empty_returns_empty(self):
        from client.metrics import _windowed
        assert _windowed([]) == []


# ═══════════════════════════════════════════════════════════════════
# reset_metrics
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestResetMetrics:
    def test_reset_clears_counters(self):
        from client.metrics import metrics, reset_metrics
        metrics["agent_runs"] = 42
        metrics["llm_calls"] = 10
        metrics["agent_times"].append((time.time(), 1.5))
        reset_metrics()
        assert metrics["agent_runs"] == 0
        assert metrics["llm_calls"] == 0
        assert metrics["agent_times"] == []

    def test_reset_clears_tool_data(self):
        from client.metrics import metrics, reset_metrics
        metrics["tool_calls"]["my_tool"] = 5
        metrics["tool_errors"]["my_tool"] = 1
        reset_metrics()
        assert len(metrics["tool_calls"]) == 0
        assert len(metrics["tool_errors"]) == 0

    def test_reset_preserves_dict_reference(self):
        from client.metrics import metrics, reset_metrics
        original_ref = metrics
        reset_metrics()
        assert metrics is original_ref  # same object, not replaced


# ═══════════════════════════════════════════════════════════════════
# prepare_metrics
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestPrepareMetrics:
    def setup_method(self):
        from client.metrics import reset_metrics
        reset_metrics()

    def test_returns_expected_top_level_keys(self):
        from client.metrics import prepare_metrics
        result = prepare_metrics()
        assert "agent" in result
        assert "llm" in result
        assert "tools" in result
        assert "overall_errors" in result
        assert "failure_kinds" in result
        assert "window_size" in result
        assert "buckets" in result

    def test_empty_metrics_returns_zeros(self):
        from client.metrics import prepare_metrics
        result = prepare_metrics()
        assert result["agent"]["runs"] == 0
        assert result["agent"]["errors"] == 0
        assert result["agent"]["avg_time"] == 0
        assert result["llm"]["calls"] == 0
        assert result["overall_errors"] == 0

    def test_agent_runs_counted(self):
        from client.metrics import metrics, prepare_metrics
        metrics["agent_runs"] = 5
        metrics["agent_errors"] = 1
        metrics["agent_times"] = [(time.time(), 1.0)] * 5
        result = prepare_metrics()
        assert result["agent"]["runs"] == 5
        assert result["agent"]["errors"] == 1
        assert result["agent"]["avg_time"] == 1.0
        assert result["agent"]["error_rate"] == 20.0

    def test_tool_stats_included(self):
        from client.metrics import metrics, prepare_metrics
        metrics["tool_calls"]["search_tool"] = 3
        metrics["tool_errors"]["search_tool"] = 1
        metrics["tool_times"]["search_tool"] = [(time.time(), 0.5)] * 3
        result = prepare_metrics()
        per_tool = result["tools"]["per_tool"]
        assert "search_tool" in per_tool
        assert per_tool["search_tool"]["calls"] == 3
        assert per_tool["search_tool"]["errors"] == 1

    def test_histogram_in_agent_stats(self):
        from client.metrics import metrics, prepare_metrics, _BUCKETS
        metrics["agent_times"] = [(time.time(), 0.05)]
        result = prepare_metrics()
        hist = result["agent"]["histogram"]
        assert set(hist.keys()) == {label for label, _ in _BUCKETS}
        assert hist["<100ms"] == 1

    def test_percentiles_in_output(self):
        from client.metrics import metrics, prepare_metrics
        metrics["llm_times"] = [(time.time(), float(i)) for i in range(1, 6)]
        result = prepare_metrics()
        assert "p50" in result["llm"]
        assert "p95" in result["llm"]
        assert "p99" in result["llm"]

    def test_failure_kinds_included(self):
        from client.metrics import metrics, prepare_metrics
        metrics["failure_kinds"]["retryable"] = 3
        result = prepare_metrics()
        assert result["failure_kinds"]["retryable"] == 3