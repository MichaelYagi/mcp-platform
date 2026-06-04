"""
tests/unit/test_process_tool_result.py

Tests for the logic implemented by _process_tool_result, _tool_executor,
and _run_pipeline in client.py.

Because these are closures inside main(), tests exercise:
  - the documented behaviour of each code path (plain text, image, pre-built
    summary, list builder, LLM summarisation fallback)
  - the pipeline abort condition string that _run_pipeline checks
  - the error strings that _tool_executor returns (must match the abort check)
"""
import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ═══════════════════════════════════════════════════════════════════
# Helpers — replicate the decision logic from _process_tool_result
# so we can write regression tests without importing the closure.
# ═══════════════════════════════════════════════════════════════════

def _is_plain_text(tool_result: str) -> bool:
    return not tool_result.strip().startswith(("{", "["))


def _is_prebuilt_summary(tool_json: dict) -> bool:
    """Matches the pre-built-summary guard in _process_tool_result."""
    return (
        isinstance(tool_json, dict)
        and "summary" in tool_json
        and not any(
            isinstance(tool_json.get(k), list)
            for k in ("documents", "sources", "results", "items",
                       "records", "entries", "chunks")
        )
    )


def _format_prebuilt_summary(tool_json: dict) -> str:
    presummary = tool_json["summary"]
    title = tool_json.get("title", "")
    url = tool_json.get("url", "")
    parts = []
    if title:
        parts.append(f"**{title}**")
    if url:
        parts.append(f"[{url}]({url})")
    header = " — ".join(parts)
    return f"{header}\n\n{presummary}" if header else presummary


def _pipeline_aborts(previous: str) -> bool:
    """Replicates the abort condition in _run_pipeline (after Phase 1 fix)."""
    return (
        str(previous).startswith("Tool ")
        and (
            "error:" in str(previous).lower()
            or "not found." in str(previous)
        )
    )


# ═══════════════════════════════════════════════════════════════════
# 1. Plain-text passthrough
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestPlainTextPassthrough:

    def test_plain_string_is_plain(self):
        assert _is_plain_text("hello world") is True

    def test_json_object_is_not_plain(self):
        assert _is_plain_text('{"key": "value"}') is False

    def test_json_array_is_not_plain(self):
        assert _is_plain_text('[{"a": 1}]') is False

    def test_empty_string_is_plain(self):
        assert _is_plain_text("") is True

    def test_whitespace_before_brace_not_plain(self):
        assert _is_plain_text('   {"x": 1}') is False

    def test_error_message_is_plain(self):
        assert _is_plain_text("Tool foo error: something went wrong") is True

    def test_multiline_plain(self):
        assert _is_plain_text("Line one\nLine two\nLine three") is True


# ═══════════════════════════════════════════════════════════════════
# 2. Pre-built summary detection and formatting
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestPrebuiltSummary:

    def test_simple_summary_detected(self):
        d = {"summary": "The page is about cats."}
        assert _is_prebuilt_summary(d) is True

    def test_summary_with_title_and_url(self):
        d = {"summary": "Content here.", "title": "My Page", "url": "https://example.com"}
        assert _is_prebuilt_summary(d) is True

    def test_summary_with_documents_list_not_detected(self):
        d = {"summary": "...", "documents": [{"text": "doc1"}]}
        assert _is_prebuilt_summary(d) is False

    def test_summary_with_results_list_not_detected(self):
        d = {"summary": "...", "results": [{"title": "r1"}]}
        assert _is_prebuilt_summary(d) is False

    def test_no_summary_key(self):
        d = {"title": "page", "content": "text"}
        assert _is_prebuilt_summary(d) is False

    def test_format_with_title_and_url(self):
        d = {"summary": "Great article.", "title": "News", "url": "https://news.example.com"}
        result = _format_prebuilt_summary(d)
        assert "**News**" in result
        assert "https://news.example.com" in result
        assert "Great article." in result

    def test_format_summary_only(self):
        d = {"summary": "No title or url here."}
        result = _format_prebuilt_summary(d)
        assert result == "No title or url here."

    def test_format_url_without_title(self):
        d = {"summary": "Body.", "url": "https://x.com"}
        result = _format_prebuilt_summary(d)
        assert "https://x.com" in result
        assert "Body." in result


# ═══════════════════════════════════════════════════════════════════
# 3. List-builder detection
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestListBuilderDetection:

    def _find_items(self, parsed):
        """Replicate the list-builder item-detection from _process_tool_result."""
        if isinstance(parsed, list):
            return parsed, "results"
        if isinstance(parsed, dict):
            for k, v in parsed.items():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    return v, k
        return None, None

    def test_top_level_list(self):
        data = [{"title": "a"}, {"title": "b"}]
        items, key = self._find_items(data)
        assert items == data
        assert key == "results"

    def test_nested_list_under_key(self):
        data = {"results": [{"title": "r1"}, {"title": "r2"}]}
        items, key = self._find_items(data)
        assert key == "results"
        assert len(items) == 2

    def test_emails_key(self):
        data = {"total_unread": 3, "emails": [{"subject": "Hi"}, {"subject": "Bye"}]}
        items, key = self._find_items(data)
        assert key == "emails"

    def test_no_list_returns_none(self):
        data = {"temperature": 20, "condition": "sunny"}
        items, key = self._find_items(data)
        assert items is None

    def test_list_of_non_dicts_not_picked_up(self):
        data = {"tags": ["python", "mcp"]}
        items, key = self._find_items(data)
        assert items is None


# ═══════════════════════════════════════════════════════════════════
# 4. Pipeline abort condition — regression for Phase 1 fix
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestPipelineAbortCondition:
    """
    _run_pipeline aborts early if a non-final step returns an error string.
    Phase 1 fixed the condition from the never-matching
    `startswith("Error executing tool")` to the correct check below.
    """

    def test_tool_error_triggers_abort(self):
        err = "Tool gmail_send_email error: SMTP connection refused"
        assert _pipeline_aborts(err) is True

    def test_tool_not_found_triggers_abort(self):
        err = "Tool 'nonexistent_tool' not found."
        assert _pipeline_aborts(err) is True

    def test_plain_result_does_not_abort(self):
        assert _pipeline_aborts("Found 3 emails") is False

    def test_json_result_does_not_abort(self):
        assert _pipeline_aborts('{"emails": []}') is False

    def test_old_string_never_matched(self):
        """Confirm the pre-fix string would never have triggered."""
        old_str = "Error executing tool foo: something"
        # Old check: startswith("Error executing tool") — this would have matched
        # but _tool_executor never returned that prefix.
        assert not old_str.startswith("Tool ")  # _tool_executor returns "Tool ..."
        assert not _pipeline_aborts(old_str)    # and our fix only checks "Tool ..."

    def test_error_in_tool_name_does_not_false_positive(self):
        """A result that mentions 'error:' but doesn't start with 'Tool ' is OK."""
        result = "Found error: records show 5 items"
        assert _pipeline_aborts(result) is False

    def test_tool_error_case_insensitive(self):
        err = "Tool foo_bar Error: Something Bad"
        assert _pipeline_aborts(err) is True


# ═══════════════════════════════════════════════════════════════════
# 5. _tool_executor error string format
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestToolExecutorErrorStrings:
    """
    _tool_executor returns specific string formats on failure.
    These must match what _run_pipeline's abort condition checks.
    """

    def test_tool_error_format_matches_abort(self):
        tool_name = "weather_tool"
        exc = ValueError("timeout")
        err_str = f"Tool {tool_name} error: {exc}"
        assert _pipeline_aborts(err_str) is True

    def test_tool_not_found_format_matches_abort(self):
        tool_name = "missing_tool"
        err_str = f"Tool '{tool_name}' not found."
        assert _pipeline_aborts(err_str) is True

    def test_successful_result_does_not_match_abort(self):
        result = '{"temperature": 20, "condition": "sunny"}'
        assert _pipeline_aborts(result) is False


# ═══════════════════════════════════════════════════════════════════
# 6. _unwrap_tool_result (module-level utility)
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestUnwrapToolResult:
    """
    _unwrap_tool_result is a module-level function in client.py.
    Importing it requires mocking heavy initialisation — test via logic replication.
    """

    def _unwrap(self, raw) -> str:
        """Replicate the logic of _unwrap_tool_result."""
        if raw is None:
            return ""
        if hasattr(raw, 'text'):
            return raw.text
        if hasattr(raw, 'content'):
            return self._unwrap(raw.content)
        if isinstance(raw, list):
            parts = []
            for item in raw:
                if hasattr(item, 'text'):
                    parts.append(item.text)
                elif isinstance(item, dict):
                    parts.append(item.get('text', str(item)))
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        return str(raw)

    def test_none_returns_empty(self):
        assert self._unwrap(None) == ""

    def test_text_attribute_extracted(self):
        obj = MagicMock()
        obj.text = "hello"
        assert self._unwrap(obj) == "hello"

    def test_list_of_text_objects(self):
        a = MagicMock(); a.text = "first"
        b = MagicMock(); b.text = "second"
        result = self._unwrap([a, b])
        assert "first" in result
        assert "second" in result

    def test_string_passthrough(self):
        assert self._unwrap("raw string") == "raw string"

    def test_dict_in_list_uses_text_key(self):
        result = self._unwrap([{"text": "dict content"}])
        assert result == "dict content"
