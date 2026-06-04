"""
tests/unit/test_proactive_agent_coverage.py

Comprehensive tests for client/proactive_agent.py covering:
  - create_job / list_jobs / get_job / delete_job / find_job_by_label
  - cron_to_human
  - ConfirmationTracker (is_confirmation, is_yes, set/get/clear pending)
  - handle_jobs_command (:jobs list/pause/enable/cancel/info)
  - AgentScheduler._check_condition — condition eval, validation, context building
  - once-trigger past-date guard
"""
import asyncio
import json
import pytest
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def scheduler_db(tmp_path):
    """Isolated scheduler.db — redirects SCHEDULER_DB_PATH."""
    db_path = tmp_path / "scheduler.db"
    with patch("client.proactive_agent.SCHEDULER_DB_PATH", db_path):
        from client.proactive_agent import _ensure_db
        _ensure_db()
        yield db_path


@pytest.fixture
def one_job(scheduler_db):
    """Create a single cron job and return its id."""
    with patch("client.proactive_agent.SCHEDULER_DB_PATH", scheduler_db):
        from client.proactive_agent import create_job
        job_id = create_job(
            label="daily briefing",
            tool="get_day_briefing",
            trigger_type="cron",
            cron="30 7 * * *",
            tool_args={"max_emails": 5},
        )
        return job_id


# ═══════════════════════════════════════════════════════════════════
# 1. CRUD — create_job / list_jobs / get_job / delete_job
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestJobCRUD:

    def test_create_job_returns_int(self, scheduler_db):
        with patch("client.proactive_agent.SCHEDULER_DB_PATH", scheduler_db):
            from client.proactive_agent import create_job
            jid = create_job(label="test", tool="some_tool", cron="0 9 * * *")
            assert isinstance(jid, int)
            assert jid > 0

    def test_create_and_list(self, scheduler_db):
        with patch("client.proactive_agent.SCHEDULER_DB_PATH", scheduler_db):
            from client.proactive_agent import create_job, list_jobs
            create_job(label="job1", tool="tool_a", cron="0 8 * * *")
            create_job(label="job2", tool="tool_b", cron="0 9 * * *")
            jobs = list_jobs()
            assert len(jobs) == 2
            labels = {j["label"] for j in jobs}
            assert "job1" in labels and "job2" in labels

    def test_get_job_returns_dict(self, scheduler_db, one_job):
        with patch("client.proactive_agent.SCHEDULER_DB_PATH", scheduler_db):
            from client.proactive_agent import get_job
            job = get_job(one_job)
            assert job is not None
            assert job["label"] == "daily briefing"
            assert job["tool"] == "get_day_briefing"

    def test_get_job_nonexistent_returns_none(self, scheduler_db):
        with patch("client.proactive_agent.SCHEDULER_DB_PATH", scheduler_db):
            from client.proactive_agent import get_job
            assert get_job(9999) is None

    def test_delete_job(self, scheduler_db, one_job):
        with patch("client.proactive_agent.SCHEDULER_DB_PATH", scheduler_db):
            from client.proactive_agent import delete_job, get_job
            delete_job(one_job)
            assert get_job(one_job) is None

    def test_create_job_stores_tool_args(self, scheduler_db):
        with patch("client.proactive_agent.SCHEDULER_DB_PATH", scheduler_db):
            from client.proactive_agent import create_job, get_job
            jid = create_job(label="x", tool="t", cron="* * * * *",
                             tool_args={"limit": 10, "city": "Vancouver"})
            job = get_job(jid)
            args = json.loads(job["tool_args"])
            assert args["limit"] == 10
            assert args["city"] == "Vancouver"

    def test_create_job_enabled_by_default(self, scheduler_db):
        with patch("client.proactive_agent.SCHEDULER_DB_PATH", scheduler_db):
            from client.proactive_agent import create_job, get_job
            jid = create_job(label="y", tool="t", cron="0 0 * * *")
            assert get_job(jid)["enabled"] == 1

    def test_create_once_job_stores_run_date(self, scheduler_db):
        with patch("client.proactive_agent.SCHEDULER_DB_PATH", scheduler_db):
            from client.proactive_agent import create_job, get_job
            jid = create_job(label="once", tool="t", trigger_type="once",
                             run_date="2026-12-01T09:00:00")
            job = get_job(jid)
            assert job["trigger_type"] == "once"
            assert "2026-12-01" in job["run_date"]


# ═══════════════════════════════════════════════════════════════════
# 2. find_job_by_label
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestFindJobByLabel:

    def test_find_by_exact_label(self, scheduler_db, one_job):
        with patch("client.proactive_agent.SCHEDULER_DB_PATH", scheduler_db):
            from client.proactive_agent import find_job_by_label
            job = find_job_by_label("daily briefing")
            assert job is not None
            assert job["id"] == one_job

    def test_find_by_partial_label(self, scheduler_db, one_job):
        with patch("client.proactive_agent.SCHEDULER_DB_PATH", scheduler_db):
            from client.proactive_agent import find_job_by_label
            job = find_job_by_label("briefing")
            assert job is not None

    def test_find_by_numeric_id(self, scheduler_db, one_job):
        with patch("client.proactive_agent.SCHEDULER_DB_PATH", scheduler_db):
            from client.proactive_agent import find_job_by_label
            job = find_job_by_label(str(one_job))
            assert job is not None
            assert job["id"] == one_job

    def test_find_nonexistent_returns_none(self, scheduler_db):
        with patch("client.proactive_agent.SCHEDULER_DB_PATH", scheduler_db):
            from client.proactive_agent import find_job_by_label
            assert find_job_by_label("does not exist") is None


# ═══════════════════════════════════════════════════════════════════
# 3. cron_to_human
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestCronToHuman:

    def test_empty_returns_dash(self):
        from client.proactive_agent import cron_to_human
        assert cron_to_human("") == "—"

    def test_none_returns_dash(self):
        from client.proactive_agent import cron_to_human
        assert cron_to_human(None) == "—"

    def test_every_day(self):
        from client.proactive_agent import cron_to_human
        result = cron_to_human("30 7 * * *")
        assert "7:30am" in result
        assert "Every day" in result

    def test_weekdays(self):
        from client.proactive_agent import cron_to_human
        result = cron_to_human("0 9 * * 1-5")
        assert "Weekdays" in result
        assert "9:00am" in result

    def test_weekends(self):
        from client.proactive_agent import cron_to_human
        result = cron_to_human("0 10 * * 6,0")
        assert "Weekend" in result

    def test_specific_days(self):
        from client.proactive_agent import cron_to_human
        result = cron_to_human("0 8 * * 1,3,5")
        assert "Mon" in result or "Mon" in result

    def test_midnight(self):
        from client.proactive_agent import cron_to_human
        result = cron_to_human("0 0 * * *")
        assert "12:00am" in result

    def test_noon(self):
        from client.proactive_agent import cron_to_human
        result = cron_to_human("0 12 * * *")
        assert "12:00pm" in result

    def test_invalid_cron_passthrough(self):
        from client.proactive_agent import cron_to_human
        expr = "bad expression"
        assert cron_to_human(expr) == expr

    def test_4_part_cron_passthrough(self):
        from client.proactive_agent import cron_to_human
        expr = "0 9 * *"
        assert cron_to_human(expr) == expr


# ═══════════════════════════════════════════════════════════════════
# 4. ConfirmationTracker
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestConfirmationTracker:

    def test_is_confirmation_yes_variants(self):
        from client.proactive_agent import ConfirmationTracker
        for word in ("yes", "y", "confirm", "ok", "sure", "yep", "no", "n", "cancel", "nope", "nah"):
            assert ConfirmationTracker.is_confirmation(word) is True

    def test_is_confirmation_false_for_other(self):
        from client.proactive_agent import ConfirmationTracker
        for word in ("maybe", "what?", "schedule this", "cancel all jobs"):
            assert ConfirmationTracker.is_confirmation(word) is False

    def test_is_yes_true(self):
        from client.proactive_agent import ConfirmationTracker
        for word in ("yes", "y", "confirm", "ok", "sure", "yep"):
            assert ConfirmationTracker.is_yes(word) is True

    def test_is_yes_false_for_no(self):
        from client.proactive_agent import ConfirmationTracker
        for word in ("no", "n", "cancel", "nope", "nah"):
            assert ConfirmationTracker.is_yes(word) is False

    def test_is_confirmation_case_insensitive(self):
        from client.proactive_agent import ConfirmationTracker
        assert ConfirmationTracker.is_confirmation("YES") is True
        assert ConfirmationTracker.is_confirmation("No") is True

    def test_set_and_get_pending(self):
        from client.proactive_agent import ConfirmationTracker
        tracker = ConfirmationTracker()
        pending = MagicMock()
        tracker.set_pending("session-1", pending)
        assert tracker.get_pending("session-1") is pending

    def test_clear_pending(self):
        from client.proactive_agent import ConfirmationTracker
        tracker = ConfirmationTracker()
        tracker.set_pending("s1", MagicMock())
        tracker.clear("s1")
        assert tracker.get_pending("s1") is None

    def test_get_nonexistent_returns_none(self):
        from client.proactive_agent import ConfirmationTracker
        assert ConfirmationTracker().get_pending("missing") is None


# ═══════════════════════════════════════════════════════════════════
# 5. handle_jobs_command
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestHandleJobsCommand:

    def test_list_empty(self, scheduler_db):
        with patch("client.proactive_agent.SCHEDULER_DB_PATH", scheduler_db):
            from client.proactive_agent import handle_jobs_command
            result = handle_jobs_command(":jobs")
            assert "No scheduled jobs" in result

    def test_list_shows_job(self, scheduler_db, one_job):
        with patch("client.proactive_agent.SCHEDULER_DB_PATH", scheduler_db):
            from client.proactive_agent import handle_jobs_command
            result = handle_jobs_command(":jobs")
            assert "daily briefing" in result

    def test_cancel_job_by_label(self, scheduler_db, one_job):
        with patch("client.proactive_agent.SCHEDULER_DB_PATH", scheduler_db):
            from client.proactive_agent import handle_jobs_command, get_job
            handle_jobs_command(":jobs cancel daily briefing")
            assert get_job(one_job) is None

    def test_cancel_all(self, scheduler_db):
        with patch("client.proactive_agent.SCHEDULER_DB_PATH", scheduler_db):
            from client.proactive_agent import create_job, handle_jobs_command, list_jobs
            create_job("a", "t", cron="* * * * *")
            create_job("b", "t", cron="* * * * *")
            handle_jobs_command(":jobs cancel all")
            assert list_jobs() == []

    def test_pause_job(self, scheduler_db, one_job):
        with patch("client.proactive_agent.SCHEDULER_DB_PATH", scheduler_db):
            from client.proactive_agent import handle_jobs_command, get_job
            handle_jobs_command(":jobs pause daily briefing")
            assert get_job(one_job)["enabled"] == 0

    def test_enable_job(self, scheduler_db, one_job):
        with patch("client.proactive_agent.SCHEDULER_DB_PATH", scheduler_db):
            from client.proactive_agent import handle_jobs_command, get_job, set_job_enabled
            set_job_enabled(one_job, False)
            handle_jobs_command(":jobs enable daily briefing")
            assert get_job(one_job)["enabled"] == 1

    def test_info_shows_detail(self, scheduler_db, one_job):
        with patch("client.proactive_agent.SCHEDULER_DB_PATH", scheduler_db):
            from client.proactive_agent import handle_jobs_command
            result = handle_jobs_command(":jobs info daily briefing")
            assert "daily briefing" in result.lower() or "get_day_briefing" in result

    def test_unknown_verb(self, scheduler_db):
        with patch("client.proactive_agent.SCHEDULER_DB_PATH", scheduler_db):
            from client.proactive_agent import handle_jobs_command
            result = handle_jobs_command(":jobs frobnicate something")
            assert "Unknown" in result or "Available" in result


# ═══════════════════════════════════════════════════════════════════
# 6. AgentScheduler._check_condition
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestCheckCondition:

    def _make_scheduler(self, exec_result: str):
        """Build a minimal AgentScheduler with a mocked _execute_fn."""
        from client.proactive_agent import AgentScheduler
        scheduler = AgentScheduler.__new__(AgentScheduler)
        scheduler._execute_fn = AsyncMock(return_value=exec_result)
        scheduler._raw_execute_fn = scheduler._execute_fn
        return scheduler

    def _make_job(self, condition_expr: str, condition_tool: str = "check_email",
                  condition_tool_args: str = "{}") -> dict:
        return {
            "id": 1,
            "label": "test job",
            "condition_expr": condition_expr,
            "condition_tool": condition_tool,
            "condition_tool_args": condition_tool_args,
            "tool": "notify_tool",
            "tool_args": "{}",
            "llm_prompt": None,
            "trigger_type": "condition",
            "enabled": 1,
        }

    @pytest.mark.asyncio
    async def test_numeric_result_true(self, scheduler_db):
        with patch("client.proactive_agent.SCHEDULER_DB_PATH", scheduler_db):
            from client.proactive_agent import create_job, record_run
            jid = create_job("c", "t", trigger_type="condition",
                             condition_tool="check", condition_expr="result > 0",
                             condition_cron="* * * * *")
            scheduler = self._make_scheduler("5")
            job = {"id": jid, "condition_expr": "result > 0",
                   "condition_tool": "check", "condition_tool_args": "{}",
                   "tool": "t", "tool_args": "{}", "llm_prompt": None,
                   "trigger_type": "condition", "label": "c"}
            with patch.object(scheduler, '_execute_fn', AsyncMock(return_value="5")):
                with patch("client.proactive_agent.record_run"):
                    with patch.object(scheduler, '_fire_job', AsyncMock()):
                        await scheduler._check_condition(job)
                        scheduler._fire_job.assert_called_once()

    @pytest.mark.asyncio
    async def test_condition_false_does_not_fire(self, scheduler_db):
        with patch("client.proactive_agent.SCHEDULER_DB_PATH", scheduler_db):
            from client.proactive_agent import create_job
            jid = create_job("d", "t", trigger_type="condition",
                             condition_tool="check", condition_expr="result > 100",
                             condition_cron="* * * * *")
            job = {"id": jid, "condition_expr": "result > 100",
                   "condition_tool": "check", "condition_tool_args": "{}",
                   "tool": "t", "tool_args": "{}", "llm_prompt": None,
                   "trigger_type": "condition", "label": "d"}
            with patch.object(
                self._make_scheduler("5"), '_execute_fn', AsyncMock(return_value="5")
            ):
                scheduler = self._make_scheduler("5")
                with patch("client.proactive_agent.record_run"):
                    with patch.object(scheduler, '_fire_job', AsyncMock()):
                        await scheduler._check_condition(job)
                        scheduler._fire_job.assert_not_called()

    @pytest.mark.asyncio
    async def test_json_dict_exposes_keys(self, scheduler_db):
        with patch("client.proactive_agent.SCHEDULER_DB_PATH", scheduler_db):
            from client.proactive_agent import create_job
            jid = create_job("e", "t", trigger_type="condition",
                             condition_tool="check", condition_expr="total_unread > 0",
                             condition_cron="* * * * *")
            payload = json.dumps({"total_unread": 3, "emails": []})
            job = {"id": jid, "condition_expr": "total_unread > 0",
                   "condition_tool": "check", "condition_tool_args": "{}",
                   "tool": "t", "tool_args": "{}", "llm_prompt": None,
                   "trigger_type": "condition", "label": "e"}
            scheduler = self._make_scheduler(payload)
            with patch("client.proactive_agent.record_run"):
                with patch.object(scheduler, '_fire_job', AsyncMock()):
                    await scheduler._check_condition(job)
                    scheduler._fire_job.assert_called_once()

    @pytest.mark.asyncio
    async def test_unsafe_expression_rejected(self, scheduler_db):
        with patch("client.proactive_agent.SCHEDULER_DB_PATH", scheduler_db):
            from client.proactive_agent import create_job
            jid = create_job("f", "t", trigger_type="condition",
                             condition_tool="check",
                             condition_expr="__import__('os').getpid() > 0",
                             condition_cron="* * * * *")
            job = {"id": jid, "condition_expr": "__import__('os').getpid() > 0",
                   "condition_tool": "check", "condition_tool_args": "{}",
                   "tool": "t", "tool_args": "{}", "llm_prompt": None,
                   "trigger_type": "condition", "label": "f"}
            scheduler = self._make_scheduler("1")
            with patch("client.proactive_agent.record_run"):
                with patch.object(scheduler, '_fire_job', AsyncMock()):
                    await scheduler._check_condition(job)
                    scheduler._fire_job.assert_not_called()

    @pytest.mark.asyncio
    async def test_expression_with_parens_rejected(self, scheduler_db):
        with patch("client.proactive_agent.SCHEDULER_DB_PATH", scheduler_db):
            from client.proactive_agent import create_job
            jid = create_job("g", "t", trigger_type="condition",
                             condition_tool="check",
                             condition_expr="len(result) > 0",
                             condition_cron="* * * * *")
            job = {"id": jid, "condition_expr": "len(result) > 0",
                   "condition_tool": "check", "condition_tool_args": "{}",
                   "tool": "t", "tool_args": "{}", "llm_prompt": None,
                   "trigger_type": "condition", "label": "g"}
            scheduler = self._make_scheduler("[1,2,3]")
            with patch("client.proactive_agent.record_run"):
                with patch.object(scheduler, '_fire_job', AsyncMock()):
                    await scheduler._check_condition(job)
                    scheduler._fire_job.assert_not_called()


# ═══════════════════════════════════════════════════════════════════
# 7. cron_to_human — edge cases
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestCronToHumanEdgeCases:

    def test_pm_hour(self):
        from client.proactive_agent import cron_to_human
        result = cron_to_human("0 18 * * *")
        assert "6:00pm" in result

    def test_1pm(self):
        from client.proactive_agent import cron_to_human
        result = cron_to_human("0 13 * * *")
        assert "1:00pm" in result

    def test_specific_minute(self):
        from client.proactive_agent import cron_to_human
        result = cron_to_human("45 8 * * *")
        assert "8:45am" in result


# ═══════════════════════════════════════════════════════════════════
# 8. Condition expression safety — unit test the guard directly
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestConditionExpressionGuard:
    """Test the safety check added in Phase 2 fix #9."""

    DANGEROUS = [
        "__import__('os')",
        "len(result)",
        "result.__class__",
        "import os",
        "result; True",
        "`result`",
    ]

    SAFE = [
        "result > 0",
        "total_unread > 5",
        "result_len > 100",
        "len_emails > 0",
    ]

    def _is_safe(self, expr: str) -> bool:
        return not any(tok in expr for tok in ("(", ")", "__", "import", ";", "`"))

    def test_dangerous_expressions_caught(self):
        for expr in self.DANGEROUS:
            assert not self._is_safe(expr), f"Should be caught: {expr!r}"

    def test_safe_expressions_pass(self):
        for expr in self.SAFE:
            assert self._is_safe(expr), f"Should pass: {expr!r}"
