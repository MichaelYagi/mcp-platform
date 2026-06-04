"""
tests/unit/test_coverage_boost3.py

Targeted tests to push total coverage above 40%.
Covers: client/vision._dedup_sentences, client/message_router enums/dataclass/router,
        and additional proactive_agent paths.
"""
import asyncio
import time
import pytest
from unittest.mock import MagicMock


# ═══════════════════════════════════════════════════════════════════
# client/vision — _dedup_sentences (pure function, no HTTP)
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestDedupSentences:

    def _dedup(self, text):
        from client.vision import _dedup_sentences
        return _dedup_sentences(text)

    def test_empty_string_unchanged(self):
        assert self._dedup("") == ""

    def test_none_unchanged(self):
        assert self._dedup(None) is None

    def test_no_repetition_unchanged(self):
        text = "The cat sat. The dog barked. The bird sang."
        assert self._dedup(text) == text

    def test_triple_repetition_truncated(self):
        sent = "The cat sat on the mat"
        text = f"{sent}. {sent}. {sent}. More content."
        result = self._dedup(text)
        # Should cut before the third repetition
        parts = result.split(sent)
        assert len(parts) <= 3

    def test_result_ends_with_period(self):
        sent = "Hello world"
        text = f"{sent}. {sent}. {sent}. Trailing."
        result = self._dedup(text)
        assert result.endswith(".")

    def test_single_sentence_unchanged(self):
        text = "Just one sentence."
        assert self._dedup(text) == text

    def test_two_repetitions_allowed(self):
        sent = "The sky is blue"
        text = f"{sent}. {sent}. New content here."
        # Two reps don't trigger cut (needs 3)
        result = self._dedup(text)
        assert sent in result

    def test_whitespace_trimmed_for_key(self):
        # Sentences with leading/trailing spaces — key is stripped
        text = "  Hello  .   Hello  .   Hello  . End."
        # Should detect repetition (keys are stripped)
        result = self._dedup(text)
        assert result  # Just ensure it doesn't crash


# ═══════════════════════════════════════════════════════════════════
# client/message_router — enums, dataclass, basic router
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestMessagePriorityEnum:

    def test_priority_values(self):
        from client.message_router import MessagePriority
        assert MessagePriority.CRITICAL.value == 0
        assert MessagePriority.HIGH.value == 1
        assert MessagePriority.NORMAL.value == 2
        assert MessagePriority.LOW.value == 3
        assert MessagePriority.BULK.value == 4

    def test_all_priorities_defined(self):
        from client.message_router import MessagePriority
        names = {p.name for p in MessagePriority}
        assert {"CRITICAL", "HIGH", "NORMAL", "LOW", "BULK"} == names


@pytest.mark.unit
class TestRoutingStrategyEnum:

    def test_strategy_values(self):
        from client.message_router import RoutingStrategy
        assert RoutingStrategy.DIRECT.value == "direct"
        assert RoutingStrategy.BROADCAST.value == "broadcast"
        assert RoutingStrategy.ROUND_ROBIN.value == "round_robin"
        assert RoutingStrategy.LOAD_BALANCED.value == "balanced"
        assert RoutingStrategy.SKILL_BASED.value == "skill_based"


@pytest.mark.unit
class TestMessageEnvelope:

    def _make_envelope(self, priority=None, strategy=None, ts=None):
        from client.message_router import MessageEnvelope, MessagePriority, RoutingStrategy
        return MessageEnvelope(
            message_id="msg-1",
            from_agent="agent-a",
            to_agent="agent-b",
            content="hello",
            priority=priority or MessagePriority.NORMAL,
            routing_strategy=strategy or RoutingStrategy.DIRECT,
            timestamp=ts or time.time(),
        )

    def test_envelope_creation(self):
        env = self._make_envelope()
        assert env.message_id == "msg-1"
        assert env.from_agent == "agent-a"
        assert env.retry_count == 0
        assert env.max_retries == 3

    def test_envelope_lt_by_priority(self):
        from client.message_router import MessagePriority
        high = self._make_envelope(priority=MessagePriority.HIGH, ts=2.0)
        normal = self._make_envelope(priority=MessagePriority.NORMAL, ts=1.0)
        # HIGH (1) < NORMAL (2) in priority queue ordering
        assert high < normal

    def test_envelope_lt_by_timestamp_same_priority(self):
        from client.message_router import MessagePriority
        older = self._make_envelope(priority=MessagePriority.NORMAL, ts=1.0)
        newer = self._make_envelope(priority=MessagePriority.NORMAL, ts=2.0)
        assert older < newer

    def test_envelope_default_metadata(self):
        env = self._make_envelope()
        assert env.metadata == {}

    def test_envelope_timeout_default(self):
        env = self._make_envelope()
        assert env.timeout == 60.0


@pytest.mark.unit
class TestMessageRouter:

    def _make_router(self):
        from client.message_router import MessageRouter
        logger = MagicMock()
        return MessageRouter(logger), logger

    def test_init_empty_registry(self):
        router, _ = self._make_router()
        assert router.agent_registry == {}

    def test_register_agent(self):
        router, _ = self._make_router()
        agent = MagicMock()
        router.register_agent("agent-1", agent)
        assert "agent-1" in router.agent_registry

    def test_unregister_agent(self):
        router, _ = self._make_router()
        agent = MagicMock()
        router.register_agent("agent-1", agent)
        router.unregister_agent("agent-1")
        assert "agent-1" not in router.agent_registry

    def test_unregister_nonexistent_is_noop(self):
        router, _ = self._make_router()
        router.unregister_agent("does-not-exist")  # Should not raise

    def test_initial_routing_stats(self):
        router, _ = self._make_router()
        assert router.routing_stats["total_routed"] == 0
        assert router.routing_stats["failed_routes"] == 0

    @pytest.mark.asyncio
    async def test_route_direct_no_to_agent(self):
        from client.message_router import MessageEnvelope, MessagePriority, RoutingStrategy
        router, _ = self._make_router()
        env = MessageEnvelope(
            message_id="m1", from_agent="a", to_agent=None,
            content="x", priority=MessagePriority.NORMAL,
            routing_strategy=RoutingStrategy.DIRECT,
            timestamp=time.time()
        )
        result = await router.route_message(env)
        assert result is False

    @pytest.mark.asyncio
    async def test_route_direct_unregistered_agent(self):
        from client.message_router import MessageEnvelope, MessagePriority, RoutingStrategy
        router, _ = self._make_router()
        env = MessageEnvelope(
            message_id="m2", from_agent="a", to_agent="missing-agent",
            content="x", priority=MessagePriority.NORMAL,
            routing_strategy=RoutingStrategy.DIRECT,
            timestamp=time.time()
        )
        result = await router.route_message(env)
        assert result is False
        assert router.routing_stats["total_routed"] == 1

    @pytest.mark.asyncio
    async def test_route_direct_success(self):
        from client.message_router import MessageEnvelope, MessagePriority, RoutingStrategy
        router, _ = self._make_router()
        router.register_agent("agent-b", MagicMock())
        env = MessageEnvelope(
            message_id="m3", from_agent="a", to_agent="agent-b",
            content="hello", priority=MessagePriority.HIGH,
            routing_strategy=RoutingStrategy.DIRECT,
            timestamp=time.time()
        )
        result = await router.route_message(env)
        assert result is True
        assert "m3" in router.pending_messages

    @pytest.mark.asyncio
    async def test_route_broadcast_empty_registry(self):
        from client.message_router import MessageEnvelope, MessagePriority, RoutingStrategy
        router, _ = self._make_router()
        env = MessageEnvelope(
            message_id="m4", from_agent="a", to_agent=None,
            content="broadcast", priority=MessagePriority.LOW,
            routing_strategy=RoutingStrategy.BROADCAST,
            timestamp=time.time()
        )
        result = await router.route_message(env)
        assert result is False

    @pytest.mark.asyncio
    async def test_route_broadcast_to_all_agents(self):
        from client.message_router import MessageEnvelope, MessagePriority, RoutingStrategy
        router, _ = self._make_router()
        router.register_agent("agent-1", MagicMock())
        router.register_agent("agent-2", MagicMock())
        env = MessageEnvelope(
            message_id="m5", from_agent="a", to_agent=None,
            content="hi all", priority=MessagePriority.NORMAL,
            routing_strategy=RoutingStrategy.BROADCAST,
            timestamp=time.time()
        )
        result = await router.route_message(env)
        assert result is True


# ═══════════════════════════════════════════════════════════════════
# Additional proactive_agent paths not yet covered
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestProactiveAgentExtra:

    @pytest.fixture
    def scheduler_db(self, tmp_path):
        db_path = tmp_path / "scheduler.db"
        with pytest.MonkeyPatch().context() as m:
            m.setattr("client.proactive_agent.SCHEDULER_DB_PATH", db_path)
            from client.proactive_agent import _ensure_db
            _ensure_db()
        return db_path

    def test_set_job_enabled_disable(self, tmp_path):
        db_path = tmp_path / "scheduler.db"
        with pytest.MonkeyPatch().context() as m:
            m.setattr("client.proactive_agent.SCHEDULER_DB_PATH", db_path)
            from client.proactive_agent import _ensure_db, create_job, set_job_enabled, get_job
            _ensure_db()
            jid = create_job("j", "t", cron="* * * * *")
            set_job_enabled(jid, False)
            assert get_job(jid)["enabled"] == 0

    def test_record_run_increments_count(self, tmp_path):
        db_path = tmp_path / "scheduler.db"
        with pytest.MonkeyPatch().context() as m:
            m.setattr("client.proactive_agent.SCHEDULER_DB_PATH", db_path)
            from client.proactive_agent import _ensure_db, create_job, record_run, get_job
            _ensure_db()
            jid = create_job("j2", "t", cron="* * * * *")
            record_run(jid)
            assert get_job(jid)["run_count"] == 1

    def test_cron_to_human_sunday(self):
        from client.proactive_agent import cron_to_human
        result = cron_to_human("0 10 * * 0")
        assert "Sun" in result or "10:00am" in result

    def test_handle_jobs_cancel_nonexistent(self, tmp_path):
        db_path = tmp_path / "scheduler.db"
        with pytest.MonkeyPatch().context() as m:
            m.setattr("client.proactive_agent.SCHEDULER_DB_PATH", db_path)
            from client.proactive_agent import _ensure_db, handle_jobs_command
            _ensure_db()
            result = handle_jobs_command(":jobs cancel nonexistent-label")
            assert result  # Returns some message without raising

    def test_confirmation_tracker_session_isolation(self):
        from client.proactive_agent import ConfirmationTracker
        tracker = ConfirmationTracker()
        tracker.set_pending("s1", MagicMock())
        tracker.set_pending("s2", MagicMock())
        tracker.clear("s1")
        assert tracker.get_pending("s1") is None
        assert tracker.get_pending("s2") is not None
