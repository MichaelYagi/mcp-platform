"""
Tests for client/commands.py
Covers: get_commands_list, list_commands, is_command,
        handle_gguf_commands, handle_a2a_commands, handle_command
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════

def make_models_module(last_model="test-model", backend="ollama"):
    """Build a minimal mock models_module."""
    m = MagicMock()
    m.load_last_model.return_value = last_model
    m.detect_backend.return_value = backend
    m.print_all_models = MagicMock()
    m.switch_model = AsyncMock(return_value=MagicMock())
    m.reload_current_model = AsyncMock(return_value=(MagicMock(), last_model))
    return m


def make_tool(name, description="A test tool", enabled=True):
    tool = MagicMock()
    tool.name = name
    tool.description = description
    return tool


async def call_handle_command(command, tools=None, model_name="test-model",
                               conversation_state=None, models_module=None,
                               logger=None, **kwargs):
    from client.commands import handle_command
    return await handle_command(
        command=command,
        tools=tools or [],
        model_name=model_name,
        conversation_state=conversation_state or {"messages": []},
        models_module=models_module or make_models_module(),
        system_prompt="Test",
        logger=logger or MagicMock(),
        **kwargs
    )


# ═══════════════════════════════════════════════════════════════════
# get_commands_list / list_commands / is_command
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestCommandsBasics:
    def test_get_commands_list_returns_list(self):
        from client.commands import get_commands_list
        cmds = get_commands_list()
        assert isinstance(cmds, list)
        assert len(cmds) > 0

    def test_get_commands_list_contains_key_commands(self):
        from client.commands import get_commands_list
        cmds = "\n".join(get_commands_list())
        assert ":stop" in cmds
        assert ":model" in cmds
        assert ":tools" in cmds
        assert ":sessions" in cmds

    def test_list_commands_prints(self, capsys):
        from client.commands import list_commands
        list_commands()
        captured = capsys.readouterr()
        assert "Commands" in captured.out

    def test_is_command_true_for_colon_prefix(self):
        from client.commands import is_command
        assert is_command(":stop") is True
        assert is_command(":model qwen2.5") is True
        assert is_command("  :tools  ") is True

    def test_is_command_false_for_normal_text(self):
        from client.commands import is_command
        assert is_command("hello world") is False
        assert is_command("what's the weather?") is False
        assert is_command("") is False


# ═══════════════════════════════════════════════════════════════════
# handle_gguf_commands
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
class TestHandleGGUFCommands:
    async def test_non_gguf_command_returns_none(self):
        from client.commands import handle_gguf_commands
        result = await handle_gguf_commands(":stop")
        assert result is None

    async def test_gguf_help_returns_help_text(self):
        from client.commands import handle_gguf_commands
        result = await handle_gguf_commands(":gguf help")
        assert "GGUF Model Commands" in result

    async def test_gguf_no_args_returns_help(self):
        from client.commands import handle_gguf_commands
        result = await handle_gguf_commands(":gguf")
        assert "GGUF Model Commands" in result

    async def test_gguf_list_returns_signal(self):
        from client.commands import handle_gguf_commands
        result = await handle_gguf_commands(":gguf list")
        assert result == "list_all_models"

    async def test_gguf_remove_calls_registry(self):
        from client.commands import handle_gguf_commands
        with patch("client.commands.GGUFModelRegistry.remove_model") as mock_remove:
            result = await handle_gguf_commands(":gguf remove my-model")
            mock_remove.assert_called_once_with("my-model")
            assert "my-model" in result

    async def test_gguf_add_success(self, temp_dir):
        from client.commands import handle_gguf_commands
        fake_gguf = temp_dir / "model.gguf"
        fake_gguf.write_bytes(b"0" * 2_000_000)
        with patch("client.llm_backend.GGUF_MODELS_FILE", str(temp_dir / "registry.json")):
            result = await handle_gguf_commands(f":gguf add {fake_gguf}")
            assert "registered" in result.lower()

    async def test_gguf_add_with_alias(self, temp_dir):
        from client.commands import handle_gguf_commands
        fake_gguf = temp_dir / "model.gguf"
        fake_gguf.write_bytes(b"0" * 2_000_000)
        with patch("client.llm_backend.GGUF_MODELS_FILE", str(temp_dir / "registry.json")):
            result = await handle_gguf_commands(f":gguf add {fake_gguf} my-alias")
            assert "my-alias" in result

    async def test_gguf_add_failure_returns_error(self):
        from client.commands import handle_gguf_commands
        result = await handle_gguf_commands(":gguf add /nonexistent/model.gguf")
        assert "❌" in result

    async def test_gguf_invalid_command(self):
        from client.commands import handle_gguf_commands
        result = await handle_gguf_commands(":gguf unknown_cmd")
        assert "Invalid" in result


# ═══════════════════════════════════════════════════════════════════
# handle_a2a_commands
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
class TestHandleA2ACommands:
    async def test_a2a_on_with_orchestrator(self):
        from client.commands import handle_a2a_commands
        orch = MagicMock()
        orch.enable_a2a = MagicMock()
        result = await handle_a2a_commands(":a2a on", orch)
        orch.enable_a2a.assert_called_once()
        assert "enabled" in result.lower()

    async def test_a2a_on_without_orchestrator(self):
        from client.commands import handle_a2a_commands
        result = await handle_a2a_commands(":a2a on", None)
        assert "not available" in result.lower()

    async def test_a2a_off_with_orchestrator(self):
        from client.commands import handle_a2a_commands
        orch = MagicMock()
        orch.disable_a2a = MagicMock()
        result = await handle_a2a_commands(":a2a off", orch)
        orch.disable_a2a.assert_called_once()

    async def test_a2a_status_disabled(self):
        from client.commands import handle_a2a_commands
        orch = MagicMock()
        orch.get_a2a_status.return_value = {"enabled": False, "agents": {}}
        result = await handle_a2a_commands(":a2a status", orch)
        assert "DISABLED" in result

    async def test_a2a_status_enabled(self):
        from client.commands import handle_a2a_commands
        orch = MagicMock()
        orch.get_a2a_status.return_value = {
            "enabled": True,
            "agents": {
                "agent1": {"is_busy": False, "tools": ["t1"], "messages_sent": 0}
            },
            "message_queue_size": 0
        }
        result = await handle_a2a_commands(":a2a status", orch)
        assert "ENABLED" in result

    async def test_a2a_unrecognized_returns_none(self):
        from client.commands import handle_a2a_commands
        result = await handle_a2a_commands(":a2a unknown", MagicMock())
        assert result is None


# ═══════════════════════════════════════════════════════════════════
# handle_command — simple commands
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
class TestHandleCommandSimple:
    async def test_commands_command(self):
        handled, response, _, _ = await call_handle_command(":commands")
        assert handled is True
        assert ":stop" in response

    async def test_stop_command(self):
        with patch("client.stop_signal.request_stop") as mock_stop:
            handled, response, _, _ = await call_handle_command(":stop")
            assert handled is True
            assert "Stop" in response

    async def test_unrecognized_command_not_handled(self):
        handled, response, _, _ = await call_handle_command(":nonexistent_command_xyz")
        assert handled is False
        assert response is None

    async def test_metrics_unavailable(self):
        with patch("client.commands.handle_command.__module__"):
            handled, response, _, _ = await call_handle_command(":metrics")
            assert handled is True
            assert "not available" in response.lower() or "Metrics" in response

    async def test_negotiations_unavailable(self):
        handled, response, _, _ = await call_handle_command(":negotiations")
        assert handled is True
        assert "not available" in response.lower()

    async def test_routing_unavailable(self):
        handled, response, _, _ = await call_handle_command(":routing")
        assert handled is True
        assert "not available" in response.lower()


# ═══════════════════════════════════════════════════════════════════
# handle_command — tool commands
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
class TestHandleCommandTools:
    async def test_tool_detail_found(self):
        tools = [make_tool("weather_tool", "Gets the weather")]
        handled, response, _, _ = await call_handle_command(":tool weather_tool", tools=tools)
        assert handled is True
        assert "weather_tool" in response
        assert "Gets the weather" in response

    async def test_tool_detail_not_found(self):
        handled, response, _, _ = await call_handle_command(":tool nonexistent", tools=[])
        assert handled is True
        assert "not found" in response

    async def test_tools_no_tools_available(self):
        handled, response, _, _ = await call_handle_command(":tools", tools=[])
        assert handled is True
        assert "No tools" in response or "available" in response.lower()


# ═══════════════════════════════════════════════════════════════════
# handle_command — model commands
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
class TestHandleCommandModel:
    async def test_model_list_command(self):
        mm = make_models_module(last_model="qwen2.5:14b")
        handled, response, _, _ = await call_handle_command(":model", models_module=mm)
        assert handled is True
        mm.print_all_models.assert_called_once()

    async def test_model_switch_success(self):
        mm = make_models_module()
        new_agent = MagicMock()
        mm.switch_model = AsyncMock(return_value=new_agent)
        handled, response, agent, model = await call_handle_command(
            ":model llama3.1:8b", models_module=mm
        )
        assert handled is True
        assert agent is new_agent
        assert model == "llama3.1:8b"
        assert "Switched" in response

    async def test_model_switch_failure(self):
        mm = make_models_module()
        mm.switch_model = AsyncMock(return_value=None)
        handled, response, agent, model = await call_handle_command(
            ":model bad-model", models_module=mm
        )
        assert handled is True
        assert agent is None
        assert "not loaded" in response

    async def test_models_legacy_command(self):
        mm = make_models_module()
        handled, response, _, _ = await call_handle_command(":models", models_module=mm)
        assert handled is True
        mm.print_all_models.assert_called_once()


# ═══════════════════════════════════════════════════════════════════
# handle_command — session commands
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
class TestHandleCommandSessions:
    async def test_clear_sessions(self, session_manager):
        conv = {"messages": ["old message"]}
        with patch("client.commands.SessionManager") as MockSM:
            mock_sm = MagicMock()
            MockSM.return_value = mock_sm
            handled, response, _, _ = await call_handle_command(
                ":clear sessions", conversation_state=conv
            )
        assert handled is True
        assert conv["messages"] == []
        assert "cleared" in response.lower()

    async def test_sessions_empty(self):
        with patch("client.commands.SessionManager") as MockSM:
            mock_sm = MagicMock()
            mock_sm.get_sessions.return_value = []
            MockSM.return_value = mock_sm
            handled, response, _, _ = await call_handle_command(":sessions")
        assert handled is True
        assert "No sessions" in response

    async def test_sessions_with_data(self):
        with patch("client.commands.SessionManager") as MockSM:
            mock_sm = MagicMock()
            mock_sm.get_sessions.return_value = [
                {"id": 1, "name": "Chat about weather"},
                {"id": 2, "name": "Code review session"},
            ]
            MockSM.return_value = mock_sm
            handled, response, _, _ = await call_handle_command(":sessions")
        assert handled is True
        assert "Session 1" in response
        assert "Session 2" in response

    async def test_clear_specific_session_found(self):
        with patch("client.commands.SessionManager") as MockSM:
            mock_sm = MagicMock()
            mock_sm.get_session.return_value = {"id": 5, "name": "My session"}
            MockSM.return_value = mock_sm
            handled, response, _, _ = await call_handle_command(":clear session 5")
        assert handled is True
        mock_sm.delete_session.assert_called_once_with(5)
        assert "5" in response

    async def test_clear_specific_session_not_found(self):
        with patch("client.commands.SessionManager") as MockSM:
            mock_sm = MagicMock()
            mock_sm.get_session.return_value = None
            MockSM.return_value = mock_sm
            handled, response, _, _ = await call_handle_command(":clear session 99")
        assert handled is True
        assert "not found" in response.lower()