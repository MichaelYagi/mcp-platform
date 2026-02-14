"""
Command Handlers for MCP Client
Compatible with existing CLI/WebSocket interfaces
UPDATED: :tools command now filters disabled tools
"""
from client.langgraph import create_langgraph_agent
from client.llm_backend import GGUFModelRegistry
from client.session_manager import SessionManager


def get_commands_list():
    """Get list of available commands"""
    return [
        ":commands - List all available commands",
        ":clear history - Clear all chat history",
        ":clear session <id> - Clear session history",
        ":sessions - List all available sessions",
        ":stop - Stop current operation (ingestion, search, etc.)",
        ":stats - Show performance metrics",
        ":tools - List available tools (disabled tools hidden)",
        ":tools --all - List all tools (shows disabled tools marked)",
        ":tool <tool> - Get the tool description",
        ":model - List all available models (Ollama + GGUF)",
        ":model <model> - Switch to model (auto-detects backend)",
        ":models - List available models (legacy)",
        ":sync - Sync agent to model in last_model.txt",
        ":gguf add <path> - Register a GGUF model",
        ":gguf remove <alias> - Remove a GGUF model",
        ":gguf list - List registered GGUF models",
        ":a2a on - Enable agent-to-agent mode",
        ":a2a off - Disable agent-to-agent mode",
        ":a2a status - Check A2A system status",
        ":health - Health overview of all servers and tools",
        ":env - Show environment configuration"
    ]


def list_commands():
    """Print all available commands"""
    print("\nAvailable Commands:")
    for cmd in get_commands_list():
        print(f"  {cmd}")


async def handle_a2a_commands(command: str, orchestrator):
    """
    Handle A2A-specific commands
    Returns result string or None if command not handled
    """
    if command == ":a2a on":
        if orchestrator:
            orchestrator.enable_a2a()
            return "✅ A2A mode enabled\n   Agents will communicate via messages\n   Use ':a2a status' to see agent status"
        return "❌ Multi-agent orchestrator not available"

    elif command == ":a2a off":
        if orchestrator:
            orchestrator.disable_a2a()
            return "🔗 A2A mode disabled\n   Falling back to multi-agent or single-agent mode"
        return "❌ Multi-agent orchestrator not available"

    elif command == ":a2a status":
        if not orchestrator:
            return "❌ Multi-agent orchestrator not available"

        status = orchestrator.get_a2a_status()
        if not status["enabled"]:
            return "A2A mode: DISABLED\n\nUse ':a2a on' to enable agent-to-agent communication"

        output = ["A2A mode: ENABLED", "=" * 60, ""]
        output.append("Agent Status:")
        output.append("-" * 60)

        for agent_name, agent_status in status["agents"].items():
            busy = "🔴 BUSY" if agent_status["is_busy"] else "🟢 IDLE"
            tools_count = len(agent_status["tools"])
            msgs = agent_status["messages_sent"]

            output.append(f"  {agent_name:15} {busy} | Tools: {tools_count:2} | Messages: {msgs:3}")

        output.append("")
        output.append(f"Message Queue: {status['message_queue_size']} messages")
        output.append("=" * 60)

        return "\n".join(output)

    return None


async def handle_multi_agent_commands(command: str, orchestrator, multi_agent_state):
    """
    Handle multi-agent commands
    Returns result string or None if command not handled
    """
    multi_agent_state["enabled"] = True
    return "✅ Multi-agent mode enabled\n   Complex queries will be broken down automatically"


async def handle_gguf_commands(command: str):
    """
    Handle GGUF model registry commands
    Returns result string or None if command not handled
    """
    if not command.startswith(":gguf"):
        return None

    parts = command[6:].strip().split(maxsplit=2)

    if not parts or parts[0] == "help":
        return (
            "\n📦 GGUF Model Commands:\n"
            "  :gguf add <path>                   - Register a GGUF model\n"
            "  :gguf remove <alias>               - Remove a GGUF model\n"
            "  :gguf list                         - List registered models\n"
            "\n"
            "Examples:\n"
            "  :gguf add /path/to/tinyllama.gguf           (uses 'tinyllama' as alias)\n"
            "  :gguf add /path/to/model.gguf my-model      (uses 'my-model' as alias)\n"
            "  :gguf remove tinyllama\n"
        )

    cmd = parts[0]

    if cmd == "add" and len(parts) >= 2:
        path = parts[1]

        if len(parts) >= 3:
            alias = parts[2]
        else:
            from pathlib import Path
            filename = Path(path).stem
            alias = filename

        try:
            GGUFModelRegistry.add_model(alias, path, "")
            return f"\n✅ Model '{alias}' registered!\n   Switch to it with: :model {alias}\n"
        except Exception as e:
            return f"❌ Error: {e}"

    elif cmd == "remove" and len(parts) >= 2:
        alias = parts[1]
        GGUFModelRegistry.remove_model(alias)
        return f"✅ Removed: {alias}"

    elif cmd == "list":
        return "list_all_models"  # Special signal

    else:
        return "❌ Invalid GGUF command. Use ':gguf help' for usage"


def is_command(text: str) -> bool:
    """Check if text is a command"""
    return text.strip().startswith(":")


async def handle_command(
    command: str,
    tools,
    model_name,
    conversation_state,
    models_module,
    system_prompt,
    agent_ref=None,
    create_agent_fn=None,
    logger=None,
    orchestrator=None,
    multi_agent_state=None,
    a2a_state=None,
    mcp_agent=None
):
    """
    Main command handler compatible with existing CLI/WebSocket interface

    Returns: (handled: bool, response: str, new_agent, new_model)
    """
    command = command.strip()

    # GGUF commands
    if command.startswith(":gguf"):
        result = await handle_gguf_commands(command)
        if result:
            if result == "list_all_models":
                models_module.print_all_models()
                return (True, "", None, None)
            return (True, result, None, None)

    # A2A commands
    if command.startswith(":a2a"):
        result = await handle_a2a_commands(command, orchestrator)
        if result:
            return (True, result, None, None)

    # .env values
    if command == ":env":
        from client.env_display import format_env_display
        return True, format_env_display(), None, None

    # Health commands — MCP server/tool health check
    if command.startswith(":health"):
        from client.health import run_health_check
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        result = await run_health_check(command[7:].strip(), mcp_agent, tools, project_root)
        return (True, result, None, None)

    # Metrics commands (A2A - no-op if not available)
    if command.startswith(":metrics"):
        if orchestrator and hasattr(orchestrator, 'performance_metrics') and orchestrator.performance_metrics:
            if command == ":metrics":
                report = orchestrator.performance_metrics.get_summary_report()
                return (True, report, None, None)
            elif command == ":metrics comparative":
                stats = orchestrator.performance_metrics.get_comparative_stats()
                output = ["📊 COMPARATIVE PERFORMANCE", "=" * 60, ""]
                if "overall" in stats:
                    output.append(f"  Avg Success Rate: {stats['overall']['avg_success_rate']:.1%}")
                    output.append(f"  Avg Duration: {stats['overall']['avg_duration']:.2f}s")
                    output.append(f"  Best Performer: {stats['overall']['best_performer']}")
                    output.append("")
                for agent_id, data in stats.get('agents', {}).items():
                    output.append(f"  {agent_id:15} | Success: {data['success_rate']:5.1%} | Avg: {data['avg_duration']:5.2f}s")
                return (True, "\n".join(output), None, None)
        return (True, "📊 Metrics not available (A2A mode only)", None, None)

    # Negotiation commands (A2A - no-op if not available)
    if command.startswith(":negotiations"):
        if orchestrator and hasattr(orchestrator, 'negotiation_engine') and orchestrator.negotiation_engine:
            stats = orchestrator.negotiation_engine.get_statistics()
            output = ["🤝 NEGOTIATION STATISTICS", "=" * 60, ""]
            output.append(f"Total Proposals: {stats['total_proposals']}")
            output.append(f"Accepted: {stats['accepted']}")
            output.append(f"Rejected: {stats['rejected']}")
            output.append(f"Success Rate: {stats['success_rate']:.1%}")
            output.append(f"Active: {stats['active_negotiations']}")
            return (True, "\n".join(output), None, None)
        return (True, "🤝 Negotiations not available (A2A mode only)", None, None)

    # Routing commands (A2A - no-op if not available)
    if command.startswith(":routing"):
        if orchestrator and hasattr(orchestrator, 'message_router') and orchestrator.message_router:
            if command == ":routing":
                stats = orchestrator.message_router.get_routing_stats()
                output = ["📡 MESSAGE ROUTING STATISTICS", "=" * 60, ""]
                output.append(f"Total Routed: {stats['total_routed']}")
                output.append(f"Failed Routes: {stats['failed_routes']}")
                output.append(f"Pending: {stats['pending_messages']}")
                output.append(f"Completed: {stats['completed_messages']}")
                return (True, "\n".join(output), None, None)
        return (True, "📡 Routing not available (A2A mode only)", None, None)

    # Multi-agent commands
    if command.startswith(":multi"):
        result = await handle_multi_agent_commands(command, orchestrator, multi_agent_state)
        if result:
            return (True, result, None, None)

    # List commands
    if command == ":commands":
        result = "\n".join(get_commands_list())
        return (True, result, None, None)

    # Sync with last_model.txt
    if command == ":sync":
        last_model = models_module.load_last_model()
        if not last_model:
            return (True, "❌ No last_model.txt found", None, None)

        logger.info(f"🔄 Syncing to last_model.txt: {last_model}")

        new_agent, new_model = await models_module.reload_current_model(
            tools, logger, create_langgraph_agent, a2a_state
        )

        if new_agent:
            return (True, f"✅ Synced to model: {new_model}", new_agent, new_model)
        else:
            return (True, f"❌ Failed to sync to: {last_model}", None, None)

    # Stop command
    if command == ":stop":
        from client.stop_signal import request_stop
        request_stop()
        return (True, "🛑 Stop signal sent - operations will halt at next checkpoint", None, None)

    # Stats command
    if command == ":stats":
        try:
            from client.metrics import prepare_metrics, format_metrics_summary
            metrics = prepare_metrics()
            summary = format_metrics_summary(metrics)
            return (True, summary, None, None)
        except ImportError:
            return (True, "📊 Stats system not available", None, None)

    # Tools command
    if command == ":tools" or command == ":tools --all":
        show_all = command == ":tools --all"

        if not tools:
            return (True, "No tools available (all servers may have failed to initialize)", None, None)

        from pathlib import Path
        from client.tool_utils import load_external_server_names, resolve_tool_server
        project_root = Path(__file__).parent.parent
        external_server_names = load_external_server_names(project_root)

        try:
            from tools.tool_control import is_tool_enabled, get_disabled_tools

            # ── Resolve tool→server via shared utility ───────────────────
            tool_to_server = await resolve_tool_server(tools, mcp_agent, project_root)

            # ── Group tools by server ────────────────────────────────────
            tools_by_server = {}
            for tool in tools:
                tool_name = getattr(tool, 'name', str(tool))

                # Skip internal skill management tools
                if tool_name in ("read_skill", "list_skills"):
                    continue

                # 1. Resolved map (live session + metadata + pattern)
                server_name = tool_to_server.get(tool_name)

                # 2. Explicit metadata check (belt-and-suspenders for coingecko etc.)
                if not server_name:
                    try:
                        meta = getattr(tool, 'meta', None) or getattr(tool, 'metadata', None)
                        if isinstance(meta, dict):
                            server_name = meta.get('source_server')
                    except Exception:
                        pass

                # 3. Last resort
                if not server_name:
                    server_name = 'other'

                enabled = is_tool_enabled(tool_name, server_name)

                if server_name not in tools_by_server:
                    tools_by_server[server_name] = {'enabled': [], 'disabled': []}

                if enabled:
                    tools_by_server[server_name]['enabled'].append(tool)
                else:
                    tools_by_server[server_name]['disabled'].append(tool)

            # ── Build output ─────────────────────────────────────────────
            output = ["\n" + "=" * 60]
            output.append("ALL TOOLS (including disabled)" if show_all else "AVAILABLE TOOLS")
            output.append("=" * 60)

            total_enabled = 0
            total_disabled = 0

            for server_name in sorted(tools_by_server.keys()):
                server_data = tools_by_server[server_name]
                enabled = server_data['enabled']
                disabled = server_data['disabled']

                if not enabled and not show_all:
                    continue

                is_external = server_name in external_server_names
                label = " [external]" if is_external else ""
                output.append(f"\n{server_name}{label}:")

                for tool in enabled:
                    tool_name = getattr(tool, 'name', str(tool))
                    tool_desc = getattr(tool, 'description', '') or ''
                    desc_line = tool_desc.split('\n')[0][:70]
                    output.append(f"  ✓ {tool_name}")
                    if desc_line:
                        output.append(f"    {desc_line}")

                total_enabled += len(enabled)

                if show_all and disabled:
                    if enabled:
                        output.append("")
                    output.append("  DISABLED:")
                    for tool in disabled:
                        tool_name = getattr(tool, 'name', str(tool))
                        tool_desc = getattr(tool, 'description', '') or ''
                        desc_line = tool_desc.split('\n')[0][:70]
                        output.append(f"  ✗ {tool_name} [DISABLED]")
                        if desc_line:
                            output.append(f"    {desc_line}")

                total_disabled += len(disabled)

            output.append("")
            output.append("=" * 60)
            output.append(f"Available: {total_enabled} tools")

            if total_disabled > 0:
                output.append(f"Disabled: {total_disabled} tools")
                if not show_all:
                    output.append("Use ':tools --all' to see disabled tools")
                output.append("Check DISABLED_TOOLS in .env to modify")

            output.append("=" * 60)

            return (True, "\n".join(output), None, None)

        except ImportError:
            tool_list = "\n".join([f"  - {tool.name}" for tool in tools])
            return (True, f"Available tools:\n{tool_list}", None, None)

    # Tool detail command
    if command.startswith(":tool "):
        tool_name = command[6:].strip()
        for tool in tools:
            if tool.name == tool_name:
                return (True, f"Tool: {tool.name}\n\n{tool.description}", None, None)
        return (True, f"Tool '{tool_name}' not found", None, None)

    # Model commands - show sync status
    if command == ":model":
        last_model = models_module.load_last_model()
        current_backend = models_module.detect_backend(last_model) if last_model else "unknown"

        output = []
        output.append(f"\n📌 Current Model (from last_model.txt):")
        output.append(f"   {current_backend}/{last_model}")
        output.append("")

        models_module.print_all_models()

        if model_name != last_model:
            output.append(f"\n⚠️  WARNING: Agent might be out of sync!")
            output.append(f"   Active: {model_name}")
            output.append(f"   Should be: {last_model}")
            output.append(f"   Run ':sync' to synchronize")

        return (True, "\n".join(output) if output else "", None, None)

    if command == ":models":
        models_module.print_all_models()
        return (True, "", None, None)

    if command.startswith(":model "):
        new_model = command[7:].strip()

        if logger:
            logger.info(f"Switching to model: {new_model}")

        new_agent = await models_module.switch_model(
            new_model,
            tools,
            logger,
            create_langgraph_agent,
            a2a_state=a2a_state
        )

        if new_agent is None:
            return (True, f"❌ Model '{new_model}' not loaded", None, None)

        return (True, f"✅ Switched to model: {new_model}\n💬 Chat history cleared", new_agent, new_model)

    # Clear history
    if command == ":clear history":
        conversation_state["messages"] = []
        sessionmanager = SessionManager()
        sessionmanager.delete_all_sessions()
        return (True, "✅ Chat history cleared", None, None)

    # List all sessions
    if command == ":sessions":
        conversation_state["messages"] = []
        sessionmanager = SessionManager()
        sessions = sessionmanager.get_sessions()
        session_list = []
        for session in sessions:
            session_id = session.get('id', 'No session ID')
            session_name = session.get('name', 'No session name')
            session_list.append(f"Session {session_id}: {session_name}")
        if len(session_list) > 0:
            return (True, "\n".join(session_list), None, None)
        else:
            return (True, "No sessions", None, None)

    # Delete session
    if command.startswith(":clear session "):
        session_id = int(command[15:].strip())
        sessionmanager = SessionManager()
        session = sessionmanager.get_session(session_id)
        if session is not None:
            sessionmanager.delete_session(session_id)
            return (True, f"✅ Session {session_id} - {session.get('name', 'No session name')} deleted", None, None)
        return (True, "✅ Session not found", None, None)

    # Command not recognized
    return (False, None, None, None)