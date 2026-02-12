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
    # if command == ":multi on":
    #     if orchestrator:
    #         multi_agent_state["enabled"] = True
    #         return "✅ Multi-agent mode enabled\n   Complex queries will be broken down automatically"
    #     return "❌ Multi-agent orchestrator not available"
    #
    # elif command == ":multi off":
    #     if orchestrator:
    #         multi_agent_state["enabled"] = False
    #         return "🤖 Multi-agent mode disabled\n   Using single-agent execution"
    #     return "❌ Multi-agent orchestrator not available"
    #
    # elif command == ":multi status":
    #     if not orchestrator:
    #         return "❌ Multi-agent orchestrator not available"
    #
    #     if multi_agent_state["enabled"]:
    #         return "Multi-agent mode: ENABLED\n   Complex queries are automatically distributed to specialized agents"
    #     else:
    #         return "Multi-agent mode: DISABLED\n   Use ':multi on' to enable"
    #
    # return None

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

        # Auto-extract alias from filename if not provided
        if len(parts) >= 3:
            alias = parts[2]
        else:
            # Extract filename without extension as alias
            from pathlib import Path
            filename = Path(path).stem  # Gets filename without .gguf extension
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
        # This will be handled by showing all models
        return "list_all_models"  # Special signal

    else:
        return "❌ Invalid GGUF command. Use ':gguf help' for usage"


# SAFE VERSION OF HEALTH COMMANDS - Replace in commands.py

async def handle_health_commands(command: str, orchestrator):
    """Handle health monitoring commands"""
    if not orchestrator or not hasattr(orchestrator, 'health_monitor') or not orchestrator.health_monitor:
        return (True, "❌ Health monitoring not available", None, None)

    if command == ":health":
        summary = orchestrator.health_monitor.get_health_summary()

        # Handle empty/no agents case
        if summary.get("status") == "no_agents" or not summary.get("total_agents"):
            return (True, "❌ No agents registered yet. Enable A2A first with ':a2a on'", None, None)

        output = ["🏥 AGENT HEALTH SUMMARY", "=" * 60, ""]
        output.append(f"Overall Status: {summary.get('status', 'unknown').upper()}")
        output.append(f"Total Agents: {summary.get('total_agents', 0)}")
        output.append(f"  💚 Healthy: {summary.get('healthy', 0)}")
        output.append(f"  💛 Degraded: {summary.get('degraded', 0)}")
        output.append(f"  🔴 Unhealthy: {summary.get('unhealthy', 0)}")
        output.append(f"  ⚫ Offline: {summary.get('offline', 0)}")
        output.append("")
        output.append(f"Performance:")
        output.append(f"  Total Tasks: {summary.get('total_tasks', 0)}")
        output.append(f"  Total Errors: {summary.get('total_errors', 0)}")
        output.append(f"  Avg Response Time: {summary.get('avg_response_time', 0):.2f}s")
        output.append(f"  Recent Alerts (5min): {summary.get('recent_alerts', 0)}")
        output.append("=" * 60)

        return (True, "\n".join(output), None, None)

    elif command == ":health alerts":
        alerts = orchestrator.health_monitor.get_recent_alerts(limit=10)

        if not alerts:
            return (True, "✅ No recent alerts", None, None)

        import time
        output = ["🚨 RECENT ALERTS", "=" * 60, ""]
        for alert in alerts:
            output.append(f"{alert.level.value.upper()} | {alert.agent_id}")
            output.append(f"  {alert.message}")
            output.append(f"  {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.timestamp))}")
            output.append("")

        return (True, "\n".join(output), None, None)

    elif command.startswith(":health "):
        agent_id = command[8:].strip()
        health = orchestrator.health_monitor.get_agent_health(agent_id)

        # Try with _1 suffix if not found
        if not health:
            health = orchestrator.health_monitor.get_agent_health(f"{agent_id}_1")
            if health:
                agent_id = f"{agent_id}_1"

        if not health:
            return (True,
                    f"❌ Agent '{agent_id}' not found. Available agents: {', '.join(orchestrator.health_monitor.agent_metrics.keys())}",
                    None, None)

        import time
        status_icon = {"healthy": "💚", "degraded": "💛", "unhealthy": "🔴", "offline": "⚫"}.get(health.status.value, "❓")

        output = [f"🏥 HEALTH REPORT: {agent_id}", "=" * 60, ""]
        output.append(f"Status: {status_icon} {health.status.value.upper()}")
        output.append(f"Uptime: {health.uptime / 60:.1f} minutes")
        output.append(f"Last Heartbeat: {time.time() - health.last_heartbeat:.1f}s ago")
        output.append("")
        output.append(f"Tasks:")
        output.append(f"  Completed: {health.tasks_completed}")
        output.append(f"  Failed: {health.tasks_failed}")
        if health.tasks_completed + health.tasks_failed > 0:
            success_rate = health.tasks_completed / (health.tasks_completed + health.tasks_failed)
            output.append(f"  Success Rate: {success_rate:.1%}")
        output.append("")
        output.append(f"Performance:")
        output.append(f"  Avg Response Time: {health.avg_response_time:.2f}s")
        output.append(f"  Queue Size: {health.queue_size}")
        output.append(f"  Error Count: {health.error_count}")

        if health.last_error:
            output.append(f"\nLast Error: {health.last_error}")
            if health.last_error_time:
                output.append(f"  {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(health.last_error_time))}")

        output.append("=" * 60)

        return (True, "\n".join(output), None, None)

    return (False, None, None, None)

async def handle_metrics_commands(command: str, orchestrator):
    """Handle performance metrics commands"""
    if not orchestrator or not orchestrator.performance_metrics:
        return (True, "❌ Performance metrics not available", None, None)

    if command == ":metrics":
        report = orchestrator.performance_metrics.get_summary_report()
        return (True, report, None, None)

    elif command == ":metrics comparative":
        stats = orchestrator.performance_metrics.get_comparative_stats()

        output = ["📊 COMPARATIVE PERFORMANCE", "=" * 60, ""]

        if "overall" in stats:
            output.append("Overall Statistics:")
            output.append(f"  Avg Success Rate: {stats['overall']['avg_success_rate']:.1%}")
            output.append(f"  Avg Duration: {stats['overall']['avg_duration']:.2f}s")
            output.append(f"  Best Performer: {stats['overall']['best_performer']}")
            output.append(f"  Fastest Agent: {stats['overall']['fastest_agent']}")
            output.append("")

        output.append("Per-Agent:")
        for agent_id, data in stats['agents'].items():
            output.append(f"  {agent_id:15} | Success: {data['success_rate']:5.1%} | Avg: {data['avg_duration']:5.2f}s")

        return (True, "\n".join(output), None, None)

    elif command == ":metrics bottlenecks":
        analysis = orchestrator.performance_metrics.get_bottleneck_analysis()

        if not analysis["bottlenecks"]:
            return (True, "✅ No performance bottlenecks detected", None, None)

        output = ["⚠️  PERFORMANCE BOTTLENECKS", "=" * 60, ""]

        for bottleneck in analysis["bottlenecks"]:
            output.append(f"{bottleneck['agent_id']}:")
            for issue in bottleneck["issues"]:
                output.append(f"  - {issue}")
            output.append("")

        return (True, "\n".join(output), None, None)

    return (False, None, None, None)

async def handle_negotiation_commands(command: str, orchestrator):
    """Handle negotiation commands"""
    if not orchestrator or not orchestrator.negotiation_engine:
        return (True, "❌ Negotiation engine not available", None, None)

    if command == ":negotiations":
        stats = orchestrator.negotiation_engine.get_statistics()

        output = ["🤝 NEGOTIATION STATISTICS", "=" * 60, ""]
        output.append(f"Total Proposals: {stats['total_proposals']}")
        output.append(f"Accepted: {stats['accepted']}")
        output.append(f"Rejected: {stats['rejected']}")
        output.append(f"Expired: {stats['expired']}")
        output.append(f"Success Rate: {stats['success_rate']:.1%}")
        output.append(f"Active: {stats['active_negotiations']}")
        output.append("=" * 60)

        return (True, "\n".join(output), None, None)

    return (False, None, None, None)

async def handle_routing_commands(command: str, orchestrator):
    """Handle message routing commands"""
    if not orchestrator or not orchestrator.message_router:
        return (True, "❌ Message router not available", None, None)

    if command == ":routing":
        stats = orchestrator.message_router.get_routing_stats()

        output = ["📡 MESSAGE ROUTING STATISTICS", "=" * 60, ""]
        output.append(f"Total Routed: {stats['total_routed']}")
        output.append(f"Failed Routes: {stats['failed_routes']}")
        output.append(f"Retries: {stats['retries']}")
        output.append(f"Timeouts: {stats['timeouts']}")
        output.append(f"Pending: {stats['pending_messages']}")
        output.append(f"Completed: {stats['completed_messages']}")
        output.append("=" * 60)

        return (True, "\n".join(output), None, None)

    elif command == ":routing queues":
        status = orchestrator.message_router.get_queue_status()

        if not status:
            return (True, "No queues active", None, None)

        output = ["📬 MESSAGE QUEUE STATUS", "=" * 60, ""]

        for agent_id, queue_data in status.items():
            output.append(f"{agent_id}:")
            output.append(f"  Queue Size: {queue_data['queue_size']}")
            output.append(f"  Pending: {queue_data['pending']}")
            output.append(f"  Critical: {queue_data['priorities']['critical']}")
            output.append(f"  High: {queue_data['priorities']['high']}")
            output.append(f"  Normal: {queue_data['priorities']['normal']}")
            output.append("")

        return (True, "\n".join(output), None, None)

    return (False, None, None, None)

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
    mcp_agent=None  # ← ADDED mcp_agent parameter
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
                # Show all models instead
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

    # Health commands
    if command.startswith(":health"):
        result = await handle_health_commands(command, orchestrator)
        if result[0]:  # If handled
            return result

    # Metrics commands
    if command.startswith(":metrics"):
        result = await handle_metrics_commands(command, orchestrator)
        if result[0]:  # If handled
            return result

    # Negotiation commands
    if command.startswith(":negotiations"):
        result = await handle_negotiation_commands(command, orchestrator)
        if result[0]:  # If handled
            return result

    # Routing commands
    if command.startswith(":routing"):
        result = await handle_routing_commands(command, orchestrator)
        if result[0]:  # If handled
            return result

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

    # Tools command - UPDATED to use mcp_agent session map with fallback
    if command == ":tools" or command == ":tools --all":
        show_all = command == ":tools --all"

        if not tools:
            return (True, "No tools available (all servers may have failed to initialize)", None, None)

        try:
            from tools.tool_control import is_tool_enabled, get_disabled_tools

            # ── Build tool→server map from MCP client sessions ──────────
            # This gives us accurate grouping including external servers
            # (e.g. deepwiki tools appear under "deepwiki" not "other")
            tool_to_server = {}
            if mcp_agent and hasattr(mcp_agent, 'client') and hasattr(mcp_agent.client, 'sessions'):
                for server_name, session in mcp_agent.client.sessions.items():
                    try:
                        session_tools = await session.list_tools()
                        for t in session_tools:
                            tool_to_server[t.name] = server_name
                    except Exception:
                        pass  # Skip broken/unavailable sessions silently

            # ── Pattern matching fallback for anything not in session map ─
            category_patterns = {
                'todo': ['todo', 'task'],
                'knowledge_base': ['entry', 'entries', 'knowledge'],
                'plex': ['plex', 'media', 'scene', 'semantic_media', 'import_plex',
                         'train_recommender', 'recommend', 'record_viewing',
                         'auto_train', 'auto_recommend'],
                'rag': ['rag_'],
                'system': ['system', 'hardware', 'process'],
                'location': ['location', 'time', 'weather'],
                'text': ['text', 'summarize', 'chunk', 'explain', 'concept'],
                'code': ['code', 'debug'],
            }

            # ── Group tools by server ────────────────────────────────────
            tools_by_server = {}
            for tool in tools:
                tool_name = getattr(tool, 'name', str(tool))

                # Skip internal skill management tools
                if tool_name in ("read_skill", "list_skills"):
                    continue

                # Session map first (accurate), then pattern match, then 'other'
                server_name = tool_to_server.get(tool_name)
                if not server_name:
                    server_name = 'other'
                    for cat, patterns in category_patterns.items():
                        if any(p in tool_name.lower() for p in patterns):
                            server_name = cat
                            break

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

                # Skip servers with no enabled tools unless --all
                if not enabled and not show_all:
                    continue

                output.append(f"\n{server_name}:")

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

        # Show if current agent might be out of sync
        if model_name != last_model:
            output.append(f"\n⚠️  WARNING: Agent might be out of sync!")
            output.append(f"   Active: {model_name}")
            output.append(f"   Should be: {last_model}")
            output.append(f"   Run ':sync' to synchronize")

        return (True, "\n".join(output) if output else "", None, None)

    if command == ":models":
        # Legacy - show all models
        models_module.print_all_models()
        return (True, "", None, None)

    if command.startswith(":model "):
        new_model = command[7:].strip()

        if logger:
            logger.info(f"Switching to model: {new_model}")

        # Use the unified switch_model that auto-detects backend
        new_agent = await models_module.switch_model(
            new_model,
            tools,
            logger,
            create_langgraph_agent,
            a2a_state=a2a_state
        )

        if new_agent is None:
            return (True, f"❌ Model '{new_model}' not loaded", None, None)

        # Clear conversation history when switching models
        # conversation_state["messages"] = []
        # if logger:
        #     logger.info("✅ Chat history cleared after model switch")

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