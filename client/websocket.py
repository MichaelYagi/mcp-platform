"""
WebSocket Module with Concurrent Processing
Uses asyncio.create_task to handle operations in background
History question workaround for weaker models (7B)
"""

import asyncio
import json
import os
import socket
import websockets

from langchain_core.messages import HumanMessage

from client.commands import handle_command, handle_a2a_commands
from client.langgraph import create_langgraph_agent
from client.stop_signal import request_stop

# Import system monitor conditionally
try:
    from tools.system_monitor import system_monitor_loop
    SYSTEM_MONITOR_AVAILABLE = True
except ImportError:
    SYSTEM_MONITOR_AVAILABLE = False

CONNECTED_WEBSOCKETS = set()
SYSTEM_MONITOR_CLIENTS = set()


async def broadcast_message(message_type, data):
    """Broadcast a message to all connected WebSocket clients"""
    if CONNECTED_WEBSOCKETS:
        message = json.dumps({"type": message_type, **data})
        await asyncio.gather(
            *[ws.send(message) for ws in CONNECTED_WEBSOCKETS],
            return_exceptions=True
        )


async def process_query(websocket, prompt, original_prompt, agent_ref, conversation_state, run_agent_fn, logger, tools,
                        session_manager=None, session_id=None, system_prompt=None):
    """Process a query in the background"""
    try:
        print(f"\n> {original_prompt}")
        await broadcast_message("user_message", {"text": original_prompt})

        # Add session_id to conversation_state for context tracking
        if session_id and "session_id" not in conversation_state:
            conversation_state["session_id"] = session_id

        # ═══════════════════════════════════════════════════════════════
        # Intercept history questions for weaker models (7B)
        # Some models refuse to follow instructions even when correct
        # ═══════════════════════════════════════════════════════════════
        agent = agent_ref[0]

        history_question_detected = False
        response_text = None

        from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

        # Check for user's previous prompt questions
        if any(phrase in prompt.lower() for phrase in [
            "last prompt", "previous prompt", "what did i just ask",
            "what was my question", "my last question", "previous question",
            "what did i say", "my previous message"
        ]):
            logger.info("🎯 History question (user prompt) detected - answering directly")
            history_question_detected = True

            # Find the most recent HumanMessage BEFORE the current one
            previous_human_message = None
            for msg in reversed(conversation_state["messages"]):
                if isinstance(msg, HumanMessage):
                    previous_human_message = msg.content
                    break

            if previous_human_message:
                response_text = f'Your last prompt was: "{previous_human_message}"'
            else:
                logger.warning("⚠️ History question but no previous message found")
                history_question_detected = False

        # Check for assistant's previous response questions
        elif any(phrase in prompt.lower() for phrase in [
            "your response", "your answer", "what did you say", "what did you respond",
            "your last response", "your last answer", "what was your response",
            "what was your answer", "your reply", "your last reply"
        ]):
            logger.info("🎯 History question (assistant response) detected - answering directly")
            history_question_detected = True

            # Find the most recent AIMessage
            previous_ai_message = None
            for msg in reversed(conversation_state["messages"]):
                if isinstance(msg, AIMessage):
                    previous_ai_message = msg.content
                    break

            if previous_ai_message:
                if len(previous_ai_message) > 500:
                    response_text = f'I said: "{previous_ai_message[:500]}..." (truncated for brevity)\n\nWould you like me to repeat the full response?'
                else:
                    response_text = f'I said: "{previous_ai_message}"'
            else:
                logger.warning("⚠️ History question but no previous AI message found")
                history_question_detected = False

        # Check for conversation summary questions
        elif any(phrase in prompt.lower() for phrase in [
            "what did we discuss", "what have we talked about", "summarize our conversation",
            "what have we been discussing", "recap our conversation", "conversation summary"
        ]):
            logger.info("🎯 Conversation summary question detected - answering directly")
            history_question_detected = True

            recent_exchanges = []
            for msg in reversed(conversation_state["messages"]):
                if isinstance(msg, (HumanMessage, AIMessage)) and not isinstance(msg, SystemMessage):
                    recent_exchanges.insert(0, msg)
                    if len(recent_exchanges) >= 20:
                        break

            if recent_exchanges:
                summary_lines = ["Here's a summary of our recent conversation:\n"]
                for i, msg in enumerate(recent_exchanges):
                    if isinstance(msg, HumanMessage):
                        summary_lines.append(f"You asked: \"{msg.content[:100]}{'...' if len(msg.content) > 100 else ''}\"")
                    elif isinstance(msg, AIMessage):
                        summary_lines.append(f"I responded: \"{msg.content[:100]}{'...' if len(msg.content) > 100 else ''}\"")
                response_text = "\n".join(summary_lines)
            else:
                response_text = "We haven't had any conversation yet in this session."

        # Check for "what have I asked" questions
        elif any(phrase in prompt.lower() for phrase in [
            "what have i asked", "list my questions", "my previous questions",
            "what questions have i asked", "show my prompts"
        ]):
            logger.info("🎯 List of user prompts question detected - answering directly")
            history_question_detected = True

            user_prompts = []
            for msg in conversation_state["messages"]:
                if isinstance(msg, HumanMessage):
                    user_prompts.append(msg.content)

            if user_prompts:
                response_text = "Here are your recent prompts:\n\n"
                for i, prompt_text in enumerate(user_prompts[-10:], 1):
                    response_text += f"{i}. \"{prompt_text}\"\n"
            else:
                response_text = "You haven't asked any questions yet in this session."

        # Check for topic-specific search: "what did you say about X"
        elif "what did you say about" in prompt.lower() or "what did i ask about" in prompt.lower():
            logger.info("🎯 Topic-specific search question detected - answering directly")
            history_question_detected = True

            topic = None
            if "what did you say about" in prompt.lower():
                topic = prompt.lower().split("what did you say about")[-1].strip().strip("?")
            elif "what did i ask about" in prompt.lower():
                topic = prompt.lower().split("what did i ask about")[-1].strip().strip("?")

            if topic:
                relevant_messages = []
                for msg in conversation_state["messages"]:
                    if isinstance(msg, AIMessage) and topic in msg.content.lower():
                        relevant_messages.append(("assistant", msg.content))
                    elif isinstance(msg, HumanMessage) and topic in msg.content.lower():
                        relevant_messages.append(("user", msg.content))

                if relevant_messages:
                    response_text = f"Here's what we discussed about '{topic}':\n\n"
                    for role, content in relevant_messages[-5:]:
                        preview = content[:200] + "..." if len(content) > 200 else content
                        if role == "user":
                            response_text += f"You: \"{preview}\"\n\n"
                        else:
                            response_text += f"Me: \"{preview}\"\n\n"
                else:
                    response_text = f"We haven't discussed '{topic}' in this conversation yet."
            else:
                history_question_detected = False

        # If we detected a history question, answer directly (bypass LLM)
        if history_question_detected and response_text:
            print("\n" + response_text + "\n")

            if session_manager and session_id:
                MAX_MESSAGE_HISTORY = int(os.getenv('MAX_MESSAGE_HISTORY', 30))
                model_name = "direct-answer"
                session_manager.add_message(session_id, "assistant", response_text, MAX_MESSAGE_HISTORY, model_name)

            await broadcast_message("assistant_message", {
                "text": response_text,
                "multi_agent": False,
                "a2a": False,
                "model": "direct-answer"
            })

            await websocket.send(json.dumps({
                "type": "complete",
                "stopped": False
            }))

            return  # Exit early - don't call LLM

        # ═══════════════════════════════════════════════════════════════
        # Normal flow - Run agent (langgraph will preserve SystemMessage)
        # ═══════════════════════════════════════════════════════════════
        result = await run_agent_fn(
            agent,
            conversation_state,
            prompt,
            logger,
            tools,
            system_prompt
        )

        # ═══════════════════════════════════════════════════════════════
        # Process result and send to user
        # ═══════════════════════════════════════════════════════════════
        final_message = result["messages"][-1]
        assistant_text = final_message.content

        print("\n" + assistant_text + "\n")

        # Save to session
        if session_manager and session_id:
            MAX_MESSAGE_HISTORY = int(os.getenv('MAX_MESSAGE_HISTORY', 30))
            model_name = result.get("current_model", "unknown")
            session_manager.add_message(session_id, "assistant", assistant_text, MAX_MESSAGE_HISTORY, model_name)

        # Broadcast to WebSocket clients
        await broadcast_message("assistant_message", {
            "text": assistant_text,
            "multi_agent": result.get("multi_agent", False),
            "a2a": result.get("a2a", False),
            "model": result.get("current_model", "unknown")
        })

        await websocket.send(json.dumps({
            "type": "complete",
            "stopped": result.get("stopped", False)
        }))

    except Exception as e:
        logger.error(f"❌ Error processing query: {e}")
        import traceback
        traceback.print_exc()

        await websocket.send(json.dumps({
            "type": "error",
            "text": str(e)
        }))

        await websocket.send(json.dumps({
            "type": "complete",
            "stopped": False
        }))

async def websocket_handler(websocket, agent_ref, tools, logger, conversation_state, run_agent_fn,
                            models_module, model_name, system_prompt, orchestrator=None,
                            multi_agent_state=None, a2a_state=None, mcp_agent=None, session_manager=None):
    """
    Handle WebSocket connections with TRUE concurrent processing

    KEY: Long operations run as background tasks, allowing :stop to be processed immediately
    """
    CONNECTED_WEBSOCKETS.add(websocket)

    # Track current background task (if any)
    current_task = None
    current_session_id = None

    # Sync with last_model.txt on connection
    last_model = models_module.load_last_model()
    if last_model and last_model != model_name:
        logger.info(f"🔄 WebSocket syncing to last_model.txt: {last_model}")
        new_agent, new_model = await models_module.reload_current_model(
            tools, logger, create_langgraph_agent, a2a_state
        )
        if new_agent:
            agent_ref[0] = new_agent
            model_name = new_model
            logger.info(f"✅ WebSocket synced to: {model_name}")

    try:
        async for raw in websocket:

            if not raw or not raw.strip():
                continue

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = {"type": "user", "text": raw}

            # ═══════════════════════════════════════════════════════════
            # SESSION MANAGEMENT
            # ═══════════════════════════════════════════════════════════

            if data.get("type") == "list_sessions" and session_manager:
                sessions = session_manager.get_all_sessions()
                await websocket.send(json.dumps({
                    "type": "sessions_list",
                    "sessions": sessions
                }))
                continue

            if data.get("type") == "load_session" and session_manager:
                session_id = data.get("session_id")
                messages = session_manager.get_session_messages(session_id)
                current_session_id = session_id

                from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

                # Preserve system prompt if it exists
                system_msg = None
                for msg in conversation_state.get("messages", []):
                    if isinstance(msg, SystemMessage):
                        system_msg = msg
                        break

                conversation_state["messages"] = []
                if system_msg:
                    conversation_state["messages"].append(system_msg)

                conversation_state["session_id"] = session_id

                MAX_MESSAGE_HISTORY = int(os.getenv("MAX_MESSAGE_HISTORY", "30"))

                for msg in messages[-MAX_MESSAGE_HISTORY:]:
                    if msg["role"] == "system":
                        conversation_state["messages"].append(SystemMessage(content=msg["text"]))
                    elif msg["role"] == "user":
                        conversation_state["messages"].append(HumanMessage(content=msg["text"]))
                    elif msg["role"] == "assistant":
                        conversation_state["messages"].append(AIMessage(content=msg["text"]))

                logger.info(f"📥 Loaded {len(conversation_state['messages'])} messages from session {session_id}")

                await websocket.send(json.dumps({
                    "type": "session_loaded",
                    "session_id": session_id,
                    "messages": messages
                }))
                continue

            if data.get("type") == "new_session":
                current_session_id = None
                conversation_state["messages"] = []
                if "session_id" in conversation_state:
                    del conversation_state["session_id"]
                logger.info("🆕 New session started - conversation history cleared")
                continue

            if data.get("type") == "rename_session" and session_manager:
                session_id = data.get("session_id")
                new_name = data.get("name", "").strip()
                if session_id and new_name:
                    session_manager.update_session_name(session_id, new_name)
                    await websocket.send(json.dumps({
                        "type": "session_renamed",
                        "session_id": session_id,
                        "name": new_name
                    }))
                continue

            if data.get("type") == "delete_session" and session_manager:
                session_id = data.get("session_id")
                if session_id:
                    session_manager.delete_session(session_id)
                    await websocket.send(json.dumps({
                        "type": "session_deleted",
                        "session_id": session_id
                    }))
                continue

            # ═══════════════════════════════════════════════════════════
            # IMMEDIATE STOP HANDLING - Always processed immediately
            # ═══════════════════════════════════════════════════════════
            if data.get("type") == "user" and data.get("text") == ":stop":
                import sys

                logger.warning("🛑 STOP SIGNAL ACTIVATED - Operations will halt at next checkpoint")
                request_stop()

                print("\n🛑 Stop requested - operation will halt at next checkpoint")
                print("   This may take a few seconds for the current step to complete.")
                print("   Watch for '🛑 Stopped' messages below.\n")
                sys.stdout.flush()

                await websocket.send(json.dumps({
                    "type": "assistant_message",
                    "text": "🛑 Stop requested - operation will halt at next checkpoint.\n\nThis may take a few seconds for the current step to complete."
                }))
                await websocket.send(json.dumps({"type": "complete", "stopped": True}))
                continue

            # ═══════════════════════════════════════════════════════════
            # Fast operations - process synchronously
            # ═══════════════════════════════════════════════════════════

            if data.get("type") == "subscribe_system_stats":
                SYSTEM_MONITOR_CLIENTS.add(websocket)
                await websocket.send(json.dumps({"type": "subscribed", "subscription": "system_stats"}))
                continue

            if data.get("type") == "unsubscribe_system_stats":
                SYSTEM_MONITOR_CLIENTS.discard(websocket)
                await websocket.send(json.dumps({"type": "unsubscribed", "subscription": "system_stats"}))
                continue

            if data.get("type") == "list_models":
                all_models = models_module.get_all_models()
                model_names = [m["name"] for m in all_models]
                last = models_module.load_last_model()
                await websocket.send(json.dumps({
                    "type": "models_list",
                    "models": model_names,
                    "all_models": all_models,
                    "last_used": last
                }))
                continue

            if data.get("type") == "history_request":
                history_payload = [
                    {"role": "user", "text": m.content} if isinstance(m, HumanMessage)
                    else {"role": "assistant", "text": m.content}
                    for m in conversation_state["messages"]
                ]
                await websocket.send(json.dumps({"type": "history_sync", "history": history_payload}))
                continue

            if data.get("type") == "metrics_request":
                try:
                    from client.metrics import prepare_metrics
                    metrics_data = prepare_metrics()
                except ImportError:
                    try:
                        from metrics import prepare_metrics
                        metrics_data = prepare_metrics()
                    except ImportError:
                        metrics_data = {
                            "agent": {"runs": 0, "errors": 0, "error_rate": 0, "avg_time": 0, "times": []},
                            "llm": {"calls": 0, "errors": 0, "avg_time": 0, "times": []},
                            "tools": {"total_calls": 0, "total_errors": 0, "per_tool": {}},
                            "overall_errors": 0
                        }
                await websocket.send(json.dumps({"type": "metrics_response", "metrics": metrics_data}))
                continue

            if data.get("type") == "switch_model":
                model_name = data.get("model")
                new_agent = await models_module.switch_model(
                    model_name, tools, logger,
                    create_agent_fn=create_langgraph_agent,
                    a2a_state=a2a_state
                )
                if new_agent is None:
                    await websocket.send(json.dumps({
                        "type": "model_error",
                        "message": f"Model '{model_name}' not loaded"
                    }))
                    continue
                agent_ref[0] = new_agent
                await websocket.send(json.dumps({"type": "model_switched", "model": model_name}))
                continue

            # ═══════════════════════════════════════════════════════════
            # User messages - Create background task for long operations
            # ═══════════════════════════════════════════════════════════
            if data.get("type") == "user" or "text" in data:
                original_prompt = data.get("text")
                prompt = original_prompt

                if data.get("session_id"):
                    current_session_id = data.get("session_id")

                from client.input_sanitizer import sanitize_user_input, sanitize_command

                if prompt.startswith(":"):
                    prompt = sanitize_command(prompt)
                else:
                    prompt = sanitize_user_input(prompt, preserve_markdown=True)

                if prompt.startswith(":a2a"):
                    result = await handle_a2a_commands(prompt, orchestrator)
                    if result:
                        await broadcast_message("assistant_message", {"text": result})
                        await websocket.send(json.dumps({"type": "complete", "stopped": False}))
                        continue

                if prompt.startswith(":multi"):
                    from client.commands import handle_multi_agent_commands
                    result = await handle_multi_agent_commands(prompt, orchestrator, multi_agent_state)
                    if result:
                        await broadcast_message("assistant_message", {"text": result})
                        await websocket.send(json.dumps({"type": "complete", "stopped": False}))
                        continue

                if prompt.startswith(":"):
                    handled, response, new_agent, new_model = await handle_command(
                        prompt, tools, model_name, conversation_state, models_module,
                        system_prompt, agent_ref=agent_ref,
                        create_agent_fn=lambda llm, t: agent_ref[0].__class__(llm, t),
                        logger=logger,
                        orchestrator=orchestrator,
                        multi_agent_state=multi_agent_state,
                        a2a_state=a2a_state,
                        mcp_agent=mcp_agent
                    )
                    if handled:
                        if response:
                            await broadcast_message("assistant_message", {"text": response})
                        if new_agent:
                            agent_ref[0] = new_agent
                        if new_model:
                            model_name = new_model
                        await websocket.send(json.dumps({"type": "complete", "stopped": False}))
                        continue

                if session_manager and not prompt.startswith(":"):
                    if not current_session_id:
                        current_session_id = session_manager.create_session()
                        await websocket.send(json.dumps({
                            "type": "session_created",
                            "session_id": current_session_id
                        }))

                    MAX_MESSAGE_HISTORY = int(os.getenv('MAX_MESSAGE_HISTORY', 30))
                    session_manager.add_message(current_session_id, "user", prompt, MAX_MESSAGE_HISTORY, model=None)

                    messages = session_manager.get_session_messages(current_session_id)
                    if len(messages) == 1:
                        try:
                            text = prompt
                            name = text.split('.')[0] if '.' in text else text
                            name = name[:50] + '...' if len(name) > 50 else name
                            session_manager.update_session_name(current_session_id, name)
                            await websocket.send(json.dumps({
                                "type": "session_name_updated",
                                "session_id": current_session_id,
                                "name": name
                            }))
                        except Exception as e:
                            logger.error(f"Failed to generate session name: {e}")

                if current_task and not current_task.done():
                    logger.warning("⚠️ Cancelling previous task")
                    current_task.cancel()

                current_task = asyncio.create_task(
                    process_query(websocket, prompt, original_prompt, agent_ref, conversation_state,
                                  run_agent_fn, logger, tools, session_manager, current_session_id, system_prompt)
                )

    finally:
        if current_task and not current_task.done():
            current_task.cancel()
        CONNECTED_WEBSOCKETS.discard(websocket)
        SYSTEM_MONITOR_CLIENTS.discard(websocket)


async def start_websocket_server(agent, tools, logger, conversation_state, run_agent_fn, models_module,
                                 model_name, system_prompt, orchestrator=None, multi_agent_state=None,
                                 a2a_state=None, mcp_agent=None, session_manager=None, host="0.0.0.0", port=8765):
    """Start the WebSocket server for chat (WITH MULTI-AGENT STATE + A2A + SESSIONS)"""

    async def handler(websocket):
        try:
            await websocket_handler(
                websocket, [agent], tools, logger, conversation_state, run_agent_fn,
                models_module, model_name, system_prompt,
                orchestrator=orchestrator,
                multi_agent_state=multi_agent_state,
                a2a_state=a2a_state,
                mcp_agent=mcp_agent,
                session_manager=session_manager
            )
        except websockets.exceptions.ConnectionClosed:
            pass

    server = await websockets.serve(handler, host, port)

    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        logger.info(f"🌐 WebSocket listening on {host}:{port}")
        logger.info(f"   Local: ws://localhost:{port}")
        logger.info(f"   Network: ws://{local_ip}:{port}")
    except:
        logger.info(f"🌐 WebSocket server at ws://{host}:{port}")

    return server


async def start_log_websocket_server(log_handler_fn, host="0.0.0.0", port=8766):
    """Start a separate WebSocket server for log streaming"""
    import logging

    server = await websockets.serve(log_handler_fn, host, port)

    logger = logging.getLogger("mcp_client")
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        logger.info(f"📊 Log WebSocket listening on {host}:{port}")
        logger.info(f"   Local: ws://localhost:{port}")
        logger.info(f"   Network: ws://{local_ip}:{port}")
    except:
        logger.info(f"📊 Log WebSocket server at ws://{host}:{port}")

    return server


def get_system_monitor_clients():
    """Get the set of system monitor WebSocket clients"""
    return SYSTEM_MONITOR_CLIENTS


def is_system_monitor_available():
    """Check if system monitor is available"""
    return SYSTEM_MONITOR_AVAILABLE