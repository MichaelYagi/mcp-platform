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
from langchain_core.messages import ToolMessage

from client.commands import handle_command, handle_a2a_commands
from client.langgraph import create_langgraph_agent
from client.stop_signal import request_stop

try:
    from tools.rag.conversation_rag import store_turn_async as _rag_store_turn, purge_session as _rag_purge_session, retrieve_context as _rag_search_turns
    _CONV_RAG_AVAILABLE = True
except ImportError:
    _CONV_RAG_AVAILABLE = False
    def _rag_store_turn(*args, **kwargs): pass
    def _rag_purge_session(*args, **kwargs): pass
    def _rag_search_turns(*args, **kwargs): return []

# Import system monitor conditionally
try:
    from tools.system_monitor import system_monitor_loop
    SYSTEM_MONITOR_AVAILABLE = True
except ImportError:
    SYSTEM_MONITOR_AVAILABLE = False

CONNECTED_WEBSOCKETS = set()
SYSTEM_MONITOR_CLIENTS = set()
# Per-session tracking: set of session_ids currently processing a query.
# Replaces the single IS_PROCESSING bool which blocked ALL connections
# when any one tab was busy. Each connection now only blocks itself.
PROCESSING_SESSIONS = set()
# Module-level task registry: session_id -> asyncio.Task
# Tasks live here instead of inside websocket_handler so they survive
# connection drops. A reconnecting client picks up the running task.
SESSION_TASKS: dict = {}

# Per-session locks — serializes load_session adoption so two rapid
# reconnects cannot both adopt the same task simultaneously.
SESSION_LOCKS: dict = {}

# Timestamp of when each task was created — used by TTL cleanup.
SESSION_TASK_CREATED: dict = {}


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
    # Keep a reference to the outer conversation_state so we can write
    # updated history back after each call. The snapshot below is a working
    # copy that protects against reconnect-triggered rebuilds mid-run.
    outer_conversation_state = conversation_state
    # Snapshot conversation messages immediately so a reconnect that rebuilds
    # conversation_state["messages"] cannot mutate what this task reads.
    conversation_state = dict(conversation_state)
    conversation_state["messages"] = list(conversation_state.get("messages", []))
    if session_id:
        PROCESSING_SESSIONS.add(session_id)
    try:
        print(f"\n> {original_prompt}")
        await broadcast_message("user_message", {"text": original_prompt})

        # Add session_id to conversation_state for context tracking
        if session_id and "session_id" not in conversation_state:
            conversation_state["session_id"] = session_id

        # ═══════════════════════════════════════════════════════════════
        # WORKAROUND: Intercept history questions for weaker models (7B)
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

            # Persist before delivery
            if session_manager and session_id:
                MAX_MESSAGE_HISTORY = int(os.getenv('MAX_MESSAGE_HISTORY', 30))
                model_name = "direct-answer"
                msg_id = session_manager.add_message(session_id, "assistant", response_text, MAX_MESSAGE_HISTORY, model_name)
                _rag_store_turn(session_id, "assistant", response_text, msg_id)

            try:
                await broadcast_message("assistant_message", {
                    "text": response_text,
                    "multi_agent": False,
                    "a2a": False,
                    "model": "direct-answer"
                })
                await broadcast_message("complete", {"stopped": False})
            except Exception as send_err:
                logger.warning(f"⚠️ Direct-answer delivery failed (session {session_id}): {send_err}")

            return  # Exit early - don't call LLM

        # ═══════════════════════════════════════════════════════════════
        # Normal flow - Run agent (langgraph will preserve SystemMessage)
        # ═══════════════════════════════════════════════════════════════
        msgs_before = len(conversation_state["messages"])
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
        import re
        assistant_text = re.sub(r'<\|[^|]+\|>', '', assistant_text).strip()

        # Extract image_base64 from the ToolMessage in message history.
        # The final AIMessage is now the vision description text, not JSON,
        # so we need to look back at the tool result that triggered the vision call.
        # Only scan messages added in THIS run — msgs_before was captured before run_agent_fn.
        image_b64 = None
        image_source = None
        place_name = None
        from langchain_core.messages import ToolMessage
        new_messages = result["messages"][msgs_before:]
        tool_messages = [m for m in reversed(new_messages) if isinstance(m, ToolMessage)]
        logger.info(f"🖼️ Scanning {len(tool_messages)} new ToolMessage(s) for image/location data")
        for msg in tool_messages:
            raw = msg.content if isinstance(msg.content, str) else ""
            if "TextContent" in raw:
                idx = raw.find("text='")
                if idx != -1:
                    raw = raw[idx + 6:]
                    depth, end, in_str, esc = 0, -1, False, False
                    for i, ch in enumerate(raw):
                        if esc: esc = False; continue
                        if ch == '\\': esc = True; continue
                        if ch == '"': in_str = not in_str
                        if not in_str:
                            if ch == '{': depth += 1
                            elif ch == '}':
                                depth -= 1
                                if depth == 0: end = i; break
                    if end != -1:
                        raw = raw[:end + 1]
                    try:
                        raw = raw.encode('raw_unicode_escape').decode('unicode_escape')
                    except Exception:
                        pass
            try:
                tool_data = json.loads(raw)
                if not isinstance(tool_data, dict):
                    continue

                # Pick up placeName from any ToolMessage that has it
                if not place_name and tool_data.get("placeName"):
                    place_name = tool_data["placeName"]

                # Pick up image from first ToolMessage that has image_base64 or image_source
                if image_b64 is None and (tool_data.get("image_base64") or tool_data.get("image_source")):
                    image_source = tool_data.get("image_source")
                    b64 = tool_data.get("image_base64")
                    if b64:
                        image_b64 = b64.split(",", 1)[1] if "," in b64 else b64
                    elif image_source:
                        try:
                            import httpx as _httpx, base64 as _b64
                            fetch_headers = {}
                            shashin_key = os.getenv("SHASHIN_API_KEY", "")
                            if shashin_key and ("192.168." in image_source or "shashin" in image_source.lower()):
                                fetch_headers = {"x-api-key": shashin_key, "Content-Type": "application/json"}
                            async with _httpx.AsyncClient(timeout=30.0) as hc:
                                img_resp = await hc.get(image_source, headers=fetch_headers)
                                img_resp.raise_for_status()
                            image_b64 = _b64.b64encode(img_resp.content).decode("utf-8")
                            logger.info(f"🖼️ Fetched image for UI display from {image_source}, length={len(image_b64)}")
                        except Exception as fetch_err:
                            logger.warning(f"🖼️ Failed to fetch image for UI: {fetch_err}")

            except Exception as parse_err:
                logger.warning(f"🖼️ JSON parse failed: {parse_err}")

        logger.info(f"🖼️ image_b64={'set' if image_b64 else 'None'}, source={image_source}, place={place_name}")

        # Write updated history back to the outer conversation_state so
        # the next call sees the accumulated Human+AI pairs.
        # Only update messages — leave other keys (session_id etc.) intact.
        outer_conversation_state["messages"] = conversation_state["messages"]

        # CLI: truncate any stray base64 blobs before printing
        print_text = re.sub(r'[A-Za-z0-9+/]{100,}={0,2}', '[base64 data]', assistant_text)
        print("\n" + print_text + "\n")

        # Persist FIRST — response is safe in SQLite before delivery attempt.
        # If the socket dies mid-send, the message is already saved and
        # broadcast will reach any reconnected socket in CONNECTED_WEBSOCKETS.
        if session_manager and session_id:
            MAX_MESSAGE_HISTORY = int(os.getenv('MAX_MESSAGE_HISTORY', 30))
            model_name = result.get("current_model", "unknown")
            msg_id = session_manager.add_message(session_id, "assistant", assistant_text, MAX_MESSAGE_HISTORY, model_name)
            _rag_store_turn(session_id, "assistant", assistant_text, msg_id)
            if image_source:
                session_manager.set_message_image_source(msg_id, image_source)

        # Broadcast to ALL connected sockets — covers reconnect case where
        # original socket died but new socket is already in CONNECTED_WEBSOCKETS
        try:
            await broadcast_message("assistant_message", {
                "text": assistant_text,
                "image": image_b64,
                "multi_agent": result.get("multi_agent", False),
                "a2a": result.get("a2a", False),
                "model": result.get("current_model", "unknown")
            })
            await broadcast_message("complete", {"stopped": result.get("stopped", False)})
        except Exception as send_err:
            logger.warning(f"⚠️ Delivery failed (session {session_id}): {send_err}")

    except Exception as e:
        logger.error(f"❌ Error processing query: {e}")
        import traceback
        traceback.print_exc()

        await broadcast_message("error", {"text": str(e)})
        await broadcast_message("complete", {"stopped": False})
    finally:
        if session_id:
            PROCESSING_SESSIONS.discard(session_id)
            SESSION_TASKS.pop(session_id, None)
            SESSION_TASK_CREATED.pop(session_id, None)
            # Remove lock only if no task is pending — keep it if a new
            # query already acquired it for this session.
            SESSION_LOCKS.pop(session_id, None)

async def websocket_handler(websocket, agent_ref, tools, logger, conversation_state, run_agent_fn,
                            models_module, model_name, system_prompt, orchestrator=None,
                            multi_agent_state=None, a2a_state=None, mcp_agent=None, session_manager=None):
    """
    Handle WebSocket connections with TRUE concurrent processing

    KEY: Long operations run as background tasks, allowing :stop to be processed immediately
    """
    global json
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

                # Re-adopt any in-flight task for this session.
                # Use a per-session Lock to prevent two rapid reconnects from
                # both adopting the same task simultaneously (race condition).
                if session_id not in SESSION_LOCKS:
                    SESSION_LOCKS[session_id] = asyncio.Lock()
                async with SESSION_LOCKS[session_id]:
                    in_flight = SESSION_TASKS.get(session_id)
                    if in_flight and not in_flight.done():
                        current_task = in_flight
                        logger.info(f"🔁 Session {session_id}: re-attached to in-flight task on reconnect")

                # Orphan / status detection based on last message role
                if messages and messages[-1]["role"] == "user":
                    if session_id in PROCESSING_SESSIONS:
                        # Still computing — new socket is in CONNECTED_WEBSOCKETS,
                        # broadcast will deliver result automatically.
                        logger.info(f"⏳ Session {session_id}: in-flight, reconnected socket will receive broadcast")
                    else:
                        # Task finished but socket was dead — response lost
                        logger.warning(f"⚠️ Session {session_id} ends on user message with no active processing")
                        await websocket.send(json.dumps({
                            "type": "assistant_message",
                            "text": "⚠️ It looks like your previous message may not have received a response. "
                                     "Please resend it.",
                            "model": "system"
                        }))
                        await websocket.send(json.dumps({"type": "complete", "stopped": False}))
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
                    _rag_purge_session(session_id)
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
                # Cancel via SESSION_TASKS so the correct task is always hit
                # regardless of whether current_task matches (e.g. after reconnect).
                if current_session_id and current_session_id in SESSION_TASKS:
                    t = SESSION_TASKS.get(current_session_id)
                    if t and not t.done():
                        t.cancel()
                        logger.info(f"🛑 Cancelled SESSION_TASKS task for session {current_session_id}")

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

            if data.get("type") == "list_tools":
                try:
                    from pathlib import Path as _Path
                    from client.tool_utils import resolve_tool_server, load_external_server_names

                    # Authoritative disabled-tool check via tool_control
                    _is_tool_enabled_fn = None
                    try:
                        from tools.tool_control import is_tool_enabled as _is_tool_enabled_fn
                    except ImportError:
                        pass

                    # Fallback: parse DISABLED_TOOLS env var when tool_control unavailable
                    _disabled_raw = os.getenv("DISABLED_TOOLS", "")
                    _disabled_entries = [e.strip() for e in _disabled_raw.split(",") if ":" in e.strip()]

                    def _is_disabled(tool_name, source_server):
                        if _is_tool_enabled_fn is not None:
                            try:
                                # Check 1: no category — catches simple "tool_name" entries
                                if not _is_tool_enabled_fn(tool_name):
                                    return True
                                # Check 2: source_server as category — catches "plex:*", "todo:*"
                                # source_server is the directory name which matches the
                                # category used in DISABLED_TOOLS for most servers
                                if not _is_tool_enabled_fn(tool_name, source_server):
                                    return True
                                return False
                            except Exception:
                                pass
                        # Fallback if import failed
                        src = (source_server or "").lower()
                        for entry in _disabled_entries:
                            cat, tname = entry.split(":", 1)
                            if cat.strip().lower() == src:
                                if tname.strip() in ("*", tool_name):
                                    return True
                        return False

                    # Infrastructure tools present on every server — never user-facing
                    _INTERNAL_TOOLS = {"list_skills", "read_skill", "parse_github_url"}

                    _project_root = _Path(__file__).resolve().parent.parent
                    _external_names = load_external_server_names(_project_root)

                    # resolve_tool_server gives authoritative tool→server mapping
                    # (same logic used by :tools command)
                    _tool_to_server = await resolve_tool_server(tools, mcp_agent, _project_root)

                    tools_payload = []
                    seen_names = set()
                    for tool in tools:
                        if tool.name in _INTERNAL_TOOLS:
                            continue
                        if tool.name in seen_names:
                            continue
                        seen_names.add(tool.name)

                        source = _tool_to_server.get(tool.name, "unknown")
                        if _is_disabled(tool.name, source):
                            continue

                        tools_payload.append({
                            "name": tool.name,
                            "description": (tool.description or "").strip(),
                            "source_server": source,
                            "external": source in _external_names,
                        })

                    await websocket.send(json.dumps({
                        "type": "tools_list",
                        "tools": tools_payload,
                    }))
                except Exception as _e:
                    logger.error(f"❌ list_tools failed: {_e}")
                    await websocket.send(json.dumps({"type": "tools_list", "tools": []}))
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

                # Block new prompts while THIS connection's task is in flight
                # Uses current_task (per-connection) not PROCESSING_SESSIONS (global)
                # so other tabs are never affected by this connection's state
                this_session_busy = (
                    current_task is not None and
                    not current_task.done() and
                    original_prompt != ":stop"
                )
                if this_session_busy:
                    await websocket.send(json.dumps({
                        "type": "assistant_message",
                        "text": "⏳ A response is already being processed. Please wait, or send `:stop` to cancel.",
                        "model": None
                    }))
                    await websocket.send(json.dumps({"type": "complete", "stopped": False}))
                    continue

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
                    msg_id = session_manager.add_message(current_session_id, "user", prompt, MAX_MESSAGE_HISTORY, model=None)
                    # Store user turn in session-scoped RAG for semantic context retrieval
                    _rag_store_turn(current_session_id, "user", prompt, msg_id)

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

                # Cancel only if a task is already running for THIS session
                existing = SESSION_TASKS.get(current_session_id)
                if existing and not existing.done():
                    logger.warning("⚠️ Cancelling previous task for session")
                    existing.cancel()

                current_task = asyncio.create_task(
                    process_query(websocket, prompt, original_prompt, agent_ref, conversation_state,
                                  run_agent_fn, logger, tools, session_manager, current_session_id, system_prompt)
                )
                # Store at module level so task survives connection drops.
                # Record creation time for TTL cleanup of abandoned tasks.
                if current_session_id:
                    SESSION_TASKS[current_session_id] = current_task
                    import time as _time
                    SESSION_TASK_CREATED[current_session_id] = _time.monotonic()
                    # Ensure a lock exists for this session
                    if current_session_id not in SESSION_LOCKS:
                        SESSION_LOCKS[current_session_id] = asyncio.Lock()

    finally:
        # Do NOT cancel the task here — it may still be computing a response.
        # Tasks are only cancelled by :stop or when a new query replaces them.
        # The task will broadcast its result to any reconnected socket.
        if current_task and not current_task.done():
            logger.info(f"🔌 Connection closed but task still running for session "
                        f"{current_session_id} — keeping alive")
        CONNECTED_WEBSOCKETS.discard(websocket)
        SYSTEM_MONITOR_CLIENTS.discard(websocket)


async def _cleanup_stale_tasks(max_age_seconds: int = 3600):
    """
    Periodically remove abandoned tasks from SESSION_TASKS.
    Handles the case where process_query raises before its finally block
    (e.g. unhandled exception in LangGraph internals) and never self-cleans.
    Runs every 5 minutes, removes tasks older than max_age_seconds (default 1hr).
    """
    import time as _time
    while True:
        await asyncio.sleep(300)  # check every 5 minutes
        now = _time.monotonic()
        stale = [
            sid for sid, created in list(SESSION_TASK_CREATED.items())
            if (now - created) > max_age_seconds
        ]
        for sid in stale:
            task = SESSION_TASKS.get(sid)
            if task and not task.done():
                logger.warning(f"🧹 TTL cleanup: cancelling stale task for session {sid} "
                               f"(age={(now - SESSION_TASK_CREATED[sid]):.0f}s)")
                task.cancel()
            SESSION_TASKS.pop(sid, None)
            SESSION_TASK_CREATED.pop(sid, None)
            SESSION_LOCKS.pop(sid, None)
            PROCESSING_SESSIONS.discard(sid)
        if stale:
            logger.info(f"🧹 TTL cleanup removed {len(stale)} stale session entries")


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

    # ping_interval keeps the TCP connection alive through sleep/idle.
    # ping_timeout gives the client 60s to respond before dropping.
    server = await websockets.serve(handler, host, port,
                                    ping_interval=30, ping_timeout=60)
    asyncio.create_task(_cleanup_stale_tasks())

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