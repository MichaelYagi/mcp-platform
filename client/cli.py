"""
CLI Module (WITH MULTI-AGENT SUPPORT + A2A - FIXED STATE + REAL-TIME STOP)
Handles command-line interface and user input
"""

import asyncio
import threading
import sys

from prompt_toolkit import prompt
from queue import Queue
from client.websocket import broadcast_message
from client.commands import handle_command, get_commands_list, handle_a2a_commands, handle_multi_agent_commands
from client.stop_signal import request_stop


def list_commands():
    """Print available CLI commands"""
    for line in get_commands_list():
        print(line)


def input_thread(input_queue, stop_event):
    """Thread to handle blocking input() calls"""
    while not stop_event.is_set():
        try:
            query = prompt("> ")
            input_queue.put(query)
        except (EOFError, KeyboardInterrupt):
            break

async def cli_input_loop(agent, logger, tools, model_name, conversation_state, run_agent_fn, models_module,
                         system_prompt, create_agent_fn, orchestrator=None, multi_agent_state=None, a2a_state=None,
                         mcp_agent=None):
    """Handle CLI input using a separate thread (with multi-agent + A2A support + REAL-TIME STOP)"""
    input_queue = Queue()
    stop_event = threading.Event()

    thread = threading.Thread(target=input_thread, args=(input_queue, stop_event), daemon=True)
    thread.start()

    # Track current running agent task
    current_agent_task = None

    try:
        while True:
            await asyncio.sleep(0.01)  # Fast polling

            # Check if model changed (from Web UI switch)
            current_last_model = models_module.load_last_model()
            if current_last_model and current_last_model != model_name:
                logger.info(f"🔄 Detected model change: {model_name} → {current_last_model}")
                new_agent, new_model = await models_module.reload_current_model(
                    tools, logger, create_agent_fn, a2a_state
                )
                if new_agent:
                    agent = new_agent
                    model_name = new_model
                    logger.info(f"✅ CLI synced to: {model_name}")

            if not input_queue.empty():
                query = input_queue.get().strip()

                # ═══════════════════════════════════════════════════════════
                # PRIORITY: Handle :stop IMMEDIATELY - even during execution
                # ═══════════════════════════════════════════════════════════
                if query == ":stop":
                    request_stop()
                    print("\n🛑 Stop requested - operation will halt at next checkpoint")
                    print("   This may take a few seconds for the current step to complete.")
                    print("   Watch for '🛑 Stopped' messages below.\n")
                    sys.stdout.flush()
                    await broadcast_message("assistant_message", {"text": "🛑 Stop requested"})

                    # If there's a running task, don't wait for it - just continue
                    # The stop signal will be picked up by the agent
                    continue

                if not query:
                    continue

                # Don't accept new queries while one is running
                if current_agent_task and not current_agent_task.done():
                    print("⚠️  Please wait for current operation to complete or type :stop")
                    continue

                # Handle A2A commands first
                if query.startswith(":a2a"):
                    result = await handle_a2a_commands(query, orchestrator)
                    if result:
                        print(result)
                        await broadcast_message("assistant_message", {"text": result})
                    continue

                # Handle multi-agent commands
                if query.startswith(":multi"):
                    result = await handle_multi_agent_commands(query, orchestrator, multi_agent_state)
                    if result:
                        print(result)
                        await broadcast_message("assistant_message", {"text": result})
                    continue

                # Handle other commands
                if query.startswith(":"):
                    handled, response, new_agent, new_model = await handle_command(
                        query,
                        tools,
                        model_name,
                        conversation_state,
                        models_module,
                        system_prompt,
                        agent_ref=[agent],
                        create_agent_fn=create_agent_fn,
                        logger=logger,
                        orchestrator=orchestrator,
                        multi_agent_state=multi_agent_state,
                        a2a_state=a2a_state,
                        mcp_agent=mcp_agent
                    )

                    if handled:
                        if response:
                            print(response)
                            await broadcast_message("assistant_message", {"text": response})
                        if new_agent:
                            agent = new_agent
                        if new_model:
                            model_name = new_model
                        continue

                logger.info(f"💬 Received query: '{query}'")

                print(f"\n> {query}")

                await broadcast_message("user_message", {"text": query})

                # ═══════════════════════════════════════════════════════════
                # RUN AGENT AS BACKGROUND TASK (non-blocking)
                # This allows the CLI loop to continue and process :stop
                # ═══════════════════════════════════════════════════════════
                async def run_and_display():
                    try:
                        result = await run_agent_fn(agent, conversation_state, query, logger, tools)

                        final_message = result["messages"][-1]
                        assistant_text = final_message.content

                        print("\n" + assistant_text + "\n")
                        logger.info("✅ Query completed successfully")

                        await broadcast_message("assistant_message", {
                            "text": assistant_text,
                            "multi_agent": result.get("multi_agent", False),
                            "a2a": result.get("a2a", False)
                        })
                    except Exception as e:
                        logger.error(f"❌ Error in agent execution: {e}")
                        import traceback
                        traceback.print_exc()

                # Start the task but DON'T await it - let it run in background
                current_agent_task = asyncio.create_task(run_and_display())

    except KeyboardInterrupt:
        print("\n👋 Exiting.")
    finally:
        stop_event.set()