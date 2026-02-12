"""
MCP Client - Main Entry Point (WITH MULTI-AGENT INTEGRATION + MULTI-A2A SUPPORT)
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from mcp_use.client.client import MCPClient
from mcp_use.agents.mcpagent import MCPAgent
from client.distributed_skills_manager import (
    DistributedSkillsManager,
    inject_relevant_skills_into_messages
)

# Import client modules
from client import logging_handler, langgraph, models, websocket, cli, utils

from client.a2a_client import A2AClient
from client.a2a_mcp_bridge import make_a2a_tool
from client.context_tracker import integrate_context_tracking

# Import multi-agent system
try:
    from client.multi_agent import MultiAgentOrchestrator, should_use_multi_agent
    MULTI_AGENT_AVAILABLE = True
except ImportError:
    print("⚠️ Multi-agent system not available. Add multi_agent.py to client/ directory.")
    MULTI_AGENT_AVAILABLE = False

# Import system monitor conditionally
try:
    from tools.system_monitor import system_monitor_loop
    SYSTEM_MONITOR_AVAILABLE = True
except ImportError:
    SYSTEM_MONITOR_AVAILABLE = False
    print("⚠️  System monitor not available. Install with: pip install psutil gputil nvidia-ml-py3")

# Load environment variables
PROJECT_ROOT = Path(__file__).parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

# Configuration
MAX_MESSAGE_HISTORY = int(os.getenv("MAX_MESSAGE_HISTORY", "20"))

# Shared multi-agent state (mutable dict so changes propagate)
MULTI_AGENT_STATE = {
    # "enabled": MULTI_AGENT_AVAILABLE and os.getenv("MULTI_AGENT_ENABLED", "false").lower() == "true"
    "enabled": True
}

A2A_STATE = {
    "enabled": False,
    "endpoints": []  # Track successfully registered endpoints
}

# Default system prompt - will be overridden if tool_usage_guide.md exists
SYSTEM_PROMPT = """# SYSTEM INSTRUCTION: YOU ARE A TOOL-USING AGENT

CRITICAL RULES:
1. ALWAYS respond in ENGLISH only
2. Read the user's intent carefully before choosing a tool
3. DO NOT make multiple redundant tool calls

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL: YOU HAVE FULL ACCESS TO CONVERSATION HISTORY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DO NOT say "I don't have access to history" - YOU DO HAVE ACCESS.
The message list contains the FULL conversation history in chronological order.

WHEN USER ASKS: "what was my last prompt"
→ Look at the most recent HumanMessage before the current one
→ Respond: "Your last prompt was: [exact text]"

Example:
User: "what's the weather?"
You: "Sunny, 22°C"
User: "what was my last prompt"
You: "Your last prompt was: what's the weather?"  ← DO THIS

TOOL SELECTION:

"add to my todo" → use add_todo_item (NOT rag_search_tool)
"remember this" → use rag_add_tool
"find a movie" → use semantic_media_search_text
"using the RAG tool" → use rag_search_tool (ONE search only)

EXAMPLES:

User: "add to my todo due tomorrow, make breakfast"
CORRECT: add_todo_item(title="make breakfast", due_by="[tomorrow date]")
WRONG: rag_search_tool(query="make breakfast") ❌

User: "remember that password is abc123"
CORRECT: rag_add_tool(text="password is abc123", source="notes")
WRONG: add_todo_item(title="password is abc123") ❌

VERIFICATION:
- "add to my todo" = add_todo_item
- "remember" = rag_add_tool
- "find movie" = semantic_media_search_text

Read the user's message carefully and call the RIGHT tool."""

# Global conversation state
GLOBAL_CONVERSATION_STATE = {
    "messages": [],
    "loop_count": 0
}


# ═════════════════════════════════════════════════════════════════════
# A2A MULTI-ENDPOINT SUPPORT
# ═════════════════════════════════════════════════════════════════════

def parse_a2a_endpoints():
    """Parse A2A endpoints from environment variables - supports both single and multiple"""
    endpoints = []

    # Check for multiple endpoints first (comma-separated)
    endpoints_str = os.getenv("A2A_ENDPOINTS", "").strip()
    if endpoints_str:
        endpoints = [ep.strip() for ep in endpoints_str.split(",") if ep.strip()]

    # Backward compatibility: single endpoint
    if not endpoints:
        single_endpoint = os.getenv("A2A_ENDPOINT", "").strip()
        if single_endpoint:
            endpoints = [single_endpoint]

    return endpoints


async def register_a2a_tools(mcp_agent, base_url: str, logger) -> bool:
    """
    Discover remote A2A tools and register them as MCP tools.
    Handles connection failures gracefully.

    Returns:
        bool: True if tools were successfully registered, False otherwise
    """
    try:
        a2a = A2AClient(base_url)
        capabilities = await a2a.discover()

    except Exception as e:
        logger.error(f"⚠️ A2A connection failed: {e}")
        return False  # Return failure status

    # If discovery succeeded, register tools
    tool_count = 0
    for tool_def in capabilities.get("tools", []):
        tool = make_a2a_tool(a2a, tool_def)
        mcp_agent._tools.append(tool)
        tool_count += 1

    return tool_count > 0  # Return success if at least one tool was registered

async def register_all_a2a_endpoints(mcp_agent, logger):
    """Register tools from all A2A endpoints"""
    endpoints = parse_a2a_endpoints()

    if not endpoints:
        logger.info("ℹ️  No A2A endpoints configured")
        return {
            "endpoints": [],
            "successful": [],
            "failed": [],
            "total_tools_added": 0
        }

    logger.info(f"🌐 Attempting to register {len(endpoints)} A2A endpoint(s)")

    successful = []
    failed = []
    initial_tool_count = len(mcp_agent._tools)  # ← SAVE INITIAL COUNT (don't modify this)

    for i, endpoint in enumerate(endpoints, 1):
        logger.info(f"   [{i}/{len(endpoints)}] Connecting to: {endpoint}")

        try:
            tools_before_this = len(mcp_agent._tools)  # ← Track before THIS endpoint
            success = await register_a2a_tools(mcp_agent, endpoint, logger)

            if success:
                successful.append(endpoint)
                tools_after_this = len(mcp_agent._tools)
                new_tools_this_endpoint = tools_after_this - tools_before_this
                logger.info(f"   ✅ [{i}/{len(endpoints)}] Registered successfully (+{new_tools_this_endpoint} tools)")
            else:
                failed.append(endpoint)
                logger.warning(f"   ❌ [{i}/{len(endpoints)}] Registration failed")

        except Exception as e:
            failed.append(endpoint)
            logger.error(f"   ❌ [{i}/{len(endpoints)}] Error: {e}")
            import traceback
            traceback.print_exc()

    # Calculate total new tools: current count - initial count
    final_tool_count = len(mcp_agent._tools)
    total_new_tools = final_tool_count - initial_tool_count

    result = {
        "endpoints": endpoints,
        "successful": successful,
        "failed": failed,
        "total_tools_added": total_new_tools
    }

    # Summary
    logger.info("=" * 60)
    logger.info(f"🔌 A2A Registration Summary:")
    logger.info(f"   Total endpoints configured: {len(endpoints)}")
    logger.info(f"   Successfully registered: {len(successful)}")
    logger.info(f"   Failed to register: {len(failed)}")
    logger.info(f"   New A2A tools added: {total_new_tools}")
    logger.info(f"   Total tools now available: {final_tool_count}")

    if successful:
        logger.info(f"   Active A2A endpoints:")
        for endpoint in successful:
            logger.info(f"      ✓ {endpoint}")

    if failed:
        logger.info(f"   Failed endpoints:")
        for endpoint in failed:
            logger.info(f"      ✗ {endpoint}")

    logger.info("=" * 60)

    return result

# ═════════════════════════════════════════════════════════════════════
# MCP SERVER AUTO-DISCOVERY
# ═════════════════════════════════════════════════════════════════════

def is_wsl2():
    """Check if running in WSL2"""
    try:
        with open("/proc/version", "r") as f:
            return "microsoft" in f.read().lower()
    except:
        return False


def convert_path_for_platform(path: str) -> str:
    """Convert WSL2 path to Windows path if needed"""
    if is_wsl2():
        return path  # Running in WSL2, use as-is

    # Running on Windows, convert /mnt/c/... to C:\...
    if path.startswith("/mnt/c/"):
        path = path.replace("/mnt/c/", "C:\\")
        path = path.replace("/", "\\")

    return path


def convert_classpath_for_platform(classpath: str) -> str:
    """Convert Java classpath separators and paths for platform"""
    if is_wsl2():
        return classpath  # WSL2 uses : separator

    # Windows: split by :, convert each path, rejoin with ;
    paths = classpath.split(":")
    windows_paths = [convert_path_for_platform(p) for p in paths]
    return ";".join(windows_paths)

def auto_discover_servers(servers_dir: Path):
    """Auto-discover all servers by scanning servers/ directory"""
    mcp_servers = {}

    # Scan local servers/ directory
    for server_dir in servers_dir.iterdir():
        if server_dir.is_dir():
            server_file = server_dir / "server.py"
            if server_file.exists():
                server_name = server_dir.name
                mcp_servers[server_name] = {
                    "command": utils.get_venv_python(PROJECT_ROOT),
                    "args": [str(server_file)],
                    "cwd": str(PROJECT_ROOT),
                    "env": {"CLIENT_IP": utils.get_public_ip()}
                }

    # External servers from config file (outside the loop!)
    external_config = PROJECT_ROOT / "client" / "external_servers.json"
    if external_config.exists():
        try:
            config = json.loads(external_config.read_text(encoding="utf-8"))
            for name, cfg in config.get("external_servers", {}).items():
                if not cfg.get("enabled", True):
                    continue

                transport = cfg.get("transport", "sse")

                if transport == "stdio":
                    command = convert_path_for_platform(cfg["command"])
                    args = cfg.get("args", [])

                    # Skip if command doesn't exist on this platform
                    if not Path(command).exists():
                        print(f"⏭️  Skipping '{name}' - command not found: {command}")
                        continue

                    # If server bridges to a port, verify it's reachable before adding
                    port = cfg.get("env", {}).get("IJ_MCP_SERVER_PORT")
                    if port:
                        import socket
                        host = cfg.get("env", {}).get("IJ_MCP_SERVER_HOST", "127.0.0.1")
                        try:
                            with socket.create_connection((host, int(port)), timeout=2.0):
                                pass
                            print(f"✅ External stdio server added: {name} (port {port} reachable)")
                        except (ConnectionRefusedError, OSError) as e:
                            print(f"⏭️  Skipping '{name}' - port {port} not reachable: {e}")
                            continue

                    converted_args = []
                    for arg in args:
                        if ";" in arg and ".jar" in arg:
                            converted_args.append(convert_classpath_for_platform(arg))
                        else:
                            converted_args.append(arg)

                    mcp_servers[name] = {
                        "command": command,
                        "args": converted_args,
                        "env": cfg.get("env", {}),
                        "cwd": cfg.get("cwd", str(PROJECT_ROOT))
                    }
                    if not port:  # Already printed above if port check passed
                        print(f"✅ External stdio server added: {name}")

                elif transport == "sse":
                    mcp_servers[name] = {"url": cfg["url"], "transport": "sse"}
                    print(f"✅ External SSE server added: {name}")

        except Exception as e:
            print(f"⚠️  Failed to load external_servers.json: {e}")

    return mcp_servers


# ═════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════

async def main():
    global SYSTEM_PROMPT

    # Setup logging
    LOG_DIR = PROJECT_ROOT / "logs"
    LOG_DIR.mkdir(exist_ok=True)

    CLIENT_LOG_FILE = LOG_DIR / "mcp-client.log"
    SERVER_LOG_FILE = LOG_DIR / "mcp-server.log"

    logging_handler.setup_logging(CLIENT_LOG_FILE)
    logger = logging.getLogger("mcp_client")

    # Set event loop for logging
    logging_handler.set_event_loop(asyncio.get_running_loop())

    # Setup MCP client with auto-discovered servers
    mcp_servers = auto_discover_servers(PROJECT_ROOT / "servers")
    client = MCPClient.from_dict({
        "mcpServers": mcp_servers
    })

    logger.info(f"🔌 Discovered {len(mcp_servers)} MCP servers: {list(mcp_servers.keys())}")

    # Load system prompt from file if it exists
    system_prompt_path = PROJECT_ROOT / "prompts/tool_usage_guide.md"
    if system_prompt_path.exists():
        logger.info(f"⚙️ System prompt loaded from {system_prompt_path}")
        file_prompt = system_prompt_path.read_text(encoding="utf-8")
        # Append conversation history awareness to file-based prompt
        history_awareness = """

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CRITICAL: YOU HAVE FULL ACCESS TO CONVERSATION HISTORY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DO NOT say "I don't have access to history" - YOU DO HAVE ACCESS.

WHEN USER ASKS: "what was my last prompt"
→ Look at the most recent HumanMessage before the current one
→ Respond: "Your last prompt was: [exact text]"

Example:
User: "what's the weather?"
You: "Sunny, 22°C"  
User: "what was my last prompt"
You: "Your last prompt was: what's the weather?"  ← DO THIS"""
        SYSTEM_PROMPT = file_prompt + history_awareness
    else:
        logger.warning(f"⚠️  System prompt file not found at {system_prompt_path}, using default")

    # Log first 200 chars of system prompt for verification
    logger.info(f"📋 System prompt preview: {SYSTEM_PROMPT[:200]}...")

    # Import backend manager
    from client.llm_backend import LLMBackendManager, GGUFModelRegistry

    # Get all available models
    all_models = models.get_all_models()
    if not all_models:
        print("❌ No models available")
        print("   Ollama: ollama pull <model>")
        print("   GGUF: :gguf add <path>")
        sys.exit(1)

    # Start with configured backend
    backend = models.get_initial_backend()
    os.environ["LLM_BACKEND"] = backend
    logger.info(f"🔧 Backend: {backend}")

    # Initialize available_models variable
    available_models = []

    # Check backend-specific requirements
    if backend == "ollama":
        try:
            await utils.ensure_ollama_running()
            available_models = [m["name"] for m in all_models if m["backend"] == "ollama"]
            if not available_models:
                raise RuntimeError("No Ollama models installed")
        except RuntimeError as e:
            # Ollama not available - try GGUF fallback
            gguf_models = [m["name"] for m in all_models if m["backend"] == "gguf"]
            if gguf_models:
                logger.warning(f"⚠️ Ollama unavailable, switching to GGUF")
                backend = "gguf"
                os.environ["LLM_BACKEND"] = "gguf"
                available_models = gguf_models
            else:
                print(f"❌ {e}")
                print("💡 Start Ollama: ollama serve")
                print("   Or add GGUF models: :gguf add <path>")
                sys.exit(1)

    elif backend == "gguf":
        available_models = [m["name"] for m in all_models if m["backend"] == "gguf"]
        if not available_models:
            print("❌ No GGUF models. Add with: :gguf add <path>")
            sys.exit(1)

    # Select model from available_models
    model_name = available_models[0]
    last = models.load_last_model()
    if last and last in available_models:
        model_name = last

    models.save_last_model(model_name)
    logger.info(f"🤖 Using {backend}/{model_name}")

    # Initialize LLM
    llm = LLMBackendManager.create_llm(model_name, temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")))

    mcp_agent = MCPAgent(
        llm=llm,
        client=client,
        max_steps=10,
        system_prompt=SYSTEM_PROMPT
    )

    mcp_agent.debug = True
    try:
        await mcp_agent.initialize()
    except Exception as e:
        logger.error(f"❌ Some MCP servers failed to initialize: {e}")

    from client.session_manager import SessionManager
    session_manager = SessionManager()
    logger.info("💾 Session manager initialized")

    tools = mcp_agent._tools
    logger.info(f"🛠️  Local MCP tools loaded: {len(tools)}")

    # ═══════════════════════════════════════════════════════════════
    # DISTRIBUTED SKILLS DISCOVERY
    # ═══════════════════════════════════════════════════════════════

    skills_manager = DistributedSkillsManager(client)
    await skills_manager.discover_all_skills()

    if skills_manager.all_skills:
        skills_summary = skills_manager.get_skills_summary()
        SYSTEM_PROMPT = SYSTEM_PROMPT + "\n\n" + skills_summary
        logger.info(f"📚 System prompt enhanced with {len(skills_manager.all_skills)} distributed skill(s)")
    else:
        logger.warning("⚠️  No skills discovered from servers")
        skills_manager = None  # Disable if no skills found

    # ═════════════════════════════════════════════════════════════════
    # MULTI-A2A REGISTRATION (UPDATED)
    # ═════════════════════════════════════════════════════════════════

    a2a_result = await register_all_a2a_endpoints(mcp_agent, logger)

    if a2a_result["successful"]:
        tools = mcp_agent._tools
        logger.info(f"🔌 A2A integration complete. Total tools: {len(tools)}")
        A2A_STATE["enabled"] = True
        A2A_STATE["endpoints"] = a2a_result["successful"]  # Store successful endpoints
    else:
        logger.warning("⚠️ No A2A endpoints registered - continuing with local tools only")
        A2A_STATE["enabled"] = False
        A2A_STATE["endpoints"] = []

    # Log any failures
    if a2a_result["failed"]:
        logger.warning(f"⚠️  {len(a2a_result['failed'])} endpoint(s) failed to register")

    # Check if Plex server is available
    plex_server_available = "plex" in mcp_servers or "plex-server" in mcp_servers

    if plex_server_available:
        logger.info("🎬 Plex server detected - testing connection...")

        # Test Plex connection before attempting import
        plex_connected = False
        try:
            # Quick connection test using a simple tool
            test_tool = None
            for tool in tools:
                if hasattr(tool, 'name') and tool.name == "plex_get_stats":
                    test_tool = tool
                    break

            if test_tool:
                # Try to get stats - this will fail if Plex is unreachable
                test_result = await test_tool.ainvoke({})

                # Parse JSON if needed
                if isinstance(test_result, str):
                    import json
                    try:
                        test_result = json.loads(test_result)
                    except json.JSONDecodeError:
                        pass

                # Check if result is valid (not an error)
                if isinstance(test_result, str) and "error" not in test_result.lower():
                    plex_connected = True
                    logger.info("   ✅ Plex connection verified")
                elif isinstance(test_result, dict) and test_result.get("total_items", 0) >= 0:
                    plex_connected = True
                    logger.info("   ✅ Plex connection verified")
                else:
                    logger.warning(f"   ⚠️  Plex connection test returned unexpected result")
            else:
                logger.warning("   ⚠️  Connection test tool not found")

        except Exception as e:
            logger.warning(f"   ⚠️  Plex connection test failed: {e}")
            plex_connected = False

        # Only proceed if Plex is actually connected
        if plex_connected:
            # CHECK IF MODEL ALREADY EXISTS
            model_file = PROJECT_ROOT / "models" / "plex_recommender.pkl"

            if model_file.exists():
                logger.info("   ✅ Model exists")
                logger.info("   ⏭️  Skipping training")
            else:
                logger.info("   📥 No model - importing...")

                try:
                    # Find the import_plex_history and train_recommender tools
                    import_tool = None
                    train_tool = None

                    for tool in tools:
                        if hasattr(tool, 'name'):
                            if tool.name == "import_plex_history":
                                import_tool = tool
                            elif tool.name == "train_recommender":
                                train_tool = tool

                    if import_tool and train_tool:
                        # Step 1: Import Plex history
                        try:
                            import_result_raw = await import_tool.ainvoke({"limit": 3000})

                            # Extract JSON from TextContent string representation
                            import re
                            import json

                            # Pattern: text='JSON_HERE'
                            match = re.search(r"text='(.*?)'(?:,|\))", str(import_result_raw), re.DOTALL)

                            if match:
                                # Get the raw string (still escaped)
                                escaped_json = match.group(1)

                                # Use Python's string decoder to properly unescape
                                # This handles \n, \", etc. correctly
                                try:
                                    # Decode escape sequences properly
                                    import codecs
                                    json_str = codecs.decode(escaped_json, 'unicode_escape')

                                    # Now parse the JSON
                                    import_result = json.loads(json_str)
                                    logger.info(f"   ✅ Successfully parsed result")
                                except Exception as e:
                                    logger.error(f"   ❌ Failed to decode/parse: {e}")
                                    # Try a simpler approach - just replace common escapes
                                    json_str = escaped_json.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
                                    try:
                                        import_result = json.loads(json_str)
                                        logger.info(f"   ✅ Successfully parsed with fallback method")
                                    except:
                                        logger.error(f"   ❌ Both parsing methods failed")
                                        logger.error(f"   Raw: {escaped_json[:200]}")
                                        import_result = {}
                            else:
                                logger.error(f"   ❌ Could not extract JSON from result")
                                import_result = {}

                            # Parse result
                            if isinstance(import_result, dict):
                                imported = import_result.get("imported", 0)
                                total_views = import_result.get("total_views_now", 0)
                                can_train = import_result.get("can_train", False)

                                logger.info(f"   ✅ Imported {imported} viewing events (total: {total_views})")

                                # Step 2: Train if we have enough data
                                if can_train:
                                    logger.info("   🤖 Training ML model...")
                                    train_result_raw = await train_tool.ainvoke({})

                                    # Extract JSON from TextContent (same as import)
                                    import re
                                    import json
                                    import codecs

                                    match = re.search(r"text='(.*?)'(?:,|\))", str(train_result_raw), re.DOTALL)

                                    if match:
                                        escaped_json = match.group(1)

                                        try:
                                            json_str = codecs.decode(escaped_json, 'unicode_escape')
                                            train_result = json.loads(json_str)
                                            logger.info(f"   ✅ Successfully parsed train result")
                                        except Exception as e:
                                            logger.error(f"   ❌ Failed to decode/parse train result: {e}")
                                            json_str = escaped_json.replace('\\n', '\n').replace('\\"', '"').replace('\\\\',
                                                                                                                     '\\')
                                            try:
                                                train_result = json.loads(json_str)
                                                logger.info(f"   ✅ Parsed train result with fallback")
                                            except:
                                                logger.error(f"   ❌ Train result parsing failed completely")
                                                train_result = {}
                                    else:
                                        logger.error(f"   ❌ Could not extract JSON from train result")
                                        train_result = {}

                                    if isinstance(train_result, dict):
                                        if train_result.get("status") == "success":
                                            accuracy = train_result.get("train_accuracy", "N/A")
                                            samples = train_result.get("training_samples", 0)
                                            logger.info(f"   ✅ Model trained! Accuracy: {accuracy}, Samples: {samples}")
                                            logger.info("   🎯 ML recommendations ready!")
                                        else:
                                            logger.warning(
                                                f"   ⚠️  Training failed: {train_result.get('message', 'Unknown error')}")
                                else:
                                    needed = 20 - total_views
                                    logger.info(f"   ℹ️  Need {needed} more viewing events to train (minimum: 20)")
                            else:
                                logger.warning(f"   ⚠️  Unexpected import result format: {type(import_result)}")

                        except Exception as e:
                            logger.error(f"   ❌ Failed to import Plex history: {e}")
                            import traceback
                            traceback.print_exc()
                    else:
                        missing = []
                        if not import_tool:
                            missing.append("import_plex_history")
                        if not train_tool:
                            missing.append("train_recommender")
                        logger.warning(f"   ⚠️  ML tools not found: {', '.join(missing)}")

                except Exception as e:
                    logger.error(f"   ❌ ML auto-training failed: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            logger.info("   ℹ️  Plex connection unavailable - skipping ML auto-training")
            logger.info("   💡 Check PLEX_URL and PLEX_TOKEN in .env")
    else:
        logger.info("ℹ️  Plex server not detected - skipping ML auto-training")

    # ═════════════════════════════════════════════════════════════════

    llm_with_tools = llm.bind_tools(tools)

    # Test tool binding
    # logger.info("=" * 60)
    # logger.info("🧪 TESTING TOOL BINDING")
    # test_messages = [
    #     SystemMessage(content="You have access to tools. Call the semantic_media_search_text tool to find movies."),
    #     HumanMessage(content="find action movies")
    # ]
    # test_response = await llm_with_tools.ainvoke(test_messages)
    # logger.info(f"Test response type: {type(test_response)}")
    # logger.info(f"Has tool_calls attr: {hasattr(test_response, 'tool_calls')}")
    # if hasattr(test_response, 'tool_calls'):
    #     tool_calls = test_response.tool_calls
    #     logger.info(f"Number of tool calls: {len(tool_calls)}")
    #     if tool_calls:
    #         for tc in tool_calls:
    #             logger.info(f"  Tool call: {tc}")
    # logger.info("=" * 60)

    logger.info("⚠️  Tool binding test skipped (manual optimization)")

    # Create LangGraph agent
    agent = langgraph.create_langgraph_agent(llm_with_tools, tools)

    # Create multi-agent orchestrator if available
    orchestrator = None
    if MULTI_AGENT_AVAILABLE:
        orchestrator = MultiAgentOrchestrator(llm, tools, logger)
        logger.info(f"🎭 Multi-agent orchestrator created (enabled: {MULTI_AGENT_STATE['enabled']})")
    else:
        logger.warning("⚠️ Multi-agent system not available")

    def user_requested_specific_tool(message: str, tools_list: list) -> bool:
        """Check if user explicitly named a tool they want to use"""
        message_lower = message.lower()
        for tool_item in tools_list:
            if hasattr(tool_item, 'name') and tool_item.name.lower() in message_lower:
                return True
        return False

    # Create enhanced agent runner with multi-agent support
    async def run_agent_wrapper(agent, conversation_state, user_message, logger, tools, system_prompt=None):
        """Enhanced agent runner with multi-agent, A2A, and skills support"""

        # Use provided system_prompt or fallback to global SYSTEM_PROMPT
        if system_prompt is None:
            system_prompt = SYSTEM_PROMPT

        if user_requested_specific_tool(user_message, tools):
            logger.info("🎯 User requested specific tool - bypassing multi-agent")
            return await langgraph.run_agent(
                agent, conversation_state, user_message,
                logger, tools, system_prompt, llm, MAX_MESSAGE_HISTORY
            )

        if orchestrator:
            try:
                from pathlib import Path

                # Read what model SHOULD be used (from last_model.txt)
                last_model_file = Path(__file__).parent  / "client/last_model.txt"

                if last_model_file.exists():
                    expected_model = last_model_file.read_text().strip()

                    # Get what model orchestrator is ACTUALLY using
                    actual_model = None
                    if hasattr(orchestrator.base_llm, 'model'):
                        actual_model = orchestrator.base_llm.model
                    elif hasattr(orchestrator.base_llm, 'model_name'):
                        actual_model = orchestrator.base_llm.model_name
                    elif hasattr(orchestrator.base_llm, 'model_path'):
                        actual_model = Path(orchestrator.base_llm.model_path).stem

                    # Sync if mismatch
                    if actual_model and expected_model != actual_model:
                        logger.info(f"🔄 Multi-agent out of sync!")
                        logger.info(f"   Expected (last_model.txt): {expected_model}")
                        logger.info(f"   Actual (orchestrator): {actual_model}")
                        logger.info(f"   Syncing to: {expected_model}")

                        # Use global llm (which should be synced to last_model.txt)
                        from client.llm_backend import LLMBackendManager
                        fresh_llm = LLMBackendManager.create_llm(expected_model, temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")))

                        if hasattr(orchestrator, 'update_llm'):
                            orchestrator.update_llm(fresh_llm)
                        else:
                            logger.error(f"❌ orchestrator.update_llm() not found!")

            except Exception as e:
                logger.warning(f"⚠️ Multi-agent sync check failed: {e}")

        if skills_manager and skills_manager.all_skills:
            conversation_state["messages"] = await inject_relevant_skills_into_messages(
                skills_manager,
                user_message,
                conversation_state["messages"],
                logger
            )

        current_session_id = conversation_state.get("session_id")

        if session_manager and current_session_id:
            try:
                context_injected = integrate_context_tracking(
                    session_manager=session_manager,
                    session_id=current_session_id,
                    prompt=user_message,
                    conversation_state=conversation_state,
                    logger=logger
                )

                if context_injected:
                    logger.info("✅ Context from previous messages injected")

            except Exception as e:
                logger.warning(f"⚠️ Context tracking failed: {e}")

        # Check if A2A should be used (highest priority)
        use_a2a = (
                A2A_STATE["enabled"] and
                MULTI_AGENT_AVAILABLE and
                orchestrator and
                await should_use_multi_agent(user_message)
        )

        if use_a2a:
            logger.info("🔗 Using A2A execution")

            try:
                # Execute with A2A
                result = await orchestrator.execute_a2a(user_message)

                # Handle dict response
                if isinstance(result, dict):
                    result_text = result.get("response", str(result))
                    current_model = result.get("current_model", "unknown")
                    stopped = result.get("stopped", False)
                else:
                    # Backwards compatibility: handle string response
                    result_text = result
                    current_model = "unknown"
                    stopped = False

                # Add to conversation
                conversation_state["messages"].append(HumanMessage(content=user_message))
                conversation_state["messages"].append(AIMessage(content=result_text))

                return {
                    "messages": conversation_state["messages"],
                    "a2a": True,
                    "current_model": current_model,
                    "stopped": stopped
                }

            except Exception as e:
                logger.error(f"❌ A2A execution failed: {e}, falling back to single agent")
                import traceback
                traceback.print_exc()
                use_a2a = False

        # Check if multi-agent should be used (second priority)
        use_multi = (
                MULTI_AGENT_STATE["enabled"] and
                MULTI_AGENT_AVAILABLE and
                not use_a2a and
                await should_use_multi_agent(user_message)
        )

        if use_multi and orchestrator:
            logger.info("🎭 Using MULTI-AGENT execution")

            try:
                # Execute with multi-agent
                result = await orchestrator.execute(user_message)

                # Handle dict response
                if isinstance(result, dict):
                    result_text = result.get("response", str(result))
                    current_model = result.get("current_model", "unknown")
                    stopped = result.get("stopped", False)
                else:
                    # Backwards compatibility: handle string response
                    result_text = result
                    current_model = "unknown"
                    stopped = False

                # Add to conversation
                conversation_state["messages"].append(HumanMessage(content=user_message))
                conversation_state["messages"].append(AIMessage(content=result_text))

                return {
                    "messages": conversation_state["messages"],
                    "multi_agent": True,
                    "current_model": current_model,
                    "stopped": stopped
                }

            except Exception as e:
                logger.error(f"❌ Multi-agent execution failed: {e}, falling back to single agent")
                import traceback
                traceback.print_exc()
                use_multi = False

        if not use_multi and not use_a2a:
            logger.info("🤖 Using SINGLE-AGENT execution")

            # Use existing single agent flow
            return await langgraph.run_agent(
                agent,
                conversation_state,
                user_message,
                logger,
                tools,
                system_prompt,
                llm,
                MAX_MESSAGE_HISTORY
            )

    print("\n🚀 Starting MCP Agent with dual interface support")
    print("=" * 60)
    print(f"🔌 Local MCP servers: {len(mcp_servers)}")
    print(f"🛠️  Total tools available: {len(tools)}")

    if A2A_STATE["enabled"]:
        print(f"🔗 A2A endpoints: {len(A2A_STATE['endpoints'])} active")
        for endpoint in A2A_STATE['endpoints']:
            print(f"   ✓ {endpoint}")

    if MULTI_AGENT_AVAILABLE:
        if A2A_STATE["enabled"]:
            print("🔗 A2A mode: ENABLED")
            print("   Agents communicate via messages for complex workflows")
            print("   Use ':a2a off' to disable")
        elif MULTI_AGENT_STATE["enabled"]:
            print("🎭 Multi-agent mode: ENABLED")
            print("   Complex queries will be broken down automatically")
            print("   Use ':multi off' to disable")
        else:
            print("🤖 Multi-agent mode: DISABLED")
            print("   Use ':multi on' or ':a2a on' to enable")
    else:
        print("⚠️  Multi-agent mode: NOT AVAILABLE")
        print("   Add multi_agent.py to client/ directory to enable")
    print()

    # Open browser
    index_path = PROJECT_ROOT / "client/ui/index.html"
    utils.open_browser_file(index_path)

    # Start HTTP server
    utils.start_http_server(port=9000)

    # Start WebSocket servers
    websocket_server = await websocket.start_websocket_server(
        agent,
        tools,
        logger,
        GLOBAL_CONVERSATION_STATE,
        run_agent_wrapper,
        models,
        model_name,
        SYSTEM_PROMPT,
        orchestrator=orchestrator,
        multi_agent_state=MULTI_AGENT_STATE,
        a2a_state=A2A_STATE,
        mcp_agent=mcp_agent,
        session_manager=session_manager,
        host="0.0.0.0",
        port=8765
    )

    log_websocket_server = await websocket.start_log_websocket_server(
        logging_handler.log_websocket_handler,
        host="0.0.0.0",
        port=8766
    )

    # Start log file tailing
    asyncio.create_task(logging_handler.tail_log_file(SERVER_LOG_FILE))

    # Start system monitor
    if SYSTEM_MONITOR_AVAILABLE:
        asyncio.create_task(system_monitor_loop(websocket.get_system_monitor_clients(), update_interval=1.0))
        print("📊 System monitor started (update interval: 1.0s)")
    else:
        print("⚠️  System monitor disabled (install psutil, gputil, nvidia-ml-py3)")

    print("🖥️  CLI interface ready")
    print("🌐 Browser interface ready at http://localhost:9000")
    print("📊 Log streaming ready at ws://localhost:8766")
    print(f"📋 Tailing server logs: {SERVER_LOG_FILE}")
    print()

    # Show file status
    if SERVER_LOG_FILE.exists():
        size = SERVER_LOG_FILE.stat().st_size
        print(f"📋 Server log file exists: {size} bytes")
    else:
        print(f"⚠️  Server log file does NOT exist yet: {SERVER_LOG_FILE}")
        print(f"   It will be created when server.py starts")
    print()
    print("=" * 60)
    print("\nBoth interfaces share the same conversation state!")
    print("\nCLI Commands:")
    cli.list_commands()
    print()

    try:
        await cli.cli_input_loop(
            agent,
            logger,
            tools,
            model_name,
            GLOBAL_CONVERSATION_STATE,
            run_agent_wrapper,
            models,
            SYSTEM_PROMPT,
            langgraph.create_langgraph_agent,
            orchestrator,
            MULTI_AGENT_STATE,
            A2A_STATE,
            mcp_agent
        )
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    finally:
        websocket_server.close()
        await websocket_server.wait_closed()
        log_websocket_server.close()
        await log_websocket_server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())