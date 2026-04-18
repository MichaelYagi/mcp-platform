"""
MCP Client - Main Entry Point (WITH MULTI-AGENT INTEGRATION + MULTI-A2A SUPPORT)
"""

import json
import logging
import socket
import sys
import asyncio
import os
import re as _re

from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
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

# Import metrics dict once — all recording sites reference this same object
# so websocket.py's prepare_metrics() always sees the current values
try:
    from client.metrics import metrics as _client_metrics
except ImportError:
    _client_metrics = None

# Import all LLM prompt templates from central prompts module
from prompts.prompts import (
    VISION_DEFAULT,
    VISION_LOCATION_INSTRUCTION,
    TOOL_RESULT_PRESENT,
    ITEM_SUMMARISE,
    NOTE_SUMMARISE,
)

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

try:
    from tools.rag.conversation_rag import retrieve_context as _rag_search_turns
    _CONV_RAG_AVAILABLE = True
except ImportError:
    _CONV_RAG_AVAILABLE = False
    def _rag_search_turns(*args, **kwargs): return []

# Load environment variables
PROJECT_ROOT = Path(__file__).parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

# Re-parse DISABLED_TOOLS now that .env is loaded.
# tool_control is imported transitively before load_dotenv runs, so its
# module-level _parse_disabled_tools() fires with an empty DISABLED_TOOLS.
# Calling it again here ensures the cached sets reflect the actual .env values.
try:
    from tools import tool_control as _tc
    _tc._DISABLED_TOOLS_RAW = os.getenv("DISABLED_TOOLS", "")
    _tc._DISABLED_TOOLS = set()
    _tc._DISABLED_CATEGORIES = {}
    _tc._parse_disabled_tools()
except Exception:
    pass

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

Read the user's message carefully and call the RIGHT tool.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PATTERN / SEQUENCE ANALYSIS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

When analyzing sequences or patterns, you MUST:
1. Check your identified pattern against ALL terms, not just the first few
2. Verify the last few terms match your formula before responding
3. If any terms don't fit, revise your pattern — the sequence may have
   multiple phases with different rules (e.g. +3 for first half, +5 for second)"""

# Global conversation state
GLOBAL_CONVERSATION_STATE = {
    "messages": [],
    "loop_count": 0
}


_OAUTH_BROWSER_PATTERN = _re.compile(r'opening\s+browser.*?https?://([^/?#\s]+)', _re.IGNORECASE)
_OAUTH_ERROR_PATTERN = _re.compile(r'(invalid_grant|unauthorized|oauth|token.*expired|auth.*required|could not locate runnable browser)', _re.IGNORECASE)

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

def convert_classpath_for_platform(classpath: str) -> str:
    """Convert Java classpath separators and paths for platform"""
    if is_wsl2():
        return classpath  # WSL2 uses : separator

    # Windows: split by :, convert each path, rejoin with ;
    paths = classpath.split(":")
    windows_paths = [convert_path_for_platform(p) for p in paths]
    return ";".join(windows_paths)

def convert_path_for_platform(path: str) -> str:
    """Convert WSL2 path to Windows path if needed"""
    if is_wsl2():
        return path  # Running in WSL2, use as-is

    # Running on Windows, convert /mnt/c/... to C:\...
    if path.startswith("/mnt/c/"):
        path = path.replace("/mnt/c/", "C:\\")
        path = path.replace("/", "\\")

    return path

def _is_private_ip(host: str) -> bool:
    """Return True if host resolves to a private/loopback RFC-1918 address."""
    try:
        import ipaddress
        addr = ipaddress.ip_address(socket.gethostbyname(host))
        return addr.is_private or addr.is_loopback
    except Exception:
        return False

async def verify_transport_reachable(host: str, port: int, timeout: float = 2.0) -> bool:
    """Check if a TCP port is open.

    Uses a shorter timeout for private/loopback addresses — they respond in
    milliseconds if they're up, so waiting 2 s for a silent firewall drop is
    wasteful and makes startup feel sluggish.
    """
    effective_timeout = 0.5 if _is_private_ip(host) else timeout
    try:
        # Use asyncio to avoid blocking the event loop during the socket check
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port),
            timeout=effective_timeout
        )
        writer.close()
        await writer.wait_closed()
        return True
    except (asyncio.TimeoutError, ConnectionRefusedError, OSError):
        return False

def resolve_env_placeholder(server_name: str, placeholder: str) -> str:
    """
    Resolve <$PLACEHOLDER> using ES_{SERVER_NAME_CAPS}_{PLACEHOLDER} convention.
    Falls back to the placeholder as-is if env var not found.
    """
    env_key = f"ES_{server_name.upper()}_{placeholder.upper()}"
    return os.getenv(env_key, f"<${placeholder}>")


def resolve_headers(server_name: str, headers: dict) -> dict:
    """Resolve all <$PLACEHOLDER> values in headers for a given server."""
    import re
    resolved = {}
    for key, value in headers.items():
        def replacer(match):
            return resolve_env_placeholder(server_name, match.group(1))
        resolved[key] = re.sub(r'<\$([A-Z0-9_a-z]+)>', replacer, value)

async def auto_discover_servers(servers_dir: Path, logger):
    """
    Auto-discover and verify MCP servers.
    Returns a dictionary of verified server configurations.
    """
    mcp_servers = {}

    # 1. Scan local servers/ directory
    if servers_dir.exists():
        for server_dir in servers_dir.iterdir():
            if server_dir.is_dir():
                server_file = server_dir / "server.py"
                if server_file.exists():
                    server_name = server_dir.name
                    venv_python = utils.get_venv_python(PROJECT_ROOT)

                    # Verify Python executable exists
                    if os.path.exists(venv_python):
                        mcp_servers[server_name] = {
                            "command": venv_python,
                            "args": [str(server_file)],
                            "cwd": str(PROJECT_ROOT),
                            "env": {"CLIENT_IP": utils.get_public_ip()}
                        }
                    else:
                        logger.warning(f"⏭️  Skipping local '{server_name}': Venv python not found at {venv_python}")

    # 2. Process external_servers.json
    external_config = PROJECT_ROOT / "external_servers.json"
    if not external_config.exists():
        return mcp_servers

    try:
        config_data = json.loads(external_config.read_text(encoding="utf-8"))
        external_definitions = config_data.get("external_servers", {})

        # We'll collect verification tasks to run them in parallel
        verification_tasks = []
        server_meta = []

        for name, cfg in external_definitions.items():
            if not cfg.get("enabled", True):
                continue

            transport = cfg.get("transport", "stdio")

            if transport == "stdio":
                command = convert_path_for_platform(cfg["command"])
                # Check if command exists/is executable
                if not Path(command).exists():
                    logger.warning(f"⏭️  Skipping '{name}': Command path does not exist: {command}")
                    continue

                # Check if this is a bridged stdio server (waiting on a port)
                port = cfg.get("env", {}).get("IJ_MCP_SERVER_PORT")
                if port:
                    host = cfg.get("env", {}).get("IJ_MCP_SERVER_HOST", "127.0.0.1")
                    verification_tasks.append(verify_transport_reachable(host, int(port)))
                    server_meta.append((name, cfg, "bridge"))
                else:
                    # Pure local stdio, no network check needed
                    mcp_servers[name] = {
                        "command": command,
                        "args": [convert_classpath_for_platform(a) if ";" in a else a for a in cfg.get("args", [])],
                        "env": cfg.get("env", {}),
                        "cwd": cfg.get("cwd", str(PROJECT_ROOT))
                    }
                    logger.info(f"✅ External stdio server verified: {name}")

            elif transport == "sse":
                url = cfg.get("url")
                parsed = urlparse(url)
                host = parsed.hostname
                port = parsed.port or (443 if parsed.scheme == "https" else 80)
                verification_tasks.append(verify_transport_reachable(host, port))
                server_meta.append((name, cfg, "sse"))

            elif transport == "http":
                url = cfg.get("url")
                parsed = urlparse(url)
                host = parsed.hostname
                port = parsed.port or (443 if parsed.scheme == "https" else 80)
                verification_tasks.append(verify_transport_reachable(host, port))
                server_meta.append((name, cfg, "http"))

        # Execute all network checks in parallel
        tcp_results = await asyncio.gather(*verification_tasks, return_exceptions=True)

        # For servers that passed TCP, run OAuth probes in parallel too
        async def _probe_oauth(url: str) -> bool:
            """Return True if the endpoint requires OAuth (HTTP 401)."""
            try:
                import httpx
                async with httpx.AsyncClient(timeout=1.5) as hc:
                    r = await hc.get(url, headers={"Accept": "text/event-stream"})
                    return r.status_code == 401
            except Exception:
                return False  # let mcp_use try

        oauth_tasks = []
        oauth_meta = []
        for (name, cfg, s_type), is_ok in zip(server_meta, tcp_results):
            if is_ok is True and s_type in ("sse", "http"):
                oauth_tasks.append(_probe_oauth(cfg["url"]))
                oauth_meta.append((name, cfg, s_type))

        oauth_results = await asyncio.gather(*oauth_tasks, return_exceptions=True)
        oauth_blocked = {
            name for (name, _, _), blocked in zip(oauth_meta, oauth_results)
            if blocked is True
        }

        for (name, cfg, s_type), is_ok in zip(server_meta, tcp_results):
            if is_ok is True:
                if s_type in ("sse", "http"):
                    if name in oauth_blocked:
                        logger.warning(f"⏭️  Skipping '{name}': OAuth required (401)")
                        continue
                    entry = {"url": cfg["url"], "transport": s_type}
                    headers = cfg.get("headers")
                    if headers:
                        entry["headers"] = resolve_headers(name, headers)
                    mcp_servers[name] = entry
                else:  # bridge
                    mcp_servers[name] = {
                        "command": convert_path_for_platform(cfg["command"]),
                        "args": cfg.get("args", []),
                        "env": cfg.get("env", {}),
                        "cwd": cfg.get("cwd", str(PROJECT_ROOT))
                    }
                logger.info(f"✅ External {s_type} server verified: {name}")
            else:
                logger.warning(f"⏭️  Skipping '{name}': Host unreachable or connection refused.")

    except Exception as e:
        logger.error(f"⚠️  Error processing external config: {e}")

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
    logging.getLogger("mcp.client.streamable_http").setLevel(logging.WARNING)
    logging.getLogger("mcp.client.sse").setLevel(logging.WARNING)
    logger = logging.getLogger("mcp_client")

    # Set event loop for logging
    logging_handler.set_event_loop(asyncio.get_running_loop())

    # Setup MCP client with auto-discovered servers
    mcp_servers = await auto_discover_servers(PROJECT_ROOT / "servers", logger)
    client = MCPClient.from_dict({
        "mcpServers": mcp_servers
    })

    logger.info(f"🔌 Discovered {len(mcp_servers)} MCP servers: {list(mcp_servers.keys())}")

    # Load system prompt from file if it exists
    system_prompt_path = PROJECT_ROOT / "prompts/system_prompt.md"
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
    llm = LLMBackendManager.create_llm(
        model_name,
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
        num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "2048"))
    )

    mcp_agent = MCPAgent(
        llm=llm,
        client=client,
        max_steps=10,
        system_prompt=SYSTEM_PROMPT
    )

    # Detect OAuth-blocked servers during init (browser opens are allowed through)
    class _OAuthDetector(logging.Handler):
        def __init__(self):
            super().__init__()
            self.blocked = set()
        def emit(self, r):
            m = _OAUTH_BROWSER_PATTERN.search(r.getMessage())
            if m:
                self.blocked.add(m.group(1).lower())

    _detector = _OAuthDetector()
    logging.getLogger("mcp_use").addHandler(_detector)

    mcp_agent.debug = False
    try:
        await mcp_agent.initialize()
    except Exception as e:
        logger.error(f"❌ Some MCP servers failed to initialize: {e}")
    finally:
        logging.getLogger("mcp_use").removeHandler(_detector)

    if _detector.blocked:
        before = len(mcp_agent._tools)
        mcp_agent._tools = [
            t for t in mcp_agent._tools
            if urlparse(getattr(getattr(t, 'tool_connector', None), 'url', '') or
                        getattr(getattr(t, 'tool_connector', None), 'base_url', '') or '').hostname
               not in _detector.blocked
        ]
        removed = before - len(mcp_agent._tools)
        for h in _detector.blocked:
            logger.warning(f"⏭️  Skipped '{h}': OAuth sign-in required — complete auth for this server and restart")
        if removed:
            logger.warning(f"   Removed {removed} tools from OAuth-blocked server(s)")

    # Don't give up — collect tools from whatever sessions succeeded
    try:
        recovered = []
        if hasattr(mcp_agent, 'client') and hasattr(mcp_agent.client, 'sessions'):
            for server_name, session in mcp_agent.client.sessions.items():
                try:
                    session_tools = await session.list_tools()
                    for t in session_tools:
                        if t.meta is None:
                            t.meta = {}
                        t.meta['source_server'] = server_name
                    recovered.extend(session_tools)
                    logger.info(f"   ✅ Recovered {len(session_tools)} tools from: {server_name}")
                except Exception as se:
                    logger.warning(f"   ⚠️  Skipping session {server_name}: {se}")
        if recovered and not mcp_agent._tools:
            from langchain_core.tools import StructuredTool
            import inspect

            def _make_tool(t):
                from pydantic import create_model
                from langchain_core.tools import StructuredTool
                source = (t.meta or {}).get('source_server') if isinstance(t.meta, dict) else None
                schema = t.inputSchema or {}
                props = schema.get('properties', {})
                required = schema.get('required', [])
                fields = {}
                type_map = {'string': str, 'integer': int, 'number': float, 'boolean': bool, 'array': list, 'object': dict}
                for fname, finfo in props.items():
                    py_type = type_map.get(finfo.get('type', 'string'), str)
                    if fname in required:
                        fields[fname] = (py_type, ...)
                    else:
                        fields[fname] = (py_type, None)
                args_model = create_model(f"{t.name}Args", **fields) if fields else None
                async def _run(**kwargs): return f"Tool {t.name} called"
                _run.__name__ = t.name
                return StructuredTool(
                    name=t.name,
                    description=(t.description or "").strip(),
                    args_schema=args_model,
                    func=lambda **kw: None,
                    coroutine=_run,
                    metadata={"source_server": source},
                )

            mcp_agent._tools = [_make_tool(t) for t in recovered]
            logger.info(f"⚠️  Partial initialization: {len(mcp_agent._tools)} tools recovered")
        # else:
        #     logger.warning("⚠️  No tools recovered — all servers may have failed")
    except Exception as re:
        logger.error(f"❌ Recovery failed: {re}")

    from client.session_manager import SessionManager
    session_manager = SessionManager()
    logger.info("💾 Session manager initialized")

    from client.session_state import SessionStateRegistry
    session_state_registry = SessionStateRegistry()
    logger.info("🗂️  Session state registry initialized")

    from client.capability_registry import CapabilityRegistry
    capability_registry = CapabilityRegistry()

    tools = mcp_agent._tools

    # ── Tag each tool with its source server via tool_connector ─────
    server_name_map = {}
    if hasattr(mcp_agent, 'client') and hasattr(mcp_agent.client, 'sessions'):
        for sname, session in mcp_agent.client.sessions.items():
            connector = getattr(session, 'connector', None) or getattr(session, '_connector', None)
            if connector:
                server_name_map[id(connector)] = sname

    for tool in tools:
        connector = getattr(tool, 'tool_connector', None)
        if connector:
            sname = server_name_map.get(id(connector))
            if sname:
                if tool.metadata is None:
                    tool.metadata = {}
                tool.metadata['source_server'] = sname

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
    # MULTI-A2A REGISTRATION
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

    # Build capability registry now that the tool list is final
    try:
        from pathlib import Path as _CRPath
        from client.tool_utils import resolve_tool_server, load_external_server_names
        from tools.tool_control import is_tool_enabled as _is_tool_enabled_cr

        _cr_project_root = _CRPath(__file__).resolve().parent
        _cr_external = load_external_server_names(_cr_project_root)
        _cr_tool_to_server = await resolve_tool_server(tools, mcp_agent, _cr_project_root)

        def _cr_is_disabled(tool_name, server_name):
            try:
                return not _is_tool_enabled_cr(tool_name) or not _is_tool_enabled_cr(tool_name, server_name)
            except Exception:
                return False

        capability_registry.build(
            tools=tools,
            tool_to_server=_cr_tool_to_server,
            external_servers=_cr_external,
            is_disabled_fn=_cr_is_disabled,
        )
    except Exception as _cr_err:
        logger.warning(f"⚠️  CapabilityRegistry build failed (non-fatal): {_cr_err}")

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

    # Matches:  "use tool_name: arg value"
    #           "use tool_name"            (no-arg tools)
    _USE_TOOL_RE = _re.compile(
        r'^\s*use\s+(\w+)\s*(?:[:\s]\s*(.*))?$',
        _re.IGNORECASE
    )

    def parse_explicit_tool(message: str, tools_list: list):
        """
        If the message matches 'use <tool_name>[: <arg>]' and <tool_name> is a
        known tool, return (tool_obj, arg_string).  Otherwise return (None, None).
        """
        m = _USE_TOOL_RE.match(message.strip())
        if not m:
            return None, None
        tool_name = m.group(1).lower()
        arg_str = (m.group(2) or "").strip()
        tool_map = {t.name.lower(): t for t in tools_list if hasattr(t, 'name')}
        tool = tool_map.get(tool_name)
        return tool, arg_str

    async def _invoke_tool_directly(tool, arg_str: str, logger) -> str:
        """
        Call a tool directly, bypassing LLM tool-call generation.

        Arg mapping strategy:
          - If arg_str contains key=value pairs (e.g. "movie_title=Yuma scene_query=train"),
            parse them directly as named args.
          - Otherwise map the whole arg_str to the first required parameter.
          - Falls back to {"query": arg_str} for search-style tools.
          - If arg_str is empty, calls with no args (for zero-arg tools).
        """
        import inspect as _inspect
        import uuid as _uuid

        # Get schema info
        first_param = None
        required_params = []
        try:
            schema = tool.args_schema.model_json_schema() if hasattr(tool, 'args_schema') and tool.args_schema else {}
            props = schema.get("properties", {})
            required_params = schema.get("required", list(props.keys()))
            if required_params:
                first_param = required_params[0]
        except Exception:
            pass

        if not arg_str:
            tool_args = {}
        else:
            # Check for key=value syntax: "key1=val1 key2=val2" or "key1=val1, key2=val2"
            # Simple pattern: word=anything-up-to-next-word= or end
            _kv_re = _re.compile(r'(\w+)=(".*?"|\'.*?\'|[^\s,]+)')
            kv_matches = _kv_re.findall(arg_str)
            if kv_matches:
                tool_args = {}
                for k, v in kv_matches:
                    # Strip surrounding quotes if present
                    if len(v) >= 2 and v[0] in ('"', "'") and v[-1] == v[0]:
                        v = v[1:-1]
                    # Drop empty-string values for optional params — user left the
                    # bracket placeholder blank (e.g. [use_thumbnail=""]).
                    # Passing "" to a bool/int pydantic field causes a validation error.
                    if v == "":
                        continue  # always drop empty strings — let pydantic defaults apply
                    tool_args[k] = v
            else:
                # If arg contains a URL or file path with preceding text, extract it
                # e.g. "describe this https://..." or "convert this /path/to/file.jpg"
                _url_re   = _re.compile(r'(https?://\S+)')
                _path_re  = _re.compile(r'((?:/|[A-Za-z]:\\\\)\S+)')
                _url_match  = _url_re.search(arg_str)
                _path_match = _path_re.search(arg_str)
                _match = _url_match or _path_match
                _is_image_param = first_param in ("image_url", "url", "image_file_path")
                if _match and _is_image_param:
                    _val    = _match.group(1)
                    _prompt = arg_str[:_match.start()].strip()
                    # Route to correct param: URLs → image_url, paths → image_file_path
                    if _url_match:
                        tool_args = {"image_url": _val}
                    else:
                        tool_args = {"image_file_path": _val}
                    if _prompt:
                        tool_args["_extra_prompt"] = _prompt
                else:
                    # Detect "city, state" or "city, state, country" natural location format
                    # when the tool has a city param (location tools)
                    _loc_params = list(props.keys()) if props else []
                    _has_loc = "city" in _loc_params
                    _parts = [p.strip() for p in arg_str.split(",")]
                    if _has_loc and 2 <= len(_parts) <= 3 and all(_parts):
                        tool_args = {"city": _parts[0]}
                        if len(_parts) >= 2:
                            tool_args["state"] = _parts[1]
                        if len(_parts) == 3:
                            tool_args["country"] = _parts[2]
                    else:
                        tool_args = {first_param: arg_str} if first_param else {"query": arg_str}

        logger.info(f"🎯 Direct tool dispatch: {tool.name}({tool_args})")

        result = await tool.ainvoke(tool_args)

        # Unwrap TextContent / list results (mirrors langgraph's own unwrapping)
        if isinstance(result, list) and result:
            first = result[0]
            # Prefer the .text attribute (mcp TextContent), then .content
            if hasattr(first, 'text') and first.text is not None:
                result = first.text
            elif hasattr(first, 'content') and first.content is not None:
                result = first.content
            else:
                # Last resort: pull text value out of the repr string
                joined = str(first)
                # Match text='...' with potentially escaped quotes inside
                import re as _re2
                m = _re2.search(r"text='(.*?)',\s*annotations=", joined, _re2.DOTALL)
                if m:
                    result = m.group(1)
                else:
                    result = joined

        result_str = str(result)

        # If the result is still a TextContent repr (unwrap failed), extract JSON directly
        if result_str.startswith("[TextContent(") or result_str.startswith("TextContent("):
            import re as _re3
            m = _re3.search(r"text='(.*)',\s*annotations=", result_str, _re3.DOTALL)
            if m:
                result_str = m.group(1)
            # Unescape Python string escapes (\n \t etc.) without corrupting UTF-8
            try:
                if "\\n" in result_str or "\\t" in result_str:
                    result_str = result_str.encode("utf-8").decode("unicode_escape").encode("latin-1").decode("utf-8")
            except Exception:
                pass  # keep original if decode fails

        return result_str

    # Create enhanced agent runner with multi-agent support
    async def run_agent_wrapper(agent, conversation_state, user_message, logger, tools, system_prompt=None):
        """Enhanced agent runner with multi-agent, A2A, and skills support"""

        # Use provided system_prompt or fallback to global SYSTEM_PROMPT
        if system_prompt is None:
            system_prompt = SYSTEM_PROMPT

        # ── Explicit tool dispatch ──────────────────────────────────────
        # When the user types "use <tool_name>[: <args>]" we call the tool
        # directly and hand the result straight to the LLM for summarization.
        # This bypasses LangGraph tool-call generation entirely, which small
        # models (llama3.2:3b etc.) handle poorly.
        explicit_tool, explicit_arg = parse_explicit_tool(user_message, tools)
        if explicit_tool:
            logger.info(f"🎯 Explicit tool dispatch: {explicit_tool.name!r} arg={explicit_arg!r}")
            import uuid as _uuid2, time as _time

            # Always use the model from last_model.txt — the single source of truth
            # after a model switch. orchestrator.base_llm may lag behind.
            try:
                from client.llm_backend import LLMBackendManager
                _last_model_file = PROJECT_ROOT / 'client' / 'last_model.txt'
                _current_model = _last_model_file.read_text().strip() if _last_model_file.exists() else None
                if _current_model:
                    active_llm = LLMBackendManager.create_llm(
                        _current_model,
                        temperature=float(os.getenv('LLM_TEMPERATURE', '0.3')),
                        num_ctx=int(os.getenv('OLLAMA_NUM_CTX', '2048'))
                    )
                else:
                    active_llm = orchestrator.base_llm if orchestrator and hasattr(orchestrator, 'base_llm') else llm
            except Exception as _model_err:
                logger.warning(f'⚠️ Could not load model from last_model.txt: {_model_err}')
                active_llm = orchestrator.base_llm if orchestrator and hasattr(orchestrator, 'base_llm') else llm
            logger.info(f"🤖 Explicit dispatch model: {getattr(active_llm, 'model', 'unknown')}")

            start = _time.time()
            try:
                tool_result = await _invoke_tool_directly(explicit_tool, explicit_arg, logger)
                _duration = _time.time() - start
                logger.info(f"✅ Direct tool {explicit_tool.name} completed in {_duration:.2f}s")
                # Record into metrics so dashboard reflects direct dispatch calls
                if _client_metrics is not None:
                    _client_metrics["tool_calls"][explicit_tool.name] += 1
                    _client_metrics["tool_times"][explicit_tool.name].append((_time.time(), _duration))
            except Exception as e:
                _duration = _time.time() - start
                logger.error(f"❌ Direct tool {explicit_tool.name} failed: {e}")
                tool_result = f"Error running {explicit_tool.name}: {e}"
                if _client_metrics is not None:
                    _client_metrics["tool_calls"][explicit_tool.name] += 1
                    _client_metrics["tool_errors"][explicit_tool.name] += 1
                    _client_metrics["tool_times"][explicit_tool.name].append((_time.time(), _duration))

                # ── Mid-session OAuth recovery ─────────────────────────────────
                # If the error looks like an OAuth/auth failure, attempt re-auth
                # and retry the tool call once.
                if _OAUTH_ERROR_PATTERN.search(str(e)):
                    _server_hostname = None
                    _connector = getattr(explicit_tool, 'tool_connector', None)
                    if _connector:
                        _url = getattr(_connector, 'url', '') or getattr(_connector, 'base_url', '')
                        _server_hostname = urlparse(_url).hostname or ""

                    # Local Google server — use auth_google.py
                    _is_local_google = (
                        not _server_hostname or
                        _server_hostname in ("localhost", "127.0.0.1") or
                        (explicit_tool.metadata or {}).get('source_server', '').lower() in ('google-server', 'google')
                    )

                    if _is_local_google:
                        _auth_script = PROJECT_ROOT / "auth_google.py"
                        if _auth_script.exists():
                            logger.warning(f"🔐 OAuth failure detected — launching auth_google.py to re-authenticate")
                            import subprocess as _oauth_sp
                            _auth_result = _oauth_sp.run(
                                [sys.executable, str(_auth_script)],
                                capture_output=False
                            )
                            if _auth_result.returncode == 0:
                                logger.info("🔐 Re-auth complete — retrying tool call")
                                try:
                                    tool_result = await _invoke_tool_directly(explicit_tool, explicit_arg, logger)
                                    logger.info(f"✅ Retry succeeded for {explicit_tool.name}")
                                    if _client_metrics is not None:
                                        _client_metrics["tool_calls"][explicit_tool.name] += 1
                                except Exception as _retry_e:
                                    logger.error(f"❌ Retry failed after re-auth: {_retry_e}")
                                    tool_result = f"Re-authentication succeeded but tool still failed: {_retry_e}"
                            else:
                                logger.error("❌ auth_google.py failed — Google tools will remain unavailable")
                                tool_result = "Google authentication failed. Run auth_google.py manually and restart."
                        else:
                            logger.error(f"❌ OAuth failure on local server but auth_google.py not found at {_auth_script}")
                    else:
                        # External MCP server OAuth failure — log clearly, no auto-recovery possible
                        logger.warning(
                            f"🔐 OAuth failure on external server '{_server_hostname}' — "
                            f"complete authentication for this server manually and restart mcp-platform"
                        )
                        tool_result = (
                            f"Authentication required for '{_server_hostname}'. "
                            f"Please complete OAuth sign-in for this server and restart mcp-platform."
                        )

            # Log a short preview so the result count is visible immediately
            # Strip base64 blobs before logging — they flood the log with useless noise
            import re as _re_b64
            _preview_safe = _re_b64.sub(
                r'"([A-Za-z0-9+/]{100,}={0,2})"',
                lambda m: f'"{m.group(1)[:12]}...[base64 {len(m.group(1))} chars]"',
                tool_result
            )
            preview_lines = _preview_safe.splitlines()
            preview = "\n".join(preview_lines[:6])
            logger.info(f"📋 Tool result preview:\n{preview}")

            # ── Pre-formatted text passthrough ──────────────────────────
            # Some tools (shashin_search_tool etc.) return pre-formatted plain
            # text rather than JSON. Detect this and pass through directly.
            # Detect double-encoded JSON: a string like "{\"city\":\"Vancouver\"...}"
            _stripped = tool_result.strip()
            if _stripped.startswith('"') and _stripped.endswith('"'):
                try:
                    _inner = json.loads(_stripped)  # unwrap outer string
                    if isinstance(_inner, str):
                        _inner_parsed = json.loads(_inner)
                        if isinstance(_inner_parsed, dict):
                            tool_result = json.dumps(_inner_parsed)
                except Exception:
                    pass

            _is_plain_text = not tool_result.strip().startswith(("{", "["))
            # Error strings (from MCPToolError) are plain text — they pass through
            # directly below without going to the LLM, which prevents safety refusals
            # when error messages contain private IP addresses or raw URLs.
            if _is_plain_text:
                summary_text = tool_result

                # Persist with ToolMessage so image scanner still runs
                import uuid as _pt_uuid
                _pt_call_id = str(_pt_uuid.uuid4())
                conversation_state["messages"].append(HumanMessage(content=user_message))
                conversation_state["messages"].append(AIMessage(
                    content="",
                    tool_calls=[{"id": _pt_call_id, "name": explicit_tool.name, "args": {}}]
                ))
                conversation_state["messages"].append(ToolMessage(
                    content=tool_result,
                    tool_call_id=_pt_call_id,
                    name=explicit_tool.name
                ))
                conversation_state["messages"].append(AIMessage(content=summary_text))
                return {
                    "messages": conversation_state["messages"],
                    "current_model": getattr(active_llm, "model", "unknown")
                }

            # ── Vision shortcut ─────────────────────────────────────────────
            # If the tool result has image_source, call Ollama vision directly
            # (same path as LangGraph) rather than summarising with text LLM.
            import re as _re4
            try:
                _tool_json = json.loads(tool_result)
            except Exception:
                _tool_json = None

            if isinstance(_tool_json, dict) and (_tool_json.get("image_source") or _tool_json.get("image_base64")):
                logger.info("[direct dispatch] 🖼️ Image result — delegating to vision")
                import httpx as _httpx2, base64 as _b642
                _img_src = _tool_json.get("image_source")
                _img_orig = _tool_json.get("image_source_original") or _img_src
                _b64img = _tool_json.get("image_base64")
                if _b64img and "," in _b64img:
                    _b64img = _b64img.split(",", 1)[1]
                _shashin_key = os.getenv("SHASHIN_API_KEY", "")
                _fetch_hdrs = {}
                try:
                    if not _b64img and _img_orig:
                        if _shashin_key and ("192.168." in _img_orig or "shashin" in _img_orig.lower()):
                            _fetch_hdrs = {"x-api-key": _shashin_key, "Content-Type": "application/json"}
                        async with _httpx2.AsyncClient(timeout=60.0) as _hc:
                            _ir = await _hc.get(_img_orig, headers=_fetch_hdrs)
                            _ir.raise_for_status()
                        _b64img = _b642.b64encode(_ir.content).decode("utf-8")
                        logger.info(f"[direct dispatch] 🖼️ Fetched {len(_ir.content)} bytes for vision")
                    else:
                        logger.info("[direct dispatch] 🖼️ Using pre-encoded base64 image")

                    # Build vision prompt using existing metadata
                    _place = _tool_json.get("placeName", "")
                    _taken = _tool_json.get("takenAt", "")
                    _desc  = _tool_json.get("description", "")
                    _fname = _tool_json.get("fileName", "")

                    # Only use place and date as context — filename is technical noise
                    # that pushes the model toward a formal/clinical tone
                    _meta_parts = [p for p in [_place, _taken] if p]
                    _meta = ", ".join(_meta_parts)

                    # Extract user's extra instruction.
                    # Priority: query field in tool result > _extra_prompt > stripped explicit_arg
                    _extra = _tool_json.get("query", "") or _tool_json.get("_extra_prompt", "")
                    if not _extra:
                        _raw_extra = _re.sub(
                            r'\w+=["\']?[^"\'"\s,]+["\']?|https?://\S+|(?:/|[A-Za-z]:\\\\)\S+',
                            '', explicit_arg
                        ).strip()
                        _extra = _raw_extra

                    if _extra and not _extra.startswith("http") and len(_extra) > 3:
                        # User gave a specific instruction — use it, add place/date context
                        _vision_prompt = _extra
                        if _meta:
                            _vision_prompt += f" (taken at {_meta})"
                    else:
                        # Default: warm, natural description — same tone for random and by-ID.
                        # Only weave in location/date when they add genuine context.
                        _location_hint = ""
                        if _place and _taken:
                            _location_hint = f" The photo was taken at {_place} on {_taken}."
                        elif _place:
                            _location_hint = f" The photo was taken at {_place}."
                        elif _taken:
                            _location_hint = f" The photo was taken on {_taken}."

                        _vision_prompt = (
                            VISION_DEFAULT
                            + (_location_hint if _location_hint else "")
                            + (f" Existing caption: {_desc}." if _desc else "")
                            + (VISION_LOCATION_INSTRUCTION if _location_hint else "")
                        )

                    # Use higher token limit for specific queries (translation, reading text,
                    # counting, detailed analysis) vs generic descriptions
                    _specific_query = bool(_extra and len(_extra) > 3)
                    _num_predict = 1000 if _specific_query else 300

                    # Call Ollama vision API directly
                    _ollama_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
                    _vision_model = os.getenv("OLLAMA_VISION_MODEL", "qwen3-vl:8b-instruct")
                    logger.info(f"[direct dispatch] 🖼️ Calling vision model: {_vision_model} (num_predict={_num_predict}, query={bool(_extra)})")
                    _vision_llm_start = _time.time()
                    async with _httpx2.AsyncClient(timeout=300.0) as _vclient:
                        _vresp = await _vclient.post(
                            f"{_ollama_url}/api/chat",
                            json={
                                "model": _vision_model,
                                "stream": False,
                                "options": {
                                    "num_predict": _num_predict,
                                    "repeat_penalty": 1.3,
                                    "temperature": 0.3,
                                },
                                "messages": [{
                                    "role": "user",
                                    "content": _vision_prompt,
                                    "images": [_b64img]
                                }]
                            }
                        )
                        _vdata = _vresp.json()
                    if _client_metrics is not None:
                        _client_metrics["llm_calls"] += 1
                        _client_metrics["llm_times"].append((_time.time(), _time.time() - _vision_llm_start))
                    logger.info(f"[direct dispatch] 🖼️ Vision response status: {_vresp.status_code}, keys: {list(_vdata.keys()) if isinstance(_vdata, dict) else type(_vdata)}")
                    summary_text = _vdata.get("message", {}).get("content", "").strip()
                    # Detect and trim repetition loops — if any sentence repeats 3+ times, truncate before it
                    if summary_text:
                        _sentences = summary_text.split(". ")
                        _seen_sentences: dict = {}
                        _cutoff = len(_sentences)
                        for _si, _sent in enumerate(_sentences):
                            _key = _sent.strip().lower()[:60]
                            if not _key:
                                continue
                            _seen_sentences[_key] = _seen_sentences.get(_key, 0) + 1
                            if _seen_sentences[_key] >= 3:
                                _cutoff = _si
                                break
                        if _cutoff < len(_sentences):
                            summary_text = ". ".join(_sentences[:_cutoff]).rstrip(".") + "."
                            logger.info(f"[direct dispatch] 🖼️ Trimmed repetition loop at sentence {_cutoff}")
                    if not summary_text:
                        logger.warning(f"[direct dispatch] 🖼️ Vision returned empty content. Full response: {str(_vdata)[:500]}")
                    logger.info(f"[direct dispatch] 🖼️ Vision description: {summary_text[:80]}")


                except Exception as _ve:
                    logger.warning(f"[direct dispatch] 🖼️ Vision failed: {type(_ve).__name__}: {_ve} — falling back to metadata summary")
                    summary_text = None  # fall through to normal summarisation below

                # Build search link — appended regardless of vision success/failure
                _image_id = _tool_json.get("image_id")
                _shashin_base = os.getenv("SHASHIN_BASE_URL", "").rstrip("/")
                _shashin_link = (
                    f"\n\n[🔗 View in Shashin]({_shashin_base}/search?term={_image_id})"
                    if _image_id and _shashin_base else ""
                )

                if summary_text:
                    summary_text += _shashin_link
                    import uuid as _vis_uuid
                    _vis_call_id = str(_vis_uuid.uuid4())
                    conversation_state["messages"].append(HumanMessage(content=user_message))
                    conversation_state["messages"].append(AIMessage(
                        content="",
                        tool_calls=[{"id": _vis_call_id, "name": explicit_tool.name, "args": {}}]
                    ))
                    conversation_state["messages"].append(ToolMessage(
                        content=tool_result,
                        tool_call_id=_vis_call_id,
                        name=explicit_tool.name
                    ))
                    conversation_state["messages"].append(AIMessage(content=summary_text))
                    # Store last image in conversation state for follow-up vision queries
                    conversation_state["_last_vision_b64"] = _b64img
                    conversation_state["_last_vision_url"] = _img_orig
                    conversation_state["_last_vision_tool_result"] = tool_result
                    return {
                        "messages": conversation_state["messages"],
                        "current_model": getattr(active_llm, "model", "unknown")
                    }

            # Build output with per-note LLM summaries assembled entirely in Python.
            # One small focused LLM call per note avoids small-model context overload.
            trilium_base = os.getenv("TRILIUM_URL", "").rstrip("/")
            note_lines = None

            if trilium_base:
                try:
                    parsed = json.loads(tool_result)
                    items = parsed if isinstance(parsed, list) else parsed.get("results", [])
                    if items and items[0].get("noteId"):
                        _tool_map = {t.name: t for t in tools if hasattr(t, "name")}
                        get_note_tool = _tool_map.get("get_note_by_id")

                        note_lines = [f'Found {len(items)} note(s) matching "{explicit_arg}":\n']

                        for i, item in enumerate(items, 1):
                            note_id = item.get("noteId")
                            title = item.get("title", note_id)
                            url = f"{trilium_base}/#root/{note_id}"

                            # Fetch full content if tool available, else use preview
                            full_content = None
                            if get_note_tool and note_id:
                                try:
                                    raw = await get_note_tool.ainvoke({"note_id": note_id})
                                    if isinstance(raw, list) and raw:
                                        raw = raw[0].text if hasattr(raw[0], "text") else str(raw[0])
                                    full_content = str(raw)
                                    full_content = _re4.sub(r'<[^>]+>', ' ', full_content).strip()
                                    full_content = _re4.sub(r' +', ' ', full_content)
                                    full_content = full_content[:2000] + "..." if len(full_content) > 2000 else full_content
                                except Exception as _e:
                                    logger.warning(f"⚠️ Could not fetch full content for {note_id}: {_e}")

                            if not full_content:
                                full_content = item.get("contentPreview", "")
                                full_content = _re4.sub(r'<[^>]+>', '', full_content).strip()

                            # One focused LLM call per note: just summarise this content
                            summary = ""
                            if full_content:
                                try:
                                    _note_llm_start = _time.time()
                                    resp = await active_llm.ainvoke([
                                        SystemMessage(content=NOTE_SUMMARISE),
                                        HumanMessage(content=full_content),
                                    ])
                                    summary = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
                                    if _client_metrics is not None:
                                        _client_metrics["llm_calls"] += 1
                                        _client_metrics["llm_times"].append((_time.time(), _time.time() - _note_llm_start))
                                except Exception as _e:
                                    logger.warning(f"⚠️ Summary failed for {note_id}: {_e}")
                                    summary = full_content[:200] + "..."

                            note_lines.append(f"{i}. [{title}]({url}) `{note_id}`")
                            if summary:
                                note_lines.append(f"   {summary}")
                            note_lines.append("")

                        logger.info(f"📋 Built {len(items)} per-note summaries")
                except (json.JSONDecodeError, AttributeError):
                    pass

            if note_lines:
                summary_text = "\n".join(note_lines)
            elif _tool_json and isinstance(_tool_json, dict) and "summary" in _tool_json and not any(
                isinstance(_tool_json.get(k), list) for k in
                ("documents", "sources", "results", "items", "records", "entries", "chunks")
            ):
                # Tool already did the summarisation internally (e.g. summarize_url_tool,
                # summarize_text_tool). Pass the summary through directly — no second LLM
                # call needed.
                # Guard: only use this path when there is NO primary list in the response.
                # Tools like rag_browse_tool and rag_list_sources_tool add a convenience
                # "summary" key alongside their document/source lists — we want the
                # list-builder to handle those so all the rich data is shown.
                _presummary = _tool_json["summary"]
                _title      = _tool_json.get("title", "")
                _url        = _tool_json.get("url", "")
                _src        = _tool_json.get("source", "")
                _orig_len   = _tool_json.get("original_length", 0)
                _truncated  = _tool_json.get("truncated", False)

                header_parts = []
                if _title:
                    header_parts.append(f"**{_title}**")
                if _url:
                    header_parts.append(f"[{_url}]({_url})")
                elif _src and _src != "text":
                    header_parts.append(f"`{_src}`")
                if _orig_len:
                    trunc_note = " *(truncated)*" if _truncated else ""
                    header_parts.append(f"({_orig_len:,} chars{trunc_note})")

                header = " — ".join(header_parts)
                summary_text = (f"{header}\n\n{_presummary}" if header else _presummary)
                logger.info(f"📋 Passing through pre-built summary ({len(_presummary)} chars)")
            else:
                # Try to build a Python-formatted list for any JSON result that
                # contains a top-level list under a common key, matching the same
                # style as the Trilium / shashin search outputs.
                # Falls back to a single LLM call only for non-list results.
                list_lines = None
                try:
                    parsed = json.loads(tool_result)

                    items = None
                    array_key = None
                    # If the result has a top-level "text" field, use it directly — no LLM
                    if isinstance(parsed, dict) and parsed.get("text") and isinstance(parsed["text"], str):
                        logger.info(f"📋 Using top-level text field directly")
                        list_lines = [parsed["text"]]
                    else:
                        # Find the first array value in the response regardless of key name
                        if isinstance(parsed, list):
                            items = parsed
                            array_key = "results"
                        else:
                            for k, v in parsed.items():
                                # Only treat as a list if items are dicts (not primitives like strings)
                                if isinstance(v, list) and v and isinstance(v[0], dict):
                                    items = v
                                    array_key = k
                                    break
                                # Handle one level of nesting: {scenes: {scenes: [...]}}
                                elif isinstance(v, dict):
                                    for k2, v2 in v.items():
                                        if isinstance(v2, list) and v2 and isinstance(v2[0], dict):
                                            items = v2
                                            array_key = k2
                                            break
                                    if items:
                                        break

                    if items and isinstance(parsed, dict):
                        _sibling_dicts = (
                            isinstance(parsed, dict) and
                            sum(1 for v in parsed.values() if isinstance(v, dict)) >= 2
                        )
                        # Also skip list path when the response has multiple substantial
                        # top-level scalar/string fields alongside the list — these are
                        # structured reports (e.g. weather: temperature/condition/humidity
                        # + forecast list) where the LLM single call renders everything.
                        _sibling_scalars = (
                            isinstance(parsed, dict) and
                            sum(1 for k, v in parsed.items()
                                if not isinstance(v, (list, dict)) and v not in (None, "", 0)
                                and k not in ("total", "count", "total_count", "summary",
                                              "message", "status", "error",
                                              # location/weather metadata — not content
                                              "city", "state", "country", "latitude",
                                              "longitude", "timezone")) >= 3
                        )
                        # Weather responses have a "current" dict + "forecast" list —
                        # the list builder only sees the forecast array and renders bare
                        # date strings. Route the whole response to the LLM instead so
                        # it can render current conditions and forecast together.
                        _is_weather_response = (
                            isinstance(parsed, dict) and
                            isinstance(parsed.get("current"), dict) and
                            isinstance(parsed.get("forecast"), list)
                        )
                        if _sibling_dicts or _sibling_scalars or _is_weather_response:
                            items = None  # skip list path → falls through to LLM call

                    if items:
                        # Use actual list length as the count — most accurate
                        total = len(items)
                        # Use the array key name as the label (sources, documents, results…)
                        label_word = array_key.rstrip("s") if array_key else "result"
                        arg_label = f' for "{explicit_arg}"' if explicit_arg else ""
                        # Prepend location line for weather/location tools
                        _loc_city    = parsed.get("city", "")
                        _loc_state   = parsed.get("state", "")
                        _loc_country = parsed.get("country", "")
                        _loc_parts   = [p for p in [_loc_city, _loc_state, _loc_country] if p]
                        _loc_prefix  = f"📍 {', '.join(_loc_parts)}\n\n" if _loc_parts else ""
                        header = f"{_loc_prefix}Found {total} {label_word}(s){arg_label}:\n"
                        list_lines = [header]

                        # For RAG search results: fetch chunk text from sessions.db
                        # Items have chunk_id but no text field
                        _sessions_db_path = os.getenv("SESSIONS_DB", str(
                            PROJECT_ROOT / "data" / "sessions.db"
                        ))
                        _chunk_cache = {}
                        _has_chunk_ids = any(
                            isinstance(it, dict) and it.get("chunk_id") and not (
                                it.get("preview") or it.get("sample") or
                                it.get("content") or it.get("contentPreview")
                            )
                            for it in items
                        )
                        if _has_chunk_ids:
                            try:
                                import sqlite3 as _sq
                                _sc = _sq.connect(_sessions_db_path)
                                for it in items:
                                    cid = it.get("chunk_id")
                                    if cid and cid not in _chunk_cache:
                                        row = _sc.execute(
                                            "SELECT text FROM chunks WHERE id = ?", (cid,)
                                        ).fetchone()
                                        _chunk_cache[cid] = row[0] if row else ""
                                _sc.close()
                            except Exception as _ce:
                                logger.warning(f"⚠️ Could not fetch chunk text: {_ce}")

                        for i, item in enumerate(items, 1):
                            if not isinstance(item, dict):
                                list_lines.append(f"{i}. {item}")
                                continue

                            # Build a title from common fields
                            title = (
                                item.get("title")
                                or item.get("name")
                                or item.get("source")
                                or item.get("date")
                                or item.get("day")
                                or item.get("id")
                                or f"Item {i}"
                            )
                            # For process items, append PID to disambiguate multiple python processes
                            if item.get("pid") and item.get("name"):
                                title = f"{item['name']} (pid {item['pid']})"
                            # Score label for search results
                            score = item.get("score")
                            score_label = f" (score: {score:.2f})" if score is not None else ""

                            # Get content: prefer cached chunk text, then common fields
                            chunk_id = item.get("chunk_id")
                            # If no standard content field, serialise all kv pairs
                            _kv_content = "; ".join(
                                f"{k}: {v}" for k, v in item.items()
                                if v not in (None, "", []) and k not in ("chunk_id", "score", "id")
                            )
                            content_text = (
                                _chunk_cache.get(chunk_id, "")
                                or item.get("preview")
                                or item.get("sample")
                                or item.get("content")
                                or item.get("contentPreview")
                                or item.get("text")
                                or item.get("dialogue")
                                or item.get("snippet")      # web search results
                                or _kv_content
                                or ""
                            )
                            # For scene results: prepend timestamp if available
                            timestamp = item.get("timestamp") or item.get("time") or item.get("start_time")
                            if timestamp and content_text:
                                content_text = f"[{timestamp}] {content_text}"
                            content_text = _re4.sub(r'<[^>]+>', ' ', str(content_text)).strip()
                            # Truncate hard — small models hallucinate on long text
                            content_text = content_text[:300]

                            # Detect metadata-only items (sources, stats) — render directly,
                            # no LLM call needed. Only summarise when real text content exists.
                            meta_fields = {
                                k: v for k, v in item.items()
                                if k in ("documents", "chars", "words", "total_words",
                                         "total_chars", "score", "created", "dateModified",
                                         # process / system fields
                                         "pid", "memory_percent", "cpu_percent", "status",
                                         "memory_mb", "cpu_time", "threads", "ppid",
                                         # generic numeric stats
                                         "count", "total", "size", "duration", "bytes",
                                         "percent", "usage", "rate", "value")
                                and v is not None
                            }
                            # Items with a preview field have real text — summarise with LLM
                            has_preview = bool(item.get("preview") or item.get("sample"))
                            # Also treat items with ONLY numeric/id fields as metadata
                            _text_fields = {"preview", "sample", "content", "contentPreview",
                                            "text", "dialogue", "description", "body", "summary",
                                            "snippet"}   # web search results
                            _has_text_field = any(item.get(f) for f in _text_fields)
                            # Web search items: snippet is the final answer, never send to LLM
                            _is_web_search_item = bool(item.get("snippet") and item.get("url"))
                            is_metadata_item = (
                                (bool(meta_fields) or not _has_text_field or _is_web_search_item)
                                and not chunk_id
                                and not _chunk_cache
                                and not has_preview
                            )

                            summary = ""
                            if is_metadata_item:
                                # Build a concise metadata line directly in Python
                                parts = []
                                # Process / system items
                                if "pid" in item:
                                    if "memory_percent" in item:
                                        parts.append(f"mem: {item['memory_percent']:.1f}%")
                                    if "cpu_percent" in item:
                                        parts.append(f"cpu: {item['cpu_percent']:.1f}%")
                                    if "status" in item:
                                        parts.append(f"status: {item['status']}")
                                    if "memory_mb" in item:
                                        parts.append(f"{item['memory_mb']:.0f} MB")
                                # Web search results — title already in heading, show url + snippet
                                elif item.get("snippet") or item.get("url"):
                                    if item.get("url"):
                                        parts.append(item["url"])
                                    if item.get("snippet"):
                                        # snippet is the useful text — show it directly, no LLM
                                        summary = item["snippet"]
                                        if item.get("url"):
                                            summary = f"{item['url']}\n   {summary}"
                                # Shashin / photo items
                                elif item.get("filename") or item.get("tags") or item.get("date"):
                                    if item.get("date"):
                                        parts.append(str(item["date"]))
                                    if item.get("tags"):
                                        tags = item["tags"]
                                        if isinstance(tags, list):
                                            parts.append(", ".join(str(t) for t in tags[:6]))
                                        else:
                                            parts.append(str(tags))
                                    if item.get("filename") and not title.startswith(item.get("filename", "")):
                                        parts.append(item["filename"])
                                # Source / RAG items
                                elif "documents" in meta_fields:
                                    parts.append(f"{meta_fields['documents']} chunk(s)")
                                # Generic char/word counts
                                if "chars" in meta_fields or "total_chars" in meta_fields:
                                    c = meta_fields.get("chars") or meta_fields.get("total_chars")
                                    parts.append(f"{c:,} chars")
                                if "score" in meta_fields:
                                    parts.append(f"score: {meta_fields['score']:.2f}")
                                if "created" in meta_fields:
                                    parts.append(f"created: {meta_fields['created']}")
                                if not summary:
                                    if content_text and not parts:
                                        # sample field — show it as-is, no LLM
                                        summary = content_text[:150] + ("..." if len(content_text) > 150 else "")
                                    elif parts:
                                        summary = ", ".join(parts)
                            elif content_text:
                                # If item has a "text" field, use it directly — no LLM
                                if item.get("text"):
                                    summary = item["text"]
                                else:
                                    try:
                                        _item_llm_start = _time.time()
                                        resp = await active_llm.ainvoke([
                                            SystemMessage(content=ITEM_SUMMARISE),
                                            HumanMessage(content=content_text),
                                        ])
                                        summary = resp.content.strip() if hasattr(resp, "content") else ""
                                        if _client_metrics is not None:
                                            _client_metrics["llm_calls"] += 1
                                            _client_metrics["llm_times"].append((_time.time(), _time.time() - _item_llm_start))
                                    except Exception:
                                        summary = content_text[:120] + "..."

                            list_lines.append(f"{i}. {title}{score_label}")
                            if summary:
                                list_lines.append(f"   {summary}")
                            list_lines.append("")

                except (json.JSONDecodeError, AttributeError, StopIteration) as _list_err:
                    logger.warning(f"⚠️ List builder failed: {_list_err}")

                if list_lines:
                    logger.info(f"📋 Built list output ({len(list_lines)} lines)")
                    summary_text = "\n".join(list_lines)
                else:
                    # Scalar / confirmation result — single LLM call
                    logger.info(f"🧠 Summarising tool result with LLM ({len(tool_result)} chars)")
                    _llm_start = _time.time()
                    resp = await active_llm.ainvoke([
                        SystemMessage(content=TOOL_RESULT_PRESENT.format(tool_name=explicit_tool.name)),
                        HumanMessage(content=tool_result),
                    ])
                    summary_text = resp.content if hasattr(resp, "content") else str(resp)
                    if _client_metrics is not None:
                        _client_metrics["llm_calls"] += 1
                        _client_metrics["llm_times"].append((_time.time(), _time.time() - _llm_start))

            # Persist the exchange in conversation history.
            # Include a ToolMessage so websocket image/place scanner finds image_source.
            import uuid as _tmsg_uuid
            _tool_call_id = str(_tmsg_uuid.uuid4())
            conversation_state["messages"].append(HumanMessage(content=user_message))
            conversation_state["messages"].append(AIMessage(
                content="",
                tool_calls=[{"id": _tool_call_id, "name": explicit_tool.name, "args": {}}]
            ))
            conversation_state["messages"].append(ToolMessage(
                content=tool_result,
                tool_call_id=_tool_call_id,
                name=explicit_tool.name
            ))
            # Append shashin link if one was built for this image result
            _final_link = locals().get("_shashin_link", "")
            conversation_state["messages"].append(AIMessage(content=summary_text + _final_link))

            return {
                "messages": conversation_state["messages"],
                "current_model": getattr(active_llm, 'model', 'unknown')
            }

        if orchestrator:
            try:
                from pathlib import Path

                last_model_file = Path(__file__).parent / "client/last_model.txt"

                if last_model_file.exists():
                    expected_model = last_model_file.read_text().strip()

                    actual_model = None
                    if hasattr(orchestrator.base_llm, 'model'):
                        actual_model = orchestrator.base_llm.model
                    elif hasattr(orchestrator.base_llm, 'model_name'):
                        actual_model = orchestrator.base_llm.model_name
                    elif hasattr(orchestrator.base_llm, 'model_path'):
                        actual_model = Path(orchestrator.base_llm.model_path).stem

                    if actual_model and expected_model != actual_model:
                        logger.info(f"🔄 Multi-agent out of sync!")
                        logger.info(f"   Expected (last_model.txt): {expected_model}")
                        logger.info(f"   Actual (orchestrator): {actual_model}")
                        logger.info(f"   Syncing to: {expected_model}")

                        from client.llm_backend import LLMBackendManager
                        fresh_llm = LLMBackendManager.create_llm(
                            expected_model,
                            temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
                            num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "2048"))
                        )

                        if hasattr(orchestrator, 'update_llm'):
                            orchestrator.update_llm(fresh_llm)
                        else:
                            logger.error(f"❌ orchestrator.update_llm() not found!")

            except Exception as e:
                logger.warning(f"⚠️ Multi-agent sync check failed: {e}")

        # ═══════════════════════════════════════════════════════════
        # VISION FOLLOW-UP — re-invoke vision if last response had an image
        # and the user is asking a follow-up question about it
        # ═══════════════════════════════════════════════════════════
        _last_b64 = conversation_state.get("_last_vision_b64")
        if _last_b64 and not explicit_tool:
            logger.info("[vision follow-up] 🖼️ Last image in context — re-invoking vision")
            if True:
                import httpx as _httpx_fu, base64 as _b64_fu, uuid as _uuid_fu, time as _time_fu
                _fu_prompt = user_message
                _fu_num_predict = 1000
                _ollama_url_fu = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
                _vision_model_fu = os.getenv("OLLAMA_VISION_MODEL", "qwen3-vl:8b-instruct")
                try:
                    async with _httpx_fu.AsyncClient(timeout=300.0) as _fvc:
                        _fvr = await _fvc.post(
                            f"{_ollama_url_fu}/api/chat",
                            json={
                                "model": _vision_model_fu,
                                "stream": False,
                                "options": {
                                    "num_predict": _fu_num_predict,
                                    "repeat_penalty": 1.3,
                                    "temperature": 0.3,
                                },
                                "messages": [{
                                    "role": "user",
                                    "content": _fu_prompt,
                                    "images": [_last_b64]
                                }]
                            }
                        )
                    _fvdata = _fvr.json()
                    _fu_text = _fvdata.get("message", {}).get("content", "").strip()
                    if _fu_text:
                        logger.info(f"[vision follow-up] 🖼️ Got response: {_fu_text[:80]}")
                        conversation_state["messages"].append(HumanMessage(content=user_message))
                        conversation_state["messages"].append(AIMessage(content=_fu_text))
                        return {
                            "messages": conversation_state["messages"],
                            "current_model": getattr(active_llm, "model", model_name)
                        }
                except Exception as _fu_err:
                    logger.warning(f"[vision follow-up] 🖼️ Failed: {_fu_err} — falling through to agent")

        # ═══════════════════════════════════════════════════════════
        # SKILL INJECTION — capture what was injected for multi-agent
        # ═══════════════════════════════════════════════════════════
        skill_context = None
        if skills_manager and skills_manager.all_skills:
            # Snapshot system message content before injection
            pre_inject_content = (
                conversation_state["messages"][0].content
                if conversation_state["messages"] and
                   hasattr(conversation_state["messages"][0], 'type') and
                   conversation_state["messages"][0].type == "system"
                else None
            )

            conversation_state["messages"] = await inject_relevant_skills_into_messages(
                skills_manager,
                user_message,
                conversation_state["messages"],
                logger
            )

            # Extract injected skill content so multi-agent can use it too
            post_inject_content = (
                conversation_state["messages"][0].content
                if conversation_state["messages"] and
                   hasattr(conversation_state["messages"][0], 'type') and
                   conversation_state["messages"][0].type == "system"
                else None
            )

            if post_inject_content and post_inject_content != pre_inject_content:
                # Pull out only the injected skill portion
                if pre_inject_content:
                    skill_context = post_inject_content[len(pre_inject_content):]
                else:
                    skill_context = post_inject_content
                logger.info("📚 Skill context captured for multi-agent")

        current_session_id = conversation_state.get("session_id")
        session_state = session_state_registry.get(current_session_id) if current_session_id else None

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
                result = await orchestrator.execute_a2a(user_message)

                if isinstance(result, dict):
                    result_text = result.get("response", str(result))
                    current_model = result.get("current_model", "unknown")
                    stopped = result.get("stopped", False)
                else:
                    result_text = result
                    current_model = "unknown"
                    stopped = False

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
                # Pass skill_context so orchestrator follows the skill workflow
                result = await orchestrator.execute(
                    user_message,
                    skill_context=skill_context,
                    session_id=conversation_state.get("session_id"),
                    rag_search_fn=_rag_search_turns if _CONV_RAG_AVAILABLE else None
                )

                if isinstance(result, dict):
                    result_text = result.get("response", str(result))
                    current_model = result.get("current_model", "unknown")
                    stopped = result.get("stopped", False)
                else:
                    result_text = result
                    current_model = "unknown"
                    stopped = False

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

            return await langgraph.run_agent(
                agent,
                conversation_state,
                user_message,
                logger,
                tools,
                system_prompt,
                llm,
                MAX_MESSAGE_HISTORY,
                session_state=session_state,
                capability_registry=capability_registry
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
    import socket
    try:
        # Connect to an external address to find the real outbound interface IP
        _s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        _s.connect(("8.8.8.8", 80))
        _host_ip = _s.getsockname()[0]
        _s.close()
    except Exception:
        _host_ip = "localhost"
    utils.open_browser_url(f"http://{_host_ip}:9000/client/ui/index.html")

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
        session_state_registry=session_state_registry,
        capability_registry=capability_registry,
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