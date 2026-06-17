"""
MCP Client - Main Entry Point (WITH MULTI-AGENT INTEGRATION + MULTI-A2A SUPPORT)
NOW WITH asyncio.gather() FOR PARALLEL A2A ENDPOINT REGISTRATION

Parallelism notes:
- register_all_a2a_endpoints(): all A2A endpoint registrations run concurrently.
- auto_discover_servers(): TCP reachability checks and OAuth probes already
  use asyncio.gather() (unchanged).
"""

import base64
import httpx
import json
import logging
import os
import re as _re
import socket
import sys
import asyncio
import time
import uuid

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

User: "remember that the server hostname is dev-box"
CORRECT: rag_add_tool(text="server hostname is dev-box", source="notes")
WRONG: add_todo_item(title="server hostname is dev-box") ❌

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
_USE_STEP_RE = _re.compile(r'use\s+(\w+)(?:\s*:\s*(.*))?', _re.IGNORECASE | _re.DOTALL)
_COND_RE = _re.compile(r'^check\s+(\w+)(?::([^i]+))?\s+if\s+(.+?)\s+then\s+(.+)$', _re.IGNORECASE | _re.DOTALL)
_KV_RE = _re.compile(r'(\w+)=(".*?"|\'.*?\'|[^\s,]+)')
_URL_RE = _re.compile(r'(https?://\S+)')
_PATH_RE = _re.compile(r'((?:/|[A-Za-z]:\\\\)\S+)')

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
    """Register tools from all A2A endpoints concurrently via asyncio.gather().

    All endpoint registrations run in parallel — network I/O for discovery and
    tool registration on each endpoint no longer blocks the others.
    """
    endpoints = parse_a2a_endpoints()

    if not endpoints:
        logger.info("ℹ️  No A2A endpoints configured")
        return {
            "endpoints": [],
            "successful": [],
            "failed": [],
            "total_tools_added": 0
        }

    logger.info(f"🌐 Attempting to register {len(endpoints)} A2A endpoint(s) concurrently")

    initial_tool_count = len(mcp_agent._tools)

    # Serialise mcp_agent._tools mutations with a lock so concurrent
    # registrations don't race when appending tools.
    tools_lock = asyncio.Lock()

    async def _register_one(i: int, endpoint: str) -> tuple:
        """Register a single endpoint; return (endpoint, success)."""
        logger.info(f"   [{i}/{len(endpoints)}] Connecting to: {endpoint}")
        try:
            a2a = A2AClient(endpoint)
            capabilities = await a2a.discover()
        except Exception as e:
            logger.error(f"   ❌ [{i}/{len(endpoints)}] A2A connection failed: {e}")
            return endpoint, False

        tool_defs = capabilities.get("tools", [])
        if not tool_defs:
            logger.warning(f"   ❌ [{i}/{len(endpoints)}] No tools returned from {endpoint}")
            return endpoint, False

        new_tools = [make_a2a_tool(a2a, t) for t in tool_defs]
        async with tools_lock:
            mcp_agent._tools.extend(new_tools)

        logger.info(f"   ✅ [{i}/{len(endpoints)}] Registered successfully (+{len(new_tools)} tools)")
        return endpoint, True

    registration_results = await asyncio.gather(
        *[_register_one(i, ep) for i, ep in enumerate(endpoints, 1)],
        return_exceptions=True
    )

    successful = []
    failed = []
    for outcome in registration_results:
        if isinstance(outcome, Exception):
            logger.error(f"   ❌ Endpoint registration raised: {outcome}")
            continue
        endpoint, success = outcome
        (successful if success else failed).append(endpoint)

    final_tool_count = len(mcp_agent._tools)
    total_new_tools = final_tool_count - initial_tool_count

    result = {
        "endpoints": endpoints,
        "successful": successful,
        "failed": failed,
        "total_tools_added": total_new_tools
    }

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
    Logs a warning and returns an empty string if the env var is not set,
    rather than injecting the literal placeholder text into auth headers.
    """
    env_key = f"ES_{server_name.upper()}_{placeholder.upper()}"
    value = os.getenv(env_key)
    if value is None:
        import logging as _rl
        _rl.getLogger("mcp_client").warning(
            f"⚠️  Header placeholder <${placeholder}> for server '{server_name}' "
            f"has no matching env var {env_key!r} — header value will be empty"
        )
        return ""
    return value


def resolve_headers(server_name: str, headers: dict) -> dict:
    """Resolve all <$PLACEHOLDER> values in headers for a given server."""
    import re
    resolved = {}
    for key, value in headers.items():
        def replacer(match):
            return resolve_env_placeholder(server_name, match.group(1))
        resolved[key] = re.sub(r'<\$([A-Z0-9_a-z]+)>', replacer, value)
    return resolved

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

    # Suppress noisy websockets handshake failures that occur when a browser
    # closes a connection mid-upgrade (normal on page refresh / reconnect).
    class _WsHandshakeFilter(logging.Filter):
        def filter(self, record):
            return "opening handshake failed" not in record.getMessage()
    logging.getLogger("websockets.server").addFilter(_WsHandshakeFilter())

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
            await utils.ensure_ollama_running(os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/"))
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
        num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "2048")), num_predict=int(os.getenv("OLLAMA_NUM_PREDICT", "2048")), repeat_penalty=float(os.getenv("OLLAMA_REPEAT_PENALTY", "1.1"))
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
                async def _run(**kwargs):
                    raise RuntimeError(f"Tool {t.name!r} is unavailable — MCP server '{source}' failed to initialize")
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

    # ─── Persistent Memory ───────────────────────────────────────────────────
    from client.memory_consolidator import (
        InactivityWatcher, inject_into_system_prompt, run_nightly_promotion,
    )

    async def _llm_fn_for_memory(system: str, user: str) -> str:
        """Background LLM call for memory consolidation.

        Acquires the background lock so consolidation jobs don't compete with
        each other or the scheduler, and caps output to 800 tokens (facts are
        short; 4096 is wasteful).
        """
        from langchain_core.messages import SystemMessage, HumanMessage
        from client.langgraph import llm_ainvoke as _llm_ainvoke
        from client.ollama_lock import background_ollama_call
        msgs = [SystemMessage(content=system), HumanMessage(content=user)]
        async with background_ollama_call():
            response = await _llm_ainvoke(llm, msgs, num_predict=800)
        return response.content if hasattr(response, "content") else str(response)

    inactivity_watcher = InactivityWatcher(_llm_fn_for_memory, session_manager)
    asyncio.create_task(inactivity_watcher.run())

    # Inject existing memories into the system prompt before agent starts
    # Use empty query on cold start — falls back to importance-sorted top-N
    SYSTEM_PROMPT = inject_into_system_prompt(SYSTEM_PROMPT)
    logger.info("🧠 Persistent memory layer initialized")

    # Nightly memory promotion loop
    async def _nightly_promotion_loop():
        while True:
            await asyncio.sleep(24 * 60 * 60)
            await run_nightly_promotion()
    asyncio.create_task(_nightly_promotion_loop())

    # ─── Proactive Agent Scheduler ───────────────────────────────────────────
    from client.proactive_agent import AgentScheduler, ScheduleParser
    from client.websocket import broadcast_proactive_result

    async def _process_image_result(tool_json: dict, tool_name: str) -> str:
        """Shared image post-processor for scheduled jobs. Fetches image, runs vision model."""
        from client.vision import call_vision_model

        _img_src  = tool_json.get("image_source")
        _img_orig = tool_json.get("image_source_original") or _img_src
        _b64img   = None
        summary_text = None

        if tool_json.get("image_base64"):
            _b64img = tool_json["image_base64"]
        elif _img_orig:
            try:
                async with httpx.AsyncClient(timeout=60.0) as _ic:
                    _ir = await _ic.get(_img_orig)
                _b64img = base64.b64encode(_ir.content).decode()
            except Exception as _fe:
                logger.warning(f"[image result] Failed to fetch image: {_fe}")

        if _b64img:
            try:
                _meta_parts = []
                for _k in ("takenAt", "placeName", "camera", "description", "fileName"):
                    _v = tool_json.get(_k)
                    if _v:
                        _meta_parts.append(f"{_k}: {_v}")
                _meta_str = "\n".join(_meta_parts)
                _vision_prompt = (
                    f"Describe this photo in 3-5 sentences. Include the setting, mood, "
                    f"and any notable subjects or details.\n\nPhoto metadata:\n{_meta_str}"
                )
                summary_text = await call_vision_model(_b64img, _vision_prompt, num_predict=300)
            except Exception as _ve:
                logger.warning(f"[image result] Vision failed: {_ve}")
                summary_text = None

        _image_id     = tool_json.get("image_id")
        _shashin_base = os.getenv("SHASHIN_BASE_URL", "").rstrip("/")
        _shashin_link = (
            f"\n\n[View in Shashin]({_shashin_base}/search?term={_image_id})"
            if _image_id and _shashin_base else ""
        )

        # Always prepend thumbnail image
        _img_line = f"![]({_img_src})" if _img_src else ""

        if summary_text:
            result_parts = [p for p in [_img_line, summary_text + _shashin_link] if p]
            return "\n\n".join(result_parts)

        # Fallback: show image inline + metadata
        _parts = [_img_line] if _img_line else []
        for _k, _label in [("placeName", "📍"), ("takenAt", "📅"), ("camera", "📷")]:
            _v = tool_json.get(_k)
            if _v:
                _parts.append(f"{_label} {_v}")
        if _shashin_link:
            _parts.append(_shashin_link)
        return "\n".join(p for p in _parts if p)

    async def _process_tool_result(tool_name: str, tool_result: str, arg_str: str, active_llm) -> str:
        """Shared post-processor for tool results.
        Handles images, plain text, structured JSON, lists, and LLM summarization.
        Used by both _tool_executor (scheduler) and run_agent_wrapper (direct dispatch)."""
        # Plain text passthrough
        _is_plain_text = not tool_result.strip().startswith(("{", "["))
        if _is_plain_text:
            return tool_result

        try:
            _tool_json = json.loads(tool_result)
        except Exception:
            return tool_result

        # Image result — vision model
        if isinstance(_tool_json, dict) and (_tool_json.get("image_source") or _tool_json.get("image_base64")):
            return await _process_image_result(_tool_json, tool_name)

        # Bare status result (e.g. discord_notify's {"status": "sent", "channel": ...,
        # "content": "..."}) — one "Key: value" line per short field, no LLM
        # summarization needed. The "content" field (what was actually sent) is
        # shown in full below the status lines, since that's the useful part.
        if isinstance(_tool_json, dict) and "status" in _tool_json and not any(
            isinstance(v, (list, dict)) for v in _tool_json.values()
        ) and all(
            k == "content" or len(str(v)) <= 200
            for k, v in _tool_json.items()
        ):
            _status_lines = [
                f"{k.replace('_', ' ').title()}: {v}"
                for k, v in _tool_json.items() if k != "content" and len(str(v)) <= 100
            ]
            _out = "\n".join(_status_lines)
            if _tool_json.get("content"):
                _out += f"\n\n{_tool_json['content']}"
            if _out:
                return _out

        # Pre-built summary passthrough
        if isinstance(_tool_json, dict) and "summary" in _tool_json and not any(
            isinstance(_tool_json.get(k), list) for k in
            ("documents", "sources", "results", "items", "records", "entries", "chunks")
        ):
            _presummary = _tool_json["summary"]
            _title = _tool_json.get("title", "")
            _url = _tool_json.get("url", "")
            header_parts = []
            if _title:
                header_parts.append(f"**{_title}**")
            if _url:
                header_parts.append(f"[{_url}]({_url})")
            header = " — ".join(header_parts)
            return f"{header}\n\n{_presummary}" if header else _presummary

        # List builder for array results
        list_lines = None
        try:
            parsed = _tool_json
            items = None
            array_key = None

            if isinstance(parsed, dict) and parsed.get("text") and isinstance(parsed["text"], str):
                return parsed["text"]

            if isinstance(parsed, list):
                items = parsed
                array_key = "results"
            elif isinstance(parsed, dict):
                for k, v in parsed.items():
                    if isinstance(v, list) and v and isinstance(v[0], dict):
                        items = v
                        array_key = k
                        break
                    elif isinstance(v, dict):
                        for k2, v2 in v.items():
                            if isinstance(v2, list) and v2 and isinstance(v2[0], dict):
                                items = v2
                                array_key = k2
                                break
                        if items:
                            break

            if items and isinstance(parsed, dict):
                _sibling_dicts = sum(1 for v in parsed.values() if isinstance(v, dict)) >= 2
                _is_weather = isinstance(parsed.get("current"), dict) and isinstance(parsed.get("forecast"), list)
                if _sibling_dicts or _is_weather:
                    items = None

            if items:
                total = len(items)
                label_word = array_key.rstrip("s") if array_key else "result"
                arg_label = f' for "{arg_str}"' if arg_str else ""
                _loc_parts = [p for p in [parsed.get("city",""), parsed.get("state",""), parsed.get("country","")] if p]
                _loc_prefix = f"📍 {', '.join(_loc_parts)}\n\n" if _loc_parts else ""
                list_lines = [f"{_loc_prefix}Found {total} {label_word}(s){arg_label}:\n"]

                for i, item in enumerate(items, 1):
                    if not isinstance(item, dict):
                        list_lines.append(f"{i}. {item}")
                        continue
                    title = (item.get("title") or item.get("name") or item.get("source")
                             or item.get("date") or item.get("id") or f"Item {i}")
                    score = item.get("score")
                    score_label = f" (score: {score:.2f})" if score is not None else ""
                    _text_val = (item.get("text") or item.get("snippet") or item.get("preview")
                                 or item.get("content") or item.get("contentPreview") or "")
                    summary = _text_val[:300] if _text_val else ""
                    list_lines.append(f"{i}. {title}{score_label}")
                    if summary:
                        list_lines.append(f"   {summary}")
                    list_lines.append("")

        except Exception:
            pass

        if list_lines:
            return "\n".join(list_lines)

        # LLM summarization fallback — cap at 400 tokens; summaries are short.
        try:
            from langchain_core.messages import SystemMessage, HumanMessage as _HM
            from client.langgraph import llm_ainvoke as _llm_ainvoke
            resp = await _llm_ainvoke(active_llm, [
                SystemMessage(content=TOOL_RESULT_PRESENT.format(tool_name=tool_name)),
                _HM(content=tool_result),
            ], num_predict=400)
            return resp.content if hasattr(resp, "content") else str(resp)
        except asyncio.CancelledError:
            logger.warning("🛑 _process_tool_result: LLM summarization cancelled")
            raise
        except Exception as _e:
            logger.error(f"[_process_tool_result] LLM summarization failed: {_e}")

        return tool_result

    def _md_to_html(md: str) -> str:
        """Convert markdown to HTML for email sending.
        Local network images are fetched and embedded as base64."""
        import urllib.request as _urllib

        def _embed_image(m):
            alt = m.group(1)
            url = m.group(2)
            if url.startswith("http://192.168.") or url.startswith("http://10.") or url.startswith("http://172."):
                try:
                    with _urllib.urlopen(url, timeout=10) as _resp:
                        _data = _resp.read()
                        _ct = _resp.headers.get("Content-Type", "image/jpeg").split(";")[0]
                    _b64_data = base64.b64encode(_data).decode()
                    src = f"data:{_ct};base64,{_b64_data}"
                except Exception:
                    src = url
            else:
                src = url
            return f'<img src="{src}" alt="{alt}" style="max-width:100%;border-radius:4px;margin:8px 0;"><br>'

        html = md
        html = _re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', _embed_image, html)
        html = _re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', html)
        html = _re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
        html = _re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)
        html = _re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=_re.MULTILINE)
        html = _re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=_re.MULTILINE)
        html = _re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=_re.MULTILINE)
        html = html.replace('\n', '<br>\n')
        return f"""<html><body style="font-family:sans-serif;max-width:700px;margin:auto;padding:16px;color:#222;">
{html}
</body></html>"""

    def _unwrap_tool_result(raw) -> str:
        """Extract plain text from TextContent objects or lists."""
        if raw is None:
            return ""
        if hasattr(raw, 'text'):
            return raw.text
        if hasattr(raw, 'content'):
            return _unwrap_tool_result(raw.content)
        if isinstance(raw, list):
            parts = []
            for item in raw:
                if hasattr(item, 'text'):
                    parts.append(item.text)
                elif isinstance(item, dict):
                    parts.append(item.get('text', str(item)))
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        s = str(raw)
        # Handle stringified TextContent: [TextContent(type='text', text='...', ...)]
        if s.startswith("[TextContent(") or s.startswith("TextContent("):
            _m = _re.search(r"text='(.*?)'(?:,\s*annotations|\))", s, _re.DOTALL)
            if _m:
                text = _m.group(1).replace("\\'", "'")
                # Unescape Python string escapes written into the repr (\n, \t, etc.)
                try:
                    if "\\n" in text or "\\t" in text:
                        text = text.encode("utf-8").decode("unicode_escape").encode("latin-1").decode("utf-8")
                except Exception:
                    pass
                return text
        return s

    async def _run_pipeline(pipe_parts: list, tools_list: list, initial_result: str = None) -> str:
        """Execute a >>-separated tool chain using _tool_executor for each step.
        Identical execution path to the scheduler pipeline — no extra transformation."""
        _STEP_RE = _USE_STEP_RE
        _NOTIF = ("discord_notify", "gmail_reply_tool")
        _EMAIL_SEND = ("gmail_send_email",)
        previous = initial_result  # allow pre-seeding with condition check result
        steps = [s.strip() for s in pipe_parts if s.strip()]
        for idx, step in enumerate(steps):
            m = _STEP_RE.match(step)
            if not m:
                continue
            tool_name = m.group(1)
            args_str = (m.group(2) or "").strip()

            # Parse explicit args from step
            args: dict = {}
            if args_str:
                try:
                    import json as _jj2
                    args = _jj2.loads(args_str)
                except Exception:
                    for km in _re.finditer(r'(\w+)\s*=\s*"((?:[^"\\]|\\.)*)"', args_str):
                        args[km.group(1)] = km.group(2).replace('\\"', '"')

            # Drop empty string values — unfilled template placeholders
            args = {k: v for k, v in args.items() if v != ""}

            # Inject previous result as content for next step
            if previous:
                _is_notif = any(k in tool_name for k in _NOTIF)
                _is_email = "gmail_send" in tool_name or "gmail_reply" in tool_name
                _has_content = any(k in args for k in ("message", "body", "content", "text", "input"))

                # Try to extract structured fields from previous JSON result
                _prev_data = {}
                try:
                    import json as _pj2
                    _prev_str = str(previous).strip().strip("'\"").replace('\\n', '\n')
                    if _prev_str.startswith('{'):
                        _prev_data = _pj2.loads(_prev_str)
                except Exception:
                    pass

                # Tool-specific injection from previous result fields
                if tool_name == "gmail_reply_tool" and "message_id" not in args:
                    _mid = _prev_data.get("id") or _prev_data.get("message_id")
                    if _mid:
                        args["message_id"] = str(_mid)
                elif tool_name == "calendar_create_event":
                    if "summary" not in args:
                        args["summary"] = _prev_data.get("title") or _prev_data.get("summary") or str(previous)[:50]
                    if "start" not in args:
                        args["start"] = _prev_data.get("start") or _prev_data.get("date") or ""
                    if "end" not in args:
                        args["end"] = _prev_data.get("end") or _prev_data.get("date") or ""
                elif tool_name == "create_note":
                    if "title" not in args:
                        args["title"] = _prev_data.get("title") or str(previous)[:80]
                    if "parent_note_id" not in args:
                        args.setdefault("parent_note_id", "root")
                    if "content" not in args:
                        args["content"] = str(previous)
                elif tool_name == "scene_locator_tool":
                    if "media_id" not in args:
                        _mid = _prev_data.get("id") or _prev_data.get("media_id") or _prev_data.get("ratingKey")
                        if _mid:
                            args["media_id"] = str(_mid)
                    if "query" not in args:
                        args["query"] = str(previous)[:100]

                # Resolve pipe_targets for this tool from its description sentinel
                _tool_obj = next((t for t in tools_list if getattr(t, "name", "") == tool_name), None)
                _pipe_targets: dict = {}
                if _tool_obj and getattr(_tool_obj, "description", None):
                    import re as _pt_re, json as _pt_json
                    _pt_m = _pt_re.search(r'__pipe_targets__:\s*(\{[^}]*\})', _tool_obj.description)
                    if _pt_m:
                        try:
                            _pipe_targets = _pt_json.loads(_pt_m.group(1))
                        except Exception:
                            pass

                # Generic content injection
                if not _has_content:
                    if _is_notif:
                        _msg = str(previous)
                        if "discord" in tool_name and len(_msg) > 1900:
                            _msg = _msg[:1900] + "\n…(truncated)"
                        args["message"] = _msg
                    elif "gmail_send" in tool_name:
                        args["body"] = str(previous)
                        if "subject" not in args:
                            args["subject"] = "Message"
                    elif _is_email:
                        args["body"] = str(previous)
                    elif tool_name not in ("gmail_reply_tool", "calendar_create_event",
                                           "create_note", "scene_locator_tool"):
                        # Only inject if the tool declares a matching pipe_targets entry.
                        # Tools with no pipe_targets run independently — don't force-feed them.
                        _text_param = next(
                            (param for param, accepts in _pipe_targets.items() if accepts == "text"),
                            None
                        )
                        _image_id_param = next(
                            (param for param, accepts in _pipe_targets.items() if accepts == "image_id"),
                            None
                        )
                        _image_url_param = next(
                            (param for param, accepts in _pipe_targets.items() if accepts == "image_url"),
                            None
                        )
                        if _image_id_param:
                            _iid = _prev_data.get("image_id")
                            if _iid:
                                args[_image_id_param] = str(_iid)
                        elif _image_url_param:
                            _iurl = (_prev_data.get("image_url")
                                     or _prev_data.get("image_source")
                                     or _prev_data.get("url"))
                            if not _iurl:
                                # Previous result might be a bare URL string
                                _prev_stripped = str(previous).strip()
                                if _prev_stripped.startswith("http"):
                                    _iurl = _prev_stripped
                            if _iurl:
                                args[_image_url_param] = str(_iurl)
                        elif _text_param:
                            args[_text_param] = str(previous)
                elif "gmail_send" in tool_name and "message" in args and "body" not in args:
                    args["body"] = args.pop("message")

            # Execute via _tool_executor for all steps
            _is_notif_step = any(k in tool_name for k in _NOTIF)
            logger.info(f"🔀 Pipeline step {idx+1}/{len(steps)}: {tool_name}({args})")
            previous = await _tool_executor(tool_name, args)
            logger.info(f"🔀 Pipeline step {idx+1} done: {str(previous)[:80]}")
            # Abort pipeline if a non-final step errored — don't pass error strings downstream
            if idx < len(steps) - 1 and (str(previous).startswith("Tool ") and ("error:" in str(previous).lower() or "not found." in str(previous))):
                logger.warning(f"🔀 Pipeline aborted at step {idx+1}: {str(previous)[:120]}")
                return str(previous)

        # Clean UI result
        last_tool = steps[-1].split()[1].split(":")[0] if steps and len(steps[-1].split()) > 1 else ""
        _last_errored = (
            str(previous).startswith("Tool ")
            and ("error:" in str(previous).lower() or "not found." in str(previous))
        )
        if not _last_errored and (any(nt in last_tool for nt in _NOTIF) or any(nt in last_tool for nt in _EMAIL_SEND)):
            return "Job completed."
        return str(previous) if previous else "Done."


    async def _tool_executor(tool_name: str, args: dict) -> str:
        """Execute a named tool via the same path as direct dispatch."""
        _current_tools = mcp_agent._tools

        # Auto-convert markdown to HTML for gmail_send_email
        if tool_name == "gmail_send_email" and args.get("body"):
            args = dict(args)
            # Keep the small /thumbnails/225/ image — _md_to_html embeds images as
            # base64 data URIs, and the /thumbnails/original/ size blows past Gmail's
            # "Limit Exceeded: Email Body Size" cap.
            args["html"] = True
            args["body"] = _md_to_html(args["body"])

        for tool in _current_tools:
            if tool.name == tool_name:
                try:
                    if args and any('\n' in str(v) or len(str(v)) > 200 for v in args.values()):
                        raw = await tool.ainvoke(args)
                        result_str = _unwrap_tool_result(raw)
                        arg_str = " ".join(f'{k}="{v}"' for k, v in args.items()) if args else ""
                        return await _process_tool_result(tool_name, result_str, arg_str, llm)
                    arg_str = " ".join(f'{k}="{v}"' for k, v in args.items()) if args else ""
                    result_str = await _invoke_tool_directly(tool, arg_str, logger)
                    return await _process_tool_result(tool_name, result_str, arg_str, llm)
                except Exception as e:
                    err_str = str(e)
                    try:
                        from pydantic import ValidationError as _PydanticVE
                        if isinstance(e, _PydanticVE):
                            missing = [str(err["loc"][0]) for err in e.errors() if err.get("type") == "missing"]
                            if missing:
                                err_str = f"Missing required field(s): {', '.join(missing)}"
                    except Exception:
                        pass
                    logger.error(f"[_tool_executor] {tool_name} error: {err_str}")
                    return f"Tool {tool_name} error: {err_str}"
        available = [t.name for t in _current_tools]
        logger.error(f"[_tool_executor] '{tool_name}' not found. Available ({len(available)}): {', '.join(available)}")
        return f"Tool '{tool_name}' not found."

    async def _scheduler_llm_fn(prompt: str) -> str:
        """Background LLM call for scheduled-job post-processing.

        Acquires the background lock and caps output to 600 tokens.
        """
        from langchain_core.messages import HumanMessage
        from client.langgraph import llm_ainvoke as _llm_ainvoke
        from client.ollama_lock import background_ollama_call
        async with background_ollama_call():
            response = await _llm_ainvoke(llm, [HumanMessage(content=prompt)], num_predict=600)
        return response.content if hasattr(response, "content") else str(response)

    async def _scheduler_agent_fn(prompt: str) -> str:
        """Full agent run for scheduled jobs with llm_prompt — gives the LLM tool access
        so it can execute compound actions (e.g. get briefing AND send to Discord).
        Multi-agent and A2A are suppressed — scheduler jobs always use single-agent execution.
        """
        _state = {"messages": [], "loop_count": 0}
        result = await run_agent_wrapper(
            agent,
            _state,
            prompt,
            logger,
            tools,
            suppress_multi_agent=True,
        )
        if isinstance(result, dict):
            msgs = result.get("messages", [])
            if msgs:
                last = msgs[-1]
                return last.content if hasattr(last, "content") else str(last)
        return str(result)

    agent_scheduler = AgentScheduler(
        execute_fn=_tool_executor,
        broadcast_fn=broadcast_proactive_result,
        llm_fn=_scheduler_llm_fn,
        session_manager=session_manager,
        agent_fn=_scheduler_agent_fn,
    )
    await agent_scheduler.start()

    async def _schedule_parser_llm_fn(system: str, user: str) -> str:
        """Background LLM call for schedule-request parsing.

        Uses temperature=0.0 — this is a structured-extraction task (the model
        must return ONLY a JSON object), so compliant/deterministic output
        matters far more than creative variance. Mirrors the routing classifier's
        temperature choice in client/langgraph.py:_get_routing_llm.
        """
        from langchain_core.messages import SystemMessage, HumanMessage
        from client.langgraph import llm_ainvoke as _llm_ainvoke
        from client.ollama_lock import background_ollama_call
        msgs = [SystemMessage(content=system), HumanMessage(content=user)]
        async with background_ollama_call():
            response = await _llm_ainvoke(llm, msgs, num_predict=800, temperature=0.0)
        return response.content if hasattr(response, "content") else str(response)

    tool_names = [t.name for t in mcp_agent._tools]
    schedule_parser = ScheduleParser(
        llm_fn=_schedule_parser_llm_fn,
        available_tools=tool_names,
        default_timezone=os.getenv("DEFAULT_TIMEZONE", "America/Vancouver"),
    )
    logger.info("⏰ Proactive agent scheduler initialized")

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
                            try:
                                import_result = json.loads(_unwrap_tool_result(import_result_raw))
                            except Exception:
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
                                    try:
                                        train_result = json.loads(_unwrap_tool_result(train_result_raw))
                                    except Exception:
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
        _re.IGNORECASE | _re.DOTALL
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
            # Pre-pass: if arg_str starts with key="value" where the value spans
            # multiple lines or contains embedded double quotes, _KV_RE would stop
            # at the first inner quote. Instead, find the closing " that is
            # immediately followed by optional whitespace then [ (next bracket arg)
            # or end of string — that boundary is unambiguous in tool templates.
            _LEADING_QUOTED_KV = _re.compile(
                r'^(\w+)="(.*?)"\s*(?=\[|\Z)', _re.DOTALL
            )
            _pre_match = _LEADING_QUOTED_KV.match(arg_str.strip())
            _pre_extracted: dict = {}
            _kv_input = arg_str
            if _pre_match:
                _pre_extracted[_pre_match.group(1)] = _pre_match.group(2)
                _kv_input = arg_str.strip()[_pre_match.end():].strip()

            # Check for key=value syntax: "key1=val1 key2=val2" or "key1=val1, key2=val2"
            # Simple pattern: word=anything-up-to-next-word= or end
            kv_matches = _KV_RE.findall(_kv_input)
            if kv_matches or _pre_extracted:
                tool_args = dict(_pre_extracted)
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
                _url_match  = _URL_RE.search(arg_str)
                _path_match = _PATH_RE.search(arg_str)
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
        return _unwrap_tool_result(result)

    # Create enhanced agent runner with multi-agent support
    async def run_agent_wrapper(agent, conversation_state, user_message, logger, tools, system_prompt=None, suppress_multi_agent=False, stream_callback=None):
        """Enhanced agent runner with multi-agent, A2A, and skills support"""

        # Bail immediately if a stop was requested — before any pre-checks
        # (memory search, skill injection, multi-agent routing) run.
        from client.stop_signal import is_stop_requested as _is_stop_requested
        if _is_stop_requested():
            logger.warning("🛑 run_agent_wrapper: stop signal set on entry — aborting")
            raise asyncio.CancelledError("run_agent_wrapper aborted: stop was requested")

        # Use provided system_prompt or fallback to global SYSTEM_PROMPT
        if system_prompt is None:
            system_prompt = SYSTEM_PROMPT

        # ── Explicit tool dispatch ──────────────────────────────────────
        # When the user types "use <tool_name>[: <args>]" we call the tool
        # directly and hand the result straight to the LLM for summarization.
        # This bypasses LangGraph tool-call generation entirely, which small
        # models (llama3.2:3b etc.) handle poorly.

        # ── Condition dispatch ─────────────────────────────────────────
        # "check <tool> if <expr> then use <tool2>" — run check tool, evaluate
        # condition, only run action tool if condition is true.
        _cond_match = _COND_RE.match(user_message.strip())
        if _cond_match:
            _check_tool_name = _cond_match.group(1).strip()
            _check_args_str  = (_cond_match.group(2) or "").strip()
            _cond_expr       = _cond_match.group(3).strip()
            _action_str      = _cond_match.group(4).strip()

            # Parse check tool args
            _check_args: dict = {}
            if _check_args_str:
                try:
                    import json as _cj
                    _check_args = _cj.loads(_check_args_str)
                except Exception:
                    for _km in _re.finditer(r'(\w+)\s*=\s*"((?:[^"\\]|\\.)*)"', _check_args_str):
                        _check_args[_km.group(1)] = _km.group(2)

            # Run check tool
            logger.info(f"🔍 Condition check: {_check_tool_name}({_check_args}) if {_cond_expr!r}")
            _check_result = await _tool_executor(_check_tool_name, _check_args)

            # Evaluate condition
            _cond_fired = False
            try:
                import json as _cj2
                _cond_data = _cj2.loads(_check_result) if isinstance(_check_result, str) else {}
                _cond_vars = {"result": _check_result, "result_len": len(str(_check_result)), "data": _cond_data}
                if isinstance(_cond_data, dict):
                    for _k, _v in _cond_data.items():
                        if isinstance(_v, (int, float, bool, str)):
                            _cond_vars[_k] = _v
                        elif isinstance(_v, list):
                            _cond_vars[f"len_{_k}"] = len(_v)
                elif isinstance(_cond_data, list):
                    _cond_vars["result"] = len(_cond_data)
                _cond_fired = bool(eval(_cond_expr, {"__builtins__": {}}, _cond_vars))
            except Exception as _ce:
                logger.warning(f"🔍 Condition eval failed: {_ce}")
                _cond_fired = False

            if _cond_fired:
                logger.info(f"🔍 Condition TRUE — running action: {_action_str}")
                # Parse action as pipeline or single use step
                _action_parts = [p.strip() for p in _action_str.split(">>") if p.strip()]
                # Inject check result as first previous_result
                _action_result = await _run_pipeline(_action_parts, tools, initial_result=_check_result)
                _cond_output = _action_result
            else:
                logger.info(f"🔍 Condition FALSE — no action taken")
                _cond_output = "Condition not met — no action taken."

            conversation_state["messages"].append(HumanMessage(content=user_message))
            conversation_state["messages"].append(AIMessage(content=_cond_output))
            return {
                "messages": conversation_state["messages"],
                "current_model": "condition",
                "multi_agent": False,
                "a2a": False,
            }

        # ── Pipeline dispatch ──────────────────────────────────────────
        # "use tool1 >> use tool2: arg=val" runs tools in sequence without LLM.
        _PIPE_STEP_RE = _USE_STEP_RE
        _pipe_parts = [p.strip() for p in user_message.split(">>") if p.strip()]
        _is_pipeline = (
            len(_pipe_parts) > 1 and
            all(_PIPE_STEP_RE.match(p) for p in _pipe_parts)
        )
        if _is_pipeline:
            _pipeline_result = await _run_pipeline(_pipe_parts, tools)
            conversation_state["messages"].append(HumanMessage(content=user_message))
            conversation_state["messages"].append(AIMessage(content=_pipeline_result))
            return {
                "messages": conversation_state["messages"],
                "current_model": "pipeline",
                "multi_agent": False,
                "a2a": False,
            }

        explicit_tool, explicit_arg = parse_explicit_tool(user_message, tools)
        if explicit_tool:
            logger.info(f"🎯 Explicit tool dispatch: {explicit_tool.name!r} arg={explicit_arg!r}")

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
                        num_ctx=int(os.getenv('OLLAMA_NUM_CTX', '2048')), num_predict=int(os.getenv('OLLAMA_NUM_PREDICT', '2048')), repeat_penalty=float(os.getenv('OLLAMA_REPEAT_PENALTY', '1.1'))
                    )
                else:
                    active_llm = orchestrator.base_llm if orchestrator and hasattr(orchestrator, 'base_llm') else llm
            except Exception as _model_err:
                logger.warning(f'⚠️ Could not load model from last_model.txt: {_model_err}')
                active_llm = orchestrator.base_llm if orchestrator and hasattr(orchestrator, 'base_llm') else llm
            logger.info(f"🤖 Explicit dispatch model: {getattr(active_llm, 'model', 'unknown')}")

            start = time.time()
            try:
                tool_result = await _invoke_tool_directly(explicit_tool, explicit_arg, logger)
                _duration = time.time() - start
                logger.info(f"✅ Direct tool {explicit_tool.name} completed in {_duration:.2f}s")
                # Record into metrics so dashboard reflects direct dispatch calls
                if _client_metrics is not None:
                    _client_metrics["tool_calls"][explicit_tool.name] += 1
                    _client_metrics["tool_times"][explicit_tool.name].append((time.time(), _duration))
            except Exception as e:
                _duration = time.time() - start
                logger.error(f"❌ Direct tool {explicit_tool.name} failed: {e}")
                tool_result = f"Error running {explicit_tool.name}: {e}"
                if _client_metrics is not None:
                    _client_metrics["tool_calls"][explicit_tool.name] += 1
                    _client_metrics["tool_errors"][explicit_tool.name] += 1
                    _client_metrics["tool_times"][explicit_tool.name].append((time.time(), _duration))

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
                _pt_call_id = str(uuid.uuid4())
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
            try:
                _tool_json = json.loads(tool_result)
            except Exception:
                _tool_json = None

            if isinstance(_tool_json, dict) and (_tool_json.get("image_source") or _tool_json.get("image_base64")):
                # For generate_image_tool — skip vision, just show image + metadata
                if explicit_tool.name == "generate_image_tool":
                    _seed = _tool_json.get("seed", "")
                    _model_used = _tool_json.get("model", "")
                    summary_text = f"🌱 Seed: `{_seed}` · Model: `{_model_used}`"
                    _gen_call_id = str(uuid.uuid4())
                    conversation_state["messages"].append(HumanMessage(content=user_message))
                    conversation_state["messages"].append(AIMessage(
                        content="",
                        tool_calls=[{"id": _gen_call_id, "name": explicit_tool.name, "args": {}}]
                    ))
                    conversation_state["messages"].append(ToolMessage(
                        content=tool_result,
                        tool_call_id=_gen_call_id,
                        name=explicit_tool.name
                    ))
                    conversation_state["messages"].append(AIMessage(content=summary_text))
                    return {
                        "messages": conversation_state["messages"],
                        "current_model": getattr(active_llm, "model", "unknown")
                    }

                logger.info("[direct dispatch] 🖼️ Image result — delegating to vision")
                from client.vision import call_vision_model as _call_vision
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
                        async with httpx.AsyncClient(timeout=60.0) as _hc:
                            _ir = await _hc.get(_img_orig, headers=_fetch_hdrs)
                            _ir.raise_for_status()
                        _b64img = base64.b64encode(_ir.content).decode("utf-8")
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

                    logger.info(f"[direct dispatch] 🖼️ Calling vision model (num_predict={_num_predict}, query={bool(_extra)})")
                    _vision_llm_start = time.time()
                    summary_text = await _call_vision(_b64img, _vision_prompt, num_predict=_num_predict)
                    if _client_metrics is not None:
                        _client_metrics["llm_calls"] += 1
                        _client_metrics["llm_times"].append((time.time(), time.time() - _vision_llm_start))
                    if not summary_text:
                        logger.warning("[direct dispatch] 🖼️ Vision returned empty content")
                    logger.info(f"[direct dispatch] 🖼️ Vision description: {summary_text[:80] if summary_text else ''}")


                except Exception as _ve:
                    logger.warning(f"[direct dispatch] 🖼️ Vision failed: {type(_ve).__name__}: {_ve} — falling back to metadata summary")
                    summary_text = None  # fall through to normal summarisation below

                # Build search link — appended regardless of vision success/failure
                _image_id = _tool_json.get("image_id")
                _shashin_base = os.getenv("SHASHIN_BASE_URL", "").rstrip("/")
                _shashin_link = (
                    f"\n\n[View in Shashin]({_shashin_base}/search?term={_image_id})"
                    if _image_id and _shashin_base else ""
                )

                if summary_text:
                    summary_text += _shashin_link
                    _vis_call_id = str(uuid.uuid4())
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

            # Route through shared post-processor (same as scheduler path)
            summary_text = await _process_tool_result(explicit_tool.name, tool_result, explicit_arg, active_llm)

            # Persist the exchange in conversation history.
            # Include a ToolMessage so websocket image/place scanner finds image_source.
            _tool_call_id = str(uuid.uuid4())
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

                last_model_file = PROJECT_ROOT / "client" / "last_model.txt"

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
                            num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "2048")), num_predict=int(os.getenv("OLLAMA_NUM_PREDICT", "2048")), repeat_penalty=float(os.getenv("OLLAMA_REPEAT_PENALTY", "1.1"))
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
            # Only re-invoke vision when the message looks like a visual question.
            # The old `if True:` here caused every subsequent message — including
            # completely unrelated ones — to be routed to the vision model with
            # the stale image, wasting time and returning nonsense answers.
            _VISUAL_PHRASES = (
                "what", "where", "who", "describe", "show", "tell me", "explain",
                "how many", "is there", "is it", "are there", "can you see", "read",
                "translate", "count", "color", "colour", "look", "identify",
                "find in", "in the image", "in the photo", "in this", "background",
                "foreground", "text in", "writing", "sign", "any ", "do you see",
            )
            _is_visual_followup = any(
                p in user_message.lower() for p in _VISUAL_PHRASES
            )
            if not _is_visual_followup:
                # Unrelated message — clear stale image so it doesn't linger
                conversation_state.pop("_last_vision_b64", None)
                conversation_state.pop("_last_vision_url", None)
                conversation_state.pop("_last_vision_tool_result", None)
            if _is_visual_followup:
                logger.info("[vision follow-up] 🖼️ Visual question detected — re-invoking vision")
                from client.vision import call_vision_model as _call_vision_fu
                try:
                    _fu_text = await _call_vision_fu(_last_b64, user_message, num_predict=1000)
                    if _fu_text:
                        logger.info(f"[vision follow-up] 🖼️ Got response: {_fu_text[:80]}")
                        conversation_state["messages"].append(HumanMessage(content=user_message))
                        conversation_state["messages"].append(AIMessage(content=_fu_text))
                        return {
                            "messages": conversation_state["messages"],
                            "current_model": model_name
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
        _is_explicit = user_message.lstrip().lower().startswith("use ")
        use_a2a = (
                not suppress_multi_agent and
                not _is_explicit and
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
                not suppress_multi_agent and
                not _is_explicit and
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
                capability_registry=capability_registry,
                stream_callback=stream_callback,
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
        proactive_agent={"scheduler": agent_scheduler, "parser": schedule_parser, "llm_fn": _scheduler_llm_fn},
        inactivity_watcher=inactivity_watcher,
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

    # ── Google re-auth notifier ───────────────────────────────────────────────
    # If a Google server wrote auth_pending.json at startup (invalid token),
    # wait for the first browser client to connect then push the auth URL into chat.
    # Uses a dedicated 'google_auth_required' message type so the frontend can
    # re-inject it after session_loaded wipes the chat.
    _AUTH_PENDING_FILE = PROJECT_ROOT / "auth_pending.json"

    async def _google_auth_notifier():
        if not _AUTH_PENDING_FILE.exists():
            return
        # Wait up to 30s for at least one browser client to connect
        for _ in range(60):
            if websocket.CONNECTED_WEBSOCKETS:
                break
            await asyncio.sleep(0.5)
        else:
            return  # No client connected in time — URL is still in the log

        # Extra delay to let the session fully load before broadcasting
        await asyncio.sleep(3.0)

        try:
            pending = json.loads(_AUTH_PENDING_FILE.read_text())
            auth_url = pending.get("auth_url", "")
            if not auth_url:
                return
            from client.websocket import broadcast_message as _ws_broadcast
            await _ws_broadcast("google_auth_required", {"auth_url": auth_url})
            logger.info("🔑 Google auth URL broadcast to UI")
            _AUTH_PENDING_FILE.unlink(missing_ok=True)
        except Exception as _e:
            logger.warning(f"🔑 Could not broadcast auth URL: {_e}")

    asyncio.create_task(_google_auth_notifier())

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