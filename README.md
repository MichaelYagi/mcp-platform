# MCP Platform

Local MCP runtime with multi-agent orchestration, distributed tool servers, and ML-powered media recommendations.

⚠️ **Experimental** — intended for personal and experimental use only, not for production deployment.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [1. Quick Start](#1-quick-start)
  - [Install Dependencies](#install-dependencies)
  - [LLM Backend](#llm-backend)
  - [Start the Client](#start-the-client)
- [2. Using MCP Servers with Other Clients](#2-using-mcp-servers-with-other-clients)
- [3. Client Configuration](#3-client-configuration)
  - [Environment Variables](#environment-variables)
  - [Recommended Setup](#recommended-setup)
  - [Configuration Details](#configuration-details)
  - [Feature Requirements](#feature-requirements)
  - [Available Commands](#available-commands)
  - [API Setup](#api-setup)
  - [Google Setup](#google-setup)
- [4. Adding Custom Tools](#4-adding-custom-tools)
  - [@tool_meta field reference](#tool_meta-field-reference)
  - [Tag vocabulary](#tag-vocabulary)
  - [Add external MCP servers (optional)](#step-4-add-external-mcp-servers-optional)
- [5. Distributed Mode (A2A Protocol)](#5-distributed-mode-a2a-protocol)
- [6. Testing](#6-testing)
- [7. Architecture](#7-architecture)
- [8. Intent Patterns & Troubleshooting](#8-intent-patterns--troubleshooting)
  - [Intent Patterns](#intent-patterns)
  - [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Prerequisites

* Python 3.12+
* 16GB+ RAM recommended
* Ollama installed

---

## 1. Quick Start

Get the client running in 3 steps:

### Install Dependencies

Clone repo and do the following

```bash
cd mcp-platform

# Create virtual environment
python -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate

# Activate (Windows PowerShell)
.venv\Scripts\activate

# Install requirements - this will take a while
pip install -r requirements.txt
```

### LLM Backend

Ollama
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama server
ollama serve

# Download a model (use 14B+ for best results)
ollama pull qwen2.5:14b-instruct-q4_K_M
```
* See [Recommended Setup](#recommended-setup)

**Optional: GGUF (local model files)**
```bash
# Download a GGUF model (example)
wget https://huggingface.co/TheRains/Qwen2.5-14B-Instruct-Q4_K_M-GGUF/blob/main/qwen2.5-14b-instruct-q4_k_m.gguf

# Register the model
# (After starting client, use `:gguf add` command to the downloaded file)
```

### Start the Client
```bash
python client.py
```

Access web UI at: `http://localhost:9000/client/ui/index.html`

**That's it!** The client auto-discovers all MCP servers and tools.

---

## 2. Using MCP Servers with Other Clients

Use these MCP servers with Claude Desktop, Cline, or any MCP-compatible client.

### Example Configuration

Add to your MCP client config (e.g., `claude_desktop_config.json`):
```json
{
    "mcpServers": {
        "code_review": {
            "command": "/path/to/mcp_a2a/.venv/bin/python",
            "args": ["/path/to/mcp_a2a/servers/code_review/server.py"]
        },
        "location": {
            "command": "/path/to/mcp_a2a/.venv/bin/python",
            "args": ["/path/to/mcp_a2a/servers/location/server.py"]
        },
        "plex": {
            "command": "/path/to/mcp_a2a/.venv/bin/python",
            "args": ["/path/to/mcp_a2a/servers/plex/server.py"]
        },
        "rag": {
            "command": "/path/to/mcp_a2a/.venv/bin/python",
            "args": ["/path/to/mcp_a2a/servers/rag/server.py"]
        },
        "system": {
            "command": "/path/to/mcp_a2a/.venv/bin/python",
            "args": ["/path/to/mcp_a2a/servers/system_tools/server.py"]
        },
        "text": {
            "command": "/path/to/mcp_a2a/.venv/bin/python",
            "args": ["/path/to/mcp_a2a/servers/text/server.py"]
        }
    }
}
```

**Windows paths:**
```json
"command": "C:\\path\\to\\mcp_a2a\\.venv\\Scripts\\python.exe"
```

**Available servers:**
- `code_assistant` - AI-powered code assistance
- `code_review` - Code analysis (5 tools)
- `github` - GitHub integration
- `google` - Gmail + Google Calendar ⚠️ *Requires one-time OAuth setup — see [Google Setup](#google-setup)*
- `image` - Image search and analysis ⚠️ *Requires `SERPER_API_KEY`*
- `location` - Weather, time, location (3 tools)
- `plex` - Media library + ML recommendations (17 tools) ⚠️ *Requires `PLEX_URL`, `PLEX_TOKEN`*
- `rag` - Vector search (4 tools) ⚠️ *Requires Ollama + `bge-large`*
- `system` - System info (4 tools)
- `text` - Text processing (7 tools)
- `trilium` - Trilium notes integration ⚠️ *Requires `TRILIUM_URL`, `TRILIUM_TOKEN`*

---

## 3. Client Configuration

### Environment Variables

Create `.env` in project root:
```bash
# === LLM Backend ===
OLLAMA_VISION_MODEL=qwen3-vl:8b-instruct
MAX_MESSAGE_HISTORY=30          # Chat history limit (default: 20)
LLM_TEMPERATURE=0.3             # Model temperature 0 to 1 (default: 0.3)

# === GGUF Configuration (if using GGUF backend) ===
GGUF_GPU_LAYERS=-1              # -1 = all GPU, 0 = CPU only, N = N layers on GPU
GGUF_CONTEXT_SIZE=4096          # Context window size
GGUF_BATCH_SIZE=512             # Batch size for processing

# === API Keys (optional services) ===
PLEX_URL=http://localhost:32400  # Plex server URL
PLEX_TOKEN=your_token_here       # Get from Plex account settings
TRILIUM_URL=http://localhost:8888
TRILIUM_TOKEN=your_token_here
SHASHIN_BASE_URL=http://localhost:6624/
SHASHIN_API_KEY=your_key_here
SERPER_API_KEY=your_key_here     # Serper image search (https://serper.dev/api-keys)
OLLAMA_TOKEN=your_token_here     # Ollama API key (https://ollama.com/settings/keys)

# === A2A Protocol (optional distributed mode) ===
A2A_ENDPOINTS=http://localhost:8010  # Comma-separated endpoints
A2A_EXPOSED_TOOLS=                   # Tool categories to expose (empty = all)

# === Performance Tuning (optional) ===
CONCURRENT_LIMIT=3              # Parallel ingestion jobs (default: 1)
EMBEDDING_BATCH_SIZE=50         # Embeddings per batch (default: 20)
DB_FLUSH_BATCH_SIZE=50          # DB inserts per batch (default: 30)

# === Tool Control (optional) ===
DISABLED_TOOLS=plex:*  # Disable specific tools/categories

# === Location Default - uses your IP location otherwise (optional) ===
DEFAULT_CITY=Vancouver
DEFAULT_STATE=BC
DEFAULT_COUNTRY=Canada
```

### Recommended Setup

Use Ollama for easy setup. Download and install Ollama at https://ollama.com/download and run:

```ollama serve```

**Recommended LLM**

```ollama pull qwen2.5:14b-instruct-q4_K_M```

**RAG requires Ollama + bge-large:** If `bge-large` has not been pulled from Ollama, RAG ingestion and semantic search will not work.

```ollama pull bge-large```

**Image tools requires Ollama + vision models:** If a vision model has not been pulled from Ollama, image tools will not work.

```ollama pull qwen3-vl:8b-instruct```

A minimal `.env` to get started with the core features:

```env
# === Vision ===
OLLAMA_VISION_MODEL=qwen3-vl:8b-instruct

# === Disable unused servers ===
DISABLED_TOOLS=plex:*,image_tools:shashin_analyze,shashin_random,shashin_search

# === API Keys ===
OLLAMA_TOKEN=<token>      # Free at https://ollama.com — required for web_search_tool
SERPER_API_KEY=<key>      # Required for web_image_search_tool (https://serper.dev)
```

### Configuration Details

**LLM Backend:**
- `ollama`: Uses Ollama server (requires `ollama serve` running)
- `gguf`: Uses local GGUF model files (GPU recommended)

**GGUF GPU Layers:**
- `-1`: Use all GPU (fastest, requires model fits in VRAM)
- `0`: CPU only (slow but works with any model size)
- `20`: Use 20 layers on GPU (balance for large models on limited VRAM)

**Performance Tuning:**
- `EMBEDDING_BATCH_SIZE=50` + `DB_FLUSH_BATCH_SIZE=50` = ~6x faster RAG ingestion
- For 12GB VRAM, can increase to 100 for even faster processing
- `CONCURRENT_LIMIT=2` enables parallel media ingestion

**Disabled Tools:**
- Format: `category:tool_name` or `category:*`
- Example: `DISABLED_TOOLS=todo:delete_all_todo_items,system:*`
- Hidden from `:tools` list, return error if called

### Feature Requirements

Some features require additional setup before they will function. The table below summarizes what's needed:

| Feature | Required env vars | Additional setup |
|---------|-------------------|------------------|
| Gmail + Google Calendar | — | One-time Google OAuth setup — see [Google Setup](#google-setup) |
| RAG ingestion & search | — | Ollama running + `bge-large` pulled |
| RAG reranking (optional) | — | `bge-reranker-v2-m3` pulled — improves result ranking, falls back to cosine if absent |
| Plex media library | `PLEX_URL`, `PLEX_TOKEN` | Plex Media Server running |
| Plex ingestion & recommendations | `PLEX_URL`, `PLEX_TOKEN` | Ollama running + `bge-large` pulled |
| Ollama web search | `OLLAMA_TOKEN` | Ollama account + API key |
| Image search | `SERPER_API_KEY` | Serper account + API key (https://serper.dev) |
| Trilium notes | `TRILIUM_URL`, `TRILIUM_TOKEN` | Trilium server running |
| Shashin photo gallery | `SHASHIN_BASE_URL`, `SHASHIN_API_KEY` | Shashin server running |
| A2A distributed mode | `A2A_ENDPOINTS` | Remote A2A server running |

### Available Commands

These work in both CLI and web UI:
```
:commands              - List all available commands
:clear sessions        - Clear all chat history
:clear session <id>    - Clear session
:sessions              - List all sessions
:stop                  - Stop current operation
:stats                 - Show performance metrics
:tools                 - List available tools (hides disabled)
:tools --all           - Show all tools including disabled
:tool <n>              - Get tool description
:model                 - List all available models
:model <n>             - Switch to a model (auto-detects backend)
:models                - List models (legacy)
:gguf add <path>       - Register a GGUF model
:gguf remove <alias>   - Remove a GGUF model
:gguf list             - List registered GGUF models
:a2a on                - Enable agent-to-agent mode
:a2a off               - Disable agent-to-agent mode
:a2a status            - Check A2A system status
:health                - Health overview of all servers and tools
:env                   - Show environment configuration
```

### API Setup

**Ollama Search API (web search):**
1. Sign up at https://ollama.com/
2. Get API key from https://ollama.com/settings/keys
3. Add to `.env`: `OLLAMA_TOKEN=your_key`

**Plex Media Server:**
1. Open Plex web interface
2. Settings → Network → Show Advanced
3. Copy server URL (e.g., `http://192.168.1.100:32400`)
4. Get token: Settings → Account → Show XML → Copy `authToken`
5. Add to `.env`:
```bash
   PLEX_URL=http://your_server_ip:32400
   PLEX_TOKEN=your_token
```

> ⚠️ **Without `PLEX_URL` and `PLEX_TOKEN`**, all Plex tools (library browsing, ingestion, ML recommendations) will be unavailable. The server will load but calls will return a configuration error.

**Google (Gmail + Calendar):** See [Google Setup](#google-setup) below.

---

## Google Setup

One-time setup. After completing these steps the server runs headlessly — the refresh token does not expire.

### Step 1 — Google Cloud Console

1. Go to https://console.cloud.google.com/
2. Create a new project — name it anything (e.g. `mcp-platform`)
3. **APIs & Services → Library** — search and enable both:
   - **Gmail API**
   - **Google Calendar API**
4. **Google Auth Platform → Get started** (left menu)
   - Fill in app name and support email → Save
   - User type: **External**
5. **Google Auth Platform → Audience**
   - Click **Publish App** to set publishing status to **In Production**
   - This prevents refresh tokens from expiring every 7 days
6. **Google Auth Platform → Clients → Create Client**
   - Application type: **Desktop app** → Create
   - Click **Download JSON** on the new client
   - Save the file as `credentials.json`

### Step 2 — Place credentials.json

```
servers/google/credentials.json
```

### Step 3 — Install dependencies

```bash
pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client
```

### Step 4 — Authenticate

Make sure mcp-platform is stopped, then run from the project root:

```bash
.venv/bin/python auth_google.py
```

Where `auth_google.py` contains:

```python
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/calendar.events",
]

flow = InstalledAppFlow.from_client_secrets_file(
    "servers/google/credentials.json", SCOPES
)
creds = flow.run_local_server(port=0)

with open("servers/google/token.json", "w") as f:
    f.write(creds.to_json())

print("✅ token.json written")
```

A browser window will open. Sign in to Google → if you see an **unverified app** warning, click **Advanced** → **Go to mcp-platform (unsafe)** → grant permissions.

`token.json` is written to `servers/google/token.json`. Because the app is published to production, this token will not expire unless unused for 6 months or your Google password changes.

### Step 5 — Restart the client

```bash
python client.py
```

The google server is auto-discovered. Gmail and Calendar tools are now available.

> ⚠️ **If token becomes invalid** — delete `servers/google/token.json` and re-run the Step 4 script.

> ⚠️ **Refresh tokens can still expire** if you don't use Google tools for 6 months, or if you change your Google password.

---

## 4. Adding Custom Tools

### Step 1: Create the server file

```bash
mkdir servers/my_tool
touch servers/my_tool/server.py
```

### Step 2: Implement the tool with `@tool_meta`

This is the **only step that requires your input.** Everything else — routing, the tools panel, the capability registry — is automatic.

```python
# servers/my_tool/server.py
import sys
import json
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env", override=True)

from mcp.server.fastmcp import FastMCP
from tools.tool_control import check_tool_enabled

try:
    from client.tool_meta import tool_meta
except Exception:
    def tool_meta(**kwargs):
        def decorator(fn): return fn
        return decorator

LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_my_tool_server")

mcp = FastMCP("my-tool-server")


@mcp.tool()
@check_tool_enabled(category="my_tool")
@tool_meta(
    tags=["read", "search"],           # what the tool does — see tag vocabulary below
    triggers=["my keyword", "my phrase"],  # natural language that routes to this tool
    example='use my_function: arg1=""',    # pre-fill text shown in the tools panel
    text_fields=["content"],           # which response field contains the main text
)
def my_function(arg1: str, arg2: int = 0) -> str:
    """
    Short description of what this tool does.

    Args:
        arg1 (str): Description of arg1
        arg2 (int, optional): Description of arg2

    Returns:
        JSON string with results.
    """
    logger.info(f"🛠 my_function called: {arg1}")
    result = {"content": f"Processed {arg1} with {arg2}"}
    return json.dumps(result, indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio")
```

That's it. Restart the client and the tool is live — routed, badged, and registered automatically.

### `@tool_meta` field reference

| Field | Required | Description |
|-------|----------|-------------|
| `tags` | ✅ | Capability tags — see vocabulary below |
| `triggers` | ✅ | Natural language words/phrases that route to this tool |
| `example` | recommended | Pre-fill text shown in the tools panel UI |
| `text_fields` | if needed | Response fields containing real text content (e.g. `["content", "preview"]`). Only needed if your tool returns a list where the text is in an unusually named field |
| `rate_limit` | no | `"100/hour"`, `"10/day"`, `"ollama"`, or `None` |
| `idempotent` | no | `True` if calling twice with the same args has no side effects (default: `True`) |
| `intent_category` | no | Override routing group name — useful when multiple tools with the same tags need separate routing (e.g. `"shashin_search"` vs `"shashin_analyze"` both tagged `["media"]`) |

### Tag vocabulary

Tags serve two purposes:
1. **Routing** — when a query matches an intent, the router calls `capability_registry.filter_by_tags([...])` to find all tools in that group. Any new tool with the right tags is automatically included — no hardcoded tool name lists.
2. **Tools panel** — tags render as coloured badges on each tool card in the UI.

| Tag | Meaning |
|-----|---------|
| `read` | Tool only reads data, never writes |
| `write` | Tool creates or modifies data |
| `destructive` | Tool deletes or irreversibly changes data |
| `search` | Primary purpose is search or query |
| `external` | Calls an external API or service |
| `vision` | Processes image or visual input |
| `media` | Operates on audio/video/image files |
| `calendar` | Interacts with calendar data |
| `email` | Interacts with email |
| `notes` | Interacts with note-taking systems |
| `code` | Operates on source code |
| `system` | Interacts with the OS or hardware |
| `rag` | Interacts with the RAG vector store |
| `ai` | Calls an LLM or ML model |

### Step 3: Create skill documentation (optional)

```bash
mkdir -p servers/my_tool/skills
touch servers/my_tool/skills/my_feature.md
```

### Step 4: Add external MCP servers (optional)

To connect external or third-party MCP servers, create `mcp-platform/external_servers.json`.
The client auto-discovers this file on startup — no code changes needed.

**SSE transport** (remote HTTP event stream):
```json
{
    "external_servers": {
        "deepwiki": {
            "transport": "sse",
            "url": "https://mcp.deepwiki.com/mcp",
            "enabled": true
        }
    }
}
```

**HTTP transport** (streamable HTTP, e.g. authenticated APIs):
```json
{
    "external_servers": {
        "neon": {
            "transport": "http",
            "url": "https://mcp.neon.tech/mcp",
            "enabled": true,
            "headers": { "Authorization": "Bearer <$TOKEN>" }
        }
    }
}
```

**Header authentication** uses the `ES_{SERVER_NAME}_{PLACEHOLDER}` convention in `.env`:
```bash
# Server "mcpserver" with <$TOKEN>   → ES_MCPSERVER_TOKEN
# Server "mcpserver" with <$API_KEY> → ES_MCPSERVER_API_KEY
ES_MCPSERVER_TOKEN=your_token_here
ES_MCPSERVER_API_KEY=your_api_key_here
```

**Stdio transport** (local process servers):
```json
{
    "external_servers": {
        "pycharm": {
            "transport": "stdio",
            "command": "/usr/lib/jvm/jdk-17/bin/java",
            "args": ["-classpath", "/path/to/mcpserver.jar", "com.intellij.mcpserver.stdio.McpStdioRunnerKt"],
            "env": { "IJ_MCP_SERVER_PORT": "64342" },
            "enabled": true
        }
    }
}
```

**Field reference:**

| Field | Required | Description |
|-------|----------|-------------|
| `transport` | ✅ | `"sse"`, `"http"`, or `"stdio"` |
| `url` | SSE/HTTP only | Full URL to the endpoint |
| `headers` | No | Request headers — use `<$PLACEHOLDER>` for secrets |
| `command` | stdio only | Path to the executable |
| `args` | stdio only | Command-line arguments |
| `env` | No | Environment variables passed to the process |
| `cwd` | No | Working directory (defaults to project root) |
| `enabled` | No | `false` skips without removing (default: `true`) |
| `notes` | No | Human-readable description, ignored by client |

> **WSL2 note:** For stdio servers bridging to Windows, set `IJ_MCP_SERVER_HOST` in `env`
> to the Windows host IP (`cat /etc/resolv.conf | grep nameserver`).

### Step 6: Test & Deploy
```bash
python client.py   # restart to auto-discover new server
```

---

## 5. Distributed Mode (A2A Protocol)

Run tools on remote servers and expose them via HTTP.

### Setup A2A Server

```bash
# Terminal 1
python a2a_server.py        # starts on http://localhost:8010

# Terminal 2
python client.py            # auto-connects to A2A endpoints in .env
```

### Control Exposed Tools

```bash
# Expose specific categories (comma-separated)
A2A_EXPOSED_TOOLS=plex,location,text

# Expose everything (default)
A2A_EXPOSED_TOOLS=
```

**Security:** Exclude `plex` to protect personal data.

### Multi-Endpoint Support

```bash
A2A_ENDPOINTS=http://localhost:8010,http://gpu-server:8020
```

---

## 6. Testing

### Running Tests

```bash
pytest                              # all tests
pytest -m unit                      # fast unit tests only
pytest -m integration               # integration tests
pytest -m e2e                       # end-to-end tests
pytest -c tests/pytest.coverage.ini # with coverage
```

### Test Structure

```
tests/
├── conftest.py
├── pytest.ini
├── unit/
│   ├── test_session_manager.py
│   ├── test_models.py
│   ├── test_context_tracker.py
│   ├── test_intent_patterns.py
│   └── test_code_review_tools.py
├── integration/
│   ├── test_websocket_flow.py
│   └── test_langgraph_agent.py
├── e2e/
│   └── test_full_conversation.py
└── results/
    ├── junit.xml
    ├── coverage.xml
    ├── test-report.html
    └── coverage-report.html
```

### CI/CD Integration

**GitHub Actions:**
```yaml
- name: Run tests
  run: pytest
- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    files: tests/results/coverage.xml
```

---

## 7. Architecture

### Multi-Server Design

```
servers/
├── code_review/       5 tools  - Code analysis
├── google/            7 tools  - Gmail + Google Calendar       [requires one-time OAuth setup]
├── location/          3 tools  - Weather, time, location
├── plex/             17 tools  - Media + ML recommendations    [requires PLEX_URL + PLEX_TOKEN]
├── rag/               4 tools  - Vector search                 [requires Ollama + bge-large]
├── system/      4 tools  - System info
└── text/        7 tools  - Text processing
```

### Directory Structure
```
mcp_a2a/
├── servers/
├── a2a_server.py
├── client.py
├── client/
│   ├── ui/
│   │   ├── index.html
│   │   └── dashboard.html
│   ├── capability_registry.py  ← Tool capability index (auto-populated from @tool_meta)
│   ├── langgraph.py
│   ├── query_patterns.py       ← Intent routing (auto-populated from @tool_meta triggers)
│   ├── session_state.py        ← Per-session context store
│   ├── tool_meta.py            ← @tool_meta decorator — single source of truth for tool metadata
│   ├── search_client.py        ← Ollama web search & fetch
│   ├── websocket.py
│   └── ...
└── tools/
```

---

## 8. Intent Patterns & Troubleshooting

### Intent Patterns

The client routes queries to the right tools without sending all 75+ tools to the LLM on every message. Routing is driven by the `triggers` you define in each tool's `@tool_meta` decorator — no manual pattern editing required.

Each intent has a priority — lower number wins when multiple patterns match. The static entries below cover the built-in servers. Any tool you add with `@tool_meta(triggers=[...])` is automatically included on startup.

#### Overriding intent routing

Prefix your message with `Using <tool_name>,` to bypass pattern matching entirely and force a specific tool:

```
Using shashin_search_tool, find photos of Noah
Using web_image_search_tool, show me a picture of a red panda
```

#### Conversational bypass

Queries that start with personal statements (`"I like…"`, `"My favourite…"`), filler words (`"yes"`, `"thanks"`), creative tasks (`"write me a poem"`), or pronoun follow-ups (`"what did he do?"`, `"tell me more about them"`) bypass routing entirely — no tools are bound and the LLM answers from context.

#### Query not routing to the right tool?

- Use explicit phrasing: `"Using shashin_search_tool, find photos of Noah"` bypasses pattern matching entirely
- Check what triggers are registered by looking at `@tool_meta(triggers=[...])` on each tool
- Add more specific trigger phrases to the `@tool_meta` decorator on your tool

---

### Troubleshooting

**Google tools not working:**
- Confirm `servers/google/token.json` exists — if not, re-run the auth script in [Google Setup](#google-setup)
- If token is expired or invalid: delete `token.json` and re-run the auth script
- Confirm Gmail API and Google Calendar API are enabled in Google Cloud Console
- Confirm the OAuth app is published to **In Production** in Google Auth Platform → Audience

**Ollama models not appearing:**
```bash
ollama serve
ollama list
python client.py
```

**RAG not working / embedding errors:**
- Ensure Ollama is running: `ollama serve`
- Confirm `bge-large` is available: `ollama list`
- If missing, pull it: `ollama pull bge-large`
- RAG requires Ollama for embeddings regardless of which LLM backend (Ollama or GGUF) you use for chat

**Plex tools returning errors:**
- Confirm `PLEX_URL` and `PLEX_TOKEN` are set in `.env`
- Verify the Plex server is reachable: `curl $PLEX_URL/identity?X-Plex-Token=$PLEX_TOKEN`
- See [API Setup](#api-setup) for how to locate your token

**GGUF model won't load:**
- Check model size vs VRAM (use models <7GB for 12GB VRAM)
- Reduce GPU layers: `export GGUF_GPU_LAYERS=20`
- CPU only: `export GGUF_GPU_LAYERS=0`

**Web UI won't load:**
```bash
netstat -an | grep LISTEN   # check ports 8765, 8766, 9000
```

**A2A server not connecting:**
```bash
curl http://localhost:8010/.well-known/agent-card.json
```

**Ollama Search not working:**
- Verify `OLLAMA_TOKEN` in `.env`
- Get API key at https://ollama.com/settings/keys
- System falls back to LLM knowledge if unavailable

**RAG search returns wrong results:**
- RAG uses semantic similarity — returns closest matches even if not exact
- Check what's in the database: `> show rag stats`
- Content is only stored after researching URLs or manually adding via `rag_add_tool`

**RAG ingestion is slow:**
- Normal: ~2.5s for 16 chunks (10,000 characters)
- If slower, check Ollama is running: `ollama list`

**Conversation history not working:**
- Smaller models (≤7B) often refuse to answer questions about conversation history
- Switch to a larger model: `:model qwen2.5:14b-instruct-q4_K_M`
- Models with good instruction following: `qwen2.5:14b` (80-95%), `llama3.1:8b` (~70%), `mistral-nemo` (~70%)
- Avoid for this use case: `qwen2.5:3b`, `qwen2.5:7b` (~10-30%)

**Query not routing to the right tool:**
- Use explicit phrasing: `"Using shashin_search_tool, find photos of Noah"` bypasses pattern matching entirely
- Check the `triggers=[...]` in `@tool_meta` on the relevant tool and add more specific phrases
- Restart the client after changing triggers — they are registered at startup

**Tools not appearing:**
```bash
:tools --all        # check if disabled
# check DISABLED_TOOLS in .env
python client.py    # restart
```

---

## License

MIT License