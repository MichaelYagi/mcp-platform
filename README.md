# MCP Platform

Local MCP runtime with multi-agent orchestration, distributed tool servers, and ML-powered media recommendations.

⚠️ **Experimental** — intended for personal and experimental use only, not for production deployment.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [1. Quick Start](#1-quick-start)
- [2. Using MCP Servers with Other Clients](#2-using-mcp-servers-with-other-clients)
- [3. Client Configuration](#3-client-configuration)
- [4. Adding Custom Tools](#4-adding-custom-tools)
- [5. Distributed Mode (A2A Protocol)](#5-distributed-mode-a2a-protocol)
- [6. Testing](#6-testing)
- [7. Architecture](#7-architecture)
- [8. Intent Patterns & Troubleshooting](#8-intent-patterns--troubleshooting)
- [License](#license)

---

## Prerequisites

* Python 3.12+
* Node.js 18+ (for JS tests)
* 16GB+ RAM recommended
* Ollama installed

---

## 1. Quick Start

### Install Dependencies

```bash
cd mcp-platform
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
.venv\Scripts\activate          # Windows PowerShell
pip install -r requirements.txt
```

### LLM Backend

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve
ollama pull qwen2.5:14b-instruct-q4_K_M
```

### Start the Client

```bash
python client.py
```

Access web UI at: `http://localhost:9000/client/ui/index.html`

---

## 2. Using MCP Servers with Other Clients

Add to your MCP client config (e.g., `claude_desktop_config.json`):

```json
{
    "mcpServers": {
        "code_assistant": {
            "command": "/path/to/mcp-platform/.venv/bin/python",
            "args": ["/path/to/mcp-platform/servers/code_assistant/server.py"]
        }
    }
}
```

**Windows paths:** `"command": "C:\\path\\to\\mcp-platform\\.venv\\Scripts\\python.exe"`

**Available servers:**
- `code_assistant` - AI-powered code analysis, generation, and refactoring (12 tools)
- `code_review` - Code review, search, and bug fixing (3 tools)
- `code_runner` - Python/bash execution sandbox (4 tools)
- `github` - GitHub repo clone, browse, and cleanup ⚠️ *Requires `GITHUB_TOKEN` for private repos*
- `google` - Gmail + Google Calendar (9 tools) ⚠️ *Requires one-time OAuth setup*
- `image` - Image search, analysis, and AI generation (6 tools) ⚠️ *Requires `SERPER_API_KEY` for search; generation is free*
- `location` - Weather, time, location (3 tools)
- `plex` - Media library + ML recommendations (18 tools) ⚠️ *Requires `PLEX_URL`, `PLEX_TOKEN`*
- `rag` - Vector search and management (7 tools) ⚠️ *Requires Ollama + `bge-large`*
- `system` - System info and processes (3 tools)
- `text` - Text processing and web search (8 tools)
- `trilium` - Trilium notes integration (11 tools) ⚠️ *Requires `TRILIUM_URL`, `TRILIUM_TOKEN`*

---

## 3. Client Configuration

### Environment Variables

Create `.env` in project root:

```bash
# === LLM Backend ===
OLLAMA_BASE_URL=http://127.0.0.1:11434  # Use 127.0.0.1 for local; LAN IP requires OLLAMA_HOST=0.0.0.0 on the server
OLLAMA_VISION_MODEL=qwen3-vl:8b-instruct
MAX_MESSAGE_HISTORY=30
LLM_TEMPERATURE=0.3

# === GGUF Configuration ===
GGUF_GPU_LAYERS=-1
GGUF_CONTEXT_SIZE=4096
GGUF_BATCH_SIZE=512

# === API Keys ===
PLEX_URL=http://localhost:32400
PLEX_TOKEN=your_token_here
TRILIUM_URL=http://localhost:8888
TRILIUM_TOKEN=your_token_here
SHASHIN_BASE_URL=http://localhost:6624/
SHASHIN_API_KEY=your_key_here
SERPER_API_KEY=your_key_here
OLLAMA_TOKEN=your_token_here

# === A2A Protocol ===
A2A_ENDPOINTS=http://localhost:8010
A2A_EXPOSED_TOOLS=

# === Performance Tuning ===
CONCURRENT_LIMIT=3
EMBEDDING_BATCH_SIZE=50
DB_FLUSH_BATCH_SIZE=50

# === Tool Control ===
DISABLED_TOOLS=plex:*

# === Location ===
DEFAULT_CITY=Vancouver
DEFAULT_STATE=BC
DEFAULT_COUNTRY=Canada
```

### Recommended Setup

```bash
ollama pull qwen2.5:14b-instruct-q4_K_M   # recommended LLM
ollama pull bge-large                       # required for RAG
ollama pull qwen3-vl:8b-instruct           # required for image tools
```

Minimal `.env` to get started:

```env
OLLAMA_VISION_MODEL=qwen3-vl:8b-instruct
DISABLED_TOOLS=plex:*,image_tools:shashin_analyze,shashin_random,shashin_search
OLLAMA_TOKEN=<token>
SERPER_API_KEY=<key>
```

### Feature Requirements

| Feature | Required env vars | Additional setup |
|---------|-------------------|------------------|
| Gmail + Google Calendar | — | One-time Google OAuth setup |
| RAG ingestion & search | — | Ollama running + `bge-large` pulled |
| Plex media library | `PLEX_URL`, `PLEX_TOKEN` | Plex Media Server running |
| Ollama web search | `OLLAMA_TOKEN` | Ollama account + API key |
| Image search | `SERPER_API_KEY` | Serper account + API key |
| AI image generation | — | Free via Pollinations.ai — no key required |
| Trilium notes | `TRILIUM_URL`, `TRILIUM_TOKEN` | Trilium server running |
| Shashin photo gallery | `SHASHIN_BASE_URL`, `SHASHIN_API_KEY` | Shashin server running |
| A2A distributed mode | `A2A_ENDPOINTS` | Remote A2A server running |

### Available Commands

```
:commands              - List all available commands
:clear sessions        - Clear all chat history
:clear session <id>    - Clear session
:sessions              - List all sessions
:stop                  - Stop current operation
:stats                 - Show performance metrics
:tools                 - List available tools
:tools --all           - Show all tools including disabled
:tool <n>              - Get tool description
:model                 - List all available models
:model <n>             - Switch to a model
:gguf add <path>       - Register a GGUF model
:gguf remove <alias>   - Remove a GGUF model
:gguf list             - List registered GGUF models
:a2a on/off/status     - Control A2A mode
:health                - Health overview of all servers
:env                   - Show environment configuration
```

### Google Setup

One-time setup. After completing these steps the server runs headlessly.

1. Go to https://console.cloud.google.com/ and create a project
2. Enable **Gmail API** and **Google Calendar API**
3. Create an OAuth **Desktop app** client and download `credentials.json`
4. Place at `servers/google/credentials.json`
5. Publish app to **In Production** (prevents token expiry every 7 days)
6. Run: `.venv/bin/python auth_google.py`
7. Restart: `python client.py`

> ⚠️ If token becomes invalid: delete `servers/google/token.json` and re-run step 6.

---

## 4. Adding Custom Tools

### Step 1: Create the server file

```bash
mkdir servers/my_tool && touch servers/my_tool/server.py
```

### Step 2: Implement with `@tool_meta`

```python
from mcp.server.fastmcp import FastMCP
from tools.tool_control import check_tool_enabled
from client.tool_meta import tool_meta

mcp = FastMCP("my-tool-server")

@mcp.tool()
@check_tool_enabled(category="my_tool")
@tool_meta(
    tags=["read", "search"],
    triggers=["my keyword", "my phrase"],
    example='use my_function: arg1=""',
)
def my_function(arg1: str) -> str:
    """Short description."""
    return json.dumps({"content": f"Processed {arg1}"})

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

Restart the client and the tool is live — routed, badged, and registered automatically.

### `@tool_meta` field reference

| Field | Required | Description |
|-------|----------|-------------|
| `tags` | ✅ | Capability tags |
| `triggers` | ✅ | Natural language phrases that route to this tool |
| `example` | recommended | Pre-fill text shown in tools panel |
| `text_fields` | if needed | Response fields containing main text |
| `rate_limit` | no | `"100/hour"`, `"10/day"`, `"ollama"`, or `None` |
| `idempotent` | no | `True` if side-effect free (default: `True`) |
| `intent_category` | no | Override routing group name |

### Tag vocabulary

| Tag | Meaning |
|-----|---------|
| `read` | Reads data only |
| `write` | Creates or modifies data |
| `destructive` | Deletes or irreversibly changes data |
| `search` | Primary purpose is search |
| `external` | Calls an external API |
| `vision` | Processes image input |
| `media` | Operates on audio/video/image |
| `calendar` | Interacts with calendar data |
| `email` | Interacts with email |
| `notes` | Interacts with note-taking systems |
| `code` | Operates on source code |
| `system` | Interacts with OS or hardware |
| `rag` | Interacts with the RAG vector store |
| `ai` | Calls an LLM or ML model |

### Adding External MCP Servers

Create `external_servers.json` in the project root:

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

Supported transports: `sse`, `http`, `stdio`. Header auth env var convention:
```bash
ES_SERVERNAME_TOKEN=your_token_here
```

---

## 5. Distributed Mode (A2A Protocol)

```bash
python a2a_server.py    # Terminal 1 — starts on http://localhost:8010
python client.py        # Terminal 2
```

```bash
A2A_ENDPOINTS=http://localhost:8010,http://gpu-server:8020
A2A_EXPOSED_TOOLS=plex,location,text   # empty = expose all
```

---

## 6. Testing

The project has two independent test suites — Python (pytest) and JavaScript (Jest) — with a single script to run both.

### First-time setup

```bash
# Python dependencies (if not already installed)
source .venv/bin/activate
pip install -r tests/requirements.txt

# JavaScript dependencies
npm install

# Make the run script executable (Linux/WSL2)
chmod +x run_tests.sh
```

### Running tests

```bash
./run_tests.sh              # both Python + JS with coverage
./run_tests.sh --py-only    # Python only
./run_tests.sh --js-only    # JS only
./run_tests.sh --no-coverage  # skip coverage (faster)
```

**Python tests directly:**
```bash
python -m pytest                 # all tests with coverage
python -m pytest --no-cov        # skip coverage (faster)
python -m pytest -m unit         # unit tests only
python -m pytest -m integration  # integration tests only
python -m pytest -m e2e          # end-to-end tests only
python -m pytest -x              # stop on first failure
python -m pytest -k "session"    # filter by name
```

**JS tests directly:**
```bash
npm test           # all JS tests with coverage
npm run test:watch # watch mode during development
```

### Test structure

```
tests/
├── conftest.py
├── js/
│   ├── setup.js                  <- browser API mocks + shared helpers
│   ├── test_index.test.js        <- tests for client/ui/js/index.js
│   └── test_dashboard.test.js    <- tests for client/ui/js/dashboard.js
├── unit/
│   ├── test_session_manager.py
│   ├── test_session_manager_extended.py
│   ├── test_models.py
│   ├── test_context_tracker.py
│   ├── test_intent_patterns.py
│   ├── test_query_patterns.py
│   ├── test_query_patterns_extended.py
│   ├── test_commands.py
│   ├── test_commands_extended.py
│   ├── test_metrics.py
│   ├── test_search_client.py
│   ├── test_langgraph.py
│   ├── test_websocket_extended.py
│   ├── test_utils_extended.py
│   ├── test_coverage_boost.py
│   ├── test_coverage_final.py
│   └── test_code_review_tools.py
├── integration/
│   ├── test_websocket_flow.py
│   └── test_langgraph_agent.py
├── e2e/
│   └── test_full_conversation.py
└── results/                      <- generated after each Python test run
    ├── junit.xml
    ├── coverage.xml
    ├── test-report.html
    ├── coverage-report.html
    └── generate_html.py

tests/js-results/                 <- generated after each JS test run
    ├── junit.xml
    ├── test-report.html
    └── coverage/
        └── lcov-report/
            └── index.html        <- interactive line-by-line JS coverage
```

### Coverage thresholds

| Suite | Threshold | Notes |
|-------|-----------|-------|
| Python | 22% | Hard ceiling ~35% due to LLM-dependent graph nodes |
| JavaScript | 30% | Pure functions and DOM logic |

Coverage drops below threshold fail the CI build and open a GitHub Issue automatically.

### CI/CD

Tests run automatically on every push and pull request via GitHub Actions (`.github/workflows/ci.yml`).

- **Test failure** → Issue opened with list of failed tests, labelled `test-failure`
- **Coverage drop** → Issue opened with coverage total, labelled `coverage needs-tests`
- Both Python and JS failures are detected and reported independently

To upload coverage to Codecov, add to the workflow:

```yaml
- name: Upload Python coverage
  uses: codecov/codecov-action@v4
  with:
    files: tests/results/coverage.xml

- name: Upload JS coverage
  uses: codecov/codecov-action@v4
  with:
    files: tests/js-results/coverage/lcov.info
```

---

## 7. Architecture

### Multi-Server Design

```
servers/
├── code_assistant/   12 tools  - AI-powered code analysis, generation, and refactoring
├── code_review/       3 tools  - Code review, search, and bug fixing
├── code_runner/       4 tools  - Python/bash execution sandbox
├── github/            4 tools  - GitHub repo clone, browse, and cleanup
├── google/            9 tools  - Gmail + Google Calendar       [requires OAuth]
├── image/             6 tools  - Image search, analysis, AI generation
├── location/          3 tools  - Weather, time, location
├── plex/             18 tools  - Media + ML recommendations    [requires PLEX_URL + PLEX_TOKEN]
├── rag/               7 tools  - Vector search and management  [requires Ollama + bge-large]
├── system/            3 tools  - System info and processes
├── text/              8 tools  - Text processing and web search
└── trilium/          11 tools  - Trilium notes integration     [requires TRILIUM_URL + TRILIUM_TOKEN]
```

Total: 88 tools across 12 servers

### Directory Structure

```
mcp-platform/
├── servers/
├── a2a_server.py
├── client.py
├── client/
│   ├── ui/
│   │   └── js/
│   │       ├── index.js        <- main chat UI
│   │       └── dashboard.js    <- metrics dashboard
│   ├── capability_registry.py  <- auto-populated from @tool_meta
│   ├── langgraph.py
│   ├── query_patterns.py       <- auto-populated from @tool_meta triggers
│   ├── tool_meta.py            <- single source of truth for tool metadata
│   ├── websocket.py
│   └── ...
├── tests/                      <- all tests + results
├── package.json                <- JS test dependencies
├── run_tests.sh                <- single script to run all tests
└── tools/
```

---

## 8. Intent Patterns & Troubleshooting

### Intent Patterns

Routing is driven by `triggers` in each tool's `@tool_meta` decorator — no manual pattern editing required.

**Force a specific tool:**
```
Using shashin_search_tool, find photos of Noah
Using web_image_search_tool, show me a red panda
```

**Conversational bypass:** Queries starting with `"I like..."`, `"yes"`, `"thanks"`, `"write me a poem"` bypass routing — the LLM answers from context with no tools bound.

---

### Troubleshooting

**Ollama not reachable / connection refused:**
- Use `OLLAMA_BASE_URL=http://127.0.0.1:11434` in `.env` for local access
- For WSL2 + LAN access, start Ollama with `OLLAMA_HOST=0.0.0.0 ollama serve`
- Add Windows port proxies (elevated PowerShell):
  ```powershell
  wsl hostname -I   # get WSL2 IP
  netsh interface portproxy add v4tov4 listenport=11434 listenaddress=0.0.0.0 connectport=11434 connectaddress=<WSL_IP>
  New-NetFirewallRule -DisplayName "WSL2 Ollama" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 11434
  ```
- WSL2 IP changes on reboot — use `Update-WSL2Proxies.ps1` via Task Scheduler to keep proxies current

**Ollama models not appearing:**
- If `OLLAMA_BASE_URL` points to a LAN IP but Ollama is bound to `127.0.0.1` only, the model list will be empty
- Fix: start Ollama with `OLLAMA_HOST=0.0.0.0 ollama serve`
- Verify: `ollama list`

**Web UI not accessible from LAN:**
- HTTP server (port 9000) and WebSockets (8765, 8766) bind to `0.0.0.0` inside WSL2 automatically
- Windows needs port proxies — same pattern as Ollama above, applied to ports 9000, 8765, 8766
- Use `Update-WSL2Proxies.ps1` to keep proxies updated after reboots

**Web UI won't load locally:**
```bash
netstat -an | grep LISTEN   # check ports 8765, 8766, 9000
```

**Google tools not working:**
- Confirm `servers/google/token.json` exists — if not, re-run `auth_google.py`
- Confirm Gmail API and Calendar API are enabled in Google Cloud Console
- Confirm OAuth app is published to **In Production**

**RAG not working:**
- Ensure Ollama is running: `ollama serve`
- Pull bge-large if missing: `ollama pull bge-large`

**Plex tools returning errors:**
```bash
curl $PLEX_URL/identity?X-Plex-Token=$PLEX_TOKEN
```

**GGUF model won't load:**
- Reduce GPU layers: `export GGUF_GPU_LAYERS=20`
- CPU only: `export GGUF_GPU_LAYERS=0`

**A2A server not connecting:**
```bash
curl http://localhost:8010/.well-known/agent-card.json
```

**Conversation history not working:**
- Switch to a larger model: `:model qwen2.5:14b-instruct-q4_K_M`
- Avoid `qwen2.5:3b` / `qwen2.5:7b` for history-aware tasks

**Tools not appearing:**
```bash
:tools --all        # check if disabled
# check DISABLED_TOOLS in .env
python client.py    # restart
```

---

## License

MIT License