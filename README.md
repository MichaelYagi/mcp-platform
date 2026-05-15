# MCP Platform

Local MCP runtime with multi-agent orchestration, distributed tool servers, ML-powered media recommendations, persistent cross-session memory, and a proactive agent scheduler.

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
- [8. RAG & Conversation Memory](#8-rag--conversation-memory)
- [9. Persistent Memory & Proactive Agents](#9-persistent-memory--proactive-agents)
- [10. Intent Patterns & Troubleshooting](#10-intent-patterns--troubleshooting)
- [License](#license)

---

## Prerequisites

* Python 3.12+
* 16GB+ RAM recommended
* Ollama installed
* `apscheduler` (`pip install apscheduler`) — required for proactive agent scheduler

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
DEFAULT_TIMEZONE=America/Vancouver
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
:jobs                  - List all scheduled jobs
:jobs pause <label>    - Pause a scheduled job
:jobs enable <label>   - Resume a scheduled job
:jobs cancel <label>   - Delete a scheduled job
:jobs info <label>     - Show full job detail
:memory                - List all persistent memories
:memory                        - List all memories
:memory semantic               - List permanent memories only
:memory episodic               - List session-derived memories
:memory forget <id>            - Delete a memory by ID
:memory clear                  - Clear all episodic memories
:memory clear session <id>     - Delete memories from one session
:memory consolidate <id>       - Extract memories from a session now
:memory add <fact>             - Manually add a permanent memory
:memory dedup                  - Remove duplicate memories
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

All configured endpoints are discovered and registered concurrently at startup via `asyncio.gather()` — connection timeouts for unreachable endpoints no longer block each other.

---

## 6. Testing

### Running Tests

Activate your virtualenv first, then run from the project root:

```bash
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\activate           # Windows PowerShell

python -m pytest                 # all tests
python -m pytest -m unit         # fast unit tests only
python -m pytest -m integration  # integration tests
python -m pytest -m e2e          # end-to-end tests
python -m pytest --no-cov        # skip coverage (faster)
python -m pytest -x              # stop on first failure
python -m pytest -k "session"    # filter by name
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
└── e2e/
    └── test_full_conversation.py

results/                    <- generated after running tests
├── junit.xml
├── coverage.xml
├── test-report.html
├── coverage-report.html
└── generate_html.py
```

### CI/CD

Tests run automatically on every push and pull request via GitHub Actions (`.github/workflows/ci.yml`). On failure, a GitHub Issue is opened automatically with a link to the failed run.

To upload coverage to Codecov, add to the workflow:

```yaml
- name: Upload coverage
  uses: codecov/codecov-action@v3
  with:
    files: results/coverage.xml
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
├── rag/               8 tools  - Vector search and management  [requires Ollama + bge-large]
├── system/            3 tools  - System info and processes
├── text/              8 tools  - Text processing and web search
└── trilium/          11 tools  - Trilium notes integration     [requires TRILIUM_URL + TRILIUM_TOKEN]
```

Total: 89 tools across 12 servers

### Concurrency & Parallelism

The platform uses `asyncio.gather()` at several layers to run non-LLM work concurrently:

| Layer | What runs in parallel |
|-------|-----------------------|
| Startup — server discovery | TCP reachability checks and OAuth probes for all external servers |
| Startup — A2A registration | All `A2A_ENDPOINTS` are discovered and registered simultaneously |
| Multi-agent task execution | Independent tasks (no unmet dependencies) run as a concurrent batch each cycle |
| A2A subtask execution | Same dependency-aware batched gather as multi-agent mode |
| WebSocket broadcast | All connected clients receive messages via a single `gather()` call |

**Hardware note:** LLM inference serialises at the Ollama GPU layer regardless of concurrency — one inference runs at a time on a single GPU. The parallelism benefit is in I/O-bound work: HTTP tool calls, database queries, and network requests. True parallel LLM execution would require multiple GPUs or cloud-hosted sub-agents.

### Directory Structure

```
mcp-platform/
├── servers/
├── a2a_server.py
├── client.py
├── client/
│   ├── ui/
│   ├── capability_registry.py  <- auto-populated from @tool_meta
│   ├── langgraph.py
│   ├── memory_consolidator.py  <- persistent cross-session memory
│   ├── proactive_agent.py      <- scheduler + condition triggers
│   ├── query_patterns.py       <- auto-populated from @tool_meta triggers
│   ├── tool_meta.py            <- single source of truth for tool metadata
│   ├── websocket.py
│   └── ...
├── data/
│   ├── sessions.db             <- session + message history
│   ├── memory.db               <- persistent memory (created on first run)
│   └── scheduler.db            <- scheduled jobs (created on first run)
└── tools/
```

---

## 8. RAG & Conversation Memory

### How context works

Every LLM call receives context in this order:

```
[System prompt + current session ID]
[Auto-RAG: semantically relevant chunks from memory]
[Last LLM_MESSAGE_WINDOW conversation turns]
[Current user message]
```

### Conversation window (`LLM_MESSAGE_WINDOW`)

Controls how many recent turns the LLM sees directly. Set in `.env`:

```bash
LLM_MESSAGE_WINDOW=15   # default: 6, recommended: 15
```

A window of 6 is too tight for normal conversation — information shared early in a session scrolls out before you can ask about it. 15 covers a full back-and-forth without hitting token limits on qwen2.5:14b. If you share something and the LLM seems to forget it a few messages later, increase this value.

### Overflow ingestion

When history exceeds the window, older turns are automatically ingested into the RAG vector database as `Human + Assistant` pairs. They remain searchable via semantic similarity even after scrolling out of the window.

### Auto-RAG retrieval

On every message, a semantic search runs against the full RAG store using the current user message as the query. Matching chunks (from old conversation turns, ingested documents, Plex subtitles, or research) are injected into context automatically — no explicit `search rag` trigger needed.

### What each retrieval method is good for

| Query type | Best tool |
|------------|-----------|
| "What did we discuss about X?" | Auto-RAG (semantic) |
| "What was my first prompt?" | `session_history_tool` (ordered DB) |
| "Summarise this session" | `session_history_tool` |
| "What was in that article we read?" | Auto-RAG |
| "What did I ask 10 messages ago?" | `session_history_tool` |

### `session_history_tool`

Added to the `rag` server. Queries the session SQLite database directly for ordered, timestamped message history. The current session ID is always injected into the system prompt so the LLM can pass it through automatically.

```
use session_history_tool: session_id="<id>" [limit="20"] [order="asc"]
```

Triggers: `first prompt`, `what did I ask`, `earlier in this session`, `summarise this session`, `session history`

### Applying the `langgraph.py` patch

Three changes are required:

**1. `langgraph.py` — function signature** (line ~2785):
```python
# Before
async def run_agent(..., capability_registry=None):

# After
async def run_agent(..., capability_registry=None, session_id=None):
```

**2. `langgraph.py` — replace STEP 1 through STEP 3 setup**

Replace the block from `# STEP 1: Save the original SystemMessage` through `tool_registry = {tool.name: tool for tool in tools}` with the contents of `langgraph_patch.py`.

**3. `websocket.py` — pass session_id at the call site** (line ~250):
```python
# Before
result = await run_agent_fn(
    agent, conversation_state, prompt, logger, tools, system_prompt
)

# After
result = await run_agent_fn(
    agent, conversation_state, prompt, logger, tools, system_prompt,
    session_id=session_id
)
```

**4. `servers/rag/server.py` — add `session_history_tool`**

Append the contents of `session_history_tool.py` to `server.py` after the existing `rag_search_tool` function.

---

## 9. Persistent Memory & Proactive Agents

### How memory and context work together

The platform has three layers of context, each serving a different purpose:

| Layer | What it is | Scope | Survives session delete? |
|-------|-----------|-------|--------------------------|
| **Message window** | Last N turns in direct LLM context | Current session | No |
| **Conversation RAG** | Older turns ingested as vectors | Per session | No |
| **Persistent memory** | Distilled facts extracted by LLM | All sessions | Yes |

**What this means in practice:** If you tell the platform your son's name and ask about it 3 messages later, the message window handles it. If you ask 20 messages later, RAG handles it (usually). If you start a new session tomorrow, only persistent memory has it.

### The memory workflow

**Step 1 — Have a conversation.** Tell the platform things you want it to remember: your name, your family, your projects, your preferences. The more declarative the better ("My wife's name is Ryuko" vs "what's my wife's name?").

**Step 2 — Memory extracts automatically.** After 15 minutes of inactivity, the `InactivityWatcher` fires and runs the LLM over your session transcript. It extracts facts and stores them in `data/memory.db` with vector embeddings.

Re-consolidation is smart: it tracks message count at last consolidation and only re-runs if new messages have been added since. Going idle overnight triggers one extraction, not dozens.

**Step 3 — Memories inject on every query.** On each new message, a vector search finds the most relevant memories and prepends them to the system prompt:

```
## Persistent Memory (from past sessions)
The following facts are KNOWN and TRUE. Use them to answer directly.

◆ The user's name is Mike
○ Mike's wife is Ryuko, a dental hygienist who plays piano
○ Mike's son Noah is 11, plays cello, excels at swimming
...
```

**Step 4 — Memories accumulate over time.** Episodic memories accessed 3+ times are promoted to semantic (permanent) tier nightly. The platform gets more useful the longer you use it.

### When memory doesn't fire automatically

The inactivity watcher fires 15 minutes after your last message. If you need memories extracted immediately:

```
:memory consolidate <session_id>
```

Use `:sessions` to find the session ID. The command clears the consolidation flag and re-runs extraction regardless of message count.

### Memory commands

```
:memory                        — list all memories (sorted by relevance)
:memory semantic               — permanent memories only
:memory episodic               — session-derived memories
:memory forget <id>            — delete one memory by ID
:memory clear                  — delete all episodic memories
:memory clear session <id>     — delete memories from one session
:memory consolidate <id>       — extract memories from a session now
:memory add <fact>             — manually add a permanent memory
:memory dedup                  — remove duplicate memories
```

Manually added memories (`:memory add`) are stored as `semantic` tier with importance 1.0 — they always rank first in retrieval.

### If the LLM forgets something mid-session

Increase `LLM_MESSAGE_WINDOW` in `.env`. The default of 6 is too tight — 15 is recommended. Information shared early in a session scrolls out of the window before you can ask about it.

Once a turn scrolls out of the window it moves into **Conversation RAG** — it's still there, but now retrieved by semantic similarity rather than direct context. This means the query phrasing needs to be close enough to the original content for the reranker to surface it. If the LLM still can't find something that was said earlier in the same session, try rephrasing the question to use the same keywords as the original statement.

For example: if you said "My son Noah plays cello" and later ask "What instrument does my son play?", the semantic match is strong. But "How about Noah?" is too vague for RAG to confidently return the cello fact — be specific.

### Two memory tiers

| Tier | How it's created | Persists |
|------|-----------------|---------|
| `episodic` | Auto-extracted from sessions | Until manually deleted or session cleared |
| `semantic` | Promoted from episodic (3+ accesses) or added via `:memory add` | Permanent |

Promotion threshold is configurable: `MEMORY_PROMOTE_THRESHOLD=3` in `.env`.

### Persistent Memory

The platform remembers facts, preferences, and outcomes across sessions. After a session ends (or after 15 minutes of inactivity), the LLM extracts memorable information from the transcript and stores it in `data/memory.db`. On every new session, relevant memories are injected into the system prompt automatically — no re-explaining required.

### Proactive Agent Scheduler

Agents can run on a schedule or when a condition becomes true — without any prompt from you.

**Scheduling via natural language:**
```
do a day briefing every day at 5:30am
run a Gmail summary every weekday at 8am
alert me when Gmail unread count is over 10
check the weather every morning at 7am
```

The platform parses the request, shows you exactly what it understood (tool, schedule, cron expression), and waits for confirmation before writing anything. Ambiguous requests are clarified with a question rather than guessed.

**Two trigger types:**

| Type | How it works |
|------|-------------|
| `cron` | Fires at a fixed time — "every day at 5:30am" |
| `condition` | Polls a check tool on an interval; fires the action tool only when the condition is true |

**Job management commands:**
```
:jobs                    — list all scheduled jobs
:jobs pause <label>      — pause a job
:jobs enable <label>     — resume a paused job
:jobs cancel <label>     — delete a job
:jobs info <label>       — full job detail
```

**New files (drop into `client/`):**
- `client/memory_consolidator.py` — memory extraction, storage, injection, and `:memory` commands
- `client/proactive_agent.py` — scheduler, condition triggers, schedule parser, and `:jobs` commands

**Required dependency:**
```bash
pip install apscheduler
```

**New data files (created automatically on first run):**
```
data/memory.db      — persistent memory store
data/scheduler.db   — scheduled jobs store
```

---

## 10. Intent Patterns & Troubleshooting

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
- WSL2 IP changes on reboot — use `results/Update-WSL2Proxies.ps1` via Task Scheduler to keep proxies current

**Ollama models not appearing:**
- If `OLLAMA_BASE_URL` points to a LAN IP but Ollama is bound to `127.0.0.1` only, the model list will be empty
- Fix: start Ollama with `OLLAMA_HOST=0.0.0.0 ollama serve`
- Verify: `ollama list`

**Web UI not accessible from LAN:**
- HTTP server (port 9000) and WebSockets (8765, 8766) bind to `0.0.0.0` inside WSL2 automatically
- Windows needs port proxies — same pattern as Ollama above, applied to ports 9000, 8765, 8766
- Use `results/Update-WSL2Proxies.ps1` to keep proxies updated after reboots

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

**Auto-RAG returning irrelevant context:**
- Raise `min_score` threshold in `rag_search_tool` call inside `run_agent`
- Check RAG store isn't polluted: `use rag_browse_tool`
- Clear and re-ingest if needed: `use rag_clear_tool`

**session_history_tool not finding messages:**
- Confirm the session ID in the system prompt matches the active session
- Check `data/sessions.db` exists and is writable

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