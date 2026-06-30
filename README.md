# MCP Platform [![Tests](https://github.com/MichaelYagi/mcp-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/MichaelYagi/mcp-platform/actions/workflows/ci.yml)

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

---

## 1. Quick Start

### Install Dependencies

```bash
cd mcp-platform
python3 -m venv .venv
source .venv/bin/activate       # Linux/macOS
.venv\Scripts\activate          # Windows PowerShell
pip install -r requirements.txt
```

> **WSL2 note:** creating `.venv` inside `/mnt/c/` fails because NTFS does not support the Unix symlinks that venvs require. Create the venv in the Linux filesystem instead:
> ```bash
> python3 -m venv /home/$USER/.virtualenvs/mcp_platform
> source /home/$USER/.virtualenvs/mcp_platform/bin/activate
> pip install -r requirements.txt
> ```

### LLM Backend

```bash
curl -fsSL https://ollama.com/install.sh | sh
OLLAMA_HOST=0.0.0.0:11434 ollama serve
ollama pull qwen2.5:14b-instruct-q4_K_M   # primary inference (recommended — any tool-calling model works)
ollama pull qwen2.5:0.5b                  # routing classifier (fast pre-flight intent detection)
ollama pull bge-large                     # RAG document embeddings
ollama pull qwen3-vl:8b-instruct          # vision / image analysis
```

> **GGUF alternative:** local GGUF files (e.g. from Hugging Face) work too — download the file and run `:gguf add <path_to_file>` in the prompt. No config editing needed.
>
> **sentence-transformers models** (memory embeddings + RAG reranker) download automatically on first use — no manual pull needed.

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
- `discord` - Discord channel notifications via webhook (2 tools) ⚠️ *Requires `DISCORD_WEBHOOK_URL`*
- `github` - GitHub repo clone, browse, and cleanup (4 tools) ⚠️ *Requires `GITHUB_TOKEN` for private repos*
- `google` - Gmail + Google Calendar (13 tools) ⚠️ *Requires Google setup (Apps Script or OAuth — see [Google Setup](#google-setup))*
- `image` - Image search, analysis, and AI generation (6 tools) ⚠️ *Requires `SERPER_API_KEY` for search; generation is free*
- `location` - Weather, time, location (3 tools) — uses Open-Meteo (free, no key); falls back to OpenWeatherMap if `OPENWEATHER_API_KEY` is set
- `plex` - Media library + ML recommendations (18 tools) ⚠️ *Requires `PLEX_URL`, `PLEX_TOKEN`*
- `rag` - Vector search and management (8 tools) ⚠️ *Requires Ollama + `bge-large`*
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
LLM_MESSAGE_WINDOW=15        # Sliding window of messages sent to LLM; older turns fall into Conversation RAG
LLM_TEMPERATURE=0.3
OLLAMA_NUM_CTX=8192          # KV cache / context window size
OLLAMA_NUM_PREDICT=4096      # Max tokens the LLM will generate per response
OLLAMA_REPEAT_PENALTY=1.1    # Penalise token repetition (1.0 = disabled)
IMAGE_MODEL=flux             # Model used for AI image generation

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
LANGSEARCH_API_KEY=your_key_here   # fallback search when Ollama weekly limit is reached
OPENWEATHER_API_KEY=your_key_here  # fallback weather when Open-Meteo is unavailable
DISCORD_WEBHOOK_URL=your_webhook_url_here

# === Google Apps Script (alternative to OAuth for Gmail/Calendar) ===
# Paste PASTE_INTO_GOOGLE_APPS_SCRIPT.js into script.google.com, deploy as a Web App,
# then set the URL with your SECRET_KEY appended as ?key=...
# When set, ALL Google tools use the script instead of OAuth.
GOOGLE_APPS_SCRIPT_URL=https://script.google.com/macros/s/.../exec?key=<strong-random-secret-32-chars>

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
ollama pull qwen2.5:14b-instruct-q4_K_M   # primary inference — any tool-calling model works; 14b q4 is a good default
ollama pull qwen2.5:0.5b                  # routing classifier — small/fast; set LLM_ROUTING_MODEL=qwen2.5:0.5b in .env
ollama pull bge-large                     # required for RAG
ollama pull qwen3-vl:8b-instruct          # required for image tools
# sentence-transformers models (memory embeddings + reranker) auto-download on first use
```

> **GGUF files:** download any GGUF (e.g. from Hugging Face) and run `:gguf add <path_to_file>` in the prompt — no config editing needed.

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
| Gmail + Google Calendar | — | Google setup — Apps Script (simpler) or OAuth (see [Google Setup](#google-setup)) |
| Discord notifications | `DISCORD_WEBHOOK_URL` | Create webhook in Discord channel settings |
| RAG ingestion & search | — | Ollama running + `bge-large` pulled |
| Plex media library | `PLEX_URL`, `PLEX_TOKEN` | Plex Media Server running |
| Web search (primary) | `OLLAMA_TOKEN` | Ollama account + API key |
| Web search (fallback) | `LANGSEARCH_API_KEY` | LangSearch account — [dashboard](https://langsearch.com/dashboard). Used automatically when Ollama's weekly limit is reached or returns empty. |
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

**Option A — Google Apps Script (simpler setup)**

1. Go to https://script.google.com and create a new project
2. Delete all existing code and paste the contents of `PASTE_INTO_GOOGLE_APPS_SCRIPT.js`
3. Replace `<SECRET_KEY>` in the script with a strong random secret (32+ characters).
   Generate one: `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`
   Anyone who knows this key can read your emails and calendar — treat it like a password.
4. Click **Deploy > New deployment > Web app**
   - Execute as: **Me**
   - Who has access: **Anyone**
   - Click **Deploy** and copy the Web App URL
5. Add to `.env`:
   ```
   GOOGLE_APPS_SCRIPT_URL=<Web App URL>?key=<your SECRET_KEY>
   ```

On subsequent script edits, use **Deploy > Manage deployments** and edit the existing deployment — do not create a new one or the URL will change.

**Option B — Google OAuth**

One-time setup. After completing these steps the server runs headlessly.

1. Go to https://console.cloud.google.com/ and create a project
2. Enable **Gmail API** and **Google Calendar API**
3. Create an OAuth **Desktop app** client and download `credentials.json`
4. Place at `servers/google/credentials.json`
5. Publish app to **In Production** — this is required. Apps left in "Testing" mode have tokens that expire every 7 days, causing `invalid_grant` errors on scheduled jobs
6. Run: `.venv/bin/python auth_google.py`
7. Restart: `python client.py`

**If the token expires later:** the platform detects it on the next Google tool call and shows a re-authorisation banner in the UI with a link. Click the link, approve access, paste the authorisation code into the chat — no server restart needed. Alternatively, delete `servers/google/token.json` and re-run step 6.

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
    template='use my_function: arg1=""',
)
def my_function(arg1: str) -> str:
    """Short description."""
    return json.dumps({"content": f"Processed {arg1}"})

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

Restart the client and the tool is live — routed, badged, and registered automatically.

### `@tool_meta` field reference

| Field             | Required | Description |
|-------------------|----------|-------------|
| `tags`            | ✅ | Capability tags |
| `triggers`        | ✅ | Natural language phrases that route to this tool |
| `template`        | recommended | Pre-fill text shown in tools panel |
| `text_fields`     | if needed | Response fields containing main text |
| `rate_limit`      | no | `"100/hour"`, `"10/day"`, `"ollama"`, or `None` |
| `idempotent`      | no | `True` if side-effect free (default: `True`) |
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
# WSL2: source /home/$USER/.virtualenvs/mcp_platform/bin/activate

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
├── unit/         <- fast isolated unit tests
├── integration/  <- multi-component tests
└── e2e/          <- full conversation tests

tests/results/    <- generated after running tests
├── junit.xml
├── coverage.xml
├── test-report.html
└── coverage-report.html
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
├── discord/           2 tools  - Discord channel notifications [requires DISCORD_WEBHOOK_URL]
├── github/            4 tools  - GitHub repo clone, browse, and cleanup
├── google/           13 tools  - Gmail + Google Calendar       [requires Google setup]
├── image/             6 tools  - Image search, analysis, AI generation
├── location/          3 tools  - Weather, time, location
├── plex/             18 tools  - Media + ML recommendations    [requires PLEX_URL + PLEX_TOKEN]
├── rag/               8 tools  - Vector search and management  [requires Ollama + bge-large]
├── system/            3 tools  - System info and processes
├── text/              8 tools  - Text processing and web search
└── trilium/          11 tools  - Trilium notes integration     [requires TRILIUM_URL + TRILIUM_TOKEN]
```

Total: 95 tools across 13 servers

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

Every LLM call receives context assembled in this exact order:

```
System prompt
  ├─ Persistent memory — top 5 by importance (always injected)
  ├─ Persistent memory — query-relevant (vector search, above threshold)
  └─ Original system instructions + session ID

Conversation RAG — turns that scrolled out of the window (if relevant)

Message window — last LLM_MESSAGE_WINDOW turns (always injected)

Current user message
```

**Lookup chain — what happens on every query:**

1. **Persistent memory (always)** — top 5 highest-importance memories are injected unconditionally into the system prompt, regardless of query relevance. This ensures core facts (your name, family, preferences) are never lost even when no query scores well. Additional query-relevant memories are added on top via vector search.

2. **Message window (always)** — the last `LLM_MESSAGE_WINDOW` turns of the current session are included directly as conversation history.

3. **Conversation RAG (always)** — a semantic search runs against turns that have scrolled out of the window. Results above the reranker threshold are injected between the system prompt and history.

4. **LLM generates response** using all of the above.

**When does each layer save you:**

| Situation | What helps |
|-----------|-----------|
| Asked about something from 3 messages ago | Message window |
| Asked about something from 20 messages ago, same session | Conversation RAG |
| Asked about your name in a new session | Persistent memory (anchor) |
| Asked a specific fact from an old session | Persistent memory (query-relevant) |
| Asked about a specific document or article | Conversation RAG |
| Fact not yet consolidated into memory | Nothing — tell the platform again |

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

---

## 9. Persistent Memory & Proactive Agents

### How memory and context work together

The platform has three layers of context, each serving a different purpose:

| Layer | What it is | Scope | Survives session delete? |
|-------|-----------|-------|--------------------------|
| **Message window** | Last N turns in direct LLM context | Current session | No |
| **Conversation RAG** | Older turns ingested as vectors | Per session | No |
| **Persistent memory** | Distilled facts extracted by LLM | All sessions | Yes — survives session deletion |

**What this means in practice:** If you tell the platform your son's name and ask about it 3 messages later, the message window handles it. If you ask 20 messages later, RAG handles it (usually). If you start a new session tomorrow, only persistent memory has it.

### The memory workflow

**Step 1 — Have a conversation.** Tell the platform things you want it to remember: your name, your family, your projects, your preferences. The more declarative the better ("My wife's name is Suzy" vs "what's my wife's name?").

**Step 2 — Memory extracts automatically.** After 15 minutes of inactivity, the `InactivityWatcher` fires and runs the LLM over your session transcript. It extracts facts and stores them in `data/memory.db` with vector embeddings.

Re-consolidation is smart: it tracks message count at last consolidation and only re-runs if new messages have been added since. Going idle overnight triggers one extraction, not dozens.

**Step 3 — Memories inject on every query.** On each new message, two things happen: the top 5 highest-importance memories are always injected into the system prompt unconditionally (so your name, family, and key preferences are never forgotten), then a vector search finds additional query-relevant memories and adds them on top. The combined block looks like this:

```
## Persistent Memory (from past sessions)
The following facts are KNOWN and TRUE. Use them to answer directly.

◆ The user's name is Bob
○ Bob's wife is Suzy, a nurse and excellent cook
○ Bob's son Sam is 11, plays accordion, excels at hockey
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

For example: if you said "My son Sam plays accordion" and later ask "What instrument does my son play?", the semantic match is strong. But "How about Sam?" is too vague for RAG to confidently return the accordion fact — be specific.

### Two memory tiers

| Tier | How it's created | Persists |
|------|-----------------|---------|
| `episodic` | Auto-extracted from sessions | Permanent — survives session deletion |
| `semantic` | Promoted from episodic (3+ accesses) or added via `:memory add` | Permanent |

Promotion threshold is configurable: `MEMORY_PROMOTE_THRESHOLD=3` in `.env`.

**Deleting a session does not delete its memories.** RAG chunks are purged, but extracted facts remain. To remove memories from a specific session before deleting it:

```
:memory clear session <id>
```

### Reading :memory output

```
PERSISTENT MEMORY
────────────────────────────────────────────────
[11] The user's name is Bob
    ◆ semantic  |  importance: 1.0  |  accessed 163x
[22] Bob's wife Suzy is a nurse and an excellent cook.
    ○ episodic  |  importance: 0.9  |  accessed 89x
```

Each entry shows:

- **`[id]`** — the memory ID, used with `:memory forget <id>` to delete it
- **Content** — the extracted fact, written in plain English
- **`◆ semantic` / `○ episodic`** — tier: semantic (permanent) or episodic (session-derived)
- **`importance`** — score assigned by the LLM at extraction time (0.0–1.0). Higher = retrieved first
- **`accessed Nx`** — how many times this memory was returned in a retrieval search. High access count drives promotion from episodic → semantic

**Memories are sorted by retrieval score** (combination of vector similarity and importance), not by ID or creation date. The most relevant memories for your last query appear first.

**Duplicate memories** can appear when the same session is consolidated multiple times, or when two sessions contain similar facts. They don't cause errors — the LLM sees both — but they waste context space. Clean them up with:

```
:memory dedup        — removes exact duplicates, keeps highest importance copy
:memory forget <id>  — removes near-duplicates manually (same fact, different wording)
```

### Persistent Memory

The platform remembers facts, preferences, and outcomes across sessions. After a session ends (or after 15 minutes of inactivity), the LLM extracts memorable information from the transcript and stores it in `data/memory.db`. On every new session, relevant memories are injected into the system prompt automatically — no re-explaining required.

### Agentic Automation (Proactive Agent Scheduler)

The platform can act autonomously or semi-autonomously — running tools on a schedule, polling for conditions, and involving the LLM to interpret results before surfacing them to you. This is the foundation for things like daily briefings, email monitoring, and eventually fully autonomous workflows.

**Just describe what you want:**
```
show me my daily briefing every day at 6am
check for new emails every 5 minutes and summarize anything urgent
show me a random photo from my gallery every morning with a short commentary
alert me when I have unread emails
run the weather check every morning at 7am and write a friendly summary
```

The LLM classifies whether your message is a scheduling request, parses the intent into a job definition, and presents it for confirmation before saving anything. Ambiguous requests are clarified with a question rather than guessed.

**Three trigger types:**

| Type | How it works | Example |
|------|-------------|---------|
| `cron` | Fires repeatedly on a fixed schedule | "every day at 6am" |
| `condition` | Polls a check tool on an interval; fires only when condition is true | "when I have unread emails" |
| `once` | Fires once at a specific date/time, then removes itself | "at 8am today", "remind me tomorrow at noon" |

**LLM involvement:** Every job can have an `llm_prompt` — an instruction passed to the LLM after the tool runs. Instead of broadcasting the raw tool output, the LLM interprets it first:

- `shashin_random_tool` + *"write a short commentary about this photo including the location and mood"*
- `gmail_get_unread` + *"summarize these emails and flag anything that needs a reply"*
- `get_weather_tool` + *"write a friendly morning weather summary"*

**Tool pipelines:** Chain multiple tools in sequence using `>>` in the `llm_prompt`. Each step's output is automatically passed to the next step — no LLM required:

```
use get_day_briefing >> use discord_notify
use gmail_get_unread >> use discord_notify
use calendar_get_today >> use discord_notify
```

When scheduling, just describe what you want:
```
Every day at 7am, use get_day_briefing then use discord_notify to send me the result
At 8am today, use get_day_briefing then use discord_notify to send me today's schedule
```

The scheduler parser will automatically produce the `>>` chain syntax.

**Condition expressions** have access to rich context from the tool's JSON output:
```
total_unread > 0       — fires when Gmail has unread emails
len_results > 0        — fires when a results list is non-empty
result > 10            — fires when a numeric value exceeds 10
result_len > 100       — fires when the raw output is non-trivial
```

**Job management commands:**
```
:jobs                    — list all scheduled jobs
:jobs pause <label>      — pause a job without deleting it
:jobs enable <label>     — resume a paused job
:jobs cancel <label>     — delete a job permanently
:jobs info <label>       — full job detail including cron and last run
```

**Files:**
- `client/memory_consolidator.py` — memory extraction, storage, injection, and `:memory` commands
- `client/proactive_agent.py` — scheduler, condition triggers, LLM-based intent parser, and `:jobs` commands

**Required dependency:**
```bash
pip install apscheduler
```

**Data files (created automatically on first run):**
```
data/memory.db      — persistent memory store
data/scheduler.db   — scheduled jobs store (label, tool, cron, llm_prompt, etc.)
```

---

## 10. Intent Patterns & Troubleshooting

### Intent Patterns

Routing is driven by the LLM classifier and `triggers` in each tool's `@tool_meta` decorator — no manual pattern editing required. Triggers support plain strings and `r:` prefixed regex patterns.

**Force a specific tool:**
```
Using shashin_search_tool, find photos of Sam
Using web_image_search_tool, show me a red panda
```

**Conversational bypass:** Queries the LLM classifies as conversational (greetings, opinions, creative writing) bypass tool routing entirely — answered from context with no tools bound.

**Scheduling detection:** The LLM determines whether a message is a scheduling/automation request — no keyword lists. If detected, it's routed to the `ScheduleParser` rather than the agent.

---

### Troubleshooting

**Ollama not reachable / connection refused:**
- Use `OLLAMA_BASE_URL=http://127.0.0.1:11434` in `.env` for local access
- For WSL2 + LAN access, start Ollama with `OLLAMA_HOST=0.0.0.0:11434 ollama serve`
- Add Windows port proxies (elevated PowerShell):
  ```powershell
  # Get WSL2 IP (use ip addr — hostname -I is not available on all WSL2 distros)
  $wslIp = (wsl -- ip -4 addr show eth0 | Select-String "inet ") -replace '.*inet (\d+\.\d+\.\d+\.\d+).*','$1'
  netsh interface portproxy add v4tov4 listenport=11434 listenaddress=0.0.0.0 connectport=11434 connectaddress=$wslIp
  New-NetFirewallRule -DisplayName "WSL2 Ollama" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 11434
  ```
- WSL2 IP changes on reboot — re-run the portproxy commands above after each reboot

**Ollama models not appearing:**
- If `OLLAMA_BASE_URL` points to a LAN IP but Ollama is bound to `127.0.0.1` only, the model list will be empty
- Fix: start Ollama with `OLLAMA_HOST=0.0.0.0:11434 ollama serve`
- Verify: `ollama list`

**Web UI not accessible from LAN:**

The platform binds ports 9000 (HTTP), 8765, and 8766 (WebSockets) to `0.0.0.0` inside the process. Whether other devices on your LAN can reach those ports depends on your environment.

*Native Linux:*
- Should work out of the box — check your firewall allows inbound on those ports:
  ```bash
  sudo ufw allow 9000 && sudo ufw allow 8765 && sudo ufw allow 8766
  ```

*WSL2 on Windows:*
WSL2 runs on a private virtual network (`172.x.x.x`) that is not directly routable from your LAN. Windows must forward LAN traffic to WSL2 using `netsh portproxy`. Run this in an elevated PowerShell each time the WSL2 IP changes (it changes on every reboot):

```powershell
# 1. Find your current WSL2 IP (hostname -I is not available on all WSL2 distros)
$wslIp = (wsl -- ip -4 addr show eth0 | Select-String "inet ") -replace '.*inet (\d+\.\d+\.\d+\.\d+).*','$1'
Write-Host "WSL2 IP: $wslIp"

# 2. Add port forwarding for each port (both 0.0.0.0 and specific LAN IP — both are needed)
netsh interface portproxy add v4tov4 listenport=9000  listenaddress=0.0.0.0 connectport=9000  connectaddress=$wslIp
netsh interface portproxy add v4tov4 listenport=8765  listenaddress=0.0.0.0 connectport=8765  connectaddress=$wslIp
netsh interface portproxy add v4tov4 listenport=8766  listenaddress=0.0.0.0 connectport=8766  connectaddress=$wslIp
$lanIp = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -like "192.168.*" } | Select-Object -First 1).IPAddress
netsh interface portproxy add v4tov4 listenport=9000  listenaddress=$lanIp connectport=9000  connectaddress=$wslIp
netsh interface portproxy add v4tov4 listenport=8765  listenaddress=$lanIp connectport=8765  connectaddress=$wslIp
netsh interface portproxy add v4tov4 listenport=8766  listenaddress=$lanIp connectport=8766  connectaddress=$wslIp

# 3. Allow inbound through Windows Firewall (first time only)
New-NetFirewallRule -DisplayName "MCP Platform 9000" -Direction Inbound -Protocol TCP -LocalPort 9000 -Action Allow
New-NetFirewallRule -DisplayName "MCP Platform 8765" -Direction Inbound -Protocol TCP -LocalPort 8765 -Action Allow
New-NetFirewallRule -DisplayName "MCP Platform 8766" -Direction Inbound -Protocol TCP -LocalPort 8766 -Action Allow

# 4. Allow inbound through WSL2 firewall (first time only — run inside WSL2)
# sudo ufw allow 9000 && sudo ufw allow 8765 && sudo ufw allow 8766
```

Access from LAN devices using your **Windows** LAN IP (e.g. `192.168.0.x`), not the WSL2 IP:
```
http://192.168.0.x:9000/client/ui/index.html
```

To find your Windows LAN IP: `ipconfig` → look for the `192.168.x.x` address on your main adapter.

> Tip: if it stops working after a reboot, just re-run the `netsh interface portproxy add` commands — the firewall rules persist but the WSL2 IP will have changed.

**Web UI won't load locally:**
```bash
netstat -an | grep LISTEN   # check ports 8765, 8766, 9000
```

**Web search returning no results / empty responses:**
- Ollama's free tier has a weekly usage cap — when hit, the API returns HTTP 200 with an empty body
- The platform automatically falls back to LangSearch when this happens
- Set `LANGSEARCH_API_KEY` in `.env` to enable the fallback (free account at https://langsearch.com/dashboard)
- To confirm which provider is being used, check logs for `🔍 Ollama web search` vs `🔍 LangSearch web search`


**Google / OAuth issues:**
- Confirm `servers/google/token.json` exists — if not, re-run `auth_google.py`
- Confirm Gmail API and Calendar API are enabled in Google Cloud Console
- Confirm OAuth app is published to **In Production** (Testing mode tokens expire every 7 days)
- `invalid_grant` at runtime: the re-auth banner will appear in the UI automatically — follow the link, approve, paste the code into chat

**RAG not working:**
- Ensure Ollama is running: `OLLAMA_HOST=0.0.0.0:11434 ollama serve`
- Pull bge-large if missing: `ollama pull bge-large`

**Auto-RAG returning irrelevant context:**
- Raise `min_score` threshold in `rag_search_tool` call inside `run_agent`
- Check RAG store isn't polluted: `use rag_browse_tool`
- Remove a source: `use rag_delete_source_tool: source="<source>"`

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
