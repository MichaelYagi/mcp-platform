# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

Local MCP runtime with a web UI, multi-agent orchestration, distributed tool servers, ML-powered media recommendations, persistent cross-session memory, and a proactive agent scheduler. All built on Ollama (local LLMs), LangGraph, and the MCP protocol. Local-first: all LLM reasoning stays on-device; only tool calls fetch from external sources.

## Running the Project

```bash
# Install (virtualenv at .venv or /home/myagi/.virtualenvs/mcp_a2a)
pip install -r requirements.txt

# Start the client (web UI at http://localhost:9000/client/ui/index.html)
python client.py

# Start A2A distributed server (separate terminal)
python a2a_server.py

# One-time Google OAuth setup
python auth_google.py
```

## Testing

```bash
# All tests (Python + JS)
./run_tests.sh

# Python tests only
python -m pytest                   # all
python -m pytest -m unit           # fast unit tests
python -m pytest -m integration    # integration tests
python -m pytest -m e2e            # end-to-end
python -m pytest --no-cov -k "session"  # filter by name, skip coverage

# JS tests only
npx jest --no-coverage

# Reports generated to tests/results/ and tests/js-results/
```

Pytest config is in `pytest.ini`. Coverage threshold is 22%. Tests require an active virtualenv.

## Linting

```bash
ruff check .      # lint
black .           # format
```

## Architecture

### Entry Points & Runtime

- **`client.py`** (root) — main entry point; discovers and starts all MCP servers, wires up LangGraph agent, starts HTTP + WebSocket servers. Contains `_process_tool_result()` (shared post-processing for all tool results) used by both `_tool_executor` (scheduler path) and `run_agent_wrapper` (direct dispatch path).
- **`client/websocket.py`** — WebSocket handler (ports 8765/8766); per-session task tracking with `SESSION_TASKS` dict; proactive agent singletons live here
- **`client/langgraph.py`** — LangGraph agent creation and execution; query routing; conversation RAG retrieval; the `run_agent()` function is the core inference loop
- **`a2a_server.py`** — A2A distributed mode server (default: port 8010)

### MCP Servers

13 servers in `servers/`, each with a `server.py` entry point:

| Server | Tools | Notes |
|--------|-------|-------|
| `code_assistant` | 12 | AI-powered code analysis, generation, refactoring |
| `code_review` | 3 | Code review, search, bug fixing |
| `code_runner` | 4 | Python/bash execution sandbox |
| `discord` | 2 | Webhook notifications — requires `DISCORD_WEBHOOK_URL` |
| `github` | 4 | Repo clone, browse, cleanup |
| `google` | 13 | Gmail + Google Calendar — requires OAuth (`servers/google/credentials.json`) |
| `image` | 6 | Image search, analysis, AI generation |
| `location` | 3 | Weather, time, location |
| `plex` | 18 | Media library + ML recommendations |
| `rag` | 8 | Vector search and management |
| `system` | 3 | System info and processes |
| `text` | 8 | Text processing and web search |
| `trilium` | 11 | Trilium notes integration |

Total: 95 tools.

### Models

Every model role is independent — swap any of them via `.env` without touching code.

#### Setup — pulling models before first run

Ollama models must be pulled manually. Python-side models (`sentence-transformers`) download automatically on first use and are cached in `~/.cache/huggingface/`.

```bash
# Primary inference (recommended — any Ollama model works, this is a good default)
ollama pull qwen2.5:14b-instruct-q4_K_M

# Routing classifier (small/fast — required if LLM_ROUTING_MODEL is set)
ollama pull qwen2.5:0.5b

# Vision
ollama pull qwen3-vl:8b-instruct

# RAG document embeddings
ollama pull bge-large

# sentence-transformers models (memory embeddings + RAG reranker) download automatically
# on first use — no manual step needed, but requires an internet connection on first run.
```

#### Model roles

| Role | Model | Where configured | Notes |
|------|-------|-----------------|-------|
| **Primary inference** | any Ollama model or local GGUF file (recommended: `qwen2.5:14b-instruct-q4_K_M`) | Ollama (auto-detected at startup) | Main agent reasoning, tool-call generation, and response synthesis. The platform picks whichever Ollama model was used last (`client/last_model.txt`), falling back to the first available model. Any model that supports tool-calling works; 14b q4 is the recommended balance of quality and speed on consumer hardware. Local GGUF files (e.g. downloaded from Hugging Face) are also supported — just download the file and run `:gguf add <path_to_gguf>` in the prompt; no config file editing needed. |
| **Routing classifier** | `qwen2.5:0.5b` | `LLM_ROUTING_MODEL` in `.env` | Pre-flight intent classifier — runs before every query to decide which tools are needed and whether RAG is required. Uses `temperature=0.0` and `num_predict=60`. Falls back to a capped copy of the primary model if unset. A 0.5b model handles this well; latency drops from 2–10 s to under 0.5 s. |
| **Vision** | `qwen3-vl:8b-instruct` | `OLLAMA_VISION_MODEL` in `.env` | Image analysis and vision follow-ups. Invoked by `client/vision.py`. |
| **Memory embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | hard-coded in `client/memory_consolidator.py` | CPU-native, 384-dim, ~22 MB. Downloads automatically on first use. Embeds personal memories in `data/memory.db`. Falls back to Ollama `bge-large` (1024-dim) if `sentence-transformers` is unavailable. Changing models triggers an automatic embedding-compat wipe so old vectors are re-embedded rather than compared across incompatible spaces. |
| **RAG document embeddings** | `bge-large` via Ollama | hard-coded in `tools/rag/` | Embeds ingested documents and conversation turns in the `chunks` table of `data/sessions.db`. 1024-dim. Must be pulled with `ollama pull bge-large`. Used by `rag_add.py`, `rag_search.py`, `rag_vector_db.py`, and `conversation_rag.py`. |
| **RAG reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `tools/rag/rag_search.py:RERANKER_MODEL` | Cross-encoder reranker loaded via `sentence-transformers`. Downloads automatically on first use. Reranks the top-20 cosine candidates from `bge-large` before returning `top_k` results. Loaded eagerly at startup to avoid per-query overhead. Gracefully disabled if unavailable. |

**Key constraint:** `bge-large` (RAG docs) and `all-MiniLM-L6-v2` (memory) are different models with different vector dimensions (1024 vs 384). Never mix their embeddings in the same cosine comparison — each pipeline uses its own table and its own model consistently.

### Tool System

Each capability lives in two places:
- **`servers/<name>/server.py`** — MCP server (entry point for `mcp run` or subprocess); wraps tools with `@mcp.tool()`
- **`tools/<name>/`** — actual business logic imported by the server

Every tool function uses three stacked decorators:
```python
@mcp.tool()
@check_tool_enabled(category="my_tool")   # respects DISABLED_TOOLS env var
@tool_meta(tags=[...], triggers=[...], template='...')  # routing metadata
def my_function(...):
```

**`client/tool_meta.py`** is the single source of truth for tool metadata — adding `@tool_meta` is all that's needed to register a tool for routing, capability tracking, and UI display.

Errors use `MCPToolError` with `FailureKind` enum — canonical definitions in `client/metrics.py`. Each server's `try/except` import block provides local fallback stubs so servers can run standalone. `JsonFormatter` (also `client/metrics.py`) is used for structured log output.

### Tool Result Post-Processing

`_process_tool_result()` in `client.py` (root) is the single shared function for all tool result handling. It covers:
1. Plain text passthrough
2. Image vision routing
3. Pre-built summary passthrough
4. List builder for arrays
5. LLM summarization fallback

Do not add post-processing logic in `_tool_executor` directly — extend `_process_tool_result()` instead.

**Exception — vision results**: `run_agent_wrapper` contains its own inline vision processing path (image fetch → `client/vision.call_vision_model()`) that runs before `_process_tool_result`. This is intentional: vision inference needs access to conversation state (prompt construction, follow-up detection) that `_process_tool_result` does not have. If you add a new post-processing path that requires conversation state, add it in `run_agent_wrapper` before the `_process_tool_result` call. Otherwise extend `_process_tool_result`.

### Routing

**`client/query_patterns.py`** is the single routing authority. `INTENT_CATALOG` maps regex patterns to tool lists. Call `classify(query)` to get a `QueryIntent`. No other module does independent pattern matching.

To add a routing rule: add one entry to `INTENT_CATALOG` in `query_patterns.py`. The rest picks it up automatically.

### Memory & Context

Every LLM call assembles context in this order:
1. System prompt + top-5 highest-importance persistent memories (always injected)
2. Query-relevant persistent memories (vector search)
3. Conversation RAG — turns scrolled out of the window, retrieved by semantic similarity
4. Last `LLM_MESSAGE_WINDOW` turns (direct history)
5. Current user message

- **`client/memory_consolidator.py`** — extracts facts from sessions into `data/memory.db`; fires after 15 min inactivity; handles `:memory` commands
- **`tools/rag/`** — RAG pipeline using SQLite `chunks` table + `sam860/qwen3-reranker:0.6b-Q8_0` reranker
- **`data/sessions.db`** — session + message history (SQLite)
- **`data/memory.db`** — persistent memory store (SQLite, created on first run)
- **`data/scheduler.db`** — scheduled jobs (SQLite, created on first run)

### Proactive Agents / Scheduler

**`client/proactive_agent.py`** — cron, interval, and `once` trigger types. Users describe jobs in natural language; the LLM parses them into job definitions. Tool pipelines use `|` in `llm_prompt`. Requires `apscheduler`.

- `once` trigger type uses `_one_time_signals` deterministic regex (not LLM-based) to detect one-time job intent
- `run_date`/`end_date` columns in `scheduler.db` track once-job lifecycle
- `human_schedule` field overrides display string for jobs where the raw schedule expression isn't human-readable
- Cron expressions must be exactly 5 fields

### Multi-Agent

**`client/multi_agent.py`** — `MultiAgentOrchestrator` with specialized agent roles (orchestrator, researcher, analyst, writer, planner, plex_ingester). Dependency-aware batched `asyncio.gather()` for parallel task execution. RAG tools are excluded from sub-agent executors via `_create_no_rag_executors()`; RAG queries are detected by `_is_rag_query()` and routed to a separate executor in `_execute_single_task()`.

### Prompt Templates

All LLM prompt strings live in **`prompts/prompts.py`** and `prompts/system_prompt.md`. Import from there — never hardcode prompts inline.

### External / A2A Servers

- External MCP servers: configure in `external_servers.json` (transports: `sse`, `http`, `stdio`)
- A2A distributed mode: set `A2A_ENDPOINTS` in `.env`; tools exposed via `A2A_EXPOSED_TOOLS`
- Header auth convention for external servers: `ES_SERVERNAME_TOKEN=...` in `.env`

## Key Configuration

`.env` in project root. Critical variables:

| Variable | Purpose |
|----------|---------|
| `OLLAMA_BASE_URL` | Ollama endpoint — use `127.0.0.1` not `localhost` for WSL2 |
| `OLLAMA_VISION_MODEL` | Model for image analysis (e.g. `qwen3-vl:8b-instruct`) |
| `OLLAMA_NUM_CTX` | Context window / KV cache size (set to 8192) |
| `OLLAMA_KEEP_ALIVE` | Keep model loaded (`-1` = always) |
| `OLLAMA_NUM_PREDICT` | Max tokens per response (4096) |
| `OLLAMA_REPEAT_PENALTY` | Token repetition penalty (1.1; 1.0 = disabled) |
| `LLM_MESSAGE_WINDOW` | Recent turns in direct context (default 6, recommend 15) |
| `LLM_TEMPERATURE` | Inference temperature (0.3) |
| `LLM_ROUTING_ENABLED` | Set to `false` to skip the pre-flight routing LLM call (saves 2-10s/query, trades routing accuracy for latency) |
| `MAX_MESSAGE_HISTORY` | Max messages stored per session (30) |
| `DISABLED_TOOLS` | Comma-separated `category:*` or `category:tool_name` |
| `CONCURRENT_LIMIT` | Max concurrent tool calls (3) |
| `A2A_ENDPOINTS` | Comma-separated A2A server URLs |
| `A2A_EXPOSED_TOOLS` | Tools exposed to A2A peers (empty = all) |
| `LANGSEARCH_API_KEY` | Fallback search when Ollama weekly limit is hit |
| `SERPER_API_KEY` | Image search |
| `DISCORD_WEBHOOK_URL` | Discord notifications |
| `PLEX_URL` / `PLEX_TOKEN` | Plex Media Server |
| `TRILIUM_URL` / `TRILIUM_TOKEN` | Trilium notes |
| `DEFAULT_CITY/STATE/COUNTRY/TIMEZONE` | Location defaults for weather/time tools |

## Hard Rules

- **No emojis or emoticons** anywhere — not in tool outputs, system messages, logs, or UI. Exception: weather condition icons in `servers/weather/` output are permitted
- **Never assume a file has been deployed** — always verify by reading the actual file content before concluding there's a deployment issue
- **Never hardcode LLM prompts inline** — all prompts go in `prompts/prompts.py` or `prompts/system_prompt.md`
- **WSL2 networking**: `utils.py` prefers `192.168.x.x` over `172.x.x.x` for LAN address detection. LAN access requires `netsh portproxy` on the Windows side for ports 9000, 8765, 8766 (web UI + WebSockets). WSL2 IP changes on reboot — use `results/Update-WSL2Proxies.ps1` via Task Scheduler to keep proxies current.
- **`load_dotenv` placement**: must be called at the top of each submodule entry point — env var ordering bugs result from calling it too late