# MCP Platform

Local MCP runtime with multi-agent orchestration, distributed tool servers, and ML-powered media recommendations.

⚠️ **Experimental** — intended for personal and experimental use only, not for production deployment.

---

## Prerequisites

* Python 3.10+
* 16GB+ RAM recommended
* One of:
  * Ollama installed OR
  * GGUF file

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

### Choose LLM Backend

**Option A: Ollama (recommended as a start)**
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama server
ollama serve

# Model required for RAG ingestion
ollama pull bge-large

# Download a model (use 14B+ for best results)
ollama pull qwen2.5:14b-instruct-q4_K_M
```

> ⚠️ **RAG requires Ollama + bge-large:** If Ollama is not running or `bge-large` has not been pulled, RAG ingestion and semantic search will not work. Run `ollama pull bge-large` before attempting to use any RAG features.

**Option B: GGUF (local model files)**
```bash
# Download a GGUF model (example)
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# Register the model
# (After starting client, use :gguf add command)
```

> ⚠️ **RAG with GGUF backend:** RAG still requires Ollama running separately for embeddings (`bge-large`), even if you use a GGUF model for chat. Ollama is not optional for RAG regardless of your LLM backend choice.

### Start the Client
```bash
python client.py
```

Access web UI at: `http://localhost:9000`

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
        "system_tools": {
            "command": "/path/to/mcp_a2a/.venv/bin/python",
            "args": ["/path/to/mcp_a2a/servers/system_tools/server.py"]
        },
        "text_tools": {
            "command": "/path/to/mcp_a2a/.venv/bin/python",
            "args": ["/path/to/mcp_a2a/servers/text_tools/server.py"]
        },
        "todo": {
            "command": "/path/to/mcp_a2a/.venv/bin/python",
            "args": ["/path/to/mcp_a2a/servers/todo/server.py"]
        },
        "knowledge_base": {
            "command": "/path/to/mcp_a2a/.venv/bin/python",
            "args": ["/path/to/mcp_a2a/servers/knowledge_base/server.py"]
        }
    }
}
```

**Windows paths:**
```json
"command": "C:\\path\\to\\mcp_a2a\\.venv\\Scripts\\python.exe"
```

**Available servers:**
- `code_review` - Code analysis (5 tools)
- `location` - Weather, time, location (3 tools)
- `plex` - Media library + ML recommendations (17 tools) ⚠️ *Requires Plex env vars*
- `rag` - Vector search (4 tools) ⚠️ *Requires Ollama + bge-large*
- `system_tools` - System info (4 tools)
- `text_tools` - Text processing (7 tools)
- `todo` - Task management (6 tools)
- `knowledge_base` - Notes management (10 tools)

---

## 3. Client Configuration

### Environment Variables

Create `.env` in project root:
```bash
# === LLM Backend ===
MAX_MESSAGE_HISTORY=30          # Chat history limit (default: 20)
LLM_TEMPERATURE=0.3             # Model temperature 0 to 1 (default: 0.3)

# === GGUF Configuration (if using GGUF backend) ===
GGUF_GPU_LAYERS=-1              # -1 = all GPU, 0 = CPU only, N = N layers on GPU
GGUF_CONTEXT_SIZE=4096          # Context window size
GGUF_BATCH_SIZE=512             # Batch size for processing

# === API Keys (optional services) ===
PLEX_URL=http://localhost:32400  # Plex server URL
PLEX_TOKEN=your_token_here       # Get from Plex account settings
OLLAMA_TOKEN=your_token_here     # Ollama API key (https://ollama.com/settings/keys)

# === A2A Protocol (optional distributed mode) ===
A2A_ENDPOINTS=http://localhost:8010  # Comma-separated endpoints
A2A_EXPOSED_TOOLS=                   # Tool categories to expose (empty = all)

# === Performance Tuning (optional) ===
CONCURRENT_LIMIT=3              # Parallel ingestion jobs (default: 1)
EMBEDDING_BATCH_SIZE=50         # Embeddings per batch (default: 20)
DB_FLUSH_BATCH_SIZE=50          # DB inserts per batch (default: 30)

# === Tool Control (optional) ===
DISABLED_TOOLS=knowledge_base:*,todo:*  # Disable specific tools/categories
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
| RAG ingestion & search | — | Ollama running + `bge-large` pulled |
| Plex media library | `PLEX_URL`, `PLEX_TOKEN` | Plex Media Server running |
| Plex ingestion & recommendations | `PLEX_URL`, `PLEX_TOKEN` | Ollama running + `bge-large` pulled |
| Ollama web search | `OLLAMA_TOKEN` | Ollama account + API key |
| A2A distributed mode | `A2A_ENDPOINTS` | Remote A2A server running |

> ⚠️ **RAG will silently fail** if Ollama is not running or `bge-large` is not available — embeddings cannot be generated without it. Run `ollama list` to confirm the model is present.

> ⚠️ **Plex tools will return errors** if `PLEX_URL` and `PLEX_TOKEN` are not set in `.env`. The Plex server must also be reachable at the configured URL. See the API Setup section below for how to obtain your token.

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
:tool <name>           - Get tool description
:model                 - List all available models
:model <name>          - Switch to a model (auto-detects backend)
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

---

## 4. Adding Tools (Developer Guide)

### Step 1: Create Tool Server
```bash
mkdir servers/my_tool
touch servers/my_tool/server.py
```

### Step 2: Implement Tool
```python
# servers/my_tool/server.py
import asyncio
from mcp.server import Server
from mcp.types import TextContent
from mcp import tool

mcp = Server("my_tool-server")

@mcp.tool()
def my_function(arg1: str, arg2: int) -> str:
    """
    Short description of what this tool does.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2

    Returns:
        Description of return value
    """
    return f"Processed {arg1} with {arg2}"

async def main():
    from mcp.server.stdio import stdio_server
    async with stdio_server() as (read_stream, write_stream):
        await mcp.run(read_stream, write_stream, mcp.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
```

### Step 3: Create Skill Documentation (Optional)
```bash
mkdir -p servers/my_tool/skills
touch servers/my_tool/skills/my_feature.md
```

### Step 4: Update Intent Patterns (Optional)

If your tool needs specific routing, update `client/langgraph.py`:
```python
INTENT_PATTERNS = {
    # ... existing patterns ...
    "my_tool": {
        "pattern": r'\bmy keyword\b|\bmy phrase\b',
        "tools": ["my_function"],
        "priority": 3
    }
}
```

### Step 5: Add External MCP Servers (Optional)

To connect external or third-party MCP servers, create `client/external_servers.json`.
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
A2A_EXPOSED_TOOLS=plex,location,text_tools

# Expose everything (default)
A2A_EXPOSED_TOOLS=
```

**Security:** Exclude `todo`, `knowledge_base`, `rag` to protect personal data.

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
├── knowledge_base/   10 tools  - Notes management
├── location/          3 tools  - Weather, time, location
├── plex/             17 tools  - Media + ML recommendations  [requires PLEX_URL + PLEX_TOKEN]
├── rag/               4 tools  - Vector search               [requires Ollama + bge-large]
├── system_tools/      4 tools  - System info
├── text_tools/        7 tools  - Text processing
└── todo/              6 tools  - Task management

Total: 56 local tools
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
│   ├── langgraph.py
│   ├── search_client.py   ← Ollama web search & fetch
│   ├── websocket.py
│   └── ...
└── tools/
```

---

## 8. Example Prompts & Troubleshooting

### Example Prompts

**Weather:**
```
> What's the weather in Vancouver?
```

**Plex ML Recommendations:**
```
> What should I watch tonight?
> Recommend unwatched SciFi movies
> Show recommender stats
```

**Code Analysis:**
```
> Analyze the code in /path/to/project
> Review this Python file for bugs
```

**Task Management:**
```
> Add "deploy feature" to my todos
> List my todos
```

**Web Search (via Ollama Search):**
```
> Who won the 2025 NBA championship?
> Latest AI developments
```

**RAG (Retrieval-Augmented Generation):**

> ⚠️ **RAG requires Ollama + bge-large.** If either is missing, ingestion and search will fail. See [Choose LLM Backend](#choose-llm-backend) for setup steps.

*Automatic ingestion from web research:*
```
> Write a report about quantum computing using 
  https://en.wikipedia.org/wiki/Quantum_computing and
  https://en.wikipedia.org/wiki/Quantum_algorithm as sources

✅ Fetches both Wikipedia pages
✅ Automatically stores content in RAG (16 chunks, ~2.5s)
✅ Generates report using the content
💾 Content available for future searches
```

*Retrieving stored content:*
```
> Use the rag_search_tool to search for "quantum entanglement"
> What do you have in the RAG about algorithm complexity?
```

*Checking RAG status:*
```
> Use rag_list_sources_tool
> Show RAG stats
```

*GitHub analysis:*
```
> Analyze architecture at https://github.com/user/repo
```

*More Examples:*
```
> Create a document using https://some_link_to_chicken_nachos as a source
> Search RAG for nachos
> Ingest 3 items from Plex
> Use rag_list_sources_tool to see what sources were used
```

**How RAG works:**
- Content is automatically chunked (350 tokens max), embedded using `bge-large`, and stored in SQLite
- URLs are deduplicated — the same page won't be stored twice
- Semantic search finds the most relevant content even if exact keywords don't match
- Search returns top 5 results with similarity scores and source URLs

### Troubleshooting

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

**Tools not appearing:**
```bash
:tools --all        # check if disabled
# check DISABLED_TOOLS in .env
python client.py    # restart
```

---

## License


MIT License