# MCP Multi-Server Architecture

A Model Context Protocol (MCP) implementation with distributed multi-server architecture, Agent-to-Agent (A2A) protocol support, and ML-powered recommendations.

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
```bash
# Clone repository
git clone <repo-url>
cd mcp_a2a

# Create virtual environment
python -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate

# Activate (Windows PowerShell)
.venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### Choose LLM Backend

**Option A: Ollama (recommended as a start)**
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama server
ollama serve

# Download a model
ollama pull llama3.1:8b
```

**Option B: GGUF (local model files)**
```bash
# Download a GGUF model (example)
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

# Register the model
# (After starting client, use :gguf add command)
```

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
- `plex` - Media library + ML recommendations (17 tools)
- `rag` - Vector search (4 tools)
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
LLM_BACKEND=ollama              # "ollama" or "gguf"
MAX_MESSAGE_HISTORY=30          # Chat history limit (default: 20)

# === GGUF Configuration (if using GGUF backend) ===
GGUF_GPU_LAYERS=-1              # -1 = all GPU, 0 = CPU only, N = N layers on GPU
GGUF_CONTEXT_SIZE=4096          # Context window size
GGUF_BATCH_SIZE=512             # Batch size for processing

# === API Keys (optional services) ===
PLEX_URL=http://localhost:32400  # Plex server URL
PLEX_TOKEN=your_token_here       # Get from Plex account settings
WEATHER_TOKEN=your_token_here    # OpenWeatherMap API key
LANGSEARCH_TOKEN=your_token_here # LangSearch API key (https://langsearch.com)

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

### Available Commands

These work in both CLI and web UI:
```
:commands              - List all available commands
:stop                  - Stop current operation
:stats                 - Show performance metrics
:tools                 - List available tools (hides disabled)
:tools --all           - Show all tools including disabled
:tool <name>           - Get tool description
:model                 - List all available models
:model <name>          - Switch to a model (auto-detects backend)
:models                - List models (legacy)
:sync                  - Sync to model in last_model.txt
:gguf add <path>       - Register a GGUF model
:gguf remove <alias>   - Remove a GGUF model
:gguf list             - List registered GGUF models
:a2a on                - Enable agent-to-agent mode
:a2a off               - Disable agent-to-agent mode
:a2a status            - Check A2A system status
:env                   - Show environment configuration
```

### API Setup

**Weather (OpenWeatherMap):**
1. Sign up at https://openweathermap.org/api
2. Get API key from account settings
3. Add to `.env`: `WEATHER_TOKEN=your_key`

**LangSearch (web search):**
1. Sign up at https://langsearch.com
2. Get API key from dashboard
3. Add to `.env`: `LANGSEARCH_TOKEN=your_key`

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

---

## 4. Adding Tools (Developer Guide)

### Step 1: Create Tool Server
```bash
# Create server directory
mkdir servers/my_tool

# Create server file
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
    
    Detailed explanation of behavior, use cases, etc.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
    
    Returns:
        Description of return value
    """
    result = f"Processed {arg1} with {arg2}"
    return result

async def main():
    from mcp.server.stdio import stdio_server
    async with stdio_server() as (read_stream, write_stream):
        await mcp.run(
            read_stream,
            write_stream,
            mcp.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
```

### Step 3: Create Skill Documentation (Optional)
```bash
# Create skills directory
mkdir -p servers/my_tool/skills

# Create skill file
touch servers/my_tool/skills/my_feature.md
```
```markdown
# My Feature Skill

This skill enables X functionality.

## When to Use
- Use case 1
- Use case 2

## Examples
User: "Do something"
Assistant: [calls my_function with appropriate args]
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

### Step 5: Test & Deploy
```bash
# Restart client (auto-discovers new server)
python client.py

# Test in CLI or web UI
> test my new tool
```

---

## 5. Distributed Mode (A2A Protocol)

Run tools on remote servers and expose them via HTTP.

### Setup A2A Server

**Terminal 1 - Start A2A server:**
```bash
python a2a_server.py
```

Server starts on `http://localhost:8010`

**Terminal 2 - Start client:**
```bash
python client.py
```

Client auto-connects to A2A endpoints in `.env`

### Control Exposed Tools

Use `A2A_EXPOSED_TOOLS` to control which categories are publicly accessible:
```bash
# Expose specific categories (comma-separated)
A2A_EXPOSED_TOOLS=plex,location,text_tools

# Expose everything (default)
A2A_EXPOSED_TOOLS=

# Available categories:
# plex, location, text_tools, system_tools, code_review,
# rag, todo, knowledge_base
```

**Security:**
- Empty = all 8 servers exposed (56 tools)
- Specified = only listed categories exposed
- Exclude `todo`, `knowledge_base`, `rag` to protect personal data

### Multi-Endpoint Support

Connect to multiple A2A servers:
```bash
# In .env
A2A_ENDPOINTS=http://localhost:8010,http://gpu-server:8020
```

Client aggregates tools from all successful connections.

### Check Available Tools

**Via HTTP:**
```bash
curl http://localhost:8010/tool-categories
```

**Via Client:**
```
> :a2a status
```

---

## 6. Architecture

### Multi-Server Design

8 specialized MCP servers communicate via stdio:
```
servers/
├── code_review/       5 tools  - Code analysis
├── knowledge_base/   10 tools  - Notes management
├── location/          3 tools  - Weather, time, location
├── plex/             17 tools  - Media + ML recommendations
├── rag/               4 tools  - Vector search
├── system_tools/      4 tools  - System info
├── text_tools/        7 tools  - Text processing
└── todo/              6 tools  - Task management

Total: 56 local tools
```

### Directory Structure
```
mcp_a2a/
├── servers/           # MCP servers (stdio)
│   ├── plex/
│   │   ├── server.py
│   │   ├── ml_recommender.py
│   │   └── skills/
│   └── ...
├── a2a_server.py     # A2A HTTP server
├── client.py         # AI agent client
├── client/
│   ├── ui/
│   │   ├── index.html      # Web UI
│   │   └── dashboard.html  # Dashboard UI
│   ├── langgraph.py  # Agent execution
│   ├── websocket.py  # WebSocket server
│   └── ...
└── tools/            # Tool implementations
```

---

## 7. Example Prompts & Troubleshooting

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

**Web Search (via LangSearch):**
```
> Who won the 2024 NBA championship?
> Latest AI developments
```

### Troubleshooting

**Ollama models not appearing:**
```bash
# Make sure Ollama is running
ollama serve

# Check models are downloaded
ollama list

# Restart client
python client.py
```

**GGUF model won't load:**
- Check model size vs VRAM (use models <7GB for 12GB VRAM)
- Reduce GPU layers: `export GGUF_GPU_LAYERS=20`
- Increase timeout: `export GGUF_LOAD_TIMEOUT=300`
- Use CPU only: `export GGUF_GPU_LAYERS=0`

**Web UI won't load:**
```bash
# Check ports are available: 8765, 8766, 9000
netstat -an | grep LISTEN

# Try localhost directly
http://localhost:9000
```

**A2A server not connecting:**
```bash
# Verify server is running
curl http://localhost:8010/.well-known/agent-card.json

# Check A2A_ENDPOINTS in .env
```

**LangSearch not working:**
- Verify `LANGSEARCH_TOKEN` in `.env`
- Check API key at https://langsearch.com
- System falls back to LLM if unavailable

**Tools not appearing:**
```bash
# Check tool is enabled
:tools --all

# Check DISABLED_TOOLS in .env

# Restart client
python client.py
```

---

## License


MIT License
