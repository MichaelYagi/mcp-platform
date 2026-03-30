"""
A2A Server Module (Lifespan Version) - A2A_EXPOSED_TOOLS Support
--------------------------------------------------------------------
Exposes your MCP tools over the A2A protocol using FastAPI's modern lifespan API.
Uses A2A_EXPOSED_TOOLS env var to control which tool categories are exposed.
"""

import os
from urllib.parse import urljoin
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi import Request
from pydantic import BaseModel
from contextlib import asynccontextmanager
from mcp_use.client.client import MCPClient
from client import utils

PROJECT_ROOT = Path(__file__).parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

# Support both single and multiple A2A endpoints
A2A_ENDPOINT = os.getenv("A2A_ENDPOINT", "").strip()
A2A_ENDPOINTS_STR = os.getenv("A2A_ENDPOINTS", "").strip()

# Parse multiple endpoints (comma-separated list)
A2A_ENDPOINTS = []
if A2A_ENDPOINTS_STR:
    A2A_ENDPOINTS = [ep.strip() for ep in A2A_ENDPOINTS_STR.split(",") if ep.strip()]
elif A2A_ENDPOINT:
    # Backward compatibility: single endpoint
    A2A_ENDPOINTS = [A2A_ENDPOINT]

# -----------------------------
# A2A RPC Request Schema
# -----------------------------
class RPCRequest(BaseModel):
    jsonrpc: str
    id: str
    method: str
    params: dict


# -----------------------------
# Auto-Discovery with A2A_EXPOSED_TOOLS Support
# -----------------------------
def auto_discover_servers_all(servers_dir: Path):
    """Discover ALL servers (no filtering) - fallback function"""
    mcp_servers = {}

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

    return mcp_servers


def auto_discover_servers(servers_dir: Path):
    """
    Auto-discover servers based on A2A_EXPOSED_TOOLS env var

    Logic:
    1. Check for A2A_EXPOSED_TOOLS in .env
    2. If found and has valid tools → use whitelist
    3. If not found OR empty OR no valid tools → use ALL tools
    """

    # Get environment variable
    exposed_tools_str = os.getenv("A2A_EXPOSED_TOOLS", "").strip()

    if exposed_tools_str:
        # Parse the whitelist
        whitelist = [s.strip() for s in exposed_tools_str.split(",") if s.strip()]
        print(f"📋 A2A_EXPOSED_TOOLS whitelist: {whitelist}")
    else:
        # No variable or empty → use all tools
        whitelist = None
        print(f"📋 A2A_EXPOSED_TOOLS not set → exposing ALL tools")

    mcp_servers = {}
    skipped = []
    included = []

    # Discover all tool categories
    for server_dir in servers_dir.iterdir():
        if server_dir.is_dir():
            tool_category = server_dir.name
            server_file = server_dir / "server.py"

            if not server_file.exists():
                continue

            # Apply whitelist if specified
            if whitelist is not None:
                if tool_category not in whitelist:
                    skipped.append(tool_category)
                    print(f"⏭️  Skipping {tool_category} (not in A2A_EXPOSED_TOOLS)")
                    continue
                else:
                    included.append(tool_category)
                    print(f"✅ Including {tool_category} (in A2A_EXPOSED_TOOLS)")
            else:
                included.append(tool_category)
                print(f"✅ Including {tool_category} (no whitelist)")

            mcp_servers[tool_category] = {
                "command": utils.get_venv_python(PROJECT_ROOT),
                "args": [str(server_file)],
                "cwd": str(PROJECT_ROOT),
                "env": {"CLIENT_IP": utils.get_public_ip()}
            }

    # If whitelist was specified but NO valid tools found → warn and use all
    if whitelist is not None and len(mcp_servers) == 0:
        print(f"⚠️  WARNING: A2A_EXPOSED_TOOLS specified but no valid tool categories found!")
        print(f"⚠️  Specified: {whitelist}")
        print(f"⚠️  None of these match available tool categories")
        print(f"⚠️  Falling back to ALL tools")

        # Retry without whitelist
        mcp_servers = auto_discover_servers_all(servers_dir)

        # Log what's actually available
        available = list(mcp_servers.keys())
        print(f"ℹ️  Available tool categories: {available}")

    return mcp_servers


# -----------------------------
# FastAPI Lifespan Manager
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting MCP session…")

    mcp_servers = auto_discover_servers(PROJECT_ROOT / "servers")

    if not mcp_servers:
        print("❌ No tool categories found! Check your servers/ directory")
        return

    client = MCPClient.from_dict({
        "mcpServers": mcp_servers
    })

    # Create sessions for all servers (not just "local")
    await client.create_all_sessions()

    # Get the first session (or create a combined session)
    # We'll use the client directly instead of storing a single session
    app.state.client = client

    print(f"✅ MCP session ready with {len(mcp_servers)} tool categories")
    yield

    # Cleanup on shutdown
    print("🛑 Shutting down MCP sessions…")
    await client.close_all_sessions()


# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(lifespan=lifespan)


# -----------------------------
# List Available Tool Categories
# -----------------------------
@app.get("/tool-categories")
async def list_tool_categories():
    """
    List all available tool categories that can be used in A2A_EXPOSED_TOOLS

    Returns:
        JSON with available categories, their tool count, and example usage
    """
    servers_dir = PROJECT_ROOT / "servers"

    categories = []
    for server_dir in servers_dir.iterdir():
        if server_dir.is_dir():
            server_file = server_dir / "server.py"
            if server_file.exists():
                categories.append(server_dir.name)

    # Get currently exposed tools
    exposed_tools_str = os.getenv("A2A_EXPOSED_TOOLS", "").strip()
    if exposed_tools_str:
        exposed = [s.strip() for s in exposed_tools_str.split(",") if s.strip()]
    else:
        exposed = categories  # All categories if not set

    return {
        "available_categories": sorted(categories),
        "total_categories": len(categories),
        "currently_exposed": sorted(exposed),
        "not_exposed": sorted(set(categories) - set(exposed)),
        "usage": {
            "description": "Set A2A_EXPOSED_TOOLS in .env to control which categories are exposed",
            "examples": [
                "A2A_EXPOSED_TOOLS=plex,location,text",
                "A2A_EXPOSED_TOOLS=  (empty = expose all)",
                "# A2A_EXPOSED_TOOLS  (commented out = expose all)"
            ]
        }
    }


# -----------------------------
# A2A Agent Card
# -----------------------------
@app.get("/.well-known/agent-card.json")
async def agent_card(request: Request):
    # Get the client to list available tools
    client = request.app.state.client

    # Fetch available tools from all sessions
    try:
        limit = 200
        all_tools = []

        # Get tools from all sessions
        for session_name, session in client.sessions.items():
            tools = await session.list_tools()
            all_tools.extend(tools)

        tool_list = [f"• {t.name}" for t in all_tools[:limit]]
        total_tools = len(all_tools)

        if total_tools > limit:
            tool_summary = "\n".join(tool_list) + f"\n... and {total_tools - limit} more tools"
        else:
            tool_summary = "\n".join(tool_list)

        description = f"""Your MCP tools exposed over A2A protocol.

Available Tools ({total_tools} total):
{tool_summary}

Use the 'a2a.discover' method to get full tool descriptions and schemas."""

    except Exception as e:
        print(f"⚠️ Error listing tools for agent card: {e}")
        import traceback
        traceback.print_exc()
        description = "Your MCP tools exposed over A2A (error listing tools)"

    card = {
        "name": "Local A2A Agent",
        "description": description,
        "version": "1.0.0",
        "capabilities": {
            "streaming": False,
            "tools": {
                "list": True,
                "execute": True
            },
            "resources": {
                "list": False,
                "read": False
            },
            "events": {
                "subscribe": False
            }
        },
        "defaultInputModes": ["text"],
        "defaultOutputModes": ["text"],
        "skills": [],
    }

    # Add endpoints if A2A_ENDPOINTS is configured
    if A2A_ENDPOINTS:
        endpoints = {}

        # Primary A2A endpoint (first in list)
        primary_endpoint = A2A_ENDPOINTS[0]
        endpoints["a2a"] = urljoin(primary_endpoint, "/a2a")

        # Additional endpoints (if any)
        if len(A2A_ENDPOINTS) > 1:
            endpoints["additional_a2a_endpoints"] = [
                urljoin(ep, "/a2a") for ep in A2A_ENDPOINTS[1:]
            ]

        card["endpoints"] = endpoints
        card["url"] = urljoin(primary_endpoint, "/.well-known/agent-card.json")

    return card


# -----------------------------
# A2A RPC Handler
# -----------------------------
@app.post("/a2a")
async def a2a_handler(req: RPCRequest, request: Request):
    client = request.app.state.client

    if req.method == "a2a.discover":
        # Collect tools from all sessions
        all_tools = []
        for session_name, session in client.sessions.items():
            tools = await session.list_tools()
            all_tools.extend(tools)

        return {
            "jsonrpc": "2.0",
            "id": req.id,
            "result": {
                "tools": [
                    {
                        "name": t.name,
                        "description": t.description or "",
                        "schema": (
                            t.args_schema.schema()
                            if hasattr(t, "args_schema") and t.args_schema
                            else {"type": "object"}
                        )
                    }
                    for t in all_tools
                ]
            }
        }

    if req.method == "a2a.call":
        tool_name = req.params["tool"]
        args = req.params["arguments"]

        # Find which session has this tool
        tool = None
        target_session = None

        for session_name, session in client.sessions.items():
            tools = await session.list_tools()
            tool = next((t for t in tools if t.name == tool_name), None)
            if tool:
                target_session = session
                break

        if tool is None or target_session is None:
            return {
                "jsonrpc": "2.0",
                "id": req.id,
                "error": f"Tool not found: {tool_name}"
            }

        # Call the tool using the correct session
        try:
            # session.call_tool returns a list of content items
            result = await target_session.call_tool(tool_name, args)

            # Result is typically a list of TextContent objects
            # Extract the text from the result
            if isinstance(result, list) and len(result) > 0:
                if hasattr(result[0], 'text'):
                    result_text = result[0].text
                else:
                    result_text = str(result[0])
            else:
                result_text = str(result)

            return {
                "jsonrpc": "2.0",
                "id": req.id,
                "result": result_text
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "jsonrpc": "2.0",
                "id": req.id,
                "error": f"Tool execution failed: {str(e)}"
            }

    return {
        "jsonrpc": "2.0",
        "id": req.id,
        "error": f"Unknown method: {req.method}"
    }

# -----------------------------
# Entrypoint
# -----------------------------
if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("🌐 A2A SERVER")
    print("=" * 60)

    # Show available tool categories
    servers_dir = PROJECT_ROOT / "servers"
    available_categories = []
    for server_dir in servers_dir.iterdir():
        if server_dir.is_dir() and (server_dir / "server.py").exists():
            available_categories.append(server_dir.name)

    print(f"📦 Available tool categories ({len(available_categories)}):")
    for cat in sorted(available_categories):
        print(f"   • {cat}")
    print()

    exposed_tools = os.getenv("A2A_EXPOSED_TOOLS", "").strip()
    if exposed_tools:
        exposed_list = [s.strip() for s in exposed_tools.split(",") if s.strip()]
        print(f"📋 A2A_EXPOSED_TOOLS: {', '.join(exposed_list)}")
        print(f"   Only these tool categories will be exposed")

        # Show what's NOT exposed
        not_exposed = set(available_categories) - set(exposed_list)
        if not_exposed:
            print(f"   Not exposed: {', '.join(sorted(not_exposed))}")
    else:
        print(f"📋 A2A_EXPOSED_TOOLS not set")
        print(f"   All tool categories will be exposed")

    print()

    if A2A_ENDPOINTS:
        print(f"🔗 A2A Server configured with {len(A2A_ENDPOINTS)} endpoint(s):")
        for i, ep in enumerate(A2A_ENDPOINTS, 1):
            print(f"   {i}. {ep}")
    else:
        print("ℹ️  No A2A_ENDPOINT or A2A_ENDPOINTS configured")

    print("=" * 60)
    print("ℹ️  View available tool categories: http://localhost:8010/tool-categories")
    print("=" * 60)
    print()

    uvicorn.run(app, host="0.0.0.0", port=8010)