"""
Shared tool utilities for MCP Client.
Centralises tool→server resolution used by :tools and :health.
"""

from pathlib import Path
import json

# Pattern matching of last resort — same list as commands.py
CATEGORY_PATTERNS = {
    'todo':           ['todo', 'task'],
    'knowledge_base': ['entry', 'entries', 'knowledge'],
    'plex':           ['plex', 'media', 'scene', 'semantic_media', 'import_plex',
                       'train_recommender', 'recommend', 'record_viewing',
                       'auto_train', 'auto_recommend'],
    'rag':            ['rag_'],
    'system':         ['system', 'hardware', 'process'],
    'location':       ['location', 'time', 'weather'],
    'text':           ['text', 'summarize', 'chunk', 'explain', 'concept'],
    'code':           ['code', 'debug'],
}


async def resolve_tool_server(tools: list, mcp_agent, project_root: Path) -> dict:
    """
    Returns {tool_name: server_name} for every tool in `tools`.

    Resolution order (mirrors :tools command):
      1. Live session.list_tools() query  — most accurate, catches deepwiki etc.
      2. tool.metadata['source_server']   — load-time tag, catches coingecko 500s
      3. Pattern matching                 — last resort
    """
    # ── Step 1: live session queries ─────────────────────────────────
    tool_to_server: dict[str, str] = {}
    if mcp_agent and hasattr(mcp_agent, 'client') and hasattr(mcp_agent.client, 'sessions'):
        for server_name, session in mcp_agent.client.sessions.items():
            try:
                session_tools = await session.list_tools()
                for t in session_tools:
                    tool_to_server[t.name] = server_name
            except Exception:
                pass  # session failed — handled by metadata fallback below

    # ── Step 2 + 3: fill gaps via metadata then pattern matching ─────
    for tool in tools:
        tool_name = getattr(tool, 'name', None)
        if not tool_name or tool_name in tool_to_server:
            continue

        # 2. metadata tag (set in client.py _make_tool recovery path)
        server_name = None
        try:
            meta = getattr(tool, 'meta', None) or getattr(tool, 'metadata', None)
            if isinstance(meta, dict):
                server_name = meta.get('source_server')
        except Exception:
            pass

        # 3. pattern matching
        if not server_name:
            server_name = 'other'
            for cat, patterns in CATEGORY_PATTERNS.items():
                if any(p in tool_name.lower() for p in patterns):
                    server_name = cat
                    break

        tool_to_server[tool_name] = server_name

    return tool_to_server


def load_external_server_names(project_root: Path) -> set:
    """Returns the set of external server names from external_servers.json."""
    cfg = project_root / "external_servers.json"
    if not cfg.exists():
        return set()
    try:
        return set(json.loads(cfg.read_text(encoding="utf-8")).get("external_servers", {}).keys())
    except Exception:
        return set()