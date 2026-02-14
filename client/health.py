"""
Health Check Module for MCP Client
Provides :health and :health <name> commands
"""

import asyncio
import json
import time
import logging
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger("mcp_client")

# ─────────────────────────────────────────────────────────────────────
# SESSION STATS — populated by call_tools_with_stop_check in langgraph.py
# ─────────────────────────────────────────────────────────────────────
tool_session_stats = {}
# Structure per tool_name:
#   { "calls": int, "errors": int, "total_time": float, "last_error": str|None }

def record_tool_call(tool_name: str, duration: float, error: str = None):
    """Called by langgraph.py after every tool execution."""
    if tool_name not in tool_session_stats:
        tool_session_stats[tool_name] = {
            "calls": 0, "errors": 0, "total_time": 0.0, "last_error": None
        }
    s = tool_session_stats[tool_name]
    s["calls"] += 1
    s["total_time"] += duration
    if error:
        s["errors"] += 1
        s["last_error"] = error


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────

def _ms(seconds: float) -> str:
    return f"{seconds * 1000:.0f}ms"

def _status(ok: bool) -> str:
    return "✅" if ok else "❌"

def _warn(ok) -> str:
    if ok is True:   return "✅"
    if ok is False:  return "❌"
    return "⚠️"


async def _tcp_ping(host: str, port: int, timeout: float = 3.0) -> float | None:
    """Returns round-trip time in seconds, or None on failure."""
    try:
        t0 = time.perf_counter()
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=timeout
        )
        elapsed = time.perf_counter() - t0
        writer.close()
        await writer.wait_closed()
        return elapsed
    except Exception:
        return None


async def _http_probe(url: str, timeout: float = 5.0) -> tuple[bool, float | None, int | None]:
    """
    Returns (success, latency_seconds, http_status_code).
    Tries a POST to the MCP endpoint with an initialize message.
    """
    try:
        import httpx
        t0 = time.perf_counter()
        async with httpx.AsyncClient(timeout=timeout) as hc:
            r = await hc.post(
                url,
                json={"jsonrpc": "2.0", "id": 1, "method": "initialize",
                      "params": {"protocolVersion": "2024-11-05",
                                 "capabilities": {},
                                 "clientInfo": {"name": "health-check", "version": "1.0"}}},
                headers={"Content-Type": "application/json"}
            )
        elapsed = time.perf_counter() - t0
        return r.status_code < 500, elapsed, r.status_code
    except Exception as e:
        return False, None, None


async def _list_tools_probe(session, timeout: float = 5.0) -> tuple[bool, float | None, list | None, str | None]:
    """Returns (success, latency, tools_list, error_msg)."""
    try:
        t0 = time.perf_counter()
        tools = await asyncio.wait_for(session.list_tools(), timeout=timeout)
        elapsed = time.perf_counter() - t0
        return True, elapsed, tools, None
    except Exception as e:
        return False, None, None, str(e)


def _validate_schemas(tools: list) -> tuple[bool, list]:
    """Returns (all_valid, list_of_issues)."""
    issues = []
    names = []
    for t in tools:
        if t.name in names:
            issues.append(f"Duplicate tool name: {t.name}")
        names.append(t.name)
        schema = getattr(t, "inputSchema", None)
        if schema is not None:
            if not isinstance(schema, dict):
                issues.append(f"{t.name}: inputSchema is not a dict")
            else:
                # Must have type or properties
                if "type" not in schema and "properties" not in schema:
                    issues.append(f"{t.name}: inputSchema missing 'type' and 'properties'")
    return len(issues) == 0, issues


# Sanity call payloads per tool name (lightweight, read-only)
SANITY_PAYLOADS = {
    "rag_search_tool":            {"query": "test", "top_k": 1},
    "rag_status_tool":            {},
    "rag_list_sources_tool":      {},
    "get_system_info":            {},
    "get_hardware_specs_tool":    {},
    "get_location_tool":          {"location": "Vancouver"},
    "get_time_tool":              {"city": "Vancouver"},
    "get_weather_tool":           {"location": "Vancouver"},
    "list_todo_items":            {},
    "list_entries":               {},
    "plex_get_stats":             {},
    "get_simple_price":           {"ids": "bitcoin", "vs_currencies": "usd", "jq_filter": "."},
    "get_list_coins_categories":  {"jq_filter": ".[0]"},
    "read_wiki_structure":        {"repoUrl": "https://github.com/torvalds/linux"},
    "get_global":                 {"jq_filter": "."},
    "search_docs":                {"query": "test"},
    "github_list_files":          {},   # will fail gracefully — that's fine
}

def _get_sanity_payload(tool_name: str) -> dict | None:
    """Return a known-safe payload, or None if we don't have one."""
    return SANITY_PAYLOADS.get(tool_name)


# ─────────────────────────────────────────────────────────────────────
# EXTERNAL SERVER CONFIG
# ─────────────────────────────────────────────────────────────────────

def _load_external_servers(project_root: Path) -> dict:
    cfg = project_root / "client" / "external_servers.json"
    if not cfg.exists():
        return {}
    try:
        return json.loads(cfg.read_text(encoding="utf-8")).get("external_servers", {})
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────
# SINGLE SERVER HEALTH
# ─────────────────────────────────────────────────────────────────────

async def check_server_health(server_name: str, session, mcp_agent, tools: list,
                               project_root: Path, verbose: bool = True) -> dict:
    """Full health check for one server (local or external)."""
    result = {
        "name": server_name,
        "reachable": False,
        "tcp_latency": None,
        "http_latency": None,
        "http_status": None,
        "list_tools_ok": False,
        "list_tools_latency": None,
        "tool_count": 0,
        "schema_ok": False,
        "schema_issues": [],
        "sanity_results": {},
        "protocol_version": None,
        "server_version": None,
        "errors": []
    }

    lines = []
    lines.append(f"\n{'═'*58}")
    lines.append(f"  SERVER: {server_name}")
    lines.append(f"{'═'*58}")

    # ── 1. Reachability ──────────────────────────────────────────
    lines.append("\n📡 1. REACHABILITY")

    ext_servers = _load_external_servers(project_root)
    server_cfg = ext_servers.get(server_name, {})
    url = server_cfg.get("url")

    if url:
        parsed = urlparse(url)
        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)

        tcp_lat = await _tcp_ping(host, port)
        result["tcp_latency"] = tcp_lat
        result["reachable"] = tcp_lat is not None
        lines.append(f"   TCP connect ({host}:{port}): {_status(tcp_lat is not None)} {_ms(tcp_lat) if tcp_lat else 'TIMEOUT'}")

        ok, http_lat, status = await _http_probe(url)
        result["http_latency"] = http_lat
        result["http_status"] = status
        lines.append(f"   HTTP handshake:             {_status(ok)} {_ms(http_lat) if http_lat else 'FAILED'} (HTTP {status})")

        if http_lat:
            result["reachable"] = True
    else:
        # Local stdio server — check via session
        result["reachable"] = session is not None
        lines.append(f"   Local stdio session:        {_status(session is not None)}")

    # ── 2. Tool Enumeration ──────────────────────────────────────
    lines.append("\n📋 2. TOOL ENUMERATION")

    if session:
        ok, lat, tools_list, err = await _list_tools_probe(session)
        result["list_tools_ok"] = ok
        result["list_tools_latency"] = lat

        if ok and tools_list:
            result["tool_count"] = len(tools_list)
            schema_ok, schema_issues = _validate_schemas(tools_list)
            result["schema_ok"] = schema_ok
            result["schema_issues"] = schema_issues

            lines.append(f"   list_tools():               {_status(ok)} {_ms(lat) if lat else ''}")
            lines.append(f"   Tool count:                 {len(tools_list)}")
            lines.append(f"   Schemas valid:              {_status(schema_ok)}")
            if schema_issues:
                for issue in schema_issues[:5]:
                    lines.append(f"      ⚠️  {issue}")
            lines.append(f"   Tool names unique:          {_status(not any('Duplicate' in i for i in schema_issues))}")
        else:
            lines.append(f"   list_tools():               ❌ {err or 'failed'}")
            result["errors"].append(f"list_tools failed: {err}")
    else:
        lines.append("   No active session — skipping")

    # ── 3. Sanity Invocations ────────────────────────────────────
    lines.append("\n🔧 3. TOOL SANITY CHECKS")

    # Use tool_to_server map if available, fall back to metadata
    server_tools = [
        t for t in tools
        if (isinstance(getattr(t, 'metadata', None), dict) and
            t.metadata.get('source_server') == server_name)
    ]

    if not server_tools:
        lines.append("   No tools found for this server")
    else:
        checked = 0
        for tool in server_tools[:5]:  # check up to 5 tools
            payload = _get_sanity_payload(tool.name)
            if payload is None:
                result["sanity_results"][tool.name] = "skipped"
                continue
            try:
                t0 = time.perf_counter()
                res = await asyncio.wait_for(tool.ainvoke(payload), timeout=10.0)
                elapsed = time.perf_counter() - t0
                res_str = str(res).strip()
                is_stub = res_str.startswith(f"Tool {tool.name} called") or res_str == ""
                if is_stub:
                    result["sanity_results"][tool.name] = "stub"
                    lines.append(f"   {tool.name[:40]:<40} ⚠️  STUB (recovery tool, not real session)")
                else:
                    result["sanity_results"][tool.name] = "ok"
                    lines.append(f"   {tool.name[:40]:<40} ✅ {_ms(elapsed)}")
                checked += 1
            except asyncio.TimeoutError:
                result["sanity_results"][tool.name] = "timeout"
                lines.append(f"   {tool.name[:40]:<40} ⏱️  TIMEOUT")
                result["errors"].append(f"{tool.name}: sanity timeout")
            except Exception as e:
                result["sanity_results"][tool.name] = f"error: {str(e)[:60]}"
                lines.append(f"   {tool.name[:40]:<40} ❌ {str(e)[:40]}")
        if checked == 0:
            lines.append("   No sanity payloads configured for this server's tools")

    # ── 4. Latency Summary ───────────────────────────────────────
    lines.append("\n⏱️  4. LATENCY SUMMARY")
    if result["tcp_latency"]:
        lines.append(f"   TCP connect:    {_ms(result['tcp_latency'])}")
    if result["http_latency"]:
        lines.append(f"   HTTP handshake: {_ms(result['http_latency'])}")
    if result["list_tools_latency"]:
        lines.append(f"   list_tools():   {_ms(result['list_tools_latency'])}")

    # Session stats
    session_tool_times = []
    for tool in server_tools:
        stats = tool_session_stats.get(tool.name)
        if stats and stats["calls"] > 0:
            avg = stats["total_time"] / stats["calls"]
            session_tool_times.append(avg)
    if session_tool_times:
        avg_all = sum(session_tool_times) / len(session_tool_times)
        lines.append(f"   Avg tool call (session): {_ms(avg_all)}")
    else:
        lines.append("   Avg tool call (session): no data yet")

    # ── 5. Version Info ──────────────────────────────────────────
    lines.append("\n🔖 5. VERSION / PROTOCOL")
    if url and result["reachable"]:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as hc:
                r = await hc.post(
                    url,
                    json={"jsonrpc": "2.0", "id": 1, "method": "initialize",
                          "params": {"protocolVersion": "2024-11-05",
                                     "capabilities": {},
                                     "clientInfo": {"name": "health-check", "version": "1.0"}}},
                    headers={"Content-Type": "application/json"}
                )
                if r.status_code == 200:
                    data = r.json()
                    res = data.get("result", {})
                    proto = res.get("protocolVersion", "unknown")
                    srv_info = res.get("serverInfo", {})
                    srv_ver = srv_info.get("version", "unknown")
                    srv_name = srv_info.get("name", "unknown")
                    result["protocol_version"] = proto
                    result["server_version"] = srv_ver
                    lines.append(f"   Protocol version: {proto}")
                    lines.append(f"   Server name:      {srv_name}")
                    lines.append(f"   Server version:   {srv_ver}")
        except Exception as e:
            lines.append(f"   Version info unavailable: {str(e)[:50]}")
    else:
        lines.append("   Version info: N/A (local stdio)")

    # ── Summary ──────────────────────────────────────────────────
    lines.append(f"\n{'─'*58}")
    overall = result["reachable"] and result["list_tools_ok"] and result["schema_ok"]
    lines.append(f"  Overall: {_status(overall)}  {'HEALTHY' if overall else 'DEGRADED'}")
    if result["errors"]:
        lines.append(f"  Errors:  {len(result['errors'])}")
        for e in result["errors"][:3]:
            lines.append(f"    • {e}")

    return result, "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# SINGLE TOOL HEALTH
# ─────────────────────────────────────────────────────────────────────

async def check_tool_health(tool_name: str, tools: list) -> str:
    """Health check for a specific tool."""
    tool = next((t for t in tools if getattr(t, 'name', None) == tool_name), None)

    lines = []
    lines.append(f"\n{'═'*58}")
    lines.append(f"  TOOL: {tool_name}")
    lines.append(f"{'═'*58}")

    if not tool:
        lines.append(f"  ❌ Tool not found")
        return "\n".join(lines)

    # Source server
    source = (getattr(tool, 'metadata', None) or {}).get('source_server', 'unknown')

    # Check if disabled
    try:
        from tools.tool_control import is_tool_enabled
        enabled = is_tool_enabled(tool_name, source)
    except ImportError:
        enabled = True

    lines.append(f"  Server:      {source}")

    if not enabled:
        lines.append(f"  Status:      ❌ DISABLED")
        lines.append(f"\n  This tool is disabled via DISABLED_TOOLS in .env")
        lines.append(f"  To enable: remove '{source}:{tool_name}' or '{source}:*' from DISABLED_TOOLS")
        return "\n".join(lines)

    lines.append(f"  Status:      ✅ Available")

    # Session stats
    stats = tool_session_stats.get(tool_name)
    lines.append(f"\n📊 SESSION STATS")
    if stats and stats["calls"] > 0:
        avg = stats["total_time"] / stats["calls"]
        lines.append(f"  Calls this session:  {stats['calls']}")
        lines.append(f"  Errors:              {stats['errors']}")
        lines.append(f"  Avg response time:   {_ms(avg)}")
        lines.append(f"  Last error:          {stats['last_error'] or 'none'}")
    else:
        lines.append("  No calls recorded this session")

    # Sanity check
    lines.append(f"\n🔧 SANITY CHECK")
    payload = _get_sanity_payload(tool_name)
    if payload is None:
        lines.append("  No sanity payload configured — skipping live call")
    else:
        try:
            t0 = time.perf_counter()
            res = await asyncio.wait_for(tool.ainvoke(payload), timeout=10.0)
            elapsed = time.perf_counter() - t0
            res_str = str(res).strip() if res is not None else ""
            is_stub = res_str.startswith(f"Tool {tool_name} called") or res_str == ""
            if is_stub:
                lines.append(f"  Live call:    ⚠️  STUB — recovery tool, not real MCP session")
                lines.append(f"  Fix:          check DISABLED_TOOLS or server init errors")
            else:
                lines.append(f"  Live call:    ✅ {_ms(elapsed)}")
                preview = res_str[:120].replace('\n', ' ')
                lines.append(f"  Response:     {preview}...")
        except asyncio.TimeoutError:
            lines.append("  Live call:    ⏱️  TIMEOUT (>10s)")
        except Exception as e:
            lines.append(f"  Live call:    ❌ {str(e)[:80]}")

    # Schema info
    lines.append(f"\n📋 SCHEMA")
    schema = getattr(tool, 'args_schema', None)
    if schema:
        try:
            fields = schema.model_fields if hasattr(schema, 'model_fields') else {}
            lines.append(f"  Parameters: {', '.join(fields.keys()) or 'none'}")
        except Exception:
            lines.append("  Schema: present")
    else:
        lines.append("  Schema: none")

    lines.append(f"\n  Description: {(tool.description or '')[:120]}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# OVERVIEW
# ─────────────────────────────────────────────────────────────────────

async def health_overview(mcp_agent, tools: list, project_root: Path) -> str:
    """Quick overview of all servers and tools."""
    lines = []
    lines.append(f"\n{'═'*58}")
    lines.append(f"  MCP HEALTH OVERVIEW")
    lines.append(f"{'═'*58}")

    # Gather server names
    local_servers = {}
    if hasattr(mcp_agent, 'client') and hasattr(mcp_agent.client, 'sessions'):
        local_servers = mcp_agent.client.sessions

    ext_servers = _load_external_servers(project_root)

    all_server_names = set(local_servers.keys()) | set(ext_servers.keys())

    # Per-server quick status
    lines.append(f"\n{'─'*58}")
    lines.append(f"  {'SERVER':<25} {'STATUS':<10} {'TOOLS':<10}  {'AVG LATENCY'}")
    lines.append(f"{'─'*58}")

    try:
        from tools.tool_control import is_tool_enabled
        total_tools = sum(
            1 for t in tools
            if is_tool_enabled(
                getattr(t, 'name', ''),
                (getattr(t, 'metadata', None) or {}).get('source_server', '')
            )
        )
    except ImportError:
        total_tools = len(tools)
    healthy = 0
    degraded = 0

    # Resolve tool→server using shared utility (same logic as :tools command)
    from client.tool_utils import resolve_tool_server
    tool_to_server = await resolve_tool_server(tools, mcp_agent, project_root)

    for sname in sorted(all_server_names):
        session = local_servers.get(sname)
        url = ext_servers.get(sname, {}).get("url")
        is_external = sname in ext_servers

        # Quick reachability
        reachable = False
        if url:
            parsed = urlparse(url)
            host = parsed.hostname
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            lat = await _tcp_ping(host, port, timeout=2.0)
            reachable = lat is not None
        elif session:
            reachable = True

        # Count total and enabled tools for this server (via resolved map)
        server_tools_all = [
            t for t in tools
            if tool_to_server.get(getattr(t, 'name', '')) == sname
        ]
        total_server = len(server_tools_all)
        try:
            from tools.tool_control import is_tool_enabled
            enabled_server = sum(
                1 for t in server_tools_all
                if is_tool_enabled(getattr(t, 'name', ''), sname)
            )
        except ImportError:
            enabled_server = total_server

        # Tool availability icon + ratio
        if total_server == 0:
            tools_str = f"{'--':>8}"
        elif enabled_server == 0:
            tools_str = f"❌ {enabled_server}/{total_server}"
        elif enabled_server == total_server:
            tools_str = f"✅ {enabled_server}/{total_server}"
        else:
            tools_str = f"⚡ {enabled_server}/{total_server}"

        # Avg session latency across all called tools on this server
        server_tool_times = []
        for t in server_tools_all:
            tname = getattr(t, 'name', '')
            stats = tool_session_stats.get(tname)
            if stats and stats["calls"] > 0:
                server_tool_times.append(stats["total_time"] / stats["calls"])

        avg_str = _ms(sum(server_tool_times) / len(server_tool_times)) if server_tool_times else "—"
        ext_tag = " [ext]" if is_external else ""
        status_icon = "✅" if reachable else "❌"

        if reachable:
            healthy += 1
        else:
            degraded += 1

        lines.append(f"  {(sname + ext_tag):<25} {status_icon:<10} {tools_str:<10}  {avg_str}")

    lines.append(f"{'─'*58}")
    try:
        from tools.tool_control import is_tool_enabled
        disabled_count = sum(
            1 for t in tools
            if not is_tool_enabled(
                getattr(t, 'name', ''),
                (getattr(t, 'metadata', None) or {}).get('source_server', '')
            )
        )
    except ImportError:
        disabled_count = 0

    lines.append(f"  Servers: {healthy} healthy, {degraded} unreachable")
    lines.append(f"  Tools:   {total_tools} available, {disabled_count} disabled")

    # Session activity
    active_tools = {k: v for k, v in tool_session_stats.items() if v["calls"] > 0}
    if active_tools:
        lines.append(f"\n{'─'*58}")
        lines.append(f"  SESSION ACTIVITY (this session)")
        lines.append(f"{'─'*58}")
        lines.append(f"  {'TOOL':<35} {'CALLS':>5}  {'ERRORS':>6}  {'AVG'}")
        for tname, stats in sorted(active_tools.items(), key=lambda x: -x[1]["calls"])[:10]:
            avg = stats["total_time"] / stats["calls"]
            err_str = str(stats["errors"]) if stats["errors"] else "—"
            lines.append(f"  {tname:<35} {stats['calls']:>5}  {err_str:>6}  {_ms(avg)}")
    else:
        lines.append(f"\n  No tool activity recorded this session yet")

    lines.append(f"\n  Run ':health <server>' for deep-dive on a server")
    lines.append(f"  Run ':health <tool>'   for deep-dive on a tool")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────

async def run_health_check(args: str, mcp_agent, tools: list, project_root: Path) -> str:
    """
    Main dispatcher for :health command.
    args = "" → overview
    args = "<name>" → server or tool deep-dive
    """
    name = args.strip()

    if not name:
        return await health_overview(mcp_agent, tools, project_root)

    # Is it a tool?
    tool_names = [getattr(t, 'name', None) for t in tools]
    if name in tool_names:
        return await check_tool_health(name, tools)

    # Is it a server?
    local_servers = {}
    if hasattr(mcp_agent, 'client') and hasattr(mcp_agent.client, 'sessions'):
        local_servers = mcp_agent.client.sessions

    ext_servers = _load_external_servers(project_root)
    all_server_names = set(local_servers.keys()) | set(ext_servers.keys())

    if name in all_server_names:
        session = local_servers.get(name)
        _, output = await check_server_health(name, session, mcp_agent, tools, project_root)
        return output

    return f"❌ '{name}' not found. Use ':health' for overview, or specify a valid server/tool name."