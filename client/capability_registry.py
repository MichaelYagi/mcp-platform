"""
Capability Registry
===================
Single source of truth for what every tool in the platform can do.

Answers the questions agents need for dynamic planning:
  - "What tools exist?"
  - "What schemas do they accept?"
  - "What constraints or rate limits apply?"
  - "What tags describe this tool's behaviour?"

Design
------
- Built once at startup from the live tool list + INTENT_CATALOG metadata
- Passed explicitly through the call stack (no globals)
- Consumed by the WebSocket list_tools/list_capabilities handlers
- Consumed by the LangGraph planner to bind only relevant tools
- Each MCP server also exposes a list_capabilities tool so remote agents
  can introspect without going through the WebSocket layer

Tags vocabulary
---------------
  read          — tool only reads, never writes
  write         — tool creates or modifies data
  destructive   — tool deletes or irreversibly mutates
  search        — primary purpose is search/query
  external      — calls an external API or service
  vision        — processes image/visual input
  media         — operates on audio/video/image files
  calendar      — interacts with calendar data
  email         — interacts with email data
  notes         — interacts with note-taking systems
  code          — operates on source code
  system        — interacts with OS/hardware
  rag           — interacts with the RAG vector store
  ai            — calls an LLM or ML model

Rate limit vocabulary (string format)
--------------------------------------
  None          — no known limit
  "10/min"      — 10 calls per minute
  "100/hour"    — 100 calls per hour
  "ollama"      — limited by local Ollama inference speed
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("mcp_client")

# ─── Tag sets per tool name ────────────────────────────────────────────────────
# Covers all tools referenced in INTENT_CATALOG plus well-known server tools.
# Tools not listed here get an empty tag set — always safe, never wrong.

_TOOL_TAGS: dict[str, list[str]] = {
    # ── Location / time / weather ─────────────────────────────────────────────
    "get_location_tool":        ["read", "external"],
    "get_time_tool":            ["read", "external"],
    "get_weather_tool":         ["read", "external"],
    # ── System ────────────────────────────────────────────────────────────────
    "get_hardware_specs_tool":  ["read", "system"],
    "get_system_info":          ["read", "system"],
    "list_system_processes":    ["read", "system"],
    "terminate_process":        ["write", "destructive", "system"],
    # ── RAG ───────────────────────────────────────────────────────────────────
    "rag_add_tool":             ["write", "rag"],
    "rag_search_tool":          ["read", "search", "rag"],
    "rag_status_tool":          ["read", "rag"],
    "rag_browse_tool":          ["read", "rag"],
    "rag_list_sources_tool":    ["read", "rag"],
    "rag_diagnose_tool":        ["read", "rag"],
    # ── Image / Shashin ───────────────────────────────────────────────────────
    "analyze_image_tool":       ["read", "vision", "ai"],
    "shashin_search_tool":      ["read", "search", "media", "external"],
    "shashin_random_tool":      ["read", "media", "external"],
    "shashin_analyze_tool":     ["read", "vision", "media", "external", "ai"],
    "web_image_search_tool":    ["read", "search", "external"],
    # ── Text tools ────────────────────────────────────────────────────────────
    "summarize_text_tool":      ["read", "ai"],
    "explain_simplified_tool":  ["read", "ai"],
    "concept_contextualizer_tool": ["read", "ai"],
    "read_file_tool_handler":   ["read"],
    "web_search_tool":          ["read", "search", "external"],
    "web_fetch_tool":           ["read", "external"],
    "summarize_url_tool":       ["read", "external", "ai"],
    # ── Code review ───────────────────────────────────────────────────────────
    "review_code":              ["read", "code", "ai"],
    "search_code_in_directory": ["read", "search", "code"],
    "scan_code_directory":      ["read", "code"],
    "summarize_code":           ["read", "code", "ai"],
    "debug_fix":                ["read", "code", "ai"],
    # ── GitHub ────────────────────────────────────────────────────────────────
    "github_clone_repo":        ["write", "external", "code"],
    "github_list_files":        ["read", "external", "code"],
    "github_get_file_content":  ["read", "external", "code"],
    "github_cleanup_repo":      ["destructive", "code"],
    "analyze_project":          ["read", "code", "ai"],
    "analyze_code_file":        ["read", "code", "ai"],
    # ── Trilium notes ─────────────────────────────────────────────────────────
    "search_notes":             ["read", "search", "notes"],
    "search_by_label":          ["read", "search", "notes"],
    "get_note_by_id":           ["read", "notes"],
    "create_note":              ["write", "notes"],
    "update_note_content":      ["write", "notes"],
    "update_note_title":        ["write", "notes"],
    "delete_note":              ["destructive", "notes"],
    "add_label_to_note":        ["write", "notes"],
    "get_note_labels":          ["read", "notes"],
    "get_note_children":        ["read", "notes"],
    "get_recent_notes":         ["read", "notes"],
    # ── Google (Gmail + Calendar) ─────────────────────────────────────────────
    "gmail_get_unread":         ["read", "search", "email", "external"],
    "gmail_get_recent":         ["read", "search", "email", "external"],
    "gmail_get_email":          ["read", "email", "external"],
    "gmail_send_email":         ["write", "email", "external"],
    "calendar_get_today":       ["read", "calendar", "external"],
    "calendar_get_this_week":   ["read", "calendar", "external"],
    "calendar_create_event":    ["write", "calendar", "external"],
    # ── Plex / ML ─────────────────────────────────────────────────────────────
    "semantic_media_search_text": ["read", "search", "media", "ai"],
    "scene_locator_tool":       ["read", "search", "media"],
    "find_scene_by_title":      ["read", "search", "media"],
    "plex_ingest_batch":        ["write", "media", "rag"],
    "plex_find_unprocessed":    ["read", "media"],
    "plex_ingest_items":        ["write", "media", "rag"],
    "plex_ingest_single":       ["write", "media", "rag"],
    "record_viewing":           ["write", "media"],
    "train_recommender":        ["write", "ai"],
    "recommend_content":        ["read", "ai"],
    "get_recommender_stats":    ["read", "ai"],
    "import_plex_history":      ["write", "media"],
    "auto_train_from_plex":     ["write", "ai"],
    "reset_recommender":        ["destructive", "ai"],
    "auto_recommend_from_plex": ["read", "ai"],
    # ── A2A ───────────────────────────────────────────────────────────────────
    "send_a2a":                 ["write", "external"],
    "discover_a2a":             ["read", "external"],
    "send_a2a_streaming":       ["write", "external"],
    "send_a2a_batch":           ["write", "external"],
}

# ─── Rate limits per tool name ─────────────────────────────────────────────────
_TOOL_RATE_LIMITS: dict[str, str] = {
    "get_weather_tool":         "100/hour",     # Open-Meteo free tier
    "web_search_tool":          "100/hour",     # Ollama web search
    "web_fetch_tool":           "100/hour",
    "summarize_url_tool":       "100/hour",
    "web_image_search_tool":    "100/hour",     # Serper free tier
    "gmail_get_unread":         "250/hour",     # Gmail API quota
    "gmail_get_recent":         "250/hour",
    "gmail_get_email":          "250/hour",
    "gmail_send_email":         "100/day",
    "calendar_get_today":       "250/hour",
    "calendar_get_this_week":   "250/hour",
    "calendar_create_event":    "250/hour",
    "analyze_image_tool":       "ollama",
    "shashin_analyze_tool":     "ollama",
    "summarize_text_tool":      "ollama",
    "explain_simplified_tool":  "ollama",
    "concept_contextualizer_tool": "ollama",
    "summarize_code":           "ollama",
    "debug_fix":                "ollama",
    "train_recommender":        "10/day",
    "github_clone_repo":        "60/hour",      # GitHub API
}

# ─── Idempotency per tool name ─────────────────────────────────────────────────
# True = calling twice with same args produces same result with no side effects
_TOOL_IDEMPOTENT: dict[str, bool] = {
    # Reads are idempotent
    "get_location_tool":        True,
    "get_time_tool":            False,   # time changes
    "get_weather_tool":         False,   # weather changes
    "get_hardware_specs_tool":  True,
    "get_system_info":          False,
    "list_system_processes":    False,
    "rag_search_tool":          True,
    "rag_status_tool":          False,
    "rag_browse_tool":          False,
    "rag_list_sources_tool":    True,
    "shashin_search_tool":      True,
    "shashin_random_tool":      False,   # random by definition
    "shashin_analyze_tool":     True,
    "web_image_search_tool":    True,
    "summarize_text_tool":      True,
    "explain_simplified_tool":  True,
    "concept_contextualizer_tool": True,
    "read_file_tool_handler":   True,
    "web_search_tool":          True,
    "web_fetch_tool":           True,
    "summarize_url_tool":       True,
    "review_code":              True,
    "search_code_in_directory": True,
    "scan_code_directory":      True,
    "summarize_code":           True,
    "search_notes":             True,
    "get_note_by_id":           True,
    "get_note_labels":          True,
    "get_note_children":        True,
    "get_recent_notes":         False,
    "gmail_get_unread":         False,
    "gmail_get_recent":         False,
    "calendar_get_today":       False,
    "calendar_get_this_week":   False,
    "semantic_media_search_text": True,
    "recommend_content":        False,
    "get_recommender_stats":    False,
    # Writes are generally not idempotent
    "rag_add_tool":             False,
    "terminate_process":        False,
    "gmail_send_email":         False,
    "calendar_create_event":    False,
    "create_note":              False,
    "update_note_content":      False,
    "update_note_title":        False,
    "delete_note":              False,
    "add_label_to_note":        False,
    "plex_ingest_batch":        False,
    "plex_ingest_items":        False,
    "record_viewing":           False,
    "train_recommender":        False,
    "reset_recommender":        False,
}


# ─── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class ParamSchema:
    name: str
    type: str
    description: str
    required: bool
    default: Any = None


@dataclass
class ToolCapability:
    name: str
    description: str
    input_schema: list[ParamSchema]
    tags: list[str]
    rate_limit: str | None
    idempotent: bool
    enabled: bool
    source_server: str
    external: bool
    example: str
    # Output schema is freeform text for now — LLMs read the description
    output_description: str = ""


@dataclass
class ServerCapability:
    name: str
    tools: list[ToolCapability]


# ─── Schema extraction helper ─────────────────────────────────────────────────

def _extract_schema(tool) -> list[ParamSchema]:
    """
    Extract full parameter schema from a LangChain tool's args_schema.
    Returns all params (required and optional) with types and descriptions.
    """
    params = []
    try:
        schema = tool.args_schema
        if schema is None:
            return params

        if callable(getattr(schema, "model_json_schema", None)):
            s = schema.model_json_schema()
        elif callable(getattr(schema, "schema", None)):
            s = schema.schema()
        else:
            return params

        props = s.get("properties", {})
        required_set = set(s.get("required", []))
        defs = s.get("$defs", {})

        for param_name, prop in props.items():
            # Resolve $ref
            if "$ref" in prop:
                ref_key = prop["$ref"].split("/")[-1]
                prop = defs.get(ref_key, prop)

            # Handle anyOf (e.g. Optional[str] → anyOf: [{type: string}, {type: null}])
            if "anyOf" in prop:
                non_null = [t for t in prop["anyOf"] if t.get("type") != "null"]
                param_type = non_null[0].get("type", "string") if non_null else "string"
            else:
                param_type = prop.get("type", "string")

            description = prop.get("description", prop.get("title", ""))
            default = prop.get("default", None)

            params.append(ParamSchema(
                name=param_name,
                type=param_type,
                description=description,
                required=param_name in required_set,
                default=default,
            ))

    except Exception as e:
        logger.debug(f"⚠️ Schema extraction failed for {getattr(tool, 'name', '?')}: {e}")

    return params


# ─── Registry ─────────────────────────────────────────────────────────────────

_INTERNAL_TOOLS = {"list_skills", "read_skill", "parse_github_url", "list_capabilities"}


class CapabilityRegistry:
    """
    Central registry of all tool capabilities.

    Built once at startup from:
      - @tool_meta decorator on each tool function (primary source)
      - _TOOL_TAGS / _TOOL_RATE_LIMITS / _TOOL_IDEMPOTENT (fallback for
        tools that haven't been decorated yet)

    Passed explicitly through the call stack — never a global.
    """

    def __init__(self):
        self._tools: dict[str, ToolCapability] = {}
        self._servers: dict[str, ServerCapability] = {}

    def build(
        self,
        tools: list,
        tool_to_server: dict[str, str],
        external_servers: set[str],
        is_disabled_fn,
    ) -> None:
        """
        Populate the registry from the live tool list.

        For each tool, metadata is resolved in priority order:
          1. @tool_meta decorator on the underlying function
          2. Hardcoded fallback tables (_TOOL_TAGS etc.) for undecorated tools

        Args:
            tools:            List of LangChain tool objects
            tool_to_server:   tool_name → server_name mapping
            external_servers: set of server names that are external
            is_disabled_fn:   callable(tool_name, server_name) → bool
        """
        try:
            from client.tool_meta import get_meta
        except ImportError:
            get_meta = lambda fn: None  # noqa: E731

        seen_per_server: dict[str, set] = {}
        server_tools: dict[str, list[ToolCapability]] = {}

        for tool in tools:
            name = getattr(tool, "name", None)
            if not name or name in _INTERNAL_TOOLS:
                continue

            source = tool_to_server.get(name, "unknown")

            # Per-server dedup: skip if this tool was already registered from
            # this same server (handles duplicate entries within one server).
            server_seen = seen_per_server.setdefault(source, set())
            if name in server_seen:
                continue
            server_seen.add(name)

            enabled = not is_disabled_fn(name, source)

            # ── Resolve metadata from @tool_meta or fallback tables ───────────
            # Walk the decorator chain to find __tool_meta__.
            # We do NOT follow __wrapped__ to avoid circular references —
            # instead check the function and its func/coroutine attributes.
            fn   = getattr(tool, "func", None) or getattr(tool, "_func", None) or tool
            meta = get_meta(fn)
            # One level deeper (e.g. check_tool_enabled wraps tool_meta-decorated fn)
            if meta is None:
                inner = getattr(fn, "func", None) or getattr(fn, "_func", None)
                if inner and inner is not fn:
                    meta = get_meta(inner)

            # Extract sentinels from tool description — these survive the MCP boundary
            # injected by @tool_meta into the docstring
            _desc = (tool.description or "")
            def _extract_sentinel(desc, key):
                sentinel = f"\n\n__{key}__: "
                if sentinel in desc:
                    return desc.split(sentinel, 1)[1].split("\n\n__")[0].strip()
                return ""

            _example_from_desc         = _extract_sentinel(_desc, "example")
            _triggers_from_desc_raw    = _extract_sentinel(_desc, "triggers")
            _intent_cat_from_desc      = _extract_sentinel(_desc, "intent_category")
            _tags_from_desc_raw        = _extract_sentinel(_desc, "tags")
            _triggers_from_desc        = [t.strip() for t in _triggers_from_desc_raw.split(",") if t.strip()] if _triggers_from_desc_raw else []
            _tags_from_desc            = [t.strip() for t in _tags_from_desc_raw.split(",") if t.strip()] if _tags_from_desc_raw else []

            if meta is not None:
                tags            = meta.get("tags") or _tags_from_desc or _TOOL_TAGS.get(name, [])
                rate_limit      = meta.get("rate_limit") if meta.get("rate_limit") is not None \
                                  else _TOOL_RATE_LIMITS.get(name)
                idempotent      = meta.get("idempotent") if meta.get("idempotent") is not None \
                                  else _TOOL_IDEMPOTENT.get(name, True)
                example         = meta.get("example") or _example_from_desc or ""
                triggers        = meta.get("triggers") or _triggers_from_desc or []
                intent_category = meta.get("intent_category") or _intent_cat_from_desc or None
            else:
                tags            = _tags_from_desc or _TOOL_TAGS.get(name, [])
                rate_limit      = _TOOL_RATE_LIMITS.get(name)
                idempotent      = _TOOL_IDEMPOTENT.get(name, True)
                example         = _example_from_desc or ""
                triggers        = _triggers_from_desc
                intent_category = _intent_cat_from_desc or None

            cap = ToolCapability(
                name=name,
                description=(tool.description or "").strip(),
                input_schema=_extract_schema(tool),
                tags=tags,
                rate_limit=rate_limit,
                idempotent=idempotent,
                enabled=enabled,
                source_server=source,
                external=source in external_servers,
                example=example,
            )

            # Store triggers and intent_category on the live tool's metadata
            # so langgraph.py can read them without needing __tool_meta__
            if triggers or intent_category:
                if tool.metadata is None:
                    tool.metadata = {}
                if triggers:
                    tool.metadata["triggers"] = triggers
                if intent_category:
                    tool.metadata["intent_category"] = intent_category
                if tags:
                    tool.metadata["tags"] = tags

            # Key by "server:tool_name" so same tool name across servers are all stored.
            self._tools[f"{source}:{name}"] = cap
            server_tools.setdefault(source, []).append(cap)

        for server_name, tool_list in server_tools.items():
            self._servers[server_name] = ServerCapability(
                name=server_name,
                tools=tool_list,
            )

        # ── Register @tool_meta triggers into query_patterns ──────────────────
        # This runs in the CLIENT process after all tools are loaded.
        # For each decorated tool, register its triggers so classify() picks
        # them up via _get_catalog(). Tools without triggers are skipped.
        try:
            from client.query_patterns import register_tool_meta, invalidate_catalog
            _registered = 0
            for _key, cap in self._tools.items():
                _tname = cap.name
                _tsource = cap.source_server
                tool = next((t for t in tools
                             if getattr(t, "name", None) == _tname
                             and tool_to_server.get(_tname, "unknown") == _tsource), None)
                if not tool:
                    continue
                fn   = getattr(tool, "func", None) or getattr(tool, "_func", None) or tool
                meta = get_meta(fn)
                if meta is None:
                    inner = getattr(fn, "func", None) or getattr(fn, "_func", None)
                    if inner and inner is not fn:
                        meta = get_meta(inner)
                if meta and meta.get("triggers"):
                    register_tool_meta(
                        tool_name=_tname,
                        tags=meta.get("tags", []),
                        triggers=meta["triggers"],
                        intent_category=meta.get("intent_category"),
                        example=meta.get("example"),
                        web_search=meta.get("web_search", False),
                        skills=meta.get("skills", False),
                        priority=meta.get("priority", 2),
                    )
                    _registered += 1
            if _registered:
                invalidate_catalog()
                logger.info(f"🗂️  Registered {_registered} tool trigger sets into query_patterns")
        except Exception as _qp_err:
            logger.warning(f"⚠️  Could not register tool triggers into query_patterns: {_qp_err}")

        decorated   = sum(1 for t in self._tools.values() if t.tags)
        undecorated = len(self._tools) - decorated
        logger.info(
            f"🗂️  CapabilityRegistry built: {len(self._tools)} tools across "
            f"{len(self._servers)} servers "
            f"({decorated} decorated, {undecorated} using fallback tables)"
        )

    # ─── Query API ────────────────────────────────────────────────────────────

    def get_tool(self, name: str, server: str | None = None) -> ToolCapability | None:
        """
        Look up a tool by name.

        If server is provided, looks up "server:name" directly.
        If not, returns the first match across all servers with that tool name.
        """
        if server:
            return self._tools.get(f"{server}:{name}")
        for cap in self._tools.values():
            if cap.name == name:
                return cap
        return None

    def get_server(self, name: str) -> ServerCapability | None:
        return self._servers.get(name)

    def all_tools(self, enabled_only: bool = True) -> list[ToolCapability]:
        if enabled_only:
            return [t for t in self._tools.values() if t.enabled]
        return list(self._tools.values())

    def filter_by_tags(self, tags: list[str], enabled_only: bool = True) -> list[ToolCapability]:
        tag_set = set(tags)
        return [
            t for t in self.all_tools(enabled_only)
            if tag_set & set(t.tags)
        ]

    def filter_by_server(self, server_name: str, enabled_only: bool = True) -> list[ToolCapability]:
        return [
            t for t in self.all_tools(enabled_only)
            if t.source_server == server_name
        ]

    # ─── Serialisation ────────────────────────────────────────────────────────

    def tool_to_dict(self, cap: ToolCapability) -> dict:
        return {
            "name":          cap.name,
            "description":   cap.description,
            "input_schema":  [
                {
                    "name":        p.name,
                    "type":        p.type,
                    "description": p.description,
                    "required":    p.required,
                    "default":     p.default,
                }
                for p in cap.input_schema
            ],
            "tags":          cap.tags,
            "rate_limit":    cap.rate_limit,
            "idempotent":    cap.idempotent,
            "enabled":       cap.enabled,
            "source_server": cap.source_server,
            "external":      cap.external,
            "example":       cap.example,
        }

    def server_to_dict(self, srv: ServerCapability) -> dict:
        return {
            "name":  srv.name,
            "tools": [self.tool_to_dict(t) for t in srv.tools],
        }

    def to_dict(self, enabled_only: bool = True) -> dict:
        """Full registry serialised for WebSocket / agent consumption."""
        return {
            "tools": [
                self.tool_to_dict(t) for t in self.all_tools(enabled_only)
            ],
            "servers": [
                self.server_to_dict(s) for s in self._servers.values()
            ],
            "total_tools":   len(self.all_tools(enabled_only)),
            "total_servers": len(self._servers),
        }

    def to_agent_prompt(self, filter_tags: list[str] | None = None) -> str:
        """
        Compact text representation for injection into LLM context.
        Used by the planner node to give the agent a live capability map
        without consuming excessive tokens.

        Example output:
            [rag] rag_search_tool(query: str) — Semantic search over RAG database
            [read,search,rag] rag_browse_tool() — Browse recent RAG documents
        """
        tools = self.filter_by_tags(filter_tags) if filter_tags else self.all_tools()
        lines = []
        for t in tools:
            tags = ",".join(t.tags) if t.tags else "general"
            required = [p for p in t.input_schema if p.required]
            sig = ", ".join(f"{p.name}: {p.type}" for p in required)
            desc = t.description[:120] + "…" if len(t.description) > 120 else t.description
            lines.append(f"[{tags}] {t.name}({sig}) — {desc}")
        return "\n".join(lines)