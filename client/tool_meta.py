"""
tool_meta — single-source tool metadata decorator
==================================================
Attach routing and capability metadata directly to an MCP tool function.
This is the ONE place you update when adding a new tool.

Usage
-----
    from client.tool_meta import tool_meta

    @mcp.tool()
    @tool_meta(
        tags=["read", "search", "rag"],
        triggers=["rag", "search my knowledge", "what do you know about"],
        rate_limit="100/hour",
        idempotent=True,
        example='use rag_search_tool: query=""',
        text_fields=["preview"],
    )
    def rag_search_tool(query: str) -> str:
        ...

Fields
------
tags (list[str], required)
    Capability tags from the standard vocabulary:
    read, write, destructive, search, external, vision, media,
    calendar, email, notes, code, system, rag, ai

triggers (list[str], optional)
    Natural language keywords/phrases that should route to this tool.
    Single words ("movie", "film") or short phrases ("what do you know about").
    Used to auto-build INTENT_CATALOG entries — no regex knowledge needed.
    Tools with the same tags share a routing group automatically.

rate_limit (str | None, optional)
    "100/hour", "10/day", "ollama", or None.

idempotent (bool, optional, default True)
    True if calling twice with the same args has no side effects.

example (str, optional)
    Pre-fill text for the tools panel UI. Defaults to auto-generated
    use <tool_name>: param1='' param2=''.

text_fields (list[str], optional)
    Which response fields contain real text content (for the client.py
    direct-dispatch result formatter). E.g. ["preview", "snippet"].
    Tells the formatter to render those fields directly instead of
    sending everything to the LLM.

intent_category (str, optional)
    Override the routing group name. Normally derived from the primary tag.
    Use when you want fine-grained control (e.g. "shashin_search" vs
    "shashin_analyze" even though both are tagged ["media"]).
"""

from __future__ import annotations

from typing import Any


# Attribute name stored on the function
_META_ATTR = "__tool_meta__"


def tool_meta(
    tags: list[str] | None = None,
    triggers: list[str] | None = None,
    rate_limit: str | None = None,
    idempotent: bool = True,
    example: str | None = None,
    text_fields: list[str] | None = None,
    intent_category: str | None = None,
    web_search: bool = False,
    skills: bool = False,
    priority: int = 2,
) -> Any:
    """
    Decorator that attaches routing + capability metadata to an MCP tool function.
    Must be applied AFTER @mcp.tool() so it wraps the original function.

    The metadata is stored as __tool_meta__ on the function object and read
    by CapabilityRegistry.build() and query_patterns.build_intent_catalog().
    Also auto-registers routing triggers into query_patterns at import time.
    """
    def decorator(fn):
        meta = {
            "tags":            tags or [],
            "triggers":        triggers or [],
            "rate_limit":      rate_limit,
            "idempotent":      idempotent,
            "example":         example,
            "text_fields":     text_fields or [],
            "intent_category": intent_category,
            "web_search":      web_search,
            "skills":          skills,
            "priority":        priority,
        }
        setattr(fn, _META_ATTR, meta)
        # Do NOT set __wrapped__ — a self-referential __wrapped__ = fn
        # causes inspect.signature() to loop forever when mcp.run() introspects tools.

        # Encode example and triggers into the docstring so they survive the MCP
        # process boundary. capability_registry.py extracts them from tool.description.
        doc_suffix = ""
        if example:
            doc_suffix += f"\n\n__example__: {example}"
        if triggers:
            doc_suffix += f"\n\n__triggers__: {','.join(triggers)}"
        if intent_category:
            doc_suffix += f"\n\n__intent_category__: {intent_category}"
        if tags:
            doc_suffix += f"\n\n__tags__: {','.join(tags)}"
        if doc_suffix:
            if fn.__doc__:
                fn.__doc__ = fn.__doc__.rstrip() + doc_suffix
            else:
                fn.__doc__ = doc_suffix.lstrip()

        return fn
    return decorator


def get_meta(fn) -> dict | None:
    """Return the __tool_meta__ dict for a function, or None if not decorated."""
    return getattr(fn, _META_ATTR, None)


def get_meta_attr(fn, key: str, default=None):
    """Convenience: read one field from __tool_meta__, with a default."""
    meta = get_meta(fn)
    if meta is None:
        return default
    return meta.get(key, default)