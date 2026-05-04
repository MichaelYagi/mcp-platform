"""
Extended tests for client/query_patterns.py
Covers: register_tool_meta, build_intent_catalog, invalidate_catalog,
        ROUTER_* patterns, extract_research_sources, WEB_SEARCH patterns
"""
import re
import pytest
from unittest.mock import patch


# ═══════════════════════════════════════════════════════════════════
# register_tool_meta & build_intent_catalog
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestRegisterToolMeta:
    def setup_method(self):
        """Clear dynamic registrations before each test."""
        from client import query_patterns as qp
        qp._DYNAMIC_REGISTRATIONS.clear()
        qp._EFFECTIVE_CATALOG = []

    def teardown_method(self):
        from client import query_patterns as qp
        qp._DYNAMIC_REGISTRATIONS.clear()
        qp._EFFECTIVE_CATALOG = []

    def test_register_appends_to_dynamic(self):
        from client.query_patterns import register_tool_meta, _DYNAMIC_REGISTRATIONS
        register_tool_meta(
            tool_name="my_tool",
            tags=["search"],
            triggers=["find me stuff"],
        )
        assert any(r["tool_name"] == "my_tool" for r in _DYNAMIC_REGISTRATIONS)

    def test_register_stores_all_fields(self):
        from client.query_patterns import register_tool_meta, _DYNAMIC_REGISTRATIONS
        register_tool_meta(
            tool_name="my_tool",
            tags=["search", "external"],
            triggers=["find", "search for"],
            intent_category="my_category",
            example='use my_tool: q=""',
            web_search=True,
            skills=False,
            priority=1,
        )
        reg = next(r for r in _DYNAMIC_REGISTRATIONS if r["tool_name"] == "my_tool")
        assert reg["tags"] == ["search", "external"]
        assert reg["triggers"] == ["find", "search for"]
        assert reg["intent_category"] == "my_category"
        assert reg["web_search"] is True
        assert reg["priority"] == 1

    def test_build_catalog_no_dynamic_returns_static(self):
        from client.query_patterns import build_intent_catalog, INTENT_CATALOG
        result = build_intent_catalog()
        assert result is INTENT_CATALOG

    def test_build_catalog_with_dynamic_entry(self):
        from client.query_patterns import register_tool_meta, build_intent_catalog
        register_tool_meta(
            tool_name="unique_novel_tool_xyz",
            tags=["ai"],
            triggers=["unique novel phrase xyz"],
            intent_category="novel_category_xyz",
        )
        catalog = build_intent_catalog()
        names = [e["name"] for e in catalog]
        assert "novel_category_xyz" in names

    def test_build_catalog_skips_static_tools(self):
        """Tools already in static catalog should not create duplicate entries."""
        from client.query_patterns import register_tool_meta, build_intent_catalog, INTENT_CATALOG
        # Pick a tool that's already in the static catalog
        existing_tool = INTENT_CATALOG[0]["tools"][0] if INTENT_CATALOG[0]["tools"] else None
        if not existing_tool:
            pytest.skip("No tools in first catalog entry")

        register_tool_meta(
            tool_name=existing_tool,
            tags=["search"],
            triggers=["some trigger"],
        )
        catalog = build_intent_catalog()
        # Count how many entries contain this tool
        count = sum(1 for e in catalog if existing_tool in e.get("tools", []))
        assert count == 1  # not duplicated

    def test_build_catalog_groups_same_category(self):
        """Two tools with same intent_category should merge into one entry."""
        from client.query_patterns import register_tool_meta, build_intent_catalog
        register_tool_meta("tool_a_xyz", ["search"], ["find a xyz"], intent_category="shared_cat_xyz")
        register_tool_meta("tool_b_xyz", ["search"], ["find b xyz"], intent_category="shared_cat_xyz")
        catalog = build_intent_catalog()
        entries = [e for e in catalog if e["name"] == "shared_cat_xyz"]
        assert len(entries) == 1
        assert "tool_a_xyz" in entries[0]["tools"]
        assert "tool_b_xyz" in entries[0]["tools"]

    def test_build_catalog_combines_triggers_into_pattern(self):
        import re
        from client.query_patterns import register_tool_meta, build_intent_catalog
        register_tool_meta("trig_tool_xyz", ["ai"], ["alpha trigger xyz", "beta trigger xyz"],
                           intent_category="trig_cat_xyz")
        catalog = build_intent_catalog()
        entry = next(e for e in catalog if e["name"] == "trig_cat_xyz")
        # Triggers are re.escape()'d so spaces become \\ in the pattern string
        # Verify both triggers match when the compiled pattern is applied
        assert entry["_compiled"].search("alpha trigger xyz")
        assert entry["_compiled"].search("beta trigger xyz")

    def test_invalidate_catalog_forces_rebuild(self):
        from client.query_patterns import register_tool_meta, invalidate_catalog, _get_catalog
        register_tool_meta("inv_tool_xyz", ["ai"], ["invalidate test xyz"],
                           intent_category="inv_cat_xyz")
        invalidate_catalog()
        catalog = _get_catalog()
        names = [e["name"] for e in catalog]
        assert "inv_cat_xyz" in names

    def test_classify_uses_dynamic_registration(self):
        from client.query_patterns import register_tool_meta, invalidate_catalog, classify
        register_tool_meta(
            tool_name="dynamic_test_tool_xyz",
            tags=["search"],
            triggers=["xyzzy dynamic search phrase"],
            intent_category="dynamic_test_cat_xyz",
        )
        invalidate_catalog()
        intent = classify("xyzzy dynamic search phrase please")
        assert intent.category == "dynamic_test_cat_xyz"
        assert "dynamic_test_tool_xyz" in intent.tools


# ═══════════════════════════════════════════════════════════════════
# ROUTER_* compiled patterns
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestRouterPatterns:
    def test_router_ingest_command_matches(self):
        from client.query_patterns import ROUTER_INGEST_COMMAND
        assert ROUTER_INGEST_COMMAND.search("ingest now")
        assert ROUTER_INGEST_COMMAND.search("ingest movies")
        assert ROUTER_INGEST_COMMAND.search("start ingesting")
        assert ROUTER_INGEST_COMMAND.search("add to rag")

    def test_router_ingest_command_no_match(self):
        from client.query_patterns import ROUTER_INGEST_COMMAND
        assert not ROUTER_INGEST_COMMAND.search("what's the weather?")
        assert not ROUTER_INGEST_COMMAND.search("search for movies")

    def test_router_status_query_matches(self):
        from client.query_patterns import ROUTER_STATUS_QUERY
        assert ROUTER_STATUS_QUERY.search("how many items have been ingested")
        assert ROUTER_STATUS_QUERY.search("show rag stats")
        assert ROUTER_STATUS_QUERY.search("what has been ingested")

    def test_router_multi_step_matches(self):
        from client.query_patterns import ROUTER_MULTI_STEP
        assert ROUTER_MULTI_STEP.search("search and then summarize")
        assert ROUTER_MULTI_STEP.search("first find the file")
        assert ROUTER_MULTI_STEP.search("research then analyze")

    def test_router_one_time_ingest_matches(self):
        from client.query_patterns import ROUTER_ONE_TIME_INGEST
        assert ROUTER_ONE_TIME_INGEST.search("then stop")
        assert ROUTER_ONE_TIME_INGEST.search("don't continue")

    def test_router_explicit_rag_matches(self):
        from client.query_patterns import ROUTER_EXPLICIT_RAG
        assert ROUTER_EXPLICIT_RAG.search("using rag")
        assert ROUTER_EXPLICIT_RAG.search("search rag for my notes")
        assert ROUTER_EXPLICIT_RAG.search("query rag")

    def test_router_knowledge_query_matches(self):
        from client.query_patterns import ROUTER_KNOWLEDGE_QUERY
        assert ROUTER_KNOWLEDGE_QUERY.search("what is quantum computing")
        assert ROUTER_KNOWLEDGE_QUERY.search("who is Alan Turing")
        assert ROUTER_KNOWLEDGE_QUERY.search("explain photosynthesis")
        assert ROUTER_KNOWLEDGE_QUERY.search("tell me about black holes")

    def test_router_exclude_media_matches(self):
        from client.query_patterns import ROUTER_EXCLUDE_MEDIA
        assert ROUTER_EXCLUDE_MEDIA.search("find a movie")
        assert ROUTER_EXCLUDE_MEDIA.search("search Plex")


# ═══════════════════════════════════════════════════════════════════
# extract_research_sources
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestExtractResearchSources:
    def test_using_as_source(self):
        from client.query_patterns import extract_research_sources
        result = extract_research_sources("summarize this using wikipedia.org as source")
        assert any("wikipedia.org" in s for s in result)

    def test_using_multiple_sources(self):
        from client.query_patterns import extract_research_sources
        result = extract_research_sources("using bbc.com and reuters.com as sources")
        assert len(result) >= 2

    def test_based_on_source(self):
        from client.query_patterns import extract_research_sources
        result = extract_research_sources("based on https://example.com write a summary")
        assert any("example.com" in s for s in result)

    def test_url_extraction(self):
        from client.query_patterns import extract_research_sources
        result = extract_research_sources("check https://docs.python.org/3/ for examples")
        assert any("docs.python.org" in s for s in result)

    def test_image_urls_excluded(self):
        from client.query_patterns import extract_research_sources
        result = extract_research_sources("see https://example.com/api/v1/thumbnails/123")
        assert not any("thumbnails" in s for s in result)

    def test_empty_string(self):
        from client.query_patterns import extract_research_sources
        result = extract_research_sources("")
        assert result == []

    def test_no_sources(self):
        from client.query_patterns import extract_research_sources
        result = extract_research_sources("what is the weather today?")
        assert result == []


# ═══════════════════════════════════════════════════════════════════
# WEB_SEARCH_EXPLICIT_PATTERN & OLLAMA_SEARCH_PATTERN
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestSearchPatterns:
    def test_web_search_explicit_matches(self):
        from client.query_patterns import WEB_SEARCH_EXPLICIT_PATTERN
        assert WEB_SEARCH_EXPLICIT_PATTERN.search("use web search to find")
        assert WEB_SEARCH_EXPLICIT_PATTERN.search("using web search for")
        assert WEB_SEARCH_EXPLICIT_PATTERN.search("web search for news")
        assert WEB_SEARCH_EXPLICIT_PATTERN.search("use web_search_tool")

    def test_web_search_explicit_no_match(self):
        from client.query_patterns import WEB_SEARCH_EXPLICIT_PATTERN
        assert not WEB_SEARCH_EXPLICIT_PATTERN.search("what is the weather")

    def test_ollama_search_pattern_matches(self):
        from client.query_patterns import OLLAMA_SEARCH_PATTERN
        assert OLLAMA_SEARCH_PATTERN.search("ollama search for news")
        assert OLLAMA_SEARCH_PATTERN.search("web search using ollama")

    def test_ollama_search_no_match(self):
        from client.query_patterns import OLLAMA_SEARCH_PATTERN
        assert not OLLAMA_SEARCH_PATTERN.search("search plex")