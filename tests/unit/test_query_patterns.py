"""
Tests for client/query_patterns.py
Covers: classify(), QueryIntent, needs_tools(), is_general_knowledge(),
        INTENT_CATALOG structure, conversational detection
"""
import pytest
from client.query_patterns import (
    classify, QueryIntent, needs_tools, is_general_knowledge,
    INTENT_CATALOG
)


# ═══════════════════════════════════════════════════════════════════
# QueryIntent dataclass
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestQueryIntent:
    def test_default_values(self):
        intent = QueryIntent(category="test")
        assert intent.tools == []
        assert intent.needs_web_search is False
        assert intent.needs_skills is False
        assert intent.is_conversational is False
        assert intent.priority == 3

    def test_custom_values(self):
        intent = QueryIntent(
            category="weather",
            tools=["get_weather"],
            needs_web_search=True,
            is_conversational=False,
            priority=1
        )
        assert intent.category == "weather"
        assert "get_weather" in intent.tools
        assert intent.needs_web_search is True
        assert intent.priority == 1


# ═══════════════════════════════════════════════════════════════════
# classify() — conversational detection
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestClassifyConversational:
    def test_yes_is_conversational(self):
        intent = classify("yes")
        assert intent.is_conversational is True
        assert intent.category == "conversational"

    def test_thanks_is_conversational(self):
        intent = classify("thanks")
        assert intent.is_conversational is True

    def test_i_like_is_conversational(self):
        intent = classify("I like that idea")
        assert intent.is_conversational is True

    def test_write_poem_is_conversational(self):
        intent = classify("write me a poem about rain")
        assert intent.is_conversational is True

    def test_conversational_has_no_tools(self):
        intent = classify("ok thanks")
        assert intent.tools == []
        assert intent.needs_web_search is False


# ═══════════════════════════════════════════════════════════════════
# classify() — intent matching
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestClassifyIntentMatching:
    def test_weather_query_matches(self):
        intent = classify("what's the weather today?")
        assert intent.category == "weather"
        assert intent.is_conversational is False

    def test_weather_has_tools(self):
        intent = classify("what's the temperature outside?")
        assert len(intent.tools) > 0

    def test_plex_query_matches(self):
        intent = classify("find me a movie on Plex")
        assert intent.category in ("plex_search", "ml_recommendation", "plex")
        assert intent.is_conversational is False

    def test_code_query_matches(self):
        intent = classify("analyze my project dependencies")
        assert intent.category == "code_assistant"
        assert len(intent.tools) > 0

    def test_general_query_no_match(self):
        intent = classify("what is the meaning of life?")
        assert intent.category == "general"
        assert intent.tools == []
        assert intent.is_conversational is False

    def test_general_query_no_web_search(self):
        # Use a query that doesn't match any specific category
        intent = classify("who was the first roman emperor?")
        assert intent.category == "general"
        assert intent.needs_web_search is False


# ═══════════════════════════════════════════════════════════════════
# classify() — explicit tool name detection
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestClassifyExplicitTool:
    def test_explicit_tool_name_detected(self):
        intent = classify(
            "use get_weather_tool to check Surrey",
            available_tool_names=["get_weather_tool", "search_tool"]
        )
        assert intent.category == "explicit_tool"
        assert "get_weather_tool" in intent.tools

    def test_no_tool_names_skips_explicit(self):
        intent = classify("use get_weather_tool please")
        # Without available_tool_names, should not be explicit_tool
        assert intent.category != "explicit_tool"

    def test_explicit_tool_not_needs_web_search(self):
        intent = classify(
            "shashin_search_tool find photos",
            available_tool_names=["shashin_search_tool"]
        )
        assert intent.needs_web_search is False


# ═══════════════════════════════════════════════════════════════════
# Legacy shims
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestLegacyShims:
    def test_needs_tools_true_for_weather(self):
        assert needs_tools("what's the weather?") is True

    def test_needs_tools_false_for_conversational(self):
        assert needs_tools("yes thanks") is False

    def test_needs_tools_false_for_general(self):
        assert needs_tools("who was the first roman emperor?") is False

    def test_is_general_knowledge_true(self):
        assert is_general_knowledge("what is the speed of light?") is True

    def test_is_general_knowledge_false_for_weather(self):
        assert is_general_knowledge("what's the weather?") is False

    def test_is_general_knowledge_false_for_conversational(self):
        assert is_general_knowledge("yes") is False


# ═══════════════════════════════════════════════════════════════════
# INTENT_CATALOG structure validation
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestIntentCatalogStructure:
    def test_catalog_not_empty(self):
        assert len(INTENT_CATALOG) > 0

    def test_all_entries_have_required_fields(self):
        for entry in INTENT_CATALOG:
            assert "name" in entry, f"Missing 'name': {entry}"
            assert "pattern" in entry, f"Missing 'pattern': {entry}"
            assert "tools" in entry, f"Missing 'tools': {entry}"
            assert "priority" in entry, f"Missing 'priority': {entry}"
            assert "web_search" in entry, f"Missing 'web_search': {entry}"
            assert "skills" in entry, f"Missing 'skills': {entry}"

    def test_tools_are_lists(self):
        for entry in INTENT_CATALOG:
            assert isinstance(entry["tools"], list), \
                f"tools must be a list in entry '{entry['name']}'"

    def test_priorities_are_integers(self):
        for entry in INTENT_CATALOG:
            assert isinstance(entry["priority"], int), \
                f"priority must be int in entry '{entry['name']}'"

    def test_web_search_is_bool(self):
        for entry in INTENT_CATALOG:
            assert isinstance(entry["web_search"], bool), \
                f"web_search must be bool in entry '{entry['name']}'"

    def test_no_duplicate_names(self):
        names = [e["name"] for e in INTENT_CATALOG]
        assert len(names) == len(set(names)), "Duplicate entry names found"

    def test_priority_range_valid(self):
        for entry in INTENT_CATALOG:
            assert entry["priority"] >= 1, \
                f"Priority must be >= 1 in entry '{entry['name']}'"

    def test_plex_priority_lower_than_ml(self):
        plex = next((e for e in INTENT_CATALOG if e["name"] == "plex_search"), None)
        ml = next((e for e in INTENT_CATALOG if e["name"] == "ml_recommendation"), None)
        if plex and ml:
            assert plex["priority"] < ml["priority"]

    def test_patterns_compile_without_error(self):
        import re
        for entry in INTENT_CATALOG:
            try:
                re.compile(entry["pattern"], re.IGNORECASE)
            except re.error as e:
                pytest.fail(f"Pattern in '{entry['name']}' failed to compile: {e}")