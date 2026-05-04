import pytest
import re
from client.query_patterns import INTENT_CATALOG


def _find_entry(name: str) -> dict:
    """Helper to find a catalog entry by name."""
    for entry in INTENT_CATALOG:
        if entry.get("name") == name or entry.get("intent") == name:
            return entry
    return {}


@pytest.mark.unit
class TestIntentPatterns:
    def test_weather_pattern(self):
        """Test weather intent detection"""
        entry = _find_entry("weather")
        assert entry, "weather entry not found in INTENT_CATALOG"
        pattern = entry["pattern"]

        assert re.search(pattern, "what's the weather", re.IGNORECASE)
        assert re.search(pattern, "temperature today", re.IGNORECASE)
        assert re.search(pattern, "forecast for tomorrow", re.IGNORECASE)

        # Should not match
        assert not re.search(pattern, "hello", re.IGNORECASE)

    def test_code_assistant_pattern(self):
        """Test code assistant intent detection"""
        entry = _find_entry("code_assistant")
        assert entry, "code_assistant entry not found in INTENT_CATALOG"
        pattern = entry["pattern"]

        assert re.search(pattern, "analyze project", re.IGNORECASE)
        assert re.search(pattern, "what dependencies", re.IGNORECASE)

        # Should not match generic questions
        assert not re.search(pattern, "hello world", re.IGNORECASE)

    def test_plex_search_priority(self):
        """Test that plex_search has higher priority than ml_recommendation"""
        plex = _find_entry("plex_search")
        ml = _find_entry("ml_recommendation")

        assert plex, "plex_search entry not found in INTENT_CATALOG"
        assert ml, "ml_recommendation entry not found in INTENT_CATALOG"

        assert plex["priority"] < ml["priority"]  # Lower number = higher priority

    def test_pattern_priority_ordering(self):
        """Test that patterns have valid priorities"""
        priorities = [e["priority"] for e in INTENT_CATALOG if "priority" in e]

        assert len(priorities) > 0
        assert min(priorities) >= 1

    def test_all_entries_have_required_fields(self):
        """Test that all catalog entries have required fields"""
        for entry in INTENT_CATALOG:
            assert "pattern" in entry, f"Entry missing 'pattern': {entry}"
            assert "priority" in entry, f"Entry missing 'priority': {entry}"

    def test_catalog_is_not_empty(self):
        """Test that the catalog has entries"""
        assert len(INTENT_CATALOG) > 0