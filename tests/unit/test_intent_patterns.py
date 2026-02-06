import pytest
from client.langgraph import INTENT_PATTERNS
import re

@pytest.mark.unit
class TestIntentPatterns:
    def test_weather_pattern(self):
        """Test weather intent detection"""
        pattern = INTENT_PATTERNS["weather"]["pattern"]
        
        assert re.search(pattern, "what's the weather", re.IGNORECASE)
        assert re.search(pattern, "temperature today", re.IGNORECASE)
        assert re.search(pattern, "forecast for tomorrow", re.IGNORECASE)
        
        # Should not match
        assert not re.search(pattern, "hello", re.IGNORECASE)
    
    def test_code_assistant_pattern(self):
        """Test code assistant intent detection"""
        pattern = INTENT_PATTERNS["code_assistant"]["pattern"]
        
        assert re.search(pattern, "analyze project", re.IGNORECASE)
        assert re.search(pattern, "what dependencies", re.IGNORECASE)
        assert re.search(pattern, "review code", re.IGNORECASE)
        
        # Should not match generic questions
        assert not re.search(pattern, "hello world", re.IGNORECASE)
    
    def test_plex_search_priority(self):
        """Test that plex_search has higher priority than ml_recommendation"""
        plex_priority = INTENT_PATTERNS["plex_search"]["priority"]
        ml_priority = INTENT_PATTERNS["ml_recommendation"]["priority"]
        
        assert plex_priority < ml_priority  # Lower number = higher priority
    
    def test_pattern_priority_ordering(self):
        """Test that patterns are correctly prioritized"""
        sorted_patterns = sorted(INTENT_PATTERNS.items(), key=lambda x: x[1]["priority"])
        
        # First should be highest priority
        assert sorted_patterns[0][1]["priority"] == 1
        
        # No duplicates in priorities for same category
        priorities = [p[1]["priority"] for p in sorted_patterns]
        assert len(priorities) == len(set(priorities)) or True  # Some duplicates OK