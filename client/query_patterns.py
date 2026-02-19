"""
Shared Query Patterns for MCP Client
Used by both LangGraph routing and Skills injection

Understands whether to inject skills or not.
This happens before LangGraph as a system prompt modification.
Returns a boolean.
"""

import re

# ═══════════════════════════════════════════════════════════════════
# TOOL-RELATED PATTERNS (queries that need MCP tools)
# ═══════════════════════════════════════════════════════════════════

# Location queries
PATTERN_LOCATION = re.compile(
    r'\b(my|what\'?s?\s+my)\s+location\b'
    r'|\bwhere\s+am\s+i\b',
    re.IGNORECASE
)

# Weather queries
PATTERN_WEATHER = re.compile(
    r'\bweather\b'
    r'|\btemperature\b'
    r'|\bforecast\b',
    re.IGNORECASE
)

# Time queries
PATTERN_TIME = re.compile(
    r'\bwhat\s+time\b'
    r'|\bcurrent\s+time\b',
    re.IGNORECASE
)

# Plex library searches
PATTERN_PLEX_SEARCH = re.compile(
    r'\b(find|search)\s+.*(plex|library|my\s+library)\b'
    r'|\b(plex|library|my\s+library)\s+.*(find|search)\b',
    re.IGNORECASE
)

# System info queries
PATTERN_SYSTEM = re.compile(
    r'\bsystem\s+info\b'
    r'|\bhardware\b'
    r'|\b(cpu|gpu|ram)\b'
    r'|\bspecs?\b'
    r'|\bprocesses?\b',
    re.IGNORECASE
)

# Code-related queries
PATTERN_CODE = re.compile(
    r'\bcode\b'
    r'|\bscan\s+code\b'
    r'|\bdebug\b'
    r'|\breview\s+code\b'
    r'|\bsummarize\s+code\b',
    re.IGNORECASE
)

# Text summarization (not code)
PATTERN_TEXT = re.compile(
    r'\b(summarize|summary|explain)\b',
    re.IGNORECASE
)

# RAG Status (read-only) - MUST CHECK BEFORE INGEST
PATTERN_RAG_STATUS = re.compile(
    r'\bwhat\'?s?\s+(in\s+)?(my\s+)?rag\b'
    r'|\b(show|list|display|check)\s+(the\s+)?(my\s+)?rag\b'
    r'|\brag\s+(status|contents?|info)\b'
    r'|\bhow\s+many\s+(items?|plex|movies?|documents?)\s+.*(ingested|in\s+rag)\b'
    r'|\bcount\s+(items?|movies?|documents?)\s+(in\s+)?rag\b'
    r'|\btotal\s+(items?|plex|movies?|documents?)\s+(in\s+)?rag\b'
    r'|\bwhat\s+(has|was)\s+been\s+ingested\b'
    r'|\bitems?\s+(have\s+been|were)\s+ingested\b',
    re.IGNORECASE
)

# Ingestion commands (action verbs, not past tense)
PATTERN_INGEST = re.compile(
    r'\bingest\s+(now|movies?|items?|\d+|batch)\b'
    r'|\bstart\s+ingesting\b'
    r'|\badd\s+to\s+(rag|knowledge)\b'
    r'|\bprocess\s+subtitles?\b'
    r'|\bextract\s+subtitles?\b',
    re.IGNORECASE
)

# A2A/remote agent queries
PATTERN_A2A = re.compile(
    r'\ba2a\b'
    r'|\bremote\s+(agent|tools?)\b'
    r'|\bdiscover\s+(agent|tools?)\b',
    re.IGNORECASE
)

# Todo/Task management
PATTERN_TODO = re.compile(
    r'\btodo\b'
    r'|\btask\b'
    r'|\bremind\s+me\b'
    r'|\badd\s+to\s+my\s+todo\b'
    r'|\blist\s+my\s+todos?\b'
    r'|\bshow\s+my\s+tasks?\b',
    re.IGNORECASE
)

# Knowledge base / Notes
PATTERN_KNOWLEDGE = re.compile(
    r'\bremember\b'
    r'|\bsave\s+this\b'
    r'|\bmake\s+a\s+note\b'
    r'|\bknowledge\s+base\b'
    r'|\bsearch\s+my\s+notes?\b'
    r'|\badd\s+entry\b',
    re.IGNORECASE
)

# ═══════════════════════════════════════════════════════════════════
# ROUTER-SPECIFIC PATTERNS (for LangGraph routing logic)
# ═══════════════════════════════════════════════════════════════════

ROUTER_INGEST_COMMAND = re.compile(
    r'\bingest\s+(now|movies?|items?|\d+|batch)\b'
    r'|\bstart\s+ingesting\b'
    r'|\badd\s+to\s+(rag|knowledge)\b'
    r'|\bprocess\s+subtitles?\b',
    re.IGNORECASE
)

ROUTER_STATUS_QUERY = re.compile(
    r'\bhow\s+many\s+.*(ingested|in\s+rag)\b'
    r'|\bwhat\s+(has|was)\s+been\s+ingested\b'
    r'|\bitems?\s+(have\s+been|were)\s+ingested\b'
    r'|\bcount\s+.*(items?|in\s+rag)\b'
    r'|\btotal\s+.*(items?|in\s+rag)\b'
    r'|\b(show|list|display)\s+rag\b',
    re.IGNORECASE
)

ROUTER_MULTI_STEP = re.compile(
    r'\s+and\s+then\s+'
    r'|\s+then\s+'
    r'|\s+after\s+that\s+'
    r'|\s+next\s+'
    r'|\bfirst\b'
    r'|\bresearch.*analyze\b'
    r'|\bfind.*summarize\b',
    re.IGNORECASE
)

ROUTER_ONE_TIME_INGEST = re.compile(
    r'\bstop\b'
    r'|\bthen\s+stop\b'
    r'|\bdon\'?t\s+continue\b'
    r'|\bdon\'?t\s+go\s+on\b',
    re.IGNORECASE
)

ROUTER_EXPLICIT_RAG = re.compile(
    r'\busing\s+rag\b'
    r'|\buse\s+rag\b'
    r'|\brag\s+tool\b'
    r'|\bwith\s+rag\b'
    r'|\bsearch\s+rag\b'
    r'|\bquery\s+rag\b',
    re.IGNORECASE
)

ROUTER_KNOWLEDGE_QUERY = re.compile(
    r'\bwhat\s+is\b'
    r'|\bwho\s+is\b'
    r'|\bexplain\b'
    r'|\btell\s+me\s+about\b',
    re.IGNORECASE
)

ROUTER_EXCLUDE_MEDIA = re.compile(
    r'\bmovie\b'
    r'|\bplex\b'
    r'|\bsearch\b'
    r'|\bfind\b'
    r'|\bshow\b'
    r'|\bmedia\b',
    re.IGNORECASE
)

# ═══════════════════════════════════════════════════════════════════
# GENERAL KNOWLEDGE PATTERNS (queries that DON'T need tools)
# ═══════════════════════════════════════════════════════════════════

# General knowledge questions
PATTERN_GENERAL_KNOWLEDGE = re.compile(
    r'\bwho\s+is\b'
    r'|\bwhat\s+is\b'
    r'|\bwhere\s+is\b'
    r'|\bwhen\s+is\b'
    r'|\bwhy\s+is\b'
    r'|\bhow\s+is\b'
    r'|\bexplain\b'
    r'|\btell\s+me\s+about\b'
    r'|\bdescribe\b'
    r'|\bdefine\b',
    re.IGNORECASE
)

# Current events
PATTERN_CURRENT_EVENTS = re.compile(
    r'\bcurrent\b'
    r'|\blatest\b'
    r'|\brecent\b'
    r'|\btoday\b'
    r'|\bnow\b'
    r'|\bthis\s+week\b'
    r'|\bwho\'?s\s+the\s+current\b'
    r'|\bwhat\'?s\s+the\s+current\b'
    r'|\bwho\s+is\s+the\s+current\b',
    re.IGNORECASE
)

# ═══════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════

# All tool-related patterns grouped together
TOOL_PATTERNS = [
    PATTERN_LOCATION,
    PATTERN_WEATHER,
    PATTERN_TIME,
    PATTERN_PLEX_SEARCH,
    PATTERN_SYSTEM,
    PATTERN_CODE,
    PATTERN_TEXT,
    PATTERN_RAG_STATUS,
    PATTERN_INGEST,
    PATTERN_A2A,
    PATTERN_TODO,
    PATTERN_KNOWLEDGE
]


def needs_tools(query: str) -> bool:
    """
    Check if a query needs MCP tools based on pattern matching.

    Returns:
        True if query matches any tool-related pattern
        False if it's a general knowledge question OR current events query
    """
    query_lower = query.lower()

    # Current events should use web search, not MCP tools
    if PATTERN_CURRENT_EVENTS.search(query_lower):
        return False

    # Check if it matches any tool pattern
    for pattern in TOOL_PATTERNS:
        if pattern.search(query_lower):
            return True

    # Check for explicit "my" references (personal data)
    if re.search(r'\bmy\b', query_lower):
        return True

    return False


def is_general_knowledge(query: str) -> bool:
    """
    Check if a query is a general knowledge question that doesn't need tools.

    Returns:
        True if it's a general knowledge question
        False otherwise
    """
    query_lower = query.lower()

    # Check if it matches general knowledge pattern
    is_general = PATTERN_GENERAL_KNOWLEDGE.search(query_lower) is not None

    # But NOT if it also needs tools
    needs_mcp_tools = needs_tools(query)

    return is_general and not needs_mcp_tools