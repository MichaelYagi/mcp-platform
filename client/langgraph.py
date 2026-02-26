"""
LangGraph Module with Centralized Pattern Configuration
Handles LangGraph agent creation, routing, and execution

Understands which tool the LLM should use.
Returns a filtered tool list
"""
import asyncio
import json
import logging
import operator
import re
import time
from typing import TypedDict, Annotated, Sequence
import requests
import urllib.parse
from html.parser import HTMLParser
from concurrent.futures import ThreadPoolExecutor

from tools.rag.rag_vector_db import should_refresh_source
from .stop_signal import is_stop_requested, clear_stop
from .search_client import get_search_client
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Import only router patterns (for router() function)
from client.query_patterns import (
    ROUTER_INGEST_COMMAND, ROUTER_STATUS_QUERY, ROUTER_MULTI_STEP,
    ROUTER_ONE_TIME_INGEST, ROUTER_EXPLICIT_RAG, ROUTER_KNOWLEDGE_QUERY,
    ROUTER_EXCLUDE_MEDIA
)

# Try to import metrics
try:
    from metrics import metrics
    METRICS_AVAILABLE = True
except ImportError:
    try:
        from client.metrics import metrics
        METRICS_AVAILABLE = True
    except ImportError:
        METRICS_AVAILABLE = False
        from collections import defaultdict
        metrics = {
            "agent_runs": 0,
            "agent_errors": 0,
            "agent_times": [],
            "llm_calls": 0,
            "llm_errors": 0,
            "llm_times": [],
            "tool_calls": defaultdict(int),
            "tool_errors": defaultdict(int),
            "tool_times": defaultdict(list),
        }

# ═══════════════════════════════════════════════════════════════════
# CENTRALIZED PATTERN CONFIGURATION
# Add new intents here - no code changes needed!
# ═══════════════════════════════════════════════════════════════════

INTENT_PATTERNS = {
    "ingest": {
        "pattern": (
            r'\bingest\b'
            r'|\bprocess\b'
            r'|\badd\s+to\s+rag\b'
            r'|\bindex\b'
            r'|\bvectorize\b'
            r'|\bembed\s+(this|it)\b'
            r'|\bupdate\s+rag\b'
            r'|\brefresh\s+rag\b'
            r'|\bscan\s+plex\b'
            r'|\bprocess\s+next\b'
            r'|\bingest\s+next\b'
            r'|\badd\s+movie\b'
            r'|\badd\s+media\b'
        ),
        "exclude_pattern": (
            r'\bhow\s+many\b'
            r'|\bwhat\s+(has|was)\b'
            r'|\bcount\b'
            r'|\btotal\b'
            r'|\bstatus\b'
            r'|\bwhat(\'s| is)\s+in\s+rag\b'
        ),
        "tools": [
            "plex_ingest_*",
            "plex_find_unprocessed",
            "plex_ingest_single",
            "plex_ingest_batch",
            "rag_add_tool"
        ],
        "priority": 2
    },
    "github_review": {
        "pattern": (
            # GitHub URLs
            r'\\bgithub\\.com/[^\\s]+'
            r'|\\breview.*github'
            r'|\\bclone.*github'
            r'|\\bcheck.*github\\.com'

            # Repository review
            r'|\\breview.*(repo|repository)'
            r'|\\banalyze.*(repo|repository)'
            r'|\\bcheck.*(repo|repository)'

            # Code from URL
            r'|\\breview.*https?://'
            r'|\\banalyze.*https?://'

            # Explicit GitHub references
            r'|\\bgithub\\s+(repo|repository|project)'
            r'|\\bfrom\\s+github'
        ),
        "tools": [
            "github_clone_repo",
            "github_list_files",
            "github_get_file_content",
            "github_cleanup_repo",
            "analyze_project",
            "analyze_code_file",
            "review_code",
            "scan_project_structure"
        ],
        "priority": 2
    },
    "code_assistant": {
        "pattern": (
            # Tech stack queries
            r'\btech\s+stack\b'
            r'|\btechnology\s+stack\b'
            r'|\bwhat.*tech\b'
            r'|\bwhat.*stack\b'
            r'|\bwhat.*technologies\b'
            r'|\bwhat.*languages\b'
            r'|\bwhat.*frameworks?\b'
            r'|\bwhat.*dependencies\b'
            r'|\bproject\s+structure\b'
            r'|\banalyze.*project\b'
            r'|\bscan.*project\b'
            r'|\bshow.*structure\b'
            r'|\blist.*dependencies\b'

            # Node.js/Package queries
            r'|\bnode\.?js\s+(packages?|dependencies|modules)\b'
            r'|\bnpm\s+(packages?|dependencies|modules)\b'
            r'|\bpackage\.json\b'
            r'|\bnode\s+(packages?|dependencies|modules)\b'
            r'|\b(about|more|explain).*node\.?js\s+(packages?|dependencies)\b'
            r'|\btell.*about.*(node|packages?|dependencies)\b'
            r'|\bmore.*about.*(node|packages?|dependencies)\b'

            # "go into depth" style questions
            r'|\bgo\s+into\s+(depth|detail)\b'
            r'|\bmore\s+detail.*about.*(packages?|dependencies|modules)\b'
            r'|\bin[\s-]?depth.*about.*(packages?|dependencies|modules)\b'
            r'|\belaborate.*on.*(packages?|dependencies|modules)\b'
            r'|\bexpand.*on.*(packages?|dependencies|modules)\b'

            # Generic package questions
            r'|\bwhat.*(do|are).*(packages?|dependencies|modules)\b'
            r'|\bexplain.*(packages?|dependencies|modules)\b'
            r'|\bwhat.*do\s+they\s+do\b'
            r'|\bwhat.*are\s+(they|those)\s+(for|used\s+for)\b'
            r'|\bwhat.*they.*used\s+for\b'

            # Code analysis
            r'|\banalyze.*code\b'
            r'|\bcheck.*code\b'
            r'|\breview.*code\b'
            r'|\blint\b'

            # File extensions
            r'|\banalyze.*\.(py|js|jsx|ts|tsx|rs|go|java|kt)\b'
            r'|\bcheck.*\.(py|js|jsx|ts|tsx|rs|go|java|kt)\b'

            # Bug fixing
            r'|\bfix.*bug\b'
            r'|\bfix.*error\b'
            r'|\bfix.*issue\b'
            r'|\bfix.*code\b'

            # Code generation
            r'|\bgenerate.*code\b'
            r'|\bcreate.*(function|class|module|component)\b'
            r'|\bwrite.*(function|class)\b'
        ),
        "tools": [
            "analyze_project",
            "get_project_dependencies",
            "scan_project_structure",
            "analyze_code_file",
            "fix_code_file",
            "suggest_improvements",
            "explain_code",
            "generate_tests",
            "refactor_code",
            "generate_code"
        ],
        "priority": 2
    },
    "location": {
        "pattern": (
            r'\b(my|what\'?s?\s+my)\s+location\b'
            r'|\bwhere\s+am\s+i\b'
            r'|\bcurrent\s+location\b'
            r'|\bwhere\s+do\s+i\s+live\b'
        ),
        "tools": ["get_location_tool"],
        "priority": 3
    },
    "weather": {
        "pattern": (
            r'\bweather\b'
            r'|\btemperature\b'
            r'|\bforecast\b'
            r'|\brain\b'
            r'|\bsnow\b'
            r'|\bwind\b'
            r'|\bconditions\b'
        ),
        "tools": ["get_location_tool", "get_weather_tool"],
        "priority": 3
    },
    "time": {
        "pattern": (
            r'\bwhat\s+time\b'
            r'|\bwhat\s+date\b'
            r'|\bcurrent\s+time\b'
            r'|\bcurrent\s+date\b'
            r'|\btime\s+now\b'
            r'|\btime\s+is\s+it\b'
        ),
        "tools": ["get_time_tool"],
        "priority": 3
    },
    # ═══════════════════════════════════════════════════════════════
    # CRITICAL FIX: plex_search BEFORE ml_recommendation (priority 2 vs 3)
    # ═══════════════════════════════════════════════════════════════
    "plex_search": {
        "pattern": (
            # Direct search phrases
            r'\b(find|search|look\s+for|show\s+me)\s+.*\b(movie|film|show|media|series)\b'

            # Plot-based searches (CRITICAL for "where hero wins")
            r'|\bmovies?\s+(about|where|with|featuring|in\s+which)\b'
            r'|\bfilms?\s+(about|where|with|featuring|in\s+which)\b'

            # "where X happens" pattern
            r'|\bwhere\s+.*\s+(wins?|loses?|dies|survives|happens|occurs|escapes)\b'

            # Library references
            r'|\bsearch\s+(plex|library|my\s+library|my\s+movies)\b'
            r'|\bfind\s+.*\s+in\s+(plex|library|my\s+library)\b'

            # Scene searches
            r'|\bscene\s+(where|with|from)\b'
            r'|\bfind\s+scene\b'
            r'|\blocate\s+scene\b'

            # Browse/explore (not recommendations)
            r'|\bbrowse\s+my\b'
            r'|\blist\s+.*\s+(movies|films|shows)\b'
        ),
        "tools": [
            "rag_search_tool",
            "semantic_media_search_text",
            "scene_locator_tool",
            "find_scene_by_title"
        ],
        "priority": 2  # HIGHER priority than ml_recommendation!
    },
    "rag": {
        "pattern": (
            # Status queries
            r'\bhow\s+many\s+.*(ingested|in\s+rag)\b'
            r'|\bwhat\s+(has|was)\s+been\s+ingested\b'
            r'|\bitems?\s+(have\s+been|were)\s+ingested\b'
            r'|\bcount\s+.*(items?|in\s+rag)\b'
            r'|\btotal\s+.*(items?|in\s+rag)\b'
            r'|\b(show|list|display)\s+rag\b'
            r'|\brag\s+(status|contents?|info|summary|overview|report|stats)\b'
            r'|\bwhat(\'s| is)\s+in\s+(the\s+)?rag\b'
            r'|\bgive\s+me\s+rag\s+(stats|status|info|details)\b'

            # Search queries
            r'|\bsearch\s+(the\s+)?rag\b'
            r'|\bfind\s+in\s+rag\b'
            r'|\blook\s+up\s+in\s+rag\b'
            r'|\brag\s+search\b'
            r'|\bquery\s+(the\s+)?rag\b'
            r'|\bdo\s+you\s+have\s+.*\s+in\s+rag\b'
            r'|\btell\s+me\s+(about|more\s+about)\b'
            r'|\bwhat\s+do\s+you\s+have\s+(about|on)\b'

            # Browse/list queries
            r'|\bbrowse\s+(the\s+)?rag\b'
            r'|\bshow\s+rag\s+(content|documents|entries|sources)\b'
            r'|\blist\s+rag\s+(sources|documents|content)\b'
            r'|\bwhat\s+sources\s+.*(in\s+)?rag\b'
            r'|\brag_list_sources\b'
            r'|\brag_browse\b'

            # General RAG references
            r'|\btell\s+me\s+(about|more\s+about)\b'
            r'|\brag\s+(database|storage|data)\b'
            r'|\btell\s+me\s+(about|more\s+about)\b'
            r'|\bwhat\s+do\s+you\s+know\s+about\b'
            r'|\bwhat\s+is\s+.+\s+in\s+my\s+(rag|knowledge|database)\b'
        ),
        "tools": [
            "rag_search_tool",
            "rag_status_tool",
            "rag_list_sources_tool",
            "rag_browse_tool",
            "rag_diagnose_tool",
            "rag_add_tool"
        ],
        "priority": 2
    },
    "ml_recommendation": {
        "pattern": (
            # Explicit recommendation requests
            r'\brecommend(ation)?s?\b'
            r'|\bsuggest(ion)?s?\b'

            # ML/training specific
            r'|\bml\s+(model|train|recommendation)\b'
            r'|\btrain\s+(model|recommender|recommendation)\b'
            r'|\bauto.?train\b'

            # History management
            r'|\bimport\s+.*\s*history\b'
            r'|\bviewing\s+history\b'
            r'|\bwatch\s+history\b'
            r'|\brecord\s+(viewing|that\s+i\s+watched)\b'

            # Personalized suggestions
            r'|\bwhat\s+should\s+i\s+watch\b'
            r'|\brank\s+(these|movies|shows)\b'
            r'|\bmy\s+best\s+unwatched\b'
            r'|\bunwatched\s+(recommendations|suggestions)\b'

            # Stats/model info
            r'|\brecommender\s+stats\b'
        ),
        "tools": [
            "record_viewing",
            "train_recommender",
            "recommend_content",
            "get_recommender_stats",
            "import_plex_history",
            "auto_train_from_plex",
            "reset_recommender",
            "auto_recommend_from_plex"
        ],
        "priority": 3  # LOWER priority - only matches if plex_search doesn't
    },
    "system": {
        "pattern": (
            r'\bsystem\s+info\b'
            r'|\bhardware\b'
            r'|\b(cpu|gpu|ram)\b'
            r'|\bspecs?\b'
            r'|\bprocesses?\b'
            r'|\bperformance\b'
            r'|\butilization\b'
            r'|\bmemory\s+usage\b'
        ),
        "tools": [
            "get_hardware_specs_tool",
            "get_system_info",
            "list_system_processes",
            "terminate_process"
        ],
        "priority": 3
    },
    "code": {
        "pattern": (
            r'\bcode\b'
            r'|\bscan\s+code\b'
            r'|\bdebug\b'
            r'|\breview\s+code\b'
            r'|\bsummarize\s+code\b'
            r'|\bfix\s+this\s+code\b'
            r'|\bexplain\s+this\s+code\b'
        ),
        "tools": [
            "review_code",
            "summarize_code_file",
            "search_code_in_directory",
            "scan_code_directory",
            "summarize_code",
            "debug_fix"
        ],
        "priority": 3
    },
    "text": {
        "pattern": (
            r'\b(summarize|summary|explain|simplify|break\s+down)\b'
        ),
        "exclude_pattern": r'\bcode\b',
        "tools": [
            "summarize_text_tool",
            "concept_contextualizer_tool",
            "summarize_direct_tool",
            "explain_simplified_tool",
            "concept_contextualizer_tool"
        ],
        "priority": 3
    },
    "todo": {
        "pattern": (
            r'\btodo\b'
            r'|\btask\b'
            r'|\bremind\s+me\b'
            r'|\bmy\s+todos?\b'
            r'|\bmy\s+tasks?\b'
            r'|\badd\s+to\s+my\s+list\b'
            r'|\btask\s+list\b'
        ),
        "tools": [
            "add_todo_item",
            "list_todo_items",
            "search_todo_items",
            "update_todo_item",
            "delete_todo_item",
            "delete_all_todo_items"
        ],
        "priority": 3
    },
    "knowledge": {
        "pattern": (
            r'\bremember\b'
            r'|\bsave\s+this\b'
            r'|\bmake\s+a\s+note\b'
            r'|\bknowledge\s+base\b'
            r'|\bsearch\s+my\s+notes?\b'
            r'|\badd\s+entry\b'
            r'|\bnote\s+this\b'
            r'|\bstore\s+this\b'
        ),
        "tools": [
            "add_entry",
            "list_entries",
            "get_entry",
            "search_entries",
            "search_by_tag",
            "search_semantic",
            "update_entry",
            "delete_entry"
        ],
        "priority": 3
    },
    "a2a": {
        "pattern": (
            r'\ba2a\b'
            r'|\bremote\s+(agent|tools?)\b'
            r'|\bdiscover\s+(agent|tools?)\b'
            r'|\bsend\s+to\s+remote\b'
            r'|\bcall\s+remote\s+tool\b'
            r'|\buse\s+remote\s+agent\b'
            r'|\bconnect\s+to\s+agent\b'
        ),
        "tools": ["send_a2a*", "discover_a2a"],
        "priority": 3
    },
    "trilium": {
        "pattern": (
            # Direct Trilium references
            r'\btrilium\b'
            r'|\bnotes?\s+(in\s+)?trilium\b'
            r'|\bmy\s+notes?\b'
            
            # Search in notes
            r'|\bsearch\s+(my\s+)?notes?\b'
            r'|\bfind\s+(in\s+)?(my\s+)?notes?\b'
            r'|\blook\s+up\s+(in\s+)?notes?\b'
            
            # Note management
            r'|\bcreate\s+(a\s+)?note\b'
            r'|\badd\s+(a\s+)?note\b'
            r'|\bupdate\s+(my\s+)?note\b'
            r'|\bdelete\s+(my\s+)?note\b'
            
            # Labels/tags
            r'|\bnotes?\s+tagged\b'
            r'|\bnotes?\s+with\s+label\b'
            r'|\badd\s+(label|tag)\s+to\s+note\b'
            
            # Recent/navigation
            r'|\brecent\s+notes?\b'
            r'|\blatest\s+notes?\b'
            r'|\bchild\s+notes?\b'
        ),
        "tools": [
            "search_notes",
            "search_by_label",
            "get_note_by_id",
            "create_note",
            "update_note_content",
            "update_note_title",
            "delete_note",
            "add_label_to_note",
            "get_note_labels",
            "get_note_children",
            "get_recent_notes"
        ],
        "priority": 2
    }
}


class AgentState(TypedDict):
    """State that gets passed between nodes in the graph"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    tools: dict
    llm: object
    ingest_completed: bool
    stopped: bool
    current_model: str
    research_source: str


# Research source detection
RESEARCH_SOURCE_PATTERN = re.compile(
    r'\busing\s+(?P<source>(?:https?://)?[\w\s\.\-/:]+?)\s+as\s+(a\s+)?source\b'
    r'|\bbased\s+on\s+(?P<source2>(?:https?://)?[\w\s\.\-/:]+?)(?:\s|,|$)'
    r'|\bfrom\s+(?P<source3>(?:https?://)?[\w\s\.\-/:]+?)\s+(?:find|search|get|tell)\b',
    re.IGNORECASE
)


def extract_research_sources(content: str) -> list:
    """
    Extract ALL sources from query (handles multiple sources).
    Returns list of sources, e.g., ['url1', 'url2', 'domain.com']
    """
    sources = []

    # Pattern 1: "using X and Y as sources" or "using X as a source"
    pattern1 = re.compile(
        r'\busing\s+(.+?)\s+as\s+(a\s+)?(source|sources)\b',
        re.IGNORECASE
    )
    match1 = pattern1.search(content)
    if match1:
        source_text = match1.group(1)
        parts = re.split(r'\s+and\s+|,\s*', source_text)
        sources.extend([p.strip().rstrip(',.;:!?') for p in parts if p.strip()])

    # Pattern 2: "based on X and Y"
    pattern2 = re.compile(
        r'\bbased\s+on\s+(.+?)(?:\s+write|\s+create|\s+explain|,|$)',
        re.IGNORECASE
    )
    match2 = pattern2.search(content)
    if match2:
        source_text = match2.group(1)
        parts = re.split(r'\s+and\s+|,\s*', source_text)
        sources.extend([p.strip().rstrip(',.;:!?') for p in parts if p.strip()])

    # Pattern 3: Find all URLs explicitly (backup)
    url_pattern = re.compile(r'https?://[^\s]+')
    urls = url_pattern.findall(content)

    # Clean URLs - remove trailing punctuation
    for url in urls:
        cleaned_url = url.rstrip(',.;:!?')
        if cleaned_url:
            sources.append(cleaned_url)

    # Deduplicate while preserving order
    unique_sources = []
    seen = set()
    for s in sources:
        s = s.strip()
        if s and s not in seen:
            unique_sources.append(s)
            seen.add(s)

    return unique_sources if unique_sources else None


# Direct source URLs for known sites
DIRECT_SOURCE_URLS = {
    "wikipedia.org": {
        "fallback_urls": []
    }
}


# HTML text extractor
class HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.skip_tags = {'script', 'style', 'nav', 'footer', 'header', 'aside', 'noscript'}
        self.current_tag = None
        self.title = None

    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        if tag == 'title' and self.title is None:
            self.title = ""

    def handle_endtag(self, tag):
        if tag == 'title' and self.title == "":
            self.title = None
        self.current_tag = None

    def handle_data(self, data):
        if self.current_tag in self.skip_tags:
            return
        if self.current_tag == 'title' and isinstance(self.title, str):
            self.title += data
            return
        text = data.strip()
        if text:
            self.text_parts.append(text)

    def get_text(self):
        text = '\n'.join(self.text_parts)
        return re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

    def get_title(self):
        return self.title.strip() if self.title else "Untitled"


def fetch_url_content_sync(url: str, timeout: int = 30) -> dict:
    try:
        from urllib.parse import urlparse, quote, urlunparse

        parsed = urlparse(url)
        encoded_path = quote(parsed.path, safe='/')
        encoded_query = quote(parsed.query, safe='&=') if parsed.query else ''
        encoded_fragment = quote(parsed.fragment, safe='') if parsed.fragment else ''

        encoded_url = urlunparse((
            parsed.scheme,
            parsed.netloc,
            encoded_path,
            parsed.params,
            encoded_query,
            encoded_fragment
        ))

        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(encoded_url, headers=headers, timeout=timeout)

        if response.status_code != 200:
            return {"success": False, "error": f"HTTP {response.status_code}"}

        parser = HTMLTextExtractor()
        parser.feed(response.text)
        text = parser.get_text()
        if len(text) > 10000:
            text = text[:10000] + "\n\n[Content truncated...]"
        if not text or len(text) < 50:
            return {"success": False, "error": "No content"}
        return {"success": True, "content": text, "title": parser.get_title(), "url": url}
    except Exception as e:
        return {"success": False, "error": str(e)}


async def fetch_url_content(url: str, timeout: int = 30) -> dict:
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=3) as executor:
        return await loop.run_in_executor(executor, fetch_url_content_sync, url, timeout)


async def fetch_from_source_directly(source: str, query: str) -> dict:
    source_lower = source.lower().replace("www.", "")
    for known_source, config in DIRECT_SOURCE_URLS.items():
        if known_source in source_lower:
            if "fallback_urls" in config and config["fallback_urls"]:
                return {"success": True, "urls": config["fallback_urls"][:3], "method": "direct"}
    return {"success": False, "method": "no_config"}


async def search_and_fetch_source(source: str, query: str, rag_add_tool=None) -> dict:
    """
    SMART HYBRID with URL support AND homepage detection:
    1. If source is a URL → Check if homepage
       a. If homepage → Auto-search site for relevant pages
       b. If specific page → Fetch that page
    2. If source is a domain → Try direct URLs
    3. If no direct URLs → Try Ollama Search
    4. Store all fetched content in RAG
    5. Always return something
    """
    logger = logging.getLogger("mcp_client")

    # ═══════════════════════════════════════════════════════════════
    # Helper function to store in RAG
    # ═══════════════════════════════════════════════════════════════
    async def store_in_rag(content: str, metadata: dict):
        """Store fetched content in RAG (with deduplication)"""
        if not rag_add_tool:
            logger.debug("ℹ️ RAG tool not provided, skipping storage")
            return False

        try:
            if not content:
                logger.warning("⚠️ Skipping RAG storage: empty content")
                return False

            source_url = metadata.get("url", "")
            if source_url:
                if not should_refresh_source(source_url, max_age_days=30):
                    logger.info(f"⏭️  Skipping recent content: {metadata.get('title')[:50]}")
                    return False

            rag_entry = {
                "text": str(content),
                "source": source_url,
                "metadata": {
                    "source_type": metadata.get("source_type", "unknown"),
                    "url": source_url,
                    "title": metadata.get("title", "Untitled"),
                    "domain": metadata.get("domain", ""),
                    "query": metadata.get("query", ""),
                    "fetch_method": metadata.get("fetch_method", "unknown"),
                    "timestamp": metadata.get("timestamp", time.time())
                }
            }

            await rag_add_tool.ainvoke(rag_entry)
            logger.debug(f"✅ Stored in RAG: {metadata.get('title', 'content')[:50]}")
            return True

        except Exception as e:
            logger.warning(f"⚠️ Failed to store in RAG: {e}")
            return False

    # ═══════════════════════════════════════════════════════════════
    # CHECK IF SOURCE IS A URL
    # ═══════════════════════════════════════════════════════════════
    if source.startswith(('http://', 'https://')):
        logger.info(f"🎯 Source is a URL: {source}")

        from urllib.parse import urlparse
        parsed = urlparse(source)
        domain = parsed.netloc
        path = parsed.path

        # ═══════════════════════════════════════════════════════════
        # DETECT HOMEPAGE
        # ═══════════════════════════════════════════════════════════
        is_homepage = (
                path in ['/', '', '/en/', '/en', '/index.html', '/index.php'] or
                path.count('/') <= 1
        )

        if is_homepage:
            logger.info(f"🏠 Homepage detected: {source}")
            logger.info(f"🔍 Auto-searching {domain}{path} for relevant content")

            # Clean query
            cleaned_query = query
            action_patterns = [
                r'create\s+(?:a\s+)?(?:\d+\s+)?(?:minute\s+)?(?:talk|essay|article)\s+(?:on|about)\s+',
                r'write\s+(?:a\s+)?(?:\d+\s+)?(?:page\s+)?(?:essay|article)\s+(?:on|about)\s+',
                r'tell\s+me\s+about\s+',
            ]
            for pattern in action_patterns:
                cleaned_query = re.sub(pattern, '', cleaned_query, flags=re.IGNORECASE)
            cleaned_query = cleaned_query.strip()

            search_query = f"site:{domain}{path} {cleaned_query}"
            logger.info(f"🔍 Site search: {search_query}")

            try:
                search = get_search_client()
                search_result = await search.search(search_query)

                if search_result.get("success"):
                    data = search_result.get("results", {})

                    if isinstance(data, dict):
                        web_pages = data.get("webPages", {})
                        value = web_pages.get("value", [])

                        summaries = []
                        for i, item in enumerate(value):
                            if isinstance(item, dict):
                                url = item.get("url", "")
                                title = item.get("name", "Untitled")
                                summary = item.get("summary", "")

                                if domain in url and summary:
                                    summaries.append({
                                        "url": url,
                                        "title": title,
                                        "summary": summary
                                    })
                                    logger.info(f"   ✅ Added: {title}")

                        if summaries:
                            logger.info(f"✅ Site search found {len(summaries)} pages with content on {domain}")

                            top_summaries = summaries[:5]

                            # ═══════════════════════════════════════════════════
                            # STORE ALL SUMMARIES IN RAG CONCURRENTLY
                            # ═══════════════════════════════════════════════════
                            if rag_add_tool:
                                storage_tasks = []
                                for item in top_summaries:
                                    task = store_in_rag(
                                        content=item['summary'],
                                        metadata={
                                            "source_type": "web_search",
                                            "url": item['url'],
                                            "title": item['title'],
                                            "domain": domain,
                                            "query": cleaned_query,
                                            "fetch_method": "ollama_search_summary",
                                            "timestamp": time.time()
                                        }
                                    )
                                    storage_tasks.append(task)

                                results = await asyncio.gather(*storage_tasks, return_exceptions=True)
                                stored_count = sum(1 for r in results if r is True)
                                logger.info(f"✅ Stored {stored_count}/{len(top_summaries)} summaries in RAG")

                            combined_content = []
                            for i, item in enumerate(top_summaries, 1):
                                logger.info(f"   {i}. {item['title']}")
                                combined_content.append(f"""
                        ═══════════════════════════════════════════════════════════════
                        SOURCE {i}: {item['title']}
                        URL: {item['url']}
                        ═══════════════════════════════════════════════════════════════

                        {item['summary']}

                        """)

                            note = f"\n\n**Note**: Auto-searched {domain} and found {len(top_summaries)} relevant page(s). All content stored in RAG."

                            return {
                                "success": True,
                                "content": "\n".join(combined_content) + note,
                                "urls_fetched": len(top_summaries),
                                "method": "ollama_search_summaries",
                                "stored_in_rag": True,
                                "rag_entries": stored_count
                            }
                        else:
                            logger.warning(f"⚠️ Site search found no relevant pages on {domain}")
                    logger.warning(f"⚠️ Site search found no relevant pages on {domain}")

            except Exception as e:
                logger.error(f"❌ Site search failed: {e}")

            # ═══════════════════════════════════════════════════════
            # FALLBACK: Fetch homepage anyway with warning
            # ═══════════════════════════════════════════════════════
            logger.info(f"📄 Site search failed, fetching homepage as fallback")

            result = await fetch_url_content(source)

            if result.get("success"):
                content = result.get("content", "")
                title = result.get("title", "Untitled")

                await store_in_rag(
                    content=content,
                    metadata={
                        "source_type": "homepage",
                        "url": source,
                        "title": title,
                        "domain": domain,
                        "query": query,
                        "fetch_method": "homepage_fallback",
                        "timestamp": time.time()
                    }
                )

                warning = f"""
⚠️ **Homepage Warning**: The source URL was a homepage with limited content.
Site search found no relevant articles. Results may be limited.

For better results, try:
- Using a specific article URL from {domain}
- Searching the site manually first
- Providing more specific keywords

"""

                combined_content = f"""
═══════════════════════════════════════════════════════════════
SOURCE 1: {title} (Homepage)
URL: {source}
═══════════════════════════════════════════════════════════════

{content}

{warning}
"""

                return {
                    "success": True,
                    "content": combined_content,
                    "urls_fetched": 1,
                    "method": "homepage_fallback",
                    "stored_in_rag": True
                }
            else:
                return {
                    "success": False,
                    "error": f"Homepage fetch failed: {result.get('error')}"
                }

        # ═══════════════════════════════════════════════════════════
        # NOT A HOMEPAGE - Fetch specific URL directly
        # ═══════════════════════════════════════════════════════════
        logger.info(f"📄 Fetching specific URL directly")

        try:
            result = await fetch_url_content(source)

            if result.get("success"):
                content = result.get("content", "")
                title = result.get("title", "Untitled")

                await store_in_rag(
                    content=content,
                    metadata={
                        "source_type": "web_page",
                        "url": source,
                        "title": title,
                        "query": query,
                        "fetch_method": "direct_fetch",
                        "timestamp": time.time()
                    }
                )

                combined_content = f"""
═══════════════════════════════════════════════════════════════
SOURCE 1: {title}
URL: {source}
═══════════════════════════════════════════════════════════════

{content}

"""
                logger.info(f"✅ Fetched specific URL: {title}")

                return {
                    "success": True,
                    "content": combined_content,
                    "urls_fetched": 1,
                    "method": "specific_url",
                    "stored_in_rag": True
                }
            else:
                logger.warning(f"⚠️ Failed to fetch URL: {result.get('error')}")
                source = parsed.netloc
                logger.info(f"🔄 Falling back to domain: {source}")

        except Exception as e:
            logger.error(f"❌ Exception fetching URL: {e}")
            try:
                source = parsed.netloc
                logger.info(f"🔄 Exception recovery - using domain: {source}")
            except:
                return {"success": False, "error": f"Invalid URL: {source}"}

    # ═══════════════════════════════════════════════════════════════
    # REST OF EXISTING CODE (Direct access, Ollama Search fallback)
    # ═══════════════════════════════════════════════════════════════

    # Try direct access (pre-configured URLs)
    direct_result = await fetch_from_source_directly(source, query)

    if direct_result.get("success"):
        urls = direct_result.get("urls", [])
        logger.info(f"✅ Got {len(urls)} URLs via direct access")
    else:
        # Fall back to Ollama Search
        logger.info(f"⚠️ No direct URLs, trying Ollama Search")
        search = get_search_client()

        if not search.is_available():
            return {"success": False, "error": "No direct URLs and Ollama Search unavailable"}

        search_query = f"{source} {query}"
        logger.info(f"🔍 Ollama Search query: {search_query[:100]}...")

        search_result = await search.search(search_query)

        if not search_result.get("success"):
            return {"success": False, "error": "Ollama Search failed"}

        results_data = search_result.get("results")

        # Try to extract URLs from the search response
        urls = []

        # Method 1: If results_data is a dict with structured data
        if isinstance(results_data, dict):
            # Try webPages.value structure (Bing-style)
            web_pages = results_data.get("webPages", {})
            if isinstance(web_pages, dict):
                for item in web_pages.get("value", []):
                    if isinstance(item, dict) and item.get("url"):
                        urls.append(item["url"])

            # Try organic results structure
            if not urls:
                for item in results_data.get("organic", []):
                    if isinstance(item, dict) and item.get("url"):
                        urls.append(item["url"])

            # Try raw_response if available
            if not urls:
                raw = search_result.get("raw_response", {})
                if isinstance(raw, dict):
                    web_pages = raw.get("webPages", {})
                    if isinstance(web_pages, dict):
                        for item in web_pages.get("value", []):
                            if isinstance(item, dict) and item.get("url"):
                                urls.append(item["url"])

        # Method 2: Extract URLs via regex from string representation
        if not urls:
            results_str = str(results_data)
            url_pattern = r'(?:url["\']?\s*[:=]\s*["\']?)?(https?://[^\s\'">\)]+)'
            matches = re.findall(url_pattern, results_str, re.IGNORECASE)

            seen = set()
            for match in matches:
                url = match.strip('",}]')
                if url and url not in seen and url.startswith('http'):
                    urls.append(url)
                    seen.add(url)

        logger.info(f"📋 Ollama Search found {len(urls)} URLs")

        if not urls:
            return {"success": False, "error": "No URLs found"}

    # Fetch actual content from discovered URLs
    unique_urls = list(dict.fromkeys(urls))[:3]
    logger.info(f"📄 Fetching {len(unique_urls)} URLs")
    for i, url in enumerate(unique_urls, 1):
        logger.info(f"   {i}. {url}")

    fetch_tasks = [fetch_url_content(url) for url in unique_urls]
    fetch_results = await asyncio.gather(*fetch_tasks)

    combined_content = []
    stored_count = 0

    for i, result in enumerate(fetch_results):
        if result.get("success"):
            url = unique_urls[i]
            title = result.get("title", "Untitled")
            content = result.get("content", "")

            await store_in_rag(
                content=content,
                metadata={
                    "source_type": "web_page",
                    "url": url,
                    "title": title,
                    "query": query,
                    "fetch_method": "multi_fetch",
                    "timestamp": time.time()
                }
            )
            stored_count += 1

            combined_content.append(f"""
═══════════════════════════════════════════════════════════════
SOURCE {i + 1}: {title}
URL: {url}
═══════════════════════════════════════════════════════════════

{content}

""")
            logger.info(f"✅ Fetched: {title}")
        else:
            logger.warning(f"⚠️ Failed to fetch {unique_urls[i]}: {result.get('error')}")

    if not combined_content:
        return {
            "success": False,
            "error": "Failed to fetch any content",
            "attempted_urls": unique_urls
        }

    return {
        "success": True,
        "content": "\n".join(combined_content),
        "urls_fetched": len(combined_content),
        "method": "direct" if direct_result.get("success") else "ollama_search",
        "stored_in_rag": stored_count > 0,
        "rag_entries": stored_count
    }


def router(state):
    """
    Route based on what the agent decided to do
    WITH STOP SIGNAL HANDLING AND A2A LOOP PREVENTION
    """
    last_message = state["messages"][-1]
    logger = logging.getLogger("mcp_client")
    logger.info(f"[LangGraph] 🎯 Router: Last message type = {type(last_message).__name__}")

    # Stop signal check
    if is_stop_requested():
        logger.warning(f"🛑 Router: Stop requested - ending graph execution")
        state["stopped"] = True
        return "continue"

    if state.get("stopped", False):
        logger.warning(f"🛑 Router: Execution already stopped - ending")
        return "continue"

    # A2A completion check
    from langchain_core.messages import ToolMessage
    if isinstance(last_message, ToolMessage):
        if hasattr(last_message, 'name') and last_message.name in ["send_a2a", "discover_a2a",
                                                                   "send_a2a_streaming", "send_a2a_batch"]:
            logger.info(f"🛑 Router: {last_message.name} result received - ending execution")
            return "continue"

    # If LLM just formatted tool results, don't re-route
    if isinstance(last_message, AIMessage):
        # ── Check for research sentinel ──
        if last_message.content == "__RESEARCH__":
            logger.info("🔬 Router: Research sentinel detected → research node")
            return "research"

        tool_calls = getattr(last_message, "tool_calls", [])
        if tool_calls:
            return "tools"
        return "continue"

    # Get user's original message
    user_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message = msg
            break

    if user_message:
        content = user_message.content

        research_source = extract_research_sources(content)
        if research_source:
            logger.info(f"🔬 Router: SOURCE-BASED RESEARCH detected: '{research_source}'")
            state["research_source"] = research_source
            return "research"

        # Status query check
        if ROUTER_STATUS_QUERY.search(content):
            logger.info(f"🎯 Router: Status query detected - continuing normally")
            return "continue"

        # Ingest routing
        if ROUTER_INGEST_COMMAND.search(content) and not ROUTER_STATUS_QUERY.search(content):
            if not state.get("ingest_completed", False):
                if ROUTER_ONE_TIME_INGEST.search(content):
                    logger.info(f"🎯 Router: ONE-TIME ingest requested")
                    return "ingest"
                if ROUTER_MULTI_STEP.search(content):
                    logger.info(f"🎯 Router: INGEST with multiple steps")
                    return "continue"
                logger.info(f"🎯 Router: INGEST requested")
                return "ingest"
            else:
                logger.info(f"🎯 Router: Ingest already completed")
                return "continue"

        # Explicit RAG requests
        if ROUTER_EXPLICIT_RAG.search(content):
            logger.info(f"🎯 Router: Explicit RAG request")
            return "rag"

    # Default: continue to END
    logger.info(f"[LangGraph] 🎯 Router: Continuing to END")
    return "continue"


async def rag_node(state):
    """Search RAG and provide context to answer the question"""
    logger = logging.getLogger("mcp_client")

    if is_stop_requested():
        logger.warning("🛑 RAG node: Stop requested")
        msg = AIMessage(content="Search cancelled by user.")
        return {"messages": [msg], "llm": state.get("llm"), "stopped": True}

    user_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message = msg
            break

    if not user_message:
        logger.error("❌ No user message found in RAG node")
        msg = AIMessage(content="Error: Could not find user's question.")
        return {"messages": [msg], "llm": state.get("llm")}

    # Find rag_search_tool
    tools_dict = state.get("tools", {})
    rag_search_tool = None
    for tool in tools_dict.values():
        if hasattr(tool, 'name') and tool.name == "rag_search_tool":
            rag_search_tool = tool
            break

    if not rag_search_tool:
        logger.error(f"❌ RAG search tool not found")
        msg = AIMessage(content="RAG search is not available.")
        return {"messages": [msg], "llm": state.get("llm")}

    try:
        result = await rag_search_tool.ainvoke({"query": user_message.content})
        context = "RAG search results here"

        augmented_messages = [
            SystemMessage(content=f"Context from RAG:\n\n{context}"),
            user_message
        ]

        llm = state.get("llm")
        response = await llm.ainvoke(augmented_messages)

        return {"messages": [response], "llm": state.get("llm")}

    except Exception as e:
        logger.error(f"❌ Error in RAG node: {e}")
        msg = AIMessage(content=f"Error searching knowledge base: {str(e)}")
        return {"messages": [msg], "llm": state.get("llm")}


# 4-TIER RESEARCH FALLBACK SYSTEM
# Tools → Direct Access → Ollama Search → LLM Knowledge
async def research_node(state):
    """Perform source-based research with multi-source support and RAG storage"""
    logger = logging.getLogger("mcp_client")

    if is_stop_requested():
        logger.warning("🛑 Research node: Stop requested")
        return {
            "messages": [AIMessage(content="Research cancelled.")],
            "llm": state.get("llm"),
            "stopped": True,
            "current_model": state.get("current_model", "unknown")
        }

    user_message = None
    for msg in reversed(state["messages"]):
        if isinstance(msg, HumanMessage):
            user_message = msg
            break

    if not user_message:
        return {
            "messages": [AIMessage(content="Error: No query found.")],
            "llm": state.get("llm"),
            "current_model": state.get("current_model", "unknown")
        }

    query = user_message.content

    sources = extract_research_sources(query)
    if not sources:
        sources = ["web"]

    logger.info(f"📚 Research sources ({len(sources)}): {sources}")

    query_cleaned = RESEARCH_SOURCE_PATTERN.sub('', query).strip()
    query_cleaned = re.sub(r'\s+', ' ', query_cleaned)
    logger.info(f"🔬 Research query: '{query_cleaned}'")

    llm = state.get("llm")

    tools_dict = state.get("tools", {})
    rag_add_tool = tools_dict.get("rag_add_tool")

    if rag_add_tool:
        logger.info("✅ RAG tool available - will store fetched content")
    else:
        logger.debug("ℹ️ RAG tool not available - skipping storage")

    try:
        all_content = []
        all_urls = []
        total_urls_fetched = 0
        failed_sources = []
        total_rag_entries = 0

        for i, source in enumerate(sources, 1):
            logger.info(f"🔍 [{i}/{len(sources)}] Fetching from: {source}")

            try:
                result = await search_and_fetch_source(source, query_cleaned, rag_add_tool)

                if result.get("success"):
                    all_content.append(result["content"])
                    total_urls_fetched += result.get("urls_fetched", 0)
                    all_urls.append(source)

                    if result.get("stored_in_rag"):
                        rag_count = result.get("rag_entries", 1)
                        total_rag_entries += rag_count
                        logger.info(f"✅ [{i}/{len(sources)}] Fetched from {source} ({rag_count} entries stored in RAG)")
                    else:
                        logger.info(f"✅ [{i}/{len(sources)}] Fetched from {source}")
                else:
                    error_msg = result.get("error", "Unknown error")
                    logger.warning(f"⚠️ [{i}/{len(sources)}] Failed: {source} - {error_msg}")
                    failed_sources.append(f"{source} ({error_msg})")

            except Exception as e:
                logger.error(f"❌ [{i}/{len(sources)}] Error: {source} - {e}")
                failed_sources.append(f"{source} (Exception: {str(e)[:50]})")

        if not all_content:
            failed_list = "\n".join([f"  - {s}" for s in failed_sources])
            return {
                "messages": [AIMessage(
                    content=f"❌ Unable to fetch content from any sources.\n\n"
                            f"**Attempted:**\n{failed_list}\n\n"
                            f"Try:\n- Checking the URLs\n- Using different sources\n- Simplifying your query"
                )],
                "llm": state.get("llm"),
                "current_model": state.get("current_model", "unknown")
            }

        combined_content = "\n\n".join(all_content)
        success_count = len(all_content)

        if total_rag_entries > 0:
            logger.info(
                f"✅ Combined content from {success_count}/{len(sources)} sources ({len(combined_content)} chars, {total_rag_entries} RAG entries)")
        else:
            logger.info(
                f"✅ Combined content from {success_count}/{len(sources)} sources ({len(combined_content)} chars)")

        if failed_sources:
            logger.warning(f"⚠️ Failed sources: {failed_sources}")

        sources_list = "\n".join([f"  {i + 1}. {url}" for i, url in enumerate(all_urls)])

        research_prompt = f"""I have fetched content from {success_count} source(s):

{sources_list}

CONTENT FROM ALL SOURCES:
{combined_content}

Question: {query_cleaned}

**Instructions:**
- Write a comprehensive answer synthesizing information from ALL {success_count} sources
- Cite each source using its ACTUAL URL (listed above)
- Example: "According to the Wikipedia article on Donald Trump (https://...), ..."
- When information comes from multiple sources, note this
- Include a References section listing all {success_count} sources at the end

Your answer:"""

        augmented_messages = state["messages"] + [HumanMessage(content=research_prompt)]

        # ATTEMPT 1: Try with full content
        try:
            response = await asyncio.wait_for(
                llm.ainvoke(augmented_messages),
                timeout=600.0
            )

            logger.info("✅ Research synthesis completed")

            notes = []

            if failed_sources:
                notes.append("⚠️ **Note**: Some sources could not be accessed:\n" +
                             "\n".join([f"- {s}" for s in failed_sources]))

            if total_rag_entries > 0:
                notes.append(
                    f"💾 **Note**: All fetched content ({total_rag_entries} pages) has been stored in RAG for future reference.")

            if notes and hasattr(response, 'content') and response.content:
                response.content += "\n\n---\n\n" + "\n\n".join(notes)

            return {
                "messages": [response],
                "llm": state.get("llm"),
                "current_model": state.get("current_model", "unknown")
            }

        except asyncio.TimeoutError:
            logger.warning(f"⏱️ Research timed out, will retry with summarized content")

        except ValueError as e:
            error_str = str(e).lower()
            if any(phrase in error_str for phrase in [
                "context window", "exceed", "token", "too long", "maximum context"
            ]):
                logger.warning(f"⚠️ Context overflow: {str(e)[:100]}")
            else:
                raise

        except Exception as e:
            error_str = str(e).lower()
            if not any(phrase in error_str for phrase in ["context", "token", "length", "exceed"]):
                logger.error(f"❌ Research failed: {e}")
                return {
                    "messages": [AIMessage(content=f"Research error: {str(e)}")],
                    "llm": state.get("llm"),
                    "current_model": state.get("current_model", "MCP Error")
                }

        # ATTEMPT 2: Retry with summarization
        logger.info("🔄 Retrying with content summarization...")

        summary_prompt = f"""Summarize this content from {success_count} sources concisely, keeping key facts relevant to: "{query_cleaned}"

{combined_content}

Provide a structured summary under 1000 words:"""

        try:
            summary_response = await asyncio.wait_for(
                llm.ainvoke([HumanMessage(content=summary_prompt)]),
                timeout=120.0
            )
            summarized_content = summary_response.content
            logger.info(f"✅ Summarized: {len(combined_content)} → {len(summarized_content)} chars")

        except asyncio.TimeoutError:
            logger.warning("⚠️ Summary timed out, using truncation")
            summarized_content = combined_content[:1500] + "\n\n[Content truncated]"

        except Exception as e:
            logger.error(f"❌ Summary failed: {e}, using truncation")
            summarized_content = combined_content[:1500] + "\n\n[Content truncated]"

        retry_prompt = f"""I have fetched and SUMMARIZED content from {success_count} sources.

Sources:
{sources_list}

SUMMARIZED CONTENT:
{summarized_content}

Question: {query_cleaned}

**Instructions:**
- Write a comprehensive answer based on the summary
- Cite each source by its URL
- Note this is based on summarized content

Your answer:"""

        retry_messages = state["messages"] + [HumanMessage(content=retry_prompt)]

        try:
            response = await asyncio.wait_for(
                llm.ainvoke(retry_messages),
                timeout=300.0
            )

            logger.info("✅ Research completed with summarized content")

            disclaimers = [
                "⚠️ **Note**: Answer based on summary due to content length."
            ]

            if failed_sources:
                disclaimers.append(
                    "⚠️ **Some sources unavailable:**\n" +
                    "\n".join([f"- {s}" for s in failed_sources])
                )

            if total_rag_entries > 0:
                disclaimers.append(
                    f"💾 **Note**: All fetched content ({total_rag_entries} pages) has been stored in RAG for future reference.")

            if hasattr(response, 'content'):
                response.content += "\n\n---\n\n" + "\n\n".join(disclaimers)

            return {
                "messages": [response],
                "llm": state.get("llm"),
                "current_model": state.get("current_model", "unknown")
            }

        except asyncio.TimeoutError:
            logger.error("❌ Retry also timed out")
            return {
                "messages": [AIMessage(
                    content=f"⏱️ Timeout: Research took too long even with {success_count} sources.\n\n"
                            f"Try:\n- More specific question\n- Fewer sources\n- Asking about specific sections"
                )],
                "llm": state.get("llm"),
                "current_model": state.get("current_model", "unknown")
            }

        except Exception as e:
            logger.error(f"❌ Retry failed: {e}")
            return {
                "messages": [AIMessage(
                    content=f"❌ Research Failed\n\n"
                            f"Could not process content from {success_count} sources.\n\n"
                            f"Error: {str(e)}\n\n"
                            f"Try:\n- More specific question\n- Fewer sources\n- Different sources"
                )],
                "llm": state.get("llm"),
                "current_model": state.get("current_model", "unknown")
            }

    except Exception as e:
        logger.error(f"❌ Research failed completely: {e}")
        return {
            "messages": [AIMessage(content=f"Research error: {str(e)}")],
            "llm": state.get("llm"),
            "current_model": state.get("current_model", "unknown")
        }

def should_continue_after_tools(state: AgentState) -> str:
    """
    Check if tools requested continuation/improvement.

    Returns:
        "agent" - Go back to LLM for refinement
        "end" - Normal termination
    """
    logger = logging.getLogger("mcp_client")
    messages = state.get("messages", [])

    # Check last few messages for feedback marker
    for msg in reversed(messages[-5:]):
        if isinstance(msg, HumanMessage) and "[Tool Feedback" in msg.content:
            logger.info("🔄 Tool feedback detected - continuing to agent")
            return "agent"

    # No feedback - normal end
    return "end"

def _needs_web_search(message: str) -> bool:
    """
    Determine if a message genuinely needs current external information.
    Filters out statements, instructions, personal facts, and creative requests.
    Exclusions are checked first — if any match, returns False immediately.
    """
    msg = message.strip()

    # Explicit exclusions — never search these
    STATEMENT_PATTERNS = [
        r"^(my |i |i'm |i am )",                                        # personal statements
        r"^(acknowledge|confirm|remember|note that|please note)",        # memory instructions
        r"^(create|write|generate|make|draft|compose|give me a|tell me a story|tell me a poem)",  # creative tasks
        r"^(yes|no|ok|okay|sure|thanks|thank you|hello|hi\b)",          # conversational
        r"\b(favourite|favorite|i like|i love|i hate|i prefer)\b",      # personal preferences
        r"\b(weather|forecast|temperature|rain|snow|wind|humidity)\b",  # weather — handled by MCP tools
        r"\b(my (location|todo|task|note|system|plex|library))\b",      # personal data — handled by MCP tools
    ]

    for pattern in STATEMENT_PATTERNS:
        if re.search(pattern, msg, re.IGNORECASE):
            return False

    # Must match a genuine question or current-info request
    SEARCH_PATTERNS = [
        r"\?",                                                           # question mark
        r"\b(what is|what are|what was|what were)\b",
        r"\b(who is|who are|who was|who were)\b",
        r"\b(when is|when was|when did)\b",
        r"\b(where is|where are|where was)\b",
        r"\b(how (much|many|long|old|far|do|does|did|is|are))\b",
        r"\b(current|latest|recent|today|right now|as of)\b",
        r"\b(news|price|score|stock|update)\b",            # weather removed — MCP tools handle it
    ]

    return any(re.search(p, msg, re.IGNORECASE) for p in SEARCH_PATTERNS)


def create_langgraph_agent(llm_with_tools, tools):
    """Create and compile the LangGraph agent"""
    logger = logging.getLogger("mcp_client")

    base_llm = llm_with_tools.bound if hasattr(llm_with_tools, 'bound') else llm_with_tools

    def get_model_name(llm):
        """Extract model name from LLM instance"""
        if hasattr(llm, 'model'):
            return llm.model
        elif hasattr(llm, 'model_name'):
            return llm.model_name
        elif hasattr(llm, 'model_path'):
            from pathlib import Path
            return Path(llm.model_path).stem
        else:
            return "unknown"

    async def call_model(state: AgentState):
        if is_stop_requested():
            logger.warning("🛑 call_model: Stop requested")
            empty_response = AIMessage(content="Operation cancelled by user.")
            return {
                "messages": [empty_response],
                "tools": state.get("tools", {}),
                "llm": state.get("llm"),
                "ingest_completed": state.get("ingest_completed", False),
                "stopped": True,
                "current_model": get_model_name(base_llm)
            }

        # Ollama search override
        messages = state["messages"]
        user_message = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_message = msg.content
                break

        # Regex pattern for Ollama Search triggers
        OLLAMA_SEARCH_PATTERN = re.compile(
            r'\bollama\s+search\b'
            r'|\bollama\s+search\s+(for|about|on)\b'
            r'|\bweb\s+search\s+using\s+ollama\b',
            re.IGNORECASE
        )

        if user_message and OLLAMA_SEARCH_PATTERN.search(user_message):
            logger.info("🔍 EXPLICIT OLLAMA SEARCH REQUESTED - bypassing all tools")

            search_client = get_search_client()

            if not search_client.is_available():
                error_response = AIMessage(
                    content="❌ Ollama Search is not available. Check OLLAMA_TOKEN in .env"
                )
                return {
                    "messages": [error_response],
                    "tools": state.get("tools", {}),
                    "llm": state.get("llm"),
                    "ingest_completed": state.get("ingest_completed", False),
                    "stopped": state.get("stopped", False),
                    "current_model": get_model_name(base_llm)
                }

            # Strip Ollama Search phrases from query
            query = OLLAMA_SEARCH_PATTERN.sub('', user_message)

            # Remove common command prefixes
            query = re.sub(r'^\s*(use|using|with|via|please|can you)\s+', '', query, flags=re.IGNORECASE)
            query = re.sub(r'\s+(for me|to)\s+', ' ', query, flags=re.IGNORECASE)
            query = query.strip().lstrip(':,.')

            if not query:
                query = user_message

            logger.info(f"🔍 Searching: '{query}'")

            try:
                search_result = await search_client.search(query)

                if search_result["success"] and search_result["results"]:
                    search_context = search_result["results"]

                    # ═══════════════════════════════════════════════════════════
                    # INGEST SEARCH RESULTS INTO RAG
                    # ═══════════════════════════════════════════════════════════
                    tools_dict = state.get("tools", {})
                    rag_add_tool = tools_dict.get("rag_add_tool")

                    if rag_add_tool:
                        logger.info("💾 Ingesting Ollama Search results into RAG...")

                        ingested_count = 0

                        try:
                            # The search_context is already the raw response
                            # Try to extract pages from different possible formats
                            pages = []

                            if isinstance(search_context, dict):
                                # Format 1: Direct webPages structure
                                if "webPages" in search_context:
                                    web_pages = search_context.get("webPages", {})
                                    if isinstance(web_pages, dict):
                                        pages = web_pages.get("value", [])
                                # Format 2: Direct list of results
                                elif "results" in search_context:
                                    pages = search_context.get("results", [])
                                # Format 3: Top-level value array
                                elif "value" in search_context:
                                    pages = search_context.get("value", [])

                            elif isinstance(search_context, list):
                                # Direct list of results
                                pages = search_context

                            elif isinstance(search_context, str):
                                # Try to parse JSON string
                                try:
                                    data = json.loads(search_context)
                                    if isinstance(data, dict):
                                        if "webPages" in data:
                                            pages = data.get("webPages", {}).get("value", [])
                                        elif "results" in data:
                                            pages = data.get("results", [])
                                        elif "value" in data:
                                            pages = data.get("value", [])
                                    elif isinstance(data, list):
                                        pages = data
                                except json.JSONDecodeError:
                                    logger.warning("⚠️ Could not parse search context as JSON")

                            logger.info(f"📋 Found {len(pages)} pages to ingest")

                            # Ingest each search result
                            for i, page in enumerate(pages[:5], 1):  # Limit to top 5 results
                                if isinstance(page, dict):
                                    # Try different field names
                                    url = page.get("url") or page.get("link") or page.get("href") or ""
                                    title = page.get("name") or page.get("title") or "Untitled"
                                    snippet = page.get("snippet") or page.get("description") or page.get(
                                        "summary") or ""

                                    if snippet and url:
                                        rag_entry = {
                                            "text": snippet,
                                            "source": url,
                                            "metadata": {
                                                "source_type": "web_search",
                                                "url": url,
                                                "title": title,
                                                "query": query,
                                                "fetch_method": "ollama_search",
                                                "timestamp": time.time()
                                            }
                                        }

                                        try:
                                            await rag_add_tool.ainvoke(rag_entry)
                                            ingested_count += 1
                                            logger.info(f"   ✅ [{i}/5] Ingested: {title[:50]}")
                                        except Exception as e:
                                            logger.warning(f"   ⚠️ [{i}/5] Failed to ingest {title[:50]}: {e}")
                                    else:
                                        logger.warning(f"   ⚠️ [{i}/5] Skipping - missing snippet or URL")

                            if ingested_count > 0:
                                logger.info(f"✅ Ingested {ingested_count}/5 search results into RAG")
                            else:
                                logger.warning("⚠️ No results ingested into RAG - check response format")
                                # Debug: log the actual structure
                                logger.debug(f"Search context type: {type(search_context)}")
                                if isinstance(search_context, dict):
                                    logger.debug(f"Keys: {list(search_context.keys())}")

                        except Exception as e:
                            logger.error(f"❌ Failed to ingest search results: {e}")
                            import traceback
                            traceback.print_exc()
                    else:
                        ingested_count = 0
                        logger.warning("⚠️ RAG tool not available - skipping ingestion")

                    augmented_prompt = f"""Web search results for: "{query}"

        {search_context}

        Based on these search results, provide a clear answer."""

                    augmented_messages = messages + [HumanMessage(content=augmented_prompt)]
                    response = await base_llm.ainvoke(augmented_messages)

                    # Add note about RAG ingestion
                    if rag_add_tool and ingested_count > 0:
                        if hasattr(response, 'content'):
                            response.content += f"\n\n💾 *{ingested_count} search results stored in RAG for future reference.*"

                    return {
                        "messages": [response],
                        "tools": state.get("tools", {}),
                        "llm": state.get("llm"),
                        "ingest_completed": state.get("ingest_completed", False),
                        "stopped": state.get("stopped", False),
                        "current_model": get_model_name(base_llm)
                    }
                else:
                    error_response = AIMessage(
                        content=f"🔍 Ollama Search returned no results for: '{query}'"
                    )
                    return {
                        "messages": [error_response],
                        "tools": state.get("tools", {}),
                        "llm": state.get("llm"),
                        "ingest_completed": state.get("ingest_completed", False),
                        "stopped": state.get("stopped", False),
                        "current_model": get_model_name(base_llm)
                    }

            except Exception as e:
                logger.error(f"❌ Ollama Search failed: {e}")
                error_response = AIMessage(
                    content=f"❌ Ollama Search error: {str(e)}"
                )
                return {
                    "messages": [error_response],
                    "tools": state.get("tools", {}),
                    "llm": state.get("llm"),
                    "ingest_completed": state.get("ingest_completed", False),
                    "stopped": state.get("stopped", False),
                    "current_model": get_model_name(base_llm)
                }

        # detect research intent before calling LLM
        user_message = None
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                user_message = msg.content
                break
        if user_message:
            sources = extract_research_sources(user_message)
            if sources:
                logger.info(f"🔬 call_model: Research sources detected → delegating to research_node")
                state["research_source"] = sources
                return {
                    "messages": [AIMessage(content="__RESEARCH__")],
                    "tools": state.get("tools", {}),
                    "llm": state.get("llm"),
                    "ingest_completed": False,
                    "stopped": False,
                    "current_model": get_model_name(base_llm),
                    "research_source": sources
                }

        messages = state["messages"]
        from langchain_core.messages import ToolMessage

        last_message = messages[-1] if messages else None

        # If formatting tool results, use base LLM
        if isinstance(last_message, ToolMessage):
            logger.info("[LangGraph] 🎯 Formatting tool results")
            start_time = time.time()
            try:
                response = await asyncio.wait_for(
                    base_llm.ainvoke(messages),
                    timeout=300.0
                )
                duration = time.time() - start_time
                if METRICS_AVAILABLE:
                    metrics["llm_calls"] += 1
                    metrics["llm_times"].append((time.time(), duration))
                return {
                    "messages": [response],
                    "tools": state.get("tools", {}),
                    "llm": state.get("llm"),
                    "ingest_completed": state.get("ingest_completed", False),
                    "stopped": state.get("stopped", False),
                    "current_model": get_model_name(base_llm)
                }
            except asyncio.TimeoutError:
                duration = time.time() - start_time
                if METRICS_AVAILABLE:
                    metrics["llm_errors"] += 1
                    metrics["llm_times"].append((time.time(), duration))
                logger.error(f"⏱️ LLM call timed out after 5m")

                timeout_message = AIMessage(content="""⏱️ Request timed out after 5 minutes.

    **The model is taking too long to respond.** This usually happens when:
    - The model is processing too many tools (58 tools detected)
    - The query is ambiguous and the model is stuck deciding
    - The model is overloaded

    **Try these solutions:**
    1. Rephrase your question more specifically
    2. Break complex questions into smaller parts
    3. Restart the Ollama service: `ollama restart`

    **Your question:** {question}""".format(question=messages[-1].content if messages else "unknown"))

                return {
                    "messages": [timeout_message],
                    "tools": state.get("tools", {}),
                    "llm": state.get("llm"),
                    "ingest_completed": state.get("ingest_completed", False),
                    "stopped": state.get("stopped", False),
                    "current_model": get_model_name(base_llm)
                }

            except Exception as e:
                duration = time.time() - start_time
                if METRICS_AVAILABLE:
                    metrics["llm_errors"] += 1
                    metrics["llm_times"].append((time.time(), duration))
                logger.error(f"❌ Model call failed: {e}")
                raise

        # Get user message
        user_message = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                user_message = msg.content
                break

        # Force web search if explicitly requested
        if user_message:
            user_lower = user_message.lower()

            web_search_patterns = [
                r'\buse\s+web\s+search\b',
                r'\busing\s+web\s+search\b',
                r'\bwith\s+web\s+search\b',
                r'\bweb\s+search\s+for\b',
                r'\bvia\s+web\s+search\b',
            ]

            should_use_web_search = any(
                re.search(pattern, user_lower)
                for pattern in web_search_patterns
            )

            if should_use_web_search:
                logger.info("[LangGraph] 🎯 FORCED WEB SEARCH: User explicitly requested web search")

                search_client = get_search_client()

                if search_client.is_available():
                    query = user_message
                    for pattern in web_search_patterns:
                        query = re.sub(pattern, '', query, flags=re.IGNORECASE)
                    query = query.strip()

                    query = re.sub(r'^,?\s*(who|what|where|when|why|how)\s+', r'\1 ', query, flags=re.IGNORECASE)
                    query = query.strip()

                    if not query:
                        query = user_message

                    logger.info(f"🔍 Performing web search: '{query}'")

                    try:
                        search_result = await search_client.search(query)

                        if search_result["success"] and search_result["results"]:
                            logger.info("[LangGraph] ✅ Web search successful - passing to LLM for processing")
                            search_context = search_result["results"]

                            augmented_prompt = f"""I searched the web and found the following results:

    {search_context}

    Based on these search results, please answer the user's question: "{user_message}"

    Provide a clear, concise answer in English. Extract the most relevant information and present it naturally."""

                            augmented_messages = messages + [HumanMessage(content=augmented_prompt)]
                            response = await base_llm.ainvoke(augmented_messages)

                            return {
                                "messages": [response],
                                "tools": state.get("tools", {}),
                                "llm": state.get("llm"),
                                "ingest_completed": state.get("ingest_completed", False),
                                "stopped": state.get("stopped", False),
                                "current_model": get_model_name(base_llm)
                            }
                        else:
                            logger.warning("⚠️ Web search returned no results")

                    except Exception as e:
                        logger.error(f"❌ Web search failed: {e}")

                else:
                    logger.warning("⚠️ Web search not available")
                    error_response = AIMessage(
                        content="Web search is not available. Please check OLLAMA_TOKEN configuration."
                    )
                    return {
                        "messages": [error_response],
                        "tools": state.get("tools", {}),
                        "llm": state.get("llm"),
                        "ingest_completed": state.get("ingest_completed", False),
                        "stopped": state.get("stopped", False),
                        "current_model": get_model_name(base_llm)
                    }

        # ═══════════════════════════════════════════════════════════
        # CENTRALIZED PATTERN MATCHING
        # ═══════════════════════════════════════════════════════════
        def match_intent(user_message: str, all_tools: list, base_llm, logger, conversation_state):
            """Match user intent using centralized pattern configuration"""

            for tool in all_tools:
                if hasattr(tool, 'name') and tool.name.lower() in user_message.lower():
                    logger.info(f"🎯 Explicit tool name detected → binding only: {tool.name}")
                    return base_llm.bind_tools([tool]), "explicit_tool"

            has_project_context = False
            for msg in reversed(conversation_state.get("messages", [])[-5:]):
                if isinstance(msg, SystemMessage) and "CONVERSATION CONTEXT" in msg.content:
                    has_project_context = True
                    logger.info("[LangGraph] 🎯 Found project context in conversation - using code_assistant")
                    break

            if has_project_context:
                config = INTENT_PATTERNS["code_assistant"]
                filtered_tools = []
                for tool in all_tools:
                    for tool_pattern in config["tools"]:
                        if "*" in tool_pattern:
                            prefix = tool_pattern.replace("*", "")
                            if tool.name.startswith(prefix):
                                filtered_tools.append(tool)
                                break
                        elif tool.name == tool_pattern:
                            filtered_tools.append(tool)
                            break

                if filtered_tools:
                    logger.info(f"   → {len(filtered_tools)} code tools (context-based routing)")
                    return base_llm.bind_tools(filtered_tools), "code_assistant"

            sorted_patterns = sorted(INTENT_PATTERNS.items(), key=lambda x: x[1]["priority"])

            for intent_name, config in sorted_patterns:
                if re.search(config["pattern"], user_message, re.IGNORECASE):
                    if "exclude_pattern" in config:
                        if re.search(config["exclude_pattern"], user_message, re.IGNORECASE):
                            continue

                    logger.info(f"🎯 {intent_name} → filtering tools")

                    filtered_tools = []
                    for tool in all_tools:
                        for tool_pattern in config["tools"]:
                            if "*" in tool_pattern:
                                prefix = tool_pattern.replace("*", "")
                                if tool.name.startswith(prefix):
                                    filtered_tools.append(tool)
                                    break
                            elif tool.name == tool_pattern:
                                filtered_tools.append(tool)
                                break

                    if filtered_tools:
                        logger.info(f"   → {len(filtered_tools)} tools: {[t.name for t in filtered_tools[:5]]}")
                        return base_llm.bind_tools(filtered_tools), intent_name

            logger.info(f"🎯 General query → all {len(all_tools)} tools")
            return base_llm.bind_tools(all_tools), "general"

        # Apply pattern matching
        if user_message:
            all_tools = list(state.get("tools", {}).values())
            llm_to_use, pattern_name = match_intent(
                user_message,
                all_tools,
                base_llm,
                logger,
                state
            )
        else:
            llm_to_use = llm_with_tools

        current_model = get_model_name(llm_to_use)

        logger.info(f"🧠 Calling LLM with {len(messages)} messages")
        logger.info(f"🤖 Model: {current_model}")

        sanitized_messages = []
        for msg in messages:
            content = msg.content if msg.content is not None else ""
            content = str(content) if not isinstance(content, str) else content

            if isinstance(msg, HumanMessage):
                sanitized_messages.append(HumanMessage(content=content))
            elif isinstance(msg, AIMessage):
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    sanitized_messages.append(AIMessage(content=content, tool_calls=msg.tool_calls))
                else:
                    sanitized_messages.append(AIMessage(content=content))
            elif isinstance(msg, ToolMessage):
                tool_name = msg.name if msg.name is not None else "unknown_tool"
                sanitized_messages.append(
                    ToolMessage(content=content, tool_call_id=msg.tool_call_id, name=tool_name))
            elif isinstance(msg, SystemMessage):
                sanitized_messages.append(SystemMessage(content=content))
            else:
                sanitized_messages.append(msg)

        start_time = time.time()
        try:
            response = await asyncio.wait_for(
                llm_to_use.ainvoke(sanitized_messages),
                timeout=300.0
            )
            duration = time.time() - start_time

            if METRICS_AVAILABLE:
                metrics["llm_calls"] += 1
                metrics["llm_times"].append((time.time(), duration))

        except Exception as e:
            error_msg_str = str(e).lower()
            if "does not support chat" in error_msg_str:
                logger.error(f"❌ Model '{current_model}' does not support chat (embedding model?)")
                error_response = AIMessage(
                    content=f"❌ **'{current_model}'** does not support chat — it may be an embedding model.\n\nSwitch to a chat model: `:model qwen2.5:14b`")
                return {
                    "messages": [error_response],
                    "tools": state.get("tools", {}),
                    "llm": state.get("llm"),
                    "ingest_completed": state.get("ingest_completed", False),
                    "stopped": state.get("stopped", False),
                    "current_model": current_model
                }

            if "does not support tools" in error_msg_str:
                logger.error(f"❌ Model '{current_model}' does not support tool calling")

                all_tools_list = list(state.get("tools", {}).values())
                tool_count = len(all_tools_list)

                error_response = AIMessage(content=f"""❌ **Model Error**: The model '{current_model}' does not support tool calling.

    This model cannot use the {tool_count} tools available in this system.

    **Recommended models with tool support:**
    • `qwen2.5:14b` - Best quality, excellent tools (recommended)
    • `qwen2.5:7b` - Fast, good balance
    • `llama3.1:8b` - Solid general purpose
    • `llama3.2:3b` - Lightweight option
    • `mistral-nemo` - Balanced performance

    **To switch models:**
    Type: `:model qwen2.5:14b`

    **To install a model:**
    Run: `ollama pull qwen2.5:14b-instruct-q4_K_M`

    **Current setup:**
    - Model: {current_model}
    - Tools available: {tool_count}
    - Tool support: ❌ Not supported""")

                return {
                    "messages": [error_response],
                    "tools": state.get("tools", {}),
                    "llm": state.get("llm"),
                    "ingest_completed": state.get("ingest_completed", False),
                    "stopped": state.get("stopped", False),
                    "current_model": current_model
                }

            duration = time.time() - start_time
            if METRICS_AVAILABLE:
                metrics["llm_errors"] += 1
                metrics["llm_times"].append((time.time(), duration))
            logger.error(f"❌ Model call failed: {e}")
            raise

        # FALLBACK CHAIN
        try:
            has_tool_calls = hasattr(response, 'tool_calls') and response.tool_calls
            has_content = hasattr(response, 'content') and response.content and response.content.strip()

            if not has_tool_calls and not has_content:
                logger.warning("⚠️ LLM returned blank - trying web search")
                search_client = get_search_client()

                if search_client.is_available():
                    search_result = await search_client.search(user_message)

                    if search_result["success"] and search_result["results"]:
                        logger.info("[LangGraph] ✅ Web search successful")
                        search_context = search_result["results"]
                        augmented_prompt = f"""WEB SEARCH RESULTS:
    {search_context}

    Please answer the question using these search results."""

                        retry_messages = messages + [HumanMessage(content=augmented_prompt)]
                        response = await base_llm.ainvoke(retry_messages)
                    else:
                        logger.warning("⚠️ Web search failed - using base LLM")
                        response = await asyncio.wait_for(
                            base_llm.ainvoke(messages),
                            timeout=300.0
                        )
                else:
                    logger.warning("⚠️ Web search unavailable - using base LLM")
                    response = await asyncio.wait_for(
                        base_llm.ainvoke(messages),
                        timeout=300.0
                    )

                current_model = get_model_name(base_llm)

            elif not has_tool_calls and has_content:
                needs_current_info = _needs_web_search(user_message)

                if needs_current_info:
                    logger.info("[LangGraph] 🔍 Trying web search fallback for current info")
                    search_client = get_search_client()

                    if search_client.is_available():
                        search_result = await search_client.search(user_message)

                        if search_result["success"] and search_result["results"]:
                            logger.info("[LangGraph] ✅ Web search successful - augmenting")
                            search_context = search_result["results"]
                            augmented_prompt = f"""Previous answer: {response.content}

    However, here are current web search results:
    {search_context}

    Please provide an updated answer using these search results."""

                            retry_messages = messages + [response, HumanMessage(content=augmented_prompt)]
                            response = await base_llm.ainvoke(retry_messages)
                            current_model = get_model_name(base_llm)

            return {
                "messages": [response],
                "tools": state.get("tools", {}),
                "llm": state.get("llm"),
                "ingest_completed": state.get("ingest_completed", False),
                "stopped": state.get("stopped", False),
                "current_model": current_model
            }

        except asyncio.TimeoutError:
            duration = time.time() - start_time
            if METRICS_AVAILABLE:
                metrics["llm_errors"] += 1
                metrics["llm_times"].append((time.time(), duration))
            logger.error(f"⏱️ LLM call timed out after 5m")

            return {
                "messages": [AIMessage(
                    content="⏱️ Request timed out after 5 minutes. Please try:\n\n1. Rephrasing your question\n2. Breaking it into smaller parts\n3. Using a simpler query")],
                "tools": state.get("tools", {}),
                "llm": state.get("llm"),
                "ingest_completed": state.get("ingest_completed", False),
                "stopped": state.get("stopped", False),
                "current_model": get_model_name(base_llm)
            }

        except Exception as e:
            duration = time.time() - start_time
            if METRICS_AVAILABLE:
                metrics["llm_errors"] += 1
                metrics["llm_times"].append((time.time(), duration))
            logger.error(f"❌ Model call failed: {e}")
            raise

    async def ingest_node(state: AgentState):
        # Extract limit from user message
        limit = 5  # default
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                m = re.search(r'\b(\d+)\b', msg.content)
                if m:
                    limit = int(m.group(1))
                break

        """Handle ingestion operations"""
        if is_stop_requested():
            logger.warning("🛑 ingest_node: Stop requested")
            msg = AIMessage(content="Ingestion cancelled by user.")
            return {
                "messages": [msg],
                "tools": state.get("tools", {}),
                "llm": state.get("llm"),
                "ingest_completed": True,
                "stopped": True
            }

        tools_dict = state.get("tools", {})
        ingest_tool = None
        for tool in tools_dict.values():
            if hasattr(tool, 'name') and tool.name == "plex_ingest_batch":
                ingest_tool = tool
                break

        if not ingest_tool:
            msg = AIMessage(content="Ingestion tool not available.")
            return {
                "messages": [msg],
                "tools": state.get("tools", {}),
                "llm": state.get("llm"),
                "ingest_completed": True,
                "stopped": False
            }

        try:
            logger.info("[LangGraph] 📥 Starting ingest operation...")
            result = await ingest_tool.ainvoke({"limit": 5})

            try:
                import json
                raw = result[0].text if isinstance(result, list) else result
                data = json.loads(raw) if isinstance(raw, str) else raw

                successful = data.get("successful_items", [])
                failed = data.get("failed_items", [])
                stats = data.get("stats", {})

                lines = [
                    f"✅ Ingested {data.get('successful', 0)}/{data.get('total_attempted', 0)} items "
                    f"({data.get('duration', 0):.1f}s)",
                    f"📊 Library: {stats.get('successfully_ingested', 0)} total ingested, "
                    f"{stats.get('remaining_unprocessed', 0)} remaining",
                ]
                if successful:
                    lines.append("\n✅ **Successful:**")
                    for item in successful:
                        lines.append(f"  • {item['title']} ({item.get('chunks', 0)} chunks)")
                if failed:
                    no_subs = [f['title'] for f in failed if 'subtitle' in f.get('reason', '').lower()]
                    errors = [f for f in failed if 'subtitle' not in f.get('reason', '').lower()]
                    if no_subs:
                        lines.append(f"\n⚠️ **No subtitles** ({len(no_subs)}): {', '.join(no_subs[:5])}"
                                     + (" ..." if len(no_subs) > 5 else ""))
                    if errors:
                        lines.append(f"\n❌ **Errors** ({len(errors)}):")
                        for f in errors:
                            lines.append(f"  • {f['title']}: {f['reason']}")

                msg = AIMessage(content="\n".join(lines))
            except Exception:
                msg = AIMessage(content=f"Ingestion complete: {result}")

            return {
                "messages": [msg],
                "tools": state.get("tools", {}),
                "llm": state.get("llm"),
                "ingest_completed": True,
                "stopped": False
            }
        except Exception as e:
            logger.error(f"❌ Error in ingest_node: {e}")
            msg = AIMessage(content=f"Ingestion failed: {str(e)}")
            return {
                "messages": [msg],
                "tools": state.get("tools", {}),
                "llm": state.get("llm"),
                "ingest_completed": True,
                "stopped": False
            }

    async def call_tools_with_stop_check(state: AgentState):
        """Execute tools with stop signal checking"""
        logger = logging.getLogger("mcp_client")

        if is_stop_requested():
            logger.warning("🛑 call_tools: Stop requested")
            empty_response = AIMessage(content="Tool execution cancelled by user.")
            return {
                "messages": [empty_response],
                "tools": state.get("tools", {}),
                "llm": state.get("llm"),
                "ingest_completed": state.get("ingest_completed", False),
                "stopped": True
            }

        from langchain_core.messages import ToolMessage
        last_message = state["messages"][-1]
        tool_calls = getattr(last_message, "tool_calls", [])

        if not tool_calls:
            logger.warning("⚠️ No tool calls found")
            return state

        context = {}
        for msg in reversed(state["messages"][-10:]):
            if isinstance(msg, SystemMessage) and "CONVERSATION CONTEXT" in msg.content:
                match = re.search(r'Active Project:\s*(.+?)(?:\n|$)', msg.content)
                if match:
                    context["project_path"] = match.group(1).strip()
                    logger.info(f"🔧 Context found: {context['project_path']}")
                break

        tool_messages = []
        for tool_call in tool_calls:
            if is_stop_requested():
                logger.warning(f"🛑 Stop requested - halting tool calls")
                break

            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            tool_id = tool_call.get("id")

            if context.get("project_path") and tool_name in ["get_project_dependencies", "analyze_code_file", "analyze_project", "scan_project_structure"]:
                if "project_path" not in tool_args:
                    logger.warning(f"🔧 AUTO-FIX: Adding missing project_path → '{context['project_path']}'")
                    tool_args["project_path"] = context["project_path"]
                    tool_call["args"]["project_path"] = context["project_path"]
                elif tool_args.get("project_path") in [".", "./"]:
                    logger.warning(f"🔧 AUTO-FIX: Replacing '.' → '{context['project_path']}'")
                    tool_args["project_path"] = context["project_path"]
                    tool_call["args"]["project_path"] = context["project_path"]

                logger.info(f"🔍 Final tool_args: {tool_args}")

            logger.info(f"🔧 Executing tool: {tool_name}")

            tools_dict = state.get("tools", {})
            tool = tools_dict.get(tool_name)

            if not tool:
                logger.error(f"❌ Tool '{tool_name}' not found")
                error_msg = ToolMessage(
                    content=f"Error: Tool '{tool_name}' not found",
                    tool_call_id=tool_id,
                    name=tool_name
                )
                tool_messages.append(error_msg)
                continue

            try:
                tool_start = time.time()
                result = await tool.ainvoke(tool_args)
                tool_duration = time.time() - tool_start

                if METRICS_AVAILABLE:
                    metrics["tool_calls"][tool_name] += 1
                    metrics["tool_times"][tool_name].append((time.time(), tool_duration))

                if isinstance(result, list) and len(result) > 0:
                    if hasattr(result[0], 'text'):
                        result = result[0].text

                if tool_name in ("plex_ingest_batch", "plex_ingest_items"):
                    try:
                        data = json.loads(result) if isinstance(result, str) else result
                        successful = data.get("successful_items", [])
                        failed = data.get("failed_items", [])
                        stats = data.get("stats", {})
                        lines = [
                            f"✅ Ingested {data.get('successful', 0)}/{data.get('total_attempted', 0)} items "
                            f"({data.get('duration', 0):.1f}s)",
                            f"📊 Library: {stats.get('successfully_ingested', 0)} total ingested, "
                            f"{stats.get('remaining_unprocessed', 0)} remaining",
                        ]
                        if successful:
                            lines.append("\nSuccessful:")
                            for item in successful:
                                lines.append(f"  • {item['title']} ({item.get('chunks', 0)} chunks)")
                        if failed:
                            no_subs = [f['title'] for f in failed if 'subtitle' in f.get('reason', '').lower()]
                            errors = [f for f in failed if 'subtitle' not in f.get('reason', '').lower()]
                            if no_subs:
                                lines.append(f"\n⚠️ No subtitles ({len(no_subs)}): {', '.join(no_subs[:5])}"
                                             + (" ..." if len(no_subs) > 5 else ""))
                            if errors:
                                lines.append(f"\n❌ Errors ({len(errors)}):")
                                for f in errors:
                                    lines.append(f"  • {f['title']}: {f['reason']}")
                        result = "\n".join(lines)
                    except Exception:
                        pass

                result_msg = ToolMessage(
                    content=str(result),
                    tool_call_id=tool_id,
                    name=tool_name
                )
                tool_messages.append(result_msg)
                logger.info(f"✅ Tool {tool_name} completed in {tool_duration:.2f}s")
                logger.info(f"✅ ToolMessage result: {str(result)}")
                from client.health import record_tool_call
                record_tool_call(tool_name, tool_duration)

            except Exception as e:
                logger.error(f"❌ Tool {tool_name} failed: {e}")
                from client.health import record_tool_call
                record_tool_call(tool_name, 0.0, error=str(e))
                if METRICS_AVAILABLE:
                    metrics["tool_errors"][tool_name] += 1
                error_msg = ToolMessage(
                    content=f"Error: {str(e)}",
                    tool_call_id=tool_id,
                    name=tool_name
                )
                tool_messages.append(error_msg)

        needs_improvement = False
        feedback_message = None

        for tool_msg in tool_messages:
            try:
                # Try to parse tool result as JSON
                result_data = json.loads(tool_msg.content)

                # Check for improvement feedback
                if isinstance(result_data, dict):
                    status = result_data.get("status")
                    feedback = result_data.get("feedback", {})

                    if status in ("needs_improvement", "low_quality"):
                        needs_improvement = True
                        reason = feedback.get("reason", "Tool suggested improvement")
                        suggestions = feedback.get("suggestions", [])

                        # Build feedback message
                        feedback_text = f"[Tool Feedback: {tool_msg.name}] {reason}"
                        if suggestions:
                            feedback_text += "\n\nSuggestions:\n" + "\n".join(f"  • {s}" for s in suggestions[:3])

                        feedback_message = feedback_text
                        logger.info(f"🔄 Tool {tool_msg.name} requested improvement: {reason}")
                        break
            except (json.JSONDecodeError, AttributeError):
                # Not JSON or doesn't have content - skip
                pass

        # If tool requested improvement, inject feedback as HumanMessage to continue loop
        if needs_improvement and feedback_message:
            tool_messages.append(
                HumanMessage(content=feedback_message)
            )

        return {
            "messages": tool_messages,
            "tools": state.get("tools", {}),
            "llm": state.get("llm"),
            "ingest_completed": state.get("ingest_completed", False),
            "stopped": state.get("stopped", False),
            "current_model": state.get("current_model")
        }

    # Build graph:
    # Each node is a function,
    # Edges define control flows between nodes depending on what the model decides
    # router decides:
    #   ├── "tools"    → tools → agent → router → …
    #   ├── "rag"      → rag → END
    #   ├── "ingest"   → ingest → END
    #   ├── "research" → research → END
    #   └── "continue" → END
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", call_tools_with_stop_check)
    workflow.add_node("rag", rag_node)
    workflow.add_node("ingest", ingest_node)
    workflow.add_node("research", research_node)

    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent",
        router,
        {
            "tools": "tools",
            "rag": "rag",
            "ingest": "ingest",
            "research": "research",
            "continue": END
        }
    )
    workflow.add_conditional_edges(
        "tools",
        should_continue_after_tools,
        {
            "agent": "agent",
            "end": "agent"  # Both go back to agent for now
        }
    )
    workflow.add_edge("ingest", END)
    workflow.add_edge("rag", END)
    workflow.add_edge("research", END)

    app = workflow.compile()
    logger.info("[LangGraph] ✅ LangGraph agent compiled successfully")
    return app

# The agent is a LangGraph compiled graph, created by langgraph.create_langgraph_agent(llm_with_tools, tools).
#   It creates nodes (unit of work), edges (connects nodes and defines what happens next)
#   In LangGraph terms it's a StateGraph — a directed graph where each node is a function and edges define the flow.
# Typically it looks something like:
# [START] → call_llm → should_use_tools?
#                          ├── yes → execute_tool → call_llm (loop)
#                          └── no  → [END]
# agent.ainvoke runs the entire LangGraph graph to completion and returns the final state
async def run_agent(agent, conversation_state, user_message, logger, tools, system_prompt, llm=None, max_history=20):
    """
    Execute the agent with the given user message and track metrics

    CONVERSATION HISTORY DESIGN:
    - The SystemMessage in conversation_state["messages"][0] is the source of truth
    - We preserve it through truncation and never recreate it
    - If no SystemMessage exists, we create one from system_prompt parameter
    """
    start_time = time.time()
    clear_stop()
    logger.info("[LangGraph] ✅ Stop signal cleared for new request")

    try:
        if METRICS_AVAILABLE:
            metrics["agent_runs"] += 1

        # STEP 1: Save the original SystemMessage (if it exists)
        original_system_msg = None
        has_system_msg = (
                conversation_state["messages"]
                and isinstance(conversation_state["messages"][0], SystemMessage)
        )

        if has_system_msg:
            original_system_msg = conversation_state["messages"][0]
            logger.info("[LangGraph] Preserving existing SystemMessage")
        else:
            original_system_msg = SystemMessage(content=system_prompt)
            conversation_state["messages"].insert(0, original_system_msg)
            logger.info("[LangGraph] Created new SystemMessage from parameter")

        # STEP 2: Add user message and truncate
        conversation_state["messages"].append(HumanMessage(content=user_message))

        system_msg = conversation_state["messages"][0]
        other_messages = conversation_state["messages"][1:]

        if len(other_messages) > max_history - 1:
            other_messages = other_messages[-(max_history - 1):]
            logger.info(f"[LangGraph] Truncated to last {max_history} messages")

        conversation_state["messages"] = [system_msg] + other_messages

        # STEP 3: Run the agent
        logger.info(f"🧠 Starting agent with {len(conversation_state['messages'])} messages")

        tool_registry = {tool.name: tool for tool in tools}

        # Hits the LLM
        # Message contains
        #   SystemMessage — system prompt / tool usage guide
        #   Previous conversation history (truncated to max_history, default 20)
        #   The new HumanMessage with the user's input (appended just before in Step 2)
        result = await agent.ainvoke({
            "messages": conversation_state["messages"],
            "tools": tool_registry,
            "llm": llm,
            "ingest_completed": False,
            "stopped": False,
            "current_model": "unknown",
            "research_source": "web"
        })

        # STEP 4: Update conversation state
        input_count = len(conversation_state["messages"])
        truly_new = result["messages"][input_count:]
        logger.info(f"📨 Agent added {len(truly_new)} new messages")
        conversation_state["messages"] = result["messages"]

        # STEP 5: Return results
        if METRICS_AVAILABLE:
            duration = time.time() - start_time
            metrics["agent_times"].append((time.time(), duration))

        final_model = result.get("current_model", "unknown")

        if METRICS_AVAILABLE:
            logger.info(f"✅ Agent run completed in {duration:.2f}s (Model: {final_model})")
        else:
            logger.info(f"✅ Agent run completed (Model: {final_model})")

        return {
            "messages": conversation_state["messages"],
            "current_model": final_model
        }

    except ValueError as e:
        error_str = str(e)
        if "exceed context window" in error_str or "Requested tokens" in error_str:
            import re
            match = re.search(r'Requested tokens \((\d+)\) exceed context window of (\d+)', error_str)

            if match:
                requested = int(match.group(1))
                available = int(match.group(2))
                overflow = requested - available
                logger.error(
                    f"❌ Context overflow: {requested} tokens requested, {available} available ({overflow} over)")
            else:
                requested = None
                available = None
                logger.error(f"❌ Context window overflow")

            current_msg_count = len(conversation_state["messages"])
            if current_msg_count > 3:
                new_limit = max(3, current_msg_count // 2)
                logger.warning(f"⚠️  Auto-recovery: Reducing history from {current_msg_count} to {new_limit} messages")

                system_msg = conversation_state["messages"][0] if isinstance(conversation_state["messages"][0],
                                                                             SystemMessage) else None
                user_msg = conversation_state["messages"][-1]
                middle_msgs = conversation_state["messages"][1:-1]

                trimmed_middle = middle_msgs[-(new_limit - 2):] if len(middle_msgs) > (new_limit - 2) else middle_msgs

                if system_msg:
                    conversation_state["messages"] = [system_msg] + trimmed_middle + [user_msg]
                else:
                    conversation_state["messages"] = trimmed_middle + [user_msg]

                error_msg = AIMessage(content=f"""⚠️ Context window overflow detected and auto-fixed.

**Issue:** Your conversation ({requested} tokens) exceeded the model's limit ({available} tokens).

**Auto-recovery:** Reduced history from {current_msg_count} to {len(conversation_state['messages'])} messages.

**Suggestions:**
1. Start a new chat
2. `:model qwen2.5:14b` - Switch to larger model (8K tokens)
3. Keep conversations shorter with small models

**You can retry your request now.**""")
            else:
                logger.error(f"❌ Cannot auto-recover: conversation already minimal ({current_msg_count} messages)")
                error_msg = AIMessage(content=f"""❌ Context window overflow - this model is too small for your task.

**Problem:** Even a minimal conversation exceeds this model's {available if available else '?'} token limit.

**Solutions:**
1. Start a new chat session
2. `:model qwen2.5:14b` - Switch to larger model (8K context)
3. Use a model with more capacity

This model cannot handle your current workload.""")

            conversation_state["messages"].append(error_msg)

            if METRICS_AVAILABLE:
                metrics["agent_errors"] += 1
                duration = time.time() - start_time
                metrics["agent_times"].append((time.time(), duration))

            return {"messages": conversation_state["messages"]}

        raise

    except Exception as e:

        if METRICS_AVAILABLE:
            metrics["agent_errors"] += 1
            duration = time.time() - start_time
            metrics["agent_times"].append((time.time(), duration))

        error_str = str(e)

        if "model runner has unexpectedly stopped" in error_str:
            logger.error("❌ Ollama model crashed - likely out of memory")

            error_msg = AIMessage(content="""❌ Model crashed due to resource limitations.

    **Common causes:**
    - Out of memory (RAM/VRAM)
    - Model too large for your system
    - Ollama server overloaded

    **Solutions:**
    1. Restart Ollama: `ollama serve`
    2. Try a smaller model: `:model llama3.2:3b`
    3. Close other applications to free memory
    4. Check Ollama logs for details

    **Quick fix:** `:model llama3.2:3b` for a lighter model.""")

            conversation_state["messages"].append(error_msg)
            return {"messages": conversation_state["messages"], "error": "ollama_crash"}

        logger.exception(f"❌ Unexpected error in agent execution")
        error_msg = AIMessage(content=f"An error occurred: {error_str}")
        conversation_state["messages"].append(error_msg)
        return {"messages": conversation_state["messages"]}