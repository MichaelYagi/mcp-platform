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
import os
import re
import time
from typing import TypedDict, Annotated, Sequence
import requests
import urllib.parse
from html.parser import HTMLParser
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from pathlib import Path
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as _SKL_STOPS
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

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
    ROUTER_EXCLUDE_MEDIA,
    classify, QueryIntent,
    RESEARCH_SOURCE_PATTERN, extract_research_sources,
    OLLAMA_SEARCH_PATTERN, WEB_SEARCH_EXPLICIT_PATTERN
)
from prompts.prompts import (
    VISION_DESCRIBE_PLAIN,
    RAG_CONTEXT,
    WEB_SEARCH_SIMPLE,
    WEB_SEARCH_WITH_QUESTION,
    WEB_SEARCH_RESULTS,
    WEB_SEARCH_UPDATE,
    RESEARCH_SYNTHESIS,
    RESEARCH_CONDENSE,
    RESEARCH_RETRY,
)

MAX_MESSAGE_HISTORY = int(os.getenv("MAX_MESSAGE_HISTORY", "20"))
LLM_MESSAGE_WINDOW = int(os.getenv("LLM_MESSAGE_WINDOW", "6"))

# Vision keyword extraction — combined sklearn + domain-specific stopwords
_VISION_STOPS = _SKL_STOPS | frozenset({
    "image", "photo", "photograph", "scene", "captured", "captures",
    "capturing", "taken", "shows", "shown", "likely", "description",
    "detail", "detailed", "high", "angle", "clear", "filled", "focused",
    "active", "background", "foreground", "division", "complex",
})

# Try to import metrics
try:
    from metrics import metrics, new_trace, get_trace, FailureKind, MCPToolError
    METRICS_AVAILABLE = True
except ImportError:
    try:
        from client.metrics import metrics, new_trace, get_trace, FailureKind, MCPToolError
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
            "failure_kinds": defaultdict(int),
        }
        def new_trace() -> str: return ""
        def get_trace() -> str: return ""
        class FailureKind: pass
        class MCPToolError(Exception): pass


def _classify_error(e: Exception) -> "FailureKind":
    """
    Map a raw exception to a FailureKind without changing any call-site logic.
    Used at every metrics["*_errors"] increment to record a structured category.
    """
    err = str(e).lower()
    # Retryable: timeouts and rate limits
    if isinstance(e, asyncio.TimeoutError):
        return FailureKind.RETRYABLE
    if any(kw in err for kw in ("timeout", "rate limit", "429", "too many requests", "connection reset", "connection refused")):
        return FailureKind.RETRYABLE
    # User error: bad input, schema violations, context overflow
    if any(kw in err for kw in ("exceed context window", "requested tokens", "too long", "maximum context",
                                 "does not support tools", "does not support chat",
                                 "invalid", "missing required", "validation")):
        return FailureKind.USER_ERROR
    # Upstream error: external services (Ollama crash, Plex, HTTP errors)
    if any(kw in err for kw in ("model runner has unexpectedly stopped", "ollama", "httpx", "http ",
                                 "502", "503", "504", "upstream", "server error", "plex")):
        return FailureKind.UPSTREAM_ERROR
    # Default: internal
    return FailureKind.INTERNAL_ERROR


def _record_failure(kind: "FailureKind") -> None:
    """Increment the failure_kinds counter if metrics are available."""
    if METRICS_AVAILABLE and hasattr(kind, "value"):
        metrics["failure_kinds"][kind.value] += 1


async def llm_ainvoke(llm, messages, poll_interval: float = 0.5):
    """
    Cancellable wrapper around llm.ainvoke().
    Polls is_stop_requested() every poll_interval seconds and cancels
    the underlying task if a stop is requested, raising asyncio.CancelledError.
    All LLM calls in this module should use this instead of llm.ainvoke() directly.
    """
    task = asyncio.create_task(llm.ainvoke(messages))
    try:
        while not task.done():
            if is_stop_requested():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                raise asyncio.CancelledError("LLM call cancelled: stop requested")
            await asyncio.sleep(poll_interval)
        return await task
    except asyncio.CancelledError:
        if not task.done():
            task.cancel()
        raise


class AgentState(TypedDict):
    """State that gets passed between nodes in the graph"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    tools: dict
    llm: object
    ingest_completed: bool
    stopped: bool
    current_model: str
    research_source: str
    session_state: object  # SessionState | None — scoped per session, never shared
    capability_registry: object  # CapabilityRegistry | None — read-only, shared across sessions

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
    import contextvars
    ctx = contextvars.copy_context()
    with ThreadPoolExecutor(max_workers=3) as executor:
        return await loop.run_in_executor(
            executor,
            lambda: ctx.run(fetch_url_content_sync, url, timeout)
        )


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
            SystemMessage(content=RAG_CONTEXT.format(context=context)),
            user_message
        ]

        llm = state.get("llm")
        response = await llm_ainvoke(llm, augmented_messages)

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

        research_prompt = RESEARCH_SYNTHESIS.format(
            source_count=success_count,
            sources_list=sources_list,
            combined_content=combined_content,
            query=query_cleaned,
        )

        augmented_messages = state["messages"] + [HumanMessage(content=research_prompt)]

        # ATTEMPT 1: Try with full content
        try:
            response = await asyncio.wait_for(
                llm_ainvoke(llm, augmented_messages),
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

        summary_prompt = RESEARCH_CONDENSE.format(
            source_count=success_count,
            combined_content=combined_content,
            query=query_cleaned,
        )

        try:
            summary_response = await asyncio.wait_for(
                llm_ainvoke(llm, [HumanMessage(content=summary_prompt)]),
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

        retry_prompt = RESEARCH_RETRY.format(
            source_count=success_count,
            sources_list=sources_list,
            summarized_content=summarized_content,
            query=query_cleaned,
        )

        retry_messages = state["messages"] + [HumanMessage(content=retry_prompt)]

        try:
            response = await asyncio.wait_for(
                llm_ainvoke(llm, retry_messages),
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
        # OLLAMA_SEARCH_PATTERN imported from query_patterns

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

                    augmented_prompt = WEB_SEARCH_SIMPLE.format(
                        query=query,
                        search_context=search_context,
                    )

                    augmented_messages = messages + [HumanMessage(content=augmented_prompt)]
                    response = await llm_ainvoke(base_llm, augmented_messages)

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

            # ── Vision shortcut ──────────────────────────────────────────────
            # If the tool result carries image_base64, bypass the LLM message
            # loop entirely. The Ollama vision API requires images to be passed
            # in the `images` array of /api/chat — not as text content — so we
            # make a direct httpx call here and return the description as the
            # final AIMessage.
            try:
                raw = last_message.content
                # Unwrap MCP TextContent: [TextContent(type='text', text='...', annotations=None, meta=None)]
                # The repr escapes newlines as \n etc, so we scan for balanced braces
                # then decode escape sequences before JSON parsing.
                if isinstance(raw, str) and "TextContent" in raw:
                    idx = raw.find("text='")
                    if idx != -1:
                        raw = raw[idx + 6:]
                        # Scan for end of JSON object using brace depth
                        depth, end, in_str, esc = 0, -1, False, False
                        for i, ch in enumerate(raw):
                            if esc:
                                esc = False; continue
                            if ch == '\\':
                                esc = True; continue
                            if ch == '"':
                                in_str = not in_str
                            if not in_str:
                                if ch == '{': depth += 1
                                elif ch == '}':
                                    depth -= 1
                                    if depth == 0:
                                        end = i; break
                        if end != -1:
                            raw = raw[:end + 1]
                        # Decode Python repr escapes (\n \t \' etc.)
                        try:
                            raw = raw.encode('raw_unicode_escape').decode('unicode_escape')
                        except Exception:
                            pass
                    try:
                        tool_data = json.loads(raw)
                    except json.JSONDecodeError:
                        tool_data = None
                    logger.info(f"[LangGraph] 🖼️ Unwrapped TextContent, keys={list(tool_data.keys()) if isinstance(tool_data, dict) else 'parse failed'}")
                else:
                    try:
                        tool_data = json.loads(raw)
                    except json.JSONDecodeError:
                        tool_data = None
                if isinstance(tool_data, dict) and (tool_data.get("image_base64") or tool_data.get("image_source")):
                    logger.info("[LangGraph] 🖼️ Image result — calling Ollama vision directly")
                    b64 = tool_data.get("image_base64")
                    image_source  = tool_data.get("image_source")           # 225px thumbnail → UI
                    vision_source = tool_data.get("image_source_original") or image_source  # original → Ollama

                    # If no base64 payload, fetch original for Ollama vision
                    if not b64 and vision_source:
                        logger.info(f"[LangGraph] 🖼️ Fetching image from source: {vision_source}")
                        import httpx as _httpx
                        fetch_headers = {}
                        shashin_key = os.getenv("SHASHIN_API_KEY", "")
                        if shashin_key and ("192.168." in vision_source or "shashin" in vision_source.lower()):
                            fetch_headers = {"x-api-key": shashin_key, "Content-Type": "application/json"}
                        async with _httpx.AsyncClient(timeout=60.0) as hc:
                            img_resp = await hc.get(vision_source, headers=fetch_headers)
                            img_resp.raise_for_status()
                        import base64 as _b64
                        b64 = _b64.b64encode(img_resp.content).decode("utf-8")
                        logger.info(f"[LangGraph] 🖼️ Fetched {len(img_resp.content)} bytes")

                    # Strip data URI prefix if present
                    if b64 and "," in b64:
                        b64 = b64.split(",", 1)[1]

                    # placeName/takenAt may be in an earlier ToolMessage (e.g. shashin_random_tool)
                    # rather than the current one — scan all messages for it.
                    image_id = tool_data.get("image_id")
                    place    = tool_data.get("placeName")
                    taken_at = tool_data.get("takenAt")
                    logger.info(f"[LangGraph] 🖼️ place={place!r}, taken_at={taken_at!r} from tool_data")
                    if not place:
                        for m in reversed(messages):
                            if not isinstance(m, ToolMessage):
                                continue
                            try:
                                m_raw = m.content if isinstance(m.content, str) else ""
                                if "TextContent" in m_raw:
                                    idx = m_raw.find("text='")
                                    if idx == -1:
                                        continue
                                    m_raw = m_raw[idx + 6:]
                                    # brace-depth scan
                                    depth, end, in_str, esc = 0, -1, False, False
                                    for i, ch in enumerate(m_raw):
                                        if esc: esc = False; continue
                                        if ch == '\\': esc = True; continue
                                        if ch == '"': in_str = not in_str
                                        if not in_str:
                                            if ch == '{': depth += 1
                                            elif ch == '}':
                                                depth -= 1
                                                if depth == 0: end = i; break
                                    if end != -1:
                                        m_raw = m_raw[:end + 1]
                                    try:
                                        m_raw = m_raw.encode('raw_unicode_escape').decode('unicode_escape')
                                    except Exception:
                                        pass
                                m_data = json.loads(m_raw)
                                if isinstance(m_data, dict) and m_data.get("placeName"):
                                    # Only borrow place if it's from the same image
                                    if m_data.get("image_id") == image_id:
                                        place    = m_data["placeName"]
                                        taken_at = m_data.get("takenAt") or taken_at
                                        logger.info(f"[LangGraph] 🖼️ place={place!r} found in earlier ToolMessage")
                                    break
                            except Exception as e:
                                logger.debug(f"[LangGraph] 🖼️ fallback scan parse error: {e}")
                    # Extract the user's actual intent from the last HumanMessage
                    # Extract user intent from the last HumanMessage, stripping
                    # tool-invocation preamble. If the message is purely a tool call
                    # with no real question (e.g. "Use shashin_analyze_tool with <id>"),
                    # fall through to the generic description prompt.
                    import re as _re
                    _TOOL_ONLY_RE = _re.compile(
                        r'^(?:use|using|run|call)\s+\w+(?:\s+(?:with|using|on|for)?\s*[\w\-]+)?\s*(?::\s*.+)?\s*$',
                        _re.IGNORECASE
                    )
                    _PREAMBLE_RE = _re.compile(
                        r'^(?:using\s+\w+[\w_]*\s*(?:tool)?\s*[,.]?\s*)',
                        _re.IGNORECASE
                    )
                    # Residual navigation phrases left after preamble stripping —
                    # these are tool-invocation language, not real vision questions.
                    _NAV_RE = _re.compile(
                        r'^(?:show\s+me\s+(?:a\s+)?(?:random\s+)?(?:photo|picture|image|pic)\b'
                        r'|show\s+me\s+(?:this|it)\b'
                        r'|get\s+(?:a\s+)?(?:random\s+)?(?:photo|picture|image)\b'
                        r'|display\s+(?:it|this|a\s+photo|an?\s+image)\b'
                        r'|fetch\s+(?:a\s+)?(?:photo|picture|image)\b)\s*$',
                        _re.IGNORECASE
                    )
                    user_intent = None
                    for m in reversed(messages):
                        if isinstance(m, HumanMessage):
                            raw_intent = m.content if isinstance(m.content, str) else None
                            if raw_intent:
                                stripped = raw_intent.strip()
                                # Skip pure tool-call messages (e.g. "Use shashin_random_tool with <id>")
                                if _TOOL_ONLY_RE.match(stripped):
                                    break
                                # Strip tool-invocation preamble
                                cleaned = _PREAMBLE_RE.sub("", stripped, count=1).strip()
                                # Discard if what remains (or the original) is just
                                # navigation language — applies both with and without preamble
                                if cleaned and not _NAV_RE.match(cleaned):
                                    user_intent = cleaned
                            break

                    if user_intent:
                        if place:
                            user_prompt = f"{user_intent}\n\nContext: this image was taken at {place}."
                        else:
                            user_prompt = user_intent
                    elif place:
                        user_prompt = f"{VISION_DESCRIBE_PLAIN} It was taken at: {place}."
                    else:
                        user_prompt = VISION_DESCRIBE_PLAIN

                    # Use the dedicated vision model from env, falling back to current model.
                    # This allows qwen2.5:14b to handle tool selection while a separate
                    # vision model handles image inference.
                    # Get Ollama URL from the already-initialized LLM client
                    ollama_url = (
                        getattr(base_llm, "base_url", None)
                        or getattr(base_llm, "client_kwargs", {}).get("base_url")
                        or os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
                    )
                    # Strip trailing slash
                    ollama_url = str(ollama_url).rstrip("/")
                    model_name = os.getenv("OLLAMA_VISION_MODEL") or get_model_name(base_llm)
                    logger.info(f"[LangGraph] 🖼️ Using vision model: {model_name}")

                    import httpx
                    payload = {
                        "model": model_name,
                        "messages": [
                            {
                                "role": "user",
                                "content": user_prompt,
                                "images": [b64]
                            }
                        ],
                        "stream": False
                    }
                    start_time = time.time()
                    async with httpx.AsyncClient(timeout=300.0) as hc:
                        vision_resp = await hc.post(
                            f"{ollama_url}/api/chat",
                            json=payload
                        )
                        vision_resp.raise_for_status()

                    duration = time.time() - start_time
                    vision_text = vision_resp.json()["message"]["content"]
                    logger.info(f"[LangGraph] 🖼️ Vision response in {duration:.2f}s: {vision_text[:80]}")

                    # Prepend location and date/time to the chat bubble
                    header_parts = []
                    if image_id:
                        shashin_base = os.getenv("SHASHIN_BASE_URL", "http://192.168.0.199:6624")
                        header_parts.append(f"🆔 {image_id}")
                        header_parts.append(f"🔗 {shashin_base}/search?term={image_id}")
                    if place:
                        place = place.split(';', 1)[0]
                        header_parts.append(f"📍 {place}")
                    if taken_at:
                        header_parts.append(f"📅 {taken_at}")
                    if header_parts:
                        vision_text = "\n".join(header_parts) + "\n\n" + vision_text

                    # Store the description in RAG keyed by the image source path/URL
                    # so future queries about the same image skip the vision model.
                    image_source = tool_data.get("image_source")
                    if image_source:
                        try:
                            rag_add_tool = state.get("tools", {}).get("rag_add_tool")
                            if rag_add_tool:
                                await rag_add_tool.ainvoke({
                                    "text": vision_text,
                                    "source": image_source,
                                    "metadata": {
                                        "source_type": "image_description",
                                        "image_source": image_source,
                                        "timestamp": time.time()
                                    }
                                })
                                logger.info(f"[LangGraph] 🖼️ Vision description stored in RAG: {image_source}")
                        except Exception as rag_err:
                            logger.warning(f"[LangGraph] 🖼️ Failed to store vision description in RAG: {rag_err}")

                    if METRICS_AVAILABLE:
                        metrics["llm_calls"] += 1
                        metrics["llm_times"].append((time.time(), duration))

                    # ── Auto-tag: write description + keywords back to Shashin ──
                    # Only tag if Shashin has no description yet for this image.
                    existing_description = tool_data.get("description", "")
                    if image_id and not existing_description:
                        try:
                            shashin_base = os.getenv("SHASHIN_BASE_URL", "http://192.168.0.199:6624")
                            shashin_key  = os.getenv("SHASHIN_API_KEY", "")
                            tag_headers  = {"x-api-key": shashin_key, "Content-Type": "application/json"}

                            # Description: first 2 sentences from vision_text,
                            # skipping header lines (🆔 📍 📅) and bold section headers
                            desc_lines = [
                                l.strip() for l in vision_text.splitlines()
                                if l.strip()
                                and not l.strip().startswith(("🆔", "📍", "📅", "**", "*"))
                            ]
                            raw_desc = " ".join(desc_lines)
                            sentences = re.split(r'(?<=[.!?])\s+', raw_desc)
                            auto_description = " ".join(sentences[:2]).strip()
                            if len(auto_description) > 500:
                                auto_description = auto_description[:497] + "..."

                            # Keywords: prioritize subject nouns over generic words
                            _STOPWORDS = _VISION_STOPS

                            kw_candidates = {}  # word -> score

                            # Bold section headers = highest priority (score 3)
                            for bold_match in re.finditer(r'\*\*([^*:]+)', vision_text):
                                word = bold_match.group(1).strip().lower()
                                if 2 < len(word) < 30 and " " not in word and word not in _STOPWORDS:
                                    kw_candidates[word] = kw_candidates.get(word, 0) + 3

                            # All words from description, scored by importance
                            words = re.findall(r'\b[a-zA-Z][a-zA-Z\-]{2,}\b', auto_description)
                            for word in words:
                                w = word.lower()
                                if w in _STOPWORDS:
                                    continue
                                score = kw_candidates.get(w, 0)
                                # Boost short meaningful nouns (pool, band, stage, choir)
                                if len(w) <= 6:
                                    score += 2
                                # Boost longer subject nouns (orchestra, swimming, musician)
                                elif len(w) <= 10:
                                    score += 1
                                # Penalize very long words that are usually adjectives/adverbs
                                else:
                                    score += 0
                                kw_candidates[w] = score

                            # Also scan full vision_text for capitalized subject words (proper nouns / subjects)
                            for word in re.findall(r'\b[A-Z][a-z]{2,}\b', vision_text):
                                w = word.lower()
                                if w not in _STOPWORDS:
                                    kw_candidates[w] = kw_candidates.get(w, 0) + 2

                            # Sort by score descending, take top 10
                            auto_keywords = ",".join(
                                w for w, _ in sorted(kw_candidates.items(), key=lambda x: -x[1])[:10]
                            )

                            async with httpx.AsyncClient(timeout=15.0) as hc:
                                if auto_description:
                                    desc_resp = await hc.put(
                                        f"{shashin_base}/api/v1/update/metadata/description/{image_id}",
                                        headers=tag_headers,
                                        json={"description": auto_description}
                                    )
                                    if desc_resp.status_code == 200:
                                        logger.info(f"[LangGraph] 🏷️ Auto-tagged description for {image_id}")
                                    else:
                                        logger.warning(f"[LangGraph] 🏷️ Description PUT failed: {desc_resp.status_code} — {desc_resp.text[:200]}")
                                if auto_keywords:
                                    kw_resp = await hc.put(
                                        f"{shashin_base}/api/v1/update/metadata/keywords/{image_id}",
                                        headers=tag_headers,
                                        json={"keywords": auto_keywords}
                                    )
                                    if kw_resp.status_code == 200:
                                        logger.info(f"[LangGraph] 🏷️ Auto-tagged keywords for {image_id}: {auto_keywords}")
                                    else:
                                        logger.warning(f"[LangGraph] 🏷️ Keywords PUT failed: {kw_resp.status_code} — {kw_resp.text[:200]}")
                        except Exception as tag_err:
                            logger.warning(f"[LangGraph] 🏷️ Auto-tag failed for {image_id}: {tag_err}")
                    elif image_id and existing_description:
                        logger.info(f"[LangGraph] 🏷️ Skipping auto-tag for {image_id} — description already exists")
                    # ── End auto-tag ─────────────────────────────────────────────

                    return {
                        "messages": [AIMessage(content=vision_text)],
                        "tools": state.get("tools", {}),
                        "llm": state.get("llm"),
                        "ingest_completed": state.get("ingest_completed", False),
                        "stopped": state.get("stopped", False),
                        "current_model": model_name
                    }
            except (json.JSONDecodeError, AttributeError, KeyError):
                pass
            except Exception as vision_err:
                logger.error(f"[LangGraph] 🖼️ Vision call failed: {vision_err} — falling back to LLM")
            # ── End vision shortcut ──────────────────────────────────────────

            # For search tools that return plain text lists, bypass the LLM
            # entirely and return the tool output directly as the AI response.
            last_tool_name = last_message.name if hasattr(last_message, "name") else ""
            if last_tool_name == "shashin_search_tool":
                raw_content = last_message.content
                if isinstance(raw_content, str) and "TextContent" in raw_content:
                    idx = raw_content.find("text='")
                    if idx != -1:
                        raw_content = raw_content[idx + 6:]
                        end = raw_content.rfind("'")
                        if end != -1:
                            raw_content = raw_content[:end]
                        raw_content = raw_content.replace("\\n", "\n").replace("\\'", "'")
                return {
                    "messages": [AIMessage(content=raw_content)],
                    "tools": state.get("tools", {}),
                    "llm": state.get("llm"),
                    "ingest_completed": state.get("ingest_completed", False),
                    "stopped": state.get("stopped", False),
                    "current_model": get_model_name(base_llm)
                }

            # ── web_image_search_tool shortcut ───────────────────────────────
            # Parse the JSON result, extract image_url, and return a plain
            # AIMessage. The websocket scanner will pick up image_url from the
            # ToolMessage and send it to the frontend for inline rendering.
            if last_tool_name == "web_image_search_tool":
                raw_content = last_message.content if isinstance(last_message.content, str) else ""
                # Unwrap MCP TextContent repr if needed
                if "TextContent" in raw_content:
                    idx = raw_content.find("text='")
                    if idx != -1:
                        raw_content = raw_content[idx + 6:]
                        end = raw_content.rfind("'")
                        if end != -1:
                            raw_content = raw_content[:end]
                        raw_content = raw_content.replace("\\n", "\n").replace("\\'", "'")

                # Images are now inline markdown ![title](url) in the list —
                # formatMessage in index.js renders them as <img> tags.
                # Just pass the text through directly.
                if raw_content and not raw_content.startswith("{"):
                    reply_text = raw_content
                else:
                    reply_text = "No images found. Try rephrasing or using a more specific name."

                return {
                    "messages": [AIMessage(content=reply_text)],
                    "tools": state.get("tools", {}),
                    "llm": state.get("llm"),
                    "ingest_completed": state.get("ingest_completed", False),
                    "stopped": state.get("stopped", False),
                    "current_model": get_model_name(base_llm)
                }
            # ── End web_image_search_tool shortcut ───────────────────────────

            start_time = time.time()
            try:
                response = await asyncio.wait_for(
                    llm_ainvoke(base_llm, messages),
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
                _record_failure(FailureKind.RETRYABLE)
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
                _record_failure(_classify_error(e))
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

            should_use_web_search = bool(WEB_SEARCH_EXPLICIT_PATTERN.search(user_lower))

            if should_use_web_search:
                logger.info("[LangGraph] 🎯 FORCED WEB SEARCH: User explicitly requested web search")

                search_client = get_search_client()

                if search_client.is_available():
                    query = user_message
                    query = WEB_SEARCH_EXPLICIT_PATTERN.sub('', query).strip()
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

                            augmented_prompt = WEB_SEARCH_WITH_QUESTION.format(
                                search_context=search_context,
                                user_message=user_message,
                            )

                            augmented_messages = messages + [HumanMessage(content=augmented_prompt)]
                            response = await llm_ainvoke(base_llm, augmented_messages)

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
        def _filter_tools(all_tools: list, tool_patterns: list) -> list:
            """Filter tool list to those matching name patterns. Supports 'prefix*' wildcards."""
            result = []
            for tool in all_tools:
                if not hasattr(tool, 'name'):
                    continue
                for pattern in tool_patterns:
                    if "*" in pattern:
                        if tool.name.startswith(pattern.replace("*", "")):
                            result.append(tool)
                            break
                    elif tool.name == pattern:
                        result.append(tool)
                        break
            return result

        def _get_tool_meta(tool):
            """Extract __tool_meta__ from a tool, checking tool.metadata first
            (populated by CapabilityRegistry from @tool_meta), then unwrapping
            decorator layers as fallback for locally-defined tools."""
            # Primary: CapabilityRegistry stores triggers/tags/intent_category on metadata
            if hasattr(tool, "metadata") and tool.metadata:
                m = tool.metadata
                if m.get("triggers") or m.get("tags"):
                    return m
            # Fallback: walk decorator chain for __tool_meta__ attribute
            fn = getattr(tool, "func", None) or getattr(tool, "_func", None) or tool
            meta = getattr(fn, "__tool_meta__", None)
            if meta is None:
                inner = getattr(fn, "func", None) or getattr(fn, "_func", None)
                if inner and inner is not fn:
                    meta = getattr(inner, "__tool_meta__", None)
            return meta

        def match_intent(user_message: str, all_tools: list, base_llm, logger, conversation_state,
                         capability_registry=None):
            """
            Route query to the right tool subset using @tool_meta triggers.
            @tool_meta is the single source of truth — query_patterns.py is not used.

            Routing priority:
            1. Active code project context → bind code-tagged tools
            2. @tool_meta trigger match — substring match against user message
            3. General fallback — bind 0 tools, LLM answers from context
            """
            msg_lower = user_message.lower()

            # ── 1. Project context override ───────────────────────────────────
            import re as _re
            _CODE_PATH_RE = _re.compile(
                r'Active Project:.*\.(py|js|ts|jsx|tsx|go|rs|java|kt|cs|cpp|c|rb|php)'
                r'|Active Project:.*(src|lib|app|packages?|node_modules|venv)',
                _re.IGNORECASE
            )
            for msg in reversed(conversation_state.get("messages", [])[-5:]):
                if isinstance(msg, SystemMessage) and "Active Project:" in msg.content:
                    if _CODE_PATH_RE.search(msg.content):
                        logger.info("[LangGraph] 🎯 Found code project context → forcing code tools")
                        if capability_registry:
                            filtered = [
                                t for t in all_tools
                                if hasattr(t, "name") and (
                                    cap := capability_registry.get_tool(t.name)
                                ) and "code" in (cap.tags if cap else [])
                            ]
                        else:
                            filtered = [
                                t for t in all_tools
                                if (_get_tool_meta(t) or {}).get("tags") and
                                "code" in (_get_tool_meta(t) or {}).get("tags", [])
                            ]
                        if filtered:
                            logger.info(f"   → {len(filtered)} code tools")
                            return base_llm.bind_tools(filtered), "code_assistant"
                    else:
                        logger.info("[LangGraph] 📁 Project context not a code path — skipping")
                    break

            # ── 2. @tool_meta trigger matching ────────────────────────────────
            # Group matched tools by intent_category so related tools are bound together
            # e.g. "weather" matches get_weather_tool AND get_location_tool
            category_tools: dict[str, list] = {}
            for tool in all_tools:
                meta = _get_tool_meta(tool)
                if not meta:
                    continue
                triggers = meta.get("triggers", [])
                category = meta.get("intent_category") or (meta.get("tags") or ["general"])[0]
                for trigger in triggers:
                    if trigger.lower() in msg_lower:
                        if category not in category_tools:
                            category_tools[category] = []
                        category_tools[category].append(tool)
                        logger.info(f"🎯 trigger match: '{trigger}' → {tool.name} [{category}]")
                        break

            if category_tools:
                # Pick the category with the most specific (longest) trigger match
                best_category = max(category_tools, key=lambda c: len(category_tools[c]))
                matched = category_tools[best_category]
                # Also include same-category tools that didn't trigger but share the category
                for tool in all_tools:
                    if tool in matched:
                        continue
                    meta = _get_tool_meta(tool)
                    if not meta:
                        continue
                    cat = meta.get("intent_category") or (meta.get("tags") or ["general"])[0]
                    if cat == best_category:
                        matched.append(tool)
                logger.info(f"🎯 tool_meta → {best_category}: {[t.name for t in matched[:5]]}")
                return base_llm.bind_tools(matched), best_category

            # ── 3. General fallback ───────────────────────────────────────────
            logger.info("🎯 General query → binding 0 tools")
            return base_llm.bind_tools([]), "general"

        # Apply pattern matching
        if user_message:
            all_tools = list(state.get("tools", {}).values())
            _cap_reg = state.get("capability_registry")
            llm_to_use, pattern_name = match_intent(
                user_message,
                all_tools,
                base_llm,
                logger,
                state,
                capability_registry=_cap_reg
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
            _record_failure(_classify_error(e))
            logger.error(f"❌ Model call failed: {e}")

            # Catch timeout and cancellation — return a clean message instead of
            # letting the ugly traceback bubble up to the user
            if isinstance(e, (TimeoutError, asyncio.TimeoutError, asyncio.CancelledError)):
                error_response = AIMessage(content=(
                    f"⏱️ **Model timed out** — `{current_model}` took too long to respond.\n\n"
                    f"This usually means the model is too large for your hardware on this query.\n\n"
                    f"**Options:**\n"
                    f"• Switch to a faster model: `:model llama3.2:3b` or `:model qwen2.5:7b`\n"
                    f"• Use explicit dispatch to bypass the LLM: `use get_weather_tool`"
                ))
                return {
                    "messages": [error_response],
                    "tools": state.get("tools", {}),
                    "llm": state.get("llm"),
                    "ingest_completed": state.get("ingest_completed", False),
                    "stopped": state.get("stopped", False),
                    "current_model": current_model
                }

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
                        augmented_prompt = WEB_SEARCH_RESULTS.format(
                            search_context=search_context,
                        )

                        retry_messages = messages + [HumanMessage(content=augmented_prompt)]
                        response = await llm_ainvoke(base_llm, retry_messages)
                    else:
                        logger.warning("⚠️ Web search failed - using base LLM")
                        response = await asyncio.wait_for(
                            llm_ainvoke(base_llm, messages),
                            timeout=300.0
                        )
                else:
                    logger.warning("⚠️ Web search unavailable - using base LLM")
                    response = await asyncio.wait_for(
                        llm_ainvoke(base_llm, messages),
                        timeout=300.0
                    )

                current_model = get_model_name(base_llm)

            elif not has_tool_calls and has_content:

                # If intent was analyze_image, shashin_analyze, or an explicit tool
                # request, but LLM answered from context instead of calling the tool,
                # force the tool call.
                if pattern_name in ("analyze_image", "shashin_analyze", "explicit_tool"):
                    import re as _img_re, uuid as _uuid
                    _file_re = _img_re.compile(
                        r'([A-Za-z]:[/\\][^\s]+\.(?:jpg|jpeg|png|gif|webp|bmp|heic)'  # Windows C:\... or C:/...
                        r'|(?:/mnt/|/home/|/tmp/|/var/|/Users/|~/)[^\s]+\.(?:jpg|jpeg|png|gif|webp|bmp|heic))',  # WSL/Linux/Mac/tilde
                        _img_re.IGNORECASE
                    )
                    _url_re = _img_re.compile(
                        r'(https?://[^\s]+\.(?:jpg|jpeg|png|gif|webp|bmp|heic))',
                        _img_re.IGNORECASE
                    )
                    _uuid_re = _img_re.compile(
                        r'\b([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b',
                        _img_re.IGNORECASE
                    )
                    file_match = _file_re.search(user_message or "")
                    url_match = _url_re.search(user_message or "")
                    uuid_match = _uuid_re.search(user_message or "")

                    forced_tool = None
                    forced_args = None
                    if file_match:
                        forced_tool = "analyze_image_tool"
                        forced_args = {"image_file_path": file_match.group(1)}
                    elif url_match:
                        forced_tool = "analyze_image_tool"
                        forced_args = {"image_url": url_match.group(1)}
                    elif uuid_match:
                        forced_tool = "shashin_analyze_tool"
                        forced_args = {"image_id": uuid_match.group(1)}

                    if forced_tool and forced_args:
                        import uuid as _uuid2
                        tool_call_id = str(_uuid2.uuid4())
                        forced = AIMessage(
                            content="",
                            tool_calls=[{
                                "id": tool_call_id,
                                "name": forced_tool,
                                "args": forced_args
                            }]
                        )
                        logger.info(f"[LangGraph] 🖼️ Forced {forced_tool} call: {forced_args}")
                        return {
                            "messages": [forced],
                            "tools": state.get("tools", {}),
                            "llm": state.get("llm"),
                            "ingest_completed": state.get("ingest_completed", False),
                            "stopped": state.get("stopped", False),
                            "current_model": current_model
                        }

                # ── Knowledge-gap detection via confidence check ──────────────
                # Ask the model directly whether it had reliable knowledge to
                # answer. One tiny call (YES/NO output) is more accurate than
                # any phrase list — catches confident-sounding wrong answers and
                # works regardless of language or phrasing.
                # Only fires for general queries to avoid interfering with
                # tool-routed intents (weather, RAG, Plex, etc.).
                _is_hedging = False
                if pattern_name == "general" and user_message:
                    try:
                        _confidence_check = await llm_ainvoke(base_llm, [
                            SystemMessage(content="Reply with only YES or NO. No other text."),
                            HumanMessage(content=(
                                f"Did you have reliable, specific knowledge to answer "
                                f"this question accurately?\n"
                                f"Question: {user_message}\n"
                                f"Your answer: {response.content[:300]}"
                            ))
                        ])
                        _is_hedging = _confidence_check.content.strip().upper().startswith("N")
                        logger.info(f"[LangGraph] 🔎 Confidence check: {'LOW — will search' if _is_hedging else 'OK'}")
                    except Exception as _cc_err:
                        logger.warning(f"[LangGraph] ⚠️ Confidence check failed: {_cc_err}")

                if _is_hedging:
                    logger.info("[LangGraph] 🔍 Low confidence detected — attempting web search fallback")
                    _search_client = get_search_client()
                    if _search_client.is_available():
                        try:
                            _search_result = await _search_client.search(user_message)
                            if _search_result.get("success") and _search_result.get("results"):
                                logger.info("[LangGraph] ✅ Web search fallback succeeded — retrying with context")
                                _augmented = WEB_SEARCH_WITH_QUESTION.format(
                                    search_context=_search_result["results"],
                                    user_message=user_message,
                                )
                                response = await llm_ainvoke(base_llm,
                                    messages + [HumanMessage(content=_augmented)]
                                )
                                current_model = get_model_name(base_llm)
                            else:
                                logger.warning("[LangGraph] ⚠️ Web search fallback returned no results")
                        except Exception as _hedge_err:
                            logger.warning(f"[LangGraph] ⚠️ Web search fallback failed: {_hedge_err}")
                    else:
                        logger.warning("[LangGraph] ⚠️ Web search unavailable for hedge fallback")

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
            _record_failure(FailureKind.RETRYABLE)
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
            _record_failure(_classify_error(e))
            logger.error(f"❌ Model call failed: {e}")

            # Catch timeout and cancellation — return a clean message instead of
            # letting the ugly traceback bubble up to the user
            if isinstance(e, (TimeoutError, asyncio.TimeoutError, asyncio.CancelledError)):
                error_response = AIMessage(content=(
                    f"⏱️ **Model timed out** — `{current_model}` took too long to respond.\n\n"
                    f"This usually means the model is too large for your hardware on this query.\n\n"
                    f"**Options:**\n"
                    f"• Switch to a faster model: `:model llama3.2:3b` or `:model qwen2.5:7b`\n"
                    f"• Use explicit dispatch to bypass the LLM: `use get_weather_tool`"
                ))
                return {
                    "messages": [error_response],
                    "tools": state.get("tools", {}),
                    "llm": state.get("llm"),
                    "ingest_completed": state.get("ingest_completed", False),
                    "stopped": state.get("stopped", False),
                    "current_model": current_model
                }

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
        session_state = state.get("session_state")

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

            # Coerce args that weak models pass as empty dicts {} or empty strings "".
            # Pydantic rejects these for int/bool fields before the function body runs.
            # Replace with None so Optional fields use their default values.
            for k, v in list(tool_args.items()):
                if (isinstance(v, dict) and len(v) == 0) or v == "":
                    tool_args[k] = None
                    tool_call["args"][k] = None

            # Auto-inject location into get_time_tool when model omits it.
            # Calls get_location_tool first (uses CLIENT_IP env var on the server)
            # and populates city/country from the result.
            if tool_name == "get_time_tool" and not any([
                tool_args.get("city"), tool_args.get("state"), tool_args.get("country")
            ]):
                try:
                    loc_tool = tools_dict.get("get_location_tool") if (tools_dict := state.get("tools", {})) else None
                    if loc_tool:
                        logger.info("🔧 AUTO-FIX: get_time_tool missing location — pre-calling get_location_tool")
                        loc_result = await loc_tool.ainvoke({})
                        import json as _json
                        loc_data = _json.loads(loc_result) if isinstance(loc_result, str) else loc_result
                        city = loc_data.get("city")
                        country = loc_data.get("country")
                        if city:
                            tool_args["city"] = city
                            tool_call["args"]["city"] = city
                        if country:
                            tool_args["country"] = country
                            tool_call["args"]["country"] = country
                        logger.info(f"🔧 AUTO-FIX: Injected location → city={city}, country={country}")
                except Exception as _le:
                    logger.warning(f"🔧 AUTO-FIX: get_location_tool pre-call failed: {_le}")

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

            # Check if tool is disabled before invoking — avoids a round-trip
            # to the server which would just return a disabled_tool_response JSON.
            try:
                from tools.tool_control import is_tool_enabled
                # Resolve server name for category check
                _tool_server = None
                try:
                    _meta = tool.metadata if hasattr(tool, "metadata") and tool.metadata else {}
                    _tool_server = _meta.get("source_server")
                except Exception:
                    pass
                if not is_tool_enabled(tool_name) and not is_tool_enabled(tool_name, _tool_server):
                    logger.warning(f"🚫 Tool '{tool_name}' is disabled — skipping execution")
                    disabled_msg = ToolMessage(
                        content=f"Tool '{tool_name}' is disabled. Check DISABLED_TOOLS in .env.",
                        tool_call_id=tool_id,
                        name=tool_name
                    )
                    tool_messages.append(disabled_msg)
                    continue
            except ImportError:
                pass

            try:
                tool_start = time.time()
                # Enrich args with session context (e.g. last_file_path) when missing
                if session_state:
                    tool_args = session_state.inject_into_args(tool_name, tool_args)
                result = await tool.ainvoke(tool_args)
                tool_duration = time.time() - tool_start

                if METRICS_AVAILABLE:
                    metrics["tool_calls"][tool_name] += 1
                    metrics["tool_times"][tool_name].append((time.time(), tool_duration))

                if isinstance(result, list) and len(result) > 0:
                    first = result[0]
                    logger.info(f"[LangGraph] 🔧 Tool result item type: {type(first).__name__}, attrs: {[a for a in dir(first) if not a.startswith('_')][:8]}")
                    if hasattr(first, 'text'):
                        result = first.text
                    elif hasattr(first, 'content'):
                        result = first.content
                    else:
                        # str(TextContent(...)) gives us the repr — extract the text value
                        joined = str(first)
                        idx = joined.find("text='")
                        if idx != -1:
                            extracted = joined[idx + 6:]
                            # find closing quote before ', annotations= or end of string
                            end_q = extracted.find("', ")
                            result = extracted[:end_q] if end_q != -1 else extracted.rstrip("')")
                        else:
                            result = joined

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

                # Update session state from successful tool result
                if session_state:
                    try:
                        result_dict = json.loads(result) if isinstance(result, str) else result
                        if isinstance(result_dict, dict):
                            session_state.update_from_tool_result(tool_name, result_dict)
                    except Exception:
                        pass

                result_msg = ToolMessage(
                    content=str(result),
                    tool_call_id=tool_id,
                    name=tool_name
                )
                tool_messages.append(result_msg)
                logger.info(f"✅ Tool {tool_name} completed in {tool_duration:.2f}s")
                logger.info(f"✅ ToolMessage result: {str(result)[:30]}")
                from client.health import record_tool_call
                record_tool_call(tool_name, tool_duration)

            except MCPToolError as e:
                # Authoritative kind from the tool itself — no inference needed
                logger.error(f"❌ Tool {tool_name} failed [{e.kind.value}]: {e.message}")
                from client.health import record_tool_call
                record_tool_call(tool_name, 0.0, error=e.message)
                if METRICS_AVAILABLE:
                    metrics["tool_errors"][tool_name] += 1
                _record_failure(e.kind)
                error_msg = ToolMessage(
                    content=f"Error: {e.message}",
                    tool_call_id=tool_id,
                    name=tool_name
                )
                tool_messages.append(error_msg)

            except Exception as e:
                # Fallback inference for tools not yet migrated to MCPToolError
                logger.error(f"❌ Tool {tool_name} failed: {e}")
                from client.health import record_tool_call
                record_tool_call(tool_name, 0.0, error=str(e))
                if METRICS_AVAILABLE:
                    metrics["tool_errors"][tool_name] += 1
                _record_failure(_classify_error(e))
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
            "current_model": state.get("current_model"),
            "session_state": session_state,
            "capability_registry": state.get("capability_registry")
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
async def run_agent(agent, conversation_state, user_message, logger, tools, system_prompt, llm=None, max_history=20, session_state=None, capability_registry=None):
    """
    Execute the agent with the given user message and track metrics

    CONVERSATION HISTORY DESIGN:
    - The SystemMessage in conversation_state["messages"][0] is the source of truth
    - We preserve it through truncation and never recreate it
    - If no SystemMessage exists, we create one from system_prompt parameter
    """
    start_time = time.time()
    clear_stop()
    trace_id = new_trace()
    logger.info(f"[LangGraph] ✅ Stop signal cleared for new request")

    try:
        if METRICS_AVAILABLE:
            metrics["agent_runs"] += 1

        # Extract session_id from conversation_state (set by websocket.py before calling run_agent)
        session_id = conversation_state.get("session_id")

        # STEP 1: Save the original SystemMessage (if it exists)
        original_system_msg = None
        has_system_msg = (
                conversation_state["messages"]
                and isinstance(conversation_state["messages"][0], SystemMessage)
        )

        if has_system_msg:
            original_system_msg = conversation_state["messages"][0]
            # Inject session_id into existing SystemMessage if not already present
            if session_id and f"Current session ID: {session_id}" not in original_system_msg.content:
                original_system_msg = SystemMessage(
                    content=original_system_msg.content
                    + f"\n\nCurrent session ID: {session_id}"
                )
                conversation_state["messages"][0] = original_system_msg
            logger.info("[LangGraph] Preserving existing SystemMessage")
        else:
            session_note = f"\n\nCurrent session ID: {session_id}" if session_id else ""
            original_system_msg = SystemMessage(content=system_prompt + session_note)
            conversation_state["messages"].insert(0, original_system_msg)
            logger.info("[LangGraph] Created new SystemMessage from parameter")

        # STEP 2: Add user message
        conversation_state["messages"].append(HumanMessage(content=user_message))

        # Build LLM-only message list: system prompt + RAG context + last N turns
        # MAX_MESSAGE_HISTORY is for the UI only — LLM_MESSAGE_WINDOW controls what the LLM sees
        system_msg = conversation_state["messages"][0]

        rag_context_msgs = [
            msg for msg in conversation_state["messages"][1:]
            if isinstance(msg, SystemMessage)
        ]

        non_system_msgs = [
            msg for msg in conversation_state["messages"][1:]
            if not isinstance(msg, SystemMessage)
        ]

        # ── STEP 2a: Ingest overflow conversation turns into RAG ──────────────
        # Turns older than the window are dropped from LLM context but stored
        # in RAG so they remain retrievable via semantic search.
        tool_registry_pre = {tool.name: tool for tool in tools}
        rag_add_tool = tool_registry_pre.get("rag_add_tool")

        overflow_turns = (
            non_system_msgs[:-LLM_MESSAGE_WINDOW]
            if len(non_system_msgs) > LLM_MESSAGE_WINDOW
            else []
        )

        if overflow_turns and rag_add_tool:
            ingested_overflow = 0
            i = 0
            while i < len(overflow_turns):
                turn = overflow_turns[i]
                if isinstance(turn, HumanMessage):
                    human_text = turn.content
                    ai_text = ""
                    if (
                        i + 1 < len(overflow_turns)
                        and isinstance(overflow_turns[i + 1], AIMessage)
                    ):
                        ai_text = overflow_turns[i + 1].content
                        i += 2
                    else:
                        i += 1
                    chunk = f"User: {human_text}"
                    if ai_text:
                        chunk += f"\nAssistant: {ai_text}"
                    try:
                        await rag_add_tool.ainvoke({
                            "text": chunk,
                            "source": (
                                f"conversation_history_{session_id}"
                                if session_id
                                else "conversation_history"
                            ),
                        })
                        ingested_overflow += 1
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to ingest overflow turn to RAG: {e}")
                else:
                    i += 1
            if ingested_overflow:
                logger.info(
                    f"💾 Ingested {ingested_overflow} overflow conversation turn(s) into RAG"
                )

        # ── STEP 2b: Auto-retrieve relevant context from RAG ─────────────────
        # Semantic search on every message so older turns (and any other
        # ingested content) surface even when outside the window.
        rag_search_tool_auto = tool_registry_pre.get("rag_search_tool")
        auto_rag_msg = None

        if rag_search_tool_auto:
            try:
                rag_result = await rag_search_tool_auto.ainvoke({"query": user_message})
                if isinstance(rag_result, str):
                    import json as _json
                    try:
                        rag_data = _json.loads(rag_result)
                    except Exception:
                        rag_data = {}
                else:
                    rag_data = rag_result if isinstance(rag_result, dict) else {}

                rag_results = rag_data.get("results", [])
                if rag_results:
                    rag_lines = []
                    for r in rag_results[:5]:
                        text = r.get("text", "").strip()
                        source = r.get("source", "")
                        if text:
                            source_note = f" [source: {source}]" if source else ""
                            rag_lines.append(f"• {text}{source_note}")
                    if rag_lines:
                        rag_context = (
                            "Relevant context from memory:\n" + "\n".join(rag_lines)
                        )
                        auto_rag_msg = SystemMessage(content=rag_context)
                        logger.info(
                            f"🔍 Auto-RAG: injected {len(rag_lines)} chunk(s) for query"
                        )
            except Exception as e:
                logger.warning(f"⚠️ Auto-RAG search failed: {e}")

        # Merge auto-RAG result with any existing RAG context messages
        if auto_rag_msg:
            rag_context_msgs = [auto_rag_msg] + rag_context_msgs

        # LLM sees: system prompt + RAG injections + last LLM_MESSAGE_WINDOW messages
        llm_messages = [system_msg] + rag_context_msgs + non_system_msgs[-LLM_MESSAGE_WINDOW:]
        logger.info(
            f"🧠 LLM context: {len(llm_messages)} messages "
            f"(window={LLM_MESSAGE_WINDOW}, rag={len(rag_context_msgs)})")

        # STEP 3: Run the agent
        logger.info(f"🧠 Starting agent with {len(llm_messages)} messages")

        tool_registry = {tool.name: tool for tool in tools}

        # Hits the LLM
        # Message contains
        #   SystemMessage — system prompt / tool usage guide
        #   Previous conversation history (truncated to max_history, default 20)
        #   The new HumanMessage with the user's input (appended just before in Step 2)
        result = await agent.ainvoke({
            "messages": llm_messages,
            "tools": tool_registry,
            "llm": llm,
            "ingest_completed": False,
            "stopped": False,
            "current_model": "unknown",
            "research_source": "web",
            "session_state": session_state,
            "capability_registry": capability_registry
        })

        # STEP 4: Update conversation state
        # result["messages"] only contains llm_messages (the windowed slice).
        # We must NOT store it back wholesale or history is lost.
        # Instead: find new messages added by LangGraph and append them
        # to the FULL conversation_state history.
        #
        # llm_messages was: [system] + [rag...] + [last N non-system]
        # result["messages"] is that same list + new AI response(s)
        # We only want the net-new AIMessage(s) at the tail.
        result_msgs = result["messages"]
        # Identify truly new messages: anything in result not in llm_messages
        truly_new = result_msgs[len(llm_messages):]
        logger.info(f"📨 Agent added {len(truly_new)} new messages")
        # Append new messages to FULL history (not the windowed slice)
        # Strip any RAG SystemMessages from the full history — they are
        # ephemeral per-query injections, not permanent conversation turns.
        conversation_state["messages"] = [
            msg for msg in conversation_state["messages"]
            if not (isinstance(msg, SystemMessage) and msg is not conversation_state["messages"][0])
        ] + truly_new

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
            _record_failure(FailureKind.USER_ERROR)

            return {"messages": conversation_state["messages"]}

        raise

    except Exception as e:

        if METRICS_AVAILABLE:
            metrics["agent_errors"] += 1
            duration = time.time() - start_time
            metrics["agent_times"].append((time.time(), duration))
        _record_failure(_classify_error(e))

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