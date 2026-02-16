"""
Ollama Web Search Client
Provides web search and URL fetching via Ollama's API.

API docs: https://docs.ollama.com/capabilities/web-search

Two endpoints:
  POST https://ollama.com/api/web_search  – search query → results[]
  POST https://ollama.com/api/web_fetch   – URL → title, content, links[]

Auth: OLLAMA_TOKEN environment variable (free Ollama account required).
"""
import os
import httpx
import logging
from typing import Optional, Dict, Any


class SearchClient:
    """
    Web search and URL fetch client backed by Ollama's cloud API.

    Exposes three methods used throughout the codebase:
      - is_available()  → True when OLLAMA_TOKEN is set
      - search(query)   → POST /api/web_search  (returns structured results)
      - fetch_url(url)  → POST /api/web_fetch   (returns clean markdown content)

    The get_search_client() factory keeps the original name so that
    existing call-sites in langgraph.py require no changes.
    """

    SEARCH_ENDPOINT = "https://ollama.com/api/web_search"
    FETCH_ENDPOINT  = "https://ollama.com/api/web_fetch"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OLLAMA_TOKEN", "").strip()
        self.logger  = logging.getLogger("mcp_client")

    # ------------------------------------------------------------------ #
    # Public interface                                                      #
    # ------------------------------------------------------------------ #

    def is_available(self) -> bool:
        """Return True when an API key is configured."""
        return bool(self.api_key)

    async def search(self, query: str, max_results: int = 5,
                     timeout: float = 60.0) -> Dict[str, Any]:
        """
        Perform a web search via Ollama's web_search API.

        Returns a dict consumed by langgraph.py's URL-extraction logic:

            {
                "success": bool,
                "results": <structured dict or str>,   # on success
                "raw_response": <raw JSON>,            # on success
                "error": str,                          # on failure
            }

        The "results" value is a dict with a "webPages.value" list so
        the existing URL-extraction logic in langgraph.py still works:

            results["webPages"]["value"] → list of {url, name, summary}
        """
        if not self.is_available():
            self.logger.warning("🔍 Ollama web search: OLLAMA_TOKEN not set")
            return {
                "success": False,
                "error": "Ollama API key not configured. Set OLLAMA_TOKEN."
            }

        self.logger.info(f"🔍 Ollama web search: '{query[:100]}'")

        payload = {
            "query": query,
            "max_results": min(max(1, max_results), 10)   # API cap is 10
        }

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    self.SEARCH_ENDPOINT,
                    headers=self._headers(),
                    json=payload
                )

            error = self._check_status(response)
            if error:
                return {"success": False, "error": error}

            data = response.json()
            raw_results = data.get("results", [])

            self.logger.info(
                f"✅ Ollama web search returned {len(raw_results)} result(s)"
            )

            # ── Normalise into the Bing-style shape langgraph.py already
            #    knows how to parse (webPages.value list).              ──
            normalised = {
                "webPages": {
                    "value": [
                        {
                            "url":     r.get("url", ""),
                            "name":    r.get("title", "Untitled"),
                            "summary": r.get("content", ""),
                        }
                        for r in raw_results
                        if isinstance(r, dict)
                    ]
                }
            }

            return {
                "success":      True,
                "results":      normalised,
                "raw_response": data,
            }

        except httpx.TimeoutException:
            self.logger.error("❌ Ollama web search: request timed out")
            return {"success": False, "error": "Ollama web search request timed out"}
        except httpx.HTTPError as exc:
            self.logger.error(f"❌ Ollama web search HTTP error: {exc}")
            return {"success": False, "error": f"HTTP error: {exc}"}
        except Exception as exc:
            self.logger.error(f"❌ Ollama web search unexpected error: {exc}")
            return {"success": False, "error": str(exc)}

    async def fetch_url(self, url: str,
                        timeout: float = 30.0) -> Dict[str, Any]:
        """
        Fetch a single web page via Ollama's web_fetch API.

        Returns:
            {
                "success": bool,
                "title":   str,
                "content": str,
                "links":   list[str],
                "error":   str   (only on failure)
            }

        This is used by search_and_fetch_source() in langgraph.py as a
        higher-quality alternative to the raw requests-based fetcher —
        Ollama's API returns clean markdown-ified content rather than raw
        HTML, so no HTMLTextExtractor pass is needed.
        """
        if not self.is_available():
            return {"success": False, "error": "OLLAMA_TOKEN not set"}

        self.logger.info(f"🌐 Ollama web fetch: {url}")

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    self.FETCH_ENDPOINT,
                    headers=self._headers(),
                    json={"url": url}
                )

            error = self._check_status(response)
            if error:
                return {"success": False, "error": error}

            data = response.json()
            content = data.get("content", "")

            if not content or len(content) < 50:
                return {"success": False, "error": "No content returned"}

            # Truncate very large pages to avoid context overflow
            if len(content) > 10_000:
                content = content[:10_000] + "\n\n[Content truncated…]"

            self.logger.info(
                f"✅ Ollama web fetch: '{data.get('title', 'Untitled')}' "
                f"({len(content)} chars)"
            )

            return {
                "success": True,
                "title":   data.get("title", "Untitled"),
                "content": content,
                "links":   data.get("links", []),
            }

        except httpx.TimeoutException:
            return {"success": False, "error": "Ollama web fetch timed out"}
        except httpx.HTTPError as exc:
            return {"success": False, "error": f"HTTP error: {exc}"}
        except Exception as exc:
            self.logger.error(f"❌ Ollama web fetch error: {exc}")
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------ #
    # Private helpers                                                       #
    # ------------------------------------------------------------------ #

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }

    def _check_status(self, response: httpx.Response) -> Optional[str]:
        """Return an error string for bad HTTP status codes, else None."""
        if response.status_code == 401:
            self.logger.error("❌ Ollama API: invalid API key")
            return "Invalid Ollama API key"
        if response.status_code == 429:
            self.logger.error("❌ Ollama API: rate limit exceeded")
            return "Ollama API rate limit exceeded — try again shortly"
        if response.status_code >= 400:
            self.logger.error(f"❌ Ollama API: HTTP {response.status_code}")
            return f"Ollama API error: HTTP {response.status_code}"
        return None


# ────────────────────────────────────────────────────────────────────── #
# Global singleton — same pattern as before                              #
# ────────────────────────────────────────────────────────────────────── #

_client: Optional[SearchClient] = None


def get_search_client() -> SearchClient:
    """Return (or lazily create) the global SearchClient instance."""
    global _client
    if _client is None:
        _client = SearchClient()
    return _client