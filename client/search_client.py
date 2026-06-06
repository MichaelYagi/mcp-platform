"""
Ollama Web Search Client
Provides web search and URL fetching via Ollama's API, with automatic
fallback to LangSearch when Ollama's weekly limit is reached or returns
an empty response.

Primary:  Ollama  — https://docs.ollama.com/capabilities/web-search
Fallback: LangSearch — https://docs.langsearch.com/api/web-search-api

Auth:
  OLLAMA_TOKEN      — free Ollama account
  LANGSEARCH_API_KEY — free LangSearch account (https://langsearch.com/dashboard)
"""
import os
import httpx
import logging
from typing import Optional, Dict, Any


class SearchClient:
    """
    Web search and URL fetch client backed by Ollama's cloud API,
    with automatic fallback to LangSearch.

    Exposes three methods used throughout the codebase:
      - is_available()  → True when at least one provider is configured
      - search(query)   → web search (returns structured results)
      - fetch_url(url)  → returns clean markdown content

    The get_search_client() factory keeps the original name so that
    existing call-sites in langgraph.py require no changes.
    """

    OLLAMA_SEARCH_ENDPOINT = "https://ollama.com/api/web_search"
    OLLAMA_FETCH_ENDPOINT  = "https://ollama.com/api/web_fetch"
    LANGSEARCH_ENDPOINT    = "https://api.langsearch.com/v1/web-search"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key          = api_key or os.getenv("OLLAMA_TOKEN", "").strip()
        self.langsearch_key   = os.getenv("LANGSEARCH_API_KEY", "").strip()
        self.logger           = logging.getLogger("mcp_client")

    # ------------------------------------------------------------------ #
    # Public interface                                                      #
    # ------------------------------------------------------------------ #

    def is_available(self) -> bool:
        """Return True when at least one provider is configured."""
        return bool(self.api_key or self.langsearch_key)

    async def search(self, query: str, max_results: int = 5,
                     timeout: float = 60.0) -> Dict[str, Any]:
        """
        Perform a web search, trying Ollama first and falling back to
        LangSearch when Ollama is unavailable, rate-limited, or returns
        an empty body (weekly limit reached).

        Returns a dict consumed by langgraph.py's URL-extraction logic:

            {
                "success": bool,
                "results": <structured dict>,   # on success
                "raw_response": <raw JSON>,     # on success
                "error": str,                   # on failure
            }

        The "results" value uses the Bing-compatible shape:

            results["webPages"]["value"] → list of {url, name, summary}
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "No search provider configured. Set OLLAMA_TOKEN or LANGSEARCH_API_KEY."
            }

        # ── Try Ollama first ──────────────────────────────────────────── #
        ollama_error: Optional[Dict[str, Any]] = None
        if self.api_key:
            result = await self._ollama_search(query, max_results, timeout)
            if result.get("success") and self._has_results(result):
                return result
            ollama_error = result
            self.logger.warning(
                f"⚠️ Ollama search unavailable or empty — falling back to LangSearch "
                f"(reason: {result.get('error', 'empty response')})"
            )

        # ── Fallback: LangSearch ─────────────────────────────────────── #
        if self.langsearch_key:
            return await self._langsearch_search(query, max_results, timeout)

        if ollama_error:
            return ollama_error

        return {"success": False, "error": "All search providers failed or unconfigured"}

    async def fetch_url(self, url: str,
                        timeout: float = 30.0) -> Dict[str, Any]:
        """
        Fetch a single web page, trying Ollama first then falling back to
        a direct httpx fetch when Ollama is unavailable.

        Returns:
            {
                "success": bool,
                "title":   str,
                "content": str,
                "links":   list[str],
                "error":   str   (only on failure)
            }
        """
        if not self.is_available():
            return {"success": False, "error": "No provider configured"}

        # ── Try Ollama first ──────────────────────────────────────────── #
        if self.api_key:
            result = await self._ollama_fetch(url, timeout)
            if result.get("success"):
                return result
            self.logger.warning(
                f"⚠️ Ollama fetch unavailable — falling back to direct fetch "
                f"(reason: {result.get('error', 'unknown')})"
            )

        # ── Fallback: direct httpx fetch ─────────────────────────────── #
        return await self._direct_fetch(url, timeout)

    # ------------------------------------------------------------------ #
    # Ollama provider                                                       #
    # ------------------------------------------------------------------ #

    async def _ollama_search(self, query: str, max_results: int,
                             timeout: float) -> Dict[str, Any]:
        self.logger.info(f"🔍 Ollama web search: '{query[:100]}'")

        payload = {
            "query":       query,
            "max_results": min(max(1, max_results), 10)
        }

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    self.OLLAMA_SEARCH_ENDPOINT,
                    headers=self._ollama_headers(),
                    json=payload
                )

            error = self._check_ollama_status(response)
            if error:
                return {"success": False, "error": error}

            raw_body = response.text.strip()
            if not raw_body:
                return {"success": False, "error": "Ollama returned empty response (weekly limit reached?)"}

            data        = response.json()
            raw_results = data.get("results", [])

            self.logger.info(f"✅ Ollama web search returned {len(raw_results)} result(s)")

            return {
                "success":      True,
                "results":      self._normalise_ollama_results(raw_results),
                "raw_response": data,
            }

        except httpx.TimeoutException:
            return {"success": False, "error": "Ollama web search timed out"}
        except httpx.HTTPError as exc:
            return {"success": False, "error": f"HTTP error: {exc}"}
        except Exception as exc:
            self.logger.error(f"❌ Ollama web search unexpected error: {exc}")
            return {"success": False, "error": str(exc)}

    async def _ollama_fetch(self, url: str, timeout: float) -> Dict[str, Any]:
        self.logger.info(f"🌐 Ollama web fetch: {url}")

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    self.OLLAMA_FETCH_ENDPOINT,
                    headers=self._ollama_headers(),
                    json={"url": url}
                )

            error = self._check_ollama_status(response)
            if error:
                return {"success": False, "error": error}

            raw_body = response.text.strip()
            if not raw_body:
                return {"success": False, "error": "Ollama returned empty response"}

            data    = response.json()
            content = data.get("content", "")

            if not content or len(content) < 50:
                return {"success": False, "error": "No content returned"}

            if len(content) > 10_000:
                content = content[:10_000] + "\n\n[Content truncated…]"

            self.logger.info(
                f"✅ Ollama web fetch: '{data.get('title', 'Untitled')}' ({len(content)} chars)"
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
    # LangSearch provider                                                   #
    # ------------------------------------------------------------------ #

    async def _langsearch_search(self, query: str, max_results: int,
                                 timeout: float) -> Dict[str, Any]:
        self.logger.info(f"🔍 LangSearch web search: '{query[:100]}'")

        payload = {
            "query":   query,
            "count":   min(max(1, max_results), 10),
            "summary": True,
        }

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    self.LANGSEARCH_ENDPOINT,
                    headers={
                        "Authorization": f"Bearer {self.langsearch_key}",
                        "Content-Type":  "application/json",
                    },
                    json=payload
                )

            if response.status_code == 401:
                return {"success": False, "error": "Invalid LangSearch API key"}
            if response.status_code >= 400:
                return {"success": False, "error": f"LangSearch API error: HTTP {response.status_code}"}

            data       = response.json()
            raw_pages  = data.get("data", {}).get("webPages", {}).get("value", [])

            self.logger.info(f"✅ LangSearch returned {len(raw_pages)} result(s)")

            normalised = {
                "webPages": {
                    "value": [
                        {
                            "url":     p.get("url", ""),
                            "name":    p.get("name", "Untitled"),
                            "summary": p.get("summary") or p.get("snippet", ""),
                        }
                        for p in raw_pages
                        if isinstance(p, dict)
                    ]
                }
            }

            return {
                "success":      True,
                "results":      normalised,
                "raw_response": data,
            }

        except httpx.TimeoutException:
            return {"success": False, "error": "LangSearch request timed out"}
        except httpx.HTTPError as exc:
            return {"success": False, "error": f"HTTP error: {exc}"}
        except Exception as exc:
            self.logger.error(f"❌ LangSearch unexpected error: {exc}")
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------ #
    # Direct fetch fallback (no Ollama dependency)                         #
    # ------------------------------------------------------------------ #

    async def _direct_fetch(self, url: str, timeout: float) -> Dict[str, Any]:
        """Basic httpx fetch used when Ollama fetch is unavailable."""
        self.logger.info(f"🌐 Direct fetch: {url}")
        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                response = await client.get(url, headers={"User-Agent": "Mozilla/5.0"})

            if response.status_code >= 400:
                return {"success": False, "error": f"HTTP {response.status_code}"}

            # Strip tags crudely — good enough for plain-text extraction
            import re
            text = re.sub(r"<[^>]+>", " ", response.text)
            text = re.sub(r"\s{2,}", " ", text).strip()

            if len(text) > 10_000:
                text = text[:10_000] + "\n\n[Content truncated…]"

            return {"success": True, "title": url, "content": text, "links": []}

        except Exception as exc:
            self.logger.error(f"❌ Direct fetch error: {exc}")
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------ #
    # Private helpers                                                       #
    # ------------------------------------------------------------------ #

    def _ollama_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }

    def _check_ollama_status(self, response: httpx.Response) -> Optional[str]:
        """Return an error string for bad HTTP status codes, else None."""
        if response.status_code == 401:
            self.logger.error("❌ Ollama API: invalid API key")
            return "Invalid Ollama API key"
        if response.status_code == 429:
            self.logger.error("❌ Ollama API: rate limit exceeded")
            return "Ollama API rate limit exceeded"
        if response.status_code >= 400:
            self.logger.error(f"❌ Ollama API: HTTP {response.status_code}")
            return f"Ollama API error: HTTP {response.status_code}"
        return None

    @staticmethod
    def _normalise_ollama_results(raw_results: list) -> Dict[str, Any]:
        return {
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

    @staticmethod
    def _has_results(result: Dict[str, Any]) -> bool:
        """Return True when the result contains at least one web page."""
        try:
            return bool(result["results"]["webPages"]["value"])
        except (KeyError, TypeError):
            return False


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