"""
Tests for client/search_client.py
Covers: SearchClient init, is_available, search, fetch_url,
        _headers, _check_status, get_search_client singleton
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx


# ═══════════════════════════════════════════════════════════════════
# SearchClient — init & is_available
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestSearchClientInit:
    def test_is_available_with_key(self):
        from client.search_client import SearchClient
        client = SearchClient(api_key="test-key-123")
        assert client.is_available() is True

    def test_is_not_available_without_key(self, monkeypatch):
        from client.search_client import SearchClient
        monkeypatch.delenv("OLLAMA_TOKEN", raising=False)
        client = SearchClient(api_key="")
        assert client.is_available() is False

    def test_reads_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_TOKEN", "env-key-xyz")
        from client.search_client import SearchClient
        client = SearchClient()
        assert client.api_key == "env-key-xyz"
        assert client.is_available() is True

    def test_explicit_key_overrides_env(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_TOKEN", "env-key")
        from client.search_client import SearchClient
        client = SearchClient(api_key="explicit-key")
        assert client.api_key == "explicit-key"


# ═══════════════════════════════════════════════════════════════════
# _headers & _check_status
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestSearchClientHelpers:
    def test_headers_include_auth(self):
        from client.search_client import SearchClient
        client = SearchClient(api_key="my-key")
        headers = client._headers()
        assert headers["Authorization"] == "Bearer my-key"
        assert headers["Content-Type"] == "application/json"

    def test_check_status_200_returns_none(self):
        from client.search_client import SearchClient
        client = SearchClient(api_key="key")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        assert client._check_status(mock_resp) is None

    def test_check_status_401_returns_error(self):
        from client.search_client import SearchClient
        client = SearchClient(api_key="key")
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        result = client._check_status(mock_resp)
        assert result is not None
        assert "invalid" in result.lower() or "key" in result.lower()

    def test_check_status_429_returns_rate_limit(self):
        from client.search_client import SearchClient
        client = SearchClient(api_key="key")
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        result = client._check_status(mock_resp)
        assert "rate limit" in result.lower()

    def test_check_status_500_returns_error(self):
        from client.search_client import SearchClient
        client = SearchClient(api_key="key")
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        result = client._check_status(mock_resp)
        assert result is not None
        assert "500" in result


# ═══════════════════════════════════════════════════════════════════
# search()
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
class TestSearchClientSearch:
    async def test_search_no_api_key(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_TOKEN", raising=False)
        from client.search_client import SearchClient
        client = SearchClient(api_key="")
        result = await client.search("test query")
        assert result["success"] is False
        assert "OLLAMA_TOKEN" in result["error"] or "key" in result["error"].lower()

    async def test_search_success(self):
        from client.search_client import SearchClient
        client = SearchClient(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {"url": "https://example.com", "title": "Example", "content": "Some content"},
                {"url": "https://other.com", "title": "Other", "content": "More content"},
            ]
        }

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)

            result = await client.search("test query")

        assert result["success"] is True
        assert "results" in result
        pages = result["results"]["webPages"]["value"]
        assert len(pages) == 2
        assert pages[0]["url"] == "https://example.com"

    async def test_search_normalises_to_bing_format(self):
        from client.search_client import SearchClient
        client = SearchClient(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [{"url": "https://x.com", "title": "X", "content": "content"}]
        }

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)
            result = await client.search("query")

        pages = result["results"]["webPages"]["value"]
        assert "url" in pages[0]
        assert "name" in pages[0]
        assert "summary" in pages[0]

    async def test_search_auth_error(self):
        from client.search_client import SearchClient
        client = SearchClient(api_key="bad-key")

        mock_response = MagicMock()
        mock_response.status_code = 401

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)
            result = await client.search("query")

        assert result["success"] is False
        assert "key" in result["error"].lower() or "invalid" in result["error"].lower()

    async def test_search_timeout(self):
        from client.search_client import SearchClient
        client = SearchClient(api_key="test-key")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)
            result = await client.search("query")

        assert result["success"] is False
        assert "timed out" in result["error"].lower()

    async def test_search_http_error(self):
        from client.search_client import SearchClient
        client = SearchClient(api_key="test-key")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(
                side_effect=httpx.HTTPError("connection failed")
            )
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)
            result = await client.search("query")

        assert result["success"] is False

    async def test_search_unexpected_error(self):
        from client.search_client import SearchClient
        client = SearchClient(api_key="test-key")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=RuntimeError("unexpected"))
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)
            result = await client.search("query")

        assert result["success"] is False
        assert "unexpected" in result["error"]

    async def test_search_max_results_capped_at_10(self):
        from client.search_client import SearchClient
        client = SearchClient(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": []}

        captured_payload = {}

        async def capture_post(url, headers, json):
            captured_payload.update(json)
            return mock_response

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = capture_post
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)
            await client.search("query", max_results=50)

        assert captured_payload.get("max_results", 0) <= 10


# ═══════════════════════════════════════════════════════════════════
# fetch_url()
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
@pytest.mark.asyncio
class TestSearchClientFetchUrl:
    async def test_fetch_no_api_key(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_TOKEN", raising=False)
        from client.search_client import SearchClient
        client = SearchClient(api_key="")
        result = await client.fetch_url("https://example.com")
        assert result["success"] is False

    async def test_fetch_success(self):
        from client.search_client import SearchClient
        client = SearchClient(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "title": "Example Page",
            "content": "This is the page content. " * 5,
            "links": ["https://link1.com", "https://link2.com"]
        }

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)
            result = await client.fetch_url("https://example.com")

        assert result["success"] is True
        assert result["title"] == "Example Page"
        assert len(result["content"]) > 0
        assert isinstance(result["links"], list)

    async def test_fetch_empty_content(self):
        from client.search_client import SearchClient
        client = SearchClient(api_key="test-key")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"title": "Empty", "content": "", "links": []}

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)
            result = await client.fetch_url("https://example.com")

        assert result["success"] is False
        assert "No content" in result["error"]

    async def test_fetch_truncates_large_content(self):
        from client.search_client import SearchClient
        client = SearchClient(api_key="test-key")

        large_content = "x" * 20_000

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "title": "Large Page",
            "content": large_content,
            "links": []
        }

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)
            result = await client.fetch_url("https://example.com")

        assert result["success"] is True
        assert len(result["content"]) <= 10_100  # 10000 + truncation suffix
        assert "truncated" in result["content"]

    async def test_fetch_timeout(self):
        from client.search_client import SearchClient
        client = SearchClient(api_key="test-key")

        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
            mock_client_cls.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=None)
            result = await client.fetch_url("https://example.com")

        assert result["success"] is False
        assert "timed out" in result["error"].lower()


# ═══════════════════════════════════════════════════════════════════
# get_search_client singleton
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestGetSearchClient:
    def test_returns_search_client_instance(self):
        from client.search_client import get_search_client, SearchClient
        import client.search_client as sc
        sc._client = None  # reset singleton
        client = get_search_client()
        assert isinstance(client, SearchClient)

    def test_returns_same_instance(self):
        from client.search_client import get_search_client
        import client.search_client as sc
        sc._client = None
        c1 = get_search_client()
        c2 = get_search_client()
        assert c1 is c2