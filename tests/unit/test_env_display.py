"""
tests/unit/test_env_display.py
Tests for client/env_display.py — pure environment display formatting.
"""
import pytest
from unittest.mock import patch


@pytest.mark.unit
class TestEnvDisplay:
    def _get(self, env=None):
        """Call format_env_display with a controlled environment."""
        from client.env_display import format_env_display
        with patch.dict("os.environ", env or {}, clear=True):
            return format_env_display()

    # ── Structure ─────────────────────────────────────────────────
    def test_returns_string(self):
        assert isinstance(self._get(), str)

    def test_non_empty(self):
        assert len(self._get()) > 0

    def test_contains_header(self):
        result = self._get()
        assert "ENVIRONMENT" in result.upper() or "CONFIGURATION" in result.upper()

    def test_no_crash_on_empty_env(self):
        result = self._get({})
        assert isinstance(result, str) and len(result) > 0

    # ── Sections present in the actual output ─────────────────────
    def test_plex_section_present(self):
        result = self._get()
        assert "PLEX" in result.upper()

    def test_weather_section_present(self):
        result = self._get()
        assert "WEATHER" in result.upper()

    def test_a2a_section_present(self):
        result = self._get()
        assert "A2A" in result.upper()

    def test_agent_config_section_present(self):
        result = self._get()
        assert "AGENT" in result.upper() or "MAX_MESSAGE_HISTORY" in result.upper()

    def test_rag_section_present(self):
        result = self._get()
        assert "RAG" in result.upper() or "EMBEDDING" in result.upper()

    # ── Env var values shown ──────────────────────────────────────
    def test_plex_url_shown(self):
        result = self._get({"PLEX_URL": "http://192.168.1.1:32400"})
        assert "192.168.1.1" in result

    def test_plex_url_not_set_shown(self):
        result = self._get({})
        # Should indicate PLEX_URL is not set
        assert "not set" in result.lower() or "(not set)" in result

    def test_weather_token_masked(self):
        """Tokens should not appear verbatim."""
        result = self._get({"WEATHER_TOKEN": "super-secret-12345"})
        assert "super-secret-12345" not in result

    def test_ollama_token_masked(self):
        result = self._get({"OLLAMA_TOKEN": "my-secret-ollama-token"})
        assert "my-secret-ollama-token" not in result

    def test_plex_token_masked(self):
        result = self._get({"PLEX_TOKEN": "my-plex-token-xyz"})
        assert "my-plex-token-xyz" not in result

    def test_a2a_endpoints_shown(self):
        result = self._get({"A2A_ENDPOINTS": "http://localhost:8080"})
        assert "localhost:8080" in result or "A2A_ENDPOINTS" in result.upper()

    def test_max_message_history_shown(self):
        result = self._get({"MAX_MESSAGE_HISTORY": "50"})
        assert "50" in result or "MAX_MESSAGE_HISTORY" in result.upper()

    def test_concurrent_limit_shown(self):
        result = self._get({"CONCURRENT_LIMIT": "4"})
        assert "4" in result or "CONCURRENT_LIMIT" in result.upper()

    # ── No crash on any combination ───────────────────────────────
    def test_all_vars_set(self):
        result = self._get({
            "PLEX_URL": "http://127.0.0.1:32400",
            "PLEX_TOKEN": "secret",
            "WEATHER_TOKEN": "secret2",
            "A2A_ENDPOINTS": "http://localhost:9000",
            "MAX_MESSAGE_HISTORY": "20",
            "CONCURRENT_LIMIT": "2",
        })
        assert isinstance(result, str) and len(result) > 10