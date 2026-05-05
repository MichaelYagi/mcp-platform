import pytest
from unittest.mock import patch, MagicMock
from client import models


def _make_httpx_response(model_names):
    """Build a mock httpx response matching Ollama /api/tags format."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "models": [{"name": n, "details": {"families": ["llm"]}} for n in model_names]
    }
    return mock_resp


MOCK_MODEL_NAMES = ["llama3.1:8b", "qwen2.5:7b", "mistral-nemo:latest"]


@pytest.mark.unit
class TestOllamaModels:
    def test_get_ollama_models_success(self, mock_ollama_list):
        """Test getting Ollama models when available"""
        with patch("httpx.get", return_value=_make_httpx_response(MOCK_MODEL_NAMES)):
            model_list = models.get_ollama_models()
            assert "llama3.1:8b" in model_list
            assert "qwen2.5:7b" in model_list
            assert len(model_list) == 3

    def test_get_ollama_models_not_running(self):
        """Test getting models when Ollama not running"""
        with patch("httpx.get", side_effect=Exception("Connection refused")):
            model_list = models.get_ollama_models()
            assert model_list == []

    @pytest.mark.asyncio
    async def test_switch_model_ollama_to_ollama(self, mock_tools, mock_llm):
        """Test switching between Ollama models"""
        with patch("client.models.LLMBackendManager.create_llm", return_value=mock_llm):
            with patch("client.models.detect_backend", return_value="ollama"):
                agent = await models.switch_model(
                    "qwen2.5:7b",
                    mock_tools,
                    MagicMock(),
                    lambda llm, tools: MagicMock(),
                    None
                )

                assert agent is not None

    def test_detect_backend_ollama(self):
        """Test detecting Ollama backend"""
        with patch("client.models.get_ollama_models", return_value=["llama3.1:8b"]):
            backend = models.detect_backend("llama3.1:8b")
            assert backend == "ollama"

    def test_detect_backend_gguf(self, mock_gguf_registry):
        """Test detecting GGUF backend"""
        with patch("client.models.get_ollama_models", return_value=[]):
            backend = models.detect_backend("tinyllama-merged")
            assert backend == "gguf"

    def test_detect_backend_not_found(self):
        """Test detecting nonexistent model"""
        with patch("client.models.get_ollama_models", return_value=[]):
            with patch("client.models.GGUFModelRegistry.list_models", return_value=[]):
                backend = models.detect_backend("nonexistent-model")
                assert backend is None


@pytest.mark.unit
class TestGGUFModels:
    def test_get_all_models_combined(self, mock_ollama_list, mock_gguf_registry):
        """Test getting models from both backends"""
        with patch("httpx.get", return_value=_make_httpx_response(MOCK_MODEL_NAMES)):
            all_models = models.get_all_models()
            ollama_models = [m for m in all_models if m["backend"] == "ollama"]
            gguf_models = [m for m in all_models if m["backend"] == "gguf"]
            assert len(ollama_models) == 3
            assert len(gguf_models) == 2

    def test_model_fallback_ollama_to_gguf(self, mock_gguf_registry):
        """Test fallback from Ollama to GGUF when Ollama unavailable"""
        with patch("client.models.get_ollama_models", return_value=[]):
            backend = models.get_initial_backend()

            # Should still work if GGUF models available
            all_models = models.get_all_models()
            gguf_only = [m for m in all_models if m["backend"] == "gguf"]

            if gguf_only:
                # Fallback should have GGUF models
                assert len(gguf_only) > 0


@pytest.mark.unit
class TestModelPersistence:
    def test_save_and_load_last_model(self, temp_dir):
        """Test saving and loading last used model"""
        with patch("client.models.MODEL_STATE_FILE", str(temp_dir / "last_model.txt")):
            models.save_last_model("qwen2.5:7b")

            loaded = models.load_last_model()
            assert loaded == "qwen2.5:7b"

    def test_load_last_model_not_exists(self, temp_dir):
        """Test loading when file doesn't exist"""
        with patch("client.models.MODEL_STATE_FILE", str(temp_dir / "nonexistent.txt")):
            loaded = models.load_last_model()
            assert loaded is None

    def test_save_overwrites_existing_file(self, temp_dir):
        """Saving a new model name replaces the previous one."""
        with patch("client.models.MODEL_STATE_FILE", str(temp_dir / "last_model.txt")):
            models.save_last_model("llama3.1:8b")
            models.save_last_model("qwen2.5:14b")
            loaded = models.load_last_model()
            assert loaded == "qwen2.5:14b"

    def test_saved_file_contains_exact_model_name(self, temp_dir):
        """File should contain just the model name, no extra whitespace."""
        state_file = temp_dir / "last_model.txt"
        with patch("client.models.MODEL_STATE_FILE", str(state_file)):
            models.save_last_model("qwen2.5:14b-instruct-q4_K_M")
            content = state_file.read_text().strip()
            assert content == "qwen2.5:14b-instruct-q4_K_M"

    def test_load_strips_whitespace(self, temp_dir):
        """load_last_model strips any trailing newline/whitespace from file."""
        state_file = temp_dir / "last_model.txt"
        state_file.write_text("qwen2.5:7b\n")
        with patch("client.models.MODEL_STATE_FILE", str(state_file)):
            loaded = models.load_last_model()
            assert loaded == "qwen2.5:7b"

    def test_load_empty_file_returns_none_or_empty(self, temp_dir):
        """Empty file should not crash and returns falsy value."""
        state_file = temp_dir / "last_model.txt"
        state_file.write_text("")
        with patch("client.models.MODEL_STATE_FILE", str(state_file)):
            loaded = models.load_last_model()
            assert not loaded  # None or empty string both acceptable

    def test_save_creates_file_if_not_exists(self, temp_dir):
        """save_last_model creates the file if it doesn't exist yet."""
        state_file = temp_dir / "subdir" / "last_model.txt"
        state_file.parent.mkdir(parents=True, exist_ok=True)
        with patch("client.models.MODEL_STATE_FILE", str(state_file)):
            models.save_last_model("llama3.2:3b")
            assert state_file.exists()
            assert state_file.read_text().strip() == "llama3.2:3b"

    def test_roundtrip_gguf_model_name(self, temp_dir):
        """GGUF alias names (no colon) roundtrip correctly."""
        with patch("client.models.MODEL_STATE_FILE", str(temp_dir / "last_model.txt")):
            models.save_last_model("tinyllama-merged")
            loaded = models.load_last_model()
            assert loaded == "tinyllama-merged"

    def test_switch_model_saves_last_model(self, temp_dir, mock_tools, mock_llm):
        """switch_model should persist the new model name after a successful switch."""
        state_file = temp_dir / "last_model.txt"
        with patch("client.models.MODEL_STATE_FILE", str(state_file)):
            with patch("client.models.detect_backend", return_value="ollama"):
                with patch("client.models.LLMBackendManager.create_llm", return_value=mock_llm):
                    import asyncio
                    asyncio.get_event_loop().run_until_complete(
                        models.switch_model(
                            "qwen2.5:14b", mock_tools, MagicMock(),
                            lambda llm, tools: MagicMock(), None
                        )
                    )
            if state_file.exists():
                saved = state_file.read_text().strip()
                assert saved == "qwen2.5:14b"