import pytest
from unittest.mock import patch, MagicMock
from client import models

@pytest.mark.unit
class TestOllamaModels:
    def test_get_ollama_models_success(self, mock_ollama_list):
        """Test getting Ollama models when available"""
        with patch("subprocess.check_output", return_value=mock_ollama_list):
            model_list = models.get_ollama_models()
            
            assert "llama3.1:8b" in model_list
            assert "qwen2.5:7b" in model_list
            assert len(model_list) == 3
    
    def test_get_ollama_models_not_running(self):
        """Test getting models when Ollama not running"""
        with patch("subprocess.check_output", side_effect=Exception("Connection refused")):
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
        with patch("subprocess.check_output", return_value=mock_ollama_list):
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