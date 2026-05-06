"""
tests/unit/test_llm_backend.py
Tests for client/llm_backend.py — LLMBackendManager and GGUFModelRegistry.
"""
import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


# ═══════════════════════════════════════════════════════════════════
# LLMBackendManager.get_backend_type
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestGetBackendType:
    def test_defaults_to_ollama(self, monkeypatch):
        monkeypatch.delenv("LLM_BACKEND", raising=False)
        from client.llm_backend import LLMBackendManager
        assert LLMBackendManager.get_backend_type() == "ollama"

    def test_reads_from_env(self, monkeypatch):
        monkeypatch.setenv("LLM_BACKEND", "gguf")
        from client.llm_backend import LLMBackendManager
        assert LLMBackendManager.get_backend_type() == "gguf"

    def test_lowercases_value(self, monkeypatch):
        monkeypatch.setenv("LLM_BACKEND", "OLLAMA")
        from client.llm_backend import LLMBackendManager
        assert LLMBackendManager.get_backend_type() == "ollama"


# ═══════════════════════════════════════════════════════════════════
# LLMBackendManager.create_llm — ollama path
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestCreateLlmOllama:
    def setup_method(self):
        import client.llm_backend as lb
        lb._CURRENT_LLM = None
        lb._CURRENT_MODEL_NAME = None

    def test_create_ollama_llm(self, monkeypatch):
        monkeypatch.setenv("LLM_BACKEND", "ollama")
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        from client.llm_backend import LLMBackendManager
        mock_llm = MagicMock()
        with patch("client.llm_backend.ChatOllama", return_value=mock_llm) as mock_cls:
            result = LLMBackendManager.create_llm("qwen2.5:7b")
            mock_cls.assert_called_once()
            assert result is mock_llm

    def test_create_passes_base_url(self, monkeypatch):
        monkeypatch.setenv("LLM_BACKEND", "ollama")
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://192.168.0.1:11434")
        from client.llm_backend import LLMBackendManager
        with patch("client.llm_backend.ChatOllama", return_value=MagicMock()) as mock_cls:
            LLMBackendManager.create_llm("llama3.1:8b")
            call_kwargs = mock_cls.call_args[1]
            assert "192.168.0.1" in call_kwargs.get("base_url", "")

    def test_cached_model_not_reloaded(self, monkeypatch):
        monkeypatch.setenv("LLM_BACKEND", "ollama")
        from client.llm_backend import LLMBackendManager
        import client.llm_backend as lb
        mock_llm = MagicMock()
        mock_llm.num_ctx = None
        lb._CURRENT_LLM = mock_llm
        lb._CURRENT_MODEL_NAME = "qwen2.5:7b"
        with patch("client.llm_backend.ChatOllama") as mock_cls:
            result = LLMBackendManager.create_llm("qwen2.5:7b")
            mock_cls.assert_not_called()
            assert result is mock_llm

    def test_different_model_reloads(self, monkeypatch):
        monkeypatch.setenv("LLM_BACKEND", "ollama")
        from client.llm_backend import LLMBackendManager
        import client.llm_backend as lb
        mock_llm_old = MagicMock()
        mock_llm_old.num_ctx = None
        lb._CURRENT_LLM = mock_llm_old
        lb._CURRENT_MODEL_NAME = "llama3.1:8b"
        mock_llm_new = MagicMock()
        with patch("client.llm_backend.ChatOllama", return_value=mock_llm_new):
            result = LLMBackendManager.create_llm("qwen2.5:14b")
            assert result is mock_llm_new
            assert lb._CURRENT_MODEL_NAME == "qwen2.5:14b"

    def test_temperature_passed_through(self, monkeypatch):
        monkeypatch.setenv("LLM_BACKEND", "ollama")
        from client.llm_backend import LLMBackendManager
        with patch("client.llm_backend.ChatOllama", return_value=MagicMock()) as mock_cls:
            LLMBackendManager.create_llm("llama3.1:8b", temperature=0.7)
            call_kwargs = mock_cls.call_args[1]
            assert call_kwargs.get("temperature") == 0.7

    def test_unknown_backend_raises(self, monkeypatch):
        monkeypatch.setenv("LLM_BACKEND", "unknown_backend_xyz")
        from client.llm_backend import LLMBackendManager
        import client.llm_backend as lb
        lb._CURRENT_LLM = None
        lb._CURRENT_MODEL_NAME = None
        with pytest.raises(Exception):
            LLMBackendManager.create_llm("some-model")


# ═══════════════════════════════════════════════════════════════════
# LLMBackendManager.create_llm — gguf path
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestCreateLlmGGUF:
    def setup_method(self):
        import client.llm_backend as lb
        lb._CURRENT_LLM = None
        lb._CURRENT_MODEL_NAME = None

    def test_gguf_model_not_in_registry_raises(self, monkeypatch, temp_dir):
        monkeypatch.setenv("LLM_BACKEND", "gguf")
        with patch("client.llm_backend.GGUF_MODELS_FILE", str(temp_dir / "models.json")):
            from client.llm_backend import LLMBackendManager
            with pytest.raises(ValueError, match="not in registry"):
                LLMBackendManager.create_llm("nonexistent-model")

    def test_gguf_file_not_found_raises(self, monkeypatch, temp_dir):
        monkeypatch.setenv("LLM_BACKEND", "gguf")
        models_file = temp_dir / "models.json"
        models_file.write_text(json.dumps({
            "mymodel": {"path": "/nonexistent/model.gguf", "description": "", "size_mb": 500}
        }))
        with patch("client.llm_backend.GGUF_MODELS_FILE", str(models_file)):
            from client.llm_backend import LLMBackendManager
            with pytest.raises(FileNotFoundError):
                LLMBackendManager.create_llm("mymodel")


# ═══════════════════════════════════════════════════════════════════
# GGUFModelRegistry
# ═══════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestGGUFRegistryIO:
    def test_load_empty_when_no_file(self, temp_dir):
        with patch("client.llm_backend.GGUF_MODELS_FILE", str(temp_dir / "none.json")):
            from client.llm_backend import GGUFModelRegistry
            assert GGUFModelRegistry.load_models() == {}

    def test_load_returns_dict(self, temp_dir):
        models_file = temp_dir / "models.json"
        models_file.write_text(json.dumps({"model1": {"path": "/a/b.gguf", "size_mb": 400}}))
        with patch("client.llm_backend.GGUF_MODELS_FILE", str(models_file)):
            from client.llm_backend import GGUFModelRegistry
            assert "model1" in GGUFModelRegistry.load_models()

    def test_load_handles_corrupt_json(self, temp_dir):
        models_file = temp_dir / "models.json"
        models_file.write_text("{{not valid json")
        with patch("client.llm_backend.GGUF_MODELS_FILE", str(models_file)):
            from client.llm_backend import GGUFModelRegistry
            assert GGUFModelRegistry.load_models() == {}

    def test_save_writes_json(self, temp_dir):
        models_file = temp_dir / "models.json"
        with patch("client.llm_backend.GGUF_MODELS_FILE", str(models_file)):
            from client.llm_backend import GGUFModelRegistry
            GGUFModelRegistry.save_models({"test": {"path": "/x.gguf"}})
            assert "test" in json.loads(models_file.read_text())


@pytest.mark.unit
class TestGGUFAddModel:
    def test_add_nonexistent_file_raises(self, temp_dir):
        with patch("client.llm_backend.GGUF_MODELS_FILE", str(temp_dir / "m.json")):
            from client.llm_backend import GGUFModelRegistry
            with pytest.raises(FileNotFoundError):
                GGUFModelRegistry.add_model("alias", "/nonexistent/model.gguf")

    def test_add_non_gguf_extension_raises(self, temp_dir):
        bad_file = temp_dir / "model.bin"
        bad_file.write_bytes(b"x" * 2_000_000)
        with patch("client.llm_backend.GGUF_MODELS_FILE", str(temp_dir / "m.json")):
            from client.llm_backend import GGUFModelRegistry
            with pytest.raises(ValueError, match="gguf"):
                GGUFModelRegistry.add_model("alias", str(bad_file))

    def test_add_too_small_file_raises(self, temp_dir):
        small_file = temp_dir / "tiny.gguf"
        small_file.write_bytes(b"x" * 100)
        with patch("client.llm_backend.GGUF_MODELS_FILE", str(temp_dir / "m.json")):
            from client.llm_backend import GGUFModelRegistry
            with pytest.raises(ValueError, match="too small"):
                GGUFModelRegistry.add_model("alias", str(small_file))

    def test_add_valid_model(self, temp_dir):
        gguf_file = temp_dir / "model.gguf"
        gguf_file.write_bytes(b"GGUF" + b"x" * 2_000_000)
        models_file = temp_dir / "models.json"
        with patch("client.llm_backend.GGUF_MODELS_FILE", str(models_file)):
            from client.llm_backend import GGUFModelRegistry
            result = GGUFModelRegistry.add_model("mymodel", str(gguf_file))
            assert "path" in result
            assert "size_mb" in result
            assert GGUFModelRegistry.load_models().get("mymodel") is not None

    def test_add_overwrites_existing_alias(self, temp_dir):
        gguf1 = temp_dir / "model1.gguf"
        gguf2 = temp_dir / "model2.gguf"
        gguf1.write_bytes(b"GGUF" + b"x" * 2_000_000)
        gguf2.write_bytes(b"GGUF" + b"y" * 2_000_000)
        models_file = temp_dir / "models.json"
        with patch("client.llm_backend.GGUF_MODELS_FILE", str(models_file)):
            from client.llm_backend import GGUFModelRegistry
            GGUFModelRegistry.add_model("mymodel", str(gguf1))
            GGUFModelRegistry.add_model("mymodel", str(gguf2))
            info = GGUFModelRegistry.load_models()["mymodel"]
            assert "model2" in info["path"]


@pytest.mark.unit
class TestGGUFRemoveModel:
    def test_remove_existing(self, temp_dir):
        models_file = temp_dir / "models.json"
        models_file.write_text(json.dumps({"mymodel": {"path": "/x.gguf", "size_mb": 400}}))
        with patch("client.llm_backend.GGUF_MODELS_FILE", str(models_file)):
            from client.llm_backend import GGUFModelRegistry
            assert GGUFModelRegistry.remove_model("mymodel") is True
            assert "mymodel" not in GGUFModelRegistry.load_models()

    def test_remove_nonexistent_returns_false(self, temp_dir):
        models_file = temp_dir / "models.json"
        models_file.write_text(json.dumps({}))
        with patch("client.llm_backend.GGUF_MODELS_FILE", str(models_file)):
            from client.llm_backend import GGUFModelRegistry
            assert GGUFModelRegistry.remove_model("ghost") is False


@pytest.mark.unit
class TestGGUFListModels:
    def test_list_empty(self, temp_dir):
        with patch("client.llm_backend.GGUF_MODELS_FILE", str(temp_dir / "none.json")):
            from client.llm_backend import GGUFModelRegistry
            assert GGUFModelRegistry.list_models() == []

    def test_list_returns_aliases(self, temp_dir):
        models_file = temp_dir / "models.json"
        models_file.write_text(json.dumps({
            "model_a": {"path": "/a.gguf", "size_mb": 400},
            "model_b": {"path": "/b.gguf", "size_mb": 800},
        }))
        with patch("client.llm_backend.GGUF_MODELS_FILE", str(models_file)):
            from client.llm_backend import GGUFModelRegistry
            result = GGUFModelRegistry.list_models()
            assert "model_a" in result and "model_b" in result

    def test_get_models_full_info(self, temp_dir):
        models_file = temp_dir / "models.json"
        models_file.write_text(json.dumps({
            "mymodel": {"path": "/x.gguf", "description": "test", "size_mb": 500}
        }))
        with patch("client.llm_backend.GGUF_MODELS_FILE", str(models_file)):
            from client.llm_backend import GGUFModelRegistry
            result = GGUFModelRegistry.get_models()
            assert len(result) == 1
            assert result[0]["alias"] == "mymodel"

    def test_get_model_info_found(self, temp_dir):
        models_file = temp_dir / "models.json"
        models_file.write_text(json.dumps({"target": {"path": "/t.gguf", "size_mb": 300}}))
        with patch("client.llm_backend.GGUF_MODELS_FILE", str(models_file)):
            from client.llm_backend import GGUFModelRegistry
            info = GGUFModelRegistry.get_model_info("target")
            assert info is not None and info["size_mb"] == 300

    def test_get_model_info_missing_returns_none(self, temp_dir):
        with patch("client.llm_backend.GGUF_MODELS_FILE", str(temp_dir / "none.json")):
            from client.llm_backend import GGUFModelRegistry
            assert GGUFModelRegistry.get_model_info("missing") is None


@pytest.mark.unit
class TestGGUFValidateModel:
    def test_validate_not_in_registry(self, temp_dir):
        with patch("client.llm_backend.GGUF_MODELS_FILE", str(temp_dir / "none.json")):
            from client.llm_backend import GGUFModelRegistry
            valid, msg = GGUFModelRegistry.validate_model("ghost")
            assert valid is False
            assert "not in registry" in msg

    def test_validate_file_missing(self, temp_dir):
        models_file = temp_dir / "models.json"
        models_file.write_text(json.dumps({
            "mymodel": {"path": "/nonexistent/model.gguf", "size_mb": 500}
        }))
        with patch("client.llm_backend.GGUF_MODELS_FILE", str(models_file)):
            from client.llm_backend import GGUFModelRegistry
            valid, msg = GGUFModelRegistry.validate_model("mymodel")
            assert valid is False
            assert "not found" in msg.lower()

    def test_validate_valid_file(self, temp_dir):
        gguf_file = temp_dir / "model.gguf"
        gguf_file.write_bytes(b"GGUF" + b"x" * 2_000_000)
        models_file = temp_dir / "models.json"
        models_file.write_text(json.dumps({"mymodel": {"path": str(gguf_file), "size_mb": 2.0}}))
        with patch("client.llm_backend.GGUF_MODELS_FILE", str(models_file)):
            from client.llm_backend import GGUFModelRegistry
            valid, msg = GGUFModelRegistry.validate_model("mymodel")
            assert valid is True
            assert msg == "Valid"