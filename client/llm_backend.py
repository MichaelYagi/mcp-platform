"""
LLM Backend Manager
Supports both Ollama and GGUF with unified model registry
"""

import os
import json
import logging
from typing import Optional, Dict, List
from pathlib import Path
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)

GGUF_MODELS_FILE = "gguf_models.json"

_CURRENT_LLM = None
_CURRENT_MODEL_NAME = None

class LLMBackendManager:
    """Factory for creating LLM instances"""

    @staticmethod
    def get_backend_type() -> str:
        """Get current backend from environment"""
        return os.getenv("LLM_BACKEND", "ollama").lower()

    @staticmethod
    def create_llm(model_name: str, temperature: float = 0, **kwargs) -> BaseChatModel:
        """
        Create LLM instance - auto-detects backend
        Uses last_model.txt as source of truth - only reloads if model changed

        Args:
            model_name: Model name (Ollama) or alias (GGUF)
            temperature: Sampling temperature

        Returns:
            BaseChatModel instance

        Raises:
            ValueError: If model not found or invalid
            FileNotFoundError: If GGUF file doesn't exist
            RuntimeError: If GGUF file is corrupted/invalid
        """
        global _CURRENT_LLM, _CURRENT_MODEL_NAME

        # ✅ CHECK: Is this the same model we already loaded?
        if _CURRENT_MODEL_NAME == model_name and _CURRENT_LLM is not None:
            cached_ctx = getattr(_CURRENT_LLM, 'num_ctx', None)
            requested_ctx = kwargs.get('num_ctx')
            if requested_ctx is None or cached_ctx == requested_ctx:
                logger.info(f"✅ Model already loaded: {model_name} (no reload)")
                return _CURRENT_LLM
            logger.info(f"🔄 Reloading: num_ctx changed ({cached_ctx} → {requested_ctx})")

        # Different model - need to load
        logger.info(f"🔄 Loading new model: {model_name}")

        backend = LLMBackendManager.get_backend_type()

        if backend == "ollama":
            ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
            llm = ChatOllama(model=model_name, temperature=temperature, base_url=ollama_base_url, **kwargs)

            # Save to cache
            _CURRENT_LLM = llm
            _CURRENT_MODEL_NAME = model_name
            return llm

        elif backend == "gguf":
            try:
                from langchain_community.chat_models import ChatLlamaCpp
            except ImportError:
                raise ImportError("Install llama-cpp-python: pip install llama-cpp-python")

            # Get model path from registry
            models = GGUFModelRegistry.load_models()
            if model_name not in models:
                available = list(models.keys())
                raise ValueError(
                    f"GGUF model '{model_name}' not in registry.\n"
                    f"Available: {', '.join(available) if available else 'none'}\n"
                    f"Add with: :gguf add <alias> <path>"
                )

            model_info = models[model_name]
            model_path = model_info["path"]

            # Validate file exists
            if not Path(model_path).exists():
                raise FileNotFoundError(
                    f"GGUF file not found: {model_path}\n"
                    f"The file may have been moved or deleted.\n"
                    f"Remove with: :gguf remove {model_name}"
                )

            n_gpu_layers = int(os.getenv("GGUF_GPU_LAYERS", "-1"))
            n_ctx = int(os.getenv("GGUF_CONTEXT_SIZE", "4096"))
            n_batch = int(os.getenv("GGUF_BATCH_SIZE", "512"))

            file_size_mb = model_info.get("size_mb", 0)

            logger.info(f"🔧 Loading GGUF: {Path(model_path).name}")
            logger.info(f"   Size: {file_size_mb} MB, GPU layers: {n_gpu_layers}, Context: {n_ctx}, Batch: {n_batch}")

            # Show loading message for large models
            if file_size_mb > 1000:  # > 1GB
                print(f"⏳ Loading large model ({file_size_mb} MB)...")
                print(f"   This may take 30-90 seconds...")
                print(f"   (If it hangs, try: export GGUF_GPU_LAYERS=0)")

            try:
                import concurrent.futures

                # Load model in thread pool with timeout
                def load_model():
                    print(f"   Loading model...")
                    return ChatLlamaCpp(
                        model_path=model_path,
                        temperature=temperature,
                        n_gpu_layers=n_gpu_layers,
                        n_ctx=n_ctx,
                        n_batch=n_batch,
                        verbose=False
                    )

                # Configurable timeout
                timeout_seconds = int(os.getenv("GGUF_LOAD_TIMEOUT", "120"))

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(load_model)
                    try:
                        llm = future.result(timeout=timeout_seconds)
                    except concurrent.futures.TimeoutError:
                        raise RuntimeError(
                            f"⏱️  Model loading timed out after {timeout_seconds} seconds.\n"
                            f"\n"
                            f"Model: {Path(model_path).name} ({file_size_mb} MB)\n"
                            f"Config: GPU layers={n_gpu_layers}, Context={n_ctx}\n"
                            f"\n"
                            f"Common causes:\n"
                            f"  - Not enough VRAM (trying to load {file_size_mb}MB model on GPU)\n"
                            f"  - Not enough RAM (needs ~{file_size_mb * 1.2:.0f} MB minimum)\n"
                            f"  - Model is too large for your system\n"
                            f"\n"
                            f"Quick fixes:\n"
                            f"  1. Try CPU only: export GGUF_GPU_LAYERS=0\n"
                            f"  2. Use fewer GPU layers: export GGUF_GPU_LAYERS=20\n"
                            f"  3. Reduce context: export GGUF_CONTEXT_SIZE=2048\n"
                            f"  4. Use a smaller model (TinyLlama, Qwen2-0.5B)\n"
                            f"\n"
                            f"To extend timeout: export GGUF_LOAD_TIMEOUT=300"
                        )

                # Validate the model loaded successfully
                if not hasattr(llm, 'client') and not hasattr(llm, 'model_path'):
                    raise RuntimeError("Model failed to initialize properly")

                logger.info(f"✅ GGUF model loaded successfully")
                print(f"✅ Model loaded!")

                # Save to cache
                _CURRENT_LLM = llm
                _CURRENT_MODEL_NAME = model_name
                return llm

            except concurrent.futures.TimeoutError:
                raise  # Re-raise the timeout error we created above

            except Exception as e:
                # Catch llama-cpp-python errors and make them user-friendly
                error_msg = str(e)

                if "Failed to load model" in error_msg or "Could not load Llama model" in error_msg:
                    raise RuntimeError(
                        f"Invalid or corrupted GGUF file: {Path(model_path).name}\n"
                        f"This file is not a valid GGUF model or is incompatible.\n"
                        f"Common causes:\n"
                        f"  - File is corrupted or incomplete\n"
                        f"  - File is not a GGUF model (wrong format)\n"
                        f"  - Model requires newer llama-cpp-python version\n"
                        f"\n"
                        f"Solutions:\n"
                        f"  1. Re-download the model from HuggingFace\n"
                        f"  2. Remove invalid model: :gguf remove {model_name}\n"
                        f"  3. Update llama-cpp-python: pip install -U llama-cpp-python"
                    )
                elif "out of memory" in error_msg.lower() or "memory" in error_msg.lower():
                    raise RuntimeError(
                        f"Not enough memory to load model: {Path(model_path).name}\n"
                        f"Model size: {file_size_mb} MB (needs ~{file_size_mb * 1.5:.0f} MB RAM)\n"
                        f"Try:\n"
                        f"  - Use CPU only: export GGUF_GPU_LAYERS=0\n"
                        f"  - Reduce context size: export GGUF_CONTEXT_SIZE=2048\n"
                        f"  - Close other applications to free memory\n"
                        f"  - Try TinyLlama (1.1B) or Qwen2-0.5B instead"
                    )
                else:
                    # Unknown error - re-raise with context
                    raise RuntimeError(
                        f"Failed to load GGUF model '{model_name}': {error_msg}\n"
                        f"File: {model_path}\n"
                        f"Size: {file_size_mb} MB\n"
                        f"Remove with: :gguf remove {model_name}"
                    )

        else:
            raise ValueError(f"Unknown backend: {backend}")

class GGUFModelRegistry:
    """Manages GGUF model registry"""

    @staticmethod
    def load_models() -> Dict[str, dict]:
        if not os.path.exists(GGUF_MODELS_FILE):
            return {}
        try:
            with open(GGUF_MODELS_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load GGUF registry: {e}")
            return {}

    @staticmethod
    def save_models(models: Dict[str, dict]):
        try:
            with open(GGUF_MODELS_FILE, 'w') as f:
                json.dump(models, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save GGUF registry: {e}")

    @staticmethod
    def add_model(alias: str, path: str, description: str = ""):
        """
        Add a GGUF model to the registry.

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is invalid
        """
        path_obj = Path(path)

        if not path_obj.exists():
            raise FileNotFoundError(f"File not found: {path}")

        if path_obj.suffix.lower() != ".gguf":
            raise ValueError(f"File must have .gguf extension, got: {path_obj.suffix}")

        # Check file size
        file_size = path_obj.stat().st_size
        if file_size < 1_000_000:  # < 1MB
            raise ValueError(
                f"File appears too small to be a valid GGUF model ({file_size / 1024:.1f} KB).\n"
                f"Valid models are typically at least 100 MB."
            )

        models = GGUFModelRegistry.load_models()

        # Check if alias already exists
        if alias in models:
            existing_path = models[alias]["path"]
            logger.warning(f"Overwriting existing model '{alias}' (was: {existing_path})")

        models[alias] = {
            "path": str(path_obj.absolute()),
            "description": description,
            "size_mb": round(file_size / (1024 * 1024), 1)
        }
        GGUFModelRegistry.save_models(models)

        logger.info(f"✅ Added GGUF model: {alias} ({models[alias]['size_mb']} MB)")
        return models[alias]

    @staticmethod
    def remove_model(alias: str) -> bool:
        """
        Remove a model from registry.

        Returns:
            True if removed, False if not found
        """
        models = GGUFModelRegistry.load_models()
        if alias in models:
            del models[alias]
            GGUFModelRegistry.save_models(models)
            logger.info(f"✅ Removed: {alias}")
            return True
        else:
            logger.warning(f"Model not found: {alias}")
            return False

    @staticmethod
    def list_models() -> List[str]:
        """Get list of registered model aliases"""
        return list(GGUFModelRegistry.load_models().keys())

    @staticmethod
    def get_models() -> List[Dict[str, str]]:
        """
        Get full model information.

        Returns:
            List of dicts with alias, path, description, size_mb
        """
        models = GGUFModelRegistry.load_models()
        return [
            {"alias": alias, **info}
            for alias, info in models.items()
        ]

    @staticmethod
    def get_model_info(alias: str) -> Optional[dict]:
        """Get info for a specific model"""
        return GGUFModelRegistry.load_models().get(alias)

    @staticmethod
    def validate_model(alias: str) -> tuple[bool, str]:
        """
        Validate a registered model.

        Returns:
            (is_valid, error_message)
        """
        models = GGUFModelRegistry.load_models()

        if alias not in models:
            return False, f"Model '{alias}' not in registry"

        model_info = models[alias]
        path = Path(model_info["path"])

        if not path.exists():
            return False, f"File not found: {path}"

        file_size = path.stat().st_size
        if file_size < 1_000_000:
            logger.warning(f"⚠️ File too small ({file_size / 1024:.1f} KB), likely corrupted")

        return True, "Valid"