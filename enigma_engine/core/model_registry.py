"""
Model Registry - Safe Model Loading and Caching

Provides thread-safe model loading with security checks.
"""

import copy
import hashlib
import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import torch

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Registry for managing AI models on disk.

    Reads/writes to models/registry.json
    """

    def __init__(self, models_dir: Union[str, Path] = None):
        if models_dir is None:
            # Default to models/ in workspace root
            models_dir = Path(__file__).parent.parent.parent / "models"
        self.models_dir = Path(models_dir)
        self.registry_file = self.models_dir / "registry.json"
        self._lock = threading.Lock()
        self.registry = {"models": {}, "created": datetime.now().isoformat()}
        self._load_registry()

    def _load_registry(self):
        """Load registry from disk."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict) or "models" not in data:
                    logger.error("Model registry missing 'models' key, resetting")
                    data = {"models": {}}
                with self._lock:
                    self.registry = data
            except json.JSONDecodeError as e:
                logger.error(f"Model registry corrupted: {e}")
            except OSError as e:
                logger.warning(f"Cannot read model registry: {e}")

    def _save_registry(self):
        """Save registry to disk."""
        from enigma_engine.core.safe_save import atomic_write_json

        with self._lock:
            data = copy.deepcopy(self.registry)
        atomic_write_json(self.registry_file, data)

    def list_models(self) -> dict:
        """Get all registered models (returns a copy)."""
        with self._lock:
            return dict(self.registry.get("models", {}))

    def get_model(self, name: str) -> Optional[dict]:
        """Get info for a specific model (returns a copy)."""
        with self._lock:
            info = self.registry.get("models", {}).get(name)
        if info is None:
            return None
        return dict(info)

    def register_model(self, name: str, info: dict):
        """Register a new model."""
        with self._lock:
            if "models" not in self.registry:
                self.registry["models"] = {}
            self.registry["models"][name] = info
        self._save_registry()

    def remove_model(self, name: str):
        """Remove a model from registry."""
        with self._lock:
            if "models" in self.registry and name in self.registry["models"]:
                del self.registry["models"][name]
            else:
                return
        self._save_registry()

    def model_exists(self, name: str) -> bool:
        """Check if a model exists."""
        with self._lock:
            return name in self.registry.get("models", {})


# Thread-safe lock for model loading
_load_lock = threading.Lock()


def safe_load_weights(
    path: Union[str, Path],
    map_location: Optional[str] = None,
) -> dict[str, Any]:
    """
    Safely load model weights from a file.

    Supports .safetensors (preferred, no code execution risk) and
    .pth/.pt/.bin (loaded with weights_only=True, no fallback).

    Args:
        path: Path to the model file (.pth, .pt, .bin, .safetensors)
        map_location: Device to load to ('cpu', 'cuda', etc.)

    Returns:
        State dict or checkpoint dict

    Raises:
        FileNotFoundError: If the model file doesn't exist
        RuntimeError: If the file cannot be loaded
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    # Determine map_location
    if map_location is None:
        if torch.cuda.is_available():
            map_location = "cuda"
        else:
            map_location = "cpu"

    logger.info(f"Loading weights from {path} to {map_location}")

    try:
        # Safetensors — no pickle, no code execution risk
        if path.suffix == ".safetensors":
            from safetensors.torch import load_file

            checkpoint = load_file(str(path), device=map_location)
        else:
            logger.warning(
                "Loading %s -- .pth/.pt/.bin files use pickle. "
                "Prefer .safetensors format for models from "
                "untrusted sources.",
                path.name,
            )
            # Always weights_only=True — no fallback to insecure loading
            checkpoint = torch.load(
                path,
                map_location=map_location,
                weights_only=True,
            )

        logger.info(f"Successfully loaded weights from {path}")
        return checkpoint

    except Exception as e:
        logger.error(f"Failed to load weights from {path}: {e}")
        raise RuntimeError(f"Failed to load model weights: {e}") from e


def get_state_dict(checkpoint: dict[str, Any], prefix: str = "") -> dict[str, torch.Tensor]:
    """
    Extract state dict from a checkpoint.

    Handles various checkpoint formats:
    - Direct state dict
    - {'model_state_dict': ...}
    - {'state_dict': ...}
    - {'model': ...}

    Args:
        checkpoint: Loaded checkpoint
        prefix: Optional prefix to strip from keys

    Returns:
        State dict
    """
    # Check for common keys
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        # Check for single-key nested checkpoint wrapping a state dict
        if len(checkpoint) == 1:
            nested_key = next(iter(checkpoint))
            nested = checkpoint[nested_key]
            if isinstance(nested, dict):
                logger.info("Unwrapping nested checkpoint key '%s'", nested_key)
                return get_state_dict(nested, prefix)
        # Assume it's already a state dict — warn in case it contains
        # non-tensor entries (optimizer state, config, etc.).
        non_tensor = [k for k in list(checkpoint.keys())[:5] if not isinstance(checkpoint.get(k), torch.Tensor)]
        if non_tensor:
            logger.warning(
                "Checkpoint has no 'model_state_dict'/'state_dict'/'model' "
                "key. Using it as-is, but it may contain non-tensor entries: %s",
                non_tensor,
            )
        state_dict = checkpoint

    if not state_dict:
        logger.warning("Extracted state dict is empty -- checkpoint may be corrupted or in an unrecognized format")

    # Strip prefix if present
    if prefix:
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith(prefix):
                new_state_dict[k[len(prefix) :]] = v
            else:
                new_state_dict[k] = v
        state_dict = new_state_dict

    return state_dict


def get_model_hash(path: Union[str, Path]) -> str:
    """Get a hash of a model file for caching purposes."""
    path = Path(path)
    if not path.exists():
        return ""

    # Use file size and modification time for quick hash (not security-sensitive)
    stat = path.stat()
    return hashlib.md5(
        f"{path}:{stat.st_size}:{stat.st_mtime}".encode(),
        usedforsecurity=False,
    ).hexdigest()


__all__ = [
    "ModelRegistry",
    "get_model_hash",
    "get_state_dict",
    "safe_load_weights",
]
