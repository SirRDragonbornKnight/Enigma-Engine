"""
================================================================================
THE CHAMBER OF CONFIGURATION - DEFAULT SETTINGS
================================================================================

Deep within the foundations of Enigma AI Engine lies the Chamber of Configuration,
where the sacred scrolls of default settings are inscribed. These ancient
texts define the very nature of the forge - its paths, its powers, and its
boundaries.

FILE: enigma_engine/config/defaults.py
TYPE: Configuration Management
MAIN EXPORT: CONFIG dictionary

    THE HIERARCHY OF TRUTH:

    When Enigma AI Engine awakens, it reads configuration from three sources,
    each layer overriding the last:

        1. DEFAULT VALUES (this file) - The ancient foundation
        2. USER CONFIG FILE (forge_config.json) - Custom modifications
        3. ENVIRONMENT VARIABLES (FORGE_*) - Runtime overrides

    THE SACRED SECTIONS:

    PATHS           - Where treasures are stored (data, models, memory)
    ARCHITECTURE    - The shape of the AI's mind (layers, dimensions)
    TRAINING        - How the AI learns (learning rate, epochs)
    INFERENCE       - How the AI speaks (temperature, sampling)
    HARDWARE        - What powers the forge (CPU, GPU, precision)
    SECURITY        - What must never be touched (blocked paths)

USAGE:
    from enigma_engine.config import CONFIG, get_config, update_config

    # Read a value
    data_dir = CONFIG["data_dir"]
    # OR
    data_dir = get_config("data_dir", default="data")

    # Update values (in memory only)
    update_config({"temperature": 0.9})

    # Persist changes to file
    save_config()

ENVIRONMENT VARIABLES:
    FORGE_DATA_DIR, FORGE_MODELS_DIR, FORGE_DEVICE, etc.
    Legacy ENIGMA_* variables are also supported.

WARNING: The blocked_paths and blocked_patterns settings are sacred
         protections that the AI cannot modify at runtime.
"""

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# THE FOUNDATION STONE - Base Directory
# =============================================================================
# All paths in Enigma AI Engine are relative to this sacred point.

BASE_DIR = Path(__file__).parent.parent.parent


# =============================================================================
# THE GREAT CODEX - Default Configuration
# =============================================================================
# These are the foundational settings upon which Enigma AI Engine is built.
# Each section governs a different aspect of the forge's operation.


class _LazyConfig(dict):
    """Dict that triggers lazy initialization on first read access."""

    def __getitem__(self, key):
        _ensure_initialized()
        return super().__getitem__(key)

    def get(self, key, default=None):
        _ensure_initialized()
        return super().get(key, default)

    def __contains__(self, key):
        _ensure_initialized()
        return super().__contains__(key)

    def __iter__(self):
        _ensure_initialized()
        return super().__iter__()

    def items(self):
        _ensure_initialized()
        return super().items()

    def values(self):
        _ensure_initialized()
        return super().values()

    def keys(self):
        _ensure_initialized()
        return super().keys()

    def update(self, __m=(), **kwargs):
        _ensure_initialized()
        return super().update(__m, **kwargs)

    def pop(self, key, *args):
        _ensure_initialized()
        return super().pop(key, *args)

    def popitem(self):
        _ensure_initialized()
        return super().popitem()

    def clear(self):
        _ensure_initialized()
        return super().clear()

    def setdefault(self, key, default=None):
        _ensure_initialized()
        return super().setdefault(key, default)


CONFIG = _LazyConfig(
    {
        # =========================================================================
        # THE MAP OF REALMS - Path Configuration
        # =========================================================================
        # Every treasure has its place. These paths define where Enigma AI Engine
        # stores its knowledge, memories, and creations.
        "root": str(BASE_DIR),
        "data_dir": str(BASE_DIR / "data"),  # Training data, icons, themes
        "info_dir": str(BASE_DIR / "information"),  # Runtime settings, tasks, reminders
        "models_dir": str(BASE_DIR / "models"),
        "memory_dir": str(BASE_DIR / "memory"),
        "outputs_dir": str(BASE_DIR / "outputs"),  # Generated images, audio, video
        "db_path": str(BASE_DIR / "memory" / "memory.db"),
        "vocab_dir": str(BASE_DIR / "enigma_engine" / "vocab_model"),
        "logs_dir": str(BASE_DIR / "logs"),
        "personas_dir": str(BASE_DIR / "data" / "personas"),  # AI persona storage
        # =========================================================================
        # THE ARCHITECT'S BLUEPRINT - Model Architecture
        # =========================================================================
        # These settings define the structure of the AI's mind - how many
        # layers of thought, how wide its neural pathways, how far it can see.
        "default_model": "enigma_engine",
        "embed_dim": 256,
        "depth": 6,
        "num_layers": 6,
        "heads": 8,
        "num_heads": 8,
        "max_len": 2048,
        "ff_mult": 4.0,
        "dropout": 0.0,
        "vocab_size": 32000,
        # =========================================================================
        # THE TEACHER'S WISDOM - Training Defaults
        # =========================================================================
        # When the AI learns, these settings guide its education - how quickly
        # it absorbs knowledge, how many times it studies the texts.
        "learning_rate": 1e-4,
        "batch_size": 32,
        "epochs": 10,
        "warmup_steps": 100,
        "gradient_accumulation_steps": 1,
        "weight_decay": 0.1,
        "max_grad_norm": 1.0,
        "use_amp": True,
        "auto_learn": True,
        "auto_train_threshold": 10,
        # =========================================================================
        # THE ORACLE'S VOICE - Inference Defaults
        # =========================================================================
        # When the AI speaks, these settings color its responses - how creative,
        # how focused, how varied its words shall be.
        "temperature": 0.8,
        "top_k": 50,
        "top_p": 0.9,
        "min_p": 0.0,
        "repetition_penalty": 1.1,
        "max_gen": 8192,
        # Pass 156z9dr (N-14): RAG retrieval backend.  "bm25" uses the
        # built-in BM25/TF-IDF index (no extra deps).  "dense" uses
        # sentence-transformers + faiss-cpu for semantic embeddings;
        # falls back to BM25 with a WARNING if either dep is missing.
        "rag_backend": "bm25",
        # =========================================================================
        # THE MESSENGER'S GATE - Server Defaults
        # =========================================================================
        # The API server allows distant travelers to commune with the AI.
        # These settings control access and security.
        "api_host": "127.0.0.1",
        "api_port": 5000,
        "enable_cors": True,
        "require_api_key": True,  # Require authentication for API access
        "enigma_api_key": None,  # Set via env ENIGMA_API_KEY or forge_config.json
        # =========================================================================
        # THE FORGE'S HEART - Hardware Configuration
        # =========================================================================
        # What powers drive the forge? CPU, GPU, or the mystical MPS of Apple?
        # The precision of calculations affects both speed and quality.
        "device": "auto",  # "auto", "cpu", "cuda", "mps"
        "precision": "auto",  # "auto", "float32", "float16", "bfloat16"
        # Backend selection for neural network operations
        # "auto" - Auto-detect (PyTorch if available, CPU fallback)
        # "torch" - Always use PyTorch
        "nn_backend": "auto",
        # =========================================================================
        # Capability Toggles
        # =========================================================================
        # Features that have backing code set to True, others commented out
        "enable_offloading": False,  # CPU+GPU layer offloading (inference.py supports this)
        "offload_folder": None,  # Folder for offloaded weights (None = temp)
        "max_gpu_layers": None,  # Max layers on GPU (None = auto)
        # =========================================================================
        # Resource Management
        # =========================================================================
        "resource_mode": "performance",  # "minimal", "balanced", "performance"
        "cpu_threads": 0,  # 0 = auto
        "memory_limit_mb": 0,  # 0 = no limit
        "gpu_memory_fraction": 0.85,  # Use 85% of GPU VRAM
        # =========================================================================
        # Security Settings
        # =========================================================================
        "blocked_paths": [
            "C:/Windows",
            "C:/Program Files",
            "C:/Program Files (x86)",
            "/etc",
            "/usr",
            "/bin",
            "/sbin",
        ],
        "blocked_patterns": [
            "*.exe",
            "*.dll",
            "*.sys",
            "*.pem",
            "*.key",
            "*password*",
            "*secret*",
            "*.env",
            ".git/config",
        ],
        # Plugin allowlist — only these plugin filenames are loaded.
        # Empty list means ALL discovered plugins are loaded (legacy behavior).
        # Example: ["hello.py", "my_tools.py"]
        "trusted_plugins": [],
        # =========================================================================
        # Logging
        # =========================================================================
        "log_level": "INFO",
        "log_to_file": False,
    }
)


# =============================================================================
# CONFIG TYPE VALIDATION — reject wrong-typed user config values
# =============================================================================

# Map of config keys to their expected Python types.
# Built once from the CONFIG defaults dict.  Keys whose default is None
# are skipped (accept any type).
_CONFIG_TYPES: dict[str, type] = {}


def _build_type_map() -> None:
    """Populate _CONFIG_TYPES from the current CONFIG defaults."""
    # Use dict.items() to bypass _LazyConfig proxy — this may be
    # called during initialization while _init_lock is held.
    for key, value in dict.items(CONFIG):
        if value is None:
            continue
        if isinstance(value, bool):
            _CONFIG_TYPES[key] = bool
        elif isinstance(value, int):
            _CONFIG_TYPES[key] = int
        elif isinstance(value, float):
            _CONFIG_TYPES[key] = (int, float)  # type: ignore[assignment]
        elif isinstance(value, str):
            _CONFIG_TYPES[key] = str
        elif isinstance(value, list):
            _CONFIG_TYPES[key] = list
        elif isinstance(value, dict):
            _CONFIG_TYPES[key] = dict


def _validate_config_types(user_config: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *user_config* with wrong-typed values stripped.

    Keys not present in CONFIG defaults are passed through unchanged
    (user extensions).  Keys whose default is None also pass through.
    """
    if not _CONFIG_TYPES:
        _build_type_map()

    cleaned: dict[str, Any] = {}
    for key, value in user_config.items():
        expected = _CONFIG_TYPES.get(key)
        if expected is None:
            # Unknown key or None-default — accept as-is
            cleaned[key] = value
            continue
        if not isinstance(value, expected):
            logger.warning(
                "Config key %r has wrong type: expected %s, got %s — skipped",
                key,
                expected,
                type(value).__name__,
            )
            continue
        cleaned[key] = value
    return cleaned


# =============================================================================
# THE RITUAL OF READING - Loading User Configuration
# =============================================================================


def _load_user_config() -> None:
    """
    The Ritual of Reading User Configuration.

    Searches the realm for custom configuration scrolls (forge_config.json)
    in several sacred locations. The first scroll found is read, and its
    contents override the default settings.

    Search Order:
        1. Current working directory
        2. User's home directory (~/.enigma_engine/)
        3. Enigma AI Engine installation directory
    """
    config_paths = [
        Path.cwd() / "forge_config.json",
        Path.home() / ".enigma_engine" / "config.json",
        BASE_DIR / "forge_config.json",
    ]

    for path in config_paths:
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    user_config = json.load(f)
                if not isinstance(user_config, dict):
                    logger.warning(f"Config in {path} is not a dictionary, skipping")
                    continue
                user_config = _validate_config_types(user_config)
                # Bypass _LazyConfig.update() to avoid deadlock —
                # we are called from inside _init_lock.
                dict.update(CONFIG, user_config)
                logger.info(f"Loaded config from {path}")
                return
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON in config file {path}: {e}")
            except Exception as e:
                logger.warning(f"Failed to load config from {path}: {e}")


# =============================================================================
# THE RITUAL OF ENVIRONMENT - Loading Environment Variables
# =============================================================================


def _load_env_config() -> None:
    """
    The Ritual of Environment Variable Reading.

    Scans the environment for FORGE_* variables that override configuration.
    This allows runtime customization without modifying files - useful for
    containers, CI/CD, and temporary adjustments.

    Supported Variables:
        FORGE_DATA_DIR, FORGE_MODELS_DIR, FORGE_MEMORY_DIR,
        FORGE_DEVICE, FORGE_API_HOST, FORGE_API_PORT,
        FORGE_LOG_LEVEL, ENIGMA_API_KEY
    """
    env_mappings = {
        "FORGE_DATA_DIR": "data_dir",
        "FORGE_MODELS_DIR": "models_dir",
        "FORGE_MEMORY_DIR": "memory_dir",
        "FORGE_DEVICE": "device",
        "FORGE_API_HOST": "api_host",
        "FORGE_API_PORT": "api_port",
        "FORGE_LOG_LEVEL": "log_level",
        "ENIGMA_API_KEY": "enigma_api_key",
    }

    for env_var, config_key in env_mappings.items():
        if env_var in os.environ:
            value: Any = os.environ[env_var]
            # Type conversion with validation for port numbers
            if config_key == "api_port":
                try:
                    value = int(value)
                    if not (1 <= value <= 65535):
                        logger.warning(f"Invalid port {value}, using default")
                        continue
                except ValueError:
                    logger.warning(f"Invalid port value {value}, using default")
                    continue
            # Bypass _LazyConfig proxy — called from inside _init_lock.
            dict.__setitem__(CONFIG, config_key, value)


# =============================================================================
# THE PUBLIC INTERFACE - Configuration Access Functions
# =============================================================================


def get_config(key: str, default: Any = None) -> Any:
    """
    Retrieve a value from the configuration.

    Args:
        key: The configuration key to retrieve
        default: Value to return if key is not found

    Returns:
        The configuration value, or default if not found
    """
    return CONFIG.get(key, default)


def update_config(updates: dict[str, Any]) -> None:
    """
    Update configuration with new values (in memory only).

    Use save_config() to persist changes to disk.

    Args:
        updates: Dictionary of configuration updates

    Raises:
        TypeError: If updates is not a dictionary
    """
    if not isinstance(updates, dict):
        raise TypeError(f"updates must be a dict, got {type(updates).__name__}")
    CONFIG.update(updates)


def save_config(path: Optional[str] = None) -> None:
    """
    Persist current configuration to a JSON file.

    Args:
        path: Destination path (default: forge_config.json in base directory)

    Raises:
        IOError: If the file cannot be written
    """
    if path is None:
        path = str(BASE_DIR / "forge_config.json")

    try:
        config_path = Path(path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        from enigma_engine.core.safe_save import atomic_write_json

        atomic_write_json(config_path, CONFIG)
    except Exception as e:
        raise OSError(f"Failed to save config to {path}: {e}") from e


# =============================================================================
# THE AWAKENING - Lazy Module Initialization
# =============================================================================
# Directory creation and config loading are deferred until CONFIG is first
# accessed.  This avoids filesystem side effects at import time, making the
# module safer to import in tests and scripts.

_initialized = False
_init_lock = threading.Lock()


def _ensure_initialized() -> None:
    """Run the one-time initialization (dirs + user config + env vars)."""
    global _initialized
    if _initialized:
        return
    with _init_lock:
        if _initialized:
            return

        # Use dict.__getitem__ to bypass _LazyConfig proxy —
        # the proxy calls _ensure_initialized() which would deadlock
        # on the non-reentrant _init_lock we already hold.
        _raw_get = dict.__getitem__
        for dir_key in ["data_dir", "models_dir", "memory_dir", "logs_dir"]:
            try:
                Path(_raw_get(CONFIG, dir_key)).mkdir(parents=True, exist_ok=True)
            except (OSError, PermissionError) as e:
                logger.warning(f"Could not create directory {_raw_get(CONFIG, dir_key)}: {e}")

        _load_user_config()
        _load_env_config()
        _initialized = True
