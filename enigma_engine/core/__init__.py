# core package — Enigma core
"""Enigma core: model architecture, training support, and tokenizers.

This is the Forge (build + train). Serving is `serve_enigma.py` at the repo
root (an OpenAI-compatible FastAPI server that loads the checkpoint directly).
"""

# Hardware detection
try:
    from .hardware_detection import HardwareProfile, get_hardware
except ImportError:
    HardwareProfile = None
    get_hardware = None

# Tokenizers
try:
    from .tokenizer import SimpleTokenizer, get_tokenizer
except ImportError:
    SimpleTokenizer = None
    get_tokenizer = None

try:
    from .bpe_tokenizer import BPETokenizer
except ImportError:
    BPETokenizer = None

# Model — lazy so importing the package doesn't pull in torch at startup.
_lazy_cache = {}


def _lazy_load_model():
    from .model import MODEL_PRESETS, Enigma, ForgeConfig, create_model

    return Enigma, ForgeConfig, create_model, MODEL_PRESETS


def __getattr__(name):
    """Lazy-load the torch-dependent model symbols only when accessed."""
    if name in ("Enigma", "ForgeConfig", "create_model", "MODEL_PRESETS"):
        if "model" not in _lazy_cache:
            Enigma, ForgeConfig, create_model, MODEL_PRESETS = _lazy_load_model()
            _lazy_cache["model"] = {
                "Enigma": Enigma,
                "ForgeConfig": ForgeConfig,
                "create_model": create_model,
                "MODEL_PRESETS": MODEL_PRESETS,
            }
        return _lazy_cache["model"][name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "HardwareProfile",
    "get_hardware",
    "SimpleTokenizer",
    "get_tokenizer",
    "BPETokenizer",
    "Enigma",
    "ForgeConfig",
    "create_model",
    "MODEL_PRESETS",
]
