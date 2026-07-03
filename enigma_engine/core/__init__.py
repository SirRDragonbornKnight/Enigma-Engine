# core package — Modkit's Forge core
"""Modkit core: the Forge (model architecture + training support), tokenizers,
and the mod-command registry.

Model *inference/serving* lives outside Modkit now — trained models are run by
Odysseus + an external runner (Ollama / llama.cpp / vLLM). This package is the
factory, not the engine.
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

# Command registry (used by the mod system / mod_tools)
try:
    from .commands import (
        CommandRegistry,
        CommandResult,
        get_registry,
        parse_commands,
    )
except ImportError:
    CommandRegistry = None
    CommandResult = None
    get_registry = None
    parse_commands = None

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
    "CommandRegistry",
    "CommandResult",
    "get_registry",
    "parse_commands",
    "Enigma",
    "ForgeConfig",
    "create_model",
    "MODEL_PRESETS",
]
