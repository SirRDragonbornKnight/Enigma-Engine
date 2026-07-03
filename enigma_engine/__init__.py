"""Modkit — the AI's capability backend (the Forge + mods).

This package forges and manages models and capabilities; the models are *run* by
Odysseus + an external runner (Ollama / llama.cpp / vLLM), not here.

(The import package is still named ``enigma_engine`` for now — the package-level
rename to ``modkit`` is a deliberate follow-up to avoid churning every import.)
"""

from .config import CONFIG

__version__ = "2.0.0"
__author__ = "SirRDragonbornKnight"

__all__ = ["CONFIG", "__version__"]
