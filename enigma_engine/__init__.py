"""Enigma Engine — a from-scratch decoder-only LLM (the Forge).

This package builds, trains, and tokenizes models. Serving is `serve_enigma.py`
at the repo root (an OpenAI-compatible FastAPI server that loads the checkpoint).
"""

from .config import CONFIG

__version__ = "2.0.0"
__author__ = "SirRDragonbornKnight"

__all__ = ["CONFIG", "__version__"]
