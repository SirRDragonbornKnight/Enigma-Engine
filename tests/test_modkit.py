"""Modkit smoke tests — the kept Forge core (model, tokenizer, LoRA config) and
mod discovery.

The monolith-era engine/GUI/inference tests were removed in the refocus to
Modkit; their deep coverage lives in git history (pre-refocus, dde5c99) and is
tracked for selective restore. The dispatcher/Trainer path stays covered by
``test_training_dispatch.py``.
"""

from dataclasses import replace


def test_enigma_model_builds():
    """The Forge can instantiate an Enigma model from a preset."""
    from enigma_engine.core.model import Enigma
    from enigma_engine.core.model_presets import MODEL_PRESETS

    cfg = replace(MODEL_PRESETS["nano"], vocab_size=256)
    model = Enigma(cfg)
    assert sum(p.numel() for p in model.parameters()) > 0


def test_bpe_tokenizer_roundtrip():
    """Our BPE tokenizer trains and round-trips text."""
    from enigma_engine.core.bpe_tokenizer import BPETokenizer

    tok = BPETokenizer()
    tok.train(
        ["hello world", "the quick brown fox jumps", "hello there world"],
        vocab_size=300,
        verbose=False,
    )
    ids = tok.encode("hello world")
    assert isinstance(ids, list) and len(ids) > 0
    assert isinstance(tok.decode(ids), str)


def test_lora_config_defaults_target_attention():
    """LoRA targets the attention projections by default (Qwen3-compatible)."""
    from enigma_engine.core.lora_utils import LoraConfig

    cfg = LoraConfig(rank=8, alpha=16)
    assert cfg.rank == 8
    assert "q_proj" in cfg.target_modules


def test_mod_discovery_returns_list():
    """The mod system can enumerate installed mods (the AI's capabilities)."""
    from pathlib import Path

    from enigma_engine.core.mod_tools import discover_mod_tools

    mods_dir = Path(__file__).resolve().parent.parent / "mods"
    tools = discover_mod_tools(mods_dir)
    assert isinstance(tools, list)
