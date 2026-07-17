"""Her own eyes (eyes.py native path, Phase 4.5 step 5).

Caption generation borrows the served model in the align-trained shape:
[projected patches][BOS -> greedy tokens -> EOS]. Tiny fixtures, no
downloads, no GPU requirement.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from enigma_engine.core.eyes import Eyes, EyesError
from enigma_engine.core.model import Enigma
from enigma_engine.core.model_presets import ForgeConfig

VDIM = 16


class _TinyEncoder(nn.Module):
    """Stand-in for the aligned VisionEncoder: [B,3,16,16] -> [B, 4, VDIM]."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(image_size=16, use_pretrained=False)
        self.proj = nn.Linear(3 * 16 * 16, 4 * VDIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.reshape(x.shape[0], -1)).reshape(x.shape[0], 4, VDIM)


class _TinyTok:
    vocab_size = 64
    eos_token_id = 63
    bos_token_id = 1

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"t{int(i)}" for i in ids)


def _tiny_model(vision: bool = True) -> Enigma:
    torch.manual_seed(7)
    return Enigma(
        ForgeConfig(
            vocab_size=64, dim=32, n_layers=2, n_heads=2,
            max_seq_len=64, dropout=0.0, use_gradient_checkpointing=False,
            vision_hidden_size=VDIM if vision else None,
        )
    )


def test_native_caption_generates_from_her_pipeline():
    pil = pytest.importorskip("PIL.Image")
    eyes = Eyes(model=_tiny_model(), tokenizer=_TinyTok(), encoder=_TinyEncoder())
    cap = eyes.describe(pil.new("RGB", (32, 32), (200, 30, 30)))
    assert isinstance(cap, str) and cap
    # greedy + fixed seed: the same image captions identically
    assert eyes.describe(pil.new("RGB", (32, 32), (200, 30, 30))) == cap


def test_model_without_vision_port_fails_honest():
    with pytest.raises(EyesError, match="vision_projection"):
        Eyes(model=_tiny_model(vision=False), tokenizer=_TinyTok(), encoder=_TinyEncoder())


def test_missing_pieces_fail_honest():
    with pytest.raises(EyesError, match="aligned"):
        Eyes()


def test_factory_injection_still_works():
    eyes = Eyes(captioner_factory=lambda: (lambda img: "a fake caption"))
    pil = pytest.importorskip("PIL.Image")
    assert eyes.describe(pil.new("RGB", (8, 8))) == "a fake caption"
