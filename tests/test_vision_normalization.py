"""The [-1,1]-vs-ImageNet normalization contract (test-suite audit 2026-07-17).

Her from-scratch vision stack trains and serves in [-1, 1]: distillation fed
the student x*2-1, train_vision (and therefore align_vision.py) derives the
flag from encoder config.use_pretrained (False for her encoders), and
eyes.py captions with imagenet_normalize=False. A silent flip anywhere in
that chain keeps every shape correct and the suite green while caption
quality is destroyed -- the audit found NO test pinned it. These pin the
serve (eyes) and train/align (train_vision) links. Honest scope note
(re-audit 2026-07-18): the distill script's own student normalization
(x*2-1 inside distill_vision_encoder.py main()) is structurally out of
test reach and remains unpinned -- re-check it before any re-distillation.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from enigma_engine.core import vision_encoder as ve
from enigma_engine.core.eyes import Eyes
from enigma_engine.core.model import Enigma
from enigma_engine.core.model_presets import ForgeConfig
from enigma_engine.training.training import Trainer, TrainingConfig

# ---------------------------------------------------------------------------
# preprocess_image: the two branches, value-level (hand-computed literals, not
# recomputed with the module's own constants -- an independent pin).
# ---------------------------------------------------------------------------


def _solid(rgb: tuple, size: int = 8):
    from PIL import Image

    return Image.new("RGB", (size, size), color=rgb)


def test_preprocess_native_is_minus_one_to_one():
    x = ve.preprocess_image(_solid((200, 40, 40)), image_size=8, imagenet_normalize=False)
    assert x.shape == (1, 3, 8, 8)
    # pixel/127.5 - 1: 200 -> 0.56863, 40 -> -0.68627
    assert torch.allclose(x[0, 0], torch.full((8, 8), 0.56863), atol=1e-4)
    assert torch.allclose(x[0, 1], torch.full((8, 8), -0.68627), atol=1e-4)
    assert torch.allclose(x[0, 2], torch.full((8, 8), -0.68627), atol=1e-4)
    assert x.min() >= -1.0 and x.max() <= 1.0


def test_preprocess_imagenet_branch_math():
    x = ve.preprocess_image(_solid((200, 40, 40)), image_size=8, imagenet_normalize=True)
    # (pixel/255 - mean) / std with ImageNet stats, hand-computed:
    # R: (200/255 - 0.485) / 0.229 = 1.30705
    # G: (40/255  - 0.456) / 0.224 = -1.33546
    # B: (40/255  - 0.406) / 0.225 = -1.10729
    assert torch.allclose(x[0, 0], torch.full((8, 8), 1.30705), atol=1e-3)
    assert torch.allclose(x[0, 1], torch.full((8, 8), -1.33546), atol=1e-3)
    assert torch.allclose(x[0, 2], torch.full((8, 8), -1.10729), atol=1e-3)


def test_the_two_branches_are_distinguishable():
    """Anti-vacuity: the values these tests pin must actually differ, so a
    branch collapse (both paths running the same math) cannot pass both."""
    img = _solid((128, 128, 128))
    native = ve.preprocess_image(img, image_size=8, imagenet_normalize=False)
    imagenet = ve.preprocess_image(img, image_size=8, imagenet_normalize=True)
    assert (native - imagenet).abs().max() > 0.05


# ---------------------------------------------------------------------------
# eyes.py: the native captioner must feed the encoder [-1,1] input. The
# recording encoder captures the REAL tensor the real preprocess produced;
# flipping imagenet_normalize in eyes.py fails this test.
# ---------------------------------------------------------------------------


class _RecordingEncoder(nn.Module):
    def __init__(self, image_size: int = 16):
        super().__init__()
        self.config = SimpleNamespace(image_size=image_size)
        self.seen: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.seen = x.detach().clone()
        return torch.zeros(x.shape[0], 4, 8)


class _EosModel(nn.Module):
    """forward_multimodal always argmaxes to EOS (id 0) -> caption loop exits
    after one step; the test only cares what the encoder received."""

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))  # gives Eyes a device
        self.vision_projection = nn.Linear(8, 8)

    def forward_multimodal(self, input_ids=None, vision_features=None):
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 8)
        logits[:, -1, 0] = 1.0
        return logits


class _StubTok:
    eos_token_id = 0
    bos_token_id = 1

    def decode(self, ids, skip_special_tokens=True):
        return "stub caption"


def test_eyes_feeds_encoder_native_normalization():
    enc = _RecordingEncoder(image_size=16)
    eyes = Eyes(model=_EosModel(), tokenizer=_StubTok(), encoder=enc)
    assert eyes.describe(_solid((128, 128, 128), size=16)) == "stub caption"

    assert enc.seen is not None and enc.seen.shape == (1, 3, 16, 16)
    # native: 128/127.5 - 1 = 0.00392 on every channel
    assert torch.allclose(enc.seen, torch.full_like(enc.seen, 0.00392), atol=1e-4)
    # and demonstrably NOT the ImageNet transform of the same gray image
    # (per-channel 0.0741 / 0.2052 / 0.4265):
    imagenet = torch.tensor([0.0741, 0.2052, 0.4265]).view(1, 3, 1, 1)
    assert (enc.seen - imagenet).abs().max() > 0.05


# ---------------------------------------------------------------------------
# train_vision (the path align_vision.py runs): the flag must follow
# encoder.config.use_pretrained -- False for her from-scratch encoders.
# ---------------------------------------------------------------------------

VISION_DIM = 16


class _CharTokenizer:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.eos_token_id = 0
        self.bos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [2 + (ord(c) % (self.vocab_size - 3)) for c in text][:16] or [2]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(65 + (int(i) % 26)) for i in ids)


class _TinyVisionEncoder(nn.Module):
    def __init__(self, use_pretrained: bool, image_size: int = 16):
        super().__init__()
        self.config = SimpleNamespace(image_size=image_size, use_pretrained=use_pretrained)
        self.proj = nn.Linear(3 * image_size * image_size, 4 * VISION_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.reshape(x.shape[0], -1)).reshape(x.shape[0], 4, VISION_DIM)


def _tiny_trainer(tmp_path) -> Trainer:
    torch.manual_seed(0)
    model = Enigma(
        ForgeConfig(
            vocab_size=64, dim=32, n_layers=2, n_heads=2,
            max_seq_len=64, dropout=0.0, use_gradient_checkpointing=False,
            vision_hidden_size=VISION_DIM,
        )
    )
    config = TrainingConfig(
        epochs=1, batch_size=1, learning_rate=1e-3, warmup_steps=1,
        checkpoint_dir=str(tmp_path), log_every=0, use_amp=False,
        use_gradient_checkpointing=False, seed=7,
    )
    return Trainer(model, _CharTokenizer(64), config)


def _recorded_flags(monkeypatch, tmp_path, use_pretrained: bool) -> list:
    flags: list = []
    real = ve.preprocess_image

    def recorder(image, image_size=224, imagenet_normalize=False):
        flags.append(imagenet_normalize)
        return real(image, image_size=image_size, imagenet_normalize=imagenet_normalize)

    monkeypatch.setattr(ve, "preprocess_image", recorder)
    data = [
        {"image": _solid((200, 40, 40)), "text": "a red square"},
        {"image": _solid((40, 40, 200)), "text": "a blue square"},
    ]
    state = _tiny_trainer(tmp_path).train_vision(
        _TinyVisionEncoder(use_pretrained=use_pretrained), data
    )
    assert not state.abort_reason
    assert flags, "train_vision never preprocessed an image -- recorder not wired"
    return flags


def test_train_vision_scratch_encoder_gets_native_flag(monkeypatch, tmp_path):
    flags = _recorded_flags(monkeypatch, tmp_path, use_pretrained=False)
    assert all(f is False for f in flags)


def test_train_vision_pretrained_encoder_gets_imagenet_flag(monkeypatch, tmp_path):
    """The tie is to config.use_pretrained, not a hardcoded False: a timm
    backbone (use_pretrained=True) must get ImageNet stats."""
    flags = _recorded_flags(monkeypatch, tmp_path, use_pretrained=True)
    assert all(f is True for f in flags)
