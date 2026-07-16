"""Encoder persistence (BACKLOG fatal, audits 2026-07-15): train_vision/
train_audio train a separately-passed encoder with a LOCAL AdamW, but
_save_checkpoint wrote only self.model plus the never-stepped self.optimizer
-- the trained encoder weights were never written anywhere and evaporated on
exit. These tests lock the fix: encoder weights ride the vision/audio
checkpoints, resume_from restores them (and the local optimizer state),
the saved optimizer state is the one that actually stepped, and text-only
checkpoints keep their exact prior structure."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from enigma_engine.core.model import Enigma
from enigma_engine.core.model_presets import ForgeConfig
from enigma_engine.training.training import Trainer, TrainingConfig

VISION_DIM = 16
AUDIO_DIM = 12


class _CharTokenizer:
    """Minimal char-level tokenizer (mirrors tests/test_rl_training.py)."""

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.eos_token_id = 0
        self.bos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [2 + (ord(c) % (self.vocab_size - 3)) for c in text][:16] or [2]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(65 + (int(i) % 26)) for i in ids)


class _TinyVisionEncoder(nn.Module):
    """Stand-in for VisionEncoder: exposes the config attrs the train loop
    reads and returns [1, n_patches, VISION_DIM] features."""

    def __init__(self, n_patches: int = 4, image_size: int = 16):
        super().__init__()
        self.config = SimpleNamespace(image_size=image_size, use_pretrained=False)
        self.n_patches = n_patches
        self.proj = nn.Linear(3 * image_size * image_size, n_patches * VISION_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.proj(x.reshape(x.shape[0], -1))
        return out.reshape(x.shape[0], self.n_patches, VISION_DIM)


class _TinyAudioEncoder(nn.Module):
    """Stand-in for AudioEncoder: mel [1, n_mels, T] -> [1, n_tokens, AUDIO_DIM].
    No config attr, so preprocess_audio uses AudioEncoderConfig defaults."""

    def __init__(self, n_mels: int = 80, n_tokens: int = 3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(n_tokens)
        self.proj = nn.Linear(n_mels, AUDIO_DIM)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(mel)  # [1, n_mels, n_tokens]
        return self.proj(pooled.transpose(1, 2))


def _tiny_model(seed: int = 0) -> Enigma:
    torch.manual_seed(seed)
    return Enigma(
        ForgeConfig(
            vocab_size=64, dim=32, n_layers=2, n_heads=2,
            max_seq_len=64, dropout=0.0, use_gradient_checkpointing=False,
            vision_hidden_size=VISION_DIM, audio_hidden_size=AUDIO_DIM,
        )
    )


def _config(checkpoint_dir, **overrides) -> TrainingConfig:
    kwargs = dict(
        epochs=1,
        batch_size=1,
        learning_rate=1e-3,
        warmup_steps=1,
        checkpoint_dir=str(checkpoint_dir),
        log_every=0,
        use_amp=False,
        use_gradient_checkpointing=False,
        seed=7,
    )
    kwargs.update(overrides)
    return TrainingConfig(**kwargs)


def _vision_data() -> list[dict]:
    from PIL import Image

    return [
        {"image": Image.new("RGB", (8, 8), color=(200, 40, 40)), "text": "a red square"},
        {"image": Image.new("RGB", (8, 8), color=(40, 40, 200)), "text": "a blue square"},
    ]


def _audio_data() -> list[dict]:
    torch.manual_seed(3)
    return [
        {"audio": torch.randn(2048), "text": "hello there"},
        {"audio": torch.randn(2048), "text": "good morning"},
    ]


def _snapshot(module: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in module.state_dict().items()}


def _same_state(a: dict, b: dict) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


@pytest.fixture(scope="module")
def vision_run(tmp_path_factory):
    """One tiny train_vision run shared by the read-only vision tests."""
    ckpt_dir = tmp_path_factory.mktemp("vision_ckpt")
    model = _tiny_model()
    trainer = Trainer(model, _CharTokenizer(64), _config(ckpt_dir))
    encoder = _TinyVisionEncoder()
    initial_encoder = _snapshot(encoder)

    state = trainer.train_vision(encoder, _vision_data())
    assert not state.abort_reason

    # _checkpoint_stem is the checkpoint_dir name (mktemp appends a counter),
    # so find the best checkpoint by suffix instead of hardcoding the stem.
    candidates = list(ckpt_dir.glob("*_vision_best.pt"))
    assert len(candidates) == 1, f"expected one best checkpoint, found {candidates}"
    ckpt_path = candidates[0]
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    return SimpleNamespace(
        trainer=trainer,
        encoder=encoder,
        initial_encoder=initial_encoder,
        ckpt_path=ckpt_path,
        checkpoint=checkpoint,
    )


def test_vision_checkpoint_contains_trained_encoder(vision_run):
    """(a) The checkpoint holds the encoder weights as they were trained."""
    saved = vision_run.checkpoint.get("vision_encoder_state_dict")
    assert saved is not None, "vision checkpoint is missing vision_encoder_state_dict"
    assert _same_state(saved, vision_run.encoder.state_dict())
    # Training must have actually moved the encoder, otherwise the equality
    # above would pass on an untouched module.
    assert not _same_state(saved, vision_run.initial_encoder)


def test_vision_checkpoint_saves_stepped_optimizer(vision_run):
    """(d) The saved optimizer state belongs to the local optimizer that
    stepped, not the untouched self.optimizer from __init__."""
    opt_state = vision_run.checkpoint["optimizer_state_dict"]
    assert len(opt_state["state"]) > 0, "saved optimizer never stepped"
    # self.optimizer never steps during vision training; before the fix its
    # (empty) state is what landed in the checkpoint.
    assert len(vision_run.trainer.optimizer.state_dict()["state"]) == 0


def test_vision_resume_restores_encoder_and_model(vision_run, tmp_path):
    """(b) resume_from restores encoder + model weights and step counters."""
    model2 = _tiny_model(seed=1)
    torch.manual_seed(1)
    encoder2 = _TinyVisionEncoder()
    saved_encoder = vision_run.checkpoint["vision_encoder_state_dict"]
    assert not _same_state(encoder2.state_dict(), saved_encoder)  # sanity

    trainer2 = Trainer(model2, _CharTokenizer(64), _config(tmp_path / "resume"))
    # Saved epoch == config.epochs, so the loop body never runs: what comes
    # out is exactly what the load path restored.
    state = trainer2.train_vision(encoder2, _vision_data(), resume_from=vision_run.ckpt_path)

    assert _same_state(encoder2.state_dict(), saved_encoder)
    assert _same_state(model2.state_dict(), vision_run.checkpoint["model_state_dict"])
    saved_state = vision_run.checkpoint["training_state"]
    assert state.epoch == saved_state["epoch"]
    assert state.step == saved_state["step"]
    assert state.best_loss == saved_state["best_loss"]


def test_vision_resume_refuses_text_only_checkpoint(vision_run, tmp_path):
    """A text-only checkpoint has no encoder weights; resuming from it must
    fail loudly instead of silently restarting the encoder from scratch."""
    model2 = _tiny_model(seed=2)
    trainer2 = Trainer(model2, _CharTokenizer(64), _config(tmp_path / "refuse"))
    text_ckpt = tmp_path / "text_only.pt"
    trainer2._save_checkpoint(text_ckpt)
    assert text_ckpt.exists()

    with pytest.raises(ValueError, match="vision_encoder_state_dict"):
        trainer2.train_vision(_TinyVisionEncoder(), _vision_data(), resume_from=text_ckpt)


def test_text_checkpoint_structure_unchanged(tmp_path):
    """(c) Text-only checkpoints carry no encoder keys and load as before."""
    model = _tiny_model()
    trainer = Trainer(model, _CharTokenizer(64), _config(tmp_path / "text"))
    path = tmp_path / "text_model.pt"
    trainer._save_checkpoint(path)

    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    assert not any("encoder" in k for k in ckpt)
    expected = {
        "model_state_dict",
        "optimizer_state_dict",
        "training_state",
        "training_config",
        "model_config",
        "config",
    }
    if trainer.scheduler is not None:
        expected.add("scheduler_state_dict")
    if trainer.scaler is not None:
        expected.add("scaler_state_dict")
    assert set(ckpt.keys()) == expected

    # Round-trip through the existing text load path
    model2 = _tiny_model(seed=1)
    trainer2 = Trainer(model2, _CharTokenizer(64), _config(tmp_path / "text2"))
    trainer2.load_checkpoint(path)
    assert _same_state(model2.state_dict(), ckpt["model_state_dict"])


def test_audio_checkpoint_and_resume_roundtrip(tmp_path):
    """train_audio has the identical local-optimizer pattern: encoder weights
    must ride the checkpoint and resume_from must restore them."""
    ckpt_dir = tmp_path / "audio"
    model = _tiny_model()
    trainer = Trainer(model, _CharTokenizer(64), _config(ckpt_dir))
    encoder = _TinyAudioEncoder()
    initial_encoder = _snapshot(encoder)

    state = trainer.train_audio(encoder, _audio_data())
    assert not state.abort_reason

    candidates = list(ckpt_dir.glob("*_audio_best.pt"))
    assert len(candidates) == 1, f"expected one best checkpoint, found {candidates}"
    ckpt = torch.load(candidates[0], map_location="cpu", weights_only=True)

    saved = ckpt.get("audio_encoder_state_dict")
    assert saved is not None, "audio checkpoint is missing audio_encoder_state_dict"
    assert _same_state(saved, encoder.state_dict())
    assert not _same_state(saved, initial_encoder)
    # Second-bug sibling check: the stepped local optimizer was saved
    assert len(ckpt["optimizer_state_dict"]["state"]) > 0
    assert len(trainer.optimizer.state_dict()["state"]) == 0

    model2 = _tiny_model(seed=1)
    torch.manual_seed(1)
    encoder2 = _TinyAudioEncoder()
    assert not _same_state(encoder2.state_dict(), saved)  # sanity
    trainer2 = Trainer(model2, _CharTokenizer(64), _config(tmp_path / "audio_resume"))
    trainer2.train_audio(encoder2, _audio_data(), resume_from=candidates[0])

    assert _same_state(encoder2.state_dict(), saved)
    assert _same_state(model2.state_dict(), ckpt["model_state_dict"])
