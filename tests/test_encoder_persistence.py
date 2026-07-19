"""Encoder persistence (BACKLOG fatal, audits 2026-07-15): train_vision
trains a separately-passed encoder with a LOCAL AdamW, but
_save_checkpoint wrote only self.model plus the never-stepped self.optimizer
-- the trained encoder weights were never written anywhere and evaporated on
exit. These tests lock the fix: encoder weights ride the vision
checkpoints, resume_from restores them (and the local optimizer state),
the saved optimizer state is the one that actually stepped, and text-only
checkpoints keep their exact prior structure.

(The train_audio twin test retired with the Forge trainer, 2026-07-18;
a future audio-align trainer must re-lock the same contract.)"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from enigma_engine.core.model import Enigma
from enigma_engine.core.model_presets import ForgeConfig
from enigma_engine.training.vision_align import Trainer, TrainingConfig

VISION_DIM = 16
AUDIO_DIM = 12


class _CharTokenizer:
    """Minimal char-level tokenizer."""

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


# ---------------------------------------------------------------------------
# Checkpoint-safety contracts (review fixes 2026-07-19): a failed best-save
# must not advance best_loss, a missing resume path must refuse instead of
# restarting, the encoder resume path must protect its source from periodic
# cleanup, every exit path must leave the model inference-safe, and
# save_every_steps must actually produce a mid-epoch checkpoint.
# ---------------------------------------------------------------------------


def test_missing_resume_path_refuses(tmp_path):
    """resume_from pointing at nothing raises instead of warn-and-restart
    (epoch 1 of a fresh run would overwrite the checkpoints being resumed)."""
    trainer = Trainer(_tiny_model(), _CharTokenizer(64), _config(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="resume_from"):
        trainer.train_vision(_TinyVisionEncoder(), _vision_data(), resume_from=tmp_path / "nope.pt")


def test_missing_resume_path_leaves_model_inference_safe(tmp_path):
    """The resume refusal fires after the freeze pass; the finally must
    re-enable requires_grad and drop back to eval on the way out."""
    model = _tiny_model()
    trainer = Trainer(model, _CharTokenizer(64), _config(tmp_path / "missing2"))
    encoder = _TinyVisionEncoder()
    with pytest.raises(FileNotFoundError):
        trainer.train_vision(encoder, _vision_data(), resume_from=tmp_path / "nope.pt")
    assert all(p.requires_grad for p in model.parameters())
    assert not model.training
    assert not encoder.training


def test_abort_restores_requires_grad_and_eval(tmp_path):
    """A max_loss abort must not leave the model frozen or the encoder in
    train() mode -- a later training pass over the same objects would
    silently optimize nothing."""
    model = _tiny_model()
    trainer = Trainer(model, _CharTokenizer(64), _config(tmp_path / "abort", max_loss=1e-9))
    encoder = _TinyVisionEncoder()
    state = trainer.train_vision(encoder, _vision_data())
    assert state.abort_reason  # the run must actually have aborted
    assert all(p.requires_grad for p in model.parameters())
    assert not model.training
    assert not encoder.training


def test_failed_best_saves_retry_and_flag_the_run(tmp_path, monkeypatch):
    """With every best-save failing, each epoch still retries the write
    (best_written stays inf) and the run ends with abort_reason set --
    never a silent success with no artifact on disk. state.best_loss keeps
    pure metric semantics so early stopping is unaffected by disk health."""
    import enigma_engine.core.safe_save as safe_save

    model = _tiny_model()
    trainer = Trainer(model, _CharTokenizer(64), _config(tmp_path / "allfail", epochs=2))
    calls = {"n": 0}

    def failing_save(obj, path, *args, **kwargs):
        calls["n"] += 1
        raise OSError("disk full (simulated)")

    monkeypatch.setattr(safe_save, "atomic_torch_save", failing_save)
    state = trainer.train_vision(_TinyVisionEncoder(), _vision_data())
    assert calls["n"] == 2  # every epoch retried the unwritten best
    assert not list((tmp_path / "allfail").glob("*_vision_best.pt"))
    assert "never reached disk" in state.abort_reason
    assert state.best_loss != float("inf")  # the metric still tracked


def test_failed_best_save_retries_and_recovers(tmp_path, monkeypatch):
    """A transient save failure heals: a later epoch rewrites the best
    checkpoint because best_written (not the metric) gates the save."""
    import enigma_engine.core.safe_save as safe_save

    model = _tiny_model()
    trainer = Trainer(model, _CharTokenizer(64), _config(tmp_path / "flaky", epochs=2))
    real_save = safe_save.atomic_torch_save
    calls = {"n": 0}

    def flaky_save(obj, path, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("disk full (simulated)")
        return real_save(obj, path, *args, **kwargs)

    monkeypatch.setattr(safe_save, "atomic_torch_save", flaky_save)
    state = trainer.train_vision(_TinyVisionEncoder(), _vision_data())
    assert calls["n"] >= 2
    assert list((tmp_path / "flaky").glob("*_vision_best.pt"))
    assert state.best_loss != float("inf")


def test_step_checkpoint_resume_winds_schedule_back(tmp_path):
    """Resuming a mid-epoch rolling checkpoint replays the interrupted
    epoch from its start: the step counter (and with it the LR schedule)
    winds back to the epoch boundary instead of double-advancing."""
    ckpt_dir = tmp_path / "wind"
    model = _tiny_model()
    trainer = Trainer(model, _CharTokenizer(64), _config(ckpt_dir, save_every_steps=1))
    state = trainer.train_vision(_TinyVisionEncoder(), _vision_data())
    clean_steps = state.step
    rolling = list(ckpt_dir.glob("*_vision_step.pt"))
    assert len(rolling) == 1
    saved = torch.load(rolling[0], map_location="cpu", weights_only=True)["training_state"]
    assert saved["epoch_start_step"] < saved["step"]  # genuinely mid-epoch

    model2 = _tiny_model(seed=1)
    trainer2 = Trainer(model2, _CharTokenizer(64), _config(tmp_path / "wind2"))
    state2 = trainer2.train_vision(_TinyVisionEncoder(), _vision_data(), resume_from=rolling[0])
    # Without the wind-back the replayed epoch stacks on the loaded step
    # count (clean_steps * 2); with it the totals match a clean run.
    assert state2.step == clean_steps


def test_load_checkpoint_accepts_str_path(tmp_path):
    """load_checkpoint worked with str paths pre-hardening; the .keep
    protection must not have broken that."""
    model = _tiny_model()
    trainer = Trainer(model, _CharTokenizer(64), _config(tmp_path / "strpath"))
    path = tmp_path / "text_model.pt"
    assert trainer._save_checkpoint(path)

    trainer2 = Trainer(_tiny_model(seed=1), _CharTokenizer(64), _config(tmp_path / "strpath2"))
    trainer2.load_checkpoint(str(path))  # must not raise
    assert (tmp_path / "text_model.pt.keep").exists()


def test_save_every_steps_fires_on_remainder_flush(tmp_path):
    """With max_grad_accumulation larger than the batches per epoch, every
    optimizer step comes from the end-of-epoch remainder flush; the rolling
    save must fire there too, not only at accumulation boundaries."""
    model = _tiny_model()
    trainer = Trainer(
        model,
        _CharTokenizer(64),
        _config(tmp_path / "flush", save_every_steps=1, max_grad_accumulation=8),
    )
    state = trainer.train_vision(_TinyVisionEncoder(), _vision_data())
    assert not state.abort_reason
    assert list((tmp_path / "flush").glob("*_vision_step.pt"))


def test_save_checkpoint_missing_encoder_key_raises(tmp_path):
    """encoder without encoder_key is a call-site bug: it must raise, not
    be swallowed into the return-False disk-failure path."""
    trainer = Trainer(_tiny_model(), _CharTokenizer(64), _config(tmp_path / "keyerr"))
    with pytest.raises(ValueError, match="encoder_key"):
        trainer._save_checkpoint(tmp_path / "x.pt", encoder=_TinyVisionEncoder())


def test_fresh_write_supersedes_stale_keep_marker(tmp_path):
    """Overwriting a checkpoint file deletes a leftover .keep marker: the
    bytes it protected are gone, and a permanent marker would make
    periodic cleanup a dead letter for that slot."""
    trainer = Trainer(_tiny_model(), _CharTokenizer(64), _config(tmp_path / "stale"))
    path = tmp_path / "gen.pt"
    marker = tmp_path / "gen.pt.keep"
    marker.write_text("protected - resumed for training", encoding="utf-8")
    assert trainer._save_checkpoint(path)
    assert not marker.exists()


def test_encoder_resume_writes_keep_marker(vision_run, tmp_path):
    """Resuming via _load_encoder_checkpoint protects the source checkpoint
    from _cleanup_periodic_checkpoints, matching load_checkpoint."""
    model2 = _tiny_model(seed=3)
    trainer2 = Trainer(model2, _CharTokenizer(64), _config(tmp_path / "keep"))
    trainer2.train_vision(_TinyVisionEncoder(), _vision_data(), resume_from=vision_run.ckpt_path)
    marker = vision_run.ckpt_path.parent / (vision_run.ckpt_path.name + ".keep")
    assert marker.exists()


def test_save_every_steps_writes_rolling_checkpoint(tmp_path):
    """save_every_steps > 0 produces a mid-epoch rolling checkpoint that
    carries the encoder weights and sits outside the periodic-cleanup
    pattern."""
    import re

    model = _tiny_model()
    trainer = Trainer(model, _CharTokenizer(64), _config(tmp_path / "step", save_every_steps=1))
    encoder = _TinyVisionEncoder()
    state = trainer.train_vision(encoder, _vision_data())
    assert not state.abort_reason
    step_ckpts = list((tmp_path / "step").glob("*_vision_step.pt"))
    assert len(step_ckpts) == 1
    ckpt = torch.load(step_ckpts[0], map_location="cpu", weights_only=True)
    assert "vision_encoder_state_dict" in ckpt
    stem = trainer._checkpoint_stem
    assert not re.match(rf"^{re.escape(stem + '_vision')}(\d+)\.pt$", step_ckpts[0].name)
