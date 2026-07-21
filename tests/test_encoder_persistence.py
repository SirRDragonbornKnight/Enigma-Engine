"""Encoder persistence (BACKLOG fatal, audits 2026-07-15): train_vision
trains a separately-passed encoder with a LOCAL AdamW, but
_save_checkpoint wrote only self.model plus the never-stepped self.optimizer
-- the trained encoder weights were never written anywhere and evaporated on
exit. These tests lock the fix: encoder weights ride the vision
checkpoints, resume_from restores them (and the local optimizer state),
the saved optimizer state is the one that actually stepped, and text-only
checkpoints keep their exact prior structure.

(The train_audio twin retired with the Forge trainer 2026-07-18 and was
re-locked 2026-07-19 against the rebuilt encoder-align train_audio -- see
the audio section at the bottom.)"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from enigma_engine.core.model import Enigma
from enigma_engine.core.model_presets import ForgeConfig
from enigma_engine.training.encoder_align import Trainer, TrainingConfig

VISION_DIM = 16
# The audio port on the tiny models feeds the audio twin tests at the
# bottom: the audio-align rebuild (2026-07-19) re-pins the same contract
# against these fixtures.
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


# ---------------------------------------------------------------------------
# Pre-align fix batch (review 2026-07-19, round 2): token-weighted val loss,
# refuse-don't-lie config knobs, and the decode-overlap pipeline's contracts
# (determinism, per-sample failure isolation).
# ---------------------------------------------------------------------------


def test_val_loss_is_batch_size_invariant(tmp_path):
    """Token-weighted validation: the same val set gives the same val loss
    regardless of batch grouping -- a ragged tail batch must not be
    overweighted, because this number drives best-checkpoint selection and
    early stopping."""
    from PIL import Image

    # Train captions collapse to 1 token, so every train sample is dropped
    # in-loop and no optimizer step runs: both runs evaluate IDENTICAL
    # initial weights and only the val aggregation differs.
    train = [{"image": Image.new("RGB", (8, 8), color=(9, 40, 40)), "text": "a"} for _ in range(2)]
    val = [
        {"image": Image.new("RGB", (8, 8), color=(30 * i % 255, 80, 120)), "text": t}
        for i, t in enumerate(["ab", "cd", "ef", "gh", "a much longer caption here"])
    ]
    losses = []
    for batch_size in (2, 3):
        model = _tiny_model(seed=5)
        torch.manual_seed(5)
        encoder = _TinyVisionEncoder()
        trainer = Trainer(model, _CharTokenizer(64), _config(tmp_path / f"bs{batch_size}", batch_size=batch_size))
        state = trainer.train_vision(encoder, train, val_data=val)
        assert state.validation_losses, "validation never ran"
        losses.append(state.validation_losses[-1])
    assert losses[0] == pytest.approx(losses[1], rel=1e-5)


def test_validate_refuses_unimplemented_forge_knobs(tmp_path):
    """Knobs the vision-align loop does not implement are refused instead
    of being stamped into every checkpoint's training_config as if they
    had applied (the same refuse-don't-lie rule as ema/swa/lisa)."""
    for kwargs, needle in (
        ({"label_smoothing": 0.05}, "label_smoothing"),
        ({"gradient_noise_eta": 0.01}, "gradient_noise_eta"),
        ({"bpe_dropout": 0.1}, "bpe_dropout"),
        ({"schedule_type": "wsd"}, "schedule_type"),
    ):
        with pytest.raises(ValueError, match=needle):
            _config(tmp_path / "knobs", **kwargs).validate()
    _config(tmp_path / "knobs").validate()  # the inert defaults stay accepted


def _png_dataset(tmp_path, n: int = 6) -> list[dict]:
    """n tiny PNGs on disk -- path inputs exercise the probe pre-pass and
    the pooled decode (in-memory PIL objects take the inline path)."""
    from PIL import Image

    tmp_path.mkdir(parents=True, exist_ok=True)
    data = []
    for i in range(n):
        p = tmp_path / f"img{i}.png"
        Image.new("RGB", (8, 8), color=(40 * i % 255, 90, 130)).save(p)
        data.append({"image": str(p), "text": f"sample caption {i}"})
    return data


def test_train_vision_deterministic_with_seed(tmp_path):
    """Two identical seeded runs produce identical loss trajectories: the
    decode-overlap pipeline must consume the augmentation RNG on the main
    thread in batch order, exactly like the serial loop it replaced. Path
    inputs, so the pooled decode path is the one under test."""
    data = _png_dataset(tmp_path / "imgs")
    results = []
    for run in range(2):
        model = _tiny_model(seed=3)
        torch.manual_seed(3)
        encoder = _TinyVisionEncoder()
        trainer = Trainer(model, _CharTokenizer(64), _config(tmp_path / f"det{run}", epochs=2, batch_size=2))
        state = trainer.train_vision(encoder, data)
        assert not state.abort_reason
        results.append((state.training_losses, state.validation_losses))
    assert results[0] == results[1]


def test_bad_image_skips_sample_not_run(tmp_path):
    """A sample whose inline decode fails is skipped with a warning; the
    batch and the run continue (per-sample failure isolation survived the
    overlap rework)."""
    data = _vision_data() + [{"image": object(), "text": "not decodable"}]
    model = _tiny_model()
    trainer = Trainer(model, _CharTokenizer(64), _config(tmp_path / "bad", batch_size=3))
    encoder = _TinyVisionEncoder()
    initial = _snapshot(encoder)
    state = trainer.train_vision(encoder, data)
    assert not state.abort_reason
    assert not _same_state(encoder.state_dict(), initial)  # the good samples trained


def test_bad_pool_decode_skips_sample_not_run(tmp_path, monkeypatch):
    """Same isolation for the POOLED decode path: a worker-side decode
    failure surfaces as a per-sample skip, not a crashed run."""
    import enigma_engine.core.vision_encoder as ve

    data = _png_dataset(tmp_path / "imgs", n=4)
    bad_path = data[-1]["image"]
    real_pre = ve.preprocess_image
    raises = {"n": 0}

    def flaky_pre(image_ref, **kwargs):
        if str(image_ref) == bad_path:
            raises["n"] += 1
            raise RuntimeError("decode blew up (simulated)")
        return real_pre(image_ref, **kwargs)

    monkeypatch.setattr(ve, "preprocess_image", flaky_pre)
    model = _tiny_model()
    trainer = Trainer(model, _CharTokenizer(64), _config(tmp_path / "badpool", batch_size=2))
    encoder = _TinyVisionEncoder()
    initial = _snapshot(encoder)
    state = trainer.train_vision(encoder, data)
    assert raises["n"] == 1  # the fault really fired on the pool path
    assert not state.abort_reason
    assert not _same_state(encoder.state_dict(), initial)


class _StopDuringValEncoder(_TinyVisionEncoder):
    """Encoder that presses STOP on its Nth forward call -- lets a test
    land the stop deterministically inside the validation pass."""

    def __init__(self, stop_at_call: int):
        super().__init__()
        self.calls = 0
        self.stop_at = stop_at_call
        self.trainer = None

    def forward(self, x):
        self.calls += 1
        if self.trainer is not None and self.calls == self.stop_at:
            self.trainer.request_stop()
        return super().forward(x)


def test_stop_during_validation_does_not_rank_the_epoch(tmp_path):
    """A stop that interrupts validation leaves the epoch unranked: the
    train mean must not stand in for the val lineage and overwrite the
    best checkpoint with the stop-epoch weights."""
    from PIL import Image

    train = [
        {"image": Image.new("RGB", (8, 8), color=(200, 40, 40)), "text": "a red square"},
        {"image": Image.new("RGB", (8, 8), color=(40, 40, 200)), "text": "a blue square"},
    ]
    val = [
        {"image": Image.new("RGB", (8, 8), color=(20, 160, 60)), "text": "a green square"},
        {"image": Image.new("RGB", (8, 8), color=(220, 200, 40)), "text": "a yellow square"},
    ]
    model = _tiny_model()
    # Call schedule (batch_size=1): epoch1 train fwd 1-2, val fwd 3-4;
    # epoch2 train fwd 5-6, val batch 1 = call 7 -> stop lands mid-val.
    encoder = _StopDuringValEncoder(stop_at_call=7)
    trainer = Trainer(
        model, _CharTokenizer(64), _config(tmp_path / "stopval", epochs=2, learning_rate=5e-2)
    )
    encoder.trainer = trainer
    state = trainer.train_vision(encoder, train, val_data=val)

    # The stop really landed inside epoch 2's val pass (call 7 = its first
    # val batch; nothing runs after the guard) -- pins the call arithmetic
    # so schedule drift can't make this test pass vacuously.
    assert encoder.calls == 7
    # Epoch 1 completed and set the val-based best; epoch 2's interrupted
    # val pass must not have re-ranked anything.
    assert len(state.validation_losses) == 1
    assert state.best_loss == pytest.approx(state.validation_losses[0])
    best = list((tmp_path / "stopval").glob("*_vision_best.pt"))
    assert len(best) == 1
    saved = torch.load(best[0], map_location="cpu", weights_only=True)["training_state"]
    assert saved["epoch"] == 1  # the best on disk is epoch 1's, not the stop epoch's


def test_stop_after_completed_validation_still_ranks_the_epoch(tmp_path):
    """The dual of the guard: a stop that lands during the LAST val batch's
    forward leaves a fully-computed val metric -- that epoch must still
    rank (discarding it silently loses a real best)."""
    from PIL import Image

    train = [
        {"image": Image.new("RGB", (8, 8), color=(200, 40, 40)), "text": "a red square"},
        {"image": Image.new("RGB", (8, 8), color=(40, 40, 200)), "text": "a blue square"},
    ]
    val = [
        {"image": Image.new("RGB", (8, 8), color=(20, 160, 60)), "text": "a green square"},
        {"image": Image.new("RGB", (8, 8), color=(220, 200, 40)), "text": "a yellow square"},
    ]
    model = _tiny_model()
    # Call 8 = epoch 2's SECOND (final) val batch: the pass completes, the
    # stop is only seen afterwards. Default lr (1e-3): epoch 2's val loss
    # lands BELOW epoch 1's, so a discarded ranking is observable as
    # best_loss != min (a hot lr overshoots and would hide it).
    encoder = _StopDuringValEncoder(stop_at_call=8)
    trainer = Trainer(model, _CharTokenizer(64), _config(tmp_path / "stopdone", epochs=2))
    encoder.trainer = trainer
    state = trainer.train_vision(encoder, train, val_data=val)

    assert encoder.calls == 8
    assert len(state.validation_losses) == 2
    # Guards the setup itself: if epoch 2 ever stops improving here, the
    # min() assertion below would go vacuous -- fail loudly instead.
    assert state.validation_losses[1] < state.validation_losses[0]
    assert state.best_loss == pytest.approx(min(state.validation_losses))


def test_lr_schedule_sized_in_optimizer_steps(tmp_path):
    """With gradient accumulation, the warmup/cosine schedule is sized in
    OPTIMIZER steps (where scheduler.step() fires), so the decay completes
    by end of run instead of stretching by the accumulation factor."""
    from PIL import Image

    data = [
        {"image": Image.new("RGB", (8, 8), color=(50 * i % 255, 60, 90)), "text": f"caption number {i}"}
        for i in range(4)
    ]
    cfg = _config(tmp_path / "accum", batch_size=1, max_grad_accumulation=2, warmup_steps=1)
    model = _tiny_model()
    trainer = Trainer(model, _CharTokenizer(64), cfg)
    state = trainer.train_vision(_TinyVisionEncoder(), data)
    assert not state.abort_reason
    assert state.step == 2  # 4 batches / accum 2 -> 2 optimizer steps
    best = list((tmp_path / "accum").glob("*_vision_best.pt"))[0]
    saved_opt = torch.load(best, map_location="cpu", weights_only=True)["optimizer_state_dict"]
    final_lr = saved_opt["param_groups"][0]["lr"]
    # Schedule sized to 2 optimizer steps: after both, cosine has decayed
    # to the floor. Micro-batch sizing (4) would leave it far above.
    assert final_lr == pytest.approx(cfg.learning_rate * cfg.min_lr_ratio, rel=0.05)


def test_loss_log_fires_once_per_optimizer_step(tmp_path):
    """The log_every gate is boundary-gated: one on_loss call per logged
    optimizer step, not one per micro-batch of a matching window."""
    from PIL import Image

    data = [
        {"image": Image.new("RGB", (8, 8), color=(70 * i % 255, 40, 40)), "text": f"a caption {i}"}
        for i in range(4)
    ]
    model = _tiny_model()
    trainer = Trainer(
        model, _CharTokenizer(64), _config(tmp_path / "loggate", batch_size=1, max_grad_accumulation=2, log_every=1)
    )
    calls = []
    trainer.on_loss = calls.append
    state = trainer.train_vision(_TinyVisionEncoder(), data)
    assert not state.abort_reason
    # 2 optimizer steps logged (log_every=1) + 1 epoch-summary emit.
    assert len(calls) == 3


def test_empty_state_dict_fails_cleanly(tmp_path):
    """A checkpoint whose model_state_dict is present-but-empty fails with
    a missing-keys error, not by falling through to the wrapper dict and
    dying on unexpected optimizer/config keys."""
    import enigma_engine.core.safe_save as safe_save

    path = tmp_path / "empty.pt"
    safe_save.atomic_torch_save(
        {"model_state_dict": {}, "optimizer_state_dict": {"state": {}, "param_groups": []}}, path
    )
    trainer = Trainer(_tiny_model(), _CharTokenizer(64), _config(tmp_path / "emptyload"))
    with pytest.raises(RuntimeError) as exc_info:
        trainer.load_checkpoint(path)
    assert "optimizer_state_dict" not in str(exc_info.value)


def test_config_from_dict_drops_retired_forge_keys(tmp_path):
    """Old training_config blobs carry the slimmed-away Forge keys; from_dict
    filters them so reconstruction works, while a refused knob that survives
    the filter still fails validate() with its clear message."""
    old_blob = {
        "epochs": 2,
        "batch_size": 8,
        "learning_rate": 5e-4,
        # retired inert keys (deleted from the schema 2026-07-19)
        "val_split": 0.1,
        "curriculum": "easy_first",
        "z_loss_weight": 1e-4,
        "gradient_noise_gamma": 0.55,
        "ademamix_beta3": 0.9999,
        "training_memory_gb": 8.0,
        "eval_test_prompts": None,
        # refused knob with a live value
        "ema_decay": 0.999,
    }
    cfg = TrainingConfig.from_dict(old_blob)
    assert cfg.epochs == 2 and cfg.batch_size == 8
    assert not hasattr(cfg, "val_split")
    with pytest.raises(ValueError, match="ema_decay"):
        cfg.validate()
    # Direct construction with a retired key stays loud.
    with pytest.raises(TypeError):
        TrainingConfig(val_split=0.1)
    # to_dict is field-derived: nothing can be silently omitted.
    import dataclasses as _dc

    assert set(TrainingConfig().to_dict().keys()) == {f.name for f in _dc.fields(TrainingConfig)}


def test_batch_size_zero_is_refused(tmp_path):
    """The 0 = auto-estimate sentinel was retired with the estimator; the
    refusal points at the hardware_detection helper instead."""
    with pytest.raises(ValueError, match="recommend_training_batch_size"):
        Trainer(_tiny_model(), _CharTokenizer(64), _config(tmp_path / "bs0", batch_size=0))


def test_load_checkpoint_skips_dead_optimizer_state(tmp_path):
    """load_checkpoint must not materialize optimizer moments into the
    never-stepped fallback optimizer (~1.5 GB dead weight at full scale):
    the lazy fallback stays unbuilt through a load."""
    trainer = Trainer(_tiny_model(), _CharTokenizer(64), _config(tmp_path / "deadopt"))
    path = tmp_path / "text.pt"
    assert trainer._save_checkpoint(path)

    trainer2 = Trainer(_tiny_model(seed=1), _CharTokenizer(64), _config(tmp_path / "deadopt2"))
    trainer2.load_checkpoint(path)
    assert trainer2._fallback_optimizer is None


def test_abort_does_not_count_unconsumed_drops(tmp_path, caplog):
    """dropped_short_captions counts CONSUMED batches only: an abort must
    not report drops from batches the prefetcher submitted but training
    never reached."""
    import logging as _logging

    from PIL import Image

    # seed=7 keeps this 2-element order after the epoch shuffle: the good
    # sample trains first and aborts on max_loss; the short caption sits
    # prefetched in batch 1, never consumed.
    data = [
        {"image": Image.new("RGB", (8, 8), color=(200, 40, 40)), "text": "a red square"},
        {"image": Image.new("RGB", (8, 8), color=(40, 200, 40)), "text": "a"},
    ]
    model = _tiny_model()
    trainer = Trainer(model, _CharTokenizer(64), _config(tmp_path / "abortcount", max_loss=1e-9))
    with caplog.at_level(_logging.WARNING, logger="enigma_engine.training.encoder_align"):
        state = trainer.train_vision(_TinyVisionEncoder(), data)
    assert state.abort_reason  # the max_loss guard fired
    assert not any("caption(s) dropped" in r.message for r in caplog.records)


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


# ---------------------------------------------------------------------------
# Audio twin (rebuilt 2026-07-19 on the shared encoder-align core): the same
# persistence contract re-locked for train_audio, plus the audio-specific
# refusals (batch_size > 1, missing audio_projection).
# ---------------------------------------------------------------------------


class _TinyAudioEncoder(nn.Module):
    """Stand-in for AudioEncoder: carries a real (tiny) AudioEncoderConfig -
    train_audio's decode closure hands it to preprocess_audio for the
    waveform->mel step - and maps [1, n_mels, n_frames] mels to
    [1, n_frames, AUDIO_DIM] features."""

    def __init__(self):
        super().__init__()
        from enigma_engine.core.audio_encoder import AudioEncoderConfig

        self.config = AudioEncoderConfig(
            n_mels=8, dim=AUDIO_DIM, n_layers=1, n_heads=2,
            max_audio_len=32, sample_rate=1000, n_fft=64, hop_length=16,
        )
        self.proj = nn.Linear(8, AUDIO_DIM)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        # lengths accepted to honour the mask-aware encoder contract the
        # audio collate now exercises; the stub ignores it (uniform mels).
        return self.proj(x.transpose(1, 2))


def _audio_data() -> list[dict]:
    """Two deterministic waveform-text pairs (no RNG in construction).
    256 samples at hop 16 -> 17 mel frames; 17 + <=16 text tokens stays
    inside the tiny model's max_seq_len=64."""
    return [
        {"audio": torch.sin(torch.linspace(0, 60.0, 256)), "text": "a rising tone"},
        {"audio": torch.sin(torch.linspace(0, 20.0, 256)), "text": "a lower tone"},
    ]


@pytest.fixture(scope="module")
def audio_run(tmp_path_factory):
    """One tiny train_audio run (2 pairs, 1 epoch, batch_size=1) shared by
    the read-only audio tests."""
    ckpt_dir = tmp_path_factory.mktemp("audio_ckpt")
    model = _tiny_model()
    trainer = Trainer(model, _CharTokenizer(64), _config(ckpt_dir))
    encoder = _TinyAudioEncoder()
    initial_encoder = _snapshot(encoder)

    state = trainer.train_audio(encoder, _audio_data())
    assert not state.abort_reason

    candidates = list(ckpt_dir.glob("*_audio_best.pt"))
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


def test_audio_checkpoint_contains_trained_encoder(audio_run):
    """The *_audio_best.pt checkpoint holds the audio encoder weights as
    they were trained (same contract as the vision twin)."""
    saved = audio_run.checkpoint.get("audio_encoder_state_dict")
    assert saved is not None, "audio checkpoint is missing audio_encoder_state_dict"
    assert _same_state(saved, audio_run.encoder.state_dict())
    # Training must have actually moved the encoder, otherwise the equality
    # above would pass on an untouched module.
    assert not _same_state(saved, audio_run.initial_encoder)
    assert audio_run.ckpt_path.name.endswith("_audio_best.pt")


def test_audio_resume_restores_encoder_and_model(audio_run, tmp_path):
    """resume_from restores audio encoder + model weights and counters."""
    model2 = _tiny_model(seed=1)
    torch.manual_seed(1)
    encoder2 = _TinyAudioEncoder()
    saved_encoder = audio_run.checkpoint["audio_encoder_state_dict"]
    assert not _same_state(encoder2.state_dict(), saved_encoder)  # sanity

    trainer2 = Trainer(model2, _CharTokenizer(64), _config(tmp_path / "aresume"))
    # Saved epoch == config.epochs, so the loop body never runs: what comes
    # out is exactly what the load path restored.
    state = trainer2.train_audio(encoder2, _audio_data(), resume_from=audio_run.ckpt_path)

    assert _same_state(encoder2.state_dict(), saved_encoder)
    assert _same_state(model2.state_dict(), audio_run.checkpoint["model_state_dict"])
    saved_state = audio_run.checkpoint["training_state"]
    assert state.epoch == saved_state["epoch"]
    assert state.step == saved_state["step"]
    assert state.best_loss == saved_state["best_loss"]


def test_audio_resume_refuses_text_only_checkpoint(tmp_path):
    """A text-only checkpoint has no audio encoder weights; resuming from
    it must fail loudly instead of silently restarting the encoder."""
    model2 = _tiny_model(seed=2)
    trainer2 = Trainer(model2, _CharTokenizer(64), _config(tmp_path / "arefuse"))
    text_ckpt = tmp_path / "text_only.pt"
    trainer2._save_checkpoint(text_ckpt)
    assert text_ckpt.exists()

    with pytest.raises(ValueError, match="audio_encoder_state_dict"):
        trainer2.train_audio(_TinyAudioEncoder(), _audio_data(), resume_from=text_ckpt)


def test_audio_refuses_batched_conformer_only(tmp_path):
    """batch_size > 1 now batches through the padding-mask collate; the
    refusal survives ONLY for conformer encoders, whose BatchNorm has no
    masked statistics (refuse-don't-lie)."""
    enc = _TinyAudioEncoder()
    enc.config.use_conformer = True
    trainer = Trainer(_tiny_model(), _CharTokenizer(64), _config(tmp_path / "abatch", batch_size=2))
    with pytest.raises(ValueError, match="masked statistics"):
        trainer.train_audio(enc, _audio_data())


def test_audio_conformer_batch1_still_trains(tmp_path):
    """Audit 2026-07-20 finding 1: the collate used to hand `lengths` to the
    encoder unconditionally, so conformer encoders (which refuse lengths --
    no masked BN) crashed at batch_size=1: the exact fallback config the
    batched-conformer refusal message recommends. A single-sample batch
    pads nothing, so _forward_ce must drop the lengths and take the
    legacy path. Uses a REAL conformer AudioEncoder -- the stub ignores
    lengths, which is precisely why the suite missed the regression."""
    from enigma_engine.core.audio_encoder import AudioEncoder, AudioEncoderConfig

    cfg = AudioEncoderConfig(
        n_mels=8, dim=AUDIO_DIM, n_layers=1, n_heads=2,
        max_audio_len=32, sample_rate=1000, n_fft=64, hop_length=16,
        use_conformer=True,
    )
    trainer = Trainer(_tiny_model(), _CharTokenizer(64), _config(tmp_path / "aconf1", batch_size=1))
    state = trainer.train_audio(AudioEncoder(cfg), _audio_data())
    assert state.epoch >= 1 and not state.abort_reason


def test_align_checkpoints_persist_encoder_config(tmp_path):
    """Align checkpoints must carry the encoder's own config beside its
    weights (audio_encoder_config / vision_encoder_config), so loaders
    rebuild the exact architecture instead of trusting a preset name
    (2026-07-20 eyes polish; serve._load_eyes prefers the stored config)."""
    trainer = Trainer(_tiny_model(), _CharTokenizer(64), _config(tmp_path / "acfg", batch_size=1))
    enc = _TinyAudioEncoder()
    trainer.train_audio(enc, _audio_data())
    ck_files = sorted((tmp_path / "acfg").glob("*_audio*.pt"))
    assert ck_files, "no audio align checkpoint written"
    ck = torch.load(ck_files[0], map_location="cpu", weights_only=False)
    assert "audio_encoder_config" in ck, list(ck)
    assert ck["audio_encoder_config"]["dim"] == enc.config.dim


def test_audio_batched_training_runs(tmp_path):
    """batch_size=2 over ragged pairs must complete an epoch through the
    collate + lengths path (the old batch=1-only refusal is gone)."""
    trainer = Trainer(_tiny_model(), _CharTokenizer(64), _config(tmp_path / "abatch2", batch_size=2))
    state = trainer.train_audio(_TinyAudioEncoder(), _audio_data())
    assert state.epoch >= 1 and not state.abort_reason


def test_audio_rolling_step_checkpoint(tmp_path):
    """save_every_steps > 0 produces the *_audio_step.pt rolling checkpoint
    carrying the audio encoder weights."""
    trainer = Trainer(_tiny_model(), _CharTokenizer(64), _config(tmp_path / "astep", save_every_steps=1))
    state = trainer.train_audio(_TinyAudioEncoder(), _audio_data())
    assert not state.abort_reason
    step_ckpts = list((tmp_path / "astep").glob("*_audio_step.pt"))
    assert len(step_ckpts) == 1
    ckpt = torch.load(step_ckpts[0], map_location="cpu", weights_only=True)
    assert "audio_encoder_state_dict" in ckpt


def test_audio_missing_projection_refused(tmp_path):
    """A model built without audio_hidden_size has no audio_projection
    port; train_audio must refuse with the enable hint."""
    torch.manual_seed(0)
    model = Enigma(
        ForgeConfig(
            vocab_size=64, dim=32, n_layers=2, n_heads=2,
            max_seq_len=64, dropout=0.0, use_gradient_checkpointing=False,
            vision_hidden_size=VISION_DIM,
        )
    )
    trainer = Trainer(model, _CharTokenizer(64), _config(tmp_path / "aport"))
    with pytest.raises(ValueError, match="audio_hidden_size"):
        trainer.train_audio(_TinyAudioEncoder(), _audio_data())


def test_audio_checkpoint_saves_stepped_optimizer(audio_run):
    """The stepped-local-optimizer leg of the persistence contract,
    audio-pinned (the vision twin alone would miss an audio-only
    divergence in what optimizer gets saved)."""
    opt_state = audio_run.checkpoint["optimizer_state_dict"]
    assert len(opt_state["state"]) > 0, "saved optimizer never stepped"
    assert len(audio_run.trainer.optimizer.state_dict()["state"]) == 0


def test_nothing_trained_run_is_refused(tmp_path):
    """A run whose every sample fails to decode must NOT report success:
    no best checkpoint (a 0.0 average would beat inf and crown untrained
    weights as best), and abort_reason is set so the align entry points
    SystemExit."""
    data = [
        {"audio": torch.zeros(4, 4), "text": "not a waveform"},
        {"audio": torch.zeros(3, 5), "text": "also not one"},
    ]  # 2-D tensors: preprocess_audio accepts paths or 1-D waveforms only
    model = _tiny_model()
    trainer = Trainer(model, _CharTokenizer(64), _config(tmp_path / "nothing"))
    state = trainer.train_audio(_TinyAudioEncoder(), data)
    assert "no samples trained" in state.abort_reason
    assert state.best_loss == float("inf")
    assert not list((tmp_path / "nothing").glob("*_audio_best.pt"))
