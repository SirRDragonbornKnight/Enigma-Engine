"""
Shared encoder-align trainer core for Enigma AI Engine (vision + audio).

Carved from the retired Forge-era training.py (2026-07-18) as the
vision-align trainer: Trainer with train_vision (LLaVA stage-1 on her own
weights), its checkpoint helpers, and the TrainingConfig/TrainingState
schema those checkpoints carry. Checkpoints saved by the old
Trainer.train_vision resume here unchanged.
Checkpoint-safety hardening added 2026-07-19 (no format change): missing
resume_from refuses instead of restarting, failed best-saves retry and
flag the run via abort_reason, resumed checkpoints get .keep cleanup
protection, every exit restores eval + requires_grad, and
save_every_steps>0 writes a rolling mid-epoch {stem}_{modality}_step.pt.
Generalized to vision+audio 2026-07-19: the hardened loop moved into the
private _train_encoder, parameterized by _Modality; train_vision and
train_audio are thin wrappers that keep the exact per-modality checkpoint
format (keys, filenames) each side always had.

Usage:
    from enigma_engine.training.encoder_align import Trainer, TrainingConfig

    config = TrainingConfig(epochs=1, batch_size=64, learning_rate=1e-4)
    trainer = Trainer(model, tokenizer, config)
    trainer.train_vision(vision_encoder=encoder, data=pairs)
"""

from __future__ import annotations

import dataclasses
import logging
import math
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LambdaLR,
    SequentialLR,
)

logger = logging.getLogger(__name__)


def _resolve_amp_dtype(amp_dtype: str = "auto") -> torch.dtype:
    """Resolve AMP dtype string to torch.dtype.

    ``"auto"`` picks BF16 on Ampere+ / Blackwell GPUs, FP16 otherwise.
    (Inlined from the retired core/rl_training.py, 2026-07-18.)
    """
    if amp_dtype == "bfloat16":
        return torch.bfloat16
    if amp_dtype == "float16":
        return torch.float16
    # "auto"
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _effective_warmup(warmup_steps: int, total_steps: int) -> int:
    """Cap warmup at 20% of total steps to prevent short-run waste.

    Sched-2 fix: With the default ``warmup_steps=100`` a 100-step SFT/DPO
    run was 100% warmup (no decay phase at all); a 200-step run was 50%.
    This cap clamps warmup to ``total_steps // 5`` so the schedule always
    has at least 80% of steps in the decay phase. Long runs are
    unaffected (cap is inactive when ``warmup_steps <= total_steps // 5``).

    Args:
        warmup_steps: Configured warmup steps from ``TrainingConfig``.
        total_steps: Total training steps for this run.

    Returns:
        Effective warmup count, always >= 1.
    """
    if total_steps <= 0:
        return max(1, warmup_steps)
    cap = max(1, total_steps // 5)
    return max(1, min(warmup_steps, cap))


def set_training_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Set random seed for reproducible training.

    Seeds Python random, PyTorch CPU, and PyTorch CUDA (if available).
    Call before training for deterministic runs.

    When ``deterministic=True`` (DET-2 opt-in), additionally pins GPU kernel
    selection by setting ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` and calling
    ``torch.use_deterministic_algorithms(True, warn_only=True)``. This is
    required for bitwise reproducibility on CUDA - same seed alone leaves
    cuBLAS/cuDNN free to pick different kernels per launch. Costs ~5-15%
    throughput per the PyTorch reproducibility docs. ``warn_only=True``
    keeps non-deterministic ops from crashing the run; they emit a
    UserWarning instead.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # CUBLAS_WORKSPACE_CONFIG must be set before the first cuBLAS call;
        # setting here covers the common pattern of seed-then-train.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
        logger.info(
            "Random seed set to %d (deterministic=True; CUBLAS_WORKSPACE_CONFIG=:4096:8, ~5-15%% throughput cost)",
            seed,
        )
    else:
        logger.info("Random seed set to %d", seed)


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class TrainingConfig:
    """Configuration for model training.

    Attributes:
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Initial learning rate
        weight_decay: L2 regularization
        warmup_steps: Steps for learning rate warmup
        gradient_clip: Max gradient norm (0 to disable)
        save_every: Save checkpoint every N epochs (0 = disabled)
        save_every_steps: Also save a rolling mid-epoch checkpoint
            ({stem}_{modality}_step.pt) every N optimizer steps (0 = disabled)
        checkpoint_dir: Directory for saving checkpoints
        log_every: Log metrics every N optimizer steps
        use_amp: Use automatic mixed precision (fp16)
        max_grad_accumulation: Gradient accumulation steps
    """

    epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    gradient_clip: float = 1.0
    save_every: int = 0
    save_every_steps: int = 0  # Save checkpoint every N steps (0 = disabled)
    checkpoint_dir: str = "models/checkpoints"
    log_every: int = 10
    use_amp: bool = True
    amp_dtype: str = "auto"  # "auto", "float16", "bfloat16"
    max_grad_accumulation: int = 1
    use_gradient_checkpointing: bool = True

    # Optimizer betas (LM-friendly defaults)
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8

    # Cosine schedule floor (Pass 156z9au).  ``eta_min`` is computed
    # as ``learning_rate * min_lr_ratio`` so the LR never decays
    # below this fraction of peak.  0.1 = decay to 10% of peak (sane
    # LM default).  0.0 reproduces the textbook cosine schedule that
    # crashes the LR to zero, which is rarely what you want for LM
    # training because the very last steps then contribute nothing.
    min_lr_ratio: float = 0.1

    # Early-stopping / safety guardrails
    early_stopping_patience: int = 5  # Stop if no improvement for 5 epochs
    max_loss: float = 100.0  # abort if loss exceeds this
    max_training_seconds: float = 0  # 0 = unlimited

    # torch.compile (10-20% throughput gain on supported hardware)
    use_compile: bool = False

    # Reproducibility: set random seed for deterministic training.
    # None = don't set seed (non-deterministic). 42 = common default.
    seed: int | None = None

    # DET-2: full bitwise GPU reproducibility. When True (and seed is set),
    # set_training_seed() also flips on torch.use_deterministic_algorithms
    # and pins CUBLAS_WORKSPACE_CONFIG so cuBLAS/cuDNN kernel selection is
    # stable across runs. Costs ~5-15% throughput; off by default.
    deterministic: bool = False

    # ── Refused Forge-era knobs ─────────────────────────────────────────
    # These stay in the schema ONLY so a config rebuilt from an old
    # checkpoint's training_config blob fails at validate() with a CLEAR
    # message instead of silently pretending the feature ran (refuse-
    # don't-lie). The rest of the Forge schema — the ~25 knobs that were
    # accepted-and-ignored (val_split, curriculum, z_loss_weight,
    # r_drop_alpha, run_evaluation, ademamix_*, gradient_noise_gamma,
    # wsd_decay_fraction, general_mix_ratio, training_memory_gb, ...) —
    # was DELETED 2026-07-19; from_dict() drops their keys from old blobs.
    label_smoothing: float = 0.0
    ema_decay: float = 0.0
    gradient_noise_eta: float = 0.0
    swa_update_interval: int = 0
    schedule_type: str = "cosine"  # warmup+cosine is the only implemented schedule
    bpe_dropout: float = 0.0
    use_lisa: bool = False
    rolling_best_k: int = 0
    optimizer: str = "adamw"
    llrd_decay: float = 0.0

    def validate(self) -> None:
        """Raise *ValueError* if any field is nonsensical."""
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.batch_size < 1:
            raise ValueError(
                f"batch_size must be >= 1, got {self.batch_size}. The old 0 = auto-estimate "
                f"sentinel was retired 2026-07-19 (its memory model predated the frozen "
                f"multimodal workload); pick explicitly - "
                f"hardware_detection.recommend_training_batch_size gives a starting point"
            )
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.gradient_clip < 0:
            raise ValueError(f"gradient_clip must be >= 0, got {self.gradient_clip}")
        if self.save_every_steps < 0:
            raise ValueError(f"save_every_steps must be >= 0 (0 = disabled), got {self.save_every_steps}")
        if self.max_grad_accumulation < 1:
            raise ValueError(f"max_grad_accumulation must be >= 1, got {self.max_grad_accumulation}")
        # Optimizer betas must be in (0, 1)
        if not 0.0 < self.adam_beta1 < 1.0:
            raise ValueError(f"adam_beta1 must be in (0, 1), got {self.adam_beta1}")
        if not 0.0 < self.adam_beta2 < 1.0:
            raise ValueError(f"adam_beta2 must be in (0, 1), got {self.adam_beta2}")
        # min_lr_ratio in [0, 1] - Pass 156z9au.
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError(f"min_lr_ratio must be in [0.0, 1.0], got {self.min_lr_ratio}")
        # Refused Forge-era knobs (see the field-block comment). This
        # trainer never runs them, so accepting a live value silently
        # would stamp the checkpoint as if the feature had applied.
        # Refuse instead of lying. NOTE for configs rebuilt from old
        # blobs: before 2026-07-19 four of these shipped as INERT
        # defaults (label_smoothing 0.05, gradient_noise_eta 0.01,
        # bpe_dropout 0.1, schedule_type 'wsd') that never applied -
        # reset them to the inert values to proceed.
        if self.ema_decay > 0:
            raise ValueError("ema_decay is not supported by the encoder-align trainer (no averaging step)")
        if self.swa_update_interval > 0:
            raise ValueError("swa_update_interval is not supported by the encoder-align trainer (no averaging step)")
        if self.use_lisa:
            raise ValueError("use_lisa was a Forge-trainer feature and is not supported by the encoder-align trainer")
        if self.rolling_best_k > 0:
            raise ValueError("rolling_best_k is not supported by the encoder-align trainer (no rolling save path)")
        if self.llrd_decay > 0:
            raise ValueError(
                "llrd_decay was a Forge-trainer feature and is not supported by the encoder-align trainer"
            )
        if self.label_smoothing > 0:
            raise ValueError(
                "label_smoothing is not implemented by the encoder-align loss "
                "(pre-2026-07-19 checkpoints recorded an inert default 0.05; use 0.0)"
            )
        if self.gradient_noise_eta > 0:
            raise ValueError(
                "gradient_noise_eta is not implemented by this trainer's backward path "
                "(pre-2026-07-19 checkpoints recorded an inert default 0.01; use 0.0)"
            )
        if self.bpe_dropout > 0:
            raise ValueError(
                "bpe_dropout is not implemented - captions are tokenized once during prep "
                "(pre-2026-07-19 checkpoints recorded an inert default 0.1; use 0.0)"
            )
        if self.schedule_type != "cosine":
            raise ValueError(
                f"schedule_type '{self.schedule_type}' is not implemented - this trainer runs "
                f"only warmup+cosine (pre-2026-07-19 checkpoints recorded an inert 'wsd' "
                f"default that never applied; use 'cosine')"
            )
        # Optimizer: the Forge trainer offered ademamix/muon; this trainer
        # implements only AdamW. Reject the others HERE rather than deep in
        # __init__ so a config reconstructed from an old checkpoint's
        # training_config blob fails at validation with a clear message.
        if self.optimizer != "adamw":
            raise ValueError(
                f"optimizer '{self.optimizer}' was a Forge-trainer feature; the "
                f"vision-align trainer supports only 'adamw'"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary (checkpoint training_config blob).

        Derived from the dataclass so a new field can never be silently
        omitted from checkpoints by a stale hand-written listing."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainingConfig":
        """Rebuild a config from a checkpoint's training_config blob.

        Filters against the dataclass fields (the ForgeConfig.from_dict
        pattern) so blobs written before the 2026-07-19 schema slimming -
        which carry the retired inert Forge keys (val_split, curriculum,
        z_loss_weight, ademamix_*, ...) - still construct; validate()
        then rules on the values that survive, and the refused knobs
        (ema_decay, schedule_type, ...) keep their clear messages."""
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class TrainingState:
    """Tracks training state for checkpointing and resume."""

    epoch: int = 0
    step: int = 0
    # Optimizer-step count at the current epoch's start. Epoch-boundary
    # checkpoints carry epoch_start_step == step; the save_every_steps
    # rolling checkpoint carries epoch_start_step < step, which the resume
    # path detects to wind the LR schedule and step counter back to the
    # epoch boundary. -1 = loaded from a checkpoint predating the field
    # (always epoch-boundary, so no wind-back is needed).
    epoch_start_step: int = 0
    best_loss: float = float("inf")
    # (total_tokens was removed 2026-07-19: its only writer was the deleted
    # Forge text loop, so every vision checkpoint stamped a lying 0.)
    training_losses: list[float] = field(default_factory=list)
    validation_losses: list[float] = field(default_factory=list)
    abort_reason: str = ""


# =============================================================================
# MODALITY
# =============================================================================


@dataclass(frozen=True)
class _Modality:
    """Per-modality plumbing for the shared _train_encoder loop.

    train_vision / train_audio each build one of these; everything
    modality-specific the hardened loop touches (data keys, checkpoint
    keys, decode/augment/probe callables, naming) lives here so the loop
    itself stays modality-blind."""

    name: str            # "vision" / "audio" -> checkpoint stems + log prefixes
    media_key: str       # "image" / "audio" key in data dicts
    encoder_key: str     # "vision_encoder_state_dict" / "audio_encoder_state_dict"
    projection_attr: str # "vision_projection" / "audio_projection"
    features_kwarg: str  # "vision_features" / "audio_features"
    # Media ref -> [1, ...] CPU tensor. NO RNG inside; must be pool-safe
    # when handed a path ref (the decode pool calls it from workers).
    decode: Callable[[object], "torch.Tensor"]
    # Train-time augmentation only, called on the MAIN thread in batch
    # order (RNG order = batch order); None = no augmentation.
    augment: "Callable[[torch.Tensor], torch.Tensor] | None"
    # Cheap validity probe for path refs (raises on a bad file), or None
    # to skip the probe pre-pass for path refs (still lazily decoded).
    probe: "Callable[[object], None] | None"
    enable_hint: str     # e.g. "Set vision_hidden_size in ForgeConfig to enable vision."
    # Batch collate for RAGGED media: list of [1, ...] tensors ->
    # (batch tensor, lengths tensor passed to encoder(x, lengths=...)).
    # None = uniform media, plain torch.cat and no lengths (vision).
    collate: "Callable[[list[torch.Tensor]], tuple[torch.Tensor, torch.Tensor]] | None" = None


# =============================================================================
# TRAINER
# =============================================================================


class Trainer:
    """
    Encoder-align trainer for Enigma models.

    Supports:
    - train_vision: encoder + projection training on image-text pairs
      (LLaVA stage-1 recipe; align_vision.py is the entry point)
    - train_audio: encoder + projection training on audio-text pairs
      (same recipe over mel spectrograms; align_audio.py is the entry point)
    - Progress callbacks
    - Checkpoint saving/loading with exact resume

    Usage:
        trainer = Trainer(model, tokenizer, config)
        trainer.on_progress = lambda pct, msg: print(f"{pct}% - {msg}")
        trainer.train_vision(vision_encoder=encoder, data=pairs)
    """

    def __init__(self, model: nn.Module, tokenizer: Any, config: TrainingConfig | None = None):
        """
        Initialize trainer.

        Args:
            model: The Enigma model to train
            tokenizer: Tokenizer for encoding text
            config: Training configuration
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or TrainingConfig()
        self.state = TrainingState()

        # A tokenizer wider than the model's head emits ids the model cannot
        # score; a NARROWER one trains real ids that mean something else to
        # the model (v1 table against a v2 head). Neither crashes on its own:
        # the loss curve looks plausible and the captions decode to garbage.
        _tok_vocab = getattr(tokenizer, "vocab_size", None)
        _model_vocab = getattr(getattr(model, "config", None), "vocab_size", None)
        if _tok_vocab and _model_vocab and _tok_vocab > _model_vocab:
            raise ValueError(
                f"tokenizer vocab {_tok_vocab} exceeds model vocab {_model_vocab}; "
                "select the tokenizer from the checkpoint (vocab_file_for_size)"
            )
        if _tok_vocab and _model_vocab and _tok_vocab < _model_vocab - 128:
            raise ValueError(
                f"tokenizer vocab {_tok_vocab} is far below model vocab {_model_vocab}; "
                "this trains correct-looking ids that mean something else to the model -- "
                "select the tokenizer from the checkpoint (vocab_file_for_size)"
            )

        self.device = next(model.parameters()).device

        # Gradient checkpointing - trades compute for VRAM savings.
        if self.config.use_gradient_checkpointing:
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
            else:
                for layer in getattr(self.model, "layers", []):
                    if hasattr(layer, "use_checkpoint"):
                        layer.use_checkpoint = True

        self.config.validate()  # fail-fast on bad config

        # Callbacks for progress updates
        self.on_progress: Callable[[int, str], None] | None = None
        self.on_loss: Callable[[float], None] | None = None
        self.on_epoch_complete: Callable[[int, float], None] | None = None
        self.on_warning: Callable[[str], None] | None = None

        # Training control
        self._stop_requested = False
        self._lock = threading.Lock()

        # Fallback optimizer for text-only _save_checkpoint calls: built
        # lazily by the `optimizer` property, never stepped.
        self._fallback_optimizer: "torch.optim.Optimizer | None" = None
        self.scheduler = None

        # Resolve AMP dtype (BF16 on Blackwell / Ampere+, FP16 otherwise)
        self._amp_dtype = self._resolve_amp_dtype()

        # Mixed precision scaler - disabled for BF16 (no loss scaling needed)
        self.scaler = None
        if self.config.use_amp and torch.cuda.is_available():
            if self._amp_dtype != torch.bfloat16:
                self.scaler = torch.amp.GradScaler("cuda")

        # torch.compile for throughput gains (PyTorch 2.0+)
        # Requires Triton backend; without it Inductor falls back to C++
        # codegen which consumes tens of GB of system RAM and can OOM-kill
        # the entire OS.  Only enable when Triton is actually importable.
        self._compiled = False
        if self.config.use_compile:
            try:
                import triton  # noqa: F401
            except ImportError:
                logger.warning(
                    "torch.compile requested but Triton is not installed - "
                    "skipping (Inductor C++ fallback is too RAM-hungry). "
                    "Install with: pip install triton"
                )
            else:
                try:
                    self.model = torch.compile(self.model)
                    self._compiled = True
                    logger.info("torch.compile enabled (Triton backend)")
                except Exception as e:
                    logger.warning(f"torch.compile not available: {e}")

        logger.info(f"Trainer initialized: device={self.device}, config={self.config.to_dict()}")

    @property
    def optimizer(self) -> "torch.optim.Optimizer":
        """Fallback AdamW for text-only _save_checkpoint calls.

        _train_encoder builds and steps its own LOCAL optimizer over the
        encoder-trainable parameter set; nothing ever steps this one. It
        exists only so the text-only checkpoint format keeps its
        optimizer_state_dict key, so it is built lazily (no dead state on
        the device) with a single param group (grouping is irrelevant for
        an optimizer that never steps)."""
        if self._fallback_optimizer is None:
            self._fallback_optimizer = AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                eps=self.config.adam_eps,
            )
        return self._fallback_optimizer

    def _resolve_amp_dtype(self) -> torch.dtype:
        """Resolve the AMP autocast dtype from config.

        ``"auto"`` picks BF16 when the GPU supports it (Ampere+, Blackwell)
        and falls back to FP16 otherwise.  BF16 has better numeric range
        which avoids many loss-scaling headaches.
        """
        return _resolve_amp_dtype(self.config.amp_dtype)

    def _emit_progress(self, percent: int, message: str) -> None:
        """Emit progress update via callback."""
        if self.on_progress:
            try:
                self.on_progress(percent, message)
            except Exception as e:
                logger.debug(f"Progress callback error: {e}")

    def _emit_loss(self, loss: float) -> None:
        """Emit loss update via callback."""
        if self.on_loss:
            try:
                self.on_loss(loss)
            except Exception as e:
                logger.debug(f"Loss callback error: {e}")

    def _emit_warning(self, message: str) -> None:
        """Emit warning via callback (S717)."""
        if self.on_warning:
            try:
                self.on_warning(message)
            except Exception as e:
                logger.debug(f"Warning callback error: {e}")

    def request_stop(self) -> None:
        """Request graceful stop of training."""
        with self._lock:
            self._stop_requested = True
        logger.info("Training stop requested")

    def _should_stop(self) -> bool:
        """Check if stop was requested."""
        with self._lock:
            return self._stop_requested

    @property
    def _checkpoint_stem(self) -> str:
        """Derive model stem from checkpoint_dir for naming files.

        GUI sets checkpoint_dir to ``models/checkpoints/{model_stem}``,
        so ``Path(checkpoint_dir).name`` is the model stem.  CLI uses
        ``models/checkpoints`` (name="checkpoints") - fall back to
        "model".
        """
        name = Path(self.config.checkpoint_dir).name
        if name == "checkpoints":
            return "model"
        return name

    def _save_checkpoint(
        self,
        path: Path,
        *,
        encoder: "nn.Module | None" = None,
        encoder_key: str | None = None,
        optimizer: "torch.optim.Optimizer | None" = None,
        scheduler: Any = None,
        scaler: Any = None,
    ) -> bool:
        """Save model checkpoint with full training state.

        Saves model weights, optimizer, scheduler, scaler, and step
        counters so training can resume exactly where it left off.

        _train_encoder trains a separately-passed encoder with a LOCAL
        optimizer/scheduler/scaler (its trainable set differs from the
        text path's). It must pass those here via the keyword args so
        the checkpoint captures what actually trained; otherwise the
        encoder weights are never written anywhere and the saved
        optimizer state belongs to the untouched self.optimizer.

        Args:
            path: Destination .pt file.
            encoder: Encoder module being trained alongside the model
                (vision/audio runs only). Requires *encoder_key*.
            encoder_key: Checkpoint key for the encoder state dict,
                e.g. "vision_encoder_state_dict".
            optimizer: Optimizer to save instead of self.optimizer.
            scheduler: Scheduler to save instead of self.scheduler.
            scaler: AMP scaler to save instead of self.scaler.

        Returns:
            True if the checkpoint reached disk, False if the save
            failed (logged + warning callback; callers that gate state
            on a successful write must check this).

        Raises:
            ValueError: If *encoder* is given without *encoder_key* -- a
                call-site bug, raised before the swallow-and-return-False
                error handling so it cannot masquerade as a disk failure.
        """
        if encoder is not None and not encoder_key:
            raise ValueError("encoder_key is required when saving an encoder")
        try:
            _opt = optimizer if optimizer is not None else self.optimizer
            _sched = scheduler if scheduler is not None else self.scheduler
            _scaler = scaler if scaler is not None else self.scaler
            # torch.compile wraps the model in OptimizedModule, which prefixes
            # every state-dict key with '_orig_mod.'. Save the RAW module so
            # checkpoints stay loadable by serve/gui/uncompiled trainers.
            _raw_model = getattr(self.model, "_orig_mod", self.model)
            checkpoint = {
                "model_state_dict": _raw_model.state_dict(),
                "optimizer_state_dict": _opt.state_dict(),
                "training_state": {
                    "epoch": self.state.epoch,
                    "step": self.state.step,
                    "epoch_start_step": self.state.epoch_start_step,
                    "best_loss": self.state.best_loss,
                    "training_losses": self.state.training_losses,
                    "validation_losses": self.state.validation_losses,
                },
                "training_config": self.config.to_dict(),
            }
            if encoder is not None:
                _raw_enc = getattr(encoder, "_orig_mod", encoder)
                checkpoint[encoder_key] = _raw_enc.state_dict()
                # Persist the encoder's own config beside its weights
                # ("vision_encoder_config"/"audio_encoder_config"), so
                # loaders rebuild the exact architecture instead of
                # trusting a hand-passed preset name to match -- the
                # silent-degrade risk the 2026-07-20 serve scout flagged.
                _enc_cfg = getattr(_raw_enc, "config", None)
                if _enc_cfg is not None:
                    cfg_key = encoder_key.replace("_state_dict", "_config")
                    if hasattr(_enc_cfg, "to_dict"):
                        checkpoint[cfg_key] = _enc_cfg.to_dict()
                    else:
                        checkpoint[cfg_key] = dict(_enc_cfg.__dict__)
            # Save scheduler state for exact resume
            if _sched is not None:
                checkpoint["scheduler_state_dict"] = _sched.state_dict()
            # Save AMP scaler state. A disabled GradScaler (vision runs
            # pass their local one) serializes to {} which
            # load_state_dict refuses - skip it, matching the text path
            # where self.scaler is None when scaling is off.
            if _scaler is not None and _scaler.state_dict():
                checkpoint["scaler_state_dict"] = _scaler.state_dict()

            if hasattr(self.model, "config"):
                checkpoint["model_config"] = self.model.config.__dict__
                # Also save as 'config' for compatibility with gui_forge loader
                checkpoint["config"] = self.model.config.__dict__

            from enigma_engine.core.safe_save import atomic_torch_save

            atomic_torch_save(checkpoint, path)
            # A fresh write supersedes any .keep protection left by an
            # earlier run that resumed from this same filename -- the bytes
            # it protected no longer exist, and a stale marker would make
            # periodic cleanup a dead letter for this slot forever.
            stale_marker = Path(path).parent / (Path(path).name + ".keep")
            if stale_marker.exists():
                try:
                    stale_marker.unlink()
                except OSError:
                    pass
            logger.info(f"Saved checkpoint: {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            self._emit_warning(f"Checkpoint save failed: {e}")
            return False

    def load_checkpoint(self, path: Path) -> None:
        """Load a TEXT-ONLY checkpoint (model weights + counters).

        Encoder runs must use :meth:`_load_encoder_checkpoint` instead — it is
        the only path that restores the separately-trained encoder and the
        LOCAL optimizer/scheduler/scaler that _train_encoder actually steps.
        """
        try:
            from enigma_engine.core.model_registry import safe_load_weights

            checkpoint = safe_load_weights(path, map_location=self.device)

            # Unwrap via the shared extractor: membership tests keep an
            # empty-but-present model_state_dict selected (clean missing-
            # keys error + empty warning) instead of falling through to
            # the whole wrapper dict and dying on unexpected optimizer/
            # config keys. Also strips torch.compile's '_orig_mod.'
            # prefix (legacy checkpoints saved from a compiled Trainer).
            from enigma_engine.core.model_registry import get_state_dict

            state_dict = get_state_dict(checkpoint, prefix="_orig_mod.")
            _raw_model = getattr(self.model, "_orig_mod", self.model)
            _raw_model.load_state_dict(state_dict)

            # The checkpoint's optimizer_state_dict is deliberately NOT
            # restored: nothing in this trainer ever steps self.optimizer
            # (_train_encoder builds its own local one), so materializing
            # full-model AdamW moments here (~1.5 GB at 182M fp32) would
            # be dead weight on the device.

            state = checkpoint.get("training_state", {})
            self.state.epoch = state.get("epoch", 0)
            self.state.step = state.get("step", 0)
            self.state.epoch_start_step = state.get("epoch_start_step", -1)
            self.state.best_loss = state.get("best_loss", float("inf"))
            self.state.training_losses = state.get("training_losses", [])
            self.state.validation_losses = state.get("validation_losses", [])

            # NOTE: scheduler/scaler/EMA/SWA state in the checkpoint is NOT
            # restored here. This trainer builds its scheduler and scaler
            # LOCALLY inside _train_encoder (they bind to the encoder+projection
            # parameter set, not self.optimizer), so _load_encoder_checkpoint
            # is the path that restores them. EMA/SWA are not supported at all
            # (see TrainingConfig.validate) — an old Forge checkpoint carrying
            # those keys loads fine, they are simply ignored.

            logger.info(f"Loaded checkpoint: {path}")

            # Mark this checkpoint as protected so auto-cleanup
            # won't delete it
            self._protect_checkpoint(path)
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

    @staticmethod
    def _protect_checkpoint(path: "Path | str") -> None:
        """Write a ``.keep`` sidecar so _cleanup_periodic_checkpoints
        never deletes a checkpoint a run resumed from."""
        path = Path(path)
        keep_marker = path.parent / (path.name + ".keep")
        if not keep_marker.exists():
            try:
                keep_marker.write_text("protected - resumed for training", encoding="utf-8")
            except OSError:
                pass

    @staticmethod
    def _cleanup_periodic_checkpoints(checkpoint_dir: Path, prefix: str, keep: int = 3) -> None:
        """Delete old periodic checkpoints, keeping only the most recent *keep*.

        Matches files like ``mymodel_checkpoint3.pt`` inside *checkpoint_dir*.
        Files are sorted by the number embedded in the filename; only the
        highest *keep* are retained.  Checkpoints with a ``.keep`` sidecar
        file are never deleted (protected).
        """
        import re

        pattern = re.compile(rf"^{re.escape(prefix)}(\d+)\.pt$")
        found: list[tuple[int, Path]] = []
        try:
            for f in checkpoint_dir.iterdir():
                m = pattern.match(f.name)
                if m:
                    found.append((int(m.group(1)), f))
        except OSError:
            return

        if len(found) <= keep:
            return

        # Sort by number descending - keep the newest
        found.sort(key=lambda x: x[0], reverse=True)
        for _num, old_path in found[keep:]:
            # Skip protected checkpoints (.keep sidecar)
            keep_marker = old_path.parent / (old_path.name + ".keep")
            if keep_marker.exists():
                logger.debug("Skipping protected checkpoint: %s", old_path.name)
                continue
            try:
                old_path.unlink()
                logger.info("Deleted old periodic checkpoint: %s", old_path.name)
            except OSError as exc:
                logger.debug("Could not delete old checkpoint %s: %s", old_path.name, exc)

    def _load_encoder_checkpoint(
        self,
        path: Path,
        *,
        encoder: nn.Module,
        encoder_key: str,
        optimizer: "torch.optim.Optimizer",
        scheduler: Any,
        scaler: Any,
    ) -> None:
        """Restore an encoder training checkpoint (encoder + local optimizer).

        _train_encoder builds its optimizer/scheduler/scaler locally
        because it trains a different parameter set than the text
        path, so a plain model-weights load cannot resume it. This
        loads the model weights, the encoder weights saved under
        *encoder_key*, the local training objects' state, and the
        step counters.

        Raises:
            ValueError: If the checkpoint lacks *encoder_key* - resuming
                an encoder run from a text-only checkpoint would silently
                restart the encoder from scratch.
        """
        from enigma_engine.core.model_registry import safe_load_weights

        checkpoint = safe_load_weights(path, map_location=self.device)

        enc_state = checkpoint.get(encoder_key)
        if enc_state is None:
            raise ValueError(
                f"Checkpoint {path} has no '{encoder_key}' entry - it was not "
                f"saved by an encoder training run, so the encoder cannot be resumed from it"
            )

        model_state = checkpoint.get("model_state_dict")
        if model_state is None:
            raise ValueError(f"Checkpoint {path} has no 'model_state_dict' entry")
        model_state = {k.removeprefix("_orig_mod."): v for k, v in model_state.items()}
        _raw_model = getattr(self.model, "_orig_mod", self.model)
        _raw_model.load_state_dict(model_state)

        getattr(encoder, "_orig_mod", encoder).load_state_dict(enc_state)

        opt_state = checkpoint.get("optimizer_state_dict")
        if opt_state:
            optimizer.load_state_dict(opt_state)

        sched_state = checkpoint.get("scheduler_state_dict")
        if sched_state is not None:
            scheduler.load_state_dict(sched_state)

        # Absent when the scaler was disabled at save time; a disabled
        # scaler's load_state_dict is a no-op, so this stays symmetric.
        scaler_state = checkpoint.get("scaler_state_dict")
        if scaler_state:
            scaler.load_state_dict(scaler_state)

        state = checkpoint.get("training_state", {})
        self.state.epoch = state.get("epoch", 0)
        self.state.step = state.get("step", 0)
        self.state.epoch_start_step = state.get("epoch_start_step", -1)
        self.state.best_loss = state.get("best_loss", float("inf"))
        self.state.training_losses = state.get("training_losses", [])
        self.state.validation_losses = state.get("validation_losses", [])

        # Same protection as load_checkpoint: a periodic checkpoint we
        # are resuming from must survive this run's own cleanup pass.
        self._protect_checkpoint(path)

        logger.info(f"Loaded encoder checkpoint: {path}")

    # -----------------------------------------------------------------
    # ENCODER ALIGN - encoder + projection alignment (LLaVA stage-1)
    # -----------------------------------------------------------------

    def train_vision(
        self,
        vision_encoder: nn.Module,
        data: list[dict[str, Any]],
        unfreeze_text_layers: int = 0,
        val_data: list[dict[str, Any]] | None = None,
        resume_from: str | Path | None = None,
        freeze_text_io: bool = False,
    ) -> "TrainingState":
        """
        Train the vision encoder and projection layer on image-text pairs.

        Both the vision encoder and the model's vision_projection layer are
        trained together. The text transformer is frozen by default (set
        unfreeze_text_layers > 0 to fine-tune the last N text layers too).

        Data format: list of dicts with:
            - "image": PIL Image object, file path string, or Path
            - "text": caption/description string

        Args:
            vision_encoder: VisionEncoder instance to train.
            data: List of image-text pair dicts.
            unfreeze_text_layers: Number of last text transformer layers to
                unfreeze (0 = freeze all text layers, only train encoder +
                projection).
            val_data: Optional held-out image-text pairs. When provided
                (and non-empty after validation), a no-grad evaluation
                pass runs after each epoch; results land in
                ``state.validation_losses`` and drive best-checkpoint /
                early-stopping decisions instead of the training avg.
                V-6: closes the "overfitting on small datasets is
                invisible" gap noted in the F/Code-6 audit.
            resume_from: Path to a checkpoint saved by a previous
                train_vision run. Restores model weights, the trained
                vision encoder weights, the local optimizer/scheduler/
                scaler state, and epoch/step counters. The checkpoint
                must contain "vision_encoder_state_dict" (text-only
                checkpoints are refused - the encoder would silently
                restart from scratch). Accepts best, periodic, and the
                save_every_steps rolling {stem}_vision_step.pt; resuming
                a mid-epoch rolling checkpoint replays the interrupted
                epoch from its start with the LR schedule wound back to
                the epoch boundary.

        Returns:
            TrainingState with loss history.

        Raises:
            ValueError: If model lacks vision_projection or data is empty.
            FileNotFoundError: If resume_from is set but the path does
                not exist (a silent fresh start would overwrite the
                checkpoints the caller meant to continue).
        """
        from enigma_engine.core.vision_encoder import augment_vision_tensor, preprocess_image

        # Preprocess kwargs come off the encoder config: the [-1,1]-vs-
        # ImageNet normalization contract ties the flag to use_pretrained
        # (False for her from-scratch encoders).
        image_size = getattr(getattr(vision_encoder, "config", None), "image_size", 224)
        use_imagenet = getattr(getattr(vision_encoder, "config", None), "use_pretrained", False)

        def _decode_image(ref: object) -> "torch.Tensor":
            """Image ref (path / PIL object) -> [1, 3, H, W] CPU tensor.
            No RNG; pool-safe for path refs."""
            return preprocess_image(ref, image_size=image_size, imagenet_normalize=use_imagenet)

        def _probe_image(p: object) -> None:
            """V-2 readability probe: PIL verify() parses the header
            without decoding pixels - cheap, and safe on the pool."""
            from PIL import Image as _PILImage

            with _PILImage.open(str(p)) as _probe:
                _probe.verify()

        modality = _Modality(
            name="vision",
            media_key="image",
            encoder_key="vision_encoder_state_dict",
            projection_attr="vision_projection",
            features_kwarg="vision_features",
            decode=_decode_image,
            augment=augment_vision_tensor,
            probe=_probe_image,
            enable_hint="Set vision_hidden_size in ForgeConfig to enable vision.",
        )
        return self._train_encoder(
            encoder=vision_encoder,
            data=data,
            modality=modality,
            unfreeze_text_layers=unfreeze_text_layers,
            val_data=val_data,
            resume_from=resume_from,
            freeze_text_io=freeze_text_io,
        )

    def train_audio(
        self,
        audio_encoder: nn.Module,
        data: list[dict[str, Any]],
        unfreeze_text_layers: int = 0,
        val_data: list[dict[str, Any]] | None = None,
        resume_from: str | Path | None = None,
        freeze_text_io: bool = False,
    ) -> "TrainingState":
        """
        Train the audio encoder and projection layer on audio-text pairs.

        Both the audio encoder and the model's audio_projection layer are
        trained together. The text transformer is frozen by default (set
        unfreeze_text_layers > 0 to fine-tune the last N text layers too).

        Batching: mel spectrograms are ragged along time; batches are
        right-padded by the audio collate and encoded with a padding
        mask (audio_encoder contract: pad rows excluded from attention,
        emitted as exact zeros -- pinned by test_audio_padding_mask).
        The zero rows remain in the media prefix as Whisper-style
        constant "silence" slots. Conformer encoders still refuse
        batch_size>1 up front: their BatchNorm has no masked statistics
        yet (refuse-don't-lie).

        Data format: list of dicts with:
            - "audio": file path string, Path, or 1-D waveform tensor
            - "text": transcript/description string

        Args:
            audio_encoder: AudioEncoder instance to train.
            data: List of audio-text pair dicts.
            unfreeze_text_layers: Number of last text transformer layers to
                unfreeze (0 = freeze all text layers, only train encoder +
                projection).
            val_data: Optional held-out audio-text pairs. When provided
                (and non-empty after validation), a no-grad evaluation
                pass runs after each epoch; results land in
                ``state.validation_losses`` and drive best-checkpoint /
                early-stopping decisions instead of the training avg.
                Val samples get a clean (un-augmented) mel; SpecAugment
                applies to training samples only.
            resume_from: Path to a checkpoint saved by a previous
                train_audio run. Restores model weights, the trained
                audio encoder weights, the local optimizer/scheduler/
                scaler state, and epoch/step counters. The checkpoint
                must contain "audio_encoder_state_dict" (text-only
                checkpoints are refused - the encoder would silently
                restart from scratch). Accepts best, periodic, and the
                save_every_steps rolling {stem}_audio_step.pt; resuming
                a mid-epoch rolling checkpoint replays the interrupted
                epoch from its start with the LR schedule wound back to
                the epoch boundary.

        Returns:
            TrainingState with loss history.

        Raises:
            ValueError: If batch_size > 1, model lacks audio_projection,
                or data is empty.
            FileNotFoundError: If resume_from is set but the path does
                not exist (a silent fresh start would overwrite the
                checkpoints the caller meant to continue).
        """
        acfg = getattr(audio_encoder, "config", None)
        if self.config.batch_size > 1 and getattr(acfg, "use_conformer", False):
            raise ValueError(
                "audio align with batch_size>1 requires a NON-conformer encoder: "
                "ConformerConv's BatchNorm has no masked statistics, so pad "
                "frames would poison every sample in the batch (refuse-don't-lie)"
            )

        from enigma_engine.core.audio_encoder import preprocess_audio, spec_augment

        def _collate_audio(mels: "list[torch.Tensor]") -> "tuple[torch.Tensor, torch.Tensor]":
            """Right-pad ragged [1, n_mels, T] mels with zeros + lengths.
            Zeros beyond a sample's length are inert: the mask-aware
            encoder excludes them from attention entirely."""
            lengths = torch.tensor([m.shape[-1] for m in mels], dtype=torch.long)
            t_max = int(lengths.max())
            batch = torch.zeros(len(mels), mels[0].shape[1], t_max, dtype=mels[0].dtype)
            for i, m in enumerate(mels):
                batch[i, :, : m.shape[-1]] = m[0]
            return batch, lengths

        audio_config = getattr(audio_encoder, "config", None)

        def _decode_audio(ref: object) -> "torch.Tensor":
            """Audio ref (path / 1-D waveform tensor) -> [1, n_mels,
            n_frames] CPU log-mel tensor. No RNG; pool-safe for path
            refs (file load + STFT release the GIL in C)."""
            return preprocess_audio(ref, config=audio_config)

        modality = _Modality(
            name="audio",
            media_key="audio",
            encoder_key="audio_encoder_state_dict",
            projection_attr="audio_projection",
            features_kwarg="audio_features",
            decode=_decode_audio,
            augment=spec_augment,
            probe=None,
            enable_hint="Set audio_hidden_size in ForgeConfig to enable audio.",
            collate=_collate_audio,
        )
        return self._train_encoder(
            encoder=audio_encoder,
            data=data,
            modality=modality,
            unfreeze_text_layers=unfreeze_text_layers,
            val_data=val_data,
            resume_from=resume_from,
            freeze_text_io=freeze_text_io,
        )

    def _train_encoder(
        self,
        encoder: nn.Module,
        data: list[dict[str, Any]],
        modality: _Modality,
        unfreeze_text_layers: int = 0,
        val_data: list[dict[str, Any]] | None = None,
        resume_from: str | Path | None = None,
        freeze_text_io: bool = False,
    ) -> "TrainingState":
        """Shared hardened align loop behind train_vision / train_audio.

        Trains *encoder* plus the model's modality projection layer on
        media-text pairs; everything modality-specific (data keys,
        checkpoint keys, decode/augment/probe, naming) comes in through
        *modality* (see _Modality). Contract, data format, and hardening
        guarantees are documented on the public wrappers."""
        # Capitalized log prefix for user-facing strings ("Vision training: ...").
        label = modality.name.capitalize()

        # ── Validation ──────────────────────────────────────────────────
        projection = getattr(self.model, modality.projection_attr, None)
        if projection is None:
            raise ValueError(
                f"Model does not have a {modality.name} projection layer. {modality.enable_hint}"
            )
        if not data:
            raise ValueError(f"No training data provided for {modality.name} training.")

        # Pass 156h: seed RNGs from config.seed when set, mirroring
        # train(). Without this the per-epoch random.shuffle(pairs)
        # below uses un-seeded global state, so two runs with the same
        # data + same seed process samples in different orders -
        # violates the AA "deterministic in infrastructure" rule.
        if self.config.seed is not None:
            set_training_seed(self.config.seed, deterministic=self.config.deterministic)

        self._stop_requested = False
        self.state = TrainingState()
        self._epochs_without_improvement = 0
        start_time = time.time()

        self._emit_progress(0, f"Preparing {modality.name} training data...")

        # Decode pool for the K7 overlap work (probe pre-pass + per-step
        # media decode). Declared before the try so the finally can shut it
        # down on every exit path.
        decode_pool: ThreadPoolExecutor | None = None

        try:
            # ── Freeze / unfreeze layers ────────────────────────────────────
            # Freeze all text model parameters
            for param in self.model.parameters():
                param.requires_grad = False

            # Unfreeze the modality projection (always trainable)
            for param in projection.parameters():
                param.requires_grad = True

            # Unfreeze output/embedding. NOT needed for gradient flow to the
            # projection (frozen modules still propagate grads) -- it lets the
            # text side co-adapt. freeze_text_io=True keeps them frozen so the
            # projection targets the base checkpoint's EXACT embedding space
            # (align recipe: serve later loads encoder+projection onto the
            # pristine served weights, so text drift here would be discarded
            # and the projection mismatched; Phase 4.5 step 4, 2026-07-17).
            if not freeze_text_io:
                for param in self.model.tok_embeddings.parameters():
                    param.requires_grad = True
                for param in self.model.output.parameters():
                    param.requires_grad = True
                for param in self.model.norm.parameters():
                    param.requires_grad = True

            # Optionally unfreeze last N text transformer layers
            if unfreeze_text_layers > 0:
                n_layers = len(self.model.layers)
                for layer in self.model.layers[max(0, n_layers - unfreeze_text_layers) :]:
                    for param in layer.parameters():
                        param.requires_grad = True

            # The encoder is fully trainable
            for param in encoder.parameters():
                param.requires_grad = True

            # Re-freeze pretrained backbone if configured (the blanket unfreeze
            # above would otherwise override freeze_backbone from __init__)
            if getattr(getattr(encoder, "config", None), "freeze_backbone", False):
                backbone = getattr(encoder, "backbone", None)
                if backbone is not None:
                    for param in backbone.parameters():
                        param.requires_grad = False

            trainable_params = list(filter(lambda p: p.requires_grad, self.model.parameters())) + list(
                filter(lambda p: p.requires_grad, encoder.parameters())
            )
            optimizer = AdamW(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                eps=self.config.adam_eps,
            )

            scaler = torch.amp.GradScaler(
                "cuda", enabled=(self.config.use_amp and self.device.type == "cuda" and self._amp_dtype != torch.bfloat16)
            )

            # Setup scheduler: SequentialLR(warmup + cosine decay)
            # One loop step per BATCH (real batching, Phase 4.5 step 4; the loop
            # was batch-1 until 2026-07-17 -- at LLaVA 558k scale that starved
            # the 5090 on Python overhead).
            train_batch = max(1, int(self.config.batch_size))
            # The schedule is sized in OPTIMIZER steps, matching where
            # scheduler.step() actually fires (accumulation boundaries +
            # the per-epoch remainder flush = ceil(batches/accum) per
            # epoch). Sizing it in micro-batches stretched warmup/decay by
            # the accumulation factor whenever max_grad_accumulation > 1.
            accum_steps = max(1, self.config.max_grad_accumulation)
            epoch_batches = (len(data) + train_batch - 1) // train_batch
            total_steps = ((epoch_batches + accum_steps - 1) // accum_steps) * self.config.epochs
            warmup = _effective_warmup(self.config.warmup_steps, total_steps)
            decay_steps = max(1, total_steps - warmup)

            def _build_scheduler() -> SequentialLR:
                warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / warmup))
                cosine_scheduler = CosineAnnealingLR(
                    optimizer, T_max=decay_steps, eta_min=(self.config.learning_rate * self.config.min_lr_ratio)
                )
                return SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup])

            scheduler = _build_scheduler()

            # Resume AFTER the local optimizer/scheduler exist - their saved
            # state loads into them here, not into self.optimizer (which
            # never steps during vision training).
            if resume_from is not None:
                ckpt_path = Path(resume_from)
                if ckpt_path.exists():
                    self._load_encoder_checkpoint(
                        ckpt_path,
                        encoder=encoder,
                        encoder_key=modality.encoder_key,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                    )
                    if 0 <= self.state.epoch_start_step < self.state.step:
                        # Mid-epoch rolling checkpoint (save_every_steps): the
                        # interrupted epoch replays from its start, so the
                        # schedule and step counter must wind back to the
                        # epoch boundary or the replayed boundaries would
                        # push the scheduler past total_steps (cosine turns
                        # back UP past T_max). The loaded weights/optimizer
                        # moments keep the mid-epoch progress.
                        scheduler = _build_scheduler()
                        for _ in range(self.state.epoch_start_step):
                            scheduler.step()
                        self.state.step = self.state.epoch_start_step
                    logger.info(
                        "Resuming %s training: epoch=%d, step=%d, best_loss=%.4f",
                        modality.name,
                        self.state.epoch,
                        self.state.step,
                        self.state.best_loss,
                    )
                else:
                    raise FileNotFoundError(
                        f"resume_from checkpoint not found: {ckpt_path}. Refusing to "
                        f"train from scratch - epoch 1 of a fresh run would overwrite "
                        f"existing checkpoints under {self.config.checkpoint_dir}. "
                        f"Drop resume_from to start a new run."
                    )

            # K7: shared thread pool for the probe pre-pass and the per-step
            # media decode. Decode is CPU/IO-bound and releases the GIL in
            # its C paths (PIL decoders, WAV/STFT), so threads (no Windows
            # spawn-pickling) give the overlap. Shut down in the finally.
            decode_pool = ThreadPoolExecutor(
                max_workers=min(8, os.cpu_count() or 4), thread_name_prefix=f"{modality.name}-decode"
            )

            def _probed_stream(items: list[dict[str, Any]], stream_label: str):
                """Yield (index, media, text, probe_future|None) in item
                order. Probes go to the pool in bounded chunks so a
                558k-item dataset overlaps its header reads without
                materializing 558k Future objects at once. When
                modality.probe is None, path refs skip the probe (still
                lazily decoded per-step)."""
                chunk = 512
                entries: list[tuple[int, object, str]] = []
                for i, item in enumerate(items):
                    media = item.get(modality.media_key)
                    text = item.get("text", "")
                    if media is None or not text:
                        logger.warning(
                            f"Skipping {modality.name} {stream_label} item {i}: "
                            f"missing {modality.media_key} or text"
                        )
                        continue
                    entries.append((i, media, text))
                for cs in range(0, len(entries), chunk):
                    window = entries[cs : cs + chunk]
                    futs = [
                        decode_pool.submit(modality.probe, m)
                        if (modality.probe is not None and isinstance(m, (str, Path)))
                        else None
                        for _, m, _ in window
                    ]
                    for (i, m, tx), fut in zip(window, futs):
                        yield i, m, tx, fut

            # V-2: lazy preprocess. Store media refs (paths or in-memory
            # objects) only - never preprocess/.to(device) eagerly. At
            # LLaVA-Pretrain scale (558K - 600 KB tensors) eager prep would
            # OOM the GPU immediately on a 16 GB VRAM budget. Tensors are
            # built per-step below and freed after backward.
            # K7: the probes run on the pool via _probed_stream; result
            # order follows item order, so skip/log semantics match the
            # old serial loop.
            pairs: list[tuple[object, list[int]]] = []
            for i, media, text, fut in _probed_stream(data, "data"):
                if fut is not None:
                    try:
                        fut.result()
                    except Exception as exc:
                        logger.warning(f"Skipping {modality.name} data item {i}: {exc}")
                        continue
                # Encode text to token IDs (tokenizer stays on the main thread)
                token_ids = self.tokenizer.encode(text)
                if len(token_ids) < 1:
                    continue
                pairs.append((media, token_ids))

            if not pairs:
                raise ValueError(f"No valid {modality.media_key}-text pairs found in training data.")

            # V-6: optional held-out validation pairs. Same lightweight
            # readability probe as train (pooled); same lazy-preprocess
            # discipline (no .to(device) until inside the eval pass). Pairs
            # whose caption is <2 tokens are skipped silently because the
            # next-token loss can't be computed on them - matches the
            # train-loop drop policy at min_len < 1.
            val_pairs: list[tuple[object, list[int]]] = []
            if val_data:
                for i, v_media, v_text, v_fut in _probed_stream(val_data, "val"):
                    if v_fut is not None:
                        try:
                            v_fut.result()
                        except Exception as exc:
                            logger.warning(f"Skipping {modality.name} val item {i}: {exc}")
                            continue
                    v_token_ids = self.tokenizer.encode(v_text)
                    if len(v_token_ids) < 2:
                        continue
                    val_pairs.append((v_media, v_token_ids))
                if val_pairs:
                    logger.info(f"{label} validation: {len(val_pairs)} held-out pairs")

            self._emit_progress(5, f"Prepared {len(pairs)} {modality.media_key}-text pairs")
            logger.info(f"{label} training: {len(pairs)} pairs, {self.config.epochs} epochs")

            # ── Training loop ───────────────────────────────────────────────
            encoder.train()
            self.model.train()

            checkpoint_dir = Path(self.config.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # V-7: count captions dropped because text length collapses to <1
            # after the next-token shift. First occurrence logs once; total
            # is summarized at the end. Avoids per-sample log spam while
            # keeping the pipeline observable.
            dropped_short_captions = 0

            def _emit_drop_summary():
                """V-7 audit fix: emit the dropped-captions summary on every
                exit path (success, NaN/Inf abort, max_loss abort, stop,
                early-stop). Otherwise the most diagnostic-critical paths
                silently lose the partial-data signal."""
                if dropped_short_captions > 0:
                    logger.warning(
                        "%s training: %d caption(s) dropped during "
                        "training for being too short after next-token "
                        "shift. Consider filtering single-token captions "
                        "during data prep.",
                        label,
                        dropped_short_captions,
                    )

            def _forward_ce(media: list[torch.Tensor], id_lists: list[list[int]], with_token_count: bool = False):
                """Shared batch-build + loss for the train loop and
                _run_validation (one implementation, so a masking/padding
                fix can never land on one side only): stack the media
                tensors, right-pad text/targets on CPU with one transfer
                each, run the multimodal forward, and return (mean CE over
                non-ignored targets, that target count or None). None when
                the batch degenerates (the media prefix swallows the text
                window, or nothing is left to predict). The token count is
                computed only on request: its .item() drains the forward
                stream -- fine after a no-grad val forward, but on a train
                batch it would wedge a pipeline bubble between forward and
                backward."""
                if modality.collate is not None:
                    media_batch, media_lengths = modality.collate(media)
                    # Nothing padded (batch of one, or equal lengths) ->
                    # drop the mask and take the legacy path. This is what
                    # keeps conformer encoders trainable at batch_size=1:
                    # they refuse `lengths` outright (no masked BN), and
                    # an all-False pad mask buys nothing (equivalence
                    # pinned by test_no_lengths_full_length_and_masked_agree).
                    # Audit 2026-07-20 finding 1: the unconditional
                    # lengths pass-through crashed exactly the fallback
                    # config the refusal message recommends.
                    if media_lengths is not None and int(media_lengths.min()) == media_batch.shape[-1]:
                        media_lengths = None
                else:
                    media_batch, media_lengths = torch.cat(media, dim=0), None
                media_batch = media_batch.to(self.device)
                max_len = max(len(t) for t in id_lists)
                text_tensor = torch.zeros(len(id_lists), max_len, dtype=torch.long)
                targets = torch.full((len(id_lists), max_len - 1), -100, dtype=torch.long)
                for bi, t in enumerate(id_lists):
                    row = torch.tensor(t, dtype=torch.long)
                    text_tensor[bi, : len(t)] = row
                    targets[bi, : len(t) - 1] = row[1:]
                text_tensor = text_tensor.to(self.device)
                targets = targets.to(self.device)
                with torch.amp.autocast(
                    "cuda",
                    dtype=self._amp_dtype,
                    enabled=self.config.use_amp and self.device.type == "cuda",
                ):
                    if media_lengths is None:
                        feats = encoder(media_batch)
                    else:
                        # Mask-aware encode: pad frames are excluded from
                        # attention and their feature rows are exact zeros
                        # (audio_encoder contract, test_audio_padding_mask).
                        # The zero rows DO remain in the prefix -- Whisper-
                        # style constant "silence" slots the text model
                        # learns to ignore; per-sample truncation cannot
                        # batch.
                        feats = encoder(media_batch, lengths=media_lengths.to(self.device))
                    logits = self.model.forward_multimodal(
                        input_ids=text_tensor,
                        **{modality.features_kwarg: feats},
                    )
                    # Loss only on the text portion: the model concatenates
                    # [media_prefix, text_tokens] internally, and causal
                    # attention means a real token never attends to the
                    # padding AFTER it, so correctness needs no attention
                    # mask -- only the ignore_index loss mask.
                    n_prefix = feats.shape[1]
                    if n_prefix >= logits.shape[1]:
                        logger.warning(
                            "%s prefix (%d) >= logits length (%d), skipping batch", label, n_prefix, logits.shape[1]
                        )
                        return None
                    text_logits = logits[:, n_prefix:-1, :]
                    min_len = min(text_logits.shape[1], targets.shape[1])
                    if min_len < 1:
                        return None
                    loss = nn.functional.cross_entropy(
                        text_logits[:, :min_len, :].reshape(-1, text_logits.size(-1)),
                        targets[:, :min_len].reshape(-1),
                        ignore_index=-100,
                    )
                if not with_token_count:
                    return loss, None
                n_tokens = int((targets[:, :min_len] != -100).sum().item())
                return loss, n_tokens

            def _run_validation() -> float | None:
                """V-6: no-grad evaluation over the held-out val_pairs.
                Returns the TOKEN-WEIGHTED mean cross-entropy across all
                valid val samples (each non-ignored target token counts
                once, so a ragged tail batch or drop-shrunken batch is not
                overweighted -- this number drives best-checkpoint
                selection and early stopping). None when val_pairs is
                empty, every sample was dropped, or the pass was
                interrupted. Caller restores train mode.

                Pass 156g audit (Bug B): polls ``self._should_stop()``
                between batches so the user's STOP press isn't ignored
                during a long val pass at LLaVA scale. An interrupted pass
                returns None -- a prefix mean is not comparable with the
                full-pass best_loss values it would be ranked against."""
                if not val_pairs:
                    return None
                encoder.eval()
                self.model.eval()
                total_loss = 0.0
                total_tokens = 0
                n_batches = 0
                stopped = False
                try:
                    with torch.no_grad():
                        for v_start in range(0, len(val_pairs), train_batch):
                            if self._should_stop():
                                stopped = True
                                logger.info(
                                    "%s validation stopped by user request after %d batch(es)", label, n_batches
                                )
                                break
                            v_media: list[torch.Tensor] = []
                            v_id_lists: list[list[int]] = []
                            for v_ref, v_ids in val_pairs[v_start : v_start + train_batch]:
                                if len(v_ids) < 2:
                                    continue
                                try:
                                    v_media.append(modality.decode(v_ref))
                                    v_id_lists.append(v_ids)
                                except Exception as exc:
                                    logger.debug("%s val: preprocess failed (%s); skipping", label, exc)
                                    continue
                            if not v_media:
                                continue
                            out = _forward_ce(v_media, v_id_lists, with_token_count=True)
                            if out is None:
                                continue
                            v_loss, v_n_tokens = out
                            if v_n_tokens == 0:
                                continue
                            total_loss += v_loss.item() * v_n_tokens
                            total_tokens += v_n_tokens
                            n_batches += 1
                finally:
                    encoder.train()
                    self.model.train()
                if stopped or total_tokens == 0:
                    return None
                return total_loss / total_tokens

            def _maybe_step_save() -> None:
                """Rolling mid-epoch checkpoint every save_every_steps
                optimizer steps (0 = disabled). One file, named outside the
                numeric periodic pattern so cleanup never deletes it; resume
                winds the schedule back to the epoch boundary (see the
                resume block above). Called from BOTH step-counter sites --
                the accumulation boundary and the remainder flush -- so
                accum-heavy configs still get their saves."""
                if self.config.save_every_steps > 0 and self.state.step % self.config.save_every_steps == 0:
                    self._save_checkpoint(
                        checkpoint_dir / f"{self._checkpoint_stem}_{modality.name}_step.pt",
                        encoder=encoder,
                        encoder_key=modality.encoder_key,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                    )

            # best_written = the best loss whose checkpoint actually reached
            # disk (seeded from the resumed lineage's best; inf on a fresh
            # run). state.best_loss stays the pure metric.
            best_written = self.state.best_loss
            failed_best_saves = 0
            # Optimizer steps taken by THIS run (state.step alone can't
            # tell: a resume loads a nonzero count) -- drives the
            # nothing-trained refusal at the end of the run.
            trained_steps_this_run = 0

            # When resuming, start from the saved epoch (mirrors train())
            start_epoch = self.state.epoch
            for epoch in range(start_epoch, self.config.epochs):
                if self._should_stop():
                    logger.info("%s training stopped by user request", label)
                    break

                # Time limit check
                if self.config.max_training_seconds > 0 and (time.time() - start_time) > self.config.max_training_seconds:
                    logger.info("%s training: time limit reached", label)
                    break

                epoch_loss = 0.0
                epoch_steps = 0
                train_interrupted = False

                # V-1: gradient accumulation (accum_steps hoisted to the
                # schedule setup). Only count successful backwards
                # (skipped/short-caption samples don't advance the
                # boundary counter).
                accum_count = 0

                # Shuffle training pairs each epoch
                random.shuffle(pairs)

                # Zero grads once at the start of the epoch; thereafter only
                # zero after a real optimizer step (boundary or remainder).
                optimizer.zero_grad()

                n_epoch_batches = (len(pairs) + train_batch - 1) // train_batch
                batch_starts = list(range(0, len(pairs), train_batch))

                # V-2: lazy preprocess, one BATCH at a time (real batching,
                # Phase 4.5 step 4). K7 overlap (2026-07-19): batch n+1's
                # media DECODE runs on the pool while batch n trains, so the
                # GPU no longer idles through serial decode work.
                # Augmentation (the only RNG consumer) stays on the MAIN
                # thread in batch order, so seeded runs draw the same
                # augment sequence as the old serial loop. Per-sample
                # failures still skip the sample, not the batch.
                def _submit_decode(bstart: int) -> list[tuple[str, Any, list[int]]]:
                    """Queue one batch's decodes. Only PATH refs go to the
                    pool (file decode is the I/O-bound win; an in-memory
                    object gains nothing from a thread and a LAZY
                    file-backed one must not be load()ed concurrently, so
                    'raw' refs decode on the main thread at consumption).
                    The <2-token filter runs here, but its counter and
                    warning fire at CONSUMPTION so an aborted run never
                    reports drops from batches that never trained."""
                    entries: list[tuple[str, Any, list[int]]] = []
                    for media_ref, token_ids in pairs[bstart : bstart + train_batch]:
                        if len(token_ids) < 2:
                            entries.append(("drop", None, token_ids))
                            continue
                        if isinstance(media_ref, (str, Path)):
                            entries.append(
                                ("fut", decode_pool.submit(modality.decode, media_ref), token_ids)
                            )
                        else:
                            entries.append(("raw", media_ref, token_ids))
                    return entries

                prefetch_depth = 2
                pending: dict[int, list[tuple[Any, list[int]]]] = {}
                for lead in range(min(prefetch_depth, len(batch_starts))):
                    pending[lead] = _submit_decode(batch_starts[lead])

                for step, bstart in enumerate(batch_starts):
                    if self._should_stop():
                        train_interrupted = True
                        break

                    entries = pending.pop(step)
                    nxt = step + prefetch_depth
                    if nxt < len(batch_starts):
                        pending[nxt] = _submit_decode(batch_starts[nxt])

                    media: list[torch.Tensor] = []
                    id_lists: list[list[int]] = []
                    for kind, payload, token_ids in entries:
                        if kind == "drop":
                            # V-7: too short for next-token loss after shift.
                            if dropped_short_captions == 0:
                                logger.warning(
                                    "%s caption too short for next-token "
                                    "loss after shift (need >=2 tokens) at "
                                    "epoch %d step %d. Further drops will be "
                                    "summed and reported at end of training.",
                                    label,
                                    epoch + 1,
                                    step,
                                )
                            dropped_short_captions += 1
                            continue
                        try:
                            if kind == "fut":
                                sample = payload.result()
                            else:  # "raw": in-memory ref, decoded here
                                sample = modality.decode(payload)
                        except Exception as exc:
                            logger.warning("%s step %d: preprocess failed (%s); skipping sample", label, step, exc)
                            continue
                        # Training-time augmentation, applied per sample on the
                        # main thread so each sample draws its own coin and
                        # seeded runs consume RNG in batch order.
                        media.append(modality.augment(sample) if modality.augment is not None else sample)
                        id_lists.append(token_ids)
                    if not media:
                        continue
                    out = _forward_ce(media, id_lists)
                    if out is None:
                        continue
                    loss, _ = out
                    # V-1: scale loss for accumulation. Recover unscaled
                    # value below for logging / NaN guards.
                    loss = loss / accum_steps

                    # Backward (always); step only at accum boundary.
                    if self.config.use_amp and self.device.type == "cuda":
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    accum_count += 1
                    is_boundary = accum_count % accum_steps == 0

                    if is_boundary:
                        if self.config.use_amp and self.device.type == "cuda":
                            if self.config.gradient_clip > 0:
                                scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(trainable_params, self.config.gradient_clip)
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            if self.config.gradient_clip > 0:
                                torch.nn.utils.clip_grad_norm_(trainable_params, self.config.gradient_clip)
                            optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                    # Recover unscaled loss for logging / guards.
                    loss_val = loss.item() * accum_steps

                    # NaN guard
                    if math.isnan(loss_val) or math.isinf(loss_val):
                        logger.error(f"{label} training: NaN/Inf loss at epoch {epoch + 1}, step {step}")
                        self._emit_progress(100, "Training aborted: NaN loss")
                        self.state.abort_reason = f"{label} NaN/Inf loss at epoch {epoch + 1}, step {step}"
                        _emit_drop_summary()  # V-7 audit fix
                        return self.state

                    # Safety guard
                    if loss_val > self.config.max_loss:
                        logger.error(f"{label} training: loss {loss_val:.2f} exceeds max {self.config.max_loss}")
                        self._emit_progress(100, "Training aborted: loss too high")
                        self.state.abort_reason = f"{label} loss {loss_val:.2f} exceeded max_loss {self.config.max_loss}"
                        _emit_drop_summary()  # V-7 audit fix
                        return self.state

                    epoch_loss += loss_val
                    epoch_steps += 1
                    if is_boundary:
                        self.state.step += 1
                        _maybe_step_save()

                    # Log every N optimizer steps. Boundary-gated: without
                    # it, every micro-batch of a matching accumulation
                    # window re-fired the callback (accum duplicate emits
                    # per logged step).
                    if self.config.log_every > 0 and is_boundary and self.state.step % self.config.log_every == 0:
                        self._emit_loss(loss_val)

                    # Progress within epoch (batch counts)
                    total_batches = n_epoch_batches * self.config.epochs
                    done_batches = epoch * n_epoch_batches + step + 1
                    pct = min(int(done_batches / max(total_batches, 1) * 95) + 5, 99)
                    self._emit_progress(pct, f"Epoch {epoch + 1}/{self.config.epochs} -- loss: {loss_val:.4f}")

                # V-1: end-of-epoch remainder flush. If samples don't divide
                # evenly into accum_steps, the trailing micro-batches have
                # accumulated grads that would be discarded by the next
                # epoch's zero_grad(). Flush them now.
                if accum_count % accum_steps != 0:
                    if self.config.use_amp and self.device.type == "cuda":
                        if self.config.gradient_clip > 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(trainable_params, self.config.gradient_clip)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        if self.config.gradient_clip > 0:
                            torch.nn.utils.clip_grad_norm_(trainable_params, self.config.gradient_clip)
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    self.state.step += 1
                    _maybe_step_save()

                # Epoch summary. A zero-step epoch (every sample dropped
                # or failed to decode) has NO train metric: a 0.0 average
                # would beat inf and crown untrained weights as best, so
                # avg_loss becomes None and only a real val_loss may rank
                # this epoch. The end-of-run check below flags a run that
                # never trained at all.
                self.state.epoch = epoch + 1
                self.state.epoch_start_step = self.state.step
                if epoch_steps == 0:
                    avg_loss = None
                    logger.warning(
                        "%s epoch %d: zero batches trained (all samples dropped or failed to decode)",
                        label,
                        epoch + 1,
                    )
                else:
                    trained_steps_this_run += epoch_steps
                    avg_loss = epoch_loss / epoch_steps
                    self.state.training_losses.append(avg_loss)
                    self._emit_loss(avg_loss)
                    logger.info(f"{label} Epoch {epoch + 1}: avg_loss={avg_loss:.4f}")

                # V-6: held-out validation pass after each epoch. Drives
                # best-checkpoint + early-stopping when available; falls
                # back to training avg_loss otherwise.
                val_loss = _run_validation()
                if val_loss is not None:
                    self.state.validation_losses.append(val_loss)
                    logger.info(f"{label} Epoch {epoch + 1}: val_loss={val_loss:.4f}")

                # Rank the epoch only when its metric is complete: a stop
                # that cut the TRAIN loop leaves a partial avg_loss, and a
                # stop that cut VALIDATION leaves val_loss None (never a
                # prefix mean) -- ranking either against full-pass
                # best_loss values could overwrite the real best
                # checkpoint with the stop-epoch weights. But a val pass
                # that COMPLETED before the stop landed is a full metric:
                # it ranks normally (discarding it would silently lose a
                # real best), and the epoch-loop top breaks right after.
                metric_complete = (
                    (val_loss is not None) if val_pairs else (avg_loss is not None and not train_interrupted)
                )
                if not metric_complete and self._should_stop():
                    logger.info("%s training stopped by user request", label)
                    break

                if self.on_epoch_complete and avg_loss is not None:
                    try:
                        self.on_epoch_complete(epoch + 1, avg_loss)
                    except Exception:
                        logger.debug("on_epoch_complete callback error", exc_info=True)

                # Best model tracking and early stopping. Prefer val_loss
                # when present (V-6); fall back to train avg_loss.
                # state.best_loss tracks the METRIC and drives early
                # stopping; best_written tracks what actually reached disk
                # and drives the save/retry. They diverge only while saves
                # fail, so a broken disk can neither mask a real plateau
                # nor stop the retries.
                tracked_loss = val_loss if val_loss is not None else avg_loss
                if tracked_loss is None:
                    # Zero-step epoch with no val signal: nothing to rank.
                    continue
                improved = tracked_loss < self.state.best_loss
                if improved:
                    self.state.best_loss = tracked_loss
                    self._epochs_without_improvement = 0
                if tracked_loss < best_written:
                    if self._save_checkpoint(
                        checkpoint_dir / f"{self._checkpoint_stem}_{modality.name}_best.pt",
                        encoder=encoder,
                        encoder_key=modality.encoder_key,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                    ):
                        best_written = tracked_loss
                    else:
                        failed_best_saves += 1
                if not improved:
                    self._epochs_without_improvement += 1
                    if (
                        self.config.early_stopping_patience > 0
                        and self._epochs_without_improvement >= self.config.early_stopping_patience
                    ):
                        logger.info("%s training: early stopping triggered", label)
                        break

                # Periodic checkpoint (0 = disabled)
                if self.config.save_every > 0 and (epoch + 1) % self.config.save_every == 0:
                    stem = self._checkpoint_stem
                    ck_num = (epoch + 1) // self.config.save_every
                    self._save_checkpoint(
                        checkpoint_dir / f"{stem}_{modality.name}{ck_num}.pt",
                        encoder=encoder,
                        encoder_key=modality.encoder_key,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                    )
                    self._cleanup_periodic_checkpoints(checkpoint_dir, f"{stem}_{modality.name}", keep=3)

            # ── Cleanup ─────────────────────────────────────────────────────
            # A run whose epoch loop RAN but never trained a single batch
            # produced nothing -- refuse to report success (a stale-path
            # dataset or a missing audio backend would otherwise 'succeed'
            # with untrained weights; repo rule: fail honestly).
            if start_epoch < self.config.epochs and trained_steps_this_run == 0:
                msg = "no samples trained: every batch was dropped or failed to decode"
                logger.error(f"{label} training: {msg}")
                self.state.abort_reason = msg
            # A best epoch whose checkpoint never reached disk makes the run
            # untrustworthy for automation: say so instead of reporting
            # success (repo rule: fail honestly).
            if best_written > self.state.best_loss:
                written = "none" if best_written == float("inf") else f"{best_written:.4f}"
                msg = (
                    f"best checkpoint never reached disk: best_loss "
                    f"{self.state.best_loss:.4f}, last written {written}, "
                    f"{failed_best_saves} failed save(s)"
                )
                logger.error(f"{label} training: {msg}")
                self.state.abort_reason = msg
                self._emit_progress(100, f"{label} training finished with errors")
            elif self.state.abort_reason:
                # e.g. the nothing-trained refusal above
                self._emit_progress(100, f"{label} training finished with errors")
            else:
                self._emit_progress(100, f"{label} training complete!")

            # V-7: end-of-run summary if any captions were dropped.
            _emit_drop_summary()

            return self.state
        finally:
            # Every exit path - normal completion, the NaN/max_loss
            # abort returns, and any raise from resume or data prep -
            # must leave the model inference-safe: eval mode, and every
            # text param trainable again for whatever runs next. Guarded
            # so a cleanup error cannot mask the in-flight exception
            # that got us here.
            try:
                if decode_pool is not None:
                    decode_pool.shutdown(wait=False, cancel_futures=True)
                self.model.eval()
                encoder.eval()
                for param in self.model.parameters():
                    param.requires_grad = True
            except Exception:
                logger.error("%s training: post-run state restore failed", label, exc_info=True)
