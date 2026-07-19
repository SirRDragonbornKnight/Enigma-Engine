"""
Vision-align trainer for Enigma AI Engine.

Carved from the retired Forge-era training.py (2026-07-18): Trainer with
train_vision (LLaVA stage-1 on her own weights), its checkpoint helpers,
and the TrainingConfig/TrainingState schema those checkpoints carry.
Checkpoints saved by the old Trainer.train_vision resume here unchanged.
Checkpoint-safety hardening added 2026-07-19 (no format change): missing
resume_from refuses instead of restarting, failed best-saves retry and
flag the run via abort_reason, resumed checkpoints get .keep cleanup
protection, every exit restores eval + requires_grad, and
save_every_steps>0 writes a rolling mid-epoch {stem}_vision_step.pt.

Usage:
    from enigma_engine.training.vision_align import Trainer, TrainingConfig

    config = TrainingConfig(epochs=1, batch_size=64, learning_rate=1e-4)
    trainer = Trainer(model, tokenizer, config)
    trainer.train_vision(vision_encoder=encoder, data=pairs)
"""

from __future__ import annotations

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
            ({stem}_vision_step.pt) every N optimizer steps (0 = disabled)
        checkpoint_dir: Directory for saving checkpoints
        eval_every: Evaluate every N steps (0 to disable)
        log_every: Log metrics every N steps
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
    eval_every: int = 0
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

    # Reasoning-weighted loss (CoT-E)
    # Weight multiplier for tokens inside <think>...</think> blocks.
    # 1.0 = normal (no extra weight), 2.0 = double weight on reasoning tokens.
    reasoning_loss_weight: float = 1.0

    # Rolling best checkpoints (CK-C)
    rolling_best_k: int = 0  # 0 = disabled, N = keep N best by loss

    # Early-stopping / safety guardrails
    early_stopping_patience: int = 5  # Stop if no improvement for 5 epochs
    max_loss: float = 100.0  # abort if loss exceeds this
    max_training_seconds: float = 0  # 0 = unlimited

    # Before/after evaluation (EV-C)
    run_evaluation: bool = False  # Evaluate before and after training
    eval_test_prompts: list[str] = None  # Custom test prompts (None = use defaults)

    # Label smoothing: a Forge-trainer feature the vision-align loss does
    # not implement. >0 is refused by validate() (refuse-don't-lie); the
    # field survives so old checkpoints' training_config dicts round-trip.
    label_smoothing: float = 0.0

    # Validation split
    val_split: float = 0.1  # 10% held out to detect overfitting early

    # EMA weight averaging (smooths training noise, use EMA for eval)
    ema_decay: float = 0.0  # 0.0 = disabled, 0.999 or 0.9999 typical

    # Gradient noise injection (Neelakantan et al.): a Forge-trainer
    # feature; this trainer's backward path injects nothing, so eta > 0
    # is refused by validate(). gamma stays as an inert companion so old
    # training_config dicts round-trip.
    gradient_noise_eta: float = 0.0
    gradient_noise_gamma: float = 0.55
    # T2-3: Noise lifecycle envelope. Ramp up during first 5% of steps,
    # anneal to zero during last 20%.
    noise_warmup_fraction: float = 0.05
    noise_decay_fraction: float = 0.2

    # SWA (Stochastic Weight Averaging, Izmailov et al.)
    # 0 = disabled.  N > 0 = collect a weight snapshot every N steps.
    swa_update_interval: int = 0

    # torch.compile (10-20% throughput gain on supported hardware)
    use_compile: bool = False

    # Cosine warm restarts (SGDR, Loshchilov & Hutter).
    # 0 = disabled (single cosine decay).  N > 0 = restart every N steps.
    cosine_restart_period: int = 0

    # LR schedule type. train_vision implements ONLY warmup + cosine, so
    # "cosine" is the sole accepted value; "wsd" was a Forge-trainer
    # feature (validate() refuses it rather than stamping a schedule into
    # the checkpoint that never ran).
    schedule_type: str = "cosine"
    # Fraction of total steps used for the decay phase of WSD.
    wsd_decay_fraction: float = 0.1

    # Sequence packing: pack short sequences into max_seq_len rows
    # separated by EOS tokens with block-diagonal attention masks.
    # 30-50% throughput gain by eliminating padding waste.
    use_sequence_packing: bool = False

    # BPE-Dropout (Provilkov et al.): a Forge-trainer feature. This
    # trainer tokenizes each caption ONCE during prep, so per-epoch
    # retokenization is structurally impossible; > 0 is refused by
    # validate().
    bpe_dropout: float = 0.0

    # T2-4: R-Drop regularization (Liang et al. 2021).
    # Two forward passes with different dropout masks; KL-divergence
    # penalty between the two output distributions.  Doubles forward
    # pass cost.  0.0 = disabled.  1.0 typical.
    r_drop_alpha: float = 0.0

    # General data mixing - prevents catastrophic forgetting.
    # When focused training data is provided alongside general data,
    # this ratio controls how much general data is mixed into each epoch.
    # 0.0 = no mixing (only focused), 1.0 = only general.
    # 0.2 = 20% general + 80% focused (good default).
    general_mix_ratio: float = 0.2
    general_data: str = ""  # Path or text of general knowledge data

    # Z-Loss (PaLM, Chowdhery et al. 2022)
    # Penalizes large logits to prevent training instability/NaN.
    # 0.0 = disabled. 1e-4 typical for large models.
    z_loss_weight: float = 0.0

    # T3-6: Cut cross-entropy - chunk size for vocab-chunked CE loss.
    # When > 0, avoids materializing the full [B*T, V] logit tensor
    # during training. Memory drops from O(B*T*V) to O(chunk*V).
    # Disabled when R-Drop/Z-Loss/reasoning weighting need full logits.
    # 0 = disabled (default). 4096 = good starting value.
    ce_chunk_size: int = 0

    # LISA (Pan et al. 2024) - retired with the Forge trainer; field kept
    # so old checkpoints' training_config dicts still round-trip.
    use_lisa: bool = False
    lisa_activated_layers: int = 2  # Number of middle layers to activate per step

    # Reproducibility: set random seed for deterministic training.
    # None = don't set seed (non-deterministic). 42 = common default.
    seed: int | None = None

    # DET-2: full bitwise GPU reproducibility. When True (and seed is set),
    # set_training_seed() also flips on torch.use_deterministic_algorithms
    # and pins CUBLAS_WORKSPACE_CONFIG so cuBLAS/cuDNN kernel selection is
    # stable across runs. Costs ~5-15% throughput; off by default.
    deterministic: bool = False

    # Golden prompt regression eval - JSON file with prompt+expected pairs.
    # Run before and after training to detect regressions.
    # Empty string = disabled.
    golden_eval_path: str = ""

    # Optimizer selection: only "adamw" is supported by this trainer
    # ("ademamix"/"muon" belonged to the retired Forge trainer; the
    # fields below survive for checkpoint schema compatibility).
    optimizer: str = "adamw"
    ademamix_beta3: float = 0.9999
    ademamix_alpha: float = 5.0
    ademamix_alpha_initial: float = 10.0
    ademamix_alpha_warmup: float = 0.1  # fraction of total steps

    # T5-7: Layer-wise Learning Rate Decay (LLRD)
    # Retired with the Forge trainer; must stay 0.0 here.
    llrd_decay: float = 0.0

    # T4-2: Curriculum learning - order training sequences by difficulty.
    # "none" = random shuffle (default).  "easy_first" = score each
    # sequence by loss on the current model, train easy half first.
    curriculum: str = "none"

    # I-3: LR range finder (Leslie Smith, 2015).
    # When True, runs a short sweep with exponentially increasing LR
    # before training to find the steepest-descent point automatically.
    auto_lr: bool = False

    # Training memory budget override (GB of system RAM to use).
    # 0 = auto-detect from hardware.  Set explicitly on constrained
    # systems (Raspberry Pi, shared servers) or to leave headroom
    # for other applications.
    training_memory_gb: float = 0.0

    def validate(self) -> None:
        """Raise *ValueError* if any field is nonsensical."""
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.batch_size < 0:
            raise ValueError(f"batch_size must be >= 0 (0 = auto), got {self.batch_size}")
        if self.training_memory_gb < 0:
            raise ValueError(f"training_memory_gb must be >= 0 (0 = auto), got {self.training_memory_gb}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.gradient_clip < 0:
            raise ValueError(f"gradient_clip must be >= 0, got {self.gradient_clip}")
        if self.save_every_steps < 0:
            raise ValueError(f"save_every_steps must be >= 0 (0 = disabled), got {self.save_every_steps}")
        if self.max_grad_accumulation < 1:
            raise ValueError(f"max_grad_accumulation must be >= 1, got {self.max_grad_accumulation}")
        if not 0.0 <= self.val_split < 1.0:
            raise ValueError(f"val_split must be in [0.0, 1.0), got {self.val_split}")
        # Optimizer betas must be in (0, 1)
        if not 0.0 < self.adam_beta1 < 1.0:
            raise ValueError(f"adam_beta1 must be in (0, 1), got {self.adam_beta1}")
        if not 0.0 < self.adam_beta2 < 1.0:
            raise ValueError(f"adam_beta2 must be in (0, 1), got {self.adam_beta2}")
        # min_lr_ratio in [0, 1] - Pass 156z9au.
        if not 0.0 <= self.min_lr_ratio <= 1.0:
            raise ValueError(f"min_lr_ratio must be in [0.0, 1.0], got {self.min_lr_ratio}")
        # EMA decay in [0, 1)
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError(f"ema_decay must be in [0.0, 1.0), got {self.ema_decay}")
        # Label smoothing in [0, 1)
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError(f"label_smoothing must be in [0.0, 1.0), got {self.label_smoothing}")
        # Z-loss weight must be non-negative
        if self.z_loss_weight < 0:
            raise ValueError(f"z_loss_weight must be >= 0, got {self.z_loss_weight}")
        # Reasoning loss weight must be positive
        if self.reasoning_loss_weight <= 0:
            raise ValueError(f"reasoning_loss_weight must be > 0, got {self.reasoning_loss_weight}")
        # General mix ratio in [0, 1]
        if not 0.0 <= self.general_mix_ratio <= 1.0:
            raise ValueError(f"general_mix_ratio must be in [0.0, 1.0], got {self.general_mix_ratio}")
        # Rolling best K must be non-negative
        if self.rolling_best_k < 0:
            raise ValueError(f"rolling_best_k must be >= 0, got {self.rolling_best_k}")
        # LLRD decay must be in [0, 1); >0 was a Forge-trainer feature and is
        # rejected here for the same reason as the optimizer above.
        if not 0.0 <= self.llrd_decay < 1.0:
            raise ValueError(f"llrd_decay must be in [0.0, 1.0), got {self.llrd_decay}")
        if self.llrd_decay > 0.0:
            raise ValueError(
                "llrd_decay was a Forge-trainer feature and is not supported by the vision-align trainer"
            )
        # Averaging/scheduling knobs the Forge trainer's own loop drove. This
        # trainer never steps them, so accepting the value silently would write
        # an INITIALIZATION snapshot into the checkpoint as if it were averaged
        # weights. Refuse instead of lying.
        if self.ema_decay > 0:
            raise ValueError("ema_decay is not supported by the vision-align trainer (no averaging step)")
        if self.swa_update_interval > 0:
            raise ValueError("swa_update_interval is not supported by the vision-align trainer (no averaging step)")
        if self.use_lisa:
            raise ValueError("use_lisa was a Forge-trainer feature and is not supported by the vision-align trainer")
        if self.rolling_best_k > 0:
            raise ValueError("rolling_best_k is not supported by the vision-align trainer (no rolling save path)")
        # NOTE for configs rebuilt from old checkpoints: before 2026-07-19
        # these four shipped as INERT defaults (label_smoothing 0.05,
        # gradient_noise_eta 0.01, bpe_dropout 0.1, schedule_type 'wsd')
        # and were recorded in training_config without ever applying, so a
        # pre-2026-07-19 blob carries them without the user having chosen
        # anything. Reset them to the inert values to proceed.
        if self.label_smoothing > 0:
            raise ValueError(
                "label_smoothing is not implemented by the vision-align loss "
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
                f"schedule_type '{self.schedule_type}' is not implemented - train_vision runs "
                f"only warmup+cosine (pre-2026-07-19 checkpoints recorded an inert 'wsd' "
                f"default that never applied; use 'cosine')"
            )
        # Gradient noise gamma must be non-negative (used as exponent)
        if self.gradient_noise_gamma < 0:
            raise ValueError(f"gradient_noise_gamma must be >= 0, got {self.gradient_noise_gamma}")
        # Optimizer: the Forge trainer offered ademamix/muon; this trainer
        # implements only AdamW. Reject the others HERE rather than deep in
        # __init__ so a config reconstructed from an old checkpoint's
        # training_config blob fails at validation with a clear message.
        if self.optimizer != "adamw":
            raise ValueError(
                f"optimizer '{self.optimizer}' was a Forge-trainer feature; the "
                f"vision-align trainer supports only 'adamw'"
            )
        # BPE-Dropout in [0, 1)
        if not 0.0 <= self.bpe_dropout < 1.0:
            raise ValueError(f"bpe_dropout must be in [0.0, 1.0), got {self.bpe_dropout}")
        # Curriculum must be a known value
        if self.curriculum not in ("none", "easy_first"):
            raise ValueError(f"curriculum must be 'none' or 'easy_first', got '{self.curriculum}'")

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_steps": self.warmup_steps,
            "gradient_clip": self.gradient_clip,
            "save_every": self.save_every,
            "save_every_steps": self.save_every_steps,
            "checkpoint_dir": self.checkpoint_dir,
            "eval_every": self.eval_every,
            "log_every": self.log_every,
            "use_amp": self.use_amp,
            "amp_dtype": self.amp_dtype,
            "max_grad_accumulation": self.max_grad_accumulation,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "adam_beta1": self.adam_beta1,
            "adam_beta2": self.adam_beta2,
            "adam_eps": self.adam_eps,
            "min_lr_ratio": self.min_lr_ratio,
            "rolling_best_k": self.rolling_best_k,
            "early_stopping_patience": self.early_stopping_patience,
            "max_loss": self.max_loss,
            "max_training_seconds": self.max_training_seconds,
            "run_evaluation": self.run_evaluation,
            "eval_test_prompts": self.eval_test_prompts,
            "label_smoothing": self.label_smoothing,
            "val_split": self.val_split,
            "ema_decay": self.ema_decay,
            "gradient_noise_eta": self.gradient_noise_eta,
            "gradient_noise_gamma": self.gradient_noise_gamma,
            "noise_warmup_fraction": self.noise_warmup_fraction,
            "noise_decay_fraction": self.noise_decay_fraction,
            "swa_update_interval": self.swa_update_interval,
            "use_compile": self.use_compile,
            "cosine_restart_period": self.cosine_restart_period,
            "schedule_type": self.schedule_type,
            "wsd_decay_fraction": self.wsd_decay_fraction,
            "use_sequence_packing": self.use_sequence_packing,
            "bpe_dropout": self.bpe_dropout,
            "r_drop_alpha": self.r_drop_alpha,
            "reasoning_loss_weight": self.reasoning_loss_weight,
            "general_mix_ratio": self.general_mix_ratio,
            "general_data": self.general_data,
            "z_loss_weight": self.z_loss_weight,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "golden_eval_path": self.golden_eval_path,
            "optimizer": self.optimizer,
            "ademamix_beta3": self.ademamix_beta3,
            "ademamix_alpha": self.ademamix_alpha,
            "ademamix_alpha_initial": self.ademamix_alpha_initial,
            "ademamix_alpha_warmup": self.ademamix_alpha_warmup,
            "ce_chunk_size": self.ce_chunk_size,
            "use_lisa": self.use_lisa,
            "lisa_activated_layers": self.lisa_activated_layers,
            "curriculum": self.curriculum,
            "llrd_decay": self.llrd_decay,
            "auto_lr": self.auto_lr,
            "training_memory_gb": self.training_memory_gb,
        }


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
    total_tokens: int = 0
    training_losses: list[float] = field(default_factory=list)
    validation_losses: list[float] = field(default_factory=list)
    abort_reason: str = ""


# =============================================================================
# TRAINER
# =============================================================================


class Trainer:
    """
    Vision-align trainer for Enigma models.

    Supports:
    - train_vision: encoder + projection training on image-text pairs
      (LLaVA stage-1 recipe; align_vision.py is the entry point)
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

        # Device - needed before validate for auto batch size
        self.device = next(model.parameters()).device

        # Gradient checkpointing - trades compute for VRAM savings.
        # Must be enabled BEFORE _estimate_batch_size() so the auto-batch
        # trial runs with the same memory profile as real training.
        if self.config.use_gradient_checkpointing:
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
            else:
                for layer in getattr(self.model, "layers", []):
                    if hasattr(layer, "use_checkpoint"):
                        layer.use_checkpoint = True

        # Auto batch size: when batch_size == 0, estimate optimal
        # based on GPU memory. Must run before validate().
        if self.config.batch_size == 0:
            self.config.batch_size = self._estimate_batch_size()

        self.config.validate()  # fail-fast on bad config

        # Callbacks for progress updates
        self.on_progress: Callable[[int, str], None] | None = None
        self.on_loss: Callable[[float], None] | None = None
        self.on_epoch_complete: Callable[[int, float], None] | None = None
        self.on_warning: Callable[[str], None] | None = None

        # Training control
        self._stop_requested = False
        self._lock = threading.Lock()

        # Setup optimizer and scheduler
        self._setup_optimizer()

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

    def _setup_optimizer(self) -> None:
        """Setup optimizer and learning rate scheduler.

        ``TrainingConfig.validate()`` (run in ``__init__`` before this) already
        rejected the Forge-only knobs — non-AdamW optimizers and llrd_decay > 0
        — so this builds the AdamW path unconditionally.
        """
        # Standard: separate weight decay for different parameter types
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Bias, norm, and embedding params should not get weight decay.
            # Embeddings are already at-scale tokens; decaying them shrinks
            # the representation magnitude and hurts convergence.
            # Note: output head is weight-tied to tok_embeddings (same tensor),
            # so it's only counted once by the optimizer.
            if "bias" in name or "norm" in name or "embed" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        # AdamW (the only supported optimizer; validate() gates the rest)
        adamw_kwargs: dict[str, Any] = {}
        if torch.cuda.is_available():
            import inspect

            if "fused" in inspect.signature(AdamW).parameters:
                adamw_kwargs["fused"] = True

        self.optimizer = AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_eps,
            **adamw_kwargs,
        )

        # train_vision builds its own local optimizer/scheduler; this one
        # exists so checkpoint helpers always have a fallback target.
        self.scheduler = None

    def _estimate_batch_size(self) -> int:
        """Estimate optimal batch size from available GPU memory.

        Runs a trial forward+backward pass with ``batch_size=2`` and
        measures peak VRAM consumption.  The per-sample activation cost
        is extrapolated to fill available VRAM minus model weights,
        optimizer states, and gradients.

        Falls back to 4 on CPU or if the trial fails.

        Returns:
            Power-of-2 batch size clamped to [1, batch_size_cap].
        """
        from enigma_engine.core.hardware_detection import TrainingMemoryBudget

        budget = TrainingMemoryBudget(ram_gb=self.config.training_memory_gb)

        if not torch.cuda.is_available() or self.device.type != "cuda":
            cpu_bs = budget.cpu_batch_size
            logger.info("Auto batch size: CPU detected, using batch_size=%d", cpu_bs)
            return cpu_bs

        cfg = getattr(self.model, "config", None)
        seq_len = getattr(cfg, "max_seq_len", 512)
        vocab_size = getattr(cfg, "vocab_size", 32000)

        total_vram = torch.cuda.get_device_properties(self.device).total_memory

        self.model.train()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        # Baseline memory: model weights already on GPU
        baseline_mem = torch.cuda.memory_allocated(self.device)

        trial_bs = 2
        dummy_ids = torch.randint(0, min(100, vocab_size), (trial_bs, seq_len), device=self.device)
        logits = None
        shift_logits = None
        shift_labels = None
        loss = None

        try:
            use_amp = self.config.use_amp and torch.cuda.is_available()
            # Resolve AMP dtype for the trial (same logic as training)
            amp_dtype = _resolve_amp_dtype(self.config.amp_dtype)

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                logits = self.model(dummy_ids)
                if isinstance(logits, tuple):
                    logits = logits[0]
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = dummy_ids[:, 1:].contiguous()
                loss = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )

            loss.backward()
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise  # Re-raise non-OOM errors (shape mismatch, etc.)
            # OOM with batch=2 - model barely fits; free everything
            dummy_ids = logits = shift_logits = shift_labels = loss = None
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad = None
            torch.cuda.empty_cache()
            logger.info("Auto batch size: trial OOM, using batch_size=1")
            return 1

        peak_mem = torch.cuda.max_memory_allocated(self.device)

        # Measure actual gradient memory (fixed per-param cost, not
        # per-sample).  Must measure BEFORE clearing gradients.
        grad_mem = sum(p.grad.numel() * p.grad.element_size() for p in self.model.parameters() if p.grad is not None)

        # Activation memory = peak minus weights minus gradients.
        # Gradients are fixed cost already tracked in the budget below,
        # so excluding them here prevents double-counting.
        activation_mem = max(1, peak_mem - baseline_mem - grad_mem)
        per_sample = max(1, activation_mem // trial_bs)

        # Cleanup - release tensors so VRAM is free for real training
        dummy_ids = logits = shift_logits = shift_labels = loss = None
        # Zero gradients so they don't persist into real training
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad = None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        # Budget calculation:
        # - Model weights: baseline_mem (already measured)
        # - Optimizer: AdamW momentum+variance in fp32 = 8 bytes/param
        # - Gradients: fp32 under AMP (same dtype as master weights),
        #   4 bytes/param
        # - Activations: per_sample * batch_size (measured above)
        # - Safety margin: 20% for CUDA fragmentation + temp allocations
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        optimizer_overhead = total_params * 8  # fp32 momentum + variance
        gradient_overhead = total_params * 4  # fp32 gradients
        fixed_cost = baseline_mem + optimizer_overhead + gradient_overhead

        # Usable = 80% of total VRAM minus all fixed costs
        usable_vram = int(total_vram * 0.80) - fixed_cost
        if usable_vram <= 0 or per_sample <= 0:
            logger.info(
                "Auto batch size: no headroom (fixed=%.1f GB, total=%.1f GB), using batch_size=1",
                fixed_cost / 1e9,
                total_vram / 1e9,
            )
            return 1

        max_batch = usable_vram // per_sample
        # Cap scales with VRAM - larger GPUs can use bigger batches
        max_batch = max(1, min(max_batch, budget.batch_size_cap))

        # Round down to nearest power of 2 for GPU efficiency
        batch = 1
        while batch * 2 <= max_batch:
            batch *= 2

        total_estimated = fixed_cost + batch * per_sample
        logger.info(
            "Auto batch size: %d  (model=%.1f GB + optimizer=%.1f GB "
            "+ grad=%.1f GB + activations=%.1f GB = %.1f/%.1f GB)",
            batch,
            baseline_mem / 1e9,
            optimizer_overhead / 1e9,
            gradient_overhead / 1e9,
            batch * per_sample / 1e9,
            total_estimated / 1e9,
            total_vram / 1e9,
        )
        return max(1, batch)

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

    def _emit_loss(self, loss: float, val_loss: float | None = None) -> None:
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

        train_vision trains a separately-passed encoder with a LOCAL
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
                    "total_tokens": self.state.total_tokens,
                    "training_losses": self.state.training_losses,
                    "validation_losses": self.state.validation_losses,
                    "dataset_fingerprint": getattr(self, "_dataset_fingerprint", None),
                },
                "training_config": self.config.to_dict(),
            }
            if encoder is not None:
                checkpoint[encoder_key] = getattr(encoder, "_orig_mod", encoder).state_dict()
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
        LOCAL optimizer/scheduler/scaler that train_vision actually steps.
        """
        try:
            from enigma_engine.core.model_registry import safe_load_weights

            checkpoint = safe_load_weights(path, map_location=self.device)

            # Unwrap state dict — handles both flat and wrapped formats
            state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint.get("model")
            if state_dict is None:
                # Assume the whole checkpoint is a bare state dict
                state_dict = checkpoint
            # Strip torch.compile's '_orig_mod.' prefix (legacy checkpoints
            # saved from a compiled Trainer) and load into the raw module.
            state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
            _raw_model = getattr(self.model, "_orig_mod", self.model)
            _raw_model.load_state_dict(state_dict)

            opt_state = checkpoint.get("optimizer_state_dict")
            if opt_state:
                self.optimizer.load_state_dict(opt_state)

            state = checkpoint.get("training_state", {})
            self.state.epoch = state.get("epoch", 0)
            self.state.step = state.get("step", 0)
            self.state.epoch_start_step = state.get("epoch_start_step", -1)
            self.state.best_loss = state.get("best_loss", float("inf"))
            self.state.total_tokens = state.get("total_tokens", 0)
            self.state.training_losses = state.get("training_losses", [])
            self.state.validation_losses = state.get("validation_losses", [])

            # NOTE: scheduler/scaler/EMA/SWA state in the checkpoint is NOT
            # restored here. This trainer builds its scheduler and scaler
            # LOCALLY inside train_vision (they bind to the encoder+projection
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
        """Restore a vision training checkpoint (encoder + local optimizer).

        train_vision builds its optimizer/scheduler/scaler locally
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
        self.state.total_tokens = state.get("total_tokens", 0)
        self.state.training_losses = state.get("training_losses", [])
        self.state.validation_losses = state.get("validation_losses", [])

        # Same protection as load_checkpoint: a periodic checkpoint we
        # are resuming from must survive this run's own cleanup pass.
        self._protect_checkpoint(path)

        logger.info(f"Loaded encoder checkpoint: {path}")

    # -----------------------------------------------------------------
    # VISION - encoder + projection alignment (LLaVA stage-1)
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

        # ── Validation ──────────────────────────────────────────────────
        if not hasattr(self.model, "vision_projection") or self.model.vision_projection is None:
            raise ValueError(
                "Model does not have a vision projection layer. Set vision_hidden_size in ForgeConfig to enable vision."
            )
        if not data:
            raise ValueError("No training data provided for vision training.")

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

        self._emit_progress(0, "Preparing vision training data...")

        # Decode pool for the K7 overlap work (probe pre-pass + per-step
        # image decode). Declared before the try so the finally can shut it
        # down on every exit path.
        decode_pool: ThreadPoolExecutor | None = None

        try:
            # ── Freeze / unfreeze layers ────────────────────────────────────
            # Freeze all text model parameters
            for param in self.model.parameters():
                param.requires_grad = False

            # Unfreeze vision projection (always trainable)
            for param in self.model.vision_projection.parameters():
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

            # Vision encoder is fully trainable
            for param in vision_encoder.parameters():
                param.requires_grad = True

            # Re-freeze pretrained backbone if configured (the blanket unfreeze
            # above would otherwise override freeze_backbone from __init__)
            if getattr(vision_encoder.config, "freeze_backbone", False):
                backbone = getattr(vision_encoder, "backbone", None)
                if backbone is not None:
                    for param in backbone.parameters():
                        param.requires_grad = False

            trainable_params = list(filter(lambda p: p.requires_grad, self.model.parameters())) + list(
                filter(lambda p: p.requires_grad, vision_encoder.parameters())
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
            vision_batch = max(1, int(self.config.batch_size))
            total_steps = ((len(data) + vision_batch - 1) // vision_batch) * self.config.epochs
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
                        encoder=vision_encoder,
                        encoder_key="vision_encoder_state_dict",
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
                        "Resuming vision training: epoch=%d, step=%d, best_loss=%.4f",
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

            image_size = getattr(getattr(vision_encoder, "config", None), "image_size", 224)
            use_imagenet = getattr(getattr(vision_encoder, "config", None), "use_pretrained", False)

            # K7: shared thread pool for the probe pre-pass and the per-step
            # image decode. Decode is CPU/IO-bound and PIL releases the GIL
            # in its C decoders, so threads (no Windows spawn-pickling) give
            # the overlap. Shut down in the finally.
            decode_pool = ThreadPoolExecutor(
                max_workers=min(8, os.cpu_count() or 4), thread_name_prefix="vision-decode"
            )

            def _probe_image(p: object) -> None:
                """V-2 readability probe: PIL verify() parses the header
                without decoding pixels - cheap, and safe on the pool."""
                from PIL import Image as _PILImage

                with _PILImage.open(str(p)) as _probe:
                    _probe.verify()

            def _probed_stream(items: list[dict[str, Any]], label: str):
                """Yield (index, image, text, probe_future|None) in item
                order. Probes go to the pool in bounded chunks so a
                558k-item dataset overlaps its header reads without
                materializing 558k Future objects at once."""
                chunk = 512
                entries: list[tuple[int, object, str]] = []
                for i, item in enumerate(items):
                    image = item.get("image")
                    text = item.get("text", "")
                    if image is None or not text:
                        logger.warning(f"Skipping vision {label} item {i}: missing image or text")
                        continue
                    entries.append((i, image, text))
                for cs in range(0, len(entries), chunk):
                    window = entries[cs : cs + chunk]
                    futs = [
                        decode_pool.submit(_probe_image, im) if isinstance(im, (str, Path)) else None
                        for _, im, _ in window
                    ]
                    for (i, im, tx), fut in zip(window, futs):
                        yield i, im, tx, fut

            # V-2: lazy preprocess. Store image refs (paths or PIL objects)
            # only - never preprocess/.to(device) eagerly. At LLaVA-Pretrain
            # scale (558K - 600 KB tensors) eager prep would OOM the GPU
            # immediately on a 16 GB VRAM budget. Tensors are built per-step
            # below and freed after backward.
            # K7: the probes run on the pool via _probed_stream; result
            # order follows item order, so skip/log semantics match the
            # old serial loop.
            pairs: list[tuple[object, list[int]]] = []
            for i, image, text, fut in _probed_stream(data, "data"):
                if fut is not None:
                    try:
                        fut.result()
                    except Exception as exc:
                        logger.warning(f"Skipping vision data item {i}: {exc}")
                        continue
                # Encode text to token IDs (tokenizer stays on the main thread)
                token_ids = self.tokenizer.encode(text)
                if len(token_ids) < 1:
                    continue
                pairs.append((image, token_ids))

            if not pairs:
                raise ValueError("No valid image-text pairs found in training data.")

            # V-6: optional held-out validation pairs. Same lightweight
            # readability probe as train (pooled); same lazy-preprocess
            # discipline (no .to(device) until inside the eval pass). Pairs
            # whose caption is <2 tokens are skipped silently because the
            # next-token loss can't be computed on them - matches the
            # train-loop drop policy at min_len < 1.
            val_pairs: list[tuple[object, list[int]]] = []
            if val_data:
                for i, v_image, v_text, v_fut in _probed_stream(val_data, "val"):
                    if v_fut is not None:
                        try:
                            v_fut.result()
                        except Exception as exc:
                            logger.warning(f"Skipping vision val item {i}: {exc}")
                            continue
                    v_token_ids = self.tokenizer.encode(v_text)
                    if len(v_token_ids) < 2:
                        continue
                    val_pairs.append((v_image, v_token_ids))
                if val_pairs:
                    logger.info(f"Vision validation: {len(val_pairs)} held-out pairs")

            self._emit_progress(5, f"Prepared {len(pairs)} image-text pairs")
            logger.info(f"Vision training: {len(pairs)} pairs, {self.config.epochs} epochs")

            # ── Training loop ───────────────────────────────────────────────
            vision_encoder.train()
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
                        "Vision training: %d caption(s) dropped during "
                        "training for being too short after next-token "
                        "shift. Consider filtering single-token captions "
                        "during data prep.",
                        dropped_short_captions,
                    )

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
                vision_encoder.eval()
                self.model.eval()
                total_loss = 0.0
                total_tokens = 0
                n_batches = 0
                stopped = False
                try:
                    with torch.no_grad():
                        for v_start in range(0, len(val_pairs), vision_batch):
                            if self._should_stop():
                                stopped = True
                                logger.info("Vision validation stopped by user request after %d batch(es)", n_batches)
                                break
                            v_imgs: list[torch.Tensor] = []
                            v_id_lists: list[list[int]] = []
                            for v_ref, v_ids in val_pairs[v_start : v_start + vision_batch]:
                                if len(v_ids) < 2:
                                    continue
                                try:
                                    v_imgs.append(
                                        preprocess_image(
                                            v_ref,
                                            image_size=image_size,
                                            imagenet_normalize=use_imagenet,
                                        )
                                    )
                                    v_id_lists.append(v_ids)
                                except Exception as exc:
                                    logger.debug("Vision val: preprocess failed (%s); skipping", exc)
                                    continue
                            if not v_imgs:
                                continue
                            v_batch = torch.cat(v_imgs, dim=0).to(self.device)
                            # Build text/target tensors on CPU, one transfer
                            # each (not per-sample device writes).
                            v_max = max(len(t) for t in v_id_lists)
                            v_text_tensor = torch.zeros(len(v_id_lists), v_max, dtype=torch.long)
                            v_targets = torch.full((len(v_id_lists), v_max - 1), -100, dtype=torch.long)
                            for bi, t in enumerate(v_id_lists):
                                row = torch.tensor(t, dtype=torch.long)
                                v_text_tensor[bi, : len(t)] = row
                                v_targets[bi, : len(t) - 1] = row[1:]
                            v_text_tensor = v_text_tensor.to(self.device)
                            v_targets = v_targets.to(self.device)
                            with torch.amp.autocast(
                                "cuda",
                                dtype=self._amp_dtype,
                                enabled=self.config.use_amp and self.device.type == "cuda",
                            ):
                                v_feats = vision_encoder(v_batch)
                                v_logits = self.model.forward_multimodal(
                                    input_ids=v_text_tensor,
                                    vision_features=v_feats,
                                )
                                v_n_patches = v_feats.shape[1]
                                if v_n_patches >= v_logits.shape[1]:
                                    continue
                                v_text_logits = v_logits[:, v_n_patches:-1, :]
                                v_min_len = min(v_text_logits.shape[1], v_targets.shape[1])
                                if v_min_len < 1:
                                    continue
                                v_loss = nn.functional.cross_entropy(
                                    v_text_logits[:, :v_min_len, :].reshape(-1, v_text_logits.size(-1)),
                                    v_targets[:, :v_min_len].reshape(-1),
                                    ignore_index=-100,
                                )
                            v_n_tokens = int((v_targets[:, :v_min_len] != -100).sum().item())
                            if v_n_tokens == 0:
                                continue
                            total_loss += v_loss.item() * v_n_tokens
                            total_tokens += v_n_tokens
                            n_batches += 1
                finally:
                    vision_encoder.train()
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
                        checkpoint_dir / f"{self._checkpoint_stem}_vision_step.pt",
                        encoder=vision_encoder,
                        encoder_key="vision_encoder_state_dict",
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                    )

            # best_written = the best loss whose checkpoint actually reached
            # disk (seeded from the resumed lineage's best; inf on a fresh
            # run). state.best_loss stays the pure metric.
            best_written = self.state.best_loss
            failed_best_saves = 0

            # When resuming, start from the saved epoch (mirrors train())
            start_epoch = self.state.epoch
            for epoch in range(start_epoch, self.config.epochs):
                if self._should_stop():
                    logger.info("Vision training stopped by user request")
                    break

                # Time limit check
                if self.config.max_training_seconds > 0 and (time.time() - start_time) > self.config.max_training_seconds:
                    logger.info("Vision training: time limit reached")
                    break

                epoch_loss = 0.0
                epoch_steps = 0
                train_interrupted = False

                # V-1: gradient accumulation. Other train_* methods honor
                # config.max_grad_accumulation; vision must too. Only count
                # successful backwards (skipped/short-caption samples don't
                # advance the boundary counter).
                accum_steps = max(1, self.config.max_grad_accumulation)
                accum_count = 0

                # Shuffle training pairs each epoch
                random.shuffle(pairs)

                # Zero grads once at the start of the epoch; thereafter only
                # zero after a real optimizer step (boundary or remainder).
                optimizer.zero_grad()

                n_epoch_batches = (len(pairs) + vision_batch - 1) // vision_batch
                batch_starts = list(range(0, len(pairs), vision_batch))

                # V-2: lazy preprocess, one BATCH at a time (real batching,
                # Phase 4.5 step 4). K7 overlap (2026-07-19): batch n+1's
                # image DECODE runs on the pool while batch n trains, so the
                # GPU no longer idles through serial PIL work. Augmentation
                # (the only RNG consumer) stays on the MAIN thread in batch
                # order, so seeded runs draw the same augment sequence as
                # the old serial loop. Per-image failures still skip the
                # sample, not the batch.
                def _submit_decode(bstart: int) -> list[tuple[str, Any, list[int]]]:
                    """Queue one batch's decodes. Only PATH refs go to the
                    pool (file decode is the I/O-bound win; an in-memory
                    PIL object gains nothing from a thread and a LAZY
                    file-backed one must not be load()ed concurrently, so
                    'raw' refs decode on the main thread at consumption).
                    The <2-token filter runs here, but its counter and
                    warning fire at CONSUMPTION so an aborted run never
                    reports drops from batches that never trained."""
                    entries: list[tuple[str, Any, list[int]]] = []
                    for image_ref, token_ids in pairs[bstart : bstart + vision_batch]:
                        if len(token_ids) < 2:
                            entries.append(("drop", None, token_ids))
                            continue
                        if isinstance(image_ref, (str, Path)):
                            entries.append(
                                (
                                    "fut",
                                    decode_pool.submit(
                                        preprocess_image,
                                        image_ref,
                                        image_size=image_size,
                                        imagenet_normalize=use_imagenet,
                                    ),
                                    token_ids,
                                )
                            )
                        else:
                            entries.append(("raw", image_ref, token_ids))
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

                    imgs: list[torch.Tensor] = []
                    id_lists: list[list[int]] = []
                    for kind, payload, token_ids in entries:
                        if kind == "drop":
                            # V-7: too short for next-token loss after shift.
                            if dropped_short_captions == 0:
                                logger.warning(
                                    "Vision caption too short for next-token "
                                    "loss after shift (need >=2 tokens) at "
                                    "epoch %d step %d. Further drops will be "
                                    "summed and reported at end of training.",
                                    epoch + 1,
                                    step,
                                )
                            dropped_short_captions += 1
                            continue
                        try:
                            if kind == "fut":
                                img = payload.result()
                            else:  # "raw": in-memory ref, decoded here
                                img = preprocess_image(
                                    payload,
                                    image_size=image_size,
                                    imagenet_normalize=use_imagenet,
                                )
                        except Exception as exc:
                            logger.warning("Vision step %d: preprocess failed (%s); skipping sample", step, exc)
                            continue
                        # Training-time augmentation (random flip, color jitter),
                        # applied per image so each sample draws its own coin.
                        imgs.append(augment_vision_tensor(img))
                        id_lists.append(token_ids)
                    if not imgs:
                        continue
                    img_batch = torch.cat(imgs, dim=0).to(self.device)

                    # Right-pad text to the batch max; padded target positions
                    # get ignore_index. Causal attention means a real token
                    # never attends to the padding AFTER it, so correctness
                    # needs no attention mask -- only the loss mask. Built on
                    # CPU, one transfer each (K7: not per-sample device writes).
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
                        vision_features = vision_encoder(img_batch)  # [B, patches, v_dim]

                        # Forward through model with vision features
                        # The model concatenates [vision_patches, text_tokens] internally
                        logits = self.model.forward_multimodal(
                            input_ids=text_tensor,
                            vision_features=vision_features,
                        )
                        # logits shape: [B, vision_patches + text_len, vocab_size]

                        # We only compute loss on the text portion
                        # The text tokens start after the vision patches
                        n_patches = vision_features.shape[1]
                        if n_patches >= logits.shape[1]:
                            logger.warning(
                                "Vision patches (%d) >= logits length (%d), skipping batch", n_patches, logits.shape[1]
                            )
                            continue
                        text_logits = logits[:, n_patches:-1, :]  # predict next token

                        # Align lengths (in case of truncation)
                        min_len = min(text_logits.shape[1], targets.shape[1])
                        if min_len < 1:
                            continue

                        loss = nn.functional.cross_entropy(
                            text_logits[:, :min_len, :].reshape(-1, text_logits.size(-1)),
                            targets[:, :min_len].reshape(-1),
                            ignore_index=-100,
                        )
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
                        logger.error(f"Vision training: NaN/Inf loss at epoch {epoch + 1}, step {step}")
                        self._emit_progress(100, "Training aborted: NaN loss")
                        self.state.abort_reason = f"Vision NaN/Inf loss at epoch {epoch + 1}, step {step}"
                        _emit_drop_summary()  # V-7 audit fix
                        return self.state

                    # Safety guard
                    if loss_val > self.config.max_loss:
                        logger.error(f"Vision training: loss {loss_val:.2f} exceeds max {self.config.max_loss}")
                        self._emit_progress(100, "Training aborted: loss too high")
                        self.state.abort_reason = f"Vision loss {loss_val:.2f} exceeded max_loss {self.config.max_loss}"
                        _emit_drop_summary()  # V-7 audit fix
                        return self.state

                    epoch_loss += loss_val
                    epoch_steps += 1
                    if is_boundary:
                        self.state.step += 1
                        _maybe_step_save()

                    # Log every N steps
                    if self.config.log_every > 0 and self.state.step % self.config.log_every == 0:
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

                # Epoch summary
                avg_loss = epoch_loss / max(epoch_steps, 1)
                self.state.training_losses.append(avg_loss)
                self.state.epoch = epoch + 1
                self.state.epoch_start_step = self.state.step
                self._emit_loss(avg_loss)
                logger.info(f"Vision Epoch {epoch + 1}: avg_loss={avg_loss:.4f}")

                # V-6: held-out validation pass after each epoch. Drives
                # best-checkpoint + early-stopping when available; falls
                # back to training avg_loss otherwise.
                val_loss = _run_validation()
                if val_loss is not None:
                    self.state.validation_losses.append(val_loss)
                    logger.info(f"Vision Epoch {epoch + 1}: val_loss={val_loss:.4f}")

                # Rank the epoch only when its metric is complete: a stop
                # that cut the TRAIN loop leaves a partial avg_loss, and a
                # stop that cut VALIDATION leaves val_loss None (never a
                # prefix mean) -- ranking either against full-pass
                # best_loss values could overwrite the real best
                # checkpoint with the stop-epoch weights. But a val pass
                # that COMPLETED before the stop landed is a full metric:
                # it ranks normally (discarding it would silently lose a
                # real best), and the epoch-loop top breaks right after.
                metric_complete = (val_loss is not None) if val_pairs else (not train_interrupted)
                if not metric_complete and self._should_stop():
                    logger.info("Vision training stopped by user request")
                    break

                if self.on_epoch_complete:
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
                improved = tracked_loss < self.state.best_loss
                if improved:
                    self.state.best_loss = tracked_loss
                    self._epochs_without_improvement = 0
                if tracked_loss < best_written:
                    if self._save_checkpoint(
                        checkpoint_dir / f"{self._checkpoint_stem}_vision_best.pt",
                        encoder=vision_encoder,
                        encoder_key="vision_encoder_state_dict",
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
                        logger.info("Vision training: early stopping triggered")
                        break

                # Periodic checkpoint (0 = disabled)
                if self.config.save_every > 0 and (epoch + 1) % self.config.save_every == 0:
                    stem = self._checkpoint_stem
                    v_num = (epoch + 1) // self.config.save_every
                    self._save_checkpoint(
                        checkpoint_dir / f"{stem}_vision{v_num}.pt",
                        encoder=vision_encoder,
                        encoder_key="vision_encoder_state_dict",
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                    )
                    self._cleanup_periodic_checkpoints(checkpoint_dir, f"{stem}_vision", keep=3)

            # ── Cleanup ─────────────────────────────────────────────────────
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
                logger.error(f"Vision training: {msg}")
                self.state.abort_reason = msg
                self._emit_progress(100, "Vision training finished with errors")
            else:
                self._emit_progress(100, "Vision training complete!")

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
                vision_encoder.eval()
                for param in self.model.parameters():
                    param.requires_grad = True
            except Exception:
                logger.error("Vision training: post-run state restore failed", exc_info=True)
