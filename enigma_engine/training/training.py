"""
Training Module for Enigma AI Engine

Provides:
- TrainingConfig: Configuration for training
- Trainer: Basic fine-tuning with progress callbacks

Usage:
    from enigma_engine.training.training import Trainer, TrainingConfig

    config = TrainingConfig(epochs=10, batch_size=4, learning_rate=1e-4)
    trainer = Trainer(model, tokenizer, config)
    trainer.train(data)
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LambdaLR,
    SequentialLR,
)

logger = logging.getLogger(__name__)


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


# =============================================================================
# AdEMAMix OPTIMIZER (Pagliardini et al. 2024, R30)
# =============================================================================


class AdEMAMix(torch.optim.Optimizer):
    """AdEMAMix: Optimizer with two exponential moving averages.

    Maintains a fast EMA (beta1) and a slow EMA (beta3) of gradients.
    The update direction is an interpolation controlled by ``alpha``:

        m_fast = beta1 * m_fast + (1 - beta1) * grad
        m_slow = beta3 * m_slow + (1 - beta3) * grad
        update = alpha * m_fast + (1 - alpha) * m_slow

    The slow EMA retains gradient information from early training that
    standard AdamW (single EMA) forgets.  Benefits emerge at longer
    training runs (>10K steps).

    Args:
        params: Iterable of parameters or param groups.
        lr: Learning rate (default: 1e-4).
        betas: Coefficients for fast EMA and second moment (default: (0.9, 0.95)).
        beta3: Coefficient for slow EMA (default: 0.9999).
        alpha: Interpolation between fast and slow EMA (default: 5.0).
            Higher alpha = more weight on fast EMA.
        eps: Term for numerical stability (default: 1e-8).
        weight_decay: Decoupled weight decay (default: 0.01).
    """

    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.95),
        beta3: float = 0.9999,
        alpha: float = 5.0,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if not 0.0 <= beta3 < 1.0:
            raise ValueError(f"Invalid beta3: {beta3}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")

        defaults = dict(
            lr=lr,
            betas=betas,
            beta3=beta3,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            beta3 = group["beta3"]
            alpha = group["alpha"]
            eps = group["eps"]
            lr = group["lr"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdEMAMix does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["m_fast"] = torch.zeros_like(p)
                    state["m_slow"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                state["step"] += 1
                step = state["step"]

                m_fast = state["m_fast"]
                m_slow = state["m_slow"]
                v = state["v"]

                # Update EMAs
                m_fast.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                m_slow.mul_(beta3).add_(grad, alpha=1.0 - beta3)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                # Bias correction
                bc_fast = 1.0 - beta1**step
                bc_slow = 1.0 - beta3**step
                bc_v = 1.0 - beta2**step

                m_fast_hat = m_fast / bc_fast
                m_slow_hat = m_slow / bc_slow
                v_hat = v / bc_v

                # Combined update direction
                update = alpha * m_fast_hat + (1.0 - alpha) * m_slow_hat
                denom = v_hat.sqrt().add_(eps)

                # Decoupled weight decay
                if wd > 0:
                    p.mul_(1.0 - lr * wd)

                p.addcdiv_(update, denom, value=-lr)

        return loss


# =============================================================================
# MUON OPTIMIZER (R34)
# =============================================================================


class Muon(torch.optim.Optimizer):
    """Muon: Matrix orthogonalization optimizer via Newton-Schulz iteration.

    For weight matrices (2D params), applies Newton-Schulz iterations
    to approximately orthogonalize the gradient, then uses it as the
    update direction.  For non-matrix params (biases, norms, 1D),
    falls back to standard momentum SGD.

    Orthogonalized gradients improve conditioning and can provide
    ~2x training efficiency vs AdamW on some workloads.

    Args:
        params: Iterable of parameters or param groups.
        lr: Learning rate (default: 0.02).
        momentum: Momentum coefficient (default: 0.95).
        ns_steps: Number of Newton-Schulz iterations (default: 5).
        weight_decay: Decoupled weight decay (default: 0.01).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        ns_steps: int = 5,
        weight_decay: float = 0.01,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if ns_steps < 1:
            raise ValueError(f"ns_steps must be >= 1, got {ns_steps}")

        defaults = dict(lr=lr, momentum=momentum, ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @staticmethod
    def _newton_schulz(G: torch.Tensor, steps: int) -> torch.Tensor:
        """Approximate orthogonalization via Newton-Schulz iteration.

        Given matrix G, iteratively refines X toward the closest
        orthogonal matrix: X_{k+1} = 0.5 * X_k * (3I - X_k^T X_k).

        For tall matrices (rows > cols), operates on G^T for efficiency.
        """
        assert G.dim() == 2
        a, b = G.shape
        transpose = a < b
        if transpose:
            G = G.T

        # Normalize to unit spectral norm (approximate)
        X = G / (G.norm() + 1e-7)

        # Newton-Schulz iterations: X = 1.5*X - 0.5*X@X^T@X
        # Using the equivalent form for stability
        if X.shape[0] >= X.shape[1]:
            eye = torch.eye(X.shape[1], device=X.device, dtype=X.dtype)
            for _ in range(steps):
                A = X.T @ X
                X = X @ (1.5 * eye - 0.5 * A)
        else:
            eye = torch.eye(X.shape[0], device=X.device, dtype=X.dtype)
            for _ in range(steps):
                A = X @ X.T
                X = (1.5 * eye - 0.5 * A) @ X

        if transpose:
            X = X.T
        return X

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            ns_steps = group["ns_steps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)

                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(grad)

                # For 2D (weight matrices): orthogonalize
                if p.dim() == 2 and min(p.shape) > 1:
                    update = self._newton_schulz(buf, ns_steps)
                else:
                    # 1D params (bias, norm): use momentum directly
                    update = buf

                # Decoupled weight decay
                if wd > 0:
                    p.mul_(1.0 - lr * wd)

                p.add_(update, alpha=-lr)

        return loss


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
    keeps non-deterministic ops (e.g. ``index_add_`` used by MoE scatter)
    from crashing the run; they emit a UserWarning instead.
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


def dataset_fingerprint(data: str) -> str:
    """Compute a SHA-256 fingerprint of training data.

    For large strings (>100 MB), hashes a sample to avoid creating
    a full copy via encode().  Returns the first 16 hex characters
    for compact logging.
    """
    if len(data) > 100_000_000:
        # Sample head + tail + length for large data
        h = hashlib.sha256()
        h.update(f"len={len(data)}".encode())
        h.update(data[:1_000_000].encode("utf-8"))
        h.update(data[-1_000_000:].encode("utf-8"))
        return h.hexdigest()[:16]
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]


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

    # Label smoothing (fairseq pattern — reduces overconfidence)
    label_smoothing: float = 0.05  # Reduces overconfidence, preserves generality

    # Validation split
    val_split: float = 0.1  # 10% held out to detect overfitting early

    # EMA weight averaging (smooths training noise, use EMA for eval)
    ema_decay: float = 0.0  # 0.0 = disabled, 0.999 or 0.9999 typical

    # Gradient noise injection (Neelakantan et al.)
    # Adds Gaussian noise N(0, eta/(1+step)^gamma) to gradients.
    # 0.0 = disabled. 0.01-0.3 typical for eta, 0.55 typical for gamma.
    gradient_noise_eta: float = 0.01
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

    # T2-6: LR schedule type.  "wsd" = Warmup-Stable-Decay (robust to
    # unknown total steps, safe to stop early).  "cosine" = warmup + cosine
    # decay (previous default).
    schedule_type: str = "wsd"
    # Fraction of total steps used for the decay phase of WSD.
    wsd_decay_fraction: float = 0.1

    # Sequence packing: pack short sequences into max_seq_len rows
    # separated by EOS tokens with block-diagonal attention masks.
    # 30-50% throughput gain by eliminating padding waste.
    use_sequence_packing: bool = False

    # BPE-Dropout (Provilkov et al.): randomly skip BPE merges during
    # training for subword regularization.  0.0 = disabled.  0.1 = skip
    # 10% of merges (recommended).  Acts as free data augmentation -
    # each epoch sees different tokenizations of the same text.
    bpe_dropout: float = 0.1

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

    # LISA (Pan et al. 2024) - Layerwise Importance Sampled AdamW
    # Always trains embed+head+first+last layer, randomly samples K
    # middle layers each step. Outperforms LoRA at same memory.
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

    # Optimizer selection (R30/R34)
    # "adamw" (default), "ademamix" (dual EMA, R30), "muon" (orthogonalization, R34)
    optimizer: str = "adamw"
    # AdEMAMix-specific: slow EMA beta (fast beta uses adam_beta1)
    ademamix_beta3: float = 0.9999
    # AdEMAMix-specific: interpolation factor (higher = more fast EMA)
    ademamix_alpha: float = 5.0
    # AdEMAMix-specific: alpha annealing (S560, Pagliardini et al. 2024)
    # Linearly decay alpha from initial to ademamix_alpha over warmup fraction
    ademamix_alpha_initial: float = 10.0
    ademamix_alpha_warmup: float = 0.1  # fraction of total steps

    # T5-7: Layer-wise Learning Rate Decay (LLRD)
    # Exponentially decay LR from top (task-specific) to bottom (general).
    # lr_layer_i = lr_base * decay^(n_layers - i).
    # 0.0 = disabled (uniform LR). 0.7-0.95 typical for fine-tuning.
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
        # LLRD decay must be in [0, 1)
        if not 0.0 <= self.llrd_decay < 1.0:
            raise ValueError(f"llrd_decay must be in [0.0, 1.0), got {self.llrd_decay}")
        # Gradient noise gamma must be non-negative (used as exponent)
        if self.gradient_noise_gamma < 0:
            raise ValueError(f"gradient_noise_gamma must be >= 0, got {self.gradient_noise_gamma}")
        # Optimizer must be one of the supported values
        if self.optimizer not in ("adamw", "ademamix", "muon"):
            raise ValueError(f"optimizer must be 'adamw', 'ademamix', or 'muon', got '{self.optimizer}'")
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
    best_loss: float = float("inf")
    total_tokens: int = 0
    training_losses: list[float] = field(default_factory=list)
    validation_losses: list[float] = field(default_factory=list)
    abort_reason: str = ""


class EMAWeightAverager:
    """Exponential Moving Average of model weights.

    Maintains shadow copies of every parameter, updated each step:
        shadow = decay * shadow + (1 - decay) * current_param

    Use ``apply()`` before eval/save, ``restore()`` after to swap
    EMA weights in and live weights back.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: list[torch.Tensor] = [p.clone().detach() for p in model.parameters()]
        self._backup: list[torch.Tensor] = []

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update shadow weights toward current model parameters."""
        for shadow, param in zip(self.shadow, model.parameters()):
            shadow.lerp_(param.detach(), 1.0 - self.decay)

    def apply(self, model: nn.Module) -> None:
        """Swap EMA weights into the model (back up live weights)."""
        self._backup = [p.clone() for p in model.parameters()]
        for param, shadow in zip(model.parameters(), self.shadow):
            param.data.copy_(shadow)

    def restore(self, model: nn.Module) -> None:
        """Restore live weights from backup (undo ``apply()``)."""
        for param, backup in zip(model.parameters(), self._backup):
            param.data.copy_(backup)
        self._backup = []

    def state_dict(self) -> dict[str, list[torch.Tensor]]:
        """Serialize EMA state for checkpointing."""
        return {"shadow": self.shadow}

    def load_state_dict(self, state: dict[str, list[torch.Tensor]]) -> None:
        """Restore EMA state from checkpoint."""
        self.shadow = [s.clone() for s in state["shadow"]]


class SWAWeightAverager:
    """Stochastic Weight Averaging (Izmailov et al., R13).

    Maintains a running arithmetic mean of model weights collected
    at fixed intervals.  Unlike EMA, all snapshots contribute
    equally - this averages over the loss landscape rather than
    decaying toward the latest weights.

    Call ``update()`` every ``update_interval`` optimizer steps
    (e.g. once per epoch or every 100 steps).  Call ``apply()``
    before eval/save to swap averaged weights in, ``restore()``
    to bring back the live training weights.
    """

    def __init__(self, model: nn.Module, update_interval: int = 1) -> None:
        self.update_interval = max(1, update_interval)
        self.n_averaged = 0
        self.avg: list[torch.Tensor] = [p.clone().detach() for p in model.parameters()]
        self._backup: list[torch.Tensor] = []

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        """Accumulate a new weight snapshot if at an averaging boundary."""
        if step % self.update_interval != 0:
            return
        self.n_averaged += 1
        for avg, param in zip(self.avg, model.parameters()):
            # Running mean: avg += (param - avg) / n
            avg.add_((param.detach() - avg) / self.n_averaged)

    def apply(self, model: nn.Module) -> None:
        """Swap SWA-averaged weights into the model."""
        self._backup = [p.clone() for p in model.parameters()]
        for param, avg in zip(model.parameters(), self.avg):
            param.data.copy_(avg)

    def restore(self, model: nn.Module) -> None:
        """Restore live training weights from backup."""
        for param, backup in zip(model.parameters(), self._backup):
            param.data.copy_(backup)
        self._backup = []

    def state_dict(self) -> dict[str, Any]:
        """Serialize SWA state for checkpointing."""
        return {"avg": self.avg, "n_averaged": self.n_averaged}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore SWA state from checkpoint."""
        self.avg = [a.clone() for a in state["avg"]]
        self.n_averaged = state.get("n_averaged", 0)


# =============================================================================
# SEQUENCE PACKING
# =============================================================================


def pack_sequences(
    encoded_seqs: list[list[int]],
    max_length: int,
    eos_id: int,
    pad_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack sequences and materialise tensors + 4D masks.

    Convenience wrapper around :func:`pack_sequences_lazy` +
    :func:`build_packing_masks` that returns ready-to-use tensors.
    Fine for small datasets / tests; for large corpora use the lazy
    variant so masks can be built per-batch.

    Returns:
        (packed_tensor, mask_tensor):
        - packed_tensor: ``(num_rows, max_length)`` long tensor.
        - mask_tensor: ``(num_rows, 1, max_length, max_length)``
          float tensor.  0 = attend, -inf = block.
    """
    rows, boundaries = pack_sequences_lazy(encoded_seqs, max_length, eos_id, pad_id)

    # Pad rows ? tensor
    packed = []
    for row in rows:
        pad_len = max_length - len(row)
        packed.append(row + [pad_id] * pad_len)
    packed_tensor = torch.tensor(packed, dtype=torch.long)

    # Build all masks at once (only safe for small datasets)
    mask_tensor = build_packing_masks(boundaries, max_length)

    return packed_tensor, mask_tensor


def pack_sequences_lazy(
    encoded_seqs: list[list[int]],
    max_length: int,
    eos_id: int,
    pad_id: int = 0,
) -> tuple[list[list[int]], list[list[tuple[int, int]]]]:
    """Pack multiple short sequences into max_length rows (no tensors).

    Packing uses Best-Fit Decreasing (multipack): sort longest first,
    place each sequence into the bin with the tightest remaining
    capacity via a min-heap.  Reduces padding waste to ~2-5%.

    Returns:
        (rows, boundaries):
        - rows: list of token-id lists (unpadded, len <= max_length).
        - boundaries: per-row list of (start, end) document spans.
    """
    if max_length < 1:
        raise ValueError(f"max_length must be >= 1, got {max_length}")

    import heapq

    sorted_seqs = sorted(encoded_seqs, key=len, reverse=True)

    heap: list[tuple[int, int]] = []  # (remaining, bin_idx)
    bin_rows: list[list[int]] = []
    bin_bounds: list[list[tuple[int, int]]] = []

    truncated_count = 0
    for seq in sorted_seqs:
        needed = len(seq) + 1  # +1 for EOS separator
        if needed > max_length:
            truncated_count += 1
            seq = seq[: max_length - 1]
            needed = max_length

        placed = False
        rejects: list[tuple[int, int]] = []
        while heap:
            remaining, idx = heapq.heappop(heap)
            if remaining >= needed:
                start = max_length - remaining
                bin_rows[idx].extend(seq)
                bin_rows[idx].append(eos_id)
                end = start + needed
                bin_bounds[idx].append((start, end))
                new_remaining = remaining - needed
                if new_remaining > 0:
                    heapq.heappush(heap, (new_remaining, idx))
                placed = True
                break
            else:
                rejects.append((remaining, idx))
        for item in rejects:
            heapq.heappush(heap, item)

        if not placed:
            idx = len(bin_rows)
            bin_rows.append(list(seq) + [eos_id])
            bin_bounds.append([(0, needed)])
            remaining = max_length - needed
            if remaining > 0:
                heapq.heappush(heap, (remaining, idx))

    if truncated_count > 0:
        logger.warning(
            "pack_sequences: %d sequence(s) exceeded max_length=%d and were truncated", truncated_count, max_length
        )

    return bin_rows, bin_bounds


def build_packing_masks(
    boundaries: list[list[tuple[int, int]]],
    max_length: int,
) -> torch.Tensor:
    """Build 4D block-diagonal causal masks from packing boundaries.

    Args:
        boundaries: Per-row list of (start, end) document spans.
        max_length: Row length (T).

    Returns:
        ``(num_rows, 1, T, T)`` float tensor.  0 = attend, -inf = block.
    """
    neg_inf = float("-inf")
    causal = torch.triu(torch.full((max_length, max_length), neg_inf), diagonal=1)

    masks = []
    for bounds in boundaries:
        mask = torch.full((max_length, max_length), neg_inf)
        for start, end in bounds:
            span = end - start
            mask[start:end, start:end] = causal[:span, :span]
        masks.append(mask)

    # Clamp -inf to a large finite negative so padded positions (where all
    # attention keys are blocked) produce softmax([-1e9,...]) - 0 instead of
    # softmax([-inf,...]) = NaN.  This NaN propagates through the residual
    # stream under torch.compile's compiled kernels and kills training.
    # Valid causal masking is unaffected: -1e9 is effectively zero after exp().
    stacked = torch.stack(masks).unsqueeze(1)  # (B, 1, T, T)
    return stacked.clamp(min=-1e9)


# =============================================================================
# NEAR-DUPLICATE DETECTION (T4-1)
# =============================================================================


def minhash_dedup(
    texts: list[str],
    threshold: float = 0.75,
    num_hashes: int = 128,
    shingle_size: int = 3,
) -> list[str]:
    """Remove near-duplicate texts using MinHash (approximate Jaccard).

    Two-pass dedup:
    1. Exact duplicates via hash set (keep first occurrence).
    2. MinHash with *num_hashes* hash functions - texts with estimated
       Jaccard similarity > *threshold* are near-duplicates.  The longest
       text in each near-duplicate cluster is kept.

    Pure Python - no external dependencies.

    Args:
        texts: Input text list.
        threshold: Jaccard similarity threshold for near-duplicate
            detection.  Default 0.75 matches the FineWeb / DCLM / SmolLM3
            industry standard (FineWeb tech report arxiv:2406.17557 section 3.4:
            "targeting documents that are at least 75% similar").
        num_hashes: Number of MinHash permutations (more = more accurate,
            slower).  128 is a good tradeoff.
        shingle_size: Character n-gram size for shingling.

    Returns:
        Deduplicated list preserving original order (longest kept per
        near-duplicate cluster).
    """
    if len(texts) <= 1:
        return list(texts)

    import hashlib
    import struct

    # -- Step 1: Exact dedup (keep first) --
    seen_exact: dict[str, int] = {}
    unique_indices: list[int] = []
    for i, t in enumerate(texts):
        if t not in seen_exact:
            seen_exact[t] = i
            unique_indices.append(i)

    if not unique_indices:
        return []

    # -- Step 2: Shingling --
    def _shingles(text: str) -> set[str]:
        if len(text) < shingle_size:
            return {text}
        return {text[i : i + shingle_size] for i in range(len(text) - shingle_size + 1)}

    shingle_sets = [_shingles(texts[i]) for i in unique_indices]

    # -- Step 3: MinHash signatures --
    # Use MurmurHash-style: h_k(x) = (a*hash(x) + b) mod large_prime
    _LARGE_PRIME = (1 << 61) - 1
    # Deterministic seeds for reproducibility
    rng_bytes = hashlib.sha256(b"minhash_dedup_seed").digest()
    coeffs: list[tuple[int, int]] = []
    for k in range(num_hashes):
        seed = hashlib.sha256(rng_bytes + struct.pack(">I", k)).digest()
        a = int.from_bytes(seed[:8], "big") % _LARGE_PRIME
        b = int.from_bytes(seed[8:16], "big") % _LARGE_PRIME
        if a == 0:
            a = 1
        coeffs.append((a, b))

    def _minhash_sig(shings: set[str]) -> list[int]:
        sig = [_LARGE_PRIME] * num_hashes
        for s in shings:
            # Deterministic hash - Python's hash() is randomised per-process
            h = int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:8], "big")
            for k, (a, b_) in enumerate(coeffs):
                val = (a * h + b_) % _LARGE_PRIME
                if val < sig[k]:
                    sig[k] = val
        return sig

    sigs = []
    for idx, s in enumerate(shingle_sets):
        sigs.append(_minhash_sig(s))
        # Yield GIL every 50 signatures so the GUI thread
        # can process events during large dedup operations.
        if (idx + 1) % 50 == 0:
            import time

            time.sleep(0)

    # -- Step 4: Find near-duplicate clusters --
    n = len(unique_indices)
    removed: set[int] = set()  # indices into unique_indices

    for i in range(n):
        if i in removed:
            continue
        # Yield GIL periodically during O(n^2) pairwise comparison
        if i % 100 == 0:
            import time

            time.sleep(0)
        for j in range(i + 1, n):
            if j in removed:
                continue
            # Estimate Jaccard similarity
            matches = sum(1 for k in range(num_hashes) if sigs[i][k] == sigs[j][k])
            sim = matches / num_hashes
            if sim >= threshold:
                # Keep the longer one, remove shorter
                if len(texts[unique_indices[i]]) >= len(texts[unique_indices[j]]):
                    removed.add(j)
                else:
                    removed.add(i)
                    break  # i is removed, stop comparing

    # -- Step 5: Build result preserving original order --
    kept = sorted(i for idx, i in enumerate(unique_indices) if idx not in removed)
    return [texts[i] for i in kept]


# =============================================================================
# DATA VALIDATION (DQ-B)
# =============================================================================


@dataclass
class DataValidationResult:
    """Result of validating training data before use.

    Attributes:
        is_valid: True if data is usable (may still have warnings).
        total_sequences: Total number of sequences found.
        warnings: Non-fatal issues (short sequences, duplicates, etc.).
        errors: Fatal issues that block training.
        stats: Summary statistics about the data.
    """

    is_valid: bool = True
    total_sequences: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


def validate_training_data(
    data: str,
    min_length: int = 5,
    max_length: int = 50000,
    warn_duplicate_ratio: float = 0.3,
) -> DataValidationResult:
    """Validate training data before training begins (DQ-B).

    Checks for:
    - Empty or whitespace-only data
    - Encoding issues (null bytes, control characters)
    - Very short sequences (< min_length chars)
    - Very long sequences (> max_length chars)
    - Duplicate sequences
    - Empty lines that produce no training signal

    Args:
        data: Raw training data text.
        min_length: Minimum chars per sequence to be useful.
        max_length: Maximum chars per sequence before truncation warning.
        warn_duplicate_ratio: Warn if duplicates exceed this fraction.

    Returns:
        DataValidationResult with is_valid, warnings, errors, and stats.
    """
    result = DataValidationResult()

    # Check for empty/whitespace data
    if not data or not data.strip():
        result.is_valid = False
        result.errors.append("Training data is empty or whitespace-only.")
        return result

    # Check for encoding issues (null bytes, stray control chars)
    null_count = data.count("\x00")
    if null_count > 0:
        result.warnings.append(f"Data contains {null_count} null bytes — may indicate encoding corruption.")

    # Count non-printable control characters (exclude newline, tab, CR)
    # Use regex instead of character-by-character Python loop - the C
    # regex engine is orders of magnitude faster on multi-GB strings.
    ctrl_count = len(re.findall(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", data[:5_000_000]))
    if ctrl_count > 0:
        result.warnings.append(f"Data contains {ctrl_count}+ control characters.")

    # Split into sequences (paragraphs or double-newline separated)
    # Use the same logic as _parse_training_data for consistency
    lines = [ln.strip() for ln in data.split("\n") if ln.strip()]

    if not lines:
        result.is_valid = False
        result.errors.append("No non-empty lines found in data.")
        return result

    result.total_sequences = len(lines)

    # Check individual sequence quality
    short_count = 0
    long_count = 0
    empty_count = data.count("\n\n\n")  # triple+ newlines = wasted space
    lengths: list[int] = []

    for line in lines:
        line_len = len(line)
        lengths.append(line_len)
        if line_len < min_length:
            short_count += 1
        if line_len > max_length:
            long_count += 1

    if short_count > 0:
        pct = short_count / len(lines) * 100
        result.warnings.append(
            f"{short_count} sequences ({pct:.0f}%) shorter than {min_length} chars — may not provide useful signal."
        )

    if long_count > 0:
        result.warnings.append(
            f"{long_count} sequences exceed {max_length:,} chars — will be truncated to model max_seq_len."
        )

    if empty_count > 5:
        result.warnings.append(f"{empty_count} runs of 3+ empty lines — consider cleaning whitespace.")

    # Check for duplicates
    unique_lines = set(lines)
    dup_count = len(lines) - len(unique_lines)
    if dup_count > 0:
        dup_ratio = dup_count / len(lines)
        msg = f"{dup_count} duplicate sequences ({dup_ratio:.0%} of data)."
        if dup_ratio >= warn_duplicate_ratio:
            result.warnings.append(msg + " High duplication may cause over-fitting.")
        else:
            result.warnings.append(msg)

    # Compute stats
    result.stats = {
        "total_chars": len(data),
        "total_lines": len(lines),
        "unique_lines": len(unique_lines),
        "duplicates": dup_count,
        "avg_length": sum(lengths) / len(lengths) if lengths else 0,
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "short_sequences": short_count,
        "long_sequences": long_count,
    }

    # Data is valid if no errors (warnings are informational)
    result.is_valid = len(result.errors) == 0

    return result


# =============================================================================
# TRAINER CLASS
# =============================================================================


class Trainer:
    """
    Trainer for fine-tuning Enigma models.

    Supports:
    - Basic fine-tuning on text data
    - Q&A pair training
    - JSONL training data
    - Progress callbacks for GUI integration
    - Checkpoint saving/loading
    - Gradient accumulation
    - Mixed precision training

    Usage:
        trainer = Trainer(model, tokenizer, config)
        trainer.on_progress = lambda pct, msg: print(f"{pct}% - {msg}")
        trainer.train(data)
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

        # Build memory budget - scales all RAM/VRAM-sensitive constants
        # to the actual hardware.  Explicit override via config field.
        from enigma_engine.core.hardware_detection import TrainingMemoryBudget

        self._budget = TrainingMemoryBudget(
            ram_gb=self.config.training_memory_gb,  # 0 = auto
            vram_gb=0.0,  # always auto-detect VRAM from device
        )

        # Callbacks for progress updates
        self.on_progress: Callable[[int, str], None] | None = None
        self.on_loss: Callable[[float], None] | None = None
        self.on_epoch_complete: Callable[[int, float], None] | None = None
        self.on_throughput: Callable[[int, float], None] | None = None
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

        # Resolve pad token ID from the tokenizer — used for
        # padding batches and as ignore_index in cross-entropy loss.
        # Built-in tokenizer uses 0, tiktoken uses base+0.
        self.pad_token_id: int = getattr(self.tokenizer, "pad_token_id", 0)

        # Rolling best checkpoint tracking (CK-C)
        # List of (loss, path) tuples sorted by loss ascending
        self._rolling_checkpoints: list[tuple[float, Path]] = []

        # EMA weight averaging
        self.ema: EMAWeightAverager | None = None
        if self.config.ema_decay > 0:
            self.ema = EMAWeightAverager(self.model, decay=self.config.ema_decay)
            logger.info(f"EMA enabled with decay={self.config.ema_decay}")

        # SWA weight averaging (R13)
        self.swa: SWAWeightAverager | None = None
        if self.config.swa_update_interval > 0:
            self.swa = SWAWeightAverager(self.model, update_interval=self.config.swa_update_interval)
            logger.info(f"SWA enabled with interval={self.config.swa_update_interval}")

        # LISA: Layerwise Importance Sampled AdamW (R26)
        self.lisa = None
        if self.config.use_lisa:
            n_layers = getattr(getattr(self.model, "config", None), "n_layers", 0)
            if n_layers >= 3:
                from enigma_engine.core.progressive_growing import LISAScheduler

                self.lisa = LISAScheduler(self.model, n_layers, self.config.lisa_activated_layers)
                logger.info(
                    f"LISA enabled: {self.config.lisa_activated_layers} "
                    f"active middle layers per step "
                    f"(of {n_layers - 2} middle layers)"
                )
            else:
                logger.warning(f"LISA disabled: need >= 3 layers, got {n_layers}")

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
        """Setup optimizer and learning rate scheduler."""
        # T5-7: Layer-wise Learning Rate Decay (LLRD)
        # When enabled, assign exponentially decaying LR per layer.
        if self.config.llrd_decay > 0.0:
            param_groups = self._build_llrd_param_groups()
        else:
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

        opt_name = self.config.optimizer

        if opt_name == "ademamix":
            # AdEMAMix (R30): dual EMA optimizer
            self.optimizer = AdEMAMix(
                param_groups,
                lr=self.config.learning_rate,
                betas=(self.config.adam_beta1, self.config.adam_beta2),
                beta3=self.config.ademamix_beta3,
                alpha=self.config.ademamix_alpha,
                eps=self.config.adam_eps,
            )
            logger.info("Using AdEMAMix optimizer (R30)")
        elif opt_name == "muon":
            # Muon (R34): Newton-Schulz orthogonalization
            self.optimizer = Muon(
                param_groups,
                lr=self.config.learning_rate,
                momentum=self.config.adam_beta1,
            )
            logger.info("Using Muon optimizer (R34)")
        else:
            # Default: AdamW
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

        # Scheduler set when we know total steps (see _create_scheduler)
        self.scheduler = None

    def _build_llrd_param_groups(self) -> list[dict]:
        """Build per-layer param groups with exponentially decaying LR (T5-7).

        Top layers (task-specific) get full LR, bottom layers (general features)
        get lower LR. Formula: ``lr_layer_i = lr_base * decay^(n_layers - i)``.

        Embed/head params get full LR. Bias/norm params get zero weight decay.
        """
        decay = self.config.llrd_decay
        lr = self.config.learning_rate
        wd = self.config.weight_decay
        n_layers = getattr(getattr(self.model, "config", None), "n_layers", 0)

        # Collect params keyed by layer index
        layer_params: dict[int, list] = {}
        layer_no_decay_params: dict[int, list] = {}
        non_layer_decay: list = []
        non_layer_no_decay: list = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Bias, norm, and embedding params get no weight decay (see D-8).
            is_no_decay = "bias" in name or "norm" in name or "embed" in name

            # Try to extract layer index from name (e.g. "layers.3.attention.wq.weight")
            layer_idx = None
            if ".layers." in name:
                parts = name.split(".")
                for i, p in enumerate(parts):
                    if p == "layers" and i + 1 < len(parts):
                        try:
                            layer_idx = int(parts[i + 1])
                        except ValueError:
                            pass
                        break

            if layer_idx is not None:
                bucket = layer_no_decay_params if is_no_decay else layer_params
                bucket.setdefault(layer_idx, []).append(param)
            else:
                if is_no_decay:
                    non_layer_no_decay.append(param)
                else:
                    non_layer_decay.append(param)

        groups = []
        # Non-layer params (embeddings, output head) get full LR
        if non_layer_decay:
            groups.append({"params": non_layer_decay, "lr": lr, "weight_decay": wd})
        if non_layer_no_decay:
            groups.append({"params": non_layer_no_decay, "lr": lr, "weight_decay": 0.0})

        # Per-layer groups with decaying LR
        for layer_idx in sorted(set(list(layer_params.keys()) + list(layer_no_decay_params.keys()))):
            # Top layer = n_layers-1 gets full LR, layer 0 gets most decay
            layer_lr = lr * (decay ** (n_layers - 1 - layer_idx))
            if layer_idx in layer_params:
                groups.append(
                    {
                        "params": layer_params[layer_idx],
                        "lr": layer_lr,
                        "weight_decay": wd,
                    }
                )
            if layer_idx in layer_no_decay_params:
                groups.append(
                    {
                        "params": layer_no_decay_params[layer_idx],
                        "lr": layer_lr,
                        "weight_decay": 0.0,
                    }
                )

        logger.info(
            "LLRD enabled: decay=%.3f, %d param groups, bottom LR=%.2e, top LR=%.2e",
            decay,
            len(groups),
            lr * (decay ** max(n_layers - 1, 0)),
            lr,
        )
        return groups

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
            from enigma_engine.core.rl_training import _resolve_amp_dtype as _amp_dt

            amp_dtype = _amp_dt(self.config.amp_dtype)

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

    def _lr_find(
        self,
        batches,
        min_lr: float = 1e-7,
        max_lr: float = 10.0,
        num_steps: int = 100,
    ) -> float:
        """Leslie Smith LR range test (2015).

        Runs a short sweep with exponentially increasing LR, records
        smoothed loss at each step, and returns the LR at the steepest
        descent point (divided by 10 for safety margin).

        The model and optimizer are restored to their original state
        after the sweep so training can proceed cleanly.

        Args:
            batches: Training batches (list or generator).
            min_lr: Starting LR for the sweep.
            max_lr: Ending LR for the sweep.
            num_steps: Number of sweep steps (capped to len(batches)
                when batches is a list).

        Returns:
            Suggested learning rate.
        """
        import copy
        import math

        # Accept both lists and generators.  For generators, pre-collect
        # up to num_steps batches so we know the count and can index.
        if hasattr(batches, "__len__"):
            num_steps = min(num_steps, len(batches))
            batch_list = batches
        else:
            import itertools

            batch_list = list(itertools.islice(batches, num_steps))
            num_steps = len(batch_list)
        if num_steps < 10:
            logger.warning("LR finder: too few batches (%d), keeping default LR", num_steps)
            return self.config.learning_rate

        # Snapshot model + optimizer so we can restore after the sweep
        model_state = copy.deepcopy(self.model.state_dict())
        optim_state = copy.deepcopy(self.optimizer.state_dict())

        # Exponential LR schedule: lr_i = min_lr * (max_lr/min_lr)^(i/N)
        mult = (max_lr / min_lr) ** (1.0 / num_steps)

        # Set starting LR
        for pg in self.optimizer.param_groups:
            pg["lr"] = min_lr

        lrs: list[float] = []
        losses: list[float] = []
        smoothed_loss = 0.0
        best_loss = float("inf")

        use_amp = self.config.use_amp and torch.cuda.is_available()
        amp_dtype = self._amp_dtype

        self.model.train()
        for i in range(num_steps):
            batch_tensor, attention_mask = batch_list[i]
            input_ids = batch_tensor[:, :-1]
            target_ids = batch_tensor[:, 1:]

            self.optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                output = self.model(input_ids)
                logits = output[0] if isinstance(output, tuple) else output
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    target_ids.reshape(-1),
                    ignore_index=self.pad_token_id,
                )

            if torch.isnan(loss) or torch.isinf(loss):
                break  # Diverged - stop sweep

            loss_val = loss.item()
            # Exponential moving average of loss
            smoothed_loss = 0.98 * smoothed_loss + 0.02 * loss_val if i > 0 else loss_val
            # Stop if loss explodes past 4x the best observed
            if smoothed_loss > 4 * best_loss and i > 10:
                break
            if smoothed_loss < best_loss:
                best_loss = smoothed_loss

            lrs.append(self.optimizer.param_groups[0]["lr"])
            losses.append(smoothed_loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip or 1.0)
            self.optimizer.step()

            # Increase LR exponentially
            for pg in self.optimizer.param_groups:
                pg["lr"] *= mult

        # Restore model + optimizer to pre-sweep state
        self.model.load_state_dict(model_state)
        self.optimizer.load_state_dict(optim_state)

        if len(losses) < 10:
            logger.warning("LR finder: too few valid steps (%d), keeping default LR", len(losses))
            return self.config.learning_rate

        # Find steepest descent: largest negative gradient in log-space
        log_lrs = [math.log10(lr) for lr in lrs]
        gradients = [
            (losses[j] - losses[j - 1]) / (log_lrs[j] - log_lrs[j - 1])
            for j in range(1, len(losses))
            if log_lrs[j] != log_lrs[j - 1]
        ]
        if not gradients:
            return self.config.learning_rate

        # The steepest descent point (most negative gradient)
        min_grad_idx = min(range(len(gradients)), key=lambda k: gradients[k])
        suggested_lr = lrs[min_grad_idx + 1]

        # Safety margin: divide by 10
        suggested_lr /= 10.0

        # Clamp to reasonable range
        suggested_lr = max(1e-6, min(suggested_lr, 1e-2))

        logger.info(
            "LR finder: swept %d steps, suggested LR=%.2e (steepest descent at %.2e)",
            len(losses),
            suggested_lr,
            lrs[min_grad_idx + 1],
        )
        return suggested_lr

    def _resolve_amp_dtype(self) -> torch.dtype:
        """Resolve the AMP autocast dtype from config.

        ``"auto"`` picks BF16 when the GPU supports it (Ampere+, Blackwell)
        and falls back to FP16 otherwise.  BF16 has better numeric range
        which avoids many loss-scaling headaches.
        """
        from enigma_engine.core.rl_training import _resolve_amp_dtype

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

    def _emit_throughput(self, tokens: int, step_time: float) -> None:
        """Emit throughput update via callback."""
        if self.on_throughput:
            try:
                self.on_throughput(tokens, step_time)
            except Exception as e:
                logger.debug(f"Throughput callback error: {e}")

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

    def _parse_training_data(self, data: str | list[dict]) -> list[str]:
        """
        Parse training data into sequences.

        Supports:
        - Raw text (split by newlines or double newlines)
        - Q&A format: "Q: question\\nA: answer"
        - JSONL format: {"prompt": "...", "completion": "..."}

        Args:
            data: Raw text or list of dicts

        Returns:
            List of training sequences
        """
        sequences = []

        if isinstance(data, list):
            # Already parsed list of dicts
            for item in data:
                if isinstance(item, dict):
                    prompt = item.get("prompt", item.get("question", ""))
                    # P5-pre-2: ``response`` accepted alongside
                    # ``completion`` / ``answer`` so anchor JSONL
                    # (router.py-style ``{prompt, response, score}``)
                    # parses through the general-data mix path. Without
                    # this fallback the JSONL silently produced empty
                    # sequences and the parser landed in last-resort
                    # line-split (raw ``{...}`` strings as training).
                    completion = item.get("completion", item.get("answer", item.get("response", "")))
                    thinking = item.get("thinking", item.get("reasoning", ""))
                    if prompt and completion:
                        # Wrap thinking in <think> tags if provided
                        if thinking:
                            from enigma_engine.core.reasoning import wrap_reasoning

                            completion = wrap_reasoning(thinking, completion)
                        sequences.append(f"User: {prompt}\nAssistant: {completion}")
                elif isinstance(item, str):
                    sequences.append(item)
            return sequences

        # Raw text - detect format
        data = data.strip()

        # Try JSONL first
        if data.startswith("{"):
            for line in data.split("\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    prompt = item.get("prompt", item.get("question", ""))
                    # P5-pre-2: ``response`` accepted alongside
                    # ``completion`` / ``answer`` (matches anchor JSONL).
                    completion = item.get("completion", item.get("answer", item.get("response", "")))
                    thinking = item.get("thinking", item.get("reasoning", ""))
                    if prompt and completion:
                        if thinking:
                            from enigma_engine.core.reasoning import wrap_reasoning

                            completion = wrap_reasoning(thinking, completion)
                        sequences.append(f"User: {prompt}\nAssistant: {completion}")
                except json.JSONDecodeError:
                    continue
            if sequences:
                return sequences

        # Try Q&A format - normalise to User/Assistant to match
        # the chat inference prompt format.
        qa_pattern = re.compile(r"Q:\s*(.+?)\s*A:\s*(.+?)(?=Q:|$)", re.DOTALL)
        matches = qa_pattern.findall(data)
        if matches:
            for q, a in matches:
                sequences.append(f"User: {q.strip()}\nAssistant: {a.strip()}")
            return sequences

        # Try User/AI dialogue format - normalise role labels
        dialogue_pattern = re.compile(
            r"(?:User|Human):\s*(.+?)\s*(?:AI|Assistant):\s*(.+?)(?=(?:User|Human):|$)", re.DOTALL | re.IGNORECASE
        )
        d_matches = dialogue_pattern.findall(data)
        if d_matches:
            for user_msg, ai_msg in d_matches:
                sequences.append(f"User: {user_msg.strip()}\nAssistant: {ai_msg.strip()}")
            return sequences

        # Fall back to paragraph splitting
        paragraphs = data.split("\n\n")
        for para in paragraphs:
            para = para.strip()
            if len(para) > 50:  # Skip short/noisy paragraphs
                sequences.append(para)

        if not sequences:
            # Last resort: split by lines
            sequences = [line.strip() for line in data.split("\n") if line.strip()]

        return sequences

    def _apply_reasoning_weight(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        base_loss: torch.Tensor,
    ) -> torch.Tensor:
        """Re-compute loss with higher weight on ``<think>`` tokens (CoT-E).

        Identifies tokens that fall inside ``<think>...</think>`` spans
        and multiplies their per-token loss by ``config.reasoning_loss_weight``.
        This forces the model to learn the reasoning *process*, not just
        memorise final answers.

        Falls back to *base_loss* if the reasoning token IDs can't be
        resolved or no think tokens exist in the batch.
        """
        try:
            # Use named attributes (available on all 4 tokenizer classes)
            start_id = getattr(self.tokenizer, "think_start_id", None)
            end_id = getattr(self.tokenizer, "think_end_id", None)

            if start_id is None or end_id is None:
                return base_loss

            # Per-token cross-entropy (not reduced)
            per_token_loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=self.pad_token_id,
                reduction="none",
            ).reshape(targets.shape)

            # Build weight mask: 1.0 everywhere, reasoning_loss_weight
            # inside <think>...</think> spans
            weight = torch.ones_like(per_token_loss)
            w = self.config.reasoning_loss_weight

            for b in range(targets.size(0)):
                in_think = False
                for t in range(targets.size(1)):
                    tok = targets[b, t].item()
                    if tok == start_id:
                        in_think = True
                    if in_think:
                        weight[b, t] = w
                    if tok == end_id:
                        in_think = False

            # Weighted mean (ignoring padding)
            valid = (targets != self.pad_token_id).float()
            weighted = per_token_loss * weight * valid
            denom = (weight * valid).sum()
            if denom > 0:
                return weighted.sum() / denom
            return base_loss

        except Exception:
            logger.debug("Reasoning-weighted loss failed, using base loss", exc_info=True)
            return base_loss

    def _pack_sequences(
        self,
        encoded_seqs: list[list[int]],
        max_length: int,
    ):
        """Pack encoded sequences into dense rows with 4D masks.

        Generator - yields one ``(packed_tensor, mask_4d)`` tuple per
        batch so peak memory stays at ``batch_size * T * T * 4`` bytes
        instead of ``num_rows * T * T * 4``.

        Yields:
            ``(packed_tensor, mask_4d)`` where ``mask_4d`` has shape
            ``(B, 1, T, T)`` with 0 = attend and ``-inf`` = block.
        """
        eos_id = getattr(self.tokenizer, "eos_token_id", 2)
        rows, boundaries = pack_sequences_lazy(
            encoded_seqs,
            max_length=max_length,
            eos_id=eos_id,
            pad_id=self.pad_token_id,
        )

        # Pad rows and build tensors + masks per-batch to stay memory-safe.
        # Keep on CPU - callers move to device just before the forward pass
        # so only one batch lives on GPU at a time.
        batch_size = self.config.batch_size
        for i in range(0, len(rows), batch_size):
            batch_rows = rows[i : i + batch_size]
            batch_bounds = boundaries[i : i + batch_size]

            # Pad each row to max_length
            padded = []
            for row in batch_rows:
                pad_len = max_length - len(row)
                padded.append(row + [self.pad_token_id] * pad_len)

            packed = torch.tensor(padded, dtype=torch.long)
            masks = build_packing_masks(batch_bounds, max_length)
            yield (packed, masks)

    def _score_sequences_by_loss(
        self,
        sequences: list[str],
        max_length: int,
    ) -> list[float]:
        """Score each sequence by model loss (lower = easier).

        Runs a single forward pass per sequence with no gradient
        tracking.  Returns a list of per-sequence losses aligned
        with the input list.
        """
        import torch

        self.model.eval()
        scores: list[float] = []
        pad_id = self.pad_token_id
        try:
            for seq in sequences:
                tokens = self.tokenizer.encode(seq)
                if len(tokens) > max_length:
                    tokens = tokens[:max_length]
                if len(tokens) < 2:
                    scores.append(0.0)
                    continue
                ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
                with torch.no_grad():
                    logits = self.model(ids)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    # Shift: predict next token from each position
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = ids[:, 1:].contiguous()
                    loss = torch.nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=pad_id,
                    )
                scores.append(loss.item())
        finally:
            self.model.train()
        return scores

    # =================================================================
    # Streaming batch generation for large datasets
    # =================================================================

    # Legacy class-level defaults - used by tests and as fallbacks
    # when _budget is not available.  Actual values come from
    # self._budget (TrainingMemoryBudget) which scales to hardware.
    _STREAMING_THRESHOLD = 50_000
    _STREAMING_WINDOW = 5_000

    @staticmethod
    def _write_sequences_to_disk(
        sequences: list[str],
        path: Path,
    ) -> list[int]:
        """Write sequences to a JSONL file and return byte offsets.

        Each sequence is JSON-encoded on one line so embedded
        newlines are safe.  Returns a list of byte offsets (one per
        sequence) that can be used to seek directly to any line.
        """
        offsets: list[int] = []
        with open(path, "wb") as f:
            for seq in sequences:
                offsets.append(f.tell())
                f.write((json.dumps(seq) + "\n").encode("utf-8"))
        return offsets

    @staticmethod
    def _read_sequences_from_disk(
        path: Path,
        offsets: list[int],
        indices: list[int],
    ) -> list[str]:
        """Read specific sequences from disk by byte offset.

        Sorts indices by offset for sequential I/O (faster on HDD
        and friendlier to OS read-ahead), then returns sequences
        in the *original* order of ``indices``.
        """
        # Sort by offset to read file sequentially
        sorted_pairs = sorted(
            ((offsets[i], i, pos) for pos, i in enumerate(indices)),
            key=lambda x: x[0],
        )
        result_map: dict[int, str] = {}
        with open(path, "rb") as f:
            for offset, idx, _pos in sorted_pairs:
                f.seek(offset)
                result_map[idx] = json.loads(f.readline().decode("utf-8"))
        return [result_map[i] for i in indices]

    def _stream_batches(
        self,
        seq_path: Path,
        seq_offsets: list[int],
        indices: list[int],
        max_length: int,
    ):
        """Yield ``(batch_tensor, attention_mask)`` from disk-backed sequences.

        Reads sequences in windows of ``_STREAMING_WINDOW``, tokenizes
        and packs each window via ``_create_batches``, yields the
        resulting batches, then frees them.  Peak memory is bounded
        to one window regardless of dataset size.
        """
        window = self._budget.streaming_window
        for win_start in range(0, len(indices), window):
            win_indices = indices[win_start : win_start + window]
            win_seqs = self._read_sequences_from_disk(seq_path, seq_offsets, win_indices)
            win_batches = self._create_batches(win_seqs, max_length=max_length)
            del win_seqs
            yield from win_batches
            del win_batches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(0)  # Yield GIL

    def _create_batches(
        self,
        sequences: list[str],
        max_length: int | None = None,
    ):
        """Create batches from sequences.

        Generator - yields ``(batch_tensor, attention_mask)`` tuples
        one at a time.  Wrap with ``list()`` when random access or
        ``len()`` is needed (small datasets / LR finder).

        Args:
            sequences: List of text sequences
            max_length: Maximum sequence length.  Defaults to the
                model's ``max_seq_len`` if available, otherwise 512.

        Yields:
            ``(batch_tensor, attention_mask)`` tuples.
            attention_mask has 1 for real tokens, 0 for padding.
            When sequence packing is enabled, attention_mask is a 4D
            tensor ``(B, 1, T, T)`` instead of 2D ``(B, T)``.
        """
        # Use model's configured context length when available
        if max_length is None:
            cfg = getattr(self.model, "config", None)
            max_length = getattr(cfg, "max_seq_len", 512)

        pad_token_id = self.pad_token_id

        # Encode all sequences
        encoded = []
        bpe_dropout = self.config.bpe_dropout
        # Only pass dropout kwarg if the tokenizer supports it
        import inspect

        _enc_sig = inspect.signature(self.tokenizer.encode)
        _supports_dropout = "dropout" in _enc_sig.parameters
        for idx, seq in enumerate(sequences):
            if _supports_dropout and bpe_dropout:
                tokens = self.tokenizer.encode(seq, dropout=bpe_dropout)
            else:
                tokens = self.tokenizer.encode(seq)
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            if len(tokens) >= 5:  # Need enough tokens for meaningful training
                encoded.append(tokens)
            # Yield GIL every 200 sequences so GUI stays responsive
            # during tokenization of large corpora.
            if (idx + 1) % 200 == 0:
                import time

                time.sleep(0)

        if not encoded:
            raise ValueError("No valid sequences after encoding")

        # ---- Sequence packing path ----
        if self.config.use_sequence_packing:
            yield from self._pack_sequences(encoded, max_length)
            return

        # ---- Standard padding path ----
        # Sort by length for efficient batching
        encoded.sort(key=len, reverse=True)

        # Create batches
        batch_size = self.config.batch_size

        for i in range(0, len(encoded), batch_size):
            batch_tokens = encoded[i : i + batch_size]

            # Pad to max length in batch
            max_len = max(len(t) for t in batch_tokens)
            padded = []
            masks = []
            for tokens in batch_tokens:
                pad_len = max_len - len(tokens)
                masks.append([1] * len(tokens) + [0] * pad_len)
                padded.append(tokens + [pad_token_id] * pad_len)

            batch_tensor = torch.tensor(padded, dtype=torch.long, device=self.device)
            attention_mask = torch.tensor(masks, dtype=torch.long, device=self.device)
            yield (batch_tensor, attention_mask)

    def train(
        self,
        data: "str | list | None" = None,
        resume_from: str | Path | None = None,
        data_path: "str | Path | None" = None,
        data_offsets: "list[int] | None" = None,
    ) -> TrainingState:
        """
        Train the model on data.

        Args:
            data: Training data - text string, list of dicts
                (prompt/completion), or list of pre-chunked strings.
                Mutually exclusive with data_path/data_offsets.
            resume_from: Path to checkpoint file to resume from.
                Restores model weights, optimizer, scheduler, scaler,
                epoch/step counters, and loss history.
            data_path: Path to a pre-written JSONL file of sequences
                (one JSON-encoded string per line).  Used with
                data_offsets to skip in-memory sequence parsing.
            data_offsets: Byte offsets into data_path for each
                sequence.  Required when data_path is set.

        Returns:
            Final training state
        """
        self._stop_requested = False

        # Resume from checkpoint if requested
        if resume_from is not None:
            ckpt_path = Path(resume_from)
            if ckpt_path.exists():
                self.load_checkpoint(ckpt_path)
                logger.info(
                    "Resuming from checkpoint: epoch=%d, step=%d, best_loss=%.4f",
                    self.state.epoch,
                    self.state.step,
                    self.state.best_loss,
                )
                self._emit_progress(0, f"Resuming from step {self.state.step}, epoch {self.state.epoch}")
            else:
                logger.warning("Checkpoint not found: %s - training from scratch", ckpt_path)

        self.model.train()
        self._training_start_time = time.monotonic()
        self._epochs_without_improvement = 0

        # Reproducibility: set seed if configured
        if self.config.seed is not None:
            set_training_seed(self.config.seed, deterministic=self.config.deterministic)

        self._emit_progress(0, "Preparing training data...")
        logger.info("Starting training")

        # -- Disk-backed path: sequences already written to JSONL --
        # When data_path + data_offsets are provided (from streaming
        # pre-train pipeline), skip in-memory parsing entirely and
        # jump straight to the streaming training loop.
        if data_path is not None and data_offsets is not None:
            data_path = Path(data_path)
            if not data_path.exists():
                raise FileNotFoundError(f"data_path not found: {data_path}")
            n_sequences = len(data_offsets)
            if n_sequences == 0:
                raise ValueError("data_offsets is empty - no sequences to train on")
            logger.info("Disk-backed training: %d sequences from %s", n_sequences, data_path)
            self._dataset_fingerprint = hashlib.sha256(str(data_path).encode()).hexdigest()[:16]

            # Val split from disk offsets - train and val sequences
            # are in the same file, distinguished by offset lists.
            val_sequences: list[str] = []
            val_offsets: list[int] = []
            seq_offsets = list(data_offsets)
            _val_file: Path | None = None
            n_val_sequences = 0
            if self.config.val_split > 0 and n_sequences > 1:
                n_val = max(1, int(n_sequences * self.config.val_split))
                indices = list(range(n_sequences))
                random.Random(42).shuffle(indices)
                val_idx_set = set(indices[:n_val])
                val_offsets = [data_offsets[i] for i in sorted(val_idx_set)]
                seq_offsets = [data_offsets[i] for i in range(n_sequences) if i not in val_idx_set]
                n_val_sequences = len(val_offsets)
                n_sequences = len(seq_offsets)
                # Val reads from the same file, different offsets
                _val_file = data_path
                logger.info(
                    "Validation split: %d train, %d val (%.0f%%)",
                    n_sequences,
                    n_val_sequences,
                    self.config.val_split * 100,
                )

            # Before-training evaluation
            before_eval = None
            if self.config.run_evaluation:
                self._emit_progress(3, "Evaluating model (before training)...")
                try:
                    from enigma_engine.training.training_evaluation import evaluate_model, DEFAULT_TEST_PROMPTS

                    test_prompts = self.config.eval_test_prompts or DEFAULT_TEST_PROMPTS
                    device = next(self.model.parameters()).device
                    before_eval = evaluate_model(self.model, self.tokenizer, test_prompts, str(device))
                    logger.info(
                        "Before training: perplexity=%.2f, loss=%.4f", before_eval["perplexity"], before_eval["loss"]
                    )
                except Exception as exc:
                    logger.warning("Before-training evaluation failed: %s", exc)

            # Golden prompt eval
            golden_before = None
            if self.config.golden_eval_path:
                try:
                    from enigma_engine.training.training_evaluation import run_golden_eval

                    device = next(self.model.parameters()).device
                    golden_before = run_golden_eval(
                        self.model, self.tokenizer, self.config.golden_eval_path, str(device)
                    )
                    logger.info(
                        "Golden eval (before): %d/%d passed (%.0f%%)",
                        golden_before["passed"],
                        golden_before["total"],
                        golden_before["pass_rate"] * 100,
                    )
                except Exception as exc:
                    logger.warning("Golden eval (before) failed: %s", exc)

            cfg = getattr(self.model, "config", None)
            max_seq_len = getattr(cfg, "max_seq_len", 512)

            use_streaming = True
            _seq_file = data_path
            est_batches_per_epoch = max(1, n_sequences // self.config.batch_size)

            # LR finder on sample window
            if self.config.auto_lr:
                self._emit_progress(6, "Running LR range test...")
                # Cap to 200 sequences - LR finder only needs ~100 batches
                lr_cap = 100 * max(1, self.config.batch_size)
                sample_n = min(lr_cap, n_sequences)
                sample_indices = list(range(sample_n))
                sample_seqs = self._read_sequences_from_disk(_seq_file, seq_offsets, sample_indices)
                sample_batches = list(self._create_batches(sample_seqs, max_length=max_seq_len))
                del sample_seqs
                found_lr = self._lr_find(sample_batches)
                del sample_batches
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.config.learning_rate = found_lr
                for pg in self.optimizer.param_groups:
                    pg["lr"] = found_lr
                self._emit_progress(6, f"LR finder: using {found_lr:.2e}")

            total_steps = est_batches_per_epoch * self.config.epochs
            batches = None
            val_batches: list[tuple[torch.Tensor, torch.Tensor]] = []

            checkpoint_dir = Path(self.config.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            # Skip to scheduler setup (shared with normal path)
            # - jumps past data parsing, dedup, curriculum, etc.
            _disk_backed = True

        else:
            _disk_backed = False
            if data is None:
                raise ValueError("Either data or data_path must be provided")

        if not _disk_backed:
            # Validate training data quality (DQ-B) before investing
            # compute in tokenization and batching.
            # For large pre-chunked lists (e.g. pre-training corpora),
            # sample instead of joining everything into one huge string
            # which would freeze the GUI via GIL starvation.
            _LARGE_LIST_THRESHOLD = 10_000
            if isinstance(data, list) and len(data) > _LARGE_LIST_THRESHOLD:
                # Sample head + middle + tail for validation
                mid = len(data) // 2
                sample = data[:500] + data[mid : mid + 500] + data[-500:]
                data_str = "\n".join(str(s) for s in sample)
                # Incremental fingerprint from metadata + sample
                _h = hashlib.sha256()
                _h.update(f"n={len(data)}".encode())
                for _item in data[:10]:
                    _h.update(str(_item).encode("utf-8"))
                for _item in data[-10:]:
                    _h.update(str(_item).encode("utf-8"))
                fingerprint = _h.hexdigest()[:16]
            elif isinstance(data, str):
                data_str = data
                fingerprint = dataset_fingerprint(data_str)
            elif isinstance(data, list):
                data_str = "\n".join(str(item) for item in data)
                fingerprint = dataset_fingerprint(data_str)
            else:
                data_str = json.dumps(data)
                fingerprint = dataset_fingerprint(data_str)

            dq_result = validate_training_data(data_str)
            for w in dq_result.warnings:
                logger.warning("Data quality: %s", w)
            for e in dq_result.errors:
                logger.error("Data quality: %s", e)
            if not dq_result.is_valid:
                raise ValueError("Training data failed validation: " + "; ".join(dq_result.errors))

            # Dataset fingerprint for reproducibility tracking
            logger.info("Dataset fingerprint: %s", fingerprint)
            self._dataset_fingerprint = fingerprint

            # Parse data
            self._emit_progress(1, "Parsing training data...")
            try:
                sequences = self._parse_training_data(data)
                logger.info(f"Parsed {len(sequences)} training sequences")
                self._emit_progress(1, f"Parsed {len(sequences):,} training sequences")
            except Exception as e:
                logger.error(f"Failed to parse training data: {e}")
                raise
            time.sleep(0)  # Yield GIL after heavy tokenization

            if not sequences:
                raise ValueError("No training sequences found in data")

            # Mix general knowledge data to prevent catastrophic forgetting.
            # When focused data is provided, keep the model general by mixing
            # a portion of general/diverse examples into each training batch.
            if self.config.general_data and self.config.general_mix_ratio > 0:
                try:
                    general_text = self.config.general_data
                    # If it looks like a file path, load it
                    gp = Path(general_text)
                    if gp.exists() and gp.is_file():
                        general_text = gp.read_text(encoding="utf-8")
                    general_seqs = self._parse_training_data(general_text)
                    if general_seqs:
                        ratio = min(self.config.general_mix_ratio, 0.9)
                        n_focused = len(sequences)
                        n_general = max(1, int(n_focused * ratio / max(0.01, 1.0 - ratio)))
                        if len(general_seqs) >= n_general:
                            mixed = random.sample(general_seqs, n_general)
                        else:
                            mixed = (general_seqs * (n_general // len(general_seqs) + 1))[:n_general]
                        sequences.extend(mixed)
                        random.shuffle(sequences)
                        logger.info(
                            f"Mixed {n_general} general sequences "
                            f"with {n_focused} focused sequences "
                            f"(ratio {ratio:.0%} general)"
                        )
                except Exception as exc:
                    logger.warning(f"Could not mix general data: {exc}")

            # Deduplicate while preserving order — prevents the model
            # from over-fitting on repeated examples.
            pre_dedup = len(sequences)
            self._emit_progress(2, f"Deduplicating {len(sequences):,} sequences...")
            sequences = list(dict.fromkeys(sequences))
            if len(sequences) < pre_dedup:
                msg = f"Removed {pre_dedup - len(sequences)} duplicates ({len(sequences)} unique remain)"
                logger.info(msg)
                self._emit_progress(2, msg)

            # T4-1: Near-duplicate detection via MinHash.
            # Removes sequences with Jaccard similarity > 0.75 (keeps longest).
            # 0.75 matches FineWeb / DCLM / SmolLM3 industry standard
            # (FineWeb tech report arxiv:2406.17557 section 3.4).
            # Skip for large pre-training corpora: the O(n^2)
            # pairwise loop + shingle sets scale with RAM.
            _MINHASH_LIMIT = self._budget.minhash_limit
            if 1 < len(sequences) <= _MINHASH_LIMIT:
                self._emit_progress(2, f"MinHash dedup on {len(sequences):,} sequences...")
                pre_minhash = len(sequences)
                sequences = minhash_dedup(sequences)
                n_removed = pre_minhash - len(sequences)
                if n_removed > 0:
                    msg = f"MinHash: removed {n_removed} near-duplicates ({len(sequences)} remain)"
                    logger.info(msg)
                    self._emit_progress(2, msg)
            elif len(sequences) > _MINHASH_LIMIT:
                logger.info(
                    "Skipping MinHash dedup: %d sequences exceeds %d limit (exact dedup already applied)",
                    len(sequences),
                    _MINHASH_LIMIT,
                )
            time.sleep(0)  # Yield GIL after dedup

            # Split train/validation when val_split > 0
            val_sequences: list[str] = []
            if self.config.val_split > 0 and len(sequences) > 1:
                n_val = max(1, int(len(sequences) * self.config.val_split))
                # Random stratified split - avoids bias from data ordering
                indices = list(range(len(sequences)))
                random.Random(42).shuffle(indices)  # deterministic seed for reproducibility
                val_indices = set(indices[:n_val])
                val_sequences = [sequences[i] for i in sorted(val_indices)]
                sequences = [sequences[i] for i in range(len(sequences)) if i not in val_indices]
                logger.info(
                    f"Validation split: {len(sequences)} train, {len(val_sequences)} val ({self.config.val_split:.0%})"
                )

            # Run before-training evaluation (EV-C)
            before_eval = None
            if self.config.run_evaluation:
                self._emit_progress(3, "Evaluating model (before training)...")
                try:
                    from enigma_engine.training.training_evaluation import (
                        evaluate_model,
                        DEFAULT_TEST_PROMPTS,
                    )

                    test_prompts = self.config.eval_test_prompts or DEFAULT_TEST_PROMPTS
                    device = next(self.model.parameters()).device
                    before_eval = evaluate_model(self.model, self.tokenizer, test_prompts, str(device))
                    logger.info(
                        f"Before training: perplexity={before_eval['perplexity']:.2f}, loss={before_eval['loss']:.4f}"
                    )
                except Exception as exc:
                    logger.warning(f"Before-training evaluation failed: {exc}")

            # Golden prompt regression eval (before training)
            golden_before = None
            if self.config.golden_eval_path:
                try:
                    from enigma_engine.training.training_evaluation import (
                        run_golden_eval,
                    )

                    device = next(self.model.parameters()).device
                    golden_before = run_golden_eval(
                        self.model, self.tokenizer, self.config.golden_eval_path, str(device)
                    )
                    logger.info(
                        "Golden eval (before): %d/%d passed (%.0f%%)",
                        golden_before["passed"],
                        golden_before["total"],
                        golden_before["pass_rate"] * 100,
                    )
                except Exception as exc:
                    logger.warning("Golden eval (before) failed: %s", exc)

            # T4-2: Curriculum learning - sort sequences by difficulty.
            # "easy_first" scores each sequence by model loss, then sorts
            # so the model learns fundamentals before tackling hard examples.
            # Skip for large datasets: scoring many sequences = many forward
            # passes.  Limit scales with available VRAM/RAM.
            _CURRICULUM_LIMIT = self._budget.curriculum_limit
            if self.config.curriculum == "easy_first" and 1 < len(sequences) <= _CURRICULUM_LIMIT:
                self._emit_progress(4, "Scoring sequences for curriculum...")
                cfg = getattr(self.model, "config", None)
                _max_len = getattr(cfg, "max_seq_len", 512)
                losses = self._score_sequences_by_loss(sequences, _max_len)
                # Sort by loss ascending (easy first)
                paired = sorted(zip(losses, sequences), key=lambda x: x[0])
                sequences = [s for _, s in paired]
                logger.info(
                    "Curriculum: sorted %d sequences by loss (easy first, range %.2f-%.2f)",
                    len(sequences),
                    paired[0][0],
                    paired[-1][0],
                )
            elif self.config.curriculum == "easy_first" and len(sequences) > _CURRICULUM_LIMIT:
                logger.info(
                    "Skipping curriculum sorting: %d sequences exceeds %d limit", len(sequences), _CURRICULUM_LIMIT
                )

            time.sleep(0)  # Yield GIL after evaluation/curriculum
            self._emit_progress(5, f"Creating batches from {len(sequences)} sequences...")

            # Use model's max_seq_len for batch creation
            cfg = getattr(self.model, "config", None)
            max_seq_len = getattr(cfg, "max_seq_len", 512)

            # I-6: Auto BPE-dropout for multi-epoch small-data training.
            # When data is limited, multiple epochs risk memorisation.
            # BPE-dropout randomises tokenisation each epoch for free
            # data augmentation.  Auto-enable when:
            #   1) epochs > 3 (multiple passes make memorisation likely)
            #   2) tokens/param < 5 (data-constrained regime)
            #   3) bpe_dropout is currently 0 (user hasn't set it)
            if self.config.bpe_dropout == 0.0 and self.config.epochs > 3:
                # Rough token estimate: avg 4 chars/token - total chars
                total_chars = sum(len(s) for s in sequences)
                est_tokens = total_chars / 4
                total_params = sum(p.numel() for p in self.model.parameters())
                tokens_per_param = est_tokens / max(1, total_params)
                if tokens_per_param < 5.0:
                    self.config.bpe_dropout = 0.1
                    logger.info(
                        "Auto BPE-dropout enabled (0.1): %d epochs, %.1f tokens/param < 5.0",
                        self.config.epochs,
                        tokens_per_param,
                    )

            # -- Streaming vs eager batch creation --
            # For large datasets (>50K sequences), writing sequences to
            # disk and streaming batches in windows prevents OOM.  The
            # raw strings are freed after writing; only byte-offset
            # indices (ints) stay in RAM.
            use_streaming = len(sequences) > self._budget.streaming_threshold
            _seq_file: Path | None = None  # temp file for cleanup
            _val_file: Path | None = None

            checkpoint_dir = Path(self.config.checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            if use_streaming:
                n_sequences = len(sequences)
                _seq_file = checkpoint_dir / "_train_sequences.jsonl"
                logger.info("Streaming mode: writing %d sequences to disk...", n_sequences)
                self._emit_progress(5, f"Writing {n_sequences:,} sequences to disk...")
                seq_offsets = self._write_sequences_to_disk(sequences, _seq_file)
                del sequences  # Free the strings from RAM
                time.sleep(0)

                est_batches_per_epoch = max(1, n_sequences // self.config.batch_size)
                logger.info(
                    "Streaming: %d sequences on disk, ~%d batches/epoch (window=%d)",
                    n_sequences,
                    est_batches_per_epoch,
                    self._budget.streaming_window,
                )

                # Val sequences to disk
                n_val_sequences = 0
                val_offsets: list[int] = []
                if val_sequences:
                    n_val_sequences = len(val_sequences)
                    _val_file = checkpoint_dir / "_val_sequences.jsonl"
                    val_offsets = self._write_sequences_to_disk(val_sequences, _val_file)
                    del val_sequences
                    logger.info("Streaming: %d val sequences on disk", n_val_sequences)

                # LR finder: build sample batches from first window
                if self.config.auto_lr:
                    self._emit_progress(6, "Running LR range test...")
                    # Cap to 200 sequences - LR finder only needs ~100 batches
                    lr_cap = 100 * max(1, self.config.batch_size)
                    sample_n = min(lr_cap, n_sequences)
                    sample_indices = list(range(sample_n))
                    sample_seqs = self._read_sequences_from_disk(_seq_file, seq_offsets, sample_indices)
                    sample_batches = list(self._create_batches(sample_seqs, max_length=max_seq_len))
                    del sample_seqs
                    found_lr = self._lr_find(sample_batches)
                    del sample_batches
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.config.learning_rate = found_lr
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = found_lr
                    self._emit_progress(6, f"LR finder: using {found_lr:.2e}")

                total_steps = est_batches_per_epoch * self.config.epochs
                batches = None  # Not used in streaming mode
                _retokenize_per_epoch = False  # streaming re-tokenizes per epoch already
                val_batches: list[tuple[torch.Tensor, torch.Tensor]] = []

            else:
                # -- Eager path (small datasets) --
                n_sequences = len(sequences)
                seq_offsets = []
                n_val_sequences = 0
                val_offsets = []

                try:
                    batches = list(self._create_batches(sequences, max_length=max_seq_len))
                    logger.info(f"Created {len(batches)} training batches")
                    self._emit_progress(5, f"Created {len(batches):,} batches (batch_size={self.config.batch_size})")
                except Exception as e:
                    logger.error(f"Failed to create batches: {e}")
                    raise

                # BPE-dropout only augments when each epoch sees a fresh
                # tokenisation; a batch list built once would freeze a single
                # noised tokenisation for the whole run, so the epoch loop
                # rebuilds from the raw sequences when dropout is active.
                import inspect as _inspect

                _retokenize_per_epoch = (
                    self.config.bpe_dropout > 0
                    and self.config.epochs > 1
                    and "dropout" in _inspect.signature(self.tokenizer.encode).parameters
                )

                val_batches = []
                if val_sequences:
                    try:
                        val_batches = list(self._create_batches(val_sequences, max_length=max_seq_len))
                        logger.info(f"Created {len(val_batches)} validation batches")
                    except Exception as exc:
                        logger.warning(f"Failed to create val batches: {exc}")

                if self.config.auto_lr:
                    self._emit_progress(6, "Running LR range test...")
                    found_lr = self._lr_find(batches)
                    self.config.learning_rate = found_lr
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = found_lr
                    self._emit_progress(6, f"LR finder: using {found_lr:.2e}")

                total_steps = len(batches) * self.config.epochs
                est_batches_per_epoch = len(batches)

        # Setup scheduler
        self._total_training_steps = total_steps

        # total_steps counts MICRO-batches (state.step's unit), but
        # scheduler.step() fires once per OPTIMIZER step — one per
        # max_grad_accumulation micro-batches (plus one per epoch-end
        # remainder flush). Size the schedule in optimizer steps or with
        # accum>1 the run ends with the LR stuck near peak.
        accum = max(1, self.config.max_grad_accumulation)
        steps_per_epoch = -(-est_batches_per_epoch // accum)  # ceil div
        sched_total = max(1, steps_per_epoch * self.config.epochs)
        # A mid-epoch (save_every_steps) checkpoint resumes at the last
        # COMPLETED epoch, so the interrupted epoch's already-scheduled steps
        # run again on top of the restored scheduler state. Stretch the
        # schedule by that surplus: past T_max CosineAnnealingLR turns back
        # UP toward peak and the WSD decay walks below its floor.
        # state.step counts MICRO-batches while the scheduler steps once per
        # OPTIMIZER step, so the surplus is computed in micro-batches against
        # the micro-batch estimate and converted with the same ceil division.
        replayed_micro = self.state.step - self.state.epoch * est_batches_per_epoch
        replayed_steps = max(0, -(-replayed_micro // accum))
        sched_total += replayed_steps
        warmup = _effective_warmup(self.config.warmup_steps, sched_total)
        decay_steps = max(1, sched_total - warmup)

        warmup_scheduler = LambdaLR(self.optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / warmup))

        sched_type = self.config.schedule_type
        if sched_type == "wsd":
            # T2-6: Warmup-Stable-Decay.  Three phases:
            # 1) warmup (handled by warmup_scheduler above)
            # 2) stable at peak LR
            # 3) linear decay in the last wsd_decay_fraction of steps
            wsd_frac = self.config.wsd_decay_fraction
            min_lr_ratio = self.config.min_lr_ratio
            _decay_steps = decay_steps  # capture for lambda
            _wsd_frac = wsd_frac

            def _wsd_lambda(step: int) -> float:
                # step is relative to post-warmup
                decay_start = int(_decay_steps * (1.0 - _wsd_frac))
                if step < decay_start:
                    return 1.0  # stable phase
                # linear decay, floored at min_lr_ratio so steps beyond the
                # planned total hold the final LR instead of going negative
                progress = (step - decay_start) / max(1, _decay_steps - decay_start)
                return max(min_lr_ratio, 1.0 - (1.0 - min_lr_ratio) * progress)

            post_warmup_scheduler = LambdaLR(self.optimizer, lr_lambda=_wsd_lambda)
        else:
            # Cosine warm restarts vs single cosine decay (R8)
            restart_t0 = self.config.cosine_restart_period
            if restart_t0 > 0:
                post_warmup_scheduler = CosineAnnealingWarmRestarts(
                    self.optimizer, T_0=restart_t0, eta_min=(self.config.learning_rate * self.config.min_lr_ratio)
                )
            else:
                post_warmup_scheduler = CosineAnnealingLR(
                    self.optimizer, T_max=decay_steps, eta_min=(self.config.learning_rate * self.config.min_lr_ratio)
                )

        self.scheduler = SequentialLR(
            self.optimizer, schedulers=[warmup_scheduler, post_warmup_scheduler], milestones=[warmup]
        )

        # Restore scheduler/scaler state from checkpoint if available
        self._restore_pending_state()

        # Training loop

        # Dump config + fingerprint for reproducibility
        try:
            from enigma_engine.core.safe_save import atomic_write_json

            run_info = {
                "training_config": self.config.to_dict(),
                "dataset_fingerprint": self._dataset_fingerprint,
                "start_epoch": self.state.epoch,
            }
            model_cfg = getattr(self.model, "config", None)
            if model_cfg is not None:
                run_info["model_config"] = model_cfg.__dict__
            atomic_write_json(checkpoint_dir / "run_config.json", run_info)
        except Exception:
            logger.debug("Could not save run config", exc_info=True)

        # When resuming, start from the saved epoch
        start_epoch = self.state.epoch

        # Snapshot one weight tensor for sanity checking after warmup.
        # Pick a dense weight (not embeddings, which are sparse).
        # We delay the check until after warmup steps so the learning
        # rate is fully ramped and weight updates are meaningful.
        _weight_check_done = False
        _weight_check_step = min(warmup, est_batches_per_epoch)  # check after warmup
        _weight_snapshot: torch.Tensor | None = None
        _weight_ref_name = ""
        for _name, _p in self.model.named_parameters():
            if (
                _p.requires_grad
                and _p.ndim >= 2
                and "embed" not in _name
                and "output" not in _name
                and "head" not in _name
            ):
                _weight_snapshot = _p.data.clone()
                _weight_ref_name = _name
                break

        def _cleanup_temp_files():
            """Remove streaming temp files if they exist.

            For disk-backed mode, the caller owns the data file and both
            _seq_file and _val_file alias it, so neither may be deleted.
            """
            files_to_clean = []
            if not _disk_backed:
                files_to_clean.extend([_val_file, _seq_file])
            for _f in files_to_clean:
                if _f is not None:
                    try:
                        _f.unlink(missing_ok=True)
                    except OSError:
                        pass

        for epoch in range(start_epoch, self.config.epochs):
            if self._should_stop():
                logger.info("Training stopped by user")
                break

            # Time-limit guard
            if (
                self.config.max_training_seconds > 0
                and time.monotonic() - self._training_start_time > self.config.max_training_seconds
            ):
                logger.info("Training stopped: time limit reached")
                break

            epoch_loss = 0.0
            epoch_tokens = 0
            n_epoch_batches = 0

            progress_base = int(5 + (epoch / self.config.epochs) * 90)
            self._emit_progress(progress_base, f"Epoch {epoch + 1}/{self.config.epochs}")

            # Build the batch iterator for this epoch
            if use_streaming:
                # Streaming: shuffle sequence indices, read from disk
                # in windows, tokenize + pack per window, yield batches
                indices = list(range(n_sequences))
                random.shuffle(indices)
                epoch_batches = self._stream_batches(_seq_file, seq_offsets, indices, max_seq_len)
            else:
                if _retokenize_per_epoch and epoch > start_epoch:
                    # Fresh BPE-dropout tokenisation for this epoch (see the
                    # eager-path setup above).
                    batches = list(self._create_batches(sequences, max_length=max_seq_len))
                random.shuffle(batches)
                epoch_batches = batches

            for batch_idx, batch in enumerate(epoch_batches):
                if self._should_stop():
                    break

                _step_start = time.monotonic()
                try:
                    batch_loss = self._train_one_batch(batch, batch_idx)
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower():
                        raise
                    # OOM recovery: clear cache, enable gradient
                    # checkpointing, and retry the batch once
                    self._handle_oom(exc)
                    try:
                        batch_loss = self._train_one_batch(batch, batch_idx)
                    except RuntimeError as exc2:
                        if "out of memory" not in str(exc2).lower():
                            raise
                        logger.error(
                            "Training aborted: OOM persists after "
                            "enabling gradient checkpointing. "
                            "Reduce batch_size or max_seq_len."
                        )
                        self._emit_progress(0, "OOM: reduce batch size or sequence length")
                        self.state.abort_reason = (
                            "OOM persists after enabling gradient checkpointing. Reduce batch_size or max_seq_len."
                        )
                        self.model.eval()
                        _cleanup_temp_files()
                        return self.state
                _step_time = time.monotonic() - _step_start

                # Loss explosion guard
                if math.isnan(batch_loss) or math.isinf(batch_loss):
                    logger.error("Training aborted: NaN/Inf loss detected")
                    self.state.abort_reason = "NaN/Inf loss detected"
                    self.model.eval()
                    _cleanup_temp_files()
                    return self.state
                if batch_loss > self.config.max_loss:
                    logger.error(f"Training aborted: loss {batch_loss:.4f} exceeded max_loss {self.config.max_loss}")
                    self.state.abort_reason = f"Loss {batch_loss:.4f} exceeded max_loss {self.config.max_loss}"
                    self.model.eval()
                    _cleanup_temp_files()
                    return self.state

                batch_tokens = batch[0].numel()
                non_pad = int((batch[0] != self.pad_token_id).sum().item())
                epoch_loss += batch_loss * non_pad
                epoch_tokens += non_pad
                self.state.step += 1
                self.state.total_tokens += batch_tokens

                # Throughput telemetry
                self._emit_throughput(batch_tokens, _step_time)

                # One-time sanity check: verify weights actually changed
                # after warmup completes (LR is near-zero during early
                # warmup, so checking sooner would produce false alarms).
                if (
                    not _weight_check_done
                    and _weight_snapshot is not None
                    and self.state.step >= _weight_check_step
                    and (batch_idx + 1) % self.config.max_grad_accumulation == 0
                ):
                    _weight_check_done = True
                    for _name, _p in self.model.named_parameters():
                        if _name == _weight_ref_name:
                            delta = (_weight_snapshot - _p.data).abs().max().item()
                            if delta == 0:
                                logger.error(
                                    "WARNING: Model weights unchanged after "
                                    "%d optimizer steps (%s). Training may "
                                    "not be effective.",
                                    self.state.step,
                                    _name,
                                )
                            else:
                                logger.info("Weight update verified: %s max_delta=%.2e", _name, delta)
                            break
                    del _weight_snapshot
                    _weight_snapshot = None

                # Log periodically
                if self.state.step % self.config.log_every == 0:
                    avg_loss = epoch_loss / max(1, epoch_tokens)
                    logger.debug(f"Step {self.state.step}: loss={avg_loss:.4f}")
                    self._emit_loss(avg_loss)

                # Update progress
                batch_progress = int(
                    progress_base + (batch_idx / max(1, est_batches_per_epoch)) * (90 / self.config.epochs)
                )
                self._emit_progress(batch_progress, f"Epoch {epoch + 1}: Batch {batch_idx + 1}/{est_batches_per_epoch}")

                # Step-based validation (eval_every > 0).
                # Critical for long single-epoch pre-training runs
                # where per-epoch validation would never fire until
                # the very end.
                if self.config.eval_every > 0 and self.state.step % self.config.eval_every == 0:
                    _step_val = None
                    if use_streaming and n_val_sequences > 0:
                        _sv_idx = list(range(n_val_sequences))
                        _sv_gen = self._stream_batches(_val_file, val_offsets, _sv_idx, max_seq_len)
                        _sv_batches = list(_sv_gen)
                        if _sv_batches:
                            _step_val = self._validate(_sv_batches)
                            del _sv_batches
                    elif val_batches:
                        _step_val = self._validate(val_batches)
                    if _step_val is not None:
                        logger.info("Step %d val_loss=%.4f", self.state.step, _step_val)
                        self._emit_loss(epoch_loss / max(1, epoch_tokens), val_loss=_step_val)

                # Step-based checkpoint save (save_every_steps > 0).
                # Critical for long single-epoch pre-training runs
                # where epoch-end saves would never fire until the
                # very end - potentially months of lost progress.
                if self.config.save_every_steps > 0 and self.state.step % self.config.save_every_steps == 0:
                    stem = self._checkpoint_stem
                    ckpt_path = checkpoint_dir / f"{stem}_step{self.state.step}.pt"
                    self._save_checkpoint(ckpt_path)
                    self._cleanup_periodic_checkpoints(checkpoint_dir, f"{stem}_step", keep=3)

                # Yield GIL so GUI/other threads stay responsive.
                # On fast GPUs the batch loop can starve the main
                # thread; sleep(0) releases our time-slice.
                time.sleep(0)
                n_epoch_batches = batch_idx + 1

            # End-of-epoch remainder flush (mirrors V-1 in train_vision): if
            # the epoch's micro-batch count doesn't divide the accumulation
            # window, the tail grads must step now — otherwise they bleed
            # into the next epoch's first window and are dropped at run end.
            if n_epoch_batches % self.config.max_grad_accumulation != 0 and not self._should_stop():
                if self.config.gradient_clip > 0:
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()
                if self.scheduler is not None:
                    self.scheduler.step()

            # Epoch complete
            avg_epoch_loss = epoch_loss / max(1, epoch_tokens)
            self.state.training_losses.append(avg_epoch_loss)
            self.state.epoch = epoch + 1

            logger.info(f"Epoch {epoch + 1} complete: loss={avg_epoch_loss:.4f}")

            # Run validation pass (if we have val data)
            val_loss = None
            if use_streaming and n_val_sequences > 0:
                # Stream val batches from disk too
                _val_indices = list(range(n_val_sequences))
                _val_gen = self._stream_batches(_val_file, val_offsets, _val_indices, max_seq_len)
                _vb = list(_val_gen)
                if _vb:
                    val_loss = self._validate(_vb)
                    del _vb
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    self.state.validation_losses.append(val_loss)
                    logger.info(f"Epoch {epoch + 1} val_loss={val_loss:.4f}")
            elif val_batches:
                val_loss = self._validate(val_batches)
                self.state.validation_losses.append(val_loss)
                logger.info(f"Epoch {epoch + 1} val_loss={val_loss:.4f}")

            if self.on_epoch_complete:
                try:
                    self.on_epoch_complete(epoch + 1, avg_epoch_loss)
                except Exception:
                    logger.debug("on_epoch_complete callback error", exc_info=True)

            # Save periodic checkpoint (0 = disabled)
            if self.config.save_every > 0 and (epoch + 1) % self.config.save_every == 0:
                stem = self._checkpoint_stem
                ckpt_num = (epoch + 1) // self.config.save_every
                ckpt_path = checkpoint_dir / f"{stem}_checkpoint{ckpt_num}.pt"
                self._save_checkpoint(ckpt_path)
                self._cleanup_periodic_checkpoints(checkpoint_dir, f"{stem}_checkpoint", keep=3)

            # Track best loss + early stopping
            # Prefer val_loss when available, fall back to train loss
            tracked_loss = val_loss if val_loss is not None else avg_epoch_loss
            if tracked_loss < self.state.best_loss:
                self.state.best_loss = tracked_loss
                self._epochs_without_improvement = 0
                self._save_checkpoint(checkpoint_dir / f"{self._checkpoint_stem}_best.pt")
                # Rolling best checkpoints (CK-C)
                if self.config.rolling_best_k > 0:
                    self._save_rolling_checkpoint(checkpoint_dir, tracked_loss, epoch + 1)
            else:
                self._epochs_without_improvement += 1
                if (
                    self.config.early_stopping_patience > 0
                    and self._epochs_without_improvement >= self.config.early_stopping_patience
                ):
                    logger.info(f"Early stopping: no improvement for {self._epochs_without_improvement} epochs")
                    break

        # Apply SWA averaged weights before final save (Izmailov et al.)
        # SWA weights are the whole point - they should be the saved model.
        if self.swa is not None and self.swa.n_averaged > 0:
            logger.info("Applying SWA averaged weights (%d snapshots) for final save", self.swa.n_averaged)
            self.swa.apply(self.model)

        # Final save
        self._emit_progress(95, "Saving final model...")
        self._save_checkpoint(checkpoint_dir / f"{self._checkpoint_stem}_final.pt")

        # Run after-training evaluation (EV-C)
        after_eval = None
        if self.config.run_evaluation:
            self._emit_progress(97, "Evaluating model (after training)...")
            try:
                from enigma_engine.training.training_evaluation import (
                    evaluate_model,
                    DEFAULT_TEST_PROMPTS,
                )

                test_prompts = self.config.eval_test_prompts or DEFAULT_TEST_PROMPTS
                device = next(self.model.parameters()).device
                after_eval = evaluate_model(self.model, self.tokenizer, test_prompts, str(device))
                logger.info(f"After training: perplexity={after_eval['perplexity']:.2f}, loss={after_eval['loss']:.4f}")
                if before_eval:
                    ppl_improvement = before_eval["perplexity"] - after_eval["perplexity"]
                    logger.info(
                        f"Perplexity improvement: {ppl_improvement:.2f} "
                        f"({ppl_improvement / before_eval['perplexity'] * 100:.1f}%)"
                    )
            except Exception as exc:
                logger.warning(f"After-training evaluation failed: {exc}")

        # Golden prompt regression eval (after training)
        golden_after = None
        if self.config.golden_eval_path:
            try:
                from enigma_engine.training.training_evaluation import (
                    run_golden_eval,
                )

                device = next(self.model.parameters()).device
                golden_after = run_golden_eval(self.model, self.tokenizer, self.config.golden_eval_path, str(device))
                logger.info(
                    "Golden eval (after): %d/%d passed (%.0f%%)",
                    golden_after["passed"],
                    golden_after["total"],
                    golden_after["pass_rate"] * 100,
                )
                if golden_before:
                    delta = golden_after["passed"] - golden_before["passed"]
                    logger.info("Golden eval delta: %+d cases", delta)
            except Exception as exc:
                logger.warning("Golden eval (after) failed: %s", exc)

        # Store evaluation results in state for later retrieval
        if before_eval:
            self.state.before_eval = before_eval  # type: ignore
        if after_eval:
            self.state.after_eval = after_eval  # type: ignore
        if golden_before:
            self.state.golden_before = golden_before  # type: ignore
        if golden_after:
            self.state.golden_after = golden_after  # type: ignore

        # Restore model to eval mode for inference
        self.model.eval()

        self._emit_progress(100, "Training complete!")
        logger.info(f"Training complete: {self.state.epoch} epochs, best_loss={self.state.best_loss:.4f}")

        _cleanup_temp_files()
        return self.state

    def _train_one_batch(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> float:
        """Run forward + backward on a single batch.

        Args:
            batch: Tuple of (token_ids, attention_mask).
            batch_idx: Index within the epoch (for grad accumulation gating).

        Returns the unscaled batch loss (float).  Raises RuntimeError
        on CUDA OOM so the caller can handle recovery.
        """
        batch_tensor, attention_mask = batch
        # Lazy device transfer: tensors stay on CPU until needed so
        # only one batch occupies VRAM at a time.
        batch_tensor = batch_tensor.to(self.device)
        attention_mask = attention_mask.to(self.device)
        is_packed = attention_mask.ndim == 4  # 4D = sequence packing

        # LISA: resample active layers each optimizer step (R26)
        if self.lisa is not None and batch_idx % self.config.max_grad_accumulation == 0:
            self.lisa.step()

        # Forward pass
        with torch.amp.autocast(
            "cuda",
            dtype=self._amp_dtype,
            enabled=self.config.use_amp and torch.cuda.is_available(),
        ):
            input_ids = batch_tensor[:, :-1]
            targets = batch_tensor[:, 1:]

            # T3-6: Use chunked CE when logits aren't needed downstream
            _needs_logits = (
                self.config.r_drop_alpha > 0 or self.config.reasoning_loss_weight > 1.0 or self.config.z_loss_weight > 0
            )
            _ce_chunk = 0 if _needs_logits else self.config.ce_chunk_size

            if is_packed:
                # 4D block-diagonal mask: slice both spatial dims
                attn_mask_2d = attention_mask[:, :, :-1, :-1]
                logits, loss = self.model(
                    input_ids,
                    targets=targets,
                    pad_token_id=self.pad_token_id,
                    label_smoothing=self.config.label_smoothing,
                    attention_mask_2d=attn_mask_2d,
                    chunked_ce=_ce_chunk,
                )
            else:
                # Standard 2D pad mask: slice last position
                attn_mask = attention_mask[:, :-1]
                logits, loss = self.model(
                    input_ids,
                    targets=targets,
                    pad_token_id=self.pad_token_id,
                    label_smoothing=self.config.label_smoothing,
                    attention_mask=attn_mask,
                    chunked_ce=_ce_chunk,
                )

            # T2-4: R-Drop regularization - second forward pass with
            # different dropout masks, plus KL-divergence penalty.
            if self.config.r_drop_alpha > 0 and loss is not None:
                if is_packed:
                    logits2, loss2 = self.model(
                        input_ids,
                        targets=targets,
                        pad_token_id=self.pad_token_id,
                        label_smoothing=self.config.label_smoothing,
                        attention_mask_2d=attn_mask_2d,
                    )
                else:
                    logits2, loss2 = self.model(
                        input_ids,
                        targets=targets,
                        pad_token_id=self.pad_token_id,
                        label_smoothing=self.config.label_smoothing,
                        attention_mask=attn_mask,
                    )
                # Average CE losses from both passes
                loss = (loss + loss2) / 2
                # Symmetric KL divergence
                p = torch.nn.functional.log_softmax(logits, dim=-1)
                q = torch.nn.functional.log_softmax(logits2, dim=-1)
                kl_pq = torch.nn.functional.kl_div(q, p, log_target=True, reduction="batchmean")
                kl_qp = torch.nn.functional.kl_div(p, q, log_target=True, reduction="batchmean")
                loss = loss + self.config.r_drop_alpha * (kl_pq + kl_qp) / 2

            # Reasoning-weighted loss (CoT-E)
            if self.config.reasoning_loss_weight > 1.0 and loss is not None:
                loss = self._apply_reasoning_weight(logits, targets, loss)

            # Z-Loss: penalizes large logits for stability (R28, PaLM)
            if self.config.z_loss_weight > 0 and loss is not None:
                z = torch.logsumexp(logits, dim=-1)
                z_loss = self.config.z_loss_weight * (z**2).mean()
                loss = loss + z_loss

            # MoE auxiliary loss
            if hasattr(self.model, "get_moe_aux_loss"):
                aux_loss = self.model.get_moe_aux_loss()
                loss = loss + aux_loss * 0.01

            # Scale for gradient accumulation
            loss = loss / self.config.max_grad_accumulation

        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient accumulation step
        if (batch_idx + 1) % self.config.max_grad_accumulation == 0:
            # Under fp16 AMP the accumulated grads are still multiplied by the
            # loss scale; unscale before touching them so noise injection and
            # clipping both operate on true-magnitude gradients. scaler.step()
            # skips its internal unscale when it already happened here.
            if self.scaler is not None and (self.config.gradient_noise_eta > 0 or self.config.gradient_clip > 0):
                self.scaler.unscale_(self.optimizer)

            # Gradient noise injection (Neelakantan et al.) with
            # warmup/decay envelope (T2-3).
            if self.config.gradient_noise_eta > 0:
                noise_std = self.config.gradient_noise_eta / (1.0 + self.state.step) ** self.config.gradient_noise_gamma
                # T2-3: Apply noise lifecycle envelope
                total = getattr(self, "_total_training_steps", 0)
                if total > 0:
                    frac = self.state.step / total
                    warmup_end = self.config.noise_warmup_fraction
                    decay_start = 1.0 - self.config.noise_decay_fraction
                    if frac < warmup_end and warmup_end > 0:
                        # Ramp up linearly
                        noise_std *= frac / warmup_end
                    elif frac > decay_start and self.config.noise_decay_fraction > 0:
                        # Anneal to zero linearly
                        noise_std *= (1.0 - frac) / self.config.noise_decay_fraction

                for p in self.model.parameters():
                    if p.grad is not None:
                        p.grad.add_(torch.randn_like(p.grad) * noise_std)

            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)

            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

            self.optimizer.zero_grad()

            # S560: AdEMAMix alpha annealing - decay alpha from initial
            # to final over the first warmup fraction of training.
            if self.config.optimizer == "ademamix" and self.config.ademamix_alpha_initial != self.config.ademamix_alpha:
                total = getattr(self, "_total_training_steps", 0)
                if total > 0:
                    warmup_steps = int(total * self.config.ademamix_alpha_warmup)
                    if warmup_steps > 0 and self.state.step <= warmup_steps:
                        frac = self.state.step / warmup_steps
                        alpha = self.config.ademamix_alpha_initial + frac * (
                            self.config.ademamix_alpha - self.config.ademamix_alpha_initial
                        )
                        for group in self.optimizer.param_groups:
                            group["alpha"] = alpha

            if self.scheduler is not None:
                self.scheduler.step()

            # EMA: update shadow weights after each optimizer step
            if self.ema is not None:
                self.ema.update(self.model)

            # SWA: collect weight snapshot at intervals (R13)
            # Also force a snapshot at cosine restart boundaries (T1-4)
            # where the model is at a local minimum - ideal for averaging.
            if self.swa is not None:
                step = self.state.step
                self.swa.update(self.model, step)
                restart_t0 = self.config.cosine_restart_period
                if restart_t0 > 0:
                    warmup = _effective_warmup(self.config.warmup_steps, self._total_training_steps)
                    step_in_cosine = step - warmup
                    if step_in_cosine > 0 and step_in_cosine % restart_t0 == 0 and step % self.swa.update_interval != 0:
                        # Restart boundary that wasn't already an
                        # interval boundary - force extra snapshot
                        self.swa.n_averaged += 1
                        for avg, param in zip(self.swa.avg, self.model.parameters()):
                            avg.add_((param.detach() - avg) / self.swa.n_averaged)
                        logger.debug(
                            "SWA restart snapshot at step %d (cosine cycle %d)", step, step_in_cosine // restart_t0
                        )

        return loss.item() * self.config.max_grad_accumulation

    @torch.no_grad()
    def _validate(self, val_batches: list[tuple[torch.Tensor, torch.Tensor]]) -> float:
        """Run a forward-only pass on validation batches.

        Returns the average validation loss (float).
        Uses EMA weights when available for more stable evaluation,
        or SWA weights when EMA is not enabled.
        """
        # Swap in averaged weights for evaluation:
        # EMA takes priority; fall back to SWA if EMA is not active.
        using_ema = self.ema is not None
        using_swa = self.swa is not None and not using_ema
        if using_ema:
            self.ema.apply(self.model)
        elif using_swa:
            self.swa.apply(self.model)

        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        try:
            for batch_tensor, attention_mask in val_batches:
                batch_tensor = batch_tensor.to(self.device)
                attention_mask = attention_mask.to(self.device)
                is_packed = attention_mask.ndim == 4
                with torch.amp.autocast(
                    "cuda",
                    dtype=self._amp_dtype,
                    enabled=self.config.use_amp and torch.cuda.is_available(),
                ):
                    input_ids = batch_tensor[:, :-1]
                    targets = batch_tensor[:, 1:]
                    if is_packed:
                        attn_mask_2d = attention_mask[:, :, :-1, :-1]
                        _logits, loss = self.model(
                            input_ids, targets=targets, pad_token_id=self.pad_token_id, attention_mask_2d=attn_mask_2d
                        )
                    else:
                        attn_mask = attention_mask[:, :-1]
                        _logits, loss = self.model(
                            input_ids, targets=targets, pad_token_id=self.pad_token_id, attention_mask=attn_mask
                        )

                if loss is not None:
                    non_pad = int((targets != self.pad_token_id).sum().item())
                    total_loss += loss.item() * non_pad
                    total_tokens += non_pad
        finally:
            self.model.train()
            # Restore live training weights (must be in finally to
            # prevent leaving model with EMA/SWA weights on exception)
            if using_ema:
                self.ema.restore(self.model)
            elif using_swa:
                self.swa.restore(self.model)

        return total_loss / max(1, total_tokens)

    def _handle_oom(self, exc: RuntimeError) -> None:
        """Handle CUDA OOM: clear cache, enable gradient checkpointing.

        Called on the first OOM per training run.  Enables gradient
        checkpointing (trades compute for VRAM) and clears the CUDA
        cache so the retry has the best chance of succeeding.
        """
        logger.warning("CUDA out of memory — clearing cache and enabling gradient checkpointing for retry: %s", exc)
        self.optimizer.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Enable gradient checkpointing if not already on. Models with shared
        # K/V cannot take the checkpointed path (the layer forward refuses the
        # combination), so the retry for them runs with cache cleared only.
        if not self.config.use_gradient_checkpointing:
            if getattr(getattr(self.model, "config", None), "kv_share_groups", 0) > 0:
                logger.warning(
                    "Not enabling gradient checkpointing for the retry: kv_share_groups is "
                    "active and checkpointed backward cannot rebuild shared K/V"
                )
            else:
                self.config.use_gradient_checkpointing = True
                if hasattr(self.model, "gradient_checkpointing_enable"):
                    self.model.gradient_checkpointing_enable()
                else:
                    for layer in getattr(self.model, "layers", []):
                        if hasattr(layer, "use_checkpoint"):
                            layer.use_checkpoint = True
                logger.info("Gradient checkpointing enabled to reduce VRAM usage")

    def _restore_pending_state(self) -> None:
        """Restore scheduler and scaler state stashed by load_checkpoint."""
        sched_state = getattr(self, "_pending_scheduler_state", None)
        if sched_state is not None and self.scheduler is not None:
            try:
                self.scheduler.load_state_dict(sched_state)
                logger.info("Restored scheduler state from checkpoint")
            except Exception as exc:
                logger.warning("Could not restore scheduler state: %s", exc)
            self._pending_scheduler_state = None

        scaler_state = getattr(self, "_pending_scaler_state", None)
        if scaler_state is not None and self.scaler is not None:
            try:
                self.scaler.load_state_dict(scaler_state)
                logger.info("Restored scaler state from checkpoint")
            except Exception as exc:
                logger.warning("Could not restore scaler state: %s", exc)
            self._pending_scaler_state = None

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

    def _save_checkpoint(self, path: Path) -> None:
        """Save model checkpoint with full training state.

        Saves model weights, optimizer, scheduler, scaler, and step
        counters so training can resume exactly where it left off.
        """
        try:
            # torch.compile wraps the model in OptimizedModule, which prefixes
            # every state-dict key with '_orig_mod.'. Save the RAW module so
            # checkpoints stay loadable by serve/gui/uncompiled trainers.
            _raw_model = getattr(self.model, "_orig_mod", self.model)
            checkpoint = {
                "model_state_dict": _raw_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "training_state": {
                    "epoch": self.state.epoch,
                    "step": self.state.step,
                    "best_loss": self.state.best_loss,
                    "total_tokens": self.state.total_tokens,
                    "training_losses": self.state.training_losses,
                    "validation_losses": self.state.validation_losses,
                    "dataset_fingerprint": getattr(self, "_dataset_fingerprint", None),
                },
                "training_config": self.config.to_dict(),
            }
            # Save scheduler state for exact resume
            if self.scheduler is not None:
                checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
            # Save AMP scaler state
            if self.scaler is not None:
                checkpoint["scaler_state_dict"] = self.scaler.state_dict()

            # Save EMA state for resume
            if self.ema is not None:
                checkpoint["ema_state_dict"] = self.ema.state_dict()

            # Save SWA state for resume
            if self.swa is not None:
                checkpoint["swa_state_dict"] = self.swa.state_dict()

            if hasattr(self.model, "config"):
                checkpoint["model_config"] = self.model.config.__dict__
                # Also save as 'config' for compatibility with gui_forge loader
                checkpoint["config"] = self.model.config.__dict__

            from enigma_engine.core.safe_save import atomic_torch_save

            atomic_torch_save(checkpoint, path)
            logger.info(f"Saved checkpoint: {path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            self._emit_warning(f"Checkpoint save failed: {e}")

    def _save_rolling_checkpoint(self, checkpoint_dir: Path, loss: float, epoch: int) -> None:
        """Save a rolling best checkpoint (CK-C).

        Keeps only the K best checkpoints by loss.  When a new
        checkpoint is better than the worst in the rolling set,
        the worst is deleted.  This prevents disk bloat during
        long training runs.

        Args:
            checkpoint_dir: Directory where checkpoints are saved.
            loss: The loss value for this checkpoint.
            epoch: The epoch number for filename.
        """
        k = self.config.rolling_best_k
        if k <= 0:
            return

        path = checkpoint_dir / (f"{self._checkpoint_stem}_rolling_e{epoch}_loss{loss:.4f}.pt")

        # If we haven't filled K slots yet, just save
        if len(self._rolling_checkpoints) < k:
            self._save_checkpoint(path)
            self._rolling_checkpoints.append((loss, path))
            self._rolling_checkpoints.sort(key=lambda x: x[0])
            logger.info("Rolling checkpoint saved: %s (%d/%d slots)", path.name, len(self._rolling_checkpoints), k)
            return

        # Check if this is better than the worst
        worst_loss, worst_path = self._rolling_checkpoints[-1]
        if loss < worst_loss:
            # Delete the worst checkpoint
            try:
                if worst_path.exists():
                    worst_path.unlink()
                    logger.info("Rolling checkpoint deleted (worst): %s", worst_path.name)
            except OSError as exc:
                logger.debug("Could not delete old rolling checkpoint: %s", exc)

            # Remove worst from tracking and add new
            self._rolling_checkpoints.pop()
            self._save_checkpoint(path)
            self._rolling_checkpoints.append((loss, path))
            self._rolling_checkpoints.sort(key=lambda x: x[0])
            logger.info("Rolling checkpoint replaced: %s (loss=%.4f)", path.name, loss)

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

    @staticmethod
    def _find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
        """Find the most recent training checkpoint in a directory.

        Looks for ``{stem}_checkpoint{N}.pt`` files (new format) and
        ``checkpoint_epoch_N.pt`` (legacy) and returns the one with
        the highest number.  Falls back to ``{stem}_best.pt``,
        ``{stem}_final.pt``, or legacy ``best_model.pt`` /
        ``final_model.pt``.

        Returns ``None`` if no usable checkpoints are found or the
        directory is missing.
        """
        import re

        if not checkpoint_dir.is_dir():
            return None

        stem = checkpoint_dir.name
        if stem == "checkpoints":
            stem = "model"

        # Match new format: {stem}_checkpoint{N}.pt
        new_pat = re.compile(rf"^{re.escape(stem)}_checkpoint(\d+)\.pt$")
        # Match legacy format: checkpoint_epoch_{N}.pt
        old_pat = re.compile(r"^checkpoint_epoch_(\d+)\.pt$")

        found: list[tuple[int, Path]] = []
        for f in checkpoint_dir.iterdir():
            m = new_pat.match(f.name) or old_pat.match(f.name)
            if m:
                found.append((int(m.group(1)), f))

        if found:
            found.sort(key=lambda x: x[0])
            return found[-1][1]

        # No periodic checkpoints - try best/final (new then legacy)
        for name in (
            f"{stem}_best.pt",
            f"{stem}_final.pt",
            "best_model.pt",
            "final_model.pt",
        ):
            candidate = checkpoint_dir / name
            if candidate.exists():
                return candidate

        return None

    def load_checkpoint(self, path: Path) -> None:
        """Load model checkpoint."""
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
            self.state.best_loss = state.get("best_loss", float("inf"))
            self.state.total_tokens = state.get("total_tokens", 0)
            self.state.training_losses = state.get("training_losses", [])
            self.state.validation_losses = state.get("validation_losses", [])

            # Stash scheduler/scaler state for deferred restore
            self._pending_scheduler_state = checkpoint.get("scheduler_state_dict")
            self._pending_scaler_state = checkpoint.get("scaler_state_dict")

            # Restore EMA state if saved and EMA is active
            ema_state = checkpoint.get("ema_state_dict")
            if ema_state is not None and self.ema is not None:
                self.ema.load_state_dict(ema_state)

            # Restore SWA state if saved and SWA is active
            swa_state = checkpoint.get("swa_state_dict")
            if swa_state is not None and self.swa is not None:
                self.swa.load_state_dict(swa_state)

            logger.info(f"Loaded checkpoint: {path}")

            # Mark this checkpoint as protected so auto-cleanup
            # won't delete it
            keep_marker = path.parent / (path.name + ".keep")
            if not keep_marker.exists():
                try:
                    keep_marker.write_text("protected - resumed for training", encoding="utf-8")
                except OSError:
                    pass
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

    # -----------------------------------------------------------------
    # DPO — Direct Preference Optimization
    # -----------------------------------------------------------------

    @staticmethod
    def _dpo_loss(
        policy_chosen_logps: "torch.Tensor",
        policy_rejected_logps: "torch.Tensor",
        ref_chosen_logps: "torch.Tensor",
        ref_rejected_logps: "torch.Tensor",
        beta: float = 0.1,
    ) -> "torch.Tensor":
        """Compute the DPO loss (Rafailov et al., 2023).

        loss = -log(sigma(beta * (log pi(y_w|x)/pi_ref(y_w|x)
                                - log pi(y_l|x)/pi_ref(y_l|x))))
        """
        import torch.nn.functional as F
        import torch

        chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
        rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
        loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
        if not torch.isfinite(loss):
            logger.warning("DPO loss is non-finite (%s), clamping to 0", loss.item())
            loss = torch.zeros_like(loss)
        return loss

    @staticmethod
    def _apo_zero_loss(
        policy_chosen_logps: "torch.Tensor",
        policy_rejected_logps: "torch.Tensor",
        ref_chosen_logps: "torch.Tensor",
        ref_rejected_logps: "torch.Tensor",
        beta: float = 0.1,
    ) -> "torch.Tensor":
        """Anchored Preference Optimization, zero variant - D-9 / Pass 156j.

        APO-zero (D'Oosterlinck et al., 2024) anchors *both sides
        independently* to the reference policy: chosen is pushed above
        ref, rejected is pushed below ref. Unlike DPO - which couples
        the two via the reward difference and can be satisfied by
        degrading the rejected response without improving chosen -
        APO-zero is satisfied only when the chosen actually rises above
        ref and the rejected actually falls below it.

        loss = sigmoid(-beta * (policy_chosen - ref_chosen))
             + sigmoid( beta * (policy_rejected - ref_rejected))

        Equivalent to TRL's `apo_zero` formulation:
            (1 - sigmoid(beta * chosen_logratio))
            + sigmoid(beta * rejected_logratio)
        """
        import torch

        chosen_logratio = policy_chosen_logps - ref_chosen_logps
        rejected_logratio = policy_rejected_logps - ref_rejected_logps
        chosen_term = torch.sigmoid(-beta * chosen_logratio)
        rejected_term = torch.sigmoid(beta * rejected_logratio)
        loss = (chosen_term + rejected_term).mean()
        if not torch.isfinite(loss):
            logger.warning(
                "APO-zero loss is non-finite (%s), clamping to 0", loss.item() if loss.numel() == 1 else "tensor"
            )
            loss = torch.zeros_like(loss)
        return loss

    @staticmethod
    def _resolve_preference_loss(loss_type: str):
        """Map a string preference-loss name to its static implementation.

        Accepted values: ``"dpo"`` (default), ``"apo_zero"``. Anything
        else raises ``ValueError`` so misspellings fail loud at the call
        site instead of silently falling back to DPO.
        """
        registry = {
            "dpo": Trainer._dpo_loss,
            "apo_zero": Trainer._apo_zero_loss,
        }
        if loss_type not in registry:
            raise ValueError(f"Unknown loss_type {loss_type!r}; must be one of {sorted(registry)}")
        return registry[loss_type]

    def _get_sequence_logps(
        self,
        model: "nn.Module",
        input_ids: "torch.Tensor",
        labels: "torch.Tensor",
        attention_mask: "torch.Tensor | None" = None,
    ) -> "torch.Tensor":
        """Compute per-sample summed log-probability of *labels*.

        Summed (not length-averaged): the DPO/APO/KTO losses downstream apply
        beta to log pi(y|x) ratios per Rafailov et al., and dpo_enigma.py's
        standalone loop scores sequences the same way, so beta values stay
        comparable across both paths and with reference implementations.

        Args:
            model: The model to evaluate.
            input_ids: (B, L) token ids.
            labels: (B, L) target token ids (-100 for ignored positions).
            attention_mask: (B, L) mask where 1=real, 0=pad.

        Returns:
            (B,) tensor of summed log-probabilities.
        """
        import torch.nn.functional as F

        attn_mask = attention_mask[:, :-1] if attention_mask is not None else None
        # Enigma.forward returns bare logits when targets is None
        out = model(input_ids[:, :-1], targets=None, attention_mask=attn_mask)
        logits = out[0] if isinstance(out, tuple) else out
        # logits: (B, L-1, V)
        log_probs = F.log_softmax(logits, dim=-1)
        targets = labels[:, 1:]  # shift right

        # Labels use -100 for ignored positions; gather() cannot index -100,
        # so clamp to a valid id and rely on the mask to zero those slots.
        mask = (targets != -100).float()
        per_token = log_probs.gather(2, targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        return (per_token * mask).sum(dim=-1)

    def _encode_dpo_pair(
        self,
        item: dict,
        pairs: list,
        max_len_dpo: int,
    ) -> bool:
        """Encode a single preference pair and append to *pairs*.

        Returns True if the pair was valid and appended, False otherwise.
        """
        import torch

        prompt = item.get("prompt", "")
        chosen = item.get("chosen", "")
        rejected = item.get("rejected", "")
        if not (prompt and chosen and rejected):
            return False

        prompt_ids = self.tokenizer.encode(f"User: {prompt}\nAssistant: ")
        chosen_ids = self.tokenizer.encode(f"User: {prompt}\nAssistant: {chosen}")
        rejected_ids = self.tokenizer.encode(f"User: {prompt}\nAssistant: {rejected}")

        chosen_ids = chosen_ids[:max_len_dpo]
        rejected_ids = rejected_ids[:max_len_dpo]

        prompt_len = len(prompt_ids)
        if len(chosen_ids) <= prompt_len or len(rejected_ids) <= prompt_len:
            return False

        chosen_labels = [-100] * min(prompt_len, len(chosen_ids)) + chosen_ids[prompt_len:]
        rejected_labels = [-100] * min(prompt_len, len(rejected_ids)) + rejected_ids[prompt_len:]

        chosen_mask = [1] * len(chosen_ids)
        rejected_mask = [1] * len(rejected_ids)

        max_len = max(len(chosen_ids), len(rejected_ids))
        chosen_mask += [0] * (max_len - len(chosen_mask))
        rejected_mask += [0] * (max_len - len(rejected_mask))
        chosen_ids += [0] * (max_len - len(chosen_ids))
        rejected_ids += [0] * (max_len - len(rejected_ids))
        chosen_labels += [-100] * (max_len - len(chosen_labels))
        rejected_labels += [-100] * (max_len - len(rejected_labels))

        pairs.append(
            (
                torch.tensor([chosen_ids, rejected_ids], dtype=torch.long, device=self.device),
                torch.tensor([chosen_labels], dtype=torch.long, device=self.device),
                torch.tensor([rejected_labels], dtype=torch.long, device=self.device),
                torch.tensor([chosen_mask, rejected_mask], dtype=torch.long, device=self.device),
            )
        )
        return True

    def _generate_online_dpo_pairs(
        self,
        prompts: list[str],
        reward_fn: "Callable[[str, str], float]",
        n_samples: int = 4,
        max_new_tokens: int = 128,
    ) -> list[dict]:
        """Generate fresh preference pairs via reward-scored sampling (T2-5).

        For each prompt, generates *n_samples* responses with different
        random truncations, scores them with *reward_fn*, and creates a
        preference pair from the best and worst scored responses.

        Returns a list of dicts with ``prompt``, ``chosen``, ``rejected``.
        """
        import torch

        self.model.eval()
        new_pairs: list[dict] = []
        eos_id = getattr(self.tokenizer, "eos_token_id", 2)
        try:
            for prompt_text in prompts:
                encoded = self.tokenizer.encode(f"User: {prompt_text}\nAssistant: ")
                input_ids = torch.tensor([encoded], dtype=torch.long, device=self.device)
                responses: list[tuple[str, float]] = []
                for _ in range(n_samples):
                    ids = input_ids.clone()
                    generated: list[int] = []
                    for _tok_idx in range(max_new_tokens):
                        with torch.no_grad():
                            out = self.model(ids, targets=None)
                        logits = out[0] if isinstance(out, tuple) else out
                        next_logit = logits[:, -1, :]
                        # Temperature sampling with T=0.8
                        probs = torch.softmax(next_logit / 0.8, dim=-1)
                        next_id = torch.multinomial(probs, 1).item()
                        generated.append(next_id)
                        if next_id == eos_id:
                            break
                        # No KV cache here: the model must see the full
                        # sequence each step, not just the newest token.
                        ids = torch.cat(
                            [ids, torch.tensor([[next_id]], dtype=torch.long, device=self.device)], dim=1
                        )
                    resp_text = self.tokenizer.decode(generated)
                    score = reward_fn(prompt_text, resp_text)
                    responses.append((resp_text, score))
                if len(responses) >= 2:
                    responses.sort(key=lambda x: x[1])
                    new_pairs.append(
                        {
                            "prompt": prompt_text,
                            "chosen": responses[-1][0],  # highest score
                            "rejected": responses[0][0],  # lowest score
                        }
                    )
        finally:
            self.model.train()
        return new_pairs

    def train_dpo(
        self,
        preference_data: list[dict],
        beta: float = 0.1,
        reward_fn: "Callable[[str, str], float] | None" = None,
        dpo_online_interval: int = 0,
        dpo_online_samples: int = 4,
        loss_type: str = "dpo",
    ) -> TrainingState:
        """Train with Direct Preference Optimization.

        Each item in *preference_data* must contain:
        - ``prompt``: the user prompt
        - ``chosen``: the preferred response
        - ``rejected``: the dis-preferred response

        A frozen copy of the model serves as the reference policy.

        Args:
            preference_data: List of preference pairs.
            beta: DPO temperature (lower = stronger preference signal).
            reward_fn: Optional callable ``(prompt, response) -> score``.
                When provided alongside a positive ``dpo_online_interval``,
                fresh responses are generated and scored periodically to
                create new preference pairs (T2-5 Online DPO).
            dpo_online_interval: Generate fresh pairs every N optimizer
                steps.  0 = disabled.
            dpo_online_samples: Number of responses to generate per prompt
                for online pair creation.
            loss_type: ``"dpo"`` (default) for the original Rafailov 2023
                loss, or ``"apo_zero"`` for the Anchored Preference
                Optimization zero variant (D-9, Pass 156j) which anchors
                chosen and rejected independently to the reference,
                avoiding the DPO failure mode where the model satisfies
                the loss by degrading rejected without improving chosen.

        Returns:
            Final :class:`TrainingState`.
        """
        import copy

        if not preference_data:
            raise ValueError("No preference data provided")

        # Resolve the preference loss variant up front so a typo fails
        # before any model setup work happens.
        loss_fn = self._resolve_preference_loss(loss_type)

        # Pass 156i (DET-1): seed RNGs from config.seed when set so
        # the per-epoch random.shuffle(pairs) is reproducible across
        # runs. Mirrors train() / train_vision() pattern.
        # Pass 156i3 (DET-2): forwards `deterministic` flag for
        # bitwise GPU reproducibility when user opts in.
        if self.config.seed is not None:
            set_training_seed(self.config.seed, deterministic=self.config.deterministic)

        self._stop_requested = False
        self.model.train()
        self._training_start_time = time.monotonic()
        self._epochs_without_improvement = 0

        self._emit_progress(0, "Preparing DPO training...")

        # Build a frozen reference model.
        # Preferred: if the model has LoRA adapters, disable them to get
        # reference logps from the frozen base (zero extra VRAM).
        # Fallback: deepcopy for non-LoRA models.
        use_lora_ref = False
        ref_model = None
        if hasattr(self.model, "disable_adapter_layers"):
            use_lora_ref = True
            logger.info("DPO: using LoRA disable - frozen base weights serve as reference policy (no extra VRAM)")
        else:
            ref_model = copy.deepcopy(self.model)
            ref_model.to(self.device)
            ref_model.eval()
            for p in ref_model.parameters():
                p.requires_grad = False
            logger.info("DPO: using deepcopy reference model")

        # Encode preference pairs
        pairs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        max_len_dpo = getattr(self.model, "config", None)
        max_len_dpo = getattr(max_len_dpo, "max_seq_len", 512) if max_len_dpo else 512
        for item in preference_data:
            self._encode_dpo_pair(item, pairs, max_len_dpo)

        if not pairs:
            raise ValueError("No valid preference pairs after encoding")

        self._emit_progress(5, f"DPO training: {len(pairs)} pairs")
        logger.info(f"DPO training: {len(pairs)} preference pairs, beta={beta}")

        # Gradient accumulation: batch N pairs before optimizer step
        accum_steps = max(1, self.config.max_grad_accumulation)

        # Setup scheduler: SequentialLR(warmup → cosine decay)
        steps_per_epoch = max(1, len(pairs) // accum_steps)
        total_steps = steps_per_epoch * self.config.epochs
        warmup = _effective_warmup(self.config.warmup_steps, total_steps)
        decay_steps = max(1, total_steps - warmup)

        warmup_sched = LambdaLR(self.optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / warmup))
        cosine_sched = CosineAnnealingLR(
            self.optimizer, T_max=decay_steps, eta_min=(self.config.learning_rate * self.config.min_lr_ratio)
        )
        self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup])

        # Restore scheduler/scaler state from checkpoint if available
        self._restore_pending_state()

        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.config.epochs):
            if self._should_stop():
                break
            if (
                self.config.max_training_seconds > 0
                and time.monotonic() - self._training_start_time > self.config.max_training_seconds
            ):
                break

            epoch_loss = 0.0
            n_epoch_pairs = 0
            random.shuffle(pairs)
            self.optimizer.zero_grad()

            progress_base = int(5 + (epoch / self.config.epochs) * 90)
            self._emit_progress(progress_base, f"DPO Epoch {epoch + 1}/{self.config.epochs}")

            # Online pairs generated mid-epoch are staged here and join the
            # pool for the NEXT epoch's shuffle; appending to `pairs` while
            # it is being iterated would train them un-shuffled at the tail
            # and skew this epoch's loss average and flush modulus.
            fresh_pairs: list = []
            for i, (input_ids, chosen_labels, rejected_labels, attention_mask) in enumerate(pairs):
                if self._should_stop():
                    break

                # Policy log-probs
                policy_chosen = self._get_sequence_logps(
                    self.model, input_ids[:1], chosen_labels, attention_mask=attention_mask[:1]
                )
                policy_rejected = self._get_sequence_logps(
                    self.model, input_ids[1:], rejected_labels, attention_mask=attention_mask[1:]
                )

                # Reference log-probs (no grad)
                with torch.no_grad():
                    if use_lora_ref:
                        self.model.eval()
                        self.model.disable_adapter_layers()
                        try:
                            ref_chosen = self._get_sequence_logps(
                                self.model, input_ids[:1], chosen_labels, attention_mask=attention_mask[:1]
                            )
                            ref_rejected = self._get_sequence_logps(
                                self.model, input_ids[1:], rejected_labels, attention_mask=attention_mask[1:]
                            )
                        finally:
                            self.model.enable_adapter_layers()
                        self.model.train()
                    else:
                        ref_chosen = self._get_sequence_logps(
                            ref_model, input_ids[:1], chosen_labels, attention_mask=attention_mask[:1]
                        )
                        ref_rejected = self._get_sequence_logps(
                            ref_model, input_ids[1:], rejected_labels, attention_mask=attention_mask[1:]
                        )

                loss = loss_fn(policy_chosen, policy_rejected, ref_chosen, ref_rejected, beta=beta)

                # Scale loss for gradient accumulation
                loss = loss / accum_steps
                loss.backward()

                batch_loss = loss.item() * accum_steps
                if math.isnan(batch_loss) or math.isinf(batch_loss):
                    logger.error("DPO aborted: NaN/Inf loss")
                    self.state.abort_reason = "DPO NaN/Inf loss detected"
                    self.model.eval()
                    if ref_model is not None:
                        del ref_model
                    return self.state

                epoch_loss += batch_loss
                n_epoch_pairs += 1

                # Step optimizer every accum_steps pairs
                if (i + 1) % accum_steps == 0:
                    if self.config.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    if self.scheduler:
                        self.scheduler.step()
                    self.state.step += 1

                    # T2-5: Online DPO - generate fresh pairs periodically
                    if reward_fn is not None and dpo_online_interval > 0 and self.state.step % dpo_online_interval == 0:
                        sample_prompts = random.sample(
                            [d["prompt"] for d in preference_data if d.get("prompt")], k=min(4, len(preference_data))
                        )
                        new_items = self._generate_online_dpo_pairs(
                            sample_prompts, reward_fn, n_samples=dpo_online_samples
                        )
                        for item in new_items:
                            self._encode_dpo_pair(item, fresh_pairs, max_len_dpo)

            pairs.extend(fresh_pairs)

            # Flush remaining accumulated gradients at epoch end
            if n_epoch_pairs % accum_steps != 0:
                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.scheduler:
                    self.scheduler.step()
                self.state.step += 1

            avg_loss = epoch_loss / max(n_epoch_pairs, 1)
            self.state.training_losses.append(avg_loss)
            self.state.epoch = epoch + 1
            self._emit_loss(avg_loss)
            logger.info(f"DPO Epoch {epoch + 1}: loss={avg_loss:.4f}")

            if self.on_epoch_complete:
                try:
                    self.on_epoch_complete(epoch + 1, avg_loss)
                except Exception:
                    logger.debug("on_epoch_complete callback error", exc_info=True)

            if avg_loss < self.state.best_loss:
                self.state.best_loss = avg_loss
                self._epochs_without_improvement = 0
                self._save_checkpoint(checkpoint_dir / f"{self._checkpoint_stem}_best.pt")
            else:
                self._epochs_without_improvement += 1
                if (
                    self.config.early_stopping_patience > 0
                    and self._epochs_without_improvement >= self.config.early_stopping_patience
                ):
                    logger.info("DPO early stopping triggered")
                    break

        self._emit_progress(100, "DPO training complete")
        self.model.eval()
        if ref_model is not None:
            del ref_model
        return self.state

    # ---------------------------------------------------------------------
    # SimPO - Reference-Free DPO (R14)
    # ---------------------------------------------------------------------

    def train_simpo(
        self,
        preference_data: list[dict],
        beta: float = 2.5,
        gamma: float = 0.5,
    ) -> TrainingState:
        """Train with Simple Preference Optimization (SimPO, R14).

        SimPO eliminates the frozen reference model - halving VRAM
        during preference training.  It uses the average log-prob of
        a response as its implicit reward:

            r(y|x) = (1/|y|) * log pi(y|x)

        Loss:
            -log(sigma(beta * (r_chosen - r_rejected) - gamma))

        where gamma is a target reward margin.

        Args:
            preference_data: List of dicts with ``prompt``, ``chosen``,
                ``rejected`` keys.
            beta: Temperature controlling preference strength.
            gamma: Reward margin (minimum gap between chosen/rejected).

        Returns:
            Final TrainingState.
        """
        if not preference_data:
            raise ValueError("No preference data provided")

        # Pass 156i (DET-1): seed RNGs from config.seed when set.
        if self.config.seed is not None:
            set_training_seed(self.config.seed, deterministic=self.config.deterministic)

        self._stop_requested = False
        self.model.train()
        self._training_start_time = time.monotonic()
        self._epochs_without_improvement = 0

        self._emit_progress(0, "Preparing SimPO training...")

        # Encode preference pairs (same encoding as DPO)
        pairs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for item in preference_data:
            prompt = item.get("prompt", "")
            chosen = item.get("chosen", "")
            rejected = item.get("rejected", "")
            if not (prompt and chosen and rejected):
                continue

            prompt_ids = self.tokenizer.encode(f"User: {prompt}\nAssistant: ")
            chosen_ids = self.tokenizer.encode(f"User: {prompt}\nAssistant: {chosen}")
            rejected_ids = self.tokenizer.encode(f"User: {prompt}\nAssistant: {rejected}")

            max_len_s = getattr(self.model, "config", None)
            max_len_s = getattr(max_len_s, "max_seq_len", 512) if max_len_s else 512
            chosen_ids = chosen_ids[:max_len_s]
            rejected_ids = rejected_ids[:max_len_s]

            # Skip pairs that are too short after truncation
            prompt_len = len(prompt_ids)
            if len(chosen_ids) <= prompt_len or len(rejected_ids) <= prompt_len:
                continue

            chosen_labels = [-100] * min(prompt_len, len(chosen_ids)) + chosen_ids[prompt_len:]
            rejected_labels = [-100] * min(prompt_len, len(rejected_ids)) + rejected_ids[prompt_len:]
            chosen_mask = [1] * len(chosen_ids)
            rejected_mask = [1] * len(rejected_ids)
            max_len = max(len(chosen_ids), len(rejected_ids))
            chosen_mask += [0] * (max_len - len(chosen_mask))
            rejected_mask += [0] * (max_len - len(rejected_mask))
            chosen_ids += [0] * (max_len - len(chosen_ids))
            rejected_ids += [0] * (max_len - len(rejected_ids))
            chosen_labels += [-100] * (max_len - len(chosen_labels))
            rejected_labels += [-100] * (max_len - len(rejected_labels))

            pairs.append(
                (
                    torch.tensor([chosen_ids, rejected_ids], dtype=torch.long, device=self.device),
                    torch.tensor([chosen_labels], dtype=torch.long, device=self.device),
                    torch.tensor([rejected_labels], dtype=torch.long, device=self.device),
                    torch.tensor([chosen_mask, rejected_mask], dtype=torch.long, device=self.device),
                )
            )

        if not pairs:
            raise ValueError("No valid preference pairs after encoding")

        self._emit_progress(5, f"SimPO training: {len(pairs)} pairs")
        logger.info(f"SimPO training: {len(pairs)} pairs, beta={beta}, gamma={gamma}")

        accum_steps = max(1, self.config.max_grad_accumulation)

        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.config.epochs):
            if self._should_stop():
                break
            epoch_loss = 0.0
            random.shuffle(pairs)
            self.optimizer.zero_grad()

            for i, (input_ids, chosen_labels, rejected_labels, attn_mask) in enumerate(pairs):
                if self._should_stop():
                    break

                # SimPO's implicit reward is LENGTH-NORMALIZED:
                # r(y|x) = (1/|y|) * log pi(y|x) -- reference-free preference
                # training is only sound when reward is not confounded by
                # response length, and beta/gamma are calibrated to that
                # scale. The shared helper returns summed logps (the DPO/KTO
                # scale), so divide by each side's scored-token count here.
                policy_chosen = self._get_sequence_logps(
                    self.model, input_ids[:1], chosen_labels, attention_mask=attn_mask[:1]
                )
                policy_rejected = self._get_sequence_logps(
                    self.model, input_ids[1:], rejected_labels, attention_mask=attn_mask[1:]
                )
                chosen_len = (chosen_labels[:, 1:] != -100).sum(dim=-1).clamp(min=1)
                rejected_len = (rejected_labels[:, 1:] != -100).sum(dim=-1).clamp(min=1)
                policy_chosen = policy_chosen / chosen_len
                policy_rejected = policy_rejected / rejected_len

                # SimPO loss: -log sigma(beta * (r_c - r_r) - gamma)
                import torch.nn.functional as _F

                loss = -_F.logsigmoid(beta * (policy_chosen - policy_rejected) - gamma).mean()

                loss = loss / accum_steps
                loss.backward()
                batch_loss = loss.item() * accum_steps
                epoch_loss += batch_loss

                if math.isnan(batch_loss) or math.isinf(batch_loss):
                    logger.error("SimPO aborted: NaN/Inf loss detected")
                    self.state.abort_reason = "SimPO NaN/Inf loss detected"
                    self.model.eval()
                    return self.state

                if (i + 1) % accum_steps == 0:
                    if self.config.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.state.step += 1

            avg_loss = epoch_loss / max(len(pairs), 1)
            self.state.training_losses.append(avg_loss)
            self.state.epoch = epoch + 1
            self._emit_loss(avg_loss)
            logger.info(f"SimPO Epoch {epoch + 1}: loss={avg_loss:.4f}")

            if avg_loss < self.state.best_loss:
                self.state.best_loss = avg_loss
                self._epochs_without_improvement = 0
                self._save_checkpoint(checkpoint_dir / f"{self._checkpoint_stem}_best.pt")
            else:
                self._epochs_without_improvement += 1
                if (
                    self.config.early_stopping_patience > 0
                    and self._epochs_without_improvement >= self.config.early_stopping_patience
                ):
                    break

        self._emit_progress(100, "SimPO training complete")
        self.model.eval()
        return self.state

    # ---------------------------------------------------------------------
    # KTO - Kahneman-Tversky Optimization (R15)
    # ---------------------------------------------------------------------

    def train_kto(
        self,
        feedback_data: list[dict],
        beta: float = 0.1,
        desirable_weight: float = 1.0,
        undesirable_weight: float = 1.0,
    ) -> TrainingState:
        """Train with Kahneman-Tversky Optimization (KTO, R15).

        Unlike DPO, KTO works with *unpaired* feedback - each sample
        is independently labeled as desirable or undesirable (no need
        for matched chosen/rejected pairs).

        Data format: list of dicts with:
            - ``prompt``: the user prompt
            - ``response``: the model response
            - ``desirable``: bool (True = good, False = bad)

        Uses a frozen reference model for KL, same as standard DPO.

        Args:
            feedback_data: List of feedback dicts.
            beta: KTO temperature.
            desirable_weight: Loss weight for desirable examples.
            undesirable_weight: Loss weight for undesirable examples.

        Returns:
            Final TrainingState.
        """
        import copy
        import torch.nn.functional as _F

        if not feedback_data:
            raise ValueError("No feedback data provided")

        desirable = [d for d in feedback_data if d.get("desirable", True)]
        undesirable = [d for d in feedback_data if not d.get("desirable", True)]

        if not desirable and not undesirable:
            raise ValueError("No valid feedback after filtering")

        # Pass 156i (DET-1): seed RNGs from config.seed when set.
        if self.config.seed is not None:
            set_training_seed(self.config.seed, deterministic=self.config.deterministic)

        self._stop_requested = False
        self.model.train()
        self._training_start_time = time.monotonic()
        self._epochs_without_improvement = 0

        self._emit_progress(0, "Preparing KTO training...")

        # Frozen reference model
        use_lora_ref = False
        ref_model = None
        if hasattr(self.model, "disable_adapter_layers"):
            use_lora_ref = True
        else:
            ref_model = copy.deepcopy(self.model)
            ref_model.eval()
            for p in ref_model.parameters():
                p.requires_grad = False

        def _encode_sample(item: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
            prompt = item.get("prompt", "")
            response = item.get("response", "")
            if not (prompt and response):
                return None
            prompt_ids = self.tokenizer.encode(f"User: {prompt}\nAssistant: ")
            full_ids = self.tokenizer.encode(f"User: {prompt}\nAssistant: {response}")
            max_len_k = getattr(self.model, "config", None)
            max_len_k = getattr(max_len_k, "max_seq_len", 512) if max_len_k else 512
            full_ids = full_ids[:max_len_k]
            prompt_len = len(prompt_ids)
            labels = [-100] * min(prompt_len, len(full_ids)) + full_ids[prompt_len:]
            mask = [1] * len(full_ids)
            return (
                torch.tensor([full_ids], dtype=torch.long, device=self.device),
                torch.tensor([labels], dtype=torch.long, device=self.device),
                torch.tensor([mask], dtype=torch.long, device=self.device),
            )

        # Encode all samples
        pos_samples = [s for d in desirable if (s := _encode_sample(d)) is not None]
        neg_samples = [s for d in undesirable if (s := _encode_sample(d)) is not None]

        all_samples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]] = []
        for ids, labels, mask in pos_samples:
            all_samples.append((ids, labels, mask, True))
        for ids, labels, mask in neg_samples:
            all_samples.append((ids, labels, mask, False))

        if not all_samples:
            raise ValueError("No valid feedback samples after encoding")

        self._emit_progress(5, f"KTO: {len(pos_samples)} good, {len(neg_samples)} bad")
        logger.info(f"KTO training: {len(pos_samples)} desirable, {len(neg_samples)} undesirable, beta={beta}")

        # Compute KL baseline from reference on desirable samples. Scaled by
        # beta and clamped non-negative so it lives on the same scale as the
        # rewards it offsets (reward = beta * logratio below), matching the
        # KTO formulation where the reference point is beta * KL(policy||ref).
        kl_baseline = torch.tensor(0.0, device=self.device)
        if pos_samples:
            with torch.no_grad():
                kl_vals = []
                for ids, labels, mask in pos_samples[:50]:  # cap for speed
                    policy_lp = self._get_sequence_logps(self.model, ids, labels, attention_mask=mask)
                    if use_lora_ref:
                        self.model.eval()
                        self.model.disable_adapter_layers()
                        ref_lp = self._get_sequence_logps(self.model, ids, labels, attention_mask=mask)
                        self.model.enable_adapter_layers()
                        self.model.train()
                    else:
                        ref_lp = self._get_sequence_logps(ref_model, ids, labels, attention_mask=mask)
                    kl_vals.append((policy_lp - ref_lp).mean())
                kl_baseline = (beta * torch.stack(kl_vals).mean()).clamp(min=0.0)

        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(self.config.epochs):
            if self._should_stop():
                break
            epoch_loss = 0.0
            random.shuffle(all_samples)
            self.optimizer.zero_grad()

            for i, (ids, labels, mask, is_desirable) in enumerate(all_samples):
                if self._should_stop():
                    break

                policy_lp = self._get_sequence_logps(self.model, ids, labels, attention_mask=mask)

                with torch.no_grad():
                    if use_lora_ref:
                        self.model.eval()
                        self.model.disable_adapter_layers()
                        ref_lp = self._get_sequence_logps(self.model, ids, labels, attention_mask=mask)
                        self.model.enable_adapter_layers()
                        self.model.train()
                    else:
                        ref_lp = self._get_sequence_logps(ref_model, ids, labels, attention_mask=mask)

                # KTO loss per sample
                reward = beta * (policy_lp - ref_lp)
                if is_desirable:
                    loss = desirable_weight * (1 - _F.sigmoid(reward - kl_baseline))
                else:
                    loss = undesirable_weight * (1 - _F.sigmoid(kl_baseline - reward))

                loss = loss.mean()
                accum = max(1, self.config.max_grad_accumulation)
                loss = loss / accum
                loss.backward()
                batch_loss = loss.item() * accum
                epoch_loss += batch_loss

                if math.isnan(batch_loss) or math.isinf(batch_loss):
                    logger.error("KTO aborted: NaN/Inf loss detected")
                    self.state.abort_reason = "KTO NaN/Inf loss detected"
                    self.model.eval()
                    return self.state

                if (i + 1) % accum == 0:
                    if self.config.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.state.step += 1

            avg_loss = epoch_loss / max(len(all_samples), 1)
            self.state.training_losses.append(avg_loss)
            self.state.epoch = epoch + 1
            self._emit_loss(avg_loss)
            logger.info(f"KTO Epoch {epoch + 1}: loss={avg_loss:.4f}")

            if avg_loss < self.state.best_loss:
                self.state.best_loss = avg_loss
                self._epochs_without_improvement = 0
                self._save_checkpoint(checkpoint_dir / f"{self._checkpoint_stem}_best.pt")
            else:
                self._epochs_without_improvement += 1
                if (
                    self.config.early_stopping_patience > 0
                    and self._epochs_without_improvement >= self.config.early_stopping_patience
                ):
                    break

        self._emit_progress(100, "KTO training complete")
        self.model.eval()
        if ref_model is not None:
            del ref_model
        return self.state

    # ---------------------------------------------------------------------
    # ORPO - Odds Ratio Preference Optimization (R32)
    # ---------------------------------------------------------------------

    def train_orpo(
        self,
        preference_data: list[dict],
        beta: float = 0.1,
    ) -> TrainingState:
        """Train with Odds Ratio Preference Optimization (ORPO, R32).

        ORPO combines SFT and alignment in a single stage - no reference
        model needed.  The loss is:

            L = L_SFT(chosen) + beta * L_OR

        where L_OR penalizes the model when the odds of generating the
        chosen response are not sufficiently higher than the rejected:

            L_OR = -log(sigma(log(odds_chosen / odds_rejected)))
            odds(y|x) = p(y|x) / (1 - p(y|x))

        This eliminates the need for a separate SFT stage before DPO.

        Args:
            preference_data: List of dicts with ``prompt``, ``chosen``,
                ``rejected`` keys.
            beta: Weight for the odds ratio loss term.

        Returns:
            Final TrainingState.
        """
        import torch.nn.functional as _F

        if not preference_data:
            raise ValueError("No preference data provided")

        # Pass 156i (DET-1): seed RNGs from config.seed when set.
        if self.config.seed is not None:
            set_training_seed(self.config.seed, deterministic=self.config.deterministic)

        self._stop_requested = False
        self.model.train()
        self._training_start_time = time.monotonic()
        self._epochs_without_improvement = 0

        self._emit_progress(0, "Preparing ORPO training...")

        # Encode preference pairs
        pairs: list[tuple[torch.Tensor, torch.Tensor, int]] = []
        for item in preference_data:
            prompt = item.get("prompt", "")
            chosen = item.get("chosen", "")
            rejected = item.get("rejected", "")
            if not (prompt and chosen and rejected):
                continue

            chosen_text = f"User: {prompt}\nAssistant: {chosen}"
            rejected_text = f"User: {prompt}\nAssistant: {rejected}"
            prompt_text = f"User: {prompt}\nAssistant: "

            chosen_ids = self.tokenizer.encode(chosen_text)
            rejected_ids = self.tokenizer.encode(rejected_text)
            prompt_len = len(self.tokenizer.encode(prompt_text))

            max_len = getattr(
                self.model, "max_seq_len", getattr(getattr(self.model, "config", None), "max_seq_len", 2048)
            )

            if len(chosen_ids) < prompt_len + 2 or len(rejected_ids) < prompt_len + 2:
                continue

            chosen_ids = chosen_ids[:max_len]
            rejected_ids = rejected_ids[:max_len]

            pairs.append(
                (
                    torch.tensor(chosen_ids, dtype=torch.long),
                    torch.tensor(rejected_ids, dtype=torch.long),
                    prompt_len,
                )
            )

        if not pairs:
            raise ValueError("No valid preference pairs after encoding")

        logger.info(f"ORPO: {len(pairs)} pairs, beta={beta}")

        self._setup_optimizer()

        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        scaler = torch.amp.GradScaler(
            "cuda", enabled=(self.config.use_amp and self.device.type == "cuda" and self._amp_dtype != torch.bfloat16)
        )

        for epoch in range(self.config.epochs):
            if self._should_stop():
                break

            random.shuffle(pairs)
            epoch_loss = 0.0

            for i, (chosen_ids, rejected_ids, prompt_len) in enumerate(pairs):
                if self._should_stop():
                    break

                chosen_ids = chosen_ids.unsqueeze(0).to(self.device)
                rejected_ids = rejected_ids.unsqueeze(0).to(self.device)

                with torch.amp.autocast(
                    self.device.type,
                    dtype=self._amp_dtype,
                    enabled=self.config.use_amp,
                ):
                    # Forward pass for chosen
                    chosen_logits = self.model(chosen_ids)
                    chosen_logps = _F.log_softmax(chosen_logits[:, :-1], dim=-1)
                    chosen_logps = chosen_logps.gather(2, chosen_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
                    # Mask prompt tokens
                    resp_mask_c = torch.zeros_like(chosen_logps[0], dtype=torch.bool)
                    resp_mask_c[max(0, prompt_len - 1) :] = True
                    chosen_resp_logps = chosen_logps[0][resp_mask_c]

                    # SFT loss on chosen (NLL)
                    sft_targets = chosen_ids[:, 1:]
                    sft_loss = _F.cross_entropy(
                        chosen_logits[:, :-1].reshape(-1, chosen_logits.size(-1)),
                        sft_targets.reshape(-1),
                        label_smoothing=self.config.label_smoothing,
                    )

                    # Forward pass for rejected
                    rejected_logits = self.model(rejected_ids)
                    rejected_logps = _F.log_softmax(rejected_logits[:, :-1], dim=-1)
                    rejected_logps = rejected_logps.gather(2, rejected_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
                    resp_mask_r = torch.zeros_like(rejected_logps[0], dtype=torch.bool)
                    resp_mask_r[max(0, prompt_len - 1) :] = True
                    rejected_resp_logps = rejected_logps[0][resp_mask_r]

                    # Average log-probs ? log-odds
                    avg_logp_c = chosen_resp_logps.mean()
                    avg_logp_r = rejected_resp_logps.mean()

                    # odds = p / (1-p) = exp(logp) / (1 - exp(logp))
                    # log_odds = logp - log(1 - exp(logp))
                    log_odds_c = avg_logp_c - torch.log1p(-avg_logp_c.exp().clamp(max=1.0 - 1e-7))
                    log_odds_r = avg_logp_r - torch.log1p(-avg_logp_r.exp().clamp(max=1.0 - 1e-7))

                    # Odds ratio loss
                    log_odds_ratio = log_odds_c - log_odds_r
                    or_loss = -_F.logsigmoid(log_odds_ratio)

                    loss = sft_loss + beta * or_loss

                accum = max(1, self.config.max_grad_accumulation)
                loss = loss / accum
                scaler.scale(loss).backward()

                if (i + 1) % accum == 0:
                    if self.config.gradient_clip > 0:
                        scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()
                    if self.scheduler:
                        self.scheduler.step()

                epoch_loss += loss.item() * accum

                pct = int(((epoch * len(pairs) + i + 1) / (self.config.epochs * len(pairs))) * 100)
                self._emit_progress(pct, f"ORPO E{epoch + 1} [{i + 1}/{len(pairs)}]")

            avg_loss = epoch_loss / max(len(pairs), 1)
            self.state.training_losses.append(avg_loss)
            self.state.epoch = epoch + 1
            self._emit_loss(avg_loss)
            logger.info(f"ORPO Epoch {epoch + 1}: loss={avg_loss:.4f}")

            if avg_loss < self.state.best_loss:
                self.state.best_loss = avg_loss
                self._epochs_without_improvement = 0
                self._save_checkpoint(checkpoint_dir / f"{self._checkpoint_stem}_best.pt")
            else:
                self._epochs_without_improvement += 1
                if (
                    self.config.early_stopping_patience > 0
                    and self._epochs_without_improvement >= self.config.early_stopping_patience
                ):
                    break

        self._emit_progress(100, "ORPO training complete")
        self.model.eval()
        return self.state

    # ─────────────────────────────────────────────────────────────────────
    # VISION TRAINING — image-text pair training
    # ─────────────────────────────────────────────────────────────────────

    def train_vision(
        self,
        vision_encoder: nn.Module,
        data: list[dict[str, Any]],
        unfreeze_text_layers: int = 0,
        val_data: list[dict[str, Any]] | None = None,
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

        Returns:
            TrainingState with loss history.

        Raises:
            ValueError: If model lacks vision_projection or data is empty.
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

        # ── Freeze / unfreeze layers ────────────────────────────────────
        # Freeze all text model parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze vision projection (always trainable)
        for param in self.model.vision_projection.parameters():
            param.requires_grad = True

        # Unfreeze output/embedding (needed for text loss)
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
        total_steps = len(data) * self.config.epochs  # 1 step per pair
        warmup = _effective_warmup(self.config.warmup_steps, total_steps)
        decay_steps = max(1, total_steps - warmup)
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / warmup))
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=decay_steps, eta_min=(self.config.learning_rate * self.config.min_lr_ratio)
        )
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup])

        image_size = getattr(getattr(vision_encoder, "config", None), "image_size", 224)
        use_imagenet = getattr(getattr(vision_encoder, "config", None), "use_pretrained", False)
        # V-2: lazy preprocess. Store image refs (paths or PIL objects)
        # only - never preprocess/.to(device) eagerly. At LLaVA-Pretrain
        # scale (558K - 600 KB tensors) eager prep would OOM the GPU
        # immediately on a 16 GB VRAM budget. Tensors are built per-step
        # below and freed after backward.
        pairs: list[tuple[object, list[int]]] = []

        for i, item in enumerate(data):
            image = item.get("image")
            text = item.get("text", "")
            if image is None or not text:
                logger.warning(f"Skipping vision data item {i}: missing image or text")
                continue

            # V-2: lightweight readability probe for path inputs so bad
            # files surface here instead of mid-training. PIL `verify()`
            # parses the header without decoding pixels - cheap.
            if isinstance(image, (str, Path)):
                try:
                    from PIL import Image as _PILImage

                    with _PILImage.open(str(image)) as _probe:
                        _probe.verify()
                except Exception as exc:
                    logger.warning(f"Skipping vision data item {i}: {exc}")
                    continue

            # Encode text to token IDs
            token_ids = self.tokenizer.encode(text)
            if len(token_ids) < 1:
                continue
            pairs.append((image, token_ids))

        if not pairs:
            raise ValueError("No valid image-text pairs found in training data.")

        # V-6: optional held-out validation pairs. Same lightweight
        # readability probe as train; same lazy-preprocess discipline
        # (no .to(device) until inside the eval pass). Pairs whose
        # caption is <2 tokens are skipped silently because the
        # next-token loss can't be computed on them - matches the
        # train-loop drop policy at min_len < 1.
        val_pairs: list[tuple[object, list[int]]] = []
        if val_data:
            for i, item in enumerate(val_data):
                v_image = item.get("image")
                v_text = item.get("text", "")
                if v_image is None or not v_text:
                    logger.warning(f"Skipping vision val item {i}: missing image or text")
                    continue
                if isinstance(v_image, (str, Path)):
                    try:
                        from PIL import Image as _PILImage

                        with _PILImage.open(str(v_image)) as _probe:
                            _probe.verify()
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
            Returns mean cross-entropy loss across all valid val
            samples, or None when val_pairs is empty / every sample
            was dropped. Caller restores train mode.

            Pass 156g audit (Bug B): polls ``self._should_stop()``
            between samples so the user's STOP press isn't ignored
            during a long val pass at LLaVA scale (~28K samples on a
            5% split). Returns the partial mean over completed samples
            on early exit; ``None`` if zero finished."""
            if not val_pairs:
                return None
            vision_encoder.eval()
            self.model.eval()
            losses: list[float] = []
            try:
                with torch.no_grad():
                    for v_ref, v_ids in val_pairs:
                        if self._should_stop():
                            logger.info("Vision validation stopped by user request after %d sample(s)", len(losses))
                            break
                        try:
                            v_img = preprocess_image(
                                v_ref,
                                image_size=image_size,
                                imagenet_normalize=use_imagenet,
                            ).to(self.device)
                        except Exception as exc:
                            logger.debug("Vision val: preprocess failed (%s); skipping", exc)
                            continue
                        with torch.amp.autocast(
                            "cuda",
                            dtype=self._amp_dtype,
                            enabled=self.config.use_amp and self.device.type == "cuda",
                        ):
                            v_feats = vision_encoder(v_img)
                            v_text_tensor = torch.tensor(
                                [v_ids],
                                dtype=torch.long,
                                device=self.device,
                            )
                            v_logits = self.model.forward_multimodal(
                                input_ids=v_text_tensor,
                                vision_features=v_feats,
                            )
                            v_n_patches = v_feats.shape[1]
                            if v_n_patches >= v_logits.shape[1]:
                                continue
                            v_text_logits = v_logits[:, v_n_patches:-1, :]
                            v_text_targets = v_text_tensor[:, 1:]
                            v_min_len = min(v_text_logits.shape[1], v_text_targets.shape[1])
                            if v_min_len < 1:
                                continue
                            v_text_logits = v_text_logits[:, :v_min_len, :]
                            v_text_targets = v_text_targets[:, :v_min_len]
                            v_loss = nn.functional.cross_entropy(
                                v_text_logits.reshape(-1, v_text_logits.size(-1)),
                                v_text_targets.reshape(-1),
                            )
                        losses.append(v_loss.item())
            finally:
                vision_encoder.train()
                self.model.train()
            if not losses:
                return None
            return sum(losses) / len(losses)

        for epoch in range(self.config.epochs):
            if self._should_stop():
                logger.info("Vision training stopped by user request")
                break

            # Time limit check
            if self.config.max_training_seconds > 0 and (time.time() - start_time) > self.config.max_training_seconds:
                logger.info("Vision training: time limit reached")
                break

            epoch_loss = 0.0
            epoch_steps = 0

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

            for step, (image_ref, token_ids) in enumerate(pairs):
                if self._should_stop():
                    break

                # V-2: lazy preprocess - build tensor here so each one
                # is freed after backward instead of accumulating across
                # the full dataset. Per-image preprocess errors skip
                # the sample (already validated by header probe in prep,
                # so this should be rare).
                try:
                    img_tensor = preprocess_image(
                        image_ref,
                        image_size=image_size,
                        imagenet_normalize=use_imagenet,
                    ).to(self.device)
                except Exception as exc:
                    logger.warning("Vision step %d: preprocess failed (%s); skipping sample", step, exc)
                    continue

                # Training-time augmentation (random flip, color jitter)
                img_tensor = augment_vision_tensor(img_tensor)

                # Encode image through vision encoder
                with torch.amp.autocast(
                    "cuda",
                    dtype=self._amp_dtype,
                    enabled=self.config.use_amp and self.device.type == "cuda",
                ):
                    vision_features = vision_encoder(img_tensor)  # [1, patches, v_dim]

                    # Build text input: token IDs as target
                    text_tensor = torch.tensor([token_ids], dtype=torch.long, device=self.device)

                    # Forward through model with vision features
                    # The model concatenates [vision_patches, text_tokens] internally
                    logits = self.model.forward_multimodal(
                        input_ids=text_tensor,
                        vision_features=vision_features,
                    )
                    # logits shape: [1, vision_patches + text_len, vocab_size]

                    # We only compute loss on the text portion
                    # The text tokens start after the vision patches
                    n_patches = vision_features.shape[1]
                    if n_patches >= logits.shape[1]:
                        logger.warning(
                            "Vision patches (%d) >= logits length (%d), skipping sample", n_patches, logits.shape[1]
                        )
                        continue
                    text_logits = logits[:, n_patches:-1, :]  # predict next token
                    text_targets = text_tensor[:, 1:]  # shift targets

                    # Align lengths (in case of truncation)
                    min_len = min(text_logits.shape[1], text_targets.shape[1])
                    if min_len < 1:
                        # V-7: caption too short for next-token loss.
                        # Log the first occurrence with explanation;
                        # subsequent drops are counted silently and
                        # summarized after training.
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

                    text_logits = text_logits[:, :min_len, :]
                    text_targets = text_targets[:, :min_len]

                    loss = nn.functional.cross_entropy(
                        text_logits.reshape(-1, text_logits.size(-1)),
                        text_targets.reshape(-1),
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
                    self.model.eval()
                    return self.state

                # Safety guard
                if loss_val > self.config.max_loss:
                    logger.error(f"Vision training: loss {loss_val:.2f} exceeds max {self.config.max_loss}")
                    self._emit_progress(100, "Training aborted: loss too high")
                    self.state.abort_reason = f"Vision loss {loss_val:.2f} exceeded max_loss {self.config.max_loss}"
                    _emit_drop_summary()  # V-7 audit fix
                    self.model.eval()
                    return self.state

                epoch_loss += loss_val
                epoch_steps += 1
                if is_boundary:
                    self.state.step += 1

                # Log every N steps
                if self.config.log_every > 0 and self.state.step % self.config.log_every == 0:
                    self._emit_loss(loss_val)

                # Progress within epoch
                total_steps = len(pairs) * self.config.epochs
                done_steps = epoch * len(pairs) + step + 1
                pct = min(int(done_steps / max(total_steps, 1) * 95) + 5, 99)
                self._emit_progress(pct, f"Epoch {epoch + 1}/{self.config.epochs} — loss: {loss_val:.4f}")

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

            # Epoch summary
            avg_loss = epoch_loss / max(epoch_steps, 1)
            self.state.training_losses.append(avg_loss)
            self.state.epoch = epoch + 1
            self._emit_loss(avg_loss)
            logger.info(f"Vision Epoch {epoch + 1}: avg_loss={avg_loss:.4f}")

            # V-6: held-out validation pass after each epoch. Drives
            # best-checkpoint + early-stopping when available; falls
            # back to training avg_loss otherwise.
            val_loss = _run_validation()
            if val_loss is not None:
                self.state.validation_losses.append(val_loss)
                logger.info(f"Vision Epoch {epoch + 1}: val_loss={val_loss:.4f}")

            if self.on_epoch_complete:
                try:
                    self.on_epoch_complete(epoch + 1, avg_loss)
                except Exception:
                    logger.debug("on_epoch_complete callback error", exc_info=True)

            # Best model tracking and early stopping. Prefer val_loss
            # when present (V-6); fall back to train avg_loss.
            tracked_loss = val_loss if val_loss is not None else avg_loss
            if tracked_loss < self.state.best_loss:
                self.state.best_loss = tracked_loss
                self._epochs_without_improvement = 0
                self._save_checkpoint(checkpoint_dir / f"{self._checkpoint_stem}_vision_best.pt")
            else:
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
                self._save_checkpoint(checkpoint_dir / f"{stem}_vision{v_num}.pt")
                self._cleanup_periodic_checkpoints(checkpoint_dir, f"{stem}_vision", keep=3)

        # ── Cleanup ─────────────────────────────────────────────────────
        self._emit_progress(100, "Vision training complete!")
        self.model.eval()
        vision_encoder.eval()

        # V-7: end-of-run summary if any captions were dropped.
        _emit_drop_summary()

        # Re-enable all model parameters for subsequent use
        for param in self.model.parameters():
            param.requires_grad = True

        return self.state

    # -----------------------------------------------------------------------
    # T4-3: REINFORCED SELF-TRAINING (ReST)
    # -----------------------------------------------------------------------

    def train_rest(
        self,
        prompts: list[str],
        reward_fn: "Callable[[str, str], float]",
        rounds: int = 3,
        n_samples: int = 4,
        beta: float = 0.1,
        max_new_tokens: int = 128,
    ) -> "TrainingState":
        """Reinforced Self-Training (ReST) loop.

        Automated self-improvement cycle:
        1. Generate *n_samples* responses per prompt.
        2. Score each response with *reward_fn*.
        3. Best ? chosen, worst ? rejected ? DPO pair.
        4. Train one DPO epoch on generated pairs.
        5. Repeat for *rounds*.

        Reuses :meth:`_generate_online_dpo_pairs` for generation/scoring
        and :meth:`train_dpo` for the preference training step.

        Args:
            prompts: Seed prompts to generate responses for.
            reward_fn: ``(prompt, response) -> float`` scoring function.
            rounds: Number of generate?train cycles.
            n_samples: Responses per prompt per round.
            beta: DPO temperature (lower = stronger signal).
            max_new_tokens: Max tokens per generated response.

        Returns:
            Final :class:`TrainingState` after all rounds.
        """
        if not prompts:
            raise ValueError("No prompts provided for ReST")

        # Pass 156i (DET-1): seed RNGs from config.seed when set so the
        # outer round-loop generation is reproducible (the inner DPO
        # call also seeds, but we want round-1 generation deterministic
        # too). Mirrors train() / train_vision().
        #
        # RNG flow note (Pass 156i2 logic-eye finding): each inner
        # train_dpo() call re-seeds back to config.seed at its own
        # entry. Round-2 DPO therefore starts from the *same* RNG
        # state as round-1 DPO, but operates on different `pairs`
        # (regenerated from the updated model). The actual sample
        # order differs because the data differs; the RNG draws are
        # identical streams across rounds. This is intentional -
        # reproducibility is the goal - but a user expecting "fresh
        # randomness per round" should derive a per-round seed
        # (config.seed + rnd) outside this method.
        if self.config.seed is not None:
            set_training_seed(self.config.seed, deterministic=self.config.deterministic)

        self._stop_requested = False
        logger.info("ReST: starting %d rounds (%d prompts, %d samples/prompt)", rounds, len(prompts), n_samples)

        for rnd in range(rounds):
            if self._should_stop():
                logger.info("ReST: stop requested at round %d", rnd)
                break

            self._emit_progress(int(rnd / rounds * 100), f"ReST round {rnd + 1}/{rounds}: generating responses...")

            # Generate and score responses
            pairs = self._generate_online_dpo_pairs(
                prompts,
                reward_fn,
                n_samples=n_samples,
                max_new_tokens=max_new_tokens,
            )

            if not pairs:
                logger.warning("ReST round %d: no valid pairs generated, skipping", rnd + 1)
                continue

            logger.info("ReST round %d/%d: training on %d preference pairs", rnd + 1, rounds, len(pairs))

            # Train one DPO epoch on generated pairs
            saved_epochs = self.config.epochs
            self.config.epochs = 1
            try:
                self.train_dpo(pairs, beta=beta)
            finally:
                self.config.epochs = saved_epochs

        self._emit_progress(100, "ReST complete!")
        logger.info("ReST: completed %d rounds", rounds)
        return self.state

    # -----------------------------------------------------------------------
    # AUDIO MULTIMODAL TRAINING
    # -----------------------------------------------------------------------

    def train_audio(
        self,
        audio_encoder: nn.Module,
        data: list[dict[str, Any]],
        unfreeze_text_layers: int = 0,
        val_data: list[dict[str, Any]] | None = None,
    ) -> "TrainingState":
        """
        Train the audio encoder and projection layer on audio-text pairs.

        Both the audio encoder and the model's audio_projection layer are
        trained together. The text transformer is frozen by default (set
        unfreeze_text_layers > 0 to fine-tune the last N text layers too).

        Data format: list of dicts with:
            - "audio": file path string, Path, or 1D waveform tensor
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
                Pass 156z9ec: mirrors V-6 vision val_data — closes the
                "overfitting on small audio datasets is invisible" gap.

        Returns:
            TrainingState with loss history.

        Raises:
            ValueError: If model lacks audio_projection or data is empty.
        """
        from enigma_engine.core.audio_encoder import preprocess_audio, spec_augment

        # -- Validation --
        if not hasattr(self.model, "audio_projection") or self.model.audio_projection is None:
            raise ValueError(
                "Model does not have an audio projection layer. Set audio_hidden_size in ForgeConfig to enable audio."
            )
        if not data:
            raise ValueError("No training data provided for audio training.")

        # Pass 156i (DET-1): seed RNGs from config.seed when set so the
        # per-epoch random.shuffle(pairs) is reproducible.
        if self.config.seed is not None:
            set_training_seed(self.config.seed, deterministic=self.config.deterministic)

        self._stop_requested = False
        self.state = TrainingState()
        self._epochs_without_improvement = 0
        start_time = time.time()

        self._emit_progress(0, "Preparing audio training data...")

        # -- Freeze / unfreeze layers --
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze audio projection (always trainable)
        for param in self.model.audio_projection.parameters():
            param.requires_grad = True

        # Unfreeze output/embedding (needed for text loss)
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

        # Audio encoder is fully trainable
        for param in audio_encoder.parameters():
            param.requires_grad = True

        trainable_params = list(filter(lambda p: p.requires_grad, self.model.parameters())) + list(
            filter(lambda p: p.requires_grad, audio_encoder.parameters())
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
        total_steps = len(data) * self.config.epochs  # 1 step per pair
        warmup = _effective_warmup(self.config.warmup_steps, total_steps)
        decay_steps = max(1, total_steps - warmup)
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / warmup))
        cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=decay_steps, eta_min=(self.config.learning_rate * self.config.min_lr_ratio)
        )
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup])

        # -- Preprocess data --
        audio_config = getattr(audio_encoder, "config", None)
        pairs: list[tuple[torch.Tensor, list[int]]] = []

        for i, item in enumerate(data):
            audio = item.get("audio")
            text = item.get("text", "")
            if audio is None or not text:
                logger.warning(f"Skipping audio data item {i}: missing audio or text")
                continue

            try:
                mel_tensor = preprocess_audio(audio, config=audio_config)
                mel_tensor = mel_tensor.to(self.device)
            except Exception as exc:
                logger.warning(f"Skipping audio data item {i}: {exc}")
                continue

            token_ids = self.tokenizer.encode(text)
            if len(token_ids) < 1:
                continue
            pairs.append((mel_tensor, token_ids))

        if not pairs:
            raise ValueError("No valid audio-text pairs found in training data.")

        # Pass 156z9ec: build held-out val_pairs eagerly, matching the
        # training preprocess pattern (mel computed once, moved to
        # device). Pairs with missing fields, preprocessing failures,
        # or <1-token captions are skipped silently — matches train
        # drop policy.
        val_pairs: list[tuple[torch.Tensor, list[int]]] = []
        if val_data:
            for i, item in enumerate(val_data):
                v_audio = item.get("audio")
                v_text = item.get("text", "")
                if v_audio is None or not v_text:
                    logger.warning(f"Skipping audio val item {i}: missing audio or text")
                    continue
                try:
                    v_mel = preprocess_audio(v_audio, config=audio_config)
                    v_mel = v_mel.to(self.device)
                except Exception as exc:
                    logger.warning(f"Skipping audio val item {i}: {exc}")
                    continue
                v_token_ids = self.tokenizer.encode(v_text)
                if len(v_token_ids) < 1:
                    continue
                val_pairs.append((v_mel, v_token_ids))
            if val_pairs:
                logger.info(f"Audio validation: {len(val_pairs)} held-out pairs")

        self._emit_progress(5, f"Prepared {len(pairs)} audio-text pairs")
        logger.info(f"Audio training: {len(pairs)} pairs, {self.config.epochs} epochs")

        # -- Training loop --
        audio_encoder.train()
        self.model.train()

        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        def _run_audio_validation() -> float | None:
            """Pass 156z9ec: no-grad evaluation over the held-out
            val_pairs. Returns mean cross-entropy loss across all valid
            val samples, or None when val_pairs is empty / every sample
            was dropped. Caller restores train mode.

            Mirrors vision's V-6 ``_run_validation`` including the
            Pass 156g audit (Bug B) stop-poll between samples so the
            user's STOP press isn't ignored during a long val pass."""
            if not val_pairs:
                return None
            audio_encoder.eval()
            self.model.eval()
            losses: list[float] = []
            try:
                with torch.no_grad():
                    for v_mel, v_ids in val_pairs:
                        if self._should_stop():
                            logger.info("Audio validation stopped by user request after %d sample(s)", len(losses))
                            break
                        try:
                            with torch.amp.autocast(
                                "cuda",
                                dtype=self._amp_dtype,
                                enabled=self.config.use_amp and self.device.type == "cuda",
                            ):
                                # No SpecAugment during validation —
                                # eval the clean mel.
                                v_feats = audio_encoder(v_mel)
                                v_text_tensor = torch.tensor(
                                    [v_ids],
                                    dtype=torch.long,
                                    device=self.device,
                                )
                                v_logits = self.model.forward_multimodal(
                                    input_ids=v_text_tensor,
                                    audio_features=v_feats,
                                )
                                v_n_audio = v_feats.shape[1]
                                if v_n_audio >= v_logits.shape[1]:
                                    continue
                                v_text_logits = v_logits[:, v_n_audio:-1, :]
                                v_text_targets = v_text_tensor[:, 1:]
                                v_min_len = min(v_text_logits.shape[1], v_text_targets.shape[1])
                                if v_min_len < 1:
                                    continue
                                v_text_logits = v_text_logits[:, :v_min_len, :]
                                v_text_targets = v_text_targets[:, :v_min_len]
                                v_loss = nn.functional.cross_entropy(
                                    v_text_logits.reshape(-1, v_text_logits.size(-1)),
                                    v_text_targets.reshape(-1),
                                )
                        except Exception as exc:
                            logger.debug("Audio val: forward failed (%s); skipping", exc)
                            continue
                        losses.append(v_loss.item())
            finally:
                audio_encoder.train()
                self.model.train()
            if not losses:
                return None
            return sum(losses) / len(losses)

        for epoch in range(self.config.epochs):
            if self._should_stop():
                logger.info("Audio training stopped by user request")
                break

            if self.config.max_training_seconds > 0 and (time.time() - start_time) > self.config.max_training_seconds:
                logger.info("Audio training: time limit reached")
                break

            epoch_loss = 0.0
            epoch_steps = 0

            random.shuffle(pairs)

            for step, (mel_tensor, token_ids) in enumerate(pairs):
                if self._should_stop():
                    break

                optimizer.zero_grad()

                with torch.amp.autocast(
                    "cuda",
                    dtype=self._amp_dtype,
                    enabled=self.config.use_amp and self.device.type == "cuda",
                ):
                    # SpecAugment: mask random frequency/time bands for regularization
                    augmented_mel = spec_augment(mel_tensor)

                    # Encode audio through audio encoder
                    audio_features = audio_encoder(augmented_mel)  # [1, T/2, a_dim]

                    text_tensor = torch.tensor([token_ids], dtype=torch.long, device=self.device)

                    logits = self.model.forward_multimodal(
                        input_ids=text_tensor,
                        audio_features=audio_features,
                    )

                    # Text loss only (skip audio tokens)
                    n_audio_tokens = audio_features.shape[1]
                    text_logits = logits[:, n_audio_tokens:-1, :]
                    text_targets = text_tensor[:, 1:]

                    min_len = min(text_logits.shape[1], text_targets.shape[1])
                    if min_len < 1:
                        continue

                    text_logits = text_logits[:, :min_len, :]
                    text_targets = text_targets[:, :min_len]

                    loss = nn.functional.cross_entropy(
                        text_logits.reshape(-1, text_logits.size(-1)),
                        text_targets.reshape(-1),
                    )

                if self.config.use_amp and self.device.type == "cuda":
                    scaler.scale(loss).backward()
                    if self.config.gradient_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(trainable_params, self.config.gradient_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if self.config.gradient_clip > 0:
                        torch.nn.utils.clip_grad_norm_(trainable_params, self.config.gradient_clip)
                    optimizer.step()

                scheduler.step()

                loss_val = loss.item()

                if math.isnan(loss_val) or math.isinf(loss_val):
                    logger.error(f"Audio training: NaN/Inf loss at epoch {epoch + 1}, step {step}")
                    self._emit_progress(100, "Training aborted: NaN loss")
                    self.state.abort_reason = f"Audio NaN/Inf loss at epoch {epoch + 1}, step {step}"
                    self.model.eval()
                    return self.state

                if loss_val > self.config.max_loss:
                    logger.error(f"Audio training: loss {loss_val:.2f} exceeds max {self.config.max_loss}")
                    self._emit_progress(100, "Training aborted: loss too high")
                    self.state.abort_reason = f"Audio loss {loss_val:.2f} exceeded max_loss {self.config.max_loss}"
                    self.model.eval()
                    return self.state

                epoch_loss += loss_val
                epoch_steps += 1
                self.state.step += 1

                if self.config.log_every > 0 and self.state.step % self.config.log_every == 0:
                    self._emit_loss(loss_val)

                total_steps = len(pairs) * self.config.epochs
                done_steps = epoch * len(pairs) + step + 1
                pct = min(int(done_steps / max(total_steps, 1) * 95) + 5, 99)
                self._emit_progress(pct, f"Epoch {epoch + 1}/{self.config.epochs} - loss: {loss_val:.4f}")

            avg_loss = epoch_loss / max(epoch_steps, 1)
            self.state.training_losses.append(avg_loss)
            self.state.epoch = epoch + 1
            self._emit_loss(avg_loss)
            logger.info(f"Audio Epoch {epoch + 1}: avg_loss={avg_loss:.4f}")

            # Pass 156z9ec: held-out validation pass after each epoch.
            # Drives best-checkpoint + early-stopping when available;
            # falls back to training avg_loss otherwise. Mirrors V-6.
            val_loss = _run_audio_validation()
            if val_loss is not None:
                self.state.validation_losses.append(val_loss)
                logger.info(f"Audio Epoch {epoch + 1}: val_loss={val_loss:.4f}")

            if self.on_epoch_complete:
                try:
                    self.on_epoch_complete(epoch + 1, avg_loss)
                except Exception:
                    logger.debug("on_epoch_complete callback error", exc_info=True)

            # Best model tracking and early stopping. Prefer val_loss
            # when present (Pass 156z9ec); fall back to train avg_loss.
            tracked_loss = val_loss if val_loss is not None else avg_loss
            if tracked_loss < self.state.best_loss:
                self.state.best_loss = tracked_loss
                self._epochs_without_improvement = 0
                self._save_checkpoint(checkpoint_dir / f"{self._checkpoint_stem}_audio_best.pt")
            else:
                self._epochs_without_improvement += 1
                if (
                    self.config.early_stopping_patience > 0
                    and self._epochs_without_improvement >= self.config.early_stopping_patience
                ):
                    logger.info("Audio training: early stopping triggered")
                    break

            if self.config.save_every > 0 and (epoch + 1) % self.config.save_every == 0:
                stem = self._checkpoint_stem
                a_num = (epoch + 1) // self.config.save_every
                self._save_checkpoint(checkpoint_dir / f"{stem}_audio{a_num}.pt")
                self._cleanup_periodic_checkpoints(checkpoint_dir, f"{stem}_audio", keep=3)

        # -- Cleanup --
        self._emit_progress(100, "Audio training complete!")
        self.model.eval()
        audio_encoder.eval()

        for param in self.model.parameters():
            param.requires_grad = True

        return self.state
