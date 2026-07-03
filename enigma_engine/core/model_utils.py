"""
Utility functions and global model registry for Enigma Engine.

Contains sampling helpers, hardware detection utilities, and the
thread-safe global model registry.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from .model import Enigma

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL MODEL REGISTRY (Thread-Safe)
# ─────────────────────────────────────────────────────────────────────────────
# Registry of all loaded model instances. Uses a lock to ensure thread safety
# when multiple threads access models concurrently (e.g., GUI + API server).

_LOADED_MODELS: dict[str, Enigma] = {}
_MODELS_LOCK = threading.RLock()  # RLock allows re-entrant locking


def get_running_models() -> dict[str, Enigma]:
    """Get a copy of all loaded model instances (thread-safe)."""
    with _MODELS_LOCK:
        return _LOADED_MODELS.copy()


def is_model_loaded(name: str) -> bool:
    """Check if a model is loaded by name (thread-safe)."""
    with _MODELS_LOCK:
        return name in _LOADED_MODELS


def register_model(name: str, model: Enigma) -> None:
    """Register a model instance (thread-safe)."""
    with _MODELS_LOCK:
        _LOADED_MODELS[name] = model
        logger.debug(f"Registered model: {name}")


def unregister_model(name: str) -> Optional[Enigma]:
    """Unregister a model and return it if found (thread-safe)."""
    with _MODELS_LOCK:
        model = _LOADED_MODELS.pop(name, None)
        if model is not None:
            logger.debug(f"Unregistered model: {name}")
        return model


def get_model(name: str) -> Optional[Enigma]:
    """Get a specific model by name (thread-safe)."""
    with _MODELS_LOCK:
        return _LOADED_MODELS.get(name)


# =============================================================================
# 🔄 REPETITION PENALTY HELPER - Efficient penalty application
# =============================================================================


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_tokens: torch.Tensor,
    penalty: float,
    window: int = 128,
) -> torch.Tensor:
    """
    Apply repetition penalty to logits based on recently generated tokens.

    Only the last *window* tokens are considered to avoid over-suppressing
    the vocabulary during long-form generation.

    Args:
        logits: Logits tensor [batch, vocab_size] or [vocab_size]
        generated_tokens: Previously generated token IDs
        penalty: Penalty factor (>1.0 reduces repetition, 1.0 = no penalty)
        window: Number of recent tokens to consider (default 128)

    Returns:
        Modified logits with repetition penalty applied (new tensor, not in-place)
    """
    if penalty == 1.0:
        return logits.clone()

    if penalty < 1.0:
        logger.warning(
            "Repetition penalty %.3f < 1.0 amplifies repeated tokens instead of suppressing them",
            penalty,
        )

    # Clone to avoid in-place mutation (important for beam search, speculative decoding)
    logits = logits.clone()

    vocab_size = logits.shape[-1]

    # Window the tokens — only consider recent history
    if generated_tokens.dim() >= 2:
        tokens = generated_tokens[:, -window:]
    else:
        tokens = generated_tokens[-window:]

    seq_len = tokens.numel()

    if seq_len < 1000:
        # Set-based for short sequences (lower overhead)
        unique_tokens = set(tokens.view(-1).tolist())
        for token_id in unique_tokens:
            if 0 <= token_id < vocab_size:
                if logits.dim() == 1:
                    score = logits[token_id]
                    logits[token_id] = score / penalty if score > 0 else score * penalty
                else:
                    scores = logits[..., token_id]
                    logits[..., token_id] = torch.where(scores > 0, scores / penalty, scores * penalty)
    else:
        # Bincount for longer sequences (better vectorization)
        flat_tokens = tokens.view(-1)
        valid_mask = (flat_tokens >= 0) & (flat_tokens < vocab_size)
        valid_tokens = flat_tokens[valid_mask]
        token_counts = torch.bincount(valid_tokens, minlength=vocab_size)
        appeared_mask = token_counts > 0
        if logits.dim() == 1:
            scores = logits[appeared_mask]
            logits[appeared_mask] = torch.where(scores > 0, scores / penalty, scores * penalty)
        else:
            scores = logits[..., appeared_mask]
            logits[..., appeared_mask] = torch.where(scores > 0, scores / penalty, scores * penalty)

    return logits


def sample_next_token(
    logits: torch.Tensor,
    generated_tokens: torch.Tensor,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    min_p: float = 0.0,
) -> torch.Tensor:
    """
    Sample one token from logits with repetition penalty, top-k, top-p, and min-p.

    Shared helper used by :meth:`Enigma.generate` and
    :meth:`Enigma.generate_stream` so the sampling logic lives in one place.

    Args:
        logits: Raw logits for the last position [batch, vocab_size].
        generated_tokens: All tokens generated so far [batch, seq_len].
        temperature: Sampling temperature (>0).
        top_k: Keep only the top-k highest-probability tokens (0 = disabled).
        top_p: Nucleus sampling threshold (<1.0 to enable).
        repetition_penalty: Penalty for previously seen tokens (1.0 = off).
        min_p: Min-p filtering threshold (0 = disabled). Removes tokens
            whose probability is below ``min_p * max_probability``.

    Returns:
        Sampled token IDs [batch, 1].
    """
    # --- Greedy decode for temperature <= 0 ---
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)

    # --- Repetition penalty on raw logits first (order matters) ---
    if repetition_penalty != 1.0:
        logits = apply_repetition_penalty(logits, generated_tokens, repetition_penalty)

    next_logits = logits / temperature

    # Save pre-filter logits for NaN fallback (S720)
    pre_filter_logits = next_logits.clone()

    # --- Min-p filtering: remove tokens below min_p * max_probability ---
    if min_p > 0.0:
        probs_for_filter = F.softmax(next_logits, dim=-1)
        max_prob = probs_for_filter.max(dim=-1, keepdim=True).values
        next_logits = next_logits.masked_fill(probs_for_filter < min_p * max_prob, float("-inf"))

    # --- Top-k filtering ---
    if top_k > 0:
        v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
        next_logits[next_logits < v[:, [-1]]] = float("-inf")

    # --- Top-p (nucleus) filtering ---
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
        cumsum = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        mask = cumsum > top_p
        mask[:, 1:] = mask[:, :-1].clone()
        mask[:, 0] = False
        indices_to_remove = mask.scatter(1, sorted_idx, mask)
        next_logits[indices_to_remove] = float("-inf")

    # --- Sample ---
    probs = F.softmax(next_logits, dim=-1)
    # Guard: if all logits were -inf, softmax produces NaN.
    # Fall back to pre-filter distribution (S720).
    if torch.isnan(probs).any():
        probs = F.softmax(pre_filter_logits, dim=-1)
        # Second-level guard: if pre-filter logits were also
        # all -inf (model produced garbage), uniform sample.
        if torch.isnan(probs).any():
            probs = torch.ones_like(probs) / probs.shape[-1]
    return torch.multinomial(probs, num_samples=1)


# =============================================================================
# 🔧 HARDWARE DETECTION HELPERS
# =============================================================================


def detect_hardware() -> dict[str, Any]:
    """
    Detect hardware capabilities for model configuration.

    Returns dict with: total_ram_gb, gpu_vram_gb, is_raspberry_pi, is_arm, etc.
    """
    try:
        import dataclasses
        from .hardware_detection import detect_hardware as _detect

        profile = _detect()
        return dataclasses.asdict(profile)
    except ImportError:
        # Fallback basic detection
        ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3) if hasattr(os, "sysconf") else 4.0
        return {
            "total_ram_gb": ram_gb,
            "is_raspberry_pi": False,
            "has_cuda": torch.cuda.is_available(),
            "recommended_model_size": "small",
        }


def recommend_model_size(hardware: Optional[dict[str, Any]] = None) -> str:
    """
    Recommend optimal model size based on hardware.

    Args:
        hardware: Hardware profile dict (from detect_hardware). If None, auto-detects.

    Returns:
        Recommended preset name (e.g., "pi_5", "small", "medium")
    """
    if hardware is None:
        hardware = detect_hardware()

    return hardware.get("recommended_model_size", "small")


def estimate_memory_usage(size: str, quantization: str = "none") -> dict[str, float]:
    """
    Estimate RAM/VRAM requirements for a model configuration.

    Args:
        size: Model size preset name
        quantization: Quantization type

    Returns:
        Dict with model_size_mb, inference_ram_mb, training_ram_mb
    """
    try:
        from .hardware_detection import estimate_memory_usage as _estimate

        use_half = quantization in ("dynamic", "int8", "int4")
        result = _estimate(size, use_half=use_half)
        # Apply quantization scaling to model_memory
        if quantization == "int4":
            result["model_memory"] *= 0.5
        elif quantization == "int8":
            result["model_memory"] *= 0.5 if use_half else 0.25
        result["total"] = result["model_memory"] + result["kv_cache"]
        return {
            "model_size_mb": result["model_memory"] * 1024,
            "inference_ram_mb": result["total"] * 1024,
            "training_ram_mb": result["total"] * 1024 * 3,
        }
    except ImportError:
        # Fallback estimation
        param_counts = {
            "pi_zero": 0.5,
            "pi_4": 3,
            "pi_5": 8,
            "nano": 1,
            "micro": 2,
            "tiny": 5,
            "mini": 10,
            "small": 27,
            "medium": 85,
            "large": 200,
        }
        params_m = param_counts.get(size, 27)
        multiplier = {"none": 4, "dynamic": 1.5, "int8": 1, "int4": 0.5}.get(quantization, 4)
        model_mb = params_m * multiplier
        return {"model_size_mb": model_mb, "inference_ram_mb": model_mb * 2.5, "training_ram_mb": model_mb * 5}


# ─────────────────────────────────────────────────────────────────────────────
# VOCAB / EMBEDDING COMPATIBILITY HELPERS
# ─────────────────────────────────────────────────────────────────────────────
# Shared between AppState.load_model() (load-time gate) and
# EnigmaEngine._encode_prompt() (per-prompt gate) to enforce the same
# tokenizer/model contract. See AA code maker.md §4 "Token-ID bounds must
# be validated before tensors hit CUDA."


def get_model_vocab_limit(model: Any) -> Optional[int]:
    """Return the model's token-id upper bound (exclusive), if known.

    Preference order:
    1. ``model.tok_embeddings.num_embeddings`` (actual embedding rows; this
       is what CUDA reads during embedding lookup)
    2. ``model.config.vocab_size`` (logical config value; fallback for
       models that don't expose ``tok_embeddings`` directly)

    Returns ``None`` when neither attribute is present — caller must
    treat that as "skip the check" rather than "limit = 0".
    """
    if model is None:
        return None

    tok_embeddings = getattr(model, "tok_embeddings", None)
    if tok_embeddings is not None:
        num_embeddings = getattr(tok_embeddings, "num_embeddings", None)
        if isinstance(num_embeddings, int) and num_embeddings > 0:
            return num_embeddings

    cfg = getattr(model, "config", None)
    vocab_size = getattr(cfg, "vocab_size", None)
    if isinstance(vocab_size, int) and vocab_size > 0:
        return vocab_size

    return None


def get_tokenizer_vocab_size(tokenizer: Any) -> Optional[int]:
    """Return tokenizer vocab size, if exposed.

    Tries ``tokenizer.vocab_size`` first (attribute on most HF / Enigma
    tokenizers), then falls back to a ``get_vocab_size()`` callable
    (Rust BPE / SentencePiece wrappers). Returns ``None`` when neither
    is available.
    """
    if tokenizer is None:
        return None

    vocab_size = getattr(tokenizer, "vocab_size", None)
    if isinstance(vocab_size, int) and vocab_size > 0:
        return vocab_size

    getter = getattr(tokenizer, "get_vocab_size", None)
    if callable(getter):
        try:
            size = getter()
        except Exception:
            return None
        if isinstance(size, int) and size > 0:
            return size

    return None
