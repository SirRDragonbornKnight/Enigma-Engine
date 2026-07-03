"""SLERP and TIES model merging for Enigma checkpoints.

Merges two or more checkpoints with identical architectures into a
single model.  Supports:
  - SLERP (Spherical Linear Interpolation) for two models
  - TIES (Trim, Elect Sign, Disjoint Merge) for two or more models
  - Linear interpolation as a fast baseline
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Sequence

import torch

from .model_presets import ForgeConfig
from .model_registry import get_state_dict, safe_load_weights
from .safe_save import atomic_torch_save

logger = logging.getLogger(__name__)

# ── Architecture fields that MUST match between merged models ──────
_ARCH_FIELDS = (
    "dim",
    "n_layers",
    "n_heads",
    "n_kv_heads",
    "hidden_dim",
    "vocab_size",
    "max_seq_len",
)

# Keys to skip during merging (runtime-only / recomputed)
_SKIP_KEYS = {"freqs_cis"}


# ── Helpers ────────────────────────────────────────────────────────


def _validate_compatible(configs: list[ForgeConfig]) -> None:
    """Raise if any architecture dimension differs."""
    base = configs[0]
    for i, cfg in enumerate(configs[1:], 1):
        for field in _ARCH_FIELDS:
            v_base = getattr(base, field)
            v_other = getattr(cfg, field)
            if v_base != v_other:
                raise ValueError(f"Architecture mismatch on '{field}': model 0 has {v_base}, model {i} has {v_other}")


def _load_checkpoint(path: str | Path, device: str = "cpu"):
    """Return (state_dict, ForgeConfig) from a checkpoint file."""
    ckpt = safe_load_weights(str(path), map_location=device)
    cfg_dict = ckpt.get("model_config", ckpt.get("config", {}))
    if isinstance(cfg_dict, dict) and "epochs" in cfg_dict:
        cfg_dict = ckpt.get("model_config", {})
    config = ForgeConfig(**{k: v for k, v in cfg_dict.items() if k in ForgeConfig.__dataclass_fields__})
    sd = get_state_dict(ckpt)
    return sd, config


def _filter_keys(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove non-parameter keys that should not be merged."""
    return {k: v for k, v in sd.items() if k not in _SKIP_KEYS}


# ── SLERP ──────────────────────────────────────────────────────────


def _slerp_tensor(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical linear interpolation between flat tensors."""
    a_flat = a.float().flatten()
    b_flat = b.float().flatten()

    a_norm = torch.nn.functional.normalize(a_flat, dim=0)
    b_norm = torch.nn.functional.normalize(b_flat, dim=0)

    dot = torch.clamp(torch.dot(a_norm, b_norm), -1.0, 1.0)
    omega = torch.acos(dot)

    # Degenerate case — vectors are (anti-)parallel
    if omega.abs() < 1e-6:
        return ((1.0 - t) * a + t * b).to(a.dtype)

    sin_omega = torch.sin(omega)
    coeff_a = torch.sin((1.0 - t) * omega) / sin_omega
    coeff_b = torch.sin(t * omega) / sin_omega

    merged = coeff_a * a_flat + coeff_b * b_flat
    return merged.reshape(a.shape).to(a.dtype)


def slerp_merge(
    path_a: str | Path,
    path_b: str | Path,
    t: float = 0.5,
    output_path: str | Path | None = None,
    device: str = "cpu",
    on_progress: Callable | None = None,
) -> dict:
    """Merge two checkpoints via SLERP.

    Parameters
    ----------
    path_a, path_b : paths to checkpoint files
    t : interpolation factor (0.0 = all A, 1.0 = all B)
    output_path : if given, save merged checkpoint here
    device : device for computation
    on_progress : optional callback(percent: int, message: str)

    Returns
    -------
    dict with 'model_state_dict' and 'config'
    """
    if not 0.0 <= t <= 1.0:
        raise ValueError(f"t must be in [0, 1], got {t}")

    sd_a, cfg_a = _load_checkpoint(path_a, device)
    sd_b, cfg_b = _load_checkpoint(path_b, device)
    _validate_compatible([cfg_a, cfg_b])

    sd_a = _filter_keys(sd_a)
    sd_b = _filter_keys(sd_b)

    keys = sorted(set(sd_a) & set(sd_b))
    merged = {}
    total = len(keys)

    for i, key in enumerate(keys):
        merged[key] = _slerp_tensor(sd_a[key], sd_b[key], t)
        if on_progress and (i % max(1, total // 20) == 0 or i == total - 1):
            pct = int((i + 1) / total * 100)
            on_progress(pct, f"SLERP {pct}% ({i + 1}/{total} tensors)")

    result = {
        "model_state_dict": merged,
        "config": cfg_a.to_dict(),
    }

    if output_path:
        atomic_torch_save(result, str(output_path))
        logger.info("SLERP merge saved to %s", output_path)

    return result


# ── TIES ───────────────────────────────────────────────────────────


def ties_merge(
    paths: Sequence[str | Path],
    base_path: str | Path | None = None,
    density: float = 0.2,
    weights: Sequence[float] | None = None,
    output_path: str | Path | None = None,
    device: str = "cpu",
    on_progress: Callable | None = None,
) -> dict:
    """Merge checkpoints via TIES (Trim, Elect Sign, Disjoint Merge).

    Parameters
    ----------
    paths : checkpoint paths to merge (at least 2)
    base_path : optional base model; task vectors are computed relative
                to this.  If None, uses the first path as base.
    density : fraction of top-magnitude deltas to keep (0 < d <= 1)
    weights : per-model weights (uniform if None)
    output_path : if given, save merged checkpoint here
    on_progress : optional callback(percent: int, message: str)

    Returns
    -------
    dict with 'model_state_dict' and 'config'
    """
    if len(paths) < 2:
        raise ValueError("TIES requires at least 2 models")
    if not 0.0 < density <= 1.0:
        raise ValueError(f"density must be in (0, 1], got {density}")

    if weights is None:
        weights = [1.0 / len(paths)] * len(paths)
    if len(weights) != len(paths):
        raise ValueError("len(weights) must match len(paths)")

    # Load base
    if base_path is not None:
        sd_base, cfg_base = _load_checkpoint(base_path, device)
    else:
        sd_base, cfg_base = _load_checkpoint(paths[0], device)

    sd_base = _filter_keys(sd_base)

    # Load all models
    all_sd = []
    all_cfg = [cfg_base]
    for p in paths:
        sd, cfg = _load_checkpoint(p, device)
        all_sd.append(_filter_keys(sd))
        all_cfg.append(cfg)
    _validate_compatible(all_cfg)

    keys = sorted(set(sd_base) & set.intersection(*(set(sd) for sd in all_sd)))
    merged = {}
    total = len(keys)

    for ki, key in enumerate(keys):
        base_tensor = sd_base[key].float()

        # Step 1: Compute task vectors (deltas from base)
        deltas = []
        for sd in all_sd:
            deltas.append(sd[key].float() - base_tensor)

        # Step 2: Trim — keep only top-density magnitudes per delta
        trimmed = []
        for delta in deltas:
            flat = delta.flatten()
            k = max(1, int(density * flat.numel()))
            threshold = flat.abs().topk(k).values[-1]
            mask = flat.abs() >= threshold
            trimmed.append((flat * mask.float()).reshape(delta.shape))

        # Step 3: Elect sign — majority vote on sign per element
        stacked = torch.stack(trimmed)
        sign_votes = torch.sign(stacked).sum(dim=0)
        elected_sign = torch.sign(sign_votes)

        # Step 4: Disjoint merge — average matching-sign values
        weighted_sum = torch.zeros_like(base_tensor)
        weight_total = torch.zeros_like(base_tensor)
        for j, trimmed_delta in enumerate(trimmed):
            agree = (torch.sign(trimmed_delta) == elected_sign).float()
            agree *= (trimmed_delta != 0).float()
            weighted_sum += weights[j] * trimmed_delta * agree
            weight_total += weights[j] * agree

        safe_denom = weight_total.clamp(min=1e-8)
        merged_delta = weighted_sum / safe_denom
        merged_delta *= (weight_total > 0).float()

        merged[key] = (base_tensor + merged_delta).to(sd_base[key].dtype)

        if on_progress and (ki % max(1, total // 20) == 0 or ki == total - 1):
            pct = int((ki + 1) / total * 100)
            on_progress(pct, f"TIES {pct}% ({ki + 1}/{total} tensors)")

    result = {
        "model_state_dict": merged,
        "config": cfg_base.to_dict(),
    }

    if output_path:
        atomic_torch_save(result, str(output_path))
        logger.info("TIES merge saved to %s", output_path)

    return result


# ── Linear interpolation ──────────────────────────────────────────


def linear_merge(
    path_a: str | Path,
    path_b: str | Path,
    t: float = 0.5,
    output_path: str | Path | None = None,
    device: str = "cpu",
    on_progress: Callable | None = None,
) -> dict:
    """Merge two checkpoints via linear interpolation.

    Parameters
    ----------
    path_a, path_b : paths to checkpoint files
    t : interpolation factor (0.0 = all A, 1.0 = all B)
    output_path : if given, save merged checkpoint here
    on_progress : optional callback(percent: int, message: str)

    Returns
    -------
    dict with 'model_state_dict' and 'config'
    """
    if not 0.0 <= t <= 1.0:
        raise ValueError(f"t must be in [0, 1], got {t}")

    sd_a, cfg_a = _load_checkpoint(path_a, device)
    sd_b, cfg_b = _load_checkpoint(path_b, device)
    _validate_compatible([cfg_a, cfg_b])

    sd_a = _filter_keys(sd_a)
    sd_b = _filter_keys(sd_b)

    keys = sorted(set(sd_a) & set(sd_b))
    merged = {}
    total = len(keys)

    for i, key in enumerate(keys):
        a = sd_a[key].float()
        b = sd_b[key].float()
        merged[key] = ((1.0 - t) * a + t * b).to(sd_a[key].dtype)
        if on_progress and (i % max(1, total // 20) == 0 or i == total - 1):
            pct = int((i + 1) / total * 100)
            on_progress(pct, f"Linear {pct}% ({i + 1}/{total} tensors)")

    result = {
        "model_state_dict": merged,
        "config": cfg_a.to_dict(),
    }

    if output_path:
        atomic_torch_save(result, str(output_path))
        logger.info("Linear merge saved to %s", output_path)

    return result
