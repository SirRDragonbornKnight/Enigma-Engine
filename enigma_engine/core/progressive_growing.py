"""
Progressive Model Growing — expand a trained model to a larger architecture.

Net2Net-inspired weight expansion:
- Width expansion (dim, heads, hidden_dim): zero-pad new dimensions
- Depth expansion (n_layers): spread old layers, identity-init new ones

Depth growth is output-preserving: identity-initialized layers zero their
attention/FFN outputs, so the residual stream passes through unchanged and
the expanded model reproduces the source model's output on day 1.

Width growth is a warm start, NOT output-preserving: zero-padding the extra
dimensions changes the RMSNorm statistics (RMS is taken over the padded dim,
scaling normalized activations by sqrt(src.dim/tgt.dim)), and growing head_dim
reassigns the RoPE frequency basis. A width-grown model needs further training
to converge.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from enigma_engine.core.model_presets import ForgeConfig

logger = logging.getLogger(__name__)


def validate_growth(source_config: ForgeConfig, target_config: ForgeConfig) -> None:
    """Validate that the target config is a valid growth target.

    Raises ValueError if the expansion is invalid.
    """
    # Target must be >= source in all dimensional parameters
    if target_config.dim < source_config.dim:
        raise ValueError(f"Target dim ({target_config.dim}) must be >= source dim ({source_config.dim})")
    if target_config.n_layers < source_config.n_layers:
        raise ValueError(
            f"Target n_layers ({target_config.n_layers}) must be >= source n_layers ({source_config.n_layers})"
        )
    if target_config.n_heads < source_config.n_heads:
        raise ValueError(
            f"Target n_heads ({target_config.n_heads}) must be >= source n_heads ({source_config.n_heads})"
        )
    if target_config.n_kv_heads < source_config.n_kv_heads:
        raise ValueError(
            f"Target n_kv_heads ({target_config.n_kv_heads}) must be >= source n_kv_heads ({source_config.n_kv_heads})"
        )
    if target_config.hidden_dim < source_config.hidden_dim:
        raise ValueError(
            f"Target hidden_dim ({target_config.hidden_dim}) must be >= source hidden_dim ({source_config.hidden_dim})"
        )

    # Architecture flags must match (can't change activation type mid-grow)
    if target_config.use_swiglu != source_config.use_swiglu:
        raise ValueError("Cannot change use_swiglu during progressive growing")
    if target_config.use_rope != source_config.use_rope:
        raise ValueError("Cannot change use_rope during progressive growing")
    if target_config.use_rms_norm != source_config.use_rms_norm:
        raise ValueError("Cannot change use_rms_norm during progressive growing")
    if target_config.use_bias != source_config.use_bias:
        raise ValueError("Cannot change use_bias during progressive growing")

    # MoE config must match
    if target_config.use_moe != source_config.use_moe:
        raise ValueError("Cannot change MoE configuration during progressive growing")


def compute_layer_mapping(old_layers: int, new_layers: int) -> list[int]:
    """Compute how old layers map into new layer positions.

    Strategy: evenly spread old layers across the new depth.
    Positions without a direct source get -1 (identity initialization).

    Returns:
        List of length new_layers. Each entry is the source layer index
        (0..old_layers-1) or -1 for identity-initialized layers.

    Example:
        compute_layer_mapping(2, 6) -> [0, -1, -1, 1, -1, -1]
        Old layer 0 goes to position 0, old layer 1 to position 3.
        Positions 1, 2, 4, 5 are identity-initialized.
    """
    if new_layers == old_layers:
        return list(range(old_layers))

    mapping = [-1] * new_layers

    # Place old layers at evenly spaced positions
    for old_idx in range(old_layers):
        # Map old_idx to proportional position in new depth
        new_idx = (old_idx * new_layers) // old_layers
        mapping[new_idx] = old_idx

    return mapping


def _expand_2d(old: "torch.Tensor", new_rows: int, new_cols: int, fill: float = 0.0) -> "torch.Tensor":
    """Expand a 2D weight tensor, preserving old values in top-left corner."""
    import torch

    old_rows, old_cols = old.shape
    if new_rows < old_rows or new_cols < old_cols:
        raise ValueError(f"Cannot shrink tensor from ({old_rows}, {old_cols}) to ({new_rows}, {new_cols})")
    new = torch.full((new_rows, new_cols), fill, dtype=old.dtype)
    new[:old_rows, :old_cols] = old
    return new


def _expand_1d(old: "torch.Tensor", new_size: int, fill: float = 0.0) -> "torch.Tensor":
    """Expand a 1D tensor (bias or norm weight), preserving old values."""
    import torch

    new = torch.full((new_size,), fill, dtype=old.dtype)
    new[: old.shape[0]] = old
    return new


def expand_model_weights(
    source_sd: dict[str, "torch.Tensor"],
    source_config: ForgeConfig,
    target_config: ForgeConfig,
) -> dict[str, "torch.Tensor"]:
    """Expand a smaller model's state dict to fit a larger architecture.

    Uses Net2Net-inspired zero-initialization:
    - Old weights are copied into their corresponding positions
    - New dimensions are zero-initialized (a width warm start: zero-padding
      shifts RMSNorm statistics and head_dim growth reassigns RoPE
      frequencies, so width growth is not output-preserving)
    - New layers use identity initialization (zero residual output, which
      IS output-preserving for depth growth)

    Args:
        source_sd: State dict from the smaller model.
        source_config: ForgeConfig of the source model.
        target_config: ForgeConfig of the target (larger) model.

    Returns:
        New state dict compatible with the target architecture.

    Raises:
        ValueError: If the expansion is invalid.
    """
    import torch

    validate_growth(source_config, target_config)

    src = source_config
    tgt = target_config

    # Vocab padding (matches Enigma model convention)
    src_padded_vocab = (src.vocab_size + 63) & ~63
    tgt_padded_vocab = (tgt.vocab_size + 63) & ~63

    src_head_dim = src.dim // src.n_heads
    tgt_head_dim = tgt.dim // tgt.n_heads

    new_sd: dict[str, torch.Tensor] = {}

    # ── Global tensors ────────────────────────────────────────────────

    # Embeddings: [padded_vocab, dim]
    old_emb = source_sd.get("tok_embeddings.weight")
    if old_emb is None:
        raise ValueError("State dict missing 'tok_embeddings.weight' -- cannot expand model weights")
    new_emb = torch.zeros(tgt_padded_vocab, tgt.dim, dtype=old_emb.dtype)
    copy_rows = min(src_padded_vocab, tgt_padded_vocab)
    new_emb[:copy_rows, : src.dim] = old_emb[:copy_rows]
    new_sd["tok_embeddings.weight"] = new_emb
    # Output is weight-tied to embeddings
    new_sd["output.weight"] = new_emb

    # Final norm: [dim]
    if "norm.weight" in source_sd:
        new_sd["norm.weight"] = _expand_1d(source_sd["norm.weight"], tgt.dim, fill=1.0)
    if "norm.bias" in source_sd:
        new_sd["norm.bias"] = _expand_1d(source_sd["norm.bias"], tgt.dim, fill=0.0)

    # ── Layer mapping ─────────────────────────────────────────────────

    layer_mapping = compute_layer_mapping(src.n_layers, tgt.n_layers)

    logger.info(
        "Progressive growing: %d→%d layers, %d→%d dim, %d→%d heads, %d→%d kv_heads, %d→%d hidden_dim",
        src.n_layers,
        tgt.n_layers,
        src.dim,
        tgt.dim,
        src.n_heads,
        tgt.n_heads,
        src.n_kv_heads,
        tgt.n_kv_heads,
        src.hidden_dim,
        tgt.hidden_dim,
    )
    logger.info("Layer mapping: %s", layer_mapping)

    # ── Per-layer expansion ───────────────────────────────────────────

    for new_idx, src_idx in enumerate(layer_mapping):
        tp = f"layers.{new_idx}"

        if src_idx >= 0:
            # Expand an existing source layer
            sp = f"layers.{src_idx}"
            _expand_layer(
                source_sd,
                new_sd,
                sp,
                tp,
                src,
                tgt,
                src_head_dim,
                tgt_head_dim,
            )
        else:
            # Identity-initialized layer (zero residual outputs)
            _init_identity_layer(new_sd, tp, tgt, tgt_head_dim, device=old_emb.device, dtype=old_emb.dtype)

    # ── Multi-token prediction heads (R25) ────────────────────────────

    if tgt.n_predict_heads > 0:
        for h_idx in range(tgt.n_predict_heads):
            src_key = f"predict_heads.{h_idx}.weight"
            tgt_key = f"predict_heads.{h_idx}.weight"
            if src_key in source_sd:
                # Expand existing head: [padded_vocab, dim]
                old_hw = source_sd[src_key]
                new_hw = torch.zeros(tgt_padded_vocab, tgt.dim, dtype=old_hw.dtype)
                copy_rows = min(src_padded_vocab, tgt_padded_vocab)
                new_hw[:copy_rows, : src.dim] = old_hw[:copy_rows]
                new_sd[tgt_key] = new_hw
            else:
                # New head — small random init
                new_sd[tgt_key] = torch.randn(tgt_padded_vocab, tgt.dim, dtype=old_emb.dtype) * 0.02

    return new_sd


def _expand_layer(
    source_sd: dict,
    new_sd: dict,
    sp: str,
    tp: str,
    src: ForgeConfig,
    tgt: ForgeConfig,
    src_head_dim: int,
    tgt_head_dim: int,
) -> None:
    """Expand one source layer's weights into a target layer position."""
    # ── Norms: [dim] ──────────────────────────────────────────────
    for suffix in ("attention_norm.weight", "ffn_norm.weight"):
        if f"{sp}.{suffix}" in source_sd:
            new_sd[f"{tp}.{suffix}"] = _expand_1d(source_sd[f"{sp}.{suffix}"], tgt.dim, fill=1.0)
    for suffix in ("attention_norm.bias", "ffn_norm.bias"):
        if f"{sp}.{suffix}" in source_sd:
            new_sd[f"{tp}.{suffix}"] = _expand_1d(source_sd[f"{sp}.{suffix}"], tgt.dim, fill=0.0)

    # ── Attention projections ─────────────────────────────────────

    # Wq: [n_heads * head_dim, dim] -> [new_n_heads * new_head_dim, new_dim]
    _expand_attn_proj(
        source_sd,
        new_sd,
        f"{sp}.attention.wq",
        f"{tp}.attention.wq",
        src.n_heads,
        tgt.n_heads,
        src_head_dim,
        tgt_head_dim,
        src.dim,
        tgt.dim,
    )

    # Wk: [n_kv_heads * head_dim, dim]
    _expand_attn_proj(
        source_sd,
        new_sd,
        f"{sp}.attention.wk",
        f"{tp}.attention.wk",
        src.n_kv_heads,
        tgt.n_kv_heads,
        src_head_dim,
        tgt_head_dim,
        src.dim,
        tgt.dim,
    )

    # Wv: [n_kv_heads * head_dim, dim]
    _expand_attn_proj(
        source_sd,
        new_sd,
        f"{sp}.attention.wv",
        f"{tp}.attention.wv",
        src.n_kv_heads,
        tgt.n_kv_heads,
        src_head_dim,
        tgt_head_dim,
        src.dim,
        tgt.dim,
    )

    # Wo: [dim, n_heads * head_dim] — transposed shape
    _expand_attn_proj_output(
        source_sd,
        new_sd,
        f"{sp}.attention.wo",
        f"{tp}.attention.wo",
        src.n_heads,
        tgt.n_heads,
        src_head_dim,
        tgt_head_dim,
        src.dim,
        tgt.dim,
    )

    # QK norm (per-head): [head_dim]
    for suffix in ("attention.q_norm.weight", "attention.k_norm.weight"):
        if f"{sp}.{suffix}" in source_sd:
            new_sd[f"{tp}.{suffix}"] = _expand_1d(source_sd[f"{sp}.{suffix}"], tgt_head_dim, fill=1.0)

    # R22: Differential attention _diff_lambda: [n_heads // 2]
    diff_key = f"{sp}.attention._diff_lambda"
    if diff_key in source_sd:
        tgt_half = tgt.n_heads // 2
        new_sd[f"{tp}.attention._diff_lambda"] = _expand_1d(source_sd[diff_key], tgt_half, fill=0.05)

    # LayerScale: [dim]
    for suffix in ("ls_attn", "ls_ffn"):
        if f"{sp}.{suffix}" in source_sd:
            new_sd[f"{tp}.{suffix}"] = _expand_1d(source_sd[f"{sp}.{suffix}"], tgt.dim, fill=1e-5)

    # ── Feed-forward ──────────────────────────────────────────────

    if tgt.use_swiglu:
        # SwiGLU: w1 [hidden, dim], w2 [dim, hidden], w3 [hidden, dim]
        for key in ("feed_forward.w1.weight", "feed_forward.w3.weight"):
            if f"{sp}.{key}" in source_sd:
                new_sd[f"{tp}.{key}"] = _expand_2d(source_sd[f"{sp}.{key}"], tgt.hidden_dim, tgt.dim)
        if f"{sp}.feed_forward.w2.weight" in source_sd:
            new_sd[f"{tp}.feed_forward.w2.weight"] = _expand_2d(
                source_sd[f"{sp}.feed_forward.w2.weight"], tgt.dim, tgt.hidden_dim
            )

        # Biases
        for key in ("feed_forward.w1.bias", "feed_forward.w3.bias"):
            if f"{sp}.{key}" in source_sd:
                new_sd[f"{tp}.{key}"] = _expand_1d(source_sd[f"{sp}.{key}"], tgt.hidden_dim)
        if f"{sp}.feed_forward.w2.bias" in source_sd:
            new_sd[f"{tp}.feed_forward.w2.bias"] = _expand_1d(source_sd[f"{sp}.feed_forward.w2.bias"], tgt.dim)
    else:
        # Standard FFN: up [hidden, dim], down [dim, hidden]
        if f"{sp}.feed_forward.up.weight" in source_sd:
            new_sd[f"{tp}.feed_forward.up.weight"] = _expand_2d(
                source_sd[f"{sp}.feed_forward.up.weight"], tgt.hidden_dim, tgt.dim
            )
        if f"{sp}.feed_forward.down.weight" in source_sd:
            new_sd[f"{tp}.feed_forward.down.weight"] = _expand_2d(
                source_sd[f"{sp}.feed_forward.down.weight"], tgt.dim, tgt.hidden_dim
            )

        for key in ("feed_forward.up.bias",):
            if f"{sp}.{key}" in source_sd:
                new_sd[f"{tp}.{key}"] = _expand_1d(source_sd[f"{sp}.{key}"], tgt.hidden_dim)
        if f"{sp}.feed_forward.down.bias" in source_sd:
            new_sd[f"{tp}.feed_forward.down.bias"] = _expand_1d(source_sd[f"{sp}.feed_forward.down.bias"], tgt.dim)


def _expand_attn_proj(
    source_sd: dict,
    new_sd: dict,
    src_key: str,
    tgt_key: str,
    src_n: int,
    tgt_n: int,
    src_hd: int,
    tgt_hd: int,
    src_dim: int,
    tgt_dim: int,
) -> None:
    """Expand Q/K/V projection weight [n*head_dim, dim]."""
    import torch

    weight_key = f"{src_key}.weight"
    if weight_key not in source_sd:
        logger.debug("Skipping expansion: %s not in source", weight_key)
        return

    old_w = source_sd[weight_key]
    new_w = torch.zeros(tgt_n * tgt_hd, tgt_dim, dtype=old_w.dtype)

    # Copy head-by-head (handles head_dim changes correctly)
    copy_heads = min(src_n, tgt_n)
    copy_hd = min(src_hd, tgt_hd)
    for h in range(copy_heads):
        new_w[h * tgt_hd : h * tgt_hd + copy_hd, :src_dim] = old_w[h * src_hd : h * src_hd + copy_hd]

    new_sd[f"{tgt_key}.weight"] = new_w

    # Bias: [n * head_dim]
    bias_key = f"{src_key}.bias"
    if bias_key in source_sd:
        old_b = source_sd[bias_key]
        new_b = torch.zeros(tgt_n * tgt_hd, dtype=old_b.dtype)
        for h in range(copy_heads):
            new_b[h * tgt_hd : h * tgt_hd + copy_hd] = old_b[h * src_hd : h * src_hd + copy_hd]
        new_sd[f"{tgt_key}.bias"] = new_b


def _expand_attn_proj_output(
    source_sd: dict,
    new_sd: dict,
    src_key: str,
    tgt_key: str,
    src_n: int,
    tgt_n: int,
    src_hd: int,
    tgt_hd: int,
    src_dim: int,
    tgt_dim: int,
) -> None:
    """Expand output projection weight [dim, n*head_dim]."""
    import torch

    weight_key = f"{src_key}.weight"
    if weight_key not in source_sd:
        logger.debug("Skipping expansion: %s not in source", weight_key)
        return

    old_w = source_sd[weight_key]  # [src_dim, src_n * src_hd]
    new_w = torch.zeros(tgt_dim, tgt_n * tgt_hd, dtype=old_w.dtype)

    copy_heads = min(src_n, tgt_n)
    copy_hd = min(src_hd, tgt_hd)
    for h in range(copy_heads):
        new_w[:src_dim, h * tgt_hd : h * tgt_hd + copy_hd] = old_w[:, h * src_hd : h * src_hd + copy_hd]

    new_sd[f"{tgt_key}.weight"] = new_w

    # Bias: [dim]
    bias_key = f"{src_key}.bias"
    if bias_key in source_sd:
        new_sd[f"{tgt_key}.bias"] = _expand_1d(source_sd[bias_key], tgt_dim)


def _init_identity_layer(
    new_sd: dict,
    tp: str,
    tgt: ForgeConfig,
    tgt_head_dim: int,
    device: str = "cpu",
    dtype: "torch.dtype | None" = None,
) -> None:
    """Initialize a new layer as identity (pass-through via residual skip).

    Attention output (wo) and FFN output (w2/down) are zeroed so the
    residual connection passes input through unchanged. Other weights
    get standard initialization.
    """
    import torch

    kw: dict = {}
    if dtype is not None:
        kw["dtype"] = dtype
    kw["device"] = device

    # Norms: weight=1 is the identity for RMSNorm/LayerNorm
    new_sd[f"{tp}.attention_norm.weight"] = torch.ones(tgt.dim, **kw)
    new_sd[f"{tp}.ffn_norm.weight"] = torch.ones(tgt.dim, **kw)

    # Attention projections: small random init for Q/K/V, zero for output
    new_sd[f"{tp}.attention.wq.weight"] = torch.randn(tgt.n_heads * tgt_head_dim, tgt.dim, **kw) * 0.02
    new_sd[f"{tp}.attention.wk.weight"] = torch.randn(tgt.n_kv_heads * tgt_head_dim, tgt.dim, **kw) * 0.02
    new_sd[f"{tp}.attention.wv.weight"] = torch.randn(tgt.n_kv_heads * tgt_head_dim, tgt.dim, **kw) * 0.02
    # Zero output → residual skip is identity
    new_sd[f"{tp}.attention.wo.weight"] = torch.zeros(tgt.dim, tgt.n_heads * tgt_head_dim, **kw)

    if tgt.use_bias:
        new_sd[f"{tp}.attention.wq.bias"] = torch.zeros(tgt.n_heads * tgt_head_dim, **kw)
        new_sd[f"{tp}.attention.wk.bias"] = torch.zeros(tgt.n_kv_heads * tgt_head_dim, **kw)
        new_sd[f"{tp}.attention.wv.bias"] = torch.zeros(tgt.n_kv_heads * tgt_head_dim, **kw)
        new_sd[f"{tp}.attention.wo.bias"] = torch.zeros(tgt.dim, **kw)

    # QK norm (identity init = weight 1.0)
    if tgt.use_qk_norm:
        new_sd[f"{tp}.attention.q_norm.weight"] = torch.ones(tgt_head_dim, **kw)
        new_sd[f"{tp}.attention.k_norm.weight"] = torch.ones(tgt_head_dim, **kw)

    # R22: Differential attention — init near zero
    if tgt.use_differential_attn and tgt.n_heads >= 2 and tgt.n_heads % 2 == 0:
        new_sd[f"{tp}.attention._diff_lambda"] = torch.full((tgt.n_heads // 2,), 0.05, **kw)

    # FFN: small random init for input projections, zero output
    if tgt.use_swiglu:
        new_sd[f"{tp}.feed_forward.w1.weight"] = torch.randn(tgt.hidden_dim, tgt.dim, **kw) * 0.02
        new_sd[f"{tp}.feed_forward.w3.weight"] = torch.randn(tgt.hidden_dim, tgt.dim, **kw) * 0.02
        # Zero output → residual skip is identity
        new_sd[f"{tp}.feed_forward.w2.weight"] = torch.zeros(tgt.dim, tgt.hidden_dim, **kw)
    else:
        new_sd[f"{tp}.feed_forward.up.weight"] = torch.randn(tgt.hidden_dim, tgt.dim, **kw) * 0.02
        new_sd[f"{tp}.feed_forward.down.weight"] = torch.zeros(tgt.dim, tgt.hidden_dim, **kw)


# =============================================================================
# GRADUAL UNFREEZING (R12, Howard & Ruder)
# =============================================================================


def setup_gradual_unfreeze(
    model: "torch.nn.Module",
    n_layers: int,
    unfreeze_schedule: list[int] | None = None,
) -> "GradualUnfreezer":
    """Set up gradual unfreezing after progressive growth.

    Freezes all transformer layers initially, then unfreezes
    them bottom-up according to a schedule.

    Args:
        model: The expanded model.
        n_layers: Total number of transformer layers.
        unfreeze_schedule: List of step counts at which to unfreeze
            the next layer group (from top to bottom).  If None,
            defaults to unfreezing one layer every 100 steps.

    Returns:
        A GradualUnfreezer that should be called each training step.
    """
    return GradualUnfreezer(model, n_layers, unfreeze_schedule)


class GradualUnfreezer:
    """Manages gradual layer unfreezing during training (R12).

    Freezes all layers initially, unfreezes top-down so the
    newest/highest layers (which need the most adaptation) train
    first, and deeper layers are progressively unlocked.
    """

    def __init__(
        self,
        model: "torch.nn.Module",
        n_layers: int,
        unfreeze_schedule: list[int] | None = None,
    ) -> None:
        self.model = model
        self.n_layers = n_layers
        # Default: unfreeze one layer every 100 steps, top-down
        if unfreeze_schedule is None:
            self.schedule = [100 * (i + 1) for i in range(n_layers)]
        else:
            self.schedule = list(unfreeze_schedule)
            if len(self.schedule) < n_layers:
                last_step = self.schedule[-1] if self.schedule else 0
                interval = (self.schedule[-1] - self.schedule[-2]) if len(self.schedule) >= 2 else 100
                while len(self.schedule) < n_layers:
                    last_step += interval
                    self.schedule.append(last_step)
                logger.warning(
                    "unfreeze_schedule had %d entries for %d layers -- padded with interval %d",
                    len(unfreeze_schedule),
                    n_layers,
                    interval,
                )
        self._layers_unfrozen = 0
        # Start with all layers frozen
        self._freeze_all_layers()
        # Always keep embeddings, output, and final norm trainable
        self._unfreeze_head()

    def _freeze_all_layers(self) -> None:
        """Freeze all transformer layer parameters."""
        for name, param in self.model.named_parameters():
            if "layers." in name:
                param.requires_grad = False

    def _unfreeze_head(self) -> None:
        """Keep non-layer params (embeddings, norm, output) trainable."""
        for name, param in self.model.named_parameters():
            if "layers." not in name:
                param.requires_grad = True

    def _unfreeze_layer(self, layer_idx: int) -> None:
        """Unfreeze a specific transformer layer."""
        prefix = f"layers.{layer_idx}."
        for name, param in self.model.named_parameters():
            if name.startswith(prefix):
                param.requires_grad = True
        logger.info("Unfroze layer %d", layer_idx)

    def step(self, global_step: int) -> None:
        """Call each training step to check if more layers should unfreeze.

        Unfreezes from top (last layer) to bottom (first layer).
        """
        while (
            self._layers_unfrozen < self.n_layers
            and self._layers_unfrozen < len(self.schedule)
            and global_step >= self.schedule[self._layers_unfrozen]
        ):
            # Unfreeze top-down: start with the last layer
            layer_to_unfreeze = self.n_layers - 1 - self._layers_unfrozen
            self._unfreeze_layer(layer_to_unfreeze)
            self._layers_unfrozen += 1

    @property
    def fully_unfrozen(self) -> bool:
        """True when all layers are trainable."""
        return self._layers_unfrozen >= self.n_layers


# =============================================================================
# LISA: LAYERWISE IMPORTANCE SAMPLED ADAMW (R26, Pan et al. 2024)
# =============================================================================


class LISAScheduler:
    """LISA: Layerwise Importance Sampled AdamW (R26, Pan et al. 2024).

    Always trains embeddings, output head, norms, first layer, and last layer.
    Each optimizer step, randomly samples K middle layers to train; the rest
    are frozen. Outperforms LoRA at the same memory budget.
    """

    def __init__(
        self,
        model: "torch.nn.Module",
        n_layers: int,
        activated_layers: int = 2,
    ) -> None:
        self.model = model
        self.n_layers = n_layers
        # Middle layers = all except first and last
        self.middle_layers = list(range(1, max(1, n_layers - 1)))
        self.activated_layers = min(activated_layers, len(self.middle_layers))
        self._active_middle: list[int] = []
        # Initial setup
        self.step()

    def step(self) -> None:
        """Resample which middle layers are trainable this step."""
        import random as _random

        # Freeze all middle layers
        for idx in self.middle_layers:
            prefix = f"layers.{idx}."
            for name, param in self.model.named_parameters():
                if name.startswith(prefix):
                    param.requires_grad = False

        # Randomly sample K middle layers to activate
        self._active_middle = []
        if self.middle_layers and self.activated_layers > 0:
            self._active_middle = _random.sample(
                self.middle_layers,
                min(self.activated_layers, len(self.middle_layers)),
            )
            for idx in self._active_middle:
                prefix = f"layers.{idx}."
                for name, param in self.model.named_parameters():
                    if name.startswith(prefix):
                        param.requires_grad = True

        # Always train first and last layer
        for idx in (0, self.n_layers - 1):
            prefix = f"layers.{idx}."
            for name, param in self.model.named_parameters():
                if name.startswith(prefix):
                    param.requires_grad = True

        # Always train non-layer params (embeddings, norm, output head)
        for name, param in self.model.named_parameters():
            if "layers." not in name:
                param.requires_grad = True


# =============================================================================
# BERT2BERT LAYER COPYING (R20, Rothe et al.)
# =============================================================================


def compute_layer_mapping_bert2bert(
    old_layers: int,
    new_layers: int,
) -> list[int]:
    """Copy-stack layer mapping: duplicate existing layers for new depth.

    Unlike zero-init identity layers, copy-stacked layers start
    functional — they produce meaningful attention and FFN outputs
    from step 0.

    Strategy: cycle through old layers, repeating them to fill
    the new depth.  This means layer i in the new model gets
    weights from layer ``i % old_layers`` of the old model.

    Args:
        old_layers: Number of layers in the source model.
        new_layers: Number of layers in the target model.

    Returns:
        List of length new_layers, each entry is a source layer index.

    Example:
        compute_layer_mapping_bert2bert(2, 6) -> [0, 1, 0, 1, 0, 1]
    """
    return [i % old_layers for i in range(new_layers)]
