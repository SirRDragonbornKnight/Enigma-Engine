"""
Enigma - Modern Transformer Language Model.

The core model class that implements a modern transformer architecture with:
- RoPE positional embeddings
- RMSNorm / LayerNorm
- SwiGLU activation
- Grouped Query Attention (GQA)
- KV cache for fast generation
- Mixture of Experts (MoE) support
- Multi-modal inputs (vision, audio)
- Universal model loading (HuggingFace, GGUF, ONNX, Safetensors)
- LoRA adapter support
- Speculative decoding
- Quantization (dynamic, INT8, INT4)

Usage:
    from enigma_engine.core.model import create_model, Enigma, ForgeConfig

    model = create_model('small')
    config = ForgeConfig(vocab_size=32000, dim=512, n_layers=8)
    model = Enigma(config=config)
"""

import json
import logging
import math
from collections.abc import Generator
from pathlib import Path
from typing import Any, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Re-exports from sub-modules
# ─────────────────────────────────────────────────────────────────────────────
from .model_presets import (  # noqa: F401
    ForgeConfig,
    QuantizationConfig,
    MODEL_PRESETS,
    MODEL_DESCRIPTIONS,
    get_preset,
    estimate_parameters,
    list_presets,
)
from .model_components import (  # noqa: F401
    RMSNorm,
    DropPath,
    Attention,
    FeedForward,
    TransformerBlock,
    precompute_rope_frequencies,
    apply_rotary_embedding,
    HAS_FLASH_ATTN,
    flash_attn_func,
)
from .model_utils import (  # noqa: F401
    apply_repetition_penalty,
    sample_next_token,
    detect_hardware,
    recommend_model_size,
    estimate_memory_usage,
    get_running_models,
    is_model_loaded,
    register_model,
    unregister_model,
    get_model,
    _LOADED_MODELS,
    _MODELS_LOCK,
)
from .safe_save import atomic_torch_save, atomic_write_json


# =============================================================================
# T3-6: Chunked cross-entropy to avoid materializing [B*T, V] logits
# =============================================================================


def _chunked_cross_entropy(
    output_proj: nn.Linear,
    hidden: torch.Tensor,
    targets: torch.Tensor,
    chunk_size: int = 4096,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Cross-entropy without materializing the full ``[B*T, V]`` logit tensor.

    Chunks over the sequence dimension, computing the output projection
    and CE per chunk.  Peak logit memory: ``O(chunk * V)`` instead of
    ``O(B*T * V)``.  Biggest win at large vocab sizes (32K+).
    """
    hidden_flat = hidden.reshape(-1, hidden.size(-1))
    targets_flat = targets.reshape(-1)
    N = hidden_flat.size(0)

    total_loss = hidden.new_zeros(())
    total_tokens = hidden.new_zeros((), dtype=torch.long)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk_logits = output_proj(hidden_flat[start:end])
        chunk_targets = targets_flat[start:end]

        valid_mask = chunk_targets != ignore_index
        n_valid = valid_mask.sum()
        if n_valid > 0:
            total_loss = total_loss + F.cross_entropy(
                chunk_logits,
                chunk_targets,
                ignore_index=ignore_index,
                label_smoothing=label_smoothing,
                reduction="sum",
            )
            total_tokens = total_tokens + n_valid

    if total_tokens == 0:
        return total_loss
    return total_loss / total_tokens.float()


# =============================================================================
# MAIN MODEL - THE FULL TRANSFORMER
# =============================================================================


class Enigma(nn.Module):
    """
    Enigma - Modern Transformer Language Model

    📖 THIS IS THE COMPLETE MODEL!
    Stacks together all the components: embeddings, transformer blocks, output.

    📐 FULL ARCHITECTURE:
    ┌────────────────────────────────────────────────────────────────────────┐
    │  Input Token IDs [batch, seq_len]                                       │
    │          ↓                                                              │
    │  ┌──────────────────┐                                                  │
    │  │ Token Embedding  │  Converts token IDs to vectors                   │
    │  └──────────────────┘                                                  │
    │          ↓                                                              │
    │  ┌──────────────────┐                                                  │
    │  │ Dropout          │  Regularization during training                  │
    │  └──────────────────┘                                                  │
    │          ↓                                                              │
    │  ╔══════════════════╗                                                  │
    │  ║ TransformerBlock ║ × n_layers                                       │
    │  ║  • Attention     ║  (the "thinking" happens here!)                  │
    │  ║  • Feed-Forward  ║                                                  │
    │  ╚══════════════════╝                                                  │
    │          ↓                                                              │
    │  ┌──────────────────┐                                                  │
    │  │ Final Norm       │  RMSNorm for stable outputs                      │
    │  └──────────────────┘                                                  │
    │          ↓                                                              │
    │  ┌──────────────────┐                                                  │
    │  │ Output Head      │  Projects to vocabulary size                     │
    │  └──────────────────┘                                                  │
    │          ↓                                                              │
    │  Output Logits [batch, seq_len, vocab_size]                            │
    └────────────────────────────────────────────────────────────────────────┘

    ⚡ FEATURES:
    - RoPE positional embeddings (knows word order)
    - RMSNorm (fast and stable)
    - SwiGLU activation (better learning)
    - GQA attention (memory efficient)
    - KV cache (fast generation)

    🔗 CONNECTS TO:
      → Uses all the components defined above
      ← Used by EnigmaEngine for inference
      ← Used by Trainer for training
    """

    def __init__(self, config: ForgeConfig, **kwargs):
        """
        Initialize the Enigma model.

        Args:
            config: ForgeConfig object defining model architecture
            **kwargs: Additional config overrides
        """
        super().__init__()

        self.config = config

        # Apply any kwargs overrides
        for k, v in kwargs.items():
            if hasattr(self.config, k):
                setattr(self.config, k, v)

        # Pad vocab to multiple of 64 for GPU matmul alignment (free speedup)
        padded_vocab = (self.config.vocab_size + 63) & ~63

        # Token embeddings (padded for GPU alignment)
        self.tok_embeddings = nn.Embedding(padded_vocab, self.config.dim)

        # ─────────────────────────────────────────────────────────────────────
        # MULTI-MODAL PROJECTION LAYERS (Optional)
        # ─────────────────────────────────────────────────────────────────────
        # These allow integrating vision/audio encoders with the text model
        if self.config.vision_hidden_size is not None:
            # LLaVA-1.5 (arxiv:2310.03744 §3.2): 2-layer MLP with GELU,
            # bias=True on both Linears. Reference impl is HF
            # `LlavaMultiModalProjector`. Was a single bias-free Linear in
            # LLaVA-1; the MLP upgrade was the headline architecture change.
            # State-dict keys: vision_projection.{0,2}.{weight,bias}.
            self.vision_projection = nn.Sequential(
                nn.Linear(self.config.vision_hidden_size, self.config.dim, bias=True),
                nn.GELU(),
                nn.Linear(self.config.dim, self.config.dim, bias=True),
            )
            logger.info(
                f"Added vision projection (LLaVA-1.5 MLP): "
                f"{self.config.vision_hidden_size} → {self.config.dim} "
                f"→ {self.config.dim}"
            )
        else:
            self.vision_projection = None

        if self.config.audio_hidden_size is not None:
            self.audio_projection = nn.Linear(self.config.audio_hidden_size, self.config.dim, bias=False)
            logger.info(f"Added audio projection: {self.config.audio_hidden_size} → {self.config.dim}")
        else:
            self.audio_projection = None

        # Position embeddings (fallback)
        if not self.config.use_rope:
            self.pos = nn.Parameter(torch.randn(1, self.config.max_seq_len, self.config.dim) * 0.02)

        # Transformer layers
        self.layers = nn.ModuleList([TransformerBlock(self.config, i) for i in range(self.config.n_layers)])

        # T3-1: Cross-layer KV sharing — group layers into bands that
        # share K, V projections.  Only the first layer in each band
        # computes K, V; followers reuse them.
        n_groups = getattr(self.config, "kv_share_groups", 0)
        if n_groups > 0 and self.config.n_layers >= n_groups:
            layers_per_group = self.config.n_layers // n_groups
            for i, layer in enumerate(self.layers):
                leader_idx = (i // layers_per_group) * layers_per_group
                if i != leader_idx:
                    layer.attention._kv_share_source = self.layers[leader_idx].attention
                else:
                    # Only leaders store _shared_kv (followers read it within
                    # the same forward); non-leader/non-sharing layers must
                    # not pin graph-attached activations on the module.
                    layer.attention._kv_is_leader = True

        # Output (padded for GPU alignment, weight-tied with embeddings)
        Norm = RMSNorm if self.config.use_rms_norm else nn.LayerNorm
        self.norm = Norm(self.config.dim)
        self.output = nn.Linear(self.config.dim, padded_vocab, bias=False)

        # Weight tying
        self.output.weight = self.tok_embeddings.weight

        # Multi-token prediction heads (R25, Gloeckle et al. 2024)
        # Extra heads predict 2nd, 3rd, ... next tokens. Train-only, dropped at inference.
        self.predict_heads = nn.ModuleList()
        if self.config.n_predict_heads > 0:
            for _ in range(self.config.n_predict_heads):
                self.predict_heads.append(nn.Linear(self.config.dim, padded_vocab, bias=False))

        # T3-2: Self-speculative decoding — early-exit head
        self.early_exit_layer = getattr(self.config, "early_exit_layer", 0)
        if self.early_exit_layer > 0:
            self.early_exit_norm = RMSNorm(self.config.dim)
            self.early_exit_head = nn.Linear(self.config.dim, padded_vocab, bias=False)

        # RoPE frequencies with optional scaling
        if self.config.use_rope:
            head_dim = self.config.dim // self.config.n_heads
            # Validate head_dim is even (required for RoPE complex number reshape)
            if head_dim % 2 != 0:
                raise ValueError(
                    f"head_dim must be even for RoPE, got {head_dim} "
                    f"(dim={self.config.dim}, n_heads={self.config.n_heads}). "
                    f"Adjust dim or n_heads so dim/n_heads is even."
                )
            self.register_buffer(
                "freqs_cis",
                precompute_rope_frequencies(
                    head_dim,
                    self.config.max_seq_len * 2,
                    self.config.rope_theta,
                    scaling_type=self.config.rope_scaling_type,
                    scaling_factor=self.config.rope_scaling_factor,
                ),
                persistent=False,
            )
        else:
            self.freqs_cis = None

        # ─────────────────────────────────────────────────────────────────────
        # CAUSAL MASK: Computed on demand and cached at the size actually used.
        # ─────────────────────────────────────────────────────────────────────
        # Previous approach pre-allocated max_seq_len² floats (e.g. 4096² =
        # 64 MB for fp32).  Now we build on first use at the actual sequence
        # length and grow only if a longer sequence appears.
        self._causal_mask: Optional[torch.Tensor] = None
        self._causal_mask_size: int = 0

        # Initialize
        self.apply(self._init_weights)
        self._init_output_weights()

        # nGPT: Apply weight normalization to all Linear layers
        # (except weight-tied output head) for training stability.
        if getattr(self.config, "use_weight_norm", False):
            self._apply_weight_norm()

    def _init_weights(self, module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _init_output_weights(self) -> None:
        for name, p in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w2.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.config.n_layers))

    def gradient_checkpointing_enable(self) -> None:
        """Enable gradient checkpointing on all transformer layers.

        Trades ~30% extra compute for ~50% VRAM savings by recomputing
        activations during the backward pass instead of storing them.

        Raises:
            ValueError: If the model uses kv_share_groups; checkpointed
                recomputation cannot rebuild K/V shared across layers.
        """
        if getattr(self.config, "kv_share_groups", 0) > 0:
            raise ValueError(
                "gradient checkpointing cannot be combined with kv_share_groups: "
                "checkpointed recomputation cannot rebuild K/V shared across layers"
            )
        for layer in self.layers:
            layer.use_checkpoint = True

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing on all transformer layers."""
        for layer in self.layers:
            layer.use_checkpoint = False

    def _apply_weight_norm(self) -> None:
        """Apply parametric weight normalization to all Linear layers.

        Decomposes W = g * (v / ||v||), separating direction from magnitude.
        Skips the output head (weight-tied with tok_embeddings) and any
        Linear inside norm layers.
        """
        from torch.nn.utils.parametrizations import weight_norm

        # Collect modules to normalize (skip output head — weight-tied)
        tied_id = id(self.output.weight)
        count = 0
        for module in self.modules():
            if isinstance(module, nn.Linear) and id(module.weight) != tied_id:
                weight_norm(module, name="weight")
                count += 1
        if count:
            logger.info(f"nGPT: applied weight normalization to {count} Linear layers")

    def _get_causal_mask(self, size: int) -> torch.Tensor:
        """Return a (size, size) causal mask, cached and grown on demand.

        Only built when the forward pass materializes an explicit attention
        mask (padded batches). Plain causal training/prefill and KV-cache
        decode take the fast ``is_causal`` path with ``mask=None``.
        """
        if self._causal_mask is None or self._causal_mask_size < size:
            # Build at the requested size (never shrink)
            new_size = max(size, self._causal_mask_size)
            device = next(self.parameters()).device
            mask = torch.full((new_size, new_size), float("-inf"), device=device)
            mask = torch.triu(mask, diagonal=1)
            self._causal_mask = mask
            self._causal_mask_size = new_size
        return self._causal_mask[:size, :size]

    def load_state_dict(self, state_dict, strict=True, **kwargs):
        """Load state dict with automatic vocab padding and buffer cleanup.

        Handles two common mismatches:
        1. freqs_cis — deterministic RoPE buffer recomputed in __init__,
           stale checkpoint values cause size errors.
        2. Vocab padding — model pads vocab to multiple of 64 for GPU
           matmul alignment, but older checkpoints have unpadded weights.
        """
        if "freqs_cis" in state_dict:
            state_dict.pop("freqs_cis")
            logger.debug("Removed stale freqs_cis buffer from checkpoint")

        padded_vocab = (self.config.vocab_size + 63) & ~63
        if padded_vocab != self.config.vocab_size:
            vocab_keys = [
                "tok_embeddings.weight",
                "output.weight",
            ]
            for key in vocab_keys:
                if key in state_dict and state_dict[key].shape[0] == self.config.vocab_size:
                    old = state_dict[key]
                    padded = torch.zeros(padded_vocab, *old.shape[1:], dtype=old.dtype)
                    padded[: self.config.vocab_size] = old
                    state_dict[key] = padded

        return super().load_state_dict(state_dict, strict=strict, **kwargs)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        start_pos: int = 0,
        return_loss: bool = False,
        pad_token_id: int = 0,
        label_smoothing: float = 0.0,
        attention_mask: Optional[torch.Tensor] = None,
        attention_mask_2d: Optional[torch.Tensor] = None,
        chunked_ce: int = 0,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Forward pass.

        Args:
            input_ids: Input token IDs (B, T)
            targets: Optional target IDs for loss computation
            use_cache: Whether to use KV cache
            start_pos: Starting position for RoPE
            return_loss: If True, always return (logits, loss) tuple
            pad_token_id: Token ID used for padding — ignored in loss
                computation via ``ignore_index``.
            attention_mask: Optional mask (B, T) where 1 = real token,
                0 = padding.  Combined with the causal mask so that
                pad positions receive -inf attention scores.
            attention_mask_2d: Optional pre-built mask (B, 1, T, T)
                with 0 = attend and -inf = block.  Used by sequence
                packing.  When provided, ``attention_mask`` is ignored.
            chunked_ce: Vocab-chunk size for cross-entropy (T3-6).
                When > 0, avoids materializing the full ``[B*T, V]``
                logit tensor.  Returns ``logits=None`` in this mode.
                0 = disabled (default, full logits computed).

        Returns:
            logits if no targets and return_loss=False, else (logits, loss)
        """
        if not (0.0 <= label_smoothing <= 1.0):
            raise ValueError(f"label_smoothing must be in [0.0, 1.0], got {label_smoothing}")

        B, T = input_ids.shape

        h = self.tok_embeddings(input_ids)

        # NEFTune: uniform noise on embeddings during training (R27, Jain et al. 2023)
        if self.training and self.config.neftune_alpha > 0:
            dims = T * h.shape[2]  # seq_len * hidden_dim
            mag = self.config.neftune_alpha / math.sqrt(dims)
            h = h + torch.empty_like(h).uniform_(-mag, mag)

        if not self.config.use_rope:
            end_pos = start_pos + T
            if end_pos > self.config.max_seq_len:
                raise ValueError(
                    f"Position {end_pos} exceeds max_seq_len {self.config.max_seq_len}. "
                    f"Use RoPE or increase max_seq_len."
                )
            h = h + self.pos[:, start_pos:end_pos]

        # Causal masking. For the PLAIN causal case (no packed/pad mask) we leave
        # mask=None so attention applies causality via is_causal=True — that lets SDPA
        # pick the fast mem-efficient/flash kernel instead of the slow additive-mask
        # path (~1.7x on the attention op; math-identical). An explicit additive mask
        # is materialized ONLY when a packed (2D) or padding mask must be combined in.
        mask = None
        if attention_mask_2d is not None:
            # Pre-built 4D mask from sequence packing — use directly
            mask = attention_mask_2d.to(device=h.device, dtype=h.dtype)
        elif T > 1 and attention_mask is not None:
            # Padding present: materialize causal + pad as one additive mask.
            # Match h.dtype (like the packed-mask branch) — a bare fp32 mask on a
            # bf16/fp16 model outside autocast raises "invalid dtype for bias" in SDPA.
            mask = self._get_causal_mask(T).to(device=h.device, dtype=h.dtype).unsqueeze(0).unsqueeze(0)
            # attention_mask shape: (B, T) → (B, 1, 1, T)
            pad_mask = (1 - attention_mask.unsqueeze(1).unsqueeze(2).to(h.dtype)) * (torch.finfo(h.dtype).min)
            mask = mask + pad_mask
        # else: plain causal (T>1, no pad) or single-token decode (T==1) → mask stays
        # None; attention applies is_causal=True (correct for this decoder-only model).

        # T3-2: Capture hidden state at early-exit layer for draft head
        _early_h = None
        for i, layer in enumerate(self.layers):
            h = layer(h, self.freqs_cis, mask, use_cache, start_pos)
            if self.early_exit_layer > 0 and i == self.early_exit_layer - 1:
                _early_h = h

        # T3-1: shared K/V is intra-forward plumbing (leader writes, followers
        # read within this same pass) — drop it so no graph-attached activation
        # outlives the forward (VRAM + deepcopy safety).
        if getattr(self.config, "kv_share_groups", 0) > 0:
            for layer in self.layers:
                layer.attention._shared_kv = None

        h_normed = self.norm(h)

        # T3-6: Chunked cross-entropy avoids full [B*T, V] logit tensor
        use_chunked = chunked_ce > 0 and targets is not None

        logits = None if use_chunked else self.output(h_normed)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            if use_chunked:
                loss = _chunked_cross_entropy(
                    self.output,
                    h_normed,
                    targets,
                    chunk_size=chunked_ce,
                    ignore_index=pad_token_id,
                    label_smoothing=label_smoothing,
                )
            else:
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=pad_token_id,
                    label_smoothing=label_smoothing,
                )

            # Multi-token prediction auxiliary loss (R25, Gloeckle et al. 2024)
            # Each extra head predicts the k-th next token (k=2,3,...).
            if self.training and self.predict_heads:
                mtp_loss = torch.tensor(0.0, device=h_normed.device, dtype=h_normed.dtype)
                for i, head in enumerate(self.predict_heads):
                    shift = i + 1
                    if targets.size(1) > shift:
                        shifted_targets = targets[:, shift:]
                        if use_chunked:
                            mtp_loss = mtp_loss + _chunked_cross_entropy(
                                head,
                                h_normed[:, :-shift],
                                shifted_targets,
                                chunk_size=chunked_ce,
                                ignore_index=pad_token_id,
                                label_smoothing=label_smoothing,
                            )
                        else:
                            head_logits = head(h_normed[:, :-shift])
                            mtp_loss = mtp_loss + F.cross_entropy(
                                head_logits.reshape(-1, head_logits.size(-1)),
                                shifted_targets.reshape(-1),
                                ignore_index=pad_token_id,
                                label_smoothing=label_smoothing,
                            )
                loss = loss + mtp_loss / max(1, len(self.predict_heads))

            # T3-2: Early-exit auxiliary loss for self-speculative decoding
            if self.training and _early_h is not None:
                ee_normed = self.early_exit_norm(_early_h)
                ee_logits = self.early_exit_head(ee_normed)
                ee_loss = F.cross_entropy(
                    ee_logits.reshape(-1, ee_logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=pad_token_id,
                    label_smoothing=label_smoothing,
                )
                loss = loss + 0.1 * ee_loss

        # Return format depends on whether loss was computed
        if targets is not None or return_loss:
            return logits, loss

        return logits

    def clear_cache(self) -> None:
        for layer in self.layers:
            layer.clear_cache()

    def rewind_cache(self, position: int) -> None:
        """Truncate all layer KV-caches back to *position*."""
        for layer in self.layers:
            layer.rewind_cache(position)

    def restore_prefix_cache(self, prefix_cache) -> None:
        """Restore prefix KV into each layer's attention cache.

        After calling this, ``current_pos`` in each layer's
        ``KVCache`` equals ``prefix_cache.prefix_len``, so the
        next token-by-token decode starts at the right position.

        Args:
            prefix_cache: A :class:`PrefixKVCache` with one entry
                per layer.
        """
        for i, layer in enumerate(self.layers):
            pk, pv = prefix_cache.get(i)
            attn = layer.attention
            # Ensure the per-layer KVCache exists (lazy-init)
            if attn._kv_cache is None:
                from enigma_engine.core.kv_cache import KVCache

                attn._kv_cache = KVCache(
                    batch_size=pk.shape[0],
                    max_seq_len=attn.max_cache_len,
                    n_kv_heads=attn.n_kv_heads,
                    head_dim=attn.head_dim,
                    device=pk.device,
                    dtype=pk.dtype,
                )
            attn._kv_cache.restore_prefix(pk, pv)

    def get_moe_aux_loss(self) -> torch.Tensor:
        """
        Get total MoE auxiliary loss from all layers.

        📖 WHAT THIS DOES:
        Collects load balancing losses from all MoE layers to encourage
        even distribution of tokens across experts during training.

        ⚠️ TRAINING ONLY:
        This loss should be added to the main training loss:
            total_loss = ce_loss + model.get_moe_aux_loss()

        Returns:
            Scalar tensor with combined auxiliary loss from all MoE layers.
            Returns 0.0 if MoE is not enabled.
        """
        if not self.config.use_moe:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        aux_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for layer in self.layers:
            aux_loss = aux_loss + layer.get_moe_aux_loss()
        return aux_loss

    def forward_multimodal(
        self,
        input_ids: Optional[torch.Tensor] = None,
        vision_features: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with multi-modal inputs.

        📖 WHAT THIS DOES:
        Processes text, vision, and/or audio inputs together. Vision and audio
        features are projected to the text embedding space and concatenated.

        Args:
            input_ids: Text token IDs [batch, text_seq_len]
            vision_features: Vision encoder output [batch, vision_seq_len, vision_dim]
            audio_features: Audio encoder output [batch, audio_seq_len, audio_dim]
            **kwargs: Additional forward pass arguments

        Returns:
            Model output logits

        Example:
            # Text + vision
            logits = model.forward_multimodal(
                input_ids=text_ids,
                vision_features=vision_output
            )
        """
        embeddings_list = []

        # Process vision features
        if vision_features is not None:
            if self.vision_projection is None:
                raise ValueError("Vision features provided but vision_hidden_size not set in config")
            vision_embeds = self.vision_projection(vision_features)
            embeddings_list.append(vision_embeds)

        # Process audio features
        if audio_features is not None:
            if self.audio_projection is None:
                raise ValueError("Audio features provided but audio_hidden_size not set in config")
            audio_embeds = self.audio_projection(audio_features)
            embeddings_list.append(audio_embeds)

        # Process text
        if input_ids is not None:
            text_embeds = self.tok_embeddings(input_ids)
            embeddings_list.append(text_embeds)

        if not embeddings_list:
            raise ValueError("At least one of input_ids, vision_features, or audio_features must be provided")

        # Concatenate all modalities
        combined_embeds = torch.cat(embeddings_list, dim=1)

        # Continue with standard forward pass
        B, T, _ = combined_embeds.shape
        h = combined_embeds

        # Add positional embeddings if not using RoPE
        if not self.config.use_rope and hasattr(self, "pos"):
            if T > self.config.max_seq_len:
                raise ValueError(
                    f"Sequence length {T} exceeds max_seq_len {self.config.max_seq_len} "
                    f"in forward_multimodal. Use RoPE or increase max_seq_len."
                )
            h = h + self.pos[:, :T]

        # Build causal mask on demand. Match h.dtype (same constraint as the
        # main forward): a bare fp32 mask on a bf16/fp16 model outside
        # autocast raises "invalid dtype for bias" in SDPA.
        mask = None
        if T > 1:
            mask = self._get_causal_mask(T).to(device=h.device, dtype=h.dtype).unsqueeze(0).unsqueeze(0)

            # Combine with attention_mask if provided via kwargs
            attention_mask = kwargs.get("attention_mask")
            if attention_mask is not None:
                pad_mask = (1 - attention_mask.unsqueeze(1).unsqueeze(2).to(h.dtype)) * (torch.finfo(h.dtype).min)
                mask = mask + pad_mask

        # Transform through layers
        for layer in self.layers:
            h = layer(h, self.freqs_cis, mask, kwargs.get("use_cache", False), kwargs.get("start_pos", 0))

        # Output projection
        logits = self.output(self.norm(h))

        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        stop_tokens: Optional[list[int]] = None,
        *,  # Force keyword-only args after this
        return_logits: bool = False,
        stream: bool = False,
        min_p: float = 0.0,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate tokens autoregressively.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top-k tokens for sampling
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repeating tokens (1.0 = no penalty)
            stop_tokens: Token IDs that stop generation
            return_logits: If True, also return the final logits
            stream: If True, use streaming generation (returns generator)
            min_p: Min-p filter threshold (0 = off); see sample_next_token

        Returns:
            Generated token IDs, or (token_ids, logits) if return_logits=True,
            or generator if stream=True

        Raises:
            ValueError: If any of the following hold — ``input_ids`` is
                not a 2D tensor of shape ``[batch, seq_len]``;
                ``input_ids.device`` does not match the model's
                parameter device.
        """
        # temperature <= 0 means GREEDY decode (sample_next_token takes the
        # argmax path) -- the deterministic contract clients expect from
        # temperature=0. It used to raise here while generate_stream let the
        # same value straight through (audit 2026-07-15).
        if temperature <= 0:
            temperature = 0.0

        # Validate input shape
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be 2D [batch, seq_len], got shape {list(input_ids.shape)}")
        # Single-sequence only: the KV-cache, per-step sampling and stop-token
        # test (next_token.squeeze().item()) are all written for batch 1. Fail
        # honestly instead of crashing mid-generation on a batched input.
        if input_ids.shape[0] != 1:
            raise ValueError(f"generate() supports batch size 1 only, got {input_ids.shape[0]}")

        # Validate device match
        model_device = next(self.parameters()).device
        if input_ids.device != model_device:
            raise ValueError(
                f"input_ids on {input_ids.device} but model on {model_device}. Move input to model device first."
            )

        # Delegate to streaming generator if requested
        if stream:
            return self.generate_stream(
                input_ids, max_new_tokens, temperature, top_k, top_p, repetition_penalty, stop_tokens, min_p=min_p
            )

        self.clear_cache()
        stop_tokens = stop_tokens or [2]

        generated = input_ids
        # Repetition penalty must see only HER tokens: penalizing prompt
        # tokens suppresses exactly the vocabulary the system prompt primes
        # her to answer with ("Enigma", "local", ...) -- measured as
        # incoherence at 182M (ultrareview #9 scope fix, 2026-07-15).
        prompt_len = input_ids.shape[1]
        logits = self.forward(input_ids, use_cache=True)
        final_logits = logits  # Track for return_logits option

        for _ in range(max_new_tokens):
            next_token = sample_next_token(
                logits[:, -1, :],
                generated[:, prompt_len:],
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                min_p=min_p,
            )
            generated = torch.cat([generated, next_token], dim=1)

            if next_token.squeeze().item() in stop_tokens:
                break

            logits = self.forward(next_token, use_cache=True, start_pos=generated.shape[1] - 1)
            final_logits = logits  # Update for return_logits

        # Return based on options
        if return_logits:
            return generated, final_logits
        return generated

    @torch.no_grad()
    def generate_stream(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        stop_tokens: Optional[list[int]] = None,
        min_p: float = 0.0,
    ) -> Generator[torch.Tensor, None, None]:
        """
        Streaming token generation - yields tokens as they're generated.

        📖 WHAT THIS DOES:
        Instead of waiting for all tokens to be generated, this yields each
        token as soon as it's produced. Essential for real-time chat UX!

        📐 STREAMING FLOW:
        ┌────────────────────────────────────────────────────────────────────┐
        │  User: "Tell me a story"                                           │
        │                                                                    │
        │  [Model starts generating]                                         │
        │       ↓                                                            │
        │  yield "Once"    ← User sees this immediately                      │
        │       ↓                                                            │
        │  yield " upon"   ← And this...                                     │
        │       ↓                                                            │
        │  yield " a"      ← And this...                                     │
        │       ↓                                                            │
        │  yield " time"   ← Progressive display!                            │
        └────────────────────────────────────────────────────────────────────┘

        💡 ADVANTAGES:
        - Better user experience (see output immediately)
        - Can cancel generation early
        - Works great with chat interfaces

        Args:
            input_ids: Input token IDs [batch, seq_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Keep only top-k tokens for sampling
            top_p: Nucleus sampling threshold
            repetition_penalty: Penalty for repeating tokens
            stop_tokens: Token IDs that stop generation
            min_p: Min-p filter threshold (0 = off); see sample_next_token

        Yields:
            Individual tokens as they're generated [1] tensor

        Example:
            for token in model.generate_stream(input_ids):
                print(tokenizer.decode([token.item()]), end='', flush=True)
        """
        if input_ids.dim() != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                f"generate_stream() supports a single 2D [1, seq_len] sequence, got shape {list(input_ids.shape)}"
            )
        self.clear_cache()
        stop_tokens = stop_tokens or [2]

        generated = input_ids
        # Same penalty-scope rule as generate(): only HER tokens, not the prompt.
        prompt_len = input_ids.shape[1]
        logits = self.forward(input_ids, use_cache=True)

        for _ in range(max_new_tokens):
            next_token = sample_next_token(
                logits[:, -1, :],
                generated[:, prompt_len:],
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                min_p=min_p,
            )

            # Yield the token immediately
            yield next_token.squeeze()

            # Check for stop tokens
            if next_token.squeeze().item() in stop_tokens:
                break

            # Update generated sequence and get next logits
            generated = torch.cat([generated, next_token], dim=1)
            logits = self.forward(next_token, use_cache=True, start_pos=generated.shape[1] - 1)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Enigma":
        return cls(config=ForgeConfig.from_dict(config))

    @classmethod
    def from_pretrained(cls, path: Path) -> "Enigma":
        from .model_registry import safe_load_weights

        path = Path(path)
        config_file = path / "config.json" if path.is_dir() else path.with_suffix(".json")

        if config_file.exists():
            with open(config_file, encoding="utf-8") as f:
                model = cls.from_config(json.load(f))
        else:
            # config is a required constructor argument; no config file means
            # default dimensions
            model = cls(ForgeConfig())

        weights_file = path / "weights.pth" if path.is_dir() else path
        if weights_file.exists():
            from .model_registry import get_state_dict

            raw = safe_load_weights(weights_file, map_location="cpu")
            state_dict = get_state_dict(raw)
            model.load_state_dict(state_dict, strict=False)

        return model

    @classmethod
    def auto_configure(cls, vocab_size: int = 32000) -> "Enigma":
        """
        Create an optimally-configured model for the current hardware.

        📖 WHAT THIS DOES:
        Detects your hardware (GPU, RAM, Pi model) and automatically creates
        a model with the best configuration for your system.

        📐 DETECTION FLOW:
        ┌─────────────────────────────────────────────────────────────────────┐
        │  1. Detect hardware (RAM, GPU, is_raspberry_pi)                     │
        │  2. Recommend model size (pi_zero, pi_4, pi_5, small, medium, etc.) │
        │  3. Recommend quantization (none, dynamic, int8, int4)              │
        │  4. Create model with optimal config                                │
        │  5. Apply quantization if recommended                               │
        └─────────────────────────────────────────────────────────────────────┘

        Args:
            vocab_size: Vocabulary size for the model

        Returns:
            Optimally configured Forge model

        Example:
            # Auto-detect and create best model for this device
            model = Forge.auto_configure()
        """
        try:
            from .hardware_detection import detect_hardware, get_optimal_config

            profile = detect_hardware()
            config = get_optimal_config(profile)

            logger.info(f"Auto-configured for: {profile.hardware_type}")
            logger.info(f"Recommended: {config['model_size']} with {config.get('quantization', 'none')} quantization")

            # Create model with recommended size
            model = create_model(config["model_size"], vocab_size=vocab_size)

            # Apply quantization if recommended
            quant = config.get("quantization", "none")
            if quant and quant != "none":
                model.quantize(quant)

            return model

        except ImportError as e:
            logger.warning(f"Hardware detection not available: {e}, using default config")
            return create_model("small", vocab_size=vocab_size)

    def quantize(self, mode: str = "dynamic") -> "Enigma":
        """
        Apply quantization to reduce model memory footprint.

        📖 WHAT THIS DOES:
        Converts model weights to lower precision to save memory and
        potentially speed up inference on CPU.

        📐 QUANTIZATION MODES:
        ┌────────────────────────────────────────────────────────────────────┐
        │ Mode     │ Memory │ Speed  │ Quality │ Best For                   │
        ├──────────┼────────┼────────┼─────────┼────────────────────────────┤
        │ none     │ 100%   │ Fast   │ Best    │ GPU, plenty of RAM         │
        │ dynamic  │ ~50%   │ Fast   │ Good    │ CPU inference, Pi 5        │
        │ int8     │ ~25%   │ Medium │ Good    │ Pi 4, limited RAM          │
        │ int4     │ ~12%   │ Slower │ Fair    │ Pi Zero, extreme limits    │
        └────────────────────────────────────────────────────────────────────┘

        ⚠️ NOTES:
        - Quantization is irreversible (on the current model instance)
        - GPU acceleration may not work after quantization
        - Quality degradation is usually minor for dynamic/int8

        Args:
            mode: Quantization mode ("none", "dynamic", "int8", "int4")

        Returns:
            Self (for method chaining)

        Example:
            model = create_model('small')
            model.quantize('dynamic')  # Now uses less memory
        """
        if mode == "none":
            logger.info("No quantization applied")
            return self

        # Ensure model is on CPU for quantization
        device = next(self.parameters()).device
        if device.type != "cpu":
            logger.info("Moving model to CPU for quantization")
            self.cpu()

        if mode == "dynamic":
            # Dynamic quantization - fastest, good quality
            try:
                self._apply_dynamic_quantization()
                logger.info("Applied dynamic INT8 quantization")
            except Exception as e:
                logger.warning(f"Dynamic quantization failed: {e}")

        elif mode == "int8":
            # Static INT8 quantization (falls back to dynamic when no
            # calibration data is wired in — see _apply_static_int8_quantization).
            # The success log lives inside the helper so it reflects the
            # path that actually ran, not the requested mode.
            try:
                self._apply_static_int8_quantization()
            except Exception as e:
                logger.warning(f"Static INT8 quantization failed: {e}")

        elif mode == "int4":
            # 4-bit quantization (most aggressive). Success log lives
            # inside _apply_int4_quantization so it reflects the actual
            # path (bitsandbytes vs dynamic fallback).
            try:
                self._apply_int4_quantization()
            except Exception as e:
                logger.warning(f"INT4 quantization failed: {e}")
        else:
            logger.warning(f"Unknown quantization mode: {mode}")

        return self

    def _apply_dynamic_quantization(self) -> None:
        """Apply PyTorch dynamic quantization to linear layers."""
        torch.quantization.quantize_dynamic(
            self,
            {nn.Linear},  # Quantize Linear layers
            dtype=torch.qint8,
            inplace=True,
        )

    def _apply_static_int8_quantization(self) -> None:
        """Apply static INT8 quantization (requires calibration data).

        Note: true static INT8 needs a calibration data pass which is not
        wired in. This currently falls back to dynamic INT8. The log
        message reflects the actual path taken so operators can trust it.
        """
        logger.info("Static INT8 calibration data absent -- fell back to dynamic INT8 quantization")
        self._apply_dynamic_quantization()

    def _apply_int4_quantization(self) -> None:
        """Apply 4-bit weight-only quantization."""
        # INT4 quantization is more complex, use dynamic as fallback
        # True INT4 would require specialized libraries like bitsandbytes
        try:
            # Try using bitsandbytes if available
            import bitsandbytes as bnb

            # Convert Linear layers to 4-bit
            for name, module in self.named_modules():
                if isinstance(module, nn.Linear):
                    # Replace with 4-bit linear
                    parent_name = ".".join(name.split(".")[:-1])
                    child_name = name.split(".")[-1]
                    parent = self.get_submodule(parent_name) if parent_name else self

                    new_layer = bnb.nn.Linear4bit(module.in_features, module.out_features, bias=module.bias is not None)
                    new_layer.weight = bnb.nn.Params4bit(module.weight.data, requires_grad=False)
                    if module.bias is not None:
                        new_layer.bias = module.bias

                    setattr(parent, child_name, new_layer)

            logger.info("Applied bitsandbytes INT4 quantization")

        except ImportError:
            # Fallback to dynamic quantization. Log reflects the actual
            # path so operators reading logs aren't told "INT4" when
            # only dynamic INT8 ran.
            logger.warning("bitsandbytes not available -- fell back to dynamic INT8 quantization (not true INT4)")
            self._apply_dynamic_quantization()

    @classmethod
    def from_huggingface(cls, model_id: str, **kwargs) -> "Enigma":
        """Absent feature: no HuggingFace-to-Enigma converter ships in
        this build, so this always raises. Load Enigma checkpoints with
        :meth:`from_pretrained` or :meth:`from_safetensors`.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "from_huggingface: no HuggingFace-to-Enigma converter ships in this build "
            "(there is no huggingface_loader module); load Enigma checkpoints with "
            "from_pretrained or from_safetensors instead"
        )

    @classmethod
    def from_safetensors(cls, path: Union[str, Path], **kwargs) -> "Enigma":
        """
        Load a model from Safetensors format.

        📖 WHAT THIS DOES:
        Loads model weights from Safetensors format, which is faster and
        safer than pickle-based formats (no arbitrary code execution).

        Args:
            path: Path to .safetensors file
            **kwargs: Additional arguments (map_location, etc.)

        Returns:
            Enigma model with loaded weights

        Example:
            model = Enigma.from_safetensors("model.safetensors")
        """
        try:
            from safetensors.torch import load_file
        except ImportError:
            logger.error("Safetensors loading requires safetensors library. Install with: pip install safetensors")
            raise

        path = Path(path)

        # Load config if available
        config_file = path.with_suffix(".json")
        if config_file.exists():
            with open(config_file, encoding="utf-8") as f:
                model = cls.from_config(json.load(f))
        else:
            logger.warning("No config file found, using default config")
            model = cls(ForgeConfig())

        # Load weights
        logger.info(f"Loading Safetensors from: {path}")
        state_dict = load_file(str(path), device=kwargs.get("map_location", "cpu"))
        model.load_state_dict(state_dict, strict=False)

        return model

    @classmethod
    def from_gguf(cls, path: Union[str, Path], **kwargs) -> "Enigma":
        """Absent feature: no GGUF-to-Enigma converter ships in this
        build, so this always raises. Load Enigma checkpoints with
        :meth:`from_pretrained` or :meth:`from_safetensors`.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "from_gguf: no GGUF-to-Enigma converter ships in this build "
            "(there is no gguf_loader module); load Enigma checkpoints with "
            "from_pretrained or from_safetensors instead"
        )

    @classmethod
    def from_onnx(cls, path: Union[str, Path], **kwargs) -> "Enigma":
        """Absent feature: no ONNX-to-Enigma converter ships in this
        build, so this always raises. Load Enigma checkpoints with
        :meth:`from_pretrained` or :meth:`from_safetensors`.

        Raises:
            NotImplementedError: Always.
        """
        raise NotImplementedError(
            "from_onnx: no ONNX-to-Enigma converter ships in this build "
            "(there is no onnx_loader module); load Enigma checkpoints with "
            "from_pretrained or from_safetensors instead"
        )

    # =========================================================================
    # 🎯 LORA & ADAPTER SUPPORT
    # =========================================================================

    def load_lora(self, path: Union[str, Path], adapter_name: str = "default", merge: bool = False) -> None:
        """
        Load LoRA (Low-Rank Adaptation) weights.

        📖 WHAT THIS DOES:
        Loads LoRA adapter weights that modify the model's behavior without
        full fine-tuning. LoRA is memory-efficient and can be quickly swapped.

        📐 HOW LORA WORKS:
        Instead of updating full weight matrix W, LoRA adds:
        W' = W + A × B  (where A, B are small low-rank matrices)

        Args:
            path: Path to LoRA weights
            adapter_name: Name for this adapter (for multi-adapter support)
            merge: If True, immediately merge LoRA into base weights

        Example:
            model.load_lora("lora_adapters/coding.pth")
            model.load_lora("lora_adapters/creative.pth", "creative")
        """
        try:
            from .lora_utils import apply_lora, load_lora_weights
        except ImportError:
            logger.error("LoRA support requires lora_utils module")
            raise

        logger.info(f"Loading LoRA adapter '{adapter_name}' from: {path}")

        # Load LoRA weights
        lora_weights = load_lora_weights(path)

        # Apply to model
        if merge:
            # Merge into base weights immediately
            apply_lora(self, lora_weights, merge=True)
            logger.info(f"Merged LoRA adapter '{adapter_name}' into base weights")
        else:
            # Keep as separate adapter; apply_lora registers it in
            # self._lora_adapters once its key-match check passes, so a
            # mismatched adapter is never left registered
            apply_lora(self, lora_weights, adapter_name=adapter_name)
            logger.info(f"Loaded LoRA adapter '{adapter_name}'")

    def merge_lora(self, adapter_name: Optional[str] = None) -> None:
        """
        Merge LoRA adapters into base model weights.

        📖 WHAT THIS DOES:
        Permanently integrates LoRA adapter weights into the base model.
        After merging, the adapter can be removed to save memory.

        Args:
            adapter_name: Specific adapter to merge (None = merge all)

        Example:
            model.load_lora("adapter.pth", "my_adapter")
            model.merge_lora("my_adapter")  # Merge into base weights
        """
        if not hasattr(self, "_lora_adapters"):
            logger.warning("No LoRA adapters loaded")
            return

        try:
            from .lora_utils import merge_lora_weights
        except ImportError:
            logger.error("LoRA support requires lora_utils module")
            raise

        if adapter_name is None:
            # Merge all adapters
            for name in list(self._lora_adapters.keys()):
                merge_lora_weights(self, self._lora_adapters[name])
                del self._lora_adapters[name]
                logger.info(f"Merged LoRA adapter: {name}")
        else:
            # Merge specific adapter
            if adapter_name not in self._lora_adapters:
                raise ValueError(f"LoRA adapter '{adapter_name}' not found")
            merge_lora_weights(self, self._lora_adapters[adapter_name])
            del self._lora_adapters[adapter_name]
            logger.info(f"Merged LoRA adapter: {adapter_name}")

    # =========================================================================
    # MODEL EXPORT METHODS
    # =========================================================================

    def export_to_safetensors(self, path: Union[str, Path]) -> None:
        """
        Export model to Safetensors format.

        📖 WHAT THIS DOES:
        Saves the model weights in Safetensors format, which is:
        - Faster to load than pickle-based formats
        - Safer (no arbitrary code execution)
        - Compatible with many frameworks (HuggingFace, etc.)

        Args:
            path: Output path for .safetensors file

        Example:
            model.export_to_safetensors("model.safetensors")
            # Also creates model.json with config
        """
        try:
            from safetensors.torch import save_file
        except ImportError:
            raise ImportError(
                "Safetensors export requires safetensors library. Install with: pip install safetensors"
            ) from None

        path = Path(path)

        # Save weights
        save_file(self.state_dict(), str(path))
        logger.info(f"Exported weights to: {path}")

        # Save config alongside
        config_path = path.with_suffix(".json")
        import json as _json
        from enigma_engine.core.safe_save import atomic_write_text

        atomic_write_text(config_path, _json.dumps(self.config.to_dict(), indent=2, ensure_ascii=False))
        logger.info(f"Exported config to: {config_path}")

    def export_to_onnx(
        self,
        path: Union[str, Path],
        opset_version: int = 14,
        input_names: Optional[list[str]] = None,
        output_names: Optional[list[str]] = None,
        dynamic_axes: Optional[dict[str, dict[int, str]]] = None,
    ) -> None:
        """
        Export model to ONNX format for deployment.

        📖 WHAT THIS DOES:
        Exports the model to ONNX format, enabling:
        - Cross-platform deployment (C++, mobile, web)
        - Hardware-specific optimizations (TensorRT, OpenVINO)
        - Framework-agnostic inference

        ⚠️ NOTES:
        - KV-cache based generation is not directly supported in ONNX
        - Export captures the model at a fixed sequence length
        - Dynamic axes allow variable batch/sequence sizes

        Args:
            path: Output path for .onnx file
            opset_version: ONNX opset version (14 is widely supported)
            input_names: Names for input tensors
            output_names: Names for output tensors
            dynamic_axes: Dict specifying variable dimensions

        Example:
            model.export_to_onnx("model.onnx")
            # Use with ONNX Runtime for fast inference
        """
        path = Path(path)

        # Default configurations
        input_names = input_names or ["input_ids"]
        output_names = output_names or ["logits"]
        dynamic_axes = dynamic_axes or {"input_ids": {0: "batch", 1: "sequence"}, "logits": {0: "batch", 1: "sequence"}}

        # Create dummy input (representative of actual input)
        dummy_input = torch.randint(
            0, self.config.vocab_size, (1, min(128, self.config.max_seq_len)), device=next(self.parameters()).device
        )

        # Export
        logger.info(f"Exporting to ONNX (opset {opset_version})...")
        torch.onnx.export(
            self,
            dummy_input,
            str(path),
            opset_version=opset_version,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,  # Optimize constants
        )
        logger.info(f"Exported to ONNX: {path}")

        # Save config alongside
        config_path = path.with_suffix(".json")
        atomic_write_json(config_path, self.config.to_dict())
        logger.info(f"Exported config to: {config_path}")

    def export_to_pytorch(self, path: Union[str, Path]) -> None:
        """
        Export model to standard PyTorch format (.pth).

        📖 WHAT THIS DOES:
        Saves model weights and config in native PyTorch format.
        This is the default format used by Enigma AI Engine.

        Args:
            path: Output path for .pth file

        Example:
            model.export_to_pytorch("models/my_model.pth")
        """
        path = Path(path)

        # Save weights in checkpoint format (compatible with gui_forge loader)
        atomic_torch_save(
            {
                "model_state_dict": self.state_dict(),
                "config": self.config.to_dict(),
            },
            path,
        )
        logger.info(f"Exported weights to: {path}")

        # Save config alongside as JSON for external tools
        config_path = path.with_suffix(".json")
        atomic_write_json(config_path, self.config.to_dict())
        logger.info(f"Exported config to: {config_path}")

    def export_to_gguf(
        self,
        path: Union[str, Path],
        quant_type: str = "F16",
        tokenizer: Any = None,
        model_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> str:
        """
        Export model to GGUF format for use with llama.cpp.

        Args:
            path: Output path for .gguf file
            quant_type: Quantization type (F16, Q4_0, Q4_K_M, Q5_K_M, Q6_K, Q8_0)
            tokenizer: Tokenizer for vocabulary metadata
            model_name: Optional model name for metadata
            description: Optional description for metadata

        Returns:
            Path to exported file

        Example:
            model.export_to_gguf("model-q8.gguf", quant_type="Q8_0")
        """
        from enigma_engine.core.gguf import export_to_gguf

        return export_to_gguf(
            self,
            str(path),
            quant_type=quant_type,
            tokenizer=tokenizer,
            model_name=model_name,
            description=description,
        )


# =============================================================================
# Factory Functions
# =============================================================================


def create_model(size: str = "small", vocab_size: Optional[int] = None, **kwargs) -> Enigma:
    """
    Create an Enigma model from a preset configuration.

    Args:
        size: Model size preset (tiny, small, medium, large, xl, etc.)
        vocab_size: Size of vocabulary. If None, auto-detects from default tokenizer.
        **kwargs: Additional config overrides (unknown keys are logged and ignored)

    Returns:
        Configured Enigma model instance

    Raises:
        ValueError: If size is invalid or vocab_size is invalid
        TypeError: If size is not a string or vocab_size is not an integer
        RuntimeError: If model initialization fails

    Example:
        >>> model = create_model('small', vocab_size=32000)
        >>> model = create_model('medium', dropout=0.2)
    """
    # Validate inputs
    if not isinstance(size, str):
        raise TypeError(f"size must be a string, got {type(size).__name__}")

    # Auto-detect vocab size from tokenizer if not specified
    if vocab_size is None:
        try:
            from .tokenizer import get_tokenizer

            tok = get_tokenizer()
            vocab_size = tok.vocab_size
            logger.info(f"Auto-detected vocab_size={vocab_size} from tokenizer")
        except Exception as e:
            logger.warning(f"Could not auto-detect vocab_size: {e}, using default 32000")
            vocab_size = 32000

    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ValueError(f"vocab_size must be a positive integer, got {vocab_size}")

    if vocab_size > 1000000:
        logger.warning(f"Very large vocab_size ({vocab_size:,}). This may use excessive memory.")

    # Get preset configuration (raises ValueError if size is invalid)
    try:
        config = get_preset(size, vocab_size)
    except ValueError as e:
        logger.error(f"Failed to create model: {e}")
        raise

    # Apply kwargs overrides with validation
    for k, v in kwargs.items():
        if not hasattr(config, k):
            logger.warning(f"Unknown config parameter '{k}' - ignoring")
            continue
        setattr(config, k, v)

    # Create model
    try:
        model = Enigma(config=config)
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise RuntimeError(f"Model creation failed: {e}") from e

    logger.info(f"Created Enigma ({size}): {model.num_parameters:,} params, {config.dim}d, {config.n_layers}L")
    return model
