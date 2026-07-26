"""
Neural network components for the Enigma transformer.

Contains RMSNorm, RoPE functions, DropPath, Attention, FeedForward,
and TransformerBlock modules.
"""

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_presets import ForgeConfig

logger = logging.getLogger(__name__)


# =============================================================================
# MODEL COMPONENTS
# =============================================================================
# These are the LEGO pieces that build the full transformer.
# Each class is a specific neural network layer with a special purpose.


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

    Normalizes the input to a consistent scale: divide x by its RMS
    (sqrt(mean(x^2))), then multiply by a learned per-dimension weight.
    Unlike LayerNorm it skips the mean subtraction, so only one statistic
    is computed instead of two; quality is the same and it is ~10% faster.

    Used by TransformerBlock before attention and the FFN.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """
        Args:
            dim: Dimension of input (should match model dim)
            eps: Small number to prevent division by zero
        """
        super().__init__()
        self.eps = eps
        # Learnable scale parameter - model learns optimal normalization
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize input tensor.

        x: [batch, sequence, dim] → normalized: [batch, sequence, dim]
        """
        # Compute in float32 for numerical stability in fp16/bf16 training
        orig_dtype = x.dtype
        x_f32 = x.float()
        rms = torch.sqrt(torch.mean(x_f32**2, dim=-1, keepdim=True) + self.eps)
        return (x_f32 / rms * self.weight.float()).to(orig_dtype)


class DropPath(nn.Module):
    """Stochastic depth — drops entire residual branches during training.

    Each sample in the batch is independently kept (scaled up) or zeroed.
    At eval time this is a no-op.  ``drop_prob`` should increase linearly
    with layer depth (deeper layers → higher drop rate).
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        if drop_prob < 0.0 or drop_prob >= 1.0:
            raise ValueError(f"DropPath drop_prob must be in [0, 1), got {drop_prob}")
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        # Random tensor per sample: shape (B, 1, 1) so it blankets the whole sample
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep, device=x.device, dtype=x.dtype))
        return x * mask / keep


# =============================================================================
# ROTARY POSITION EMBEDDINGS (RoPE)
# =============================================================================
# Without position info, "dog bites man" = "man bites dog" to the model!
# RoPE encodes position by ROTATING the vectors - elegant and effective.


def precompute_rope_frequencies(
    dim: int, max_seq_len: int, theta: float = 10000.0, scaling_type: Optional[str] = None, scaling_factor: float = 1.0
) -> torch.Tensor:
    """
    Precompute RoPE frequencies for all positions with optional scaling.

    Creates a table of rotation angles for each position and dimension pair:
    for dimension pair i, frequency = 1 / (theta^(2i/dim)); for position p,
    angle = p * frequency. Each dimension pair rotates at its own speed,
    which is what lets the model read position back out of the rotations.

    Scaling modes (extend context beyond the training length):
    - linear: freqs = freqs / scaling_factor (simple compression)
    - dynamic: Adaptive NTK-aware scaling (better quality)
    - yarn: Yet another RoPE extension (best for very long contexts)

    Args:
        dim: Dimension per head (must be even)
        max_seq_len: Maximum sequence length
        theta: Base frequency (higher = better long context)
        scaling_type: Type of scaling ("linear", "dynamic", "yarn", None)
        scaling_factor: Scaling multiplier (>1.0 extends context)

    Returns:
        Complex tensor of shape [max_seq_len, dim/2] with rotation values
    """
    if dim % 2 != 0:
        raise ValueError(f"RoPE dimension must be even, got {dim}")
    if theta <= 0:
        raise ValueError(f"RoPE theta must be positive, got {theta}")
    if max_seq_len <= 0:
        raise ValueError(f"RoPE max_seq_len must be positive, got {max_seq_len}")

    # Calculate base frequencies: lower dimensions rotate faster
    # freqs[i] = 1 / (theta^(2i/dim))
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))

    # Apply RoPE scaling if specified
    if scaling_type == "linear":
        # Linear scaling: compress frequencies uniformly
        freqs = freqs / scaling_factor
        logger.debug(f"Applied linear RoPE scaling (factor={scaling_factor})")

    elif scaling_type == "dynamic":
        # Dynamic NTK-aware scaling: adjust theta based on extension
        # Better quality than linear for moderate extensions
        if dim < 4:
            raise ValueError(f"RoPE dynamic scaling requires dim >= 4, got {dim}")
        alpha = scaling_factor
        # Adjust base frequency with NTK-aware interpolation
        adjusted_theta = theta * (alpha ** (dim / (dim - 2)))
        freqs = 1.0 / (adjusted_theta ** (torch.arange(0, dim, 2).float() / dim))
        logger.debug(f"Applied dynamic NTK RoPE scaling (factor={scaling_factor})")

    elif scaling_type == "yarn":
        # YaRN (Yet another RoPE extensioN): Best for very long contexts
        # Uses attention-aware scaling with ramp function
        alpha = scaling_factor
        # YaRN applies different scaling to different frequency bands
        beta_fast = 32  # Low frequency threshold
        beta_slow = 1  # High frequency threshold

        # Compute frequency-dependent scaling
        dim_indices = torch.arange(0, dim, 2).float()
        # Ramp function: smoothly transition between fast and slow scaling
        denom = beta_fast / dim - beta_slow
        if abs(denom) < 1e-9:
            # When dim == beta_fast (e.g. 32), ramp is undefined —
            # fall back to uniform scaling (ramp = 0.5).
            ramp = torch.full_like(dim_indices, 0.5)
        else:
            ramp = (dim_indices / dim - beta_slow) / denom
        ramp = torch.clamp(ramp, 0, 1)

        # Apply scaled freqs with ramp
        freqs_scaled = freqs / alpha
        freqs = freqs_scaled * ramp + freqs * (1 - ramp)
        logger.debug(f"Applied YaRN RoPE scaling (factor={scaling_factor})")

    # Create position indices: [0, 1, 2, ..., max_seq_len-1]
    positions = torch.arange(max_seq_len)

    # Outer product: angles[pos, dim] = pos * freq[dim]
    angles = torch.outer(positions, freqs)

    # Convert to complex numbers for rotation: e^(i*angle) = cos(angle) + i*sin(angle)
    return torch.polar(torch.ones_like(angles), angles)


def apply_rotary_embedding(x: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
    """
    Apply rotary embeddings to Q and K tensors.

    Rotates the query/key vectors by their position: pairs of dimensions are
    treated as complex numbers, multiplied by the precomputed rotation, and
    converted back to real numbers.

    Args:
        x: Input tensor [batch, seq, heads, dim]
        freqs_cis: Precomputed rotation frequencies
        start_pos: Starting position (for KV-cache continuation)

    Returns:
        Rotated tensor, same shape as input
    """
    if not freqs_cis.is_complex():
        # Module.to(bf16/fp16) casts complex buffers real, silently dropping the
        # imaginary (sin) half -- every position rotation would be wrong with no
        # error. Fail honestly: keep weights fp32 and use autocast instead.
        raise TypeError(
            "freqs_cis is not complex (model.half()/.to(dtype) corrupted the RoPE "
            "buffer); keep the model fp32 and use torch.autocast for mixed precision"
        )
    seq_len = x.shape[1]
    end_pos = start_pos + seq_len
    if end_pos > freqs_cis.shape[0]:
        raise ValueError(
            f"freqs_cis length {freqs_cis.shape[0]} too short for start_pos={start_pos} + seq_len={seq_len} = {end_pos}"
        )
    # Get the right slice of frequencies for our positions
    freqs = freqs_cis[start_pos:end_pos]

    # Reshape x to treat pairs of dims as complex: [batch, seq, heads, dim/2, 2]
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

    # Add batch and head dimensions to freqs for broadcasting
    freqs = freqs.unsqueeze(0).unsqueeze(2)

    # Complex multiplication = rotation!
    x_rotated = x_complex * freqs

    # Convert back to real numbers and original shape
    return torch.view_as_real(x_rotated).flatten(-2).type_as(x)


class Attention(nn.Module):
    """
    Multi-Head Attention with Grouped Query Attention (GQA).

    Computes scaled dot-product attention: project the input to Q, K, V;
    scores = Q @ K.T / sqrt(dim); softmax the scores; output = scores @ V.

    GQA lets multiple query heads share one K/V head (8 Q heads with 2 KV
    heads means 4 Q heads per KV head), saving 2-4x KV memory versus one
    K/V pair per head.

    During generation only one new token arrives at a time, so K and V for
    previous tokens are cached instead of recomputed - O(n) per token
    instead of O(n^2).

    Uses RoPE (apply_rotary_embedding) for position encoding; used by
    TransformerBlock.
    """

    # Maximum KV-cache size (sliding window for memory efficiency)
    MAX_CACHE_SEQ_LEN = 4096

    def __init__(self, config: ForgeConfig) -> None:
        """
        Initialize attention layer.

        Args:
            config: Model configuration with n_heads, n_kv_heads, dim, etc.
        """
        super().__init__()
        self.n_heads = config.n_heads  # Number of query heads
        self.n_kv_heads = config.n_kv_heads  # Number of key/value heads (for GQA)
        if self.n_kv_heads > self.n_heads:
            raise ValueError(f"n_kv_heads ({self.n_kv_heads}) must be <= n_heads ({self.n_heads})")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})")
        self.head_dim = config.dim // config.n_heads  # Dimension per head
        self.n_rep = self.n_heads // self.n_kv_heads  # How many Q heads per KV head

        # Cache size follows the config's context window. The old
        # min(..., MAX_CACHE_SEQ_LEN) clamp silently capped the KV cache at
        # 4096 for any longer-context config (v2 targets 4k-8k), making the
        # cache slide and forget the earliest tokens with no error. The class
        # constant remains only as a fallback for config objects without a
        # max_seq_len attribute (2026-07-20 v2 gap audit).
        self.max_cache_len = config.max_seq_len if hasattr(config, "max_seq_len") else self.MAX_CACHE_SEQ_LEN

        # ─────────────────────────────────────────────────────────────────────
        # PROJECTION LAYERS: Transform input into Q, K, V, and output
        # ─────────────────────────────────────────────────────────────────────
        # Wq: Project to queries (one per head)
        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=config.use_bias)
        # Wk: Project to keys (fewer for GQA)
        self.wk = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=config.use_bias)
        # Wv: Project to values (same as keys)
        self.wv = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=config.use_bias)
        # Wo: Project attention output back to model dimension
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=config.use_bias)

        self.dropout = nn.Dropout(config.dropout)
        self.use_rope = config.use_rope
        self.use_qk_norm = config.use_qk_norm
        # v2 option: norm Q/K BEFORE rotating (the convention every current
        # QK-norm model uses); v1 normed after RoPE and its checkpoints must
        # keep that math, so the default stays False.
        self.qk_norm_before_rope = getattr(config, "qk_norm_before_rope", False)

        # Pre-computed attention scale — avoids math.sqrt() per forward pass
        self._scale = 1.0 / math.sqrt(self.head_dim)

        # Learned QK norms (Qwen3-style RMSNorm per head)
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        # ─────────────────────────────────────────────────────────────────────
        # KV-CACHE: Pre-allocated for O(1) per-token writes during generation
        # Uses the optimized KVCache from kv_cache.py instead of torch.cat()
        # which caused O(n) reallocation every token.
        # ─────────────────────────────────────────────────────────────────────
        self._kv_cache: Optional[object] = None

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """
        Forward pass through attention.

        Args:
            x: Input tensor [batch, seq_len, dim]
            freqs_cis: RoPE frequencies for position encoding
            mask: Attention mask (prevents looking at future tokens)
            use_cache: Whether to use/update KV-cache
            start_pos: Starting position (for cache continuation)

        Returns:
            Output tensor [batch, seq_len, dim]
        """
        B, T, _ = x.shape  # Batch, Time (seq_len), _ (dim)

        # ─────────────────────────────────────────────────────────────────────
        # STEP 1: Project input to Q, K, V
        # ─────────────────────────────────────────────────────────────────────
        q = self.wq(x).reshape(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).reshape(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).reshape(B, T, self.n_kv_heads, self.head_dim)

        # ─────────────────────────────────────────────────────────────────────
        # STEP 2: QK norm + RoPE. v1 order is rotate-then-norm (its checkpoints
        # depend on that math); the v2 convention normalizes BEFORE rotating.
        # ─────────────────────────────────────────────────────────────────────
        if self.use_qk_norm and self.qk_norm_before_rope:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.use_rope and freqs_cis is not None:
            q = apply_rotary_embedding(q, freqs_cis, start_pos)
            k = apply_rotary_embedding(k, freqs_cis, start_pos)

        # QK normalization: prevents fp16 attention overflow on long sequences
        if self.use_qk_norm and not self.qk_norm_before_rope:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # ─────────────────────────────────────────────────────────────────────
        # STEP 3: Handle KV-cache (for efficient generation)
        # ─────────────────────────────────────────────────────────────────────
        if use_cache:
            # Detach K, V from computation graph to prevent memory explosion
            # if someone accidentally backprops with use_cache=True
            k = k.detach()
            v = v.detach()

            # Lazy-init pre-allocated cache on first use
            if self._kv_cache is None:
                from enigma_engine.core.kv_cache import KVCache

                self._kv_cache = KVCache(
                    batch_size=B,
                    max_seq_len=self.max_cache_len,
                    n_kv_heads=self.n_kv_heads,
                    head_dim=self.head_dim,
                    device=k.device,
                    dtype=k.dtype,
                )

            # O(1) index write instead of O(n) torch.cat() + realloc
            self._kv_cache.update(k, v)
            k, v = self._kv_cache.get()

        # ─────────────────────────────────────────────────────────────────────
        # STEP 4: Repeat K, V for GQA (if using fewer KV heads)
        # ─────────────────────────────────────────────────────────────────────
        if self.n_rep > 1:
            # Each KV head serves multiple Q heads
            k = k.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)

        # ─────────────────────────────────────────────────────────────────────
        # STEP 5: Compute attention (SDPA or Standard)
        # ─────────────────────────────────────────────────────────────────────
        if hasattr(F, "scaled_dot_product_attention") and x.is_cuda:
            # ─────────────────────────────────────────────────────────────────
            # SDPA PATH: PyTorch 2.0+ built-in, auto-dispatches to
            # Flash/xFormers/math backend.  Free 2-4x speedup when
            # flash-attn package is not installed.
            # ─────────────────────────────────────────────────────────────────
            q_s = q.transpose(1, 2)  # [B, heads, T, dim]
            k_s = k.transpose(1, 2)
            v_s = v.transpose(1, 2)
            drop_p = self.dropout.p if self.training else 0.0
            if mask is not None:
                output = F.scaled_dot_product_attention(
                    q_s, k_s, v_s, attn_mask=mask, dropout_p=drop_p, scale=self._scale
                )
            elif q_s.shape[-2] == k_s.shape[-2]:
                # Square attention (prefill / training): plain causal via the fast kernel.
                output = F.scaled_dot_product_attention(
                    q_s, k_s, v_s, is_causal=True, dropout_p=drop_p, scale=self._scale
                )
            else:
                # KV-cache incremental decode: q_len < k_len. The q_len new queries are
                # the LAST q_len positions of the length-k_len sequence, so query i
                # (absolute pos k_len-q_len+i) may attend to keys 0..k_len-q_len+i.
                # is_causal=True here top-left-aligns the (q_len, k_len) mask and wrongly
                # leaves each query able to see ONLY key 0 — silent KV-cache corruption
                # that makes served generation collapse. Build the bottom-right-aligned
                # causal mask instead (for q_len==1 this is all-True = attend to full cache).
                Tq, Tk = q_s.shape[-2], k_s.shape[-2]
                if Tq == 1:
                    # The one new query attends to the entire cache, so the
                    # bottom-right mask below is all-True. Building it every
                    # layer every token costs kernels for nothing, and passing
                    # ANY attn_mask disqualifies the flash backend.
                    output = F.scaled_dot_product_attention(
                        q_s, k_s, v_s, dropout_p=drop_p, scale=self._scale
                    )
                else:
                    attn_causal = torch.ones(Tq, Tk, dtype=torch.bool, device=q_s.device).tril(diagonal=Tk - Tq)
                    output = F.scaled_dot_product_attention(
                        q_s, k_s, v_s, attn_mask=attn_causal, dropout_p=drop_p, scale=self._scale
                    )
            output = output.transpose(1, 2).reshape(B, T, -1)
        else:
            # ─────────────────────────────────────────────────────────────────
            # STANDARD ATTENTION PATH: Works everywhere (CPU, MPS, any dtype)
            # ─────────────────────────────────────────────────────────────────
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

            scores = torch.matmul(q, k.transpose(-2, -1)) * self._scale
            if mask is not None:
                scores = scores + mask  # Mask is -inf for blocked positions
            elif T > 1:
                # Plain causal case: the main model now passes mask=None so the SDPA
                # path can use is_causal=True. This non-SDPA / CPU fallback must apply
                # causality itself. (T==1 decode needs none: the 1 query sees all keys.)
                #
                # The mask must be BOTTOM-RIGHT aligned against the key axis, which is
                # longer than T whenever a cached continuation feeds >1 new token
                # (T queries against T_k = cache + T keys). A square (T, T) mask used
                # to broadcast-crash there; query i is at absolute position T_k-T+i and
                # may attend to keys 0..T_k-T+i, which is exactly tril(diagonal=T_k-T).
                # Square prefill (T_k == T) reduces to the plain triu mask.
                # One allocation + one in-place op: -inf strictly above the
                # bottom-right-aligned diagonal (blocked j >= i + T_k-T+1,
                # identical to the tril form for every T_k >= T; the square
                # case is the pre-cache triu(full(-inf), 1) idiom).
                T_k = scores.shape[-1]
                causal = torch.full((T, T_k), float("-inf"), device=scores.device, dtype=scores.dtype).triu_(
                    diagonal=T_k - T + 1
                )
                scores = scores + causal

            # Softmax and dropout, then weighted sum of values
            attn = self.dropout(F.softmax(scores, dim=-1))
            output = torch.matmul(attn, v)

            # Reshape back: [batch, heads, seq, dim] -> [batch, seq, heads*dim]
            output = output.transpose(1, 2).reshape(B, T, -1)

        # ─────────────────────────────────────────────────────────────────────
        # STEP 6: Project back to model dimension
        # ─────────────────────────────────────────────────────────────────────
        return self.wo(output)

    def clear_cache(self) -> None:
        """Clear the KV-cache (call between different sequences)."""
        if self._kv_cache is not None:
            self._kv_cache.clear()
        self._kv_cache = None

    def rewind_cache(self, position: int) -> None:
        """Truncate KV-cache back to *position* (keep cache alive)."""
        if self._kv_cache is not None:
            self._kv_cache.rewind_to(position)


class FeedForward(nn.Module):
    """
    SwiGLU Feed-Forward Network.

    Standard FFN: output = W2(ReLU(W1(x)))
    SwiGLU:       output = W2(Swish(W1(x)) * W3(x))

    Swish is smoother than ReLU and the W3 gating helps information flow;
    SwiGLU is empirically shown to train faster and reach lower loss.

    Used by TransformerBlock after attention.
    """

    def __init__(self, config: ForgeConfig) -> None:
        """
        Args:
            config: Model config with dim, hidden_dim, use_swiglu flag
        """
        super().__init__()
        self.use_swiglu = config.use_swiglu

        if self.use_swiglu:
            # ─────────────────────────────────────────────────────────────────
            # SWIGLU: 3 linear layers
            # ─────────────────────────────────────────────────────────────────
            # W1: Projects to hidden dim (for the gate)
            self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=config.use_bias)
            # W2: Projects back to model dim
            self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=config.use_bias)
            # W3: Projects to hidden dim (for the value)
            self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=config.use_bias)
        else:
            # ─────────────────────────────────────────────────────────────────
            # STANDARD FFN: 2 linear layers with ReLU
            # ─────────────────────────────────────────────────────────────────
            self.up = nn.Linear(config.dim, config.hidden_dim, bias=config.use_bias)
            self.down = nn.Linear(config.hidden_dim, config.dim, bias=config.use_bias)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feed-forward network.

        SwiGLU computation:
        1. gate = swish(W1 @ x)
        2. value = W3 @ x
        3. hidden = gate * value
        4. output = W2 @ hidden
        """
        if self.use_swiglu:
            # SwiGLU: swish(W1(x)) * W3(x), then W2
            # F.silu = swish = x * sigmoid(x)
            return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))
        # Standard FFN: GELU(W1(x)), then W2
        return self.down(self.dropout(F.gelu(self.up(x))))


class TransformerBlock(nn.Module):
    """
    Single Transformer block with pre-norm architecture.

    One layer of the transformer - stack N of these for the full model.

    Pre-norm architecture:
    x → [Norm] → [Attention] → + → [Norm] → [FFN] → + → output
         │                     ↑         │           ↑
         └─────────────────────┘         └───────────┘
              (residual skip)           (residual skip)

    Pre-norm (norm before attention, the modern convention) trains more
    stably than the original post-norm ordering, especially in deep models.
    The residual skips (the + signs) let gradients flow directly through
    the network; without them deep networks are nearly impossible to train.

    Gradient checkpointing, when enabled, recomputes activations during the
    backward pass instead of storing them - trades ~30% compute for ~50%
    memory savings, essential for training large models on limited hardware.
    """

    def __init__(self, config: ForgeConfig, layer_id: int) -> None:
        """
        Args:
            config: Model configuration
            layer_id: Which layer this is (for debugging/logging)
        """
        super().__init__()
        self.layer_id = layer_id
        self.use_checkpoint = config.use_gradient_checkpointing

        # Choose normalization type based on config
        Norm = RMSNorm if config.use_rms_norm else nn.LayerNorm

        # Two normalizations: one before attention, one before FFN
        self.attention_norm = Norm(config.dim)
        self.ffn_norm = Norm(config.dim)

        # Peri-LN (v2 option): ALSO normalize each sub-block's OUTPUT inside
        # the residual -- h = x + norm_out(block(norm_in(x))). Every shipped
        # small model converged there (2026-07-18 research verdicts); "pre"
        # (v1 default) adds no parameters, so old checkpoints load unchanged.
        self.norm_scheme = getattr(config, "norm_scheme", "pre")
        if self.norm_scheme == "peri":
            self.attention_post_norm = Norm(config.dim)
            self.ffn_post_norm = Norm(config.dim)

        # The actual computation modules
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)

        # LayerScale: learnable per-channel scaling of residual outputs
        # Initialized to a tiny value so early training is stable
        self.use_layer_scale = config.use_layer_scale
        if self.use_layer_scale:
            self.ls_attn = nn.Parameter(torch.full((config.dim,), 1e-5))
            self.ls_ffn = nn.Parameter(torch.full((config.dim,), 1e-5))

        # Drop path (stochastic depth): linearly increasing per layer
        drop_rate = config.drop_path_rate
        n_layers = config.n_layers
        layer_drop = drop_rate * layer_id / max(n_layers - 1, 1) if drop_rate > 0 else 0.0
        self.drop_path_attn = DropPath(layer_drop)
        self.drop_path_ffn = DropPath(layer_drop)

    def _forward_impl(
        self,
        x: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """Internal forward implementation (used by checkpointing)."""
        # Attention sub-layer with residual connection
        attn_out = self.attention(self.attention_norm(x), freqs_cis, mask, use_cache, start_pos)
        if self.norm_scheme == "peri":
            attn_out = self.attention_post_norm(attn_out)
        if self.use_layer_scale:
            attn_out = attn_out * self.ls_attn
        h = x + self.drop_path_attn(attn_out)

        # Standard FFN sub-layer with residual connection
        ffn_out = self.feed_forward(self.ffn_norm(h))
        if self.norm_scheme == "peri":
            ffn_out = self.ffn_post_norm(ffn_out)
        if self.use_layer_scale:
            ffn_out = ffn_out * self.ls_ffn
        return h + self.drop_path_ffn(ffn_out)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """
        Forward pass: Norm → Attention → Add → Norm → FFN → Add

        Uses gradient checkpointing during training if enabled, which
        recomputes activations during backward pass to save memory.

        Args:
            x: Input [batch, seq_len, dim]
            freqs_cis: RoPE frequencies
            mask: Causal attention mask
            use_cache: Whether to use KV-cache
            start_pos: Position for KV-cache

        Returns:
            Output tensor, same shape as input
        """
        # Use gradient checkpointing during training if enabled
        # Don't use with KV-cache as it doesn't make sense (inference only)
        if self.use_checkpoint and self.training and not use_cache:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl,
                x,
                freqs_cis,
                mask,
                use_cache,
                start_pos,
                use_reentrant=False,  # Recommended for newer PyTorch
            )
        return self._forward_impl(x, freqs_cis, mask, use_cache, start_pos)

    def clear_cache(self) -> None:
        """Clear KV-cache in the attention layer."""
        self.attention.clear_cache()

    def rewind_cache(self, position: int) -> None:
        """Truncate KV-cache back to *position*."""
        self.attention.rewind_cache(position)
