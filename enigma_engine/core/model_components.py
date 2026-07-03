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

# ─────────────────────────────────────────────────────────────────────────────
# FLASH ATTENTION: Optional high-performance attention (2-4x faster)
# ─────────────────────────────────────────────────────────────────────────────
# Requires: pip install flash-attn (CUDA only, Ampere+ GPU recommended)
# Falls back silently to standard attention if not available.
try:
    from flash_attn import flash_attn_func

    HAS_FLASH_ATTN = True
    logger.info("Flash Attention available - will use for fp16/bf16 CUDA tensors")
except ImportError:
    HAS_FLASH_ATTN = False
    flash_attn_func = None  # type: ignore


# =============================================================================
# 🧱 MODEL COMPONENTS - The Building Blocks
# =============================================================================
# These are the LEGO pieces that build the full transformer.
# Each class is a specific neural network layer with a special purpose.


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization - faster than LayerNorm!

    📖 WHAT THIS DOES:
    Normalizes the input to have consistent scale. Like adjusting volume
    on speakers so nothing is too loud or too quiet.

    📐 THE MATH (simplified):
    1. Calculate RMS: sqrt(mean(x²))
    2. Divide x by RMS (now values are normalized)
    3. Multiply by learned weight (model learns optimal scale)

    💡 WHY RMSNorm INSTEAD OF LAYERNORM?
    LayerNorm: Subtracts mean, divides by std (2 stats to compute)
    RMSNorm: Just divides by RMS (1 stat to compute)
    Result: Same quality, ~10% faster!

    🔗 USED BY:
      ← TransformerBlock uses this before attention and FFN
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
# 🌀 ROTARY POSITION EMBEDDINGS (RoPE) - How the model knows word order
# =============================================================================
# Without position info, "dog bites man" = "man bites dog" to the model!
# RoPE encodes position by ROTATING the vectors - elegant and effective.


def precompute_rope_frequencies(
    dim: int, max_seq_len: int, theta: float = 10000.0, scaling_type: Optional[str] = None, scaling_factor: float = 1.0
) -> torch.Tensor:
    """
    Precompute RoPE frequencies for all positions with optional scaling.

    📖 WHAT THIS DOES:
    Creates a table of rotation angles for each position and dimension.
    These rotations encode "position 0", "position 1", etc.
    With scaling, extends context length beyond training length.

    📐 THE MATH:
    For dimension pair i, frequency = 1 / (theta^(2i/dim))
    For position p, angle = p * frequency

    🎯 ROPE SCALING:
    - linear: freqs = freqs / scaling_factor (simple compression)
    - dynamic: Adaptive NTK-aware scaling (better quality)
    - yarn: Yet another RoPE extension (best for very long contexts)

    💡 WHY THIS WORKS:
    - Different dimensions get different rotation speeds
    - Position 5 at dim 0 rotates differently than position 5 at dim 10
    - Model can learn to "read" these rotations to understand order
    - Scaling lets model handle longer contexts than it was trained on

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

    📖 WHAT THIS DOES:
    Rotates the query/key vectors based on their position.
    This lets the model know "this word is at position 5" vs "position 10".

    📐 HOW IT WORKS:
    1. Treat pairs of dimensions as complex numbers
    2. Multiply by rotation (complex multiplication = rotation!)
    3. Convert back to real numbers

    Args:
        x: Input tensor [batch, seq, heads, dim]
        freqs_cis: Precomputed rotation frequencies
        start_pos: Starting position (for KV-cache continuation)

    Returns:
        Rotated tensor, same shape as input
    """
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

    📖 WHAT THIS DOES:
    Attention is how the model "looks at" different parts of the input.
    "The cat sat on the mat" - when processing "sat", attention lets
    the model look back at "cat" to know WHO sat.

    📐 THE MATH (simplified):
    1. Create Query (Q), Key (K), Value (V) from input
    2. Attention scores = Q @ K.T / sqrt(dim)  (which words to look at?)
    3. Softmax → probabilities (normalize scores)
    4. Output = scores @ V  (weighted combination of values)

    ⚡ GROUPED QUERY ATTENTION (GQA):
    Normal: Each head has its own K and V (memory hungry!)
    GQA: Multiple Q heads share the same K,V (saves 2-4x memory!)

    Example: 8 Q heads, 2 KV heads → 4 Q heads share each KV head

    💾 KV-CACHE:
    During generation, we only add ONE new token at a time.
    Instead of recomputing K,V for all previous tokens, we cache them!
    This makes generation O(n) instead of O(n²) - HUGE speedup!

    🔗 CONNECTS TO:
      → Uses RoPE (apply_rotary_embedding) for position encoding
      ← Used by TransformerBlock
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

        # Cache size limit from config or default
        self.max_cache_len = min(
            config.max_seq_len if hasattr(config, "max_seq_len") else self.MAX_CACHE_SEQ_LEN, self.MAX_CACHE_SEQ_LEN
        )

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

        # Pre-computed attention scale — avoids math.sqrt() per forward pass
        self._scale = 1.0 / math.sqrt(self.head_dim)

        # Learned QK norms (Qwen3-style RMSNorm per head)
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        # R22: Differential attention — split heads into two groups and
        # subtract: attn = softmax(Q1@K1^T) - λ * softmax(Q2@K2^T).
        # Cancels noise / uninformative attention mass.
        self.use_differential_attn = config.use_differential_attn
        if self.use_differential_attn and self.n_heads >= 2 and self.n_heads % 2 == 0:
            # Per-head learnable lambda (initialized near zero so early
            # training behaves close to standard attention).
            self._diff_lambda = nn.Parameter(torch.full((self.n_heads // 2,), 0.05))
        else:
            self.use_differential_attn = False

        # ─────────────────────────────────────────────────────────────────────
        # KV-CACHE: Pre-allocated for O(1) per-token writes during generation
        # Uses the optimized KVCache from kv_cache.py instead of torch.cat()
        # which caused O(n) reallocation every token.
        # ─────────────────────────────────────────────────────────────────────
        self._kv_cache: Optional[object] = None

        # T3-1: Cross-layer KV sharing (YOCO-style)
        # Set by Enigma.__init__() to point followers at the leader's Attention.
        self._kv_share_source: Optional["Attention"] = None
        self._shared_kv: Optional[tuple[torch.Tensor, torch.Tensor]] = None

        # T5-6: Multi-Head Latent Attention (MLA) — low-rank KV compression
        # Instead of projecting dim → n_kv_heads*head_dim directly,
        # factor through a smaller latent: dim → latent → K, V.
        # Reduces parameter count and acts as regularization bottleneck.
        self._use_mla = config.mla_latent_dim > 0
        if self._use_mla:
            ld = config.mla_latent_dim
            self.wkv_down = nn.Linear(config.dim, ld, bias=False)
            self.wk_up = nn.Linear(ld, self.n_kv_heads * self.head_dim, bias=False)
            self.wv_up = nn.Linear(ld, self.n_kv_heads * self.head_dim, bias=False)

        # T3-8: LongLoRA shifted sparse attention
        self.use_shifted_attention = config.use_shifted_attention
        self._shifted_group_size = config.shifted_group_size

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
        # T3-1: Cross-layer KV sharing — follower layers skip K, V
        # projection and reuse the leader layer's K, V.
        # ─────────────────────────────────────────────────────────────────────
        if self._kv_share_source is not None:
            q = self.wq(x).reshape(B, T, self.n_heads, self.head_dim)
            if self.use_rope and freqs_cis is not None:
                q = apply_rotary_embedding(q, freqs_cis, start_pos)
            if self.use_qk_norm:
                q = self.q_norm(q)

            # BUG-1 fix: source may not have run yet on this forward
            # (first forward, after clear_cache, or out-of-order layer
            # config). _kv_cache lazy-inits in the leader's STEP 3
            # block, and _shared_kv is set only in the leader's
            # training-mode branch. Either can be None when the
            # follower reaches this point — compute K, V locally as a
            # fallback rather than crashing.
            source = self._kv_share_source
            source_cache = source._kv_cache if use_cache else None
            source_shared = source._shared_kv if not use_cache else None
            if (use_cache and source_cache is not None) or (not use_cache and source_shared is not None):
                if use_cache:
                    k, v = source_cache.get()
                else:
                    k, v = source_shared
            else:
                # Source not warm yet — compute K, V from this
                # follower's own wk/wv projection weights. Mirrors the
                # leader path's STEP 1 + STEP 2 logic (MLA-aware).
                if self._use_mla:
                    kv_latent = self.wkv_down(x)
                    k = self.wk_up(kv_latent).reshape(B, T, self.n_kv_heads, self.head_dim)
                    v = self.wv_up(kv_latent).reshape(B, T, self.n_kv_heads, self.head_dim)
                else:
                    k = self.wk(x).reshape(B, T, self.n_kv_heads, self.head_dim)
                    v = self.wv(x).reshape(B, T, self.n_kv_heads, self.head_dim)
                if self.use_rope and freqs_cis is not None:
                    k = apply_rotary_embedding(k, freqs_cis, start_pos)
                if self.use_qk_norm:
                    k = self.k_norm(k)
        else:
            # ─────────────────────────────────────────────────────────────────
            # STEP 1: Project input to Q, K, V
            # ─────────────────────────────────────────────────────────────────
            q = self.wq(x).reshape(B, T, self.n_heads, self.head_dim)
            if self._use_mla:
                # T5-6: MLA — factored KV through latent bottleneck
                kv_latent = self.wkv_down(x)
                k = self.wk_up(kv_latent).reshape(B, T, self.n_kv_heads, self.head_dim)
                v = self.wv_up(kv_latent).reshape(B, T, self.n_kv_heads, self.head_dim)
            else:
                k = self.wk(x).reshape(B, T, self.n_kv_heads, self.head_dim)
                v = self.wv(x).reshape(B, T, self.n_kv_heads, self.head_dim)

            # ─────────────────────────────────────────────────────────────────
            # STEP 2: Apply RoPE position embeddings to Q and K
            # ─────────────────────────────────────────────────────────────────
            if self.use_rope and freqs_cis is not None:
                q = apply_rotary_embedding(q, freqs_cis, start_pos)
                k = apply_rotary_embedding(k, freqs_cis, start_pos)

            # QK normalization: prevents fp16 attention overflow on long sequences
            if self.use_qk_norm:
                q = self.q_norm(q)
                k = self.k_norm(k)

            # ─────────────────────────────────────────────────────────────────
            # STEP 3: Handle KV-cache (for efficient generation)
            # ─────────────────────────────────────────────────────────────────
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
            else:
                # T3-1: Store K, V for follower layers (training mode)
                self._shared_kv = (k, v)

        # ─────────────────────────────────────────────────────────────────────
        # STEP 4: Repeat K, V for GQA (if using fewer KV heads)
        # ─────────────────────────────────────────────────────────────────────
        if self.n_rep > 1:
            # Each KV head serves multiple Q heads
            k = k.repeat_interleave(self.n_rep, dim=2)
            v = v.repeat_interleave(self.n_rep, dim=2)

        # ─────────────────────────────────────────────────────────────────────
        # T3-8: LongLoRA shifted sparse attention (training only)
        # ─────────────────────────────────────────────────────────────────────
        if self.use_shifted_attention and T > 1 and not use_cache and T > self._shifted_group_size:
            output = self._shifted_sparse_attention(q, k, v, B, T)
            return self.wo(output)

        # ─────────────────────────────────────────────────────────────────────
        # STEP 5: Compute attention (Flash or Standard)
        # ─────────────────────────────────────────────────────────────────────
        # Flash Attention conditions (all must be true):
        #   1. flash_attn package is installed (pip install flash-attn)
        #   2. Running on CUDA GPU
        #   3. Using half precision (fp16 or bf16) - Flash doesn't support fp32
        #   4. NOT using KV-cache (Flash doesn't support incremental decoding)
        #   5. Processing full sequence (not continuing from cached K/V)
        #
        # ⚠️ IMPORTANT LIMITATION:
        # Flash Attention is DISABLED during generation (use_cache=True) because:
        #   - Flash computes the full attention matrix efficiently but atomically
        #   - Incremental KV-cache decoding needs to attend to cached K/V
        #   - This is a fundamental limitation, not a bug
        #
        # Flash is used during: Training, prompt encoding, non-cached inference
        # Flash is NOT used during: Token-by-token generation with KV-cache
        #
        # Performance impact: Training gets 2-4x speedup. Generation uses standard
        # attention which is still efficient due to KV-cache (O(1) per token).
        use_flash = (
            HAS_FLASH_ATTN
            and x.is_cuda
            and x.dtype in (torch.float16, torch.bfloat16)
            and not use_cache  # Flash doesn't support incremental decode
            and (mask is None or T == k.shape[1])  # Full sequence, not cached
        )

        if use_flash:
            # ─────────────────────────────────────────────────────────────────
            # FLASH ATTENTION PATH: O(n) memory, 2-4x faster
            # ─────────────────────────────────────────────────────────────────
            # flash_attn expects [batch, seq, heads, dim] - we already have that!
            # It handles causal masking internally with is_causal=True
            output = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.dropout.p if self.training else 0.0,
                causal=True,  # Autoregressive masking
                softmax_scale=self._scale,
            )
            # output is [batch, seq, heads, dim], need [batch, seq, dim]
            output = output.reshape(B, T, -1)
        elif not self.use_differential_attn and hasattr(F, "scaled_dot_product_attention") and x.is_cuda:
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
                attn_causal = torch.ones(Tq, Tk, dtype=torch.bool, device=q_s.device).tril(diagonal=Tk - Tq)
                output = F.scaled_dot_product_attention(
                    q_s, k_s, v_s, attn_mask=attn_causal, dropout_p=drop_p, scale=self._scale
                )
            output = output.transpose(1, 2).reshape(B, T, -1)
        else:
            # ─────────────────────────────────────────────────────────────────
            # STANDARD ATTENTION PATH: Works everywhere (CPU, MPS, any dtype)
            # ─────────────────────────────────────────────────────────────────
            # Transpose for batched matrix multiply: [batch, heads, seq, dim]
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

            # scores = Q @ K.T / sqrt(head_dim) - scaled dot-product attention
            scores = torch.matmul(q, k.transpose(-2, -1)) * self._scale
            if mask is not None:
                scores = scores + mask  # Mask is -inf for blocked positions
            elif T > 1:
                # Plain causal case: the main model now passes mask=None so the SDPA
                # path can use is_causal=True. This non-SDPA / CPU fallback must apply
                # causality itself. (T==1 decode needs none: the 1 query sees all keys.)
                causal = torch.triu(
                    torch.full((T, T), float("-inf"), device=scores.device, dtype=scores.dtype),
                    diagonal=1,
                )
                scores = scores + causal

            # R22: Differential attention — noise cancellation.
            if self.use_differential_attn:
                # Group 1 (even heads) and Group 2 (odd heads)
                s1 = scores[:, 0::2, :, :]  # (B, half_h, T, Tk)
                s2 = scores[:, 1::2, :, :]  # (B, half_h, T, Tk)
                a1 = F.softmax(s1, dim=-1)
                a2 = F.softmax(s2, dim=-1)
                lam = torch.sigmoid(self._diff_lambda)  # (half_h,)
                lam = lam[None, :, None, None]  # broadcast
                diff_attn = self.dropout(a1 - lam * a2)  # (B, half_h, T, Tk)
                # Apply to V: average even/odd V heads
                v1 = v[:, 0::2, :, :]
                v2 = v[:, 1::2, :, :]
                diff_attn = diff_attn.to(v1.dtype)
                out1 = torch.matmul(diff_attn, v1)
                out2 = torch.matmul(diff_attn, v2)
                # Interleave back to full head count
                output = torch.stack([out1, out2], dim=2).reshape(B, self.n_heads, T, self.head_dim)
            else:
                # Softmax and dropout, then weighted sum of values
                attn = self.dropout(F.softmax(scores, dim=-1))
                output = torch.matmul(attn, v)

            # Reshape back: [batch, heads, seq, dim] -> [batch, seq, heads*dim]
            output = output.transpose(1, 2).reshape(B, T, -1)

        # ─────────────────────────────────────────────────────────────────────
        # STEP 6: Project back to model dimension
        # ─────────────────────────────────────────────────────────────────────
        return self.wo(output)

    def _shifted_sparse_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        B: int,
        T: int,
    ) -> torch.Tensor:
        """LongLoRA shifted sparse attention (T3-8).

        Both head groups use local chunked attention (O(T * group_size)
        instead of O(T^2)). Group B shifts by half the group size so
        overlapping windows provide cross-boundary information flow.

        Note: the circular shift creates a minor causality edge case in
        Group B's first chunk. This is standard LongLoRA behavior — the
        unshifted Group A heads maintain strict causality.

        Args:
            q, k, v: [B, T, n_heads, head_dim] after GQA expansion.
            B: Batch size.
            T: Sequence length (must be > group_size).

        Returns:
            [B, T, dim] attention output.
        """
        gs = self._shifted_group_size
        shift = gs // 2
        half = self.n_heads // 2
        n_hb = self.n_heads - half

        # Pad T to multiple of group_size
        pad = (gs - T % gs) % gs
        if pad > 0:
            q = F.pad(q, (0, 0, 0, 0, 0, pad))
            k = F.pad(k, (0, 0, 0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, 0, 0, pad))
        T_padded = q.shape[1]
        n_chunks = T_padded // gs

        # Split heads: Group A (normal), Group B (shifted)
        q_a, q_b = q[:, :, :half], q[:, :, half:]
        k_a, k_b = k[:, :, :half], k[:, :, half:]
        v_a, v_b = v[:, :, :half], v[:, :, half:]

        # Shift Group B by half the group size
        q_b = torch.roll(q_b, shifts=shift, dims=1)
        k_b = torch.roll(k_b, shifts=shift, dims=1)
        v_b = torch.roll(v_b, shifts=shift, dims=1)

        # Reshape to chunks: [B, T_p, n_h, D] -> [B*n_chunks, gs, n_h, D]
        def to_chunks(t: torch.Tensor) -> torch.Tensor:
            return t.reshape(B, n_chunks, gs, -1, self.head_dim).reshape(B * n_chunks, gs, -1, self.head_dim)

        q_a, k_a, v_a = to_chunks(q_a), to_chunks(k_a), to_chunks(v_a)
        q_b, k_b, v_b = to_chunks(q_b), to_chunks(k_b), to_chunks(v_b)

        # Causal mask for each local chunk
        chunk_mask = torch.triu(
            torch.full((gs, gs), float("-inf"), device=q.device, dtype=q.dtype),
            diagonal=1,
        )
        scale = self._scale

        # Local attention per group
        def local_attn(qg: torch.Tensor, kg: torch.Tensor, vg: torch.Tensor) -> torch.Tensor:
            qg = qg.transpose(1, 2)  # [B*C, n_h, gs, D]
            kg = kg.transpose(1, 2)
            vg = vg.transpose(1, 2)
            scores = torch.matmul(qg, kg.transpose(-2, -1)) * scale
            scores = scores + chunk_mask
            attn = self.dropout(F.softmax(scores, dim=-1))
            out = torch.matmul(attn, vg)
            return out.transpose(1, 2)  # [B*C, gs, n_h, D]

        out_a = local_attn(q_a, k_a, v_a)
        out_b = local_attn(q_b, k_b, v_b)

        # Un-chunk: [B*n_chunks, gs, n_h, D] -> [B, T_padded, n_h, D]
        def from_chunks(t: torch.Tensor, n_h: int) -> torch.Tensor:
            return t.reshape(B, n_chunks, gs, n_h, self.head_dim).reshape(B, T_padded, n_h, self.head_dim)

        out_a = from_chunks(out_a, half)
        out_b = from_chunks(out_b, n_hb)

        # Unshift Group B
        out_b = torch.roll(out_b, shifts=-shift, dims=1)

        # Remove padding and combine
        if pad > 0:
            out_a = out_a[:, :T]
            out_b = out_b[:, :T]

        output = torch.cat([out_a, out_b], dim=2)  # [B, T, n_heads, D]
        return output.reshape(B, T, -1)

    def clear_cache(self) -> None:
        """Clear the KV-cache (call between different sequences)."""
        if self._kv_cache is not None:
            self._kv_cache.clear()
        self._kv_cache = None
        self._shared_kv = None

    def rewind_cache(self, position: int) -> None:
        """Truncate KV-cache back to *position* (keep cache alive)."""
        if self._kv_cache is not None:
            self._kv_cache.rewind_to(position)
            self._shared_kv = None  # invalidate expanded view


class FeedForward(nn.Module):
    """
    SwiGLU Feed-Forward Network.

    📖 WHAT THIS DOES:
    After attention decides WHAT to look at, the FFN decides
    WHAT TO DO with that information. It's the "thinking" part!

    📐 SWIGLU FORMULA:
    Standard FFN: output = W2(ReLU(W1(x)))
    SwiGLU:       output = W2(Swish(W1(x)) * W3(x))

    💡 WHY SWIGLU IS BETTER:
    - Swish activation is smoother than ReLU (no hard corners)
    - Gating mechanism (the W3 multiplication) helps information flow
    - Empirically shown to train faster and achieve lower loss

    🔗 CONNECTS TO:
      ← Used by TransformerBlock after attention
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

        📐 SwiGLU computation:
        1. gate = swish(W1 @ x)  ← Smooth activation
        2. value = W3 @ x        ← Unactivated projection
        3. hidden = gate * value ← Gated combination
        4. output = W2 @ hidden  ← Project back

        The "gating" (multiplication) is what makes SwiGLU special!
        """
        if self.use_swiglu:
            # SwiGLU: swish(W1(x)) * W3(x), then W2
            # F.silu = swish = x * sigmoid(x)
            return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))
        # Standard FFN: GELU(W1(x)), then W2
        return self.down(self.dropout(F.gelu(self.up(x))))


# =============================================================================
# T5-4: Token Merging (ToMe) — merge similar tokens mid-forward-pass
# =============================================================================


def _bipartite_soft_matching(
    metric: torch.Tensor,
    r: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Bipartite soft matching for Token Merging (Bolya et al. 2023).

    Splits tokens into two sets (even/odd indices), finds the most
    similar pairs, and merges the top-r pairs.

    Args:
        metric: (B, T, D) normalized token features for similarity.
        r: Number of tokens to merge (remove).

    Returns:
        merge_dst: (B, T) indices mapping merged tokens to destinations.
        merge_dst: Duplicate of first value (unmerge uses merge_dst directly).
        merged_mask: (B, T) boolean mask — True for tokens that were merged.
    """
    B, T, D = metric.shape
    if r <= 0 or T <= 2:
        idx = torch.arange(T, device=metric.device).unsqueeze(0).expand(B, -1)
        return idx, idx, torch.zeros(B, T, dtype=torch.bool, device=metric.device)

    # O(T²) similarity matrix — skip for very long sequences to avoid OOM
    _TOME_MAX_TOKENS = 4096
    if T > _TOME_MAX_TOKENS:
        idx = torch.arange(T, device=metric.device).unsqueeze(0).expand(B, -1)
        return idx, idx, torch.zeros(B, T, dtype=torch.bool, device=metric.device)

    r = min(r, T // 2)

    # Split into two alternating sets: A (even) and B (odd)
    a_idx = torch.arange(0, T, 2, device=metric.device)  # [T//2]
    b_idx = torch.arange(1, T, 2, device=metric.device)  # [T//2]

    a_tokens = metric[:, a_idx]  # (B, |A|, D)
    b_tokens = metric[:, b_idx]  # (B, |B|, D)

    # Similarity: dot product of L2-normalized features
    scores = torch.bmm(a_tokens, b_tokens.transpose(1, 2))  # (B, |A|, |B|)

    # For each A token, find its most similar B token
    max_scores, most_similar = scores.max(dim=-1)  # (B, |A|)

    # Pick the top-r A tokens by max similarity
    _, top_r_idx = max_scores.topk(r, dim=-1)  # (B, r)

    # Build merge mapping: merged A-tokens map to their most-similar B-token
    merge_dst = torch.arange(T, device=metric.device).unsqueeze(0).expand(B, -1).clone()

    merged_mask = torch.zeros(B, T, dtype=torch.bool, device=metric.device)
    for b in range(B):
        for i in range(r):
            a_pos = a_idx[top_r_idx[b, i]]
            b_pos = b_idx[most_similar[b, top_r_idx[b, i]]]
            merge_dst[b, a_pos] = b_pos
            merged_mask[b, a_pos] = True

    return merge_dst, merge_dst, merged_mask


def _tome_merge(
    x: torch.Tensor,
    merged_mask: torch.Tensor,
) -> torch.Tensor:
    """Merge tokens: average merged source into destination, remove sources.

    Args:
        x: (B, T, D) input tensor.
        merged_mask: (B, T) boolean — True for tokens to remove (sources).

    Returns:
        (B, T', D) tensor with merged tokens removed.
    """
    B, T, D = x.shape
    # For simplicity, just remove the merged tokens (source tokens)
    # The destination tokens already contain similar information
    keep_mask = ~merged_mask
    # Collect kept tokens per batch
    max_kept = keep_mask.sum(dim=1).max().item()
    result = x.new_zeros(B, max_kept, D)
    for b in range(B):
        kept = x[b, keep_mask[b]]
        result[b, : kept.shape[0]] = kept
    return result


def _tome_unmerge(
    x: torch.Tensor,
    original_len: int,
    merged_mask: torch.Tensor,
    merge_dst: torch.Tensor,
) -> torch.Tensor:
    """Restore merged tokens by copying from their destinations.

    Args:
        x: (B, T', D) tensor after processing.
        original_len: Original sequence length T.
        merged_mask: (B, T) — True for merged (removed) tokens.
        merge_dst: (B, T) — destination index for each merged token.

    Returns:
        (B, T, D) tensor with original sequence length restored.
    """
    B, _, D = x.shape
    result = x.new_zeros(B, original_len, D)
    for b in range(B):
        keep_positions = (~merged_mask[b]).nonzero(as_tuple=True)[0]
        # Place kept tokens back
        for i, pos in enumerate(keep_positions):
            if i < x.shape[1]:
                result[b, pos] = x[b, i]
        # Copy merged tokens from their destinations
        merge_positions = merged_mask[b].nonzero(as_tuple=True)[0]
        for pos in merge_positions:
            dst = merge_dst[b, pos].item()
            result[b, pos] = result[b, dst]
    return result


class TransformerBlock(nn.Module):
    """
    Single Transformer block with pre-norm architecture.

    📖 WHAT THIS DOES:
    One "layer" of the transformer - stack N of these for the full model.

    📐 PRE-NORM ARCHITECTURE (better than original post-norm!):
    x → [Norm] → [Attention] → + → [Norm] → [FFN] → + → output
         │                     ↑         │           ↑
         └─────────────────────┘         └───────────┘
              (residual skip)           (residual skip)

    💡 WHY PRE-NORM?
    Original transformers: Attention → Norm (post-norm)
    Modern transformers: Norm → Attention (pre-norm)
    Pre-norm is more stable during training, especially for deep models!

    ⚡ RESIDUAL CONNECTIONS (the + signs):
    Skip connections let gradients flow directly through the network.
    Without them, deep networks are nearly impossible to train.

    ⚡ GRADIENT CHECKPOINTING:
    When enabled, recomputes activations during backward pass instead of
    storing them. Trades ~30% compute for ~50% memory savings - essential
    for training large models on limited hardware.
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
        self.use_moe = config.use_moe

        # Choose normalization type based on config
        Norm = RMSNorm if config.use_rms_norm else nn.LayerNorm

        # Two normalizations: one before attention, one before FFN
        self.attention_norm = Norm(config.dim)
        self.ffn_norm = Norm(config.dim)

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

        # T3-4: Mixture of Depths — per-token routing for FFN
        self.use_mod = config.use_mixture_of_depths
        self._mod_capacity = config.mod_capacity_factor
        if self.use_mod:
            self.depth_router = nn.Linear(config.dim, 1, bias=False)
            self._mod_aux_loss = torch.tensor(0.0)

        # T5-4: Token Merging (ToMe)
        self._tome_ratio = config.tome_ratio

    def _forward_impl(
        self,
        x: torch.Tensor,
        freqs_cis: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        start_pos: int = 0,
    ) -> torch.Tensor:
        """Internal forward implementation (used by checkpointing)."""
        B, T, D = x.shape

        # T5-4: Token Merging — reduce seq length before attention
        tome_active = self._tome_ratio > 0.0 and T > 4 and not use_cache and self.training
        tome_merged_mask = None
        tome_merge_dst = None
        original_T = T

        if tome_active:
            r = max(1, int(T * self._tome_ratio))
            normed = F.normalize(x, dim=-1)
            tome_merge_dst, _, tome_merged_mask = _bipartite_soft_matching(normed, r)
            x = _tome_merge(x, tome_merged_mask)
            T = x.shape[1]
            # Rebuild mask for shorter sequence
            if mask is not None and mask.shape[-1] >= original_T:
                # Build a new causal mask for the merged sequence length.
                # SDPA/Flash handle is_causal=True internally, but the
                # standard attention path needs an explicit mask.
                causal = torch.full((T, T), float("-inf"), device=x.device, dtype=x.dtype)
                causal = torch.triu(causal, diagonal=1)
                mask = causal.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

        # Attention sub-layer with residual connection
        attn_out = self.attention(self.attention_norm(x), freqs_cis, mask, use_cache, start_pos)
        if self.use_layer_scale:
            attn_out = attn_out * self.ls_attn
        h = x + self.drop_path_attn(attn_out)

        # T5-4: Unmerge tokens back to original length after attention
        if tome_active and tome_merged_mask is not None:
            h = _tome_unmerge(h, original_T, tome_merged_mask, tome_merge_dst)
            T = original_T

        # T3-4: Mixture of Depths — route tokens for FFN
        if self.use_mod and T > 1 and not use_cache:
            scores = self.depth_router(h).squeeze(-1)  # [B, T]
            k = max(1, int(T * self._mod_capacity))
            topk_vals, topk_idx = torch.topk(scores, k, dim=1)
            weights = torch.sigmoid(topk_vals).unsqueeze(-1)  # [B, k, 1]

            # Gather → FFN → weight → scatter back
            idx_expand = topk_idx.unsqueeze(-1).expand(-1, -1, D)
            selected = torch.gather(h, 1, idx_expand)
            ffn_out = self.feed_forward(self.ffn_norm(selected))
            if self.use_layer_scale:
                ffn_out = ffn_out * self.ls_ffn
            ffn_out = self.drop_path_ffn(ffn_out) * weights
            result = h.clone()
            result.scatter_add_(1, idx_expand, ffn_out)

            # Load-balancing auxiliary loss
            probs = torch.sigmoid(scores)
            self._mod_aux_loss = (probs.mean(dim=1) - self._mod_capacity).pow(2).mean()
            return result

        # Standard FFN sub-layer with residual connection
        ffn_out = self.feed_forward(self.ffn_norm(h))
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

    def get_moe_aux_loss(self) -> torch.Tensor:
        """Get MoE auxiliary loss for load balancing during training."""
        if self.use_moe and hasattr(self.feed_forward, "get_aux_loss"):
            return self.feed_forward.get_aux_loss()
        return torch.tensor(0.0)

    def get_mod_aux_loss(self) -> torch.Tensor:
        """Get Mixture-of-Depths auxiliary loss (T3-4)."""
        if self.use_mod:
            return self._mod_aux_loss
        return torch.tensor(0.0)
