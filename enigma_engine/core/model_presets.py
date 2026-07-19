"""
Model presets and configuration dataclasses for Enigma Engine.

Contains ForgeConfig, QuantizationConfig, MODEL_PRESETS, and related
utility functions. This module has no torch dependency at import time.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


# =============================================================================
# ⚙️ CONFIGURATION - Model Settings
# =============================================================================
# ForgeConfig holds ALL the settings that define a model's architecture.
# Think of it as a blueprint - same settings = same model structure.


@dataclass
class ForgeConfig:
    """
    Model configuration with sensible defaults.

    📖 WHAT EACH SETTING DOES:

    CORE ARCHITECTURE:
    ┌────────────────────────────────────────────────────────────────────────┐
    │ vocab_size  │ How many unique tokens the model knows (like vocabulary)│
    │ dim         │ Hidden dimension - the "width" of neural pathways       │
    │ n_layers    │ Number of transformer blocks (depth of the network)     │
    │ n_heads     │ Attention heads (parallel attention computations)       │
    │ n_kv_heads  │ Key/Value heads for GQA (memory optimization)          │
    │ hidden_dim  │ FFN hidden size (typically 4x dim for expansion)       │
    └────────────────────────────────────────────────────────────────────────┘

    LIMITS & REGULARIZATION:
    ┌────────────────────────────────────────────────────────────────────────┐
    │ max_seq_len │ Maximum tokens in one sequence (context window)         │
    │ dropout     │ Randomly zero neurons during training (prevents overfit)│
    └────────────────────────────────────────────────────────────────────────┘

    ARCHITECTURE FLAGS (modern transformer tricks):
    ┌────────────────────────────────────────────────────────────────────────┐
    │ use_rope    │ Rotary Position Embeddings (better position awareness)  │
    │ use_rms_norm│ RMSNorm instead of LayerNorm (faster, equally good)    │
    │ use_swiglu  │ SwiGLU activation (better than ReLU/GELU)              │
    │ use_bias    │ Add bias terms (usually False in modern models)        │
    └────────────────────────────────────────────────────────────────────────┘

    UNIVERSAL MODEL ENHANCEMENTS:
    ┌────────────────────────────────────────────────────────────────────────┐
    │ rope_scaling_type   │ RoPE scaling for extended context              │
    │ rope_scaling_factor │ Scaling multiplier for context extension       │
    └────────────────────────────────────────────────────────────────────────┘
    """

    # ─────────────────────────────────────────────────────────────────────────
    # CORE PARAMETERS
    # ─────────────────────────────────────────────────────────────────────────
    vocab_size: int = 32000  # Size of vocabulary (tokenizer determines this)
    dim: int = 512  # Model hidden dimension (larger = smarter but slower)
    n_layers: int = 8  # Number of transformer layers (deeper = more capable)
    n_heads: int = 8  # Attention heads (more = better pattern recognition)
    n_kv_heads: Optional[int] = None  # KV heads for GQA (None = same as n_heads)
    hidden_dim: Optional[int] = None  # FFN dimension (None = auto-calculate)
    max_seq_len: int = 1024  # Maximum sequence length (context window)
    dropout: float = 0.1  # Dropout rate (0.1 = 10% neurons randomly zeroed)

    # ─────────────────────────────────────────────────────────────────────────
    # ARCHITECTURE FLAGS - Modern transformer improvements
    # ─────────────────────────────────────────────────────────────────────────
    use_rope: bool = True  # RoPE: Better position encoding than absolute
    use_rms_norm: bool = True  # RMSNorm: Faster normalization, works just as well
    use_swiglu: bool = True  # SwiGLU: Superior activation function
    use_bias: bool = False  # Bias: Usually disabled in modern transformers
    rope_theta: float = 10000.0  # RoPE base frequency (higher = longer context)

    # ─────────────────────────────────────────────────────────────────────────
    # ROPE SCALING - Extended context support
    # ─────────────────────────────────────────────────────────────────────────
    rope_scaling_type: Optional[str] = None  # "linear", "dynamic", "yarn", None
    rope_scaling_factor: float = 1.0  # Context extension multiplier (>1.0 extends)

    # ─────────────────────────────────────────────────────────────────────────
    # MEMORY OPTIMIZATION
    # ─────────────────────────────────────────────────────────────────────────
    use_gradient_checkpointing: bool = True  # Trade compute for memory during training

    # ─────────────────────────────────────────────────────────────────────────
    # MULTI-MODAL SUPPORT
    # ─────────────────────────────────────────────────────────────────────────
    vision_hidden_size: Optional[int] = None  # Vision encoder dimension
    audio_hidden_size: Optional[int] = None  # Audio encoder dimension

    # ─────────────────────────────────────────────────────────────────────────
    # TRAINING STABILITY & REGULARIZATION
    # ─────────────────────────────────────────────────────────────────────────
    use_qk_norm: bool = True  # Normalize Q and K in attention (stabilizes, prevents head collapse)
    use_layer_scale: bool = False  # Learnable residual scaling (stabilizes deep training)
    drop_path_rate: float = 0.0  # Stochastic depth (0.0 = disabled, 0.1-0.3 typical)

    # Track if config is frozen (immutable after creation)
    _frozen: bool = False

    def __post_init__(self) -> None:
        """
        Post-initialization: validate and set computed defaults.
        Called automatically after __init__ (dataclass magic).
        """
        # ─────────────────────────────────────────────────────────────────────
        # AUTO-CALCULATE KV HEADS: Default to same as n_heads (no GQA)
        # ─────────────────────────────────────────────────────────────────────
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads

        # ─────────────────────────────────────────────────────────────────────
        # AUTO-CALCULATE HIDDEN DIM: The "expansion" in feed-forward layers
        # ─────────────────────────────────────────────────────────────────────
        # Standard: hidden_dim = 4 * dim (4x expansion)
        # SwiGLU: Needs 2/3 of that because it has 3 matrices instead of 2
        # We also round up to nearest 64 for GPU efficiency
        if self.hidden_dim is None:
            if self.use_swiglu:
                # SwiGLU formula: 2/3 * (4 * dim), rounded to multiple of 64
                self.hidden_dim = int(2 * (4 * self.dim) / 3)
                self.hidden_dim = 64 * ((self.hidden_dim + 63) // 64)
            else:
                self.hidden_dim = 4 * self.dim

        # ─────────────────────────────────────────────────────────────────────
        # VALIDATION: Catch configuration errors early
        # ─────────────────────────────────────────────────────────────────────
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")

        if self.dim <= 0:
            raise ValueError(f"dim must be positive, got {self.dim}")

        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")

        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {self.n_heads}")

        if not (0 <= self.dropout <= 1):
            raise ValueError(f"dropout must be between 0 and 1, got {self.dropout}")

        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}")

        # dim must be divisible by n_heads (each head gets dim/n_heads dimensions)
        if self.dim % self.n_heads != 0:
            # Calculate helpful suggestions
            head_dim = self.dim // self.n_heads
            suggested_dim = self.n_heads * (head_dim + 1)
            suggested_heads = self.dim // (head_dim + 1) if head_dim + 1 > 0 else self.n_heads
            raise ValueError(
                f"n_heads ({self.n_heads}) must divide evenly into dim ({self.dim}). "
                f"Got remainder: {self.dim % self.n_heads}. "
                f"Try: dim={suggested_dim} (with {self.n_heads} heads) or "
                f"n_heads={suggested_heads} (with dim={self.dim})"
            )

        # n_heads must be divisible by n_kv_heads (for GQA grouping)
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_kv_heads ({self.n_kv_heads}) must divide evenly into n_heads ({self.n_heads}). "
                f"Got remainder: {self.n_heads % self.n_kv_heads}"
            )

        # ─────────────────────────────────────────────────────────────────────
        # VALIDATE NEW FEATURES
        # ─────────────────────────────────────────────────────────────────────
        # RoPE scaling validation
        if self.rope_scaling_type is not None:
            valid_scaling = {"linear", "dynamic", "yarn"}
            if self.rope_scaling_type not in valid_scaling:
                raise ValueError(f"rope_scaling_type must be one of {valid_scaling}, got {self.rope_scaling_type}")
            if self.rope_scaling_factor <= 0:
                raise ValueError(f"rope_scaling_factor must be positive, got {self.rope_scaling_factor}")

    def validate(self) -> bool:
        """
        Run read-only validation on the config.

        This does not auto-calculate fields or assign attributes,
        so it is safe to call on frozen configs.

        Returns:
            True if valid

        Raises:
            ValueError: If any validation fails
        """
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.dim <= 0:
            raise ValueError(f"dim must be positive, got {self.dim}")
        if self.n_layers <= 0:
            raise ValueError(f"n_layers must be positive, got {self.n_layers}")
        if self.n_heads <= 0:
            raise ValueError(f"n_heads must be positive, got {self.n_heads}")
        if not (0 <= self.dropout <= 1):
            raise ValueError(f"dropout must be between 0 and 1, got {self.dropout}")
        if self.max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {self.max_seq_len}")
        if self.dim % self.n_heads != 0:
            raise ValueError(f"n_heads ({self.n_heads}) must divide evenly into dim ({self.dim})")
        n_kv = self.n_kv_heads if self.n_kv_heads is not None else self.n_heads
        if self.n_heads % n_kv != 0:
            raise ValueError(f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({n_kv})")
        if self.use_rope and self.rope_scaling_type is not None:
            valid = {"linear", "dynamic", "yarn"}
            if self.rope_scaling_type not in valid:
                raise ValueError(f"rope_scaling_type must be one of {valid}, got {self.rope_scaling_type}")
            if self.rope_scaling_factor <= 0:
                raise ValueError(f"rope_scaling_factor must be positive, got {self.rope_scaling_factor}")
        return True

    def freeze(self) -> ForgeConfig:
        """
        Freeze the config to prevent further modifications.

        Once frozen, any attempt to modify attributes will raise an error.
        This is useful for ensuring config immutability after model creation.

        Returns:
            self (for chaining)
        """
        object.__setattr__(self, "_frozen", True)
        return self

    def __setattr__(self, name: str, value: Any) -> None:
        """Override setattr to enforce frozen state."""
        if getattr(self, "_frozen", False) and name != "_frozen":
            raise AttributeError(
                "Cannot modify frozen ForgeConfig. Create a new config with the desired changes instead."
            )
        object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "vocab_size": self.vocab_size,
            "dim": self.dim,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "hidden_dim": self.hidden_dim,
            "max_seq_len": self.max_seq_len,
            "dropout": self.dropout,
            "use_rope": self.use_rope,
            "use_rms_norm": self.use_rms_norm,
            "use_swiglu": self.use_swiglu,
            "use_bias": self.use_bias,
            "rope_theta": self.rope_theta,
            # New parameters
            "rope_scaling_type": self.rope_scaling_type,
            "rope_scaling_factor": self.rope_scaling_factor,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "vision_hidden_size": self.vision_hidden_size,
            "audio_hidden_size": self.audio_hidden_size,
            # Training stability & regularization
            "use_qk_norm": self.use_qk_norm,
            "use_layer_scale": self.use_layer_scale,
            "drop_path_rate": self.drop_path_rate,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ForgeConfig:
        known = {
            "vocab_size",
            "dim",
            "n_layers",
            "n_heads",
            "n_kv_heads",
            "hidden_dim",
            "max_seq_len",
            "dropout",
            "use_rope",
            "use_rms_norm",
            "use_swiglu",
            "use_bias",
            "rope_theta",
            # New parameters
            "rope_scaling_type",
            "rope_scaling_factor",
            "use_gradient_checkpointing",
            "vision_hidden_size",
            "audio_hidden_size",
            "use_qk_norm",
            "use_layer_scale",
            "drop_path_rate",
        }
        return cls(**{k: v for k, v in d.items() if k in known})


# =============================================================================
# ⚡ QUANTIZATION CONFIG - Memory-Efficient Model Deployment
# =============================================================================


@dataclass
class QuantizationConfig:
    """
    Configuration for model quantization.

    📖 WHAT THIS DOES:
    Quantization reduces model precision to save memory and speed up inference.

    📐 QUANTIZATION TYPES:
    ┌────────────┬────────────────────────────────────────────────────────────┐
    │ Type       │ Description                                                │
    ├────────────┼────────────────────────────────────────────────────────────┤
    │ none       │ Full FP32 precision (default, most accurate)              │
    │ dynamic    │ Dynamic INT8 quantization (good for CPU)                  │
    │ int8       │ Static INT8 quantization (requires calibration)           │
    │ int4       │ 4-bit quantization (smallest, some quality loss)          │
    └────────────┴────────────────────────────────────────────────────────────┘

    ⚡ MEMORY SAVINGS:
    - FP32: 4 bytes/param (baseline)
    - FP16: 2 bytes/param (50% savings)
    - INT8: 1 byte/param (75% savings)
    - INT4: 0.5 bytes/param (87.5% savings)

    🍓 PI RECOMMENDATIONS:
    - Pi Zero: int4 quantization (fits in 512MB RAM)
    - Pi 4 (4GB): int8 quantization
    - Pi 5 (8GB): dynamic quantization
    """

    mode: str = "none"  # "none", "dynamic", "int8", "int4"

    # Static quantization options
    calibration_data: Optional[list[torch.Tensor]] = None
    num_calibration_batches: int = 100

    # Dynamic quantization options
    dtype: Optional[torch.dtype] = None  # torch.qint8 for dynamic

    # Which layers to quantize
    quantize_linear: bool = True
    quantize_embedding: bool = False  # Usually keep embeddings in FP32

    # INT4 specific
    group_size: int = 128  # For grouped quantization

    def __post_init__(self) -> None:
        valid_modes = {"none", "dynamic", "int8", "int4"}
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid quantization mode: {self.mode}. Valid: {valid_modes}")


# =============================================================================
# 📊 MODEL PRESETS - From Raspberry Pi to Server Farm!
# =============================================================================
# These presets make it easy to create models of different sizes.
# Just pick a preset name and the config is ready to go!
#
# HOW TO CHOOSE A PRESET:
#   1. What hardware do you have? (RAM, GPU VRAM)
#   2. What quality do you need?
#   3. How fast does it need to be?
#
# ROUGH GUIDELINES:
#   • 4GB RAM/VRAM → tiny or mini
#   • 8GB VRAM → small or medium
#   • 16GB VRAM → base
#   • 32GB VRAM → large (measured: 28 GB at batch=1)
#   • 24GB+ VRAM (3090/4090/5090) → xl (~12 GB with grad ckpt)
#   • Multi-GPU → xxl and above

MODEL_PRESETS = {
    # ─────────────────────────────────────────────────────────────────────────
    # RASPBERRY PI OPTIMIZED (~500K-8M params) - Specifically tuned for Pi
    # ─────────────────────────────────────────────────────────────────────────
    # Pi Zero 2W (512MB RAM): Ultra-minimal footprint
    "pi_zero": ForgeConfig(dim=64, n_layers=2, n_heads=2, n_kv_heads=1, max_seq_len=256, dropout=0.0),
    # Pi 4 (4GB RAM): Good balance of capability and memory
    "pi_4": ForgeConfig(dim=192, n_layers=4, n_heads=4, n_kv_heads=2, max_seq_len=512, dropout=0.05),
    # Pi 5 (8GB RAM): Maximum Pi capability
    "pi_5": ForgeConfig(dim=256, n_layers=6, n_heads=8, n_kv_heads=4, max_seq_len=1024, dropout=0.05),
    # ─────────────────────────────────────────────────────────────────────────
    # EMBEDDED / IoT (~1-2M params) - For microcontrollers and tiny devices
    # ─────────────────────────────────────────────────────────────────────────
    "nano": ForgeConfig(dim=128, n_layers=4, n_heads=4, n_kv_heads=2, max_seq_len=256),
    "micro": ForgeConfig(dim=192, n_layers=4, n_heads=4, n_kv_heads=2, max_seq_len=384),
    # ─────────────────────────────────────────────────────────────────────────
    # EDGE / Raspberry Pi (~5-15M params) - For single-board computers
    # ─────────────────────────────────────────────────────────────────────────
    "tiny": ForgeConfig(dim=256, n_layers=6, n_heads=8, n_kv_heads=4, max_seq_len=512),
    "mini": ForgeConfig(dim=384, n_layers=6, n_heads=6, n_kv_heads=3, max_seq_len=512),
    # ─────────────────────────────────────────────────────────────────────────
    # CONSUMER GPU (~27-85M params) - RTX 2080 to RTX 3070
    # ─────────────────────────────────────────────────────────────────────────
    "small": ForgeConfig(dim=512, n_layers=8, n_heads=8, n_kv_heads=4, max_seq_len=1024),
    "medium": ForgeConfig(dim=768, n_layers=12, n_heads=12, n_kv_heads=4, max_seq_len=2048, rope_theta=500000.0),
    "base": ForgeConfig(dim=896, n_layers=14, n_heads=14, n_kv_heads=2, max_seq_len=2048, rope_theta=500000.0),
    # ─────────────────────────────────────────────────────────────────────────
    # PROSUMER GPU (~200M-600M params) - RTX 4090 (large) or A100 (xl)
    # ─────────────────────────────────────────────────────────────────────────
    "large": ForgeConfig(dim=1024, n_layers=16, n_heads=16, n_kv_heads=4, max_seq_len=4096, rope_theta=500000.0),
    "xl": ForgeConfig(
        dim=1536, n_layers=24, n_heads=24, n_kv_heads=6, max_seq_len=4096, rope_theta=500000.0, dropout=0.05
    ),
    # ─────────────────────────────────────────────────────────────────────────
    # MULTI-GPU / SERVER (~1B-3B params) - 2-4x A100, workstation setups
    # ─────────────────────────────────────────────────────────────────────────
    "xxl": ForgeConfig(
        dim=2048, n_layers=32, n_heads=32, n_kv_heads=8, max_seq_len=8192, rope_theta=500000.0, dropout=0.05
    ),
    "huge": ForgeConfig(
        dim=2560, n_layers=40, n_heads=40, n_kv_heads=8, max_seq_len=8192, rope_theta=500000.0, dropout=0.05
    ),
    # ─────────────────────────────────────────────────────────────────────────
    # DATACENTER / CLOUD (~7B-13B params) - 8x A100, cloud instances
    # ─────────────────────────────────────────────────────────────────────────
    "giant": ForgeConfig(
        dim=4096, n_layers=32, n_heads=32, n_kv_heads=8, max_seq_len=8192, rope_theta=500000.0, dropout=0.05
    ),
    "colossal": ForgeConfig(
        dim=4096, n_layers=48, n_heads=32, n_kv_heads=8, max_seq_len=16384, rope_theta=500000.0, dropout=0.05
    ),
    # ─────────────────────────────────────────────────────────────────────────
    # MAXIMUM SCALE (~30B+ params) - Full datacenter, research frontier
    # ─────────────────────────────────────────────────────────────────────────
    "titan": ForgeConfig(
        dim=6144, n_layers=48, n_heads=48, n_kv_heads=12, max_seq_len=16384, rope_theta=500000.0, dropout=0.05
    ),
    "omega": ForgeConfig(
        dim=8192, n_layers=64, n_heads=64, n_kv_heads=16, max_seq_len=32768, rope_theta=500000.0, dropout=0.05
    ),
}

# Human-readable descriptions
MODEL_DESCRIPTIONS = {
    # Pi-optimized presets
    "pi_zero": "Raspberry Pi Zero 2W - needs <1 GB RAM",
    "pi_4": "Raspberry Pi 4 - needs <1 GB RAM",
    "pi_5": "Raspberry Pi 5 - needs <1 GB RAM",
    # Standard presets (VRAM = training with gradient checkpointing, batch=1)
    "nano": "Microcontrollers - needs <1 GB",
    "micro": "IoT devices - needs <1 GB",
    "tiny": "Edge devices - needs <1 GB",
    "mini": "Mobile, low-power - needs <1 GB",
    "small": "Entry GPU - needs ~1 GB VRAM",
    "medium": "Mid-range GPU - needs ~2.5 GB VRAM",
    "base": "Mid-range GPU - needs ~3 GB VRAM",
    "large": "Good GPU - needs ~6 GB VRAM",
    "xl": "RTX 4090 / 16 GB+ GPU - needs ~12 GB VRAM",
    "xxl": "RTX 4090 / 32 GB+ GPU - needs ~34 GB VRAM",
    "huge": "Multi-GPU / 48 GB+ - needs ~50 GB VRAM",
    "giant": "Multi-GPU / A100 - needs ~77 GB VRAM",
    "colossal": "Multi-node - needs ~153 GB VRAM",
    "titan": "Datacenter - needs ~280 GB VRAM",
    "omega": "Large cluster - needs ~904 GB VRAM",
}


def estimate_training_vram(
    config: ForgeConfig,
    gradient_checkpointing: bool = True,
) -> float:
    """
    Estimate VRAM needed for training a model config, in GB.

    Accounts for weights (bf16), optimizer states (bf16 AdamW),
    gradients (bf16), attention score matrices (quadratic in seq_len),
    linear activations saved for backward, and output logits.

    Args:
        config: Model configuration.
        gradient_checkpointing: If True (default), assumes gradient
            checkpointing is enabled — only ~2 layers of activations
            are stored at a time instead of all layers.  This matches
            real training behavior (the OOM handler auto-enables it).
    """
    params = estimate_parameters(config)
    # bf16 training: weights(2) + grads(2) + Adam momentum(2) + variance(2)
    # = 8 bytes/param (optimizer states are bf16 when model is bf16)
    fixed_bytes = params * 8

    # Per-layer activation memory for backward pass (batch=1 minimum)
    # Linear term: autograd saves inputs to all linear layers, norms,
    # activation functions, and element-wise ops. Coefficient calibrated
    # against measured data: ~60 * dim bytes per (token * layer) in bf16.
    linear_per_layer = config.max_seq_len * config.dim * 60 * 2

    # Quadratic term: standard attention materializes full seq*seq score
    # matrix per head (no Flash Attention). This dominates for seq>1024.
    attn_per_layer = config.n_heads * config.max_seq_len**2 * 2

    act_per_layer = linear_per_layer + attn_per_layer

    if gradient_checkpointing:
        # GC recomputes activations during backward — only ~1 layer
        # held at a time.  Use 2 for safety margin.
        act_bytes = act_per_layer * min(2, config.n_layers)
    else:
        act_bytes = act_per_layer * config.n_layers

    # Output logits tensor (float32 for cross-entropy loss computation)
    logits_bytes = config.max_seq_len * config.vocab_size * 4

    # 1.5x empirical correction: PyTorch autograd stores additional
    # intermediate values beyond minimal saved tensors (residual copies,
    # dropout masks, norm statistics, etc.)
    act_total = (act_bytes + logits_bytes) * 1.5

    # 10% safety margin for CUDA allocator overhead and fragmentation
    total_bytes = (fixed_bytes + act_total) * 1.1
    vram_gb = total_bytes / (1024**3)
    return round(max(0.5, vram_gb), 1)


def recommend_preset_for_vram(vram_gb: float) -> str:
    """
    Return the largest preset that fits within the given VRAM budget.

    Args:
        vram_gb: Available GPU VRAM in gigabytes.

    Returns:
        Preset name (e.g. 'large', 'xl').
    """
    best_name = "pi_zero"
    best_vram = 0.0

    for name, config in MODEL_PRESETS.items():
        config_copy = copy.deepcopy(config)
        config_copy.vocab_size = 32000  # Standard for estimation
        needed = estimate_training_vram(config_copy)
        if needed <= vram_gb and needed >= best_vram:
            best_vram = needed
            best_name = name
    return best_name


def recommend_preset_for_tokens(
    token_count: int,
    vocab_size: int = 32000,
    vram_gb: float | None = None,
) -> tuple[str, int]:
    """Return the largest preset whose Chinchilla-optimal token count
    is at or below *token_count*.

    Chinchilla scaling (Hoffmann et al., 2022): optimal training uses
    ~20 tokens per parameter.  So a 27M-param model needs ~540M tokens;
    an 85M model needs ~1.7B tokens.

    When *vram_gb* is given, results are further capped to presets
    that fit the GPU for training.

    Args:
        token_count: Estimated number of tokens in the training data.
        vocab_size: Vocabulary size for parameter estimation.
        vram_gb: Available GPU VRAM in GB (optional).

    Returns:
        (preset_name, estimated_params) of the recommended preset.
    """
    best_name = "pi_zero"
    best_params = 0

    for name, cfg in MODEL_PRESETS.items():
        cfg_copy = copy.deepcopy(cfg)
        cfg_copy.vocab_size = vocab_size
        params = estimate_parameters(cfg_copy)
        # Chinchilla: need ~20 tokens per param
        needed_tokens = params * 20
        if needed_tokens > token_count:
            continue
        # Optionally filter by VRAM
        if vram_gb is not None:
            needed_vram = estimate_training_vram(cfg_copy)
            if needed_vram > vram_gb:
                continue
        if params > best_params:
            best_params = params
            best_name = name

    if best_params == 0:
        # Even the smallest preset is too large — still return it
        cfg_copy = copy.deepcopy(MODEL_PRESETS["pi_zero"])
        cfg_copy.vocab_size = vocab_size
        best_params = estimate_parameters(cfg_copy)

    return best_name, best_params


def get_preset(name: str, vocab_size: int = 32000) -> ForgeConfig:
    """Get a preset configuration."""
    if name not in MODEL_PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(MODEL_PRESETS.keys())}")

    # Create a copy with vocab_size
    preset = MODEL_PRESETS[name]
    return ForgeConfig(
        vocab_size=vocab_size,
        dim=preset.dim,
        n_layers=preset.n_layers,
        n_heads=preset.n_heads,
        n_kv_heads=preset.n_kv_heads,
        max_seq_len=preset.max_seq_len,
        dropout=preset.dropout,
        rope_theta=preset.rope_theta,
    )


# =============================================================================
# Parameter target parsing — lets users type "8b", "500m", etc.
# =============================================================================

import re as _re


def parse_param_target(text: str) -> Optional[int]:
    """
    Parse a human-friendly parameter count string.

    Accepted formats:
        "8b"   → 8,000,000,000
        "1.5b" → 1,500,000,000
        "500m" → 500,000,000
        "27M"  → 27,000,000
        "8000000000" → 8,000,000,000 (raw number)

    Returns None if the input cannot be parsed.
    """
    if not text or not isinstance(text, str):
        return None

    text = text.strip().lower()
    if not text:
        return None

    # Match number + optional suffix (b or m)
    match = _re.match(r"^(\d+(?:\.\d+)?)\s*(b|m)?$", text)
    if not match:
        return None

    number = float(match.group(1))
    suffix = match.group(2)

    if suffix == "b":
        return int(number * 1_000_000_000)
    elif suffix == "m":
        return int(number * 1_000_000)
    else:
        # Raw number — must be a reasonable integer
        if number < 1:
            return None
        return int(number)


def config_for_param_target(target: int, vocab_size: int = 32000) -> tuple:
    """
    Build a ForgeConfig that matches a target parameter count.

    First checks presets for a close match (within 20%).  If no preset
    is close enough, computes a custom config by scaling dim from the
    nearest preset to hit the target.  This means there is **no upper
    limit** — any parameter count can be requested.

    Args:
        target: Target number of parameters (e.g. 8_000_000_000).
        vocab_size: Vocabulary size for estimation (default 32000).

    Returns:
        (name, ForgeConfig) — preset name if matched, or 'custom_<target>'
        for computed configs.
    """
    import math

    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")

    best_name = "small"
    best_distance = float("inf")
    best_est = 0

    for name, config in MODEL_PRESETS.items():
        config_copy = copy.deepcopy(config)
        config_copy.vocab_size = vocab_size
        est = estimate_parameters(config_copy)
        distance = abs(est - target)
        if distance < best_distance:
            best_distance = distance
            best_name = name
            best_est = est

    # If closest preset is within 20% of target, use it directly.
    # Deepcopy the whole preset: rebuilding from a handful of fields
    # silently resets everything else (rope_theta, feature toggles) to
    # the dataclass defaults.
    if best_est > 0 and best_distance / best_est < 0.2:
        result = copy.deepcopy(MODEL_PRESETS[best_name])
        result.vocab_size = vocab_size
        return best_name, result

    # No close preset — compute a custom config to match the target
    # Use the closest preset as a template for layer/head ratios
    ref = MODEL_PRESETS[best_name]
    n_layers = ref.n_layers
    head_ratio = ref.n_heads / ref.n_kv_heads if ref.n_kv_heads else 1

    # Solve for dim from: target ≈ vocab*dim + n_layers*(4*dim² + 8*dim²) + dim
    # Simplified: target ≈ 12 * n_layers * dim² + vocab * dim
    # Quadratic: a*dim² + b*dim - target = 0
    a = 12 * n_layers
    b = vocab_size
    discriminant = b * b + 4 * a * target
    if discriminant < 0:
        # Negative discriminant means target is too small;
        # fall back to minimum dimension.
        dim = 64
    else:
        dim = int((-b + math.sqrt(discriminant)) / (2 * a))

    # Ensure dim is at least 64
    dim = max(64, dim)

    # Pick n_heads so dim is divisible by n_heads
    # Start from the ref ratio and find the best fit
    n_heads = ref.n_heads
    # Scale n_heads proportionally with dim
    if dim > ref.dim:
        n_heads = max(ref.n_heads, dim // (ref.dim // ref.n_heads))
    # Round dim UP to nearest multiple of 2*n_heads so that
    # head_dim (dim // n_heads) is always even — RoPE requires it.
    step = 2 * n_heads
    if dim % step != 0:
        dim = step * ((dim + step - 1) // step)

    # Compute n_kv_heads preserving the original ratio
    n_kv_heads = max(1, n_heads // int(head_ratio))
    # Ensure n_heads is divisible by n_kv_heads
    while n_heads % n_kv_heads != 0 and n_kv_heads > 1:
        n_kv_heads -= 1

    # Scale n_layers if dim alone can't reach the target
    # (e.g. for very large targets, increase depth too)
    test_cfg = ForgeConfig(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        max_seq_len=ref.max_seq_len,
    )
    est = estimate_parameters(test_cfg)
    if est < target * 0.8:
        # Scale layers proportionally to close the gap
        scale = target / max(1, est)
        n_layers = max(n_layers, int(n_layers * scale))

    # Scale max_seq_len with model size (larger models get more context)
    if target >= 30_000_000_000:
        max_seq_len = 32768
    elif target >= 7_000_000_000:
        max_seq_len = 16384
    elif target >= 1_000_000_000:
        max_seq_len = 8192
    else:
        max_seq_len = ref.max_seq_len

    # Format a human-readable name
    if target >= 1_000_000_000:
        label = f"{target / 1_000_000_000:.1f}b"
    else:
        label = f"{target / 1_000_000:.0f}m"

    result = ForgeConfig(
        vocab_size=vocab_size,
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        max_seq_len=max_seq_len,
        dropout=ref.dropout,
    )
    return f"custom_{label}", result


def estimate_parameters(config: ForgeConfig) -> int:
    """Estimate number of parameters for a config.

    Accounts for GQA (smaller K,V projections), per-layer norms,
    SwiGLU FFN, weight-tied output head, and MTP predict heads.
    """
    # Embedding (weight-tied with output, counted once)
    embed = config.vocab_size * config.dim

    # Per-layer attention with GQA
    head_dim = config.dim // config.n_heads
    kv_dim = config.n_kv_heads * head_dim
    # Wq + Wo: full dim×dim; Wk + Wv: dim×kv_dim (smaller with GQA)
    attn = 2 * config.dim * config.dim + 2 * config.dim * kv_dim

    # SwiGLU FFN: w1, w2, w3 (3 matrices)
    hidden = config.hidden_dim or int(2 * (4 * config.dim) / 3)
    ffn = 3 * config.dim * hidden

    # 2 RMSNorm per layer
    norms = 2 * config.dim

    per_layer = attn + ffn + norms

    # Final norm
    final_norm = config.dim

    return embed + (per_layer * config.n_layers) + final_norm


def list_presets() -> dict:
    """List all presets with descriptions and estimated parameters."""
    result = {}
    for name, config in MODEL_PRESETS.items():
        # IMPORTANT: Create a copy to avoid mutating the global preset!
        # Without copy, setting vocab_size corrupts the shared config object.
        config_copy = copy.deepcopy(config)
        config_copy.vocab_size = 32000  # Standard for estimation
        result[name] = {
            "description": MODEL_DESCRIPTIONS.get(name, ""),
            "estimated_params": estimate_parameters(config_copy),
            "dim": config_copy.dim,
            "layers": config_copy.n_layers,
            "heads": config_copy.n_heads,
            "max_seq_len": config_copy.max_seq_len,
        }
    return result
