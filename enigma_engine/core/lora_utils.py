"""
LoRA/QLoRA Training Utilities for Enigma AI Engine

Provides:
- LoraConfig: Configuration for LoRA adapters
- QLoraConfig: Configuration for 4-bit quantized LoRA
- OffloadConfig: Configuration for CPU offloading
- LoraTrainer: Training class for LoRA/QLoRA fine-tuning
- Memory management utilities for VRAM and RAM

LoRA = Low-Rank Adaptation - trains small adapter weights instead of full model
QLoRA = Quantized LoRA - uses 4-bit quantization to reduce VRAM by ~75%
CPU Offload = Stores optimizer states and some weights in system RAM

Usage:
    from enigma_engine.core.lora_utils import LoraConfig, LoraTrainer

    config = LoraConfig(rank=8, alpha=16)
    trainer = LoraTrainer(model, tokenizer, lora_config=config)
    trainer.train(data)
    trainer.save_adapter("my_lora.pth")
"""

import gc
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Optional imports for PEFT and quantization
PEFT_AVAILABLE = False
BITSANDBYTES_AVAILABLE = False
ACCELERATE_AVAILABLE = False

try:
    from peft import (
        LoraConfig as PeftLoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        TaskType,
    )

    PEFT_AVAILABLE = True
except ImportError:
    logger.debug("PEFT not installed - install with: pip install peft")

try:
    import bitsandbytes as bnb  # noqa: F401
    from transformers import BitsAndBytesConfig

    BITSANDBYTES_AVAILABLE = True
except ImportError:
    logger.debug("bitsandbytes not installed - install with: pip install bitsandbytes")

try:
    from accelerate import Accelerator, dispatch_model, infer_auto_device_map  # noqa: F401

    ACCELERATE_AVAILABLE = True
except ImportError:
    logger.debug("accelerate not installed - install with: pip install accelerate")


# =============================================================================
# MEMORY MANAGEMENT
# =============================================================================


def clear_vram() -> None:
    """
    Clear GPU VRAM by emptying PyTorch cache.

    Call this when VRAM is full to free cached memory.
    Does NOT unload models - just releases cached tensors.

    Example:
        clear_vram()  # Free up VRAM
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info("VRAM cache cleared")

    # Also run Python garbage collection
    gc.collect()


def get_memory_info() -> Dict[str, float]:
    """
    Get current memory usage info for both RAM and VRAM.

    Returns:
        Dict with memory stats in GB:
        - ram_total_gb: Total system RAM
        - ram_available_gb: Available RAM
        - ram_used_gb: Used RAM
        - vram_total_gb: Total GPU VRAM (0 if no GPU)
        - vram_used_gb: Used GPU VRAM
        - vram_available_gb: Available VRAM

    Example:
        info = get_memory_info()
        print(f"RAM: {info['ram_available_gb']:.1f}GB free")
        print(f"VRAM: {info['vram_available_gb']:.1f}GB free")
    """
    try:
        import psutil

        ram = psutil.virtual_memory()
        ram_total = ram.total / (1024**3)
        ram_available = ram.available / (1024**3)
        ram_used = ram.used / (1024**3)
    except ImportError:
        logger.debug("psutil not installed -- RAM stats unavailable")
        ram_total = 0.0
        ram_available = 0.0
        ram_used = 0.0

    info = {
        "ram_total_gb": ram_total,
        "ram_available_gb": ram_available,
        "ram_used_gb": ram_used,
        "vram_total_gb": 0.0,
        "vram_used_gb": 0.0,
        "vram_available_gb": 0.0,
    }

    # GPU VRAM (if available)
    if torch.cuda.is_available():
        try:
            # Get VRAM info for default device
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)

            info["vram_total_gb"] = props.total_memory / (1024**3)
            info["vram_used_gb"] = torch.cuda.memory_allocated(device) / (1024**3)
            info["vram_available_gb"] = info["vram_total_gb"] - info["vram_used_gb"]
        except Exception as e:
            logger.debug(f"Could not get VRAM info: {e}")

    return info


def estimate_training_memory(
    model_params: int,
    batch_size: int = 1,
    seq_length: int = 512,
    use_lora: bool = True,
    lora_rank: int = 8,
    use_qlora: bool = False,
    gradient_checkpointing: bool = False,
    hidden_size: int = 768,
) -> Dict[str, float]:
    """
    Estimate memory requirements for training.

    Args:
        model_params: Number of parameters in model
        batch_size: Training batch size
        seq_length: Sequence length
        use_lora: Whether using LoRA (much less memory)
        lora_rank: LoRA rank (higher = more memory)
        use_qlora: Whether using 4-bit quantization
        gradient_checkpointing: Whether using gradient checkpointing

    Returns:
        Dict with estimated memory in GB:
        - model_memory_gb: Memory for model weights
        - optimizer_memory_gb: Memory for optimizer states
        - gradient_memory_gb: Memory for gradients
        - activation_memory_gb: Memory for activations
        - total_vram_gb: Total VRAM needed
        - recommended_ram_gb: Recommended system RAM for offloading

    Example:
        mem = estimate_training_memory(7_000_000_000, use_qlora=True)
        print(f"Need ~{mem['total_vram_gb']:.1f}GB VRAM")
    """
    bytes_per_param = 4  # fp32

    if use_qlora:
        bytes_per_param = 0.5  # 4-bit = 0.5 bytes
    elif use_lora:
        bytes_per_param = 2  # fp16 for base model

    # Model memory
    model_memory = model_params * bytes_per_param

    # LoRA adds small number of trainable params
    if use_lora:
        # Rough estimate: LoRA adds ~0.1-1% trainable params
        lora_params = int(model_params * 0.01 * (lora_rank / 8))
        model_memory += lora_params * 4  # LoRA weights in fp32

    # Optimizer states (AdamW needs 2x params for momentum + variance)
    if use_lora:
        optimizer_memory = lora_params * 4 * 2  # Only LoRA params
    else:
        optimizer_memory = model_params * 4 * 2  # Full model

    # Gradients
    if use_lora:
        gradient_memory = lora_params * 4
    else:
        gradient_memory = model_params * 4

    # Activations (rough estimate based on batch size and seq length)
    activation_memory = batch_size * seq_length * hidden_size * 4
    if gradient_checkpointing:
        activation_memory *= 0.3  # Checkpointing reduces by ~70%

    # Convert to GB
    total_vram = (model_memory + gradient_memory + activation_memory) / (1024**3)
    optimizer_gb = optimizer_memory / (1024**3)

    return {
        "model_memory_gb": model_memory / (1024**3),
        "optimizer_memory_gb": optimizer_gb,
        "gradient_memory_gb": gradient_memory / (1024**3),
        "activation_memory_gb": activation_memory / (1024**3),
        "total_vram_gb": total_vram,
        "recommended_ram_gb": optimizer_gb + 4,  # Optimizer + buffer
    }


# =============================================================================
# LORA CONFIGURATION
# =============================================================================


@dataclass
class LoraConfig:
    """
    Configuration for LoRA (Low-Rank Adaptation) training.

    LoRA adds small trainable matrices to attention layers instead of
    fine-tuning the entire model. This drastically reduces memory usage
    and training time.

    Attributes:
        rank: Rank of the low-rank matrices (4-64, default 8)
              Lower = faster + less memory, Higher = more capacity
        alpha: Scaling factor (usually 2x rank)
               Controls how much LoRA affects output
        dropout: Dropout rate for LoRA layers (0.0-0.3)
        target_modules: Which modules to add LoRA to
                       Default targets attention projections
        bias: How to handle bias ("none", "all", "lora_only")
        task_type: Type of task ("CAUSAL_LM", "SEQ_CLS", etc.)

    Example:
        config = LoraConfig(rank=16, alpha=32)  # Higher capacity
        config = LoraConfig(rank=4, alpha=8)    # Faster, less memory
    """

    rank: int = 8
    alpha: int = 16
    dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    # LoRA+ (Hayou et al.): use a higher LR for B matrices than A matrices.
    # lambda=1.0 means same LR for both (standard LoRA). lambda=16 is the
    # paper's recommended ratio for small-to-mid models.
    lora_plus_lambda: float = 1.0

    # DoRA (R17): Weight-Decomposed Low-Rank Adaptation.
    # Decomposes the weight update into magnitude + direction components.
    # LoRA is applied only to the direction — better matches full fine-tuning
    # behavior at the same rank.  Requires PEFT >= 0.8.0 when using PEFT path.
    use_dora: bool = False

    def to_peft_config(self):
        """Convert to PEFT LoraConfig."""
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT not installed. Run: pip install peft")

        kwargs: dict[str, Any] = dict(
            r=self.rank,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            target_modules=self.target_modules,
            bias=self.bias,
            task_type=(TaskType.CAUSAL_LM if self.task_type == "CAUSAL_LM" else self.task_type),
        )
        if self.use_dora:
            kwargs["use_dora"] = True
        return PeftLoraConfig(**kwargs)


@dataclass
class QLoraConfig(LoraConfig):
    """
    Configuration for QLoRA (Quantized LoRA) training.

    QLoRA uses 4-bit quantization for the base model, reducing VRAM
    usage by ~75% compared to full precision. Only LoRA weights are
    trained in full precision.

    Attributes:
        All from LoraConfig, plus:
        load_in_4bit: Enable 4-bit quantization (default True)
        bnb_4bit_quant_type: Quantization type ("nf4" or "fp4")
        bnb_4bit_compute_dtype: Compute dtype (bfloat16 or float16)
        bnb_4bit_use_double_quant: Use double quantization (saves more memory)

    Example:
        config = QLoraConfig(rank=16)  # 4-bit base + 16-rank LoRA
    """

    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True

    def to_bnb_config(self):
        """Convert to BitsAndBytesConfig for quantization."""
        if not BITSANDBYTES_AVAILABLE:
            raise ImportError("bitsandbytes not installed. Run: pip install bitsandbytes")

        compute_dtype = torch.bfloat16 if self.bnb_4bit_compute_dtype == "bfloat16" else torch.float16

        return BitsAndBytesConfig(
            load_in_4bit=self.load_in_4bit,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
        )


@dataclass
class OffloadConfig:
    """
    Configuration for CPU offloading during training.

    When VRAM is limited, offloading stores optimizer states and
    some model weights in system RAM. This is slower but allows
    training larger models than would fit in VRAM alone.

    Attributes:
        cpu_offload: Offload some layers to CPU (default True)
        offload_optimizer: Offload optimizer states to CPU (default True)
        gradient_checkpointing: Trade compute for memory (default True)
        pin_memory: Pin CPU memory for faster GPU transfers
        max_memory_gpu: Max GB to use on GPU (None = auto)
        max_memory_cpu: Max GB to use on CPU (None = auto)

    Example:
        # For 8GB VRAM GPU with 32GB RAM:
        config = OffloadConfig(max_memory_gpu=7, max_memory_cpu=24)
    """

    cpu_offload: bool = True
    offload_optimizer: bool = True
    gradient_checkpointing: bool = True
    pin_memory: bool = True
    max_memory_gpu: Optional[float] = None
    max_memory_cpu: Optional[float] = None


# =============================================================================
# DoRA LINEAR LAYER  (R17 — standalone, no PEFT required)
# =============================================================================


class DoRALinear(nn.Module):
    """Weight-Decomposed Low-Rank Adaptation linear layer (R17).

    Decomposes the pretrained weight ``W`` into a magnitude scalar ``m``
    and a direction matrix ``V / ||V||``.  A standard LoRA update
    ``B @ A`` is applied only to the direction component::

        V' = W + B @ A
        output = x @ (m * V' / ||V'||_col)^T

    This matches the learning dynamics of full fine-tuning more
    closely than standard LoRA at the same rank.

    Args:
        base_linear: The frozen ``nn.Linear`` to wrap.
        rank: LoRA rank.
        alpha: LoRA scaling factor.
        dropout: Dropout probability on the LoRA path.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        in_features = base_linear.in_features
        out_features = base_linear.out_features

        # Freeze the original weight (keep a reference)
        self.weight = base_linear.weight  # not a param — frozen
        self.weight.requires_grad_(False)
        self.bias = base_linear.bias
        if self.bias is not None:
            self.bias.requires_grad_(False)

        # Magnitude vector m — one scalar per output neuron (column norm)
        with torch.no_grad():
            col_norms = self.weight.norm(p=2, dim=1)  # (out_features,)
        self.m = nn.Parameter(col_norms)

        # LoRA A and B
        self.lora_a = nn.Parameter(torch.zeros(rank, in_features))
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.zeros_(self.lora_b)

        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Direction = W + scaling * B @ A
        lora_update = (self.lora_b @ self.lora_a) * self.scaling
        direction = self.weight + lora_update

        # Column-wise L2 norm of direction
        col_norms = direction.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
        normed = direction / col_norms

        # Apply magnitude
        weight_dora = self.m.unsqueeze(1) * normed

        out = F.linear(self.dropout(x), weight_dora, self.bias)
        return out


def apply_dora(
    model: nn.Module,
    target_modules: list[str] | None = None,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
) -> nn.Module:
    """Replace matching ``nn.Linear`` layers with :class:`DoRALinear`.

    This is a standalone DoRA injection that does NOT require PEFT.
    Only layers whose name contains one of *target_modules* are replaced.

    Args:
        model: The model to modify in-place.
        target_modules: Substrings to match layer names against.
            Defaults to ``["q_proj", "v_proj", "k_proj", "o_proj"]``.
        rank: LoRA rank.
        alpha: LoRA scaling.
        dropout: LoRA dropout.

    Returns:
        The same model (modified in-place).
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(t in name for t in target_modules):
            continue

        # Walk to parent
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)

        dora_layer = DoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, parts[-1], dora_layer)
        replaced += 1

    logger.info("DoRA: replaced %d linear layers (rank=%d)", replaced, rank)
    return model


# =============================================================================
# LORA MODEL CREATION
# =============================================================================


def create_lora_model(model: nn.Module, config: Optional[LoraConfig] = None) -> nn.Module:
    """
    Wrap a model with LoRA adapters.

    Args:
        model: Base PyTorch model
        config: LoRA configuration (uses defaults if None)

    Returns:
        Model with LoRA adapters added

    Example:
        model = load_model("my_model.pth")
        lora_model = create_lora_model(model, LoraConfig(rank=16))
        # Now train lora_model - only LoRA weights update
    """
    if not PEFT_AVAILABLE:
        raise ImportError("PEFT not installed. Run: pip install peft")

    config = config or LoraConfig()
    peft_config = config.to_peft_config()

    logger.info(f"Creating LoRA model with rank={config.rank}, alpha={config.alpha}")

    # Wrap model with PEFT
    lora_model = get_peft_model(model, peft_config)

    # Log trainable params
    trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in lora_model.parameters())
    pct = 100 * trainable / total if total > 0 else 0.0
    logger.info(f"LoRA trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")

    return lora_model


def create_qlora_model(model: nn.Module, config: Optional[QLoraConfig] = None) -> nn.Module:
    """
    Wrap a model with QLoRA (4-bit quantized LoRA).

    QLoRA quantizes the base model to 4-bit while keeping LoRA
    weights in full precision. This reduces VRAM by ~75%.

    Args:
        model: Base PyTorch model
        config: QLoRA configuration (uses defaults if None)

    Returns:
        Quantized model with LoRA adapters

    Example:
        model = load_model("my_model.pth")
        qlora_model = create_qlora_model(model, QLoraConfig(rank=16))
        # 4-bit base + full precision LoRA
    """
    if not PEFT_AVAILABLE:
        raise ImportError("PEFT not installed. Run: pip install peft")
    if not BITSANDBYTES_AVAILABLE:
        raise ImportError("bitsandbytes not installed. Run: pip install bitsandbytes")

    config = config or QLoraConfig()

    logger.info(f"Creating QLoRA model (4-bit) with rank={config.rank}")

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Add LoRA adapters
    peft_config = config.to_peft_config()
    qlora_model = get_peft_model(model, peft_config)

    # Log memory savings
    trainable = sum(p.numel() for p in qlora_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in qlora_model.parameters())
    pct = 100 * trainable / total if total > 0 else 0.0
    logger.info(f"QLoRA trainable params: {trainable:,} / {total:,} ({pct:.2f}%)")

    return qlora_model


# =============================================================================
# LORA WEIGHT MANAGEMENT
# =============================================================================


def load_lora_weights(path: Union[str, Path]) -> Dict[str, torch.Tensor]:
    """
    Load LoRA adapter weights from file.

    Args:
        path: Path to LoRA weights file (.pth or .safetensors)

    Returns:
        Dictionary of LoRA weight tensors

    Example:
        weights = load_lora_weights("my_adapter.pth")
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"LoRA weights not found: {path}")

    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file

            weights = load_file(str(path))
        except ImportError:
            raise ImportError("safetensors required for .safetensors files") from None
    else:
        weights = torch.load(path, map_location="cpu", weights_only=True)

    logger.info(f"Loaded LoRA weights from: {path} ({len(weights)} tensors)")
    return weights


def apply_lora(
    model: nn.Module, lora_weights: Dict[str, torch.Tensor], adapter_name: str = "default", merge: bool = False
) -> None:
    """
    Apply LoRA weights to a model.

    Args:
        model: Target model
        lora_weights: Dictionary of LoRA weight tensors
        adapter_name: Name for the adapter
        merge: If True, merge into base weights permanently

    Raises:
        ValueError: If no key in ``lora_weights`` matches the model's
            state_dict (same silent-no-op case that
            :func:`merge_lora_weights` guards against).

    Example:
        weights = load_lora_weights("coding_adapter.pth")
        apply_lora(model, weights, "coding")
    """
    if merge:
        merge_lora_weights(model, lora_weights)
        return

    # Load weights into model's LoRA layers
    state_dict = model.state_dict()
    matched = 0
    skipped: list[str] = []
    for key, value in lora_weights.items():
        if key in state_dict:
            state_dict[key] = value
            matched += 1
        else:
            skipped.append(key)

    if matched == 0:
        # Silent-corruption guard, mirroring merge_lora_weights: every
        # key was filtered out, so the adapter would have been
        # registered with zero effect on the model. Fail loud instead
        # of logging "Applied" and returning.
        raise ValueError(
            "apply_lora: no keys in lora_weights matched the model's "
            "state_dict. The adapter would have had no effect. "
            "Pre-process the LoRA state dict to use this model's "
            f"weight names. First 3 unmatched keys: {skipped[:3]}"
        )

    if skipped:
        logger.warning(
            "apply_lora: %d/%d keys not in model state_dict (loaded %d). First skipped: %s",
            len(skipped),
            len(lora_weights),
            matched,
            skipped[:3],
        )

    # Register the adapter only after the match check passes
    if not hasattr(model, "_lora_adapters"):
        model._lora_adapters = {}
    model._lora_adapters[adapter_name] = lora_weights

    model.load_state_dict(state_dict, strict=False)
    logger.info("Applied LoRA adapter '%s' (%d tensors)", adapter_name, matched)


def merge_lora_weights(model: nn.Module, lora_weights: Dict[str, torch.Tensor]) -> None:
    """
    Permanently merge LoRA weights into base model.

    After merging, LoRA weights become part of base weights
    and the adapter can be deleted to save memory.

    The non-PEFT (manual) path expects ``lora_weights`` to be a dict
    of *already-deltas* (i.e. ``B @ A * scaling``) keyed by base
    weight names that exist in ``model.state_dict()``. If you have
    raw LoRA A/B matrices, multiply them yourself before calling.

    Args:
        model: Target model with LoRA adapters
        lora_weights: LoRA delta-weight dictionary keyed by base
            weight name. Each value must have the same shape as the
            target base weight.

    Raises:
        ValueError: If the manual path is taken (model has no
            ``merge_and_unload``) and either (a) no key in
            ``lora_weights`` matches the model's state_dict, or
            (b) a matched value's shape disagrees with the base
            weight. Closes the silent-corruption case where the
            function appears to succeed but no merge happened
            because every key was filtered out.

    Example:
        merge_lora_weights(model, weights)  # Now permanent
    """
    # If model is a PEFT model, use built-in merge_and_unload
    if hasattr(model, "merge_and_unload"):
        model.merge_and_unload()
        logger.info("Merged LoRA using PEFT merge_and_unload")
        return

    # Manual merge for non-PEFT models
    state_dict = model.state_dict()

    matched = 0
    skipped: list[str] = []
    for key, lora_weight in lora_weights.items():
        if key in state_dict:
            base = state_dict[key]
            if tuple(base.shape) != tuple(lora_weight.shape):
                raise ValueError(
                    f"merge_lora_weights: shape mismatch for '{key}'. "
                    f"base shape {tuple(base.shape)} vs delta shape "
                    f"{tuple(lora_weight.shape)}. The manual path "
                    "expects pre-multiplied LoRA deltas (B @ A * "
                    "scaling), not raw A/B matrices."
                )
            state_dict[key] = base + lora_weight
            matched += 1
        else:
            skipped.append(key)

    if matched == 0:
        # Silent-corruption guard: every key was filtered out. The
        # most common cause is passing a raw PEFT adapter state_dict
        # (keys like ``base_model.model.layers.0.attention.wq.lora_A
        # .default.weight``) to a non-PEFT model expecting base
        # weight names. Fail loud instead of logging "Merged" and
        # returning with no effect.
        raise ValueError(
            "merge_lora_weights: no keys in lora_weights matched the "
            "model's state_dict. The function would have been a "
            "silent no-op. Either pass a PEFT-wrapped model "
            "(use merge_and_unload) or pre-process the LoRA state "
            f"dict to use base-weight names. "
            f"First 3 unmatched keys: {skipped[:3]}"
        )

    if skipped:
        logger.warning(
            "merge_lora_weights: %d/%d keys not in base state_dict (merged %d). First skipped: %s",
            len(skipped),
            len(lora_weights),
            matched,
            skipped[:3],
        )

    model.load_state_dict(state_dict)
    logger.info("Merged %d LoRA delta weight(s) into base model", matched)


# =============================================================================
# LORA TRAINER
# =============================================================================


class LoraTrainer:
    """
    Trainer for LoRA/QLoRA fine-tuning.

    Features:
    - Automatic VRAM management
    - CPU offloading when needed
    - Gradient checkpointing
    - Progress callbacks for GUI
    - Checkpoint saving

    Example:
        trainer = LoraTrainer(
            model=my_model,
            tokenizer=my_tokenizer,
            lora_config=LoraConfig(rank=16),
            offload_config=OffloadConfig(cpu_offload=True)
        )
        trainer.train(training_data)
        trainer.save_adapter("my_lora.pth")
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        lora_config: Optional[LoraConfig] = None,
        offload_config: Optional[OffloadConfig] = None,
        output_dir: str = "models/lora_adapters",
        learning_rate: float = 1e-4,
        batch_size: int = 4,
        epochs: int = 3,
        gradient_accumulation_steps: int = 4,
        weight_decay: float = 0.01,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.95,
        adam_eps: float = 1e-8,
        min_lr_ratio: float = 0.1,
    ):
        """
        Initialize LoRA trainer.

        Args:
            model: Model to fine-tune
            tokenizer: Tokenizer for encoding text
            lora_config: LoRA configuration (default: rank=8)
            offload_config: CPU offloading settings
            output_dir: Where to save adapters
            learning_rate: Training learning rate
            batch_size: Training batch size
            epochs: Number of epochs
            gradient_accumulation_steps: Accumulate gradients
            weight_decay: AdamW weight decay
            adam_beta1: AdamW beta1
            adam_beta2: AdamW beta2 (0.95 = LM-friendly default)
            adam_eps: AdamW epsilon
            min_lr_ratio: Cosine schedule floor as fraction of peak LR
                (default 0.1 = decay to 10% of peak; Pass 156z9au).
        """
        self.tokenizer = tokenizer
        self.lora_config = lora_config or LoraConfig()
        self.offload_config = offload_config or OffloadConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        if gradient_accumulation_steps < 1:
            raise ValueError(f"gradient_accumulation_steps must be >= 1, got {gradient_accumulation_steps}")
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.weight_decay = weight_decay
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_eps = adam_eps
        if not 0.0 <= min_lr_ratio <= 1.0:
            raise ValueError(f"min_lr_ratio must be in [0.0, 1.0], got {min_lr_ratio}")
        self.min_lr_ratio = min_lr_ratio

        # Apply LoRA to model
        if isinstance(self.lora_config, QLoraConfig):
            self.model = create_qlora_model(model, self.lora_config)
        else:
            self.model = create_lora_model(model, self.lora_config)

        # Enable gradient checkpointing if configured
        if self.offload_config.gradient_checkpointing:
            if hasattr(self.model, "gradient_checkpointing_enable"):
                self.model.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")

        # Callbacks
        self.on_progress: Optional[Callable[[int, str], None]] = None
        self.on_loss: Optional[Callable[[float], None]] = None

        # Training state
        self._stop_requested = False
        self._lock = threading.Lock()

        # Clear VRAM before training
        clear_vram()

        logger.info(f"LoraTrainer initialized: rank={self.lora_config.rank}, offload={self.offload_config.cpu_offload}")

    def request_stop(self) -> None:
        """Request graceful stop of training."""
        with self._lock:
            self._stop_requested = True

    def _should_stop(self) -> bool:
        """Check if stop was requested."""
        with self._lock:
            return self._stop_requested

    def _emit_progress(self, percent: int, message: str) -> None:
        """Emit progress callback."""
        if self.on_progress:
            try:
                self.on_progress(percent, message)
            except Exception:
                pass

    def train(self, data: Union[str, List[Dict[str, str]]], max_length: int = 512) -> Dict[str, Any]:
        """
        Train LoRA adapter on data.

        Args:
            data: Training data - text or list of {"prompt": "...", "completion": "..."}
            max_length: Maximum sequence length

        Returns:
            Training results dictionary

        Example:
            results = trainer.train([
                {"prompt": "Hello", "completion": "Hi there!"},
                {"prompt": "How are you?", "completion": "I'm doing great!"}
            ])
        """
        self._stop_requested = False
        self._emit_progress(0, "Preparing training data...")

        # Parse data
        if isinstance(data, str):
            data = [{"prompt": "", "completion": data}]

        # Clear VRAM before starting
        clear_vram()

        # Check memory
        mem_info = get_memory_info()
        logger.info(
            f"Starting training - VRAM: {mem_info['vram_available_gb']:.1f}GB, "
            f"RAM: {mem_info['ram_available_gb']:.1f}GB"
        )

        # Setup device (with CPU offload if needed)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.offload_config.cpu_offload and ACCELERATE_AVAILABLE:
            # Use accelerate for smart device placement. cpu= must reflect
            # whether we truly have no GPU -- NOT offload_optimizer. Mapping
            # offload_optimizer onto cpu=True forces the ENTIRE run (model +
            # data) onto the CPU, which then crashes because the batch loop
            # below moves inputs to 'cuda' (device mismatch), and is not what
            # "offload optimizer states" means anyway.
            if self.offload_config.offload_optimizer and device == "cuda":
                # Plain AdamW + a bare Accelerator cannot offload optimizer
                # states to CPU (that needs a DeepSpeed/FSDP plugin). Fail
                # honestly: the flag is a no-op on this path rather than a
                # silent CPU-only fallback.
                logger.warning(
                    "offload_optimizer is set but optimizer-state offload is not "
                    "implemented on this path (plain AdamW); training on GPU without it."
                )
            accelerator = Accelerator(
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                cpu=(device == "cpu"),
            )
            self.model, optimizer = accelerator.prepare(
                self.model,
                self._get_optimizer(),
            )
            # accelerate owns device placement now; follow the device it chose
            # so the batch tensors below land where the model actually is.
            device = str(accelerator.device)
        else:
            self.model = self.model.to(device)
            optimizer = self._get_optimizer()

        self.model.train()

        # Learning rate scheduler (cosine decay)
        from torch.optim.lr_scheduler import CosineAnnealingLR

        n_batches = len(self._create_batches(data, max_length))
        if n_batches == 0:
            logger.warning("No valid training batches created -- all items may be too short (< 2 tokens) or empty.")
            self._emit_progress(100, "No valid batches -- skipped.")
            return {
                "total_loss": 0.0,
                "epochs": 0,
                "steps": 0,
                "final_loss": 0.0,
            }
        steps_per_epoch = max(1, n_batches // self.gradient_accumulation_steps)
        total_steps = steps_per_epoch * self.epochs
        scheduler = CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps), eta_min=self.learning_rate * self.min_lr_ratio
        )

        # Training loop
        total_loss = 0.0
        epoch_loss = 0.0
        step = 0
        pad_id = getattr(self.tokenizer, "pad_token_id", 0) or 0

        # Get the forward-callable model (bypass PEFT outer wrapper)
        fwd_model = self.model
        if hasattr(self.model, "get_base_model"):
            fwd_model = self.model.get_base_model()

        self._emit_progress(5, f"Training for {self.epochs} epochs...")

        for epoch in range(self.epochs):
            if self._should_stop():
                break

            epoch_loss = 0.0
            oom_retries = 0
            optimizer.zero_grad()

            for batch_idx, (input_ids, attention_mask) in enumerate(self._create_batches(data, max_length)):
                if self._should_stop():
                    break

                # Move batch to device
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                # Next-token prediction: shift input/target
                inputs = input_ids[:, :-1]
                targets = input_ids[:, 1:]
                attn = attention_mask[:, :-1]

                # Forward pass
                try:
                    outputs = fwd_model(inputs, attention_mask=attn)
                    logits = outputs[0] if isinstance(outputs, tuple) else outputs
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=pad_id
                    )
                    oom_retries = 0  # Reset on success
                except Exception as e:
                    # Handle OOM
                    if "out of memory" in str(e).lower():
                        oom_retries += 1
                        if oom_retries > 2:
                            logger.error("OOM persists after %d retries -- aborting LoRA training", oom_retries)
                            raise
                        logger.warning("OOM detected - clearing VRAM and retrying (%d/2)", oom_retries)
                        clear_vram()
                        optimizer.zero_grad()
                        continue
                    raise

                # Backward pass with gradient accumulation
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                    # Clear cache periodically
                    if batch_idx % 10 == 0:
                        clear_vram()

                batch_loss = loss.item() * self.gradient_accumulation_steps
                epoch_loss += batch_loss
                step += 1

                # Progress update
                progress = int(5 + 90 * (epoch * len(data) + batch_idx) / (self.epochs * len(data)))
                self._emit_progress(progress, f"Epoch {epoch + 1}/{self.epochs}, Loss: {batch_loss:.4f}")

                if self.on_loss:
                    self.on_loss(batch_loss)

            # Flush remaining accumulated gradients
            if n_batches % self.gradient_accumulation_steps != 0:
                optimizer.step()
                optimizer.zero_grad()
                if scheduler:
                    scheduler.step()

            total_loss += epoch_loss
            logger.info(f"Epoch {epoch + 1}/{self.epochs} - Loss: {epoch_loss:.4f}")

        # Final cleanup
        clear_vram()

        self._emit_progress(100, "Training complete!")

        return {
            "total_loss": total_loss,
            "epochs": self.epochs,
            "steps": step,
            "final_loss": epoch_loss,
        }

    def _get_optimizer(self):
        """Create optimizer for training.

        Supports LoRA+ differential learning rates: B matrices get
        ``learning_rate * lora_plus_lambda``, A matrices get base LR.
        When ``lora_plus_lambda == 1.0`` this degenerates to standard
        single-group AdamW.
        """
        from torch.optim import AdamW

        lam = getattr(self.lora_config, "lora_plus_lambda", 1.0)

        if lam != 1.0:
            # Split into A-matrix and B-matrix groups
            group_a: list[torch.Tensor] = []
            group_b: list[torch.Tensor] = []
            group_other: list[torch.Tensor] = []

            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                lower = name.lower()
                if "lora_a" in lower:
                    group_a.append(param)
                elif "lora_b" in lower:
                    group_b.append(param)
                else:
                    group_other.append(param)

            param_groups = []
            if group_a:
                param_groups.append({"params": group_a, "lr": self.learning_rate})
            if group_b:
                param_groups.append({"params": group_b, "lr": self.learning_rate * lam})
            if group_other:
                param_groups.append({"params": group_other, "lr": self.learning_rate})

            if not param_groups:
                raise ValueError("No trainable LoRA parameters found. Check that LoRA layers are properly attached.")

            return AdamW(
                param_groups,
                weight_decay=self.weight_decay,
                betas=(self.adam_beta1, self.adam_beta2),
                eps=self.adam_eps,
            )

        # Standard single-group path
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        return AdamW(
            trainable_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(self.adam_beta1, self.adam_beta2),
            eps=self.adam_eps,
        )

    def _create_batches(self, data: List[Dict[str, str]], max_length: int) -> List[tuple]:
        """Create padded batches from training data.

        Returns:
            List of (batch_tensor, attention_mask) tuples.
            batch_tensor shape: (batch_size, max_seq_in_batch)
            attention_mask shape: (batch_size, max_seq_in_batch)
        """
        encoded = []
        for item in data:
            text = item.get("prompt", "") + item.get("completion", "")
            if not text:
                continue
            tokens = self.tokenizer.encode(text)
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            if len(tokens) < 2:
                continue
            encoded.append(tokens)

        if not encoded:
            return []

        # Sort by length for efficient padding
        encoded.sort(key=len, reverse=True)

        pad_id = getattr(self.tokenizer, "pad_token_id", 0) or 0
        batches = []
        for i in range(0, len(encoded), self.batch_size):
            batch_tokens = encoded[i : i + self.batch_size]
            max_len = max(len(t) for t in batch_tokens)
            padded = []
            masks = []
            for tokens in batch_tokens:
                pad_len = max_len - len(tokens)
                padded.append(tokens + [pad_id] * pad_len)
                masks.append([1] * len(tokens) + [0] * pad_len)
            batches.append(
                (
                    torch.tensor(padded, dtype=torch.long),
                    torch.tensor(masks, dtype=torch.long),
                )
            )

        return batches

    def _create_dataloader(self, data, max_length):
        """Create a DataLoader for accelerate."""
        from torch.utils.data import DataLoader, Dataset

        class SimpleDataset(Dataset):
            def __init__(self, data, tokenizer, max_length):
                self.data = data
                self.tokenizer = tokenizer
                self.max_length = max_length

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                item = self.data[idx]
                text = item.get("prompt", "") + item.get("completion", "")
                tokens = self.tokenizer.encode(text)[: self.max_length]
                return torch.tensor(tokens)

        dataset = SimpleDataset(data, self.tokenizer, max_length)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def save_adapter(self, path: Optional[Union[str, Path]] = None) -> Path:
        """
        Save LoRA adapter as a PEFT directory.

        Pass 156s (LoRA-1b): always emits the canonical PEFT directory
        format (``adapter_config.json`` + ``adapter_model.safetensors``)
        so adapter rank/alpha/target_modules travel with the weights
        and inference can apply the adapter without sidecar metadata.
        The previous manual-fallback ``.pth`` save path was unreachable
        in normal flow (``LoraTrainer.__init__`` wraps via
        ``create_lora_model`` / ``create_qlora_model`` which always
        produce a PEFT model) and produced metadata-less files that
        the chat engine could not safely apply. Removed.

        Args:
            path: Output path. The PEFT directory is written next to
                this path using its stem (e.g. ``my_lora.pth`` →
                ``my_lora/``). The ``.pth`` extension is accepted for
                back-compat with older callers but the actual artefact
                is always a directory.

        Returns:
            Path to the saved PEFT directory.

        Example:
            trainer.save_adapter("my_coding_lora.pth")
            # writes models/checkpoints/my_coding_lora/
        """
        if path is None:
            path = self.output_dir / "adapter"
        else:
            path = Path(path)

        # Strip any extension — PEFT writes a directory, not a file.
        save_dir = path.parent / path.stem
        save_dir.parent.mkdir(parents=True, exist_ok=True)

        if not hasattr(self.model, "save_pretrained"):
            # Defensive: should never happen because __init__ always
            # wraps with create_lora_model / create_qlora_model.
            raise RuntimeError(
                "LoraTrainer.model is not a PEFT model -- cannot save "
                "adapter. This is a bug; LoraTrainer should always "
                "wrap with create_lora_model() in __init__."
            )

        self.model.save_pretrained(save_dir)
        logger.info(f"Saved PEFT adapter to: {save_dir}")
        return save_dir


# =============================================================================
# LORA ADAPTER MANAGER (FP-D) — per-task LoRA adapters
# =============================================================================


class LoRAAdapterManager:
    """Manage multiple LoRA adapters per task/skill.

    Each task (e.g. "coding", "math", "writing") gets its own LoRA
    weights stored in a subdirectory.  You can create, switch, list,
    and merge adapters without losing specializations.

    This prevents catastrophic forgetting: the base model stays frozen
    and each skill lives in a separate adapter file.

    Args:
        base_dir: Root directory for adapter storage.
            Defaults to ``models/lora_adapters/``.

    Example::

        mgr = LoRAAdapterManager()
        mgr.create("coding", model, LoraConfig(rank=16))
        mgr.save("coding", model)
        mgr.switch("math", model)   # loads math adapter weights
    """

    def __init__(self, base_dir: Union[str, Path, None] = None):
        self.base_dir = Path(base_dir or "models/lora_adapters")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._active_task: Optional[str] = None

    # -- directory helpers --

    def _task_dir(self, task: str) -> Path:
        result = (self.base_dir / task).resolve()
        result.relative_to(self.base_dir.resolve())
        return result

    def _weights_path(self, task: str) -> Path:
        return self._task_dir(task) / "adapter.pth"

    def _meta_path(self, task: str) -> Path:
        return self._task_dir(task) / "meta.json"

    # -- public API --

    def list_tasks(self) -> List[str]:
        """Return sorted list of available task adapters."""
        tasks = []
        if self.base_dir.exists():
            for child in sorted(self.base_dir.iterdir()):
                if child.is_dir() and (child / "adapter.pth").exists():
                    tasks.append(child.name)
        return tasks

    @property
    def active_task(self) -> Optional[str]:
        return self._active_task

    def create(
        self,
        task: str,
        model: nn.Module,
        config: Optional[LoraConfig] = None,
    ) -> Path:
        """Create a new LoRA adapter for a task and save initial weights.

        If the task already exists, this is a no-op (returns existing path).
        """
        wpath = self._weights_path(task)
        if wpath.exists():
            logger.info("Adapter '%s' already exists at %s", task, wpath)
            return wpath

        self._task_dir(task).mkdir(parents=True, exist_ok=True)
        config = config or LoraConfig()

        # Save metadata
        meta = {
            "task": task,
            "rank": config.rank,
            "alpha": config.alpha,
            "target_modules": config.target_modules,
        }
        from enigma_engine.core.safe_save import atomic_write_json

        atomic_write_json(self._meta_path(task), meta)

        # Save initial adapter (trainable params only)
        lora_weights: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                lora_weights[name] = param.data.cpu()

        from enigma_engine.core.safe_save import atomic_torch_save

        atomic_torch_save(lora_weights, wpath)

        logger.info("Created LoRA adapter '%s' (%d tensors)", task, len(lora_weights))
        return wpath

    def save(
        self,
        task: str,
        model: nn.Module,
    ) -> Path:
        """Save current trainable weights as an adapter for *task*."""
        self._task_dir(task).mkdir(parents=True, exist_ok=True)
        wpath = self._weights_path(task)

        lora_weights: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                lora_weights[name] = param.data.cpu()

        from enigma_engine.core.safe_save import atomic_torch_save

        atomic_torch_save(lora_weights, wpath)

        self._active_task = task
        logger.info("Saved adapter '%s' (%d tensors)", task, len(lora_weights))
        return wpath

    def switch(
        self,
        task: str,
        model: nn.Module,
        save_current: bool = True,
    ) -> None:
        """Switch the model to a different task adapter.

        If *save_current* is True, saves the active adapter first.
        Then loads the requested task's adapter weights.
        """
        if save_current and self._active_task is not None:
            self.save(self._active_task, model)

        wpath = self._weights_path(task)
        if not wpath.exists():
            raise FileNotFoundError(f"No adapter for task '{task}'. Available: {self.list_tasks()}")

        weights = load_lora_weights(wpath)
        apply_lora(model, weights, adapter_name=task)
        self._active_task = task
        logger.info("Switched to adapter '%s'", task)

    def delete(self, task: str) -> None:
        """Delete a task adapter from disk."""
        tdir = self._task_dir(task)
        if tdir.exists():
            import shutil

            shutil.rmtree(tdir)
            if self._active_task == task:
                self._active_task = None
            logger.info("Deleted adapter '%s'", task)

    def merge_into_base(
        self,
        task: str,
        model: nn.Module,
    ) -> None:
        """Permanently merge a task adapter into the base model.

        After merging, the adapter's specialization becomes part of
        the base weights and the adapter file can be deleted.
        """
        wpath = self._weights_path(task)
        if not wpath.exists():
            raise FileNotFoundError(f"No adapter for task '{task}'")

        weights = load_lora_weights(wpath)
        merge_lora_weights(model, weights)
        logger.info("Merged adapter '%s' into base model", task)
