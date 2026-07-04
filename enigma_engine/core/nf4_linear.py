"""T3-7: NF4 (NormalFloat 4-bit) quantization for QLoRA fine-tuning.

NF4 stores each weight as a 4-bit index into a 16-entry lookup table
derived from normal distribution quantiles.  Weights are grouped into
blocks of ``block_size``, each with its own absmax scale factor.

Usage::

    from enigma_engine.core.nf4_linear import NF4Linear, quantize_linear_nf4

    # Quantize a single layer
    nf4_layer = NF4Linear.from_linear(model.some_linear)

    # Quantize all Linear layers in a model (freeze base, add LoRA on top)
    quantize_linear_nf4(model)
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Pre-computed NF4 lookup table: 16 quantiles of N(0,1) mapped to [-1, 1].
# From Dettmers et al. 2023, "QLoRA: Efficient Finetuning of Quantized LLMs".
NF4_TABLE = torch.tensor(
    [
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ],
    dtype=torch.float32,
)

# Decision boundaries: midpoints between adjacent NF4 values.
NF4_BOUNDARIES = (NF4_TABLE[:-1] + NF4_TABLE[1:]) / 2


class NF4Linear(nn.Module):
    """Linear layer with NF4-quantized frozen weights.

    Base weights are stored in 4-bit NF4 format (2 values per byte).
    During forward, weights are dequantized to the input dtype on the
    fly.  Memory: ~4x smaller than fp16.

    Designed to be combined with LoRA adapters: the NF4 base is frozen,
    and trainable fp16 LoRA low-rank matrices are added on top.

    Args:
        in_features: Input dimension.
        out_features: Output dimension.
        bias: Whether the original layer had bias.
        block_size: Quantization block size for per-block scaling.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        block_size: int = 64,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size

        n_elements = in_features * out_features
        n_bytes = (n_elements + 1) // 2  # 2 indices per byte
        n_blocks = (n_elements + block_size - 1) // block_size

        self.register_buffer("_packed", torch.zeros(n_bytes, dtype=torch.uint8))
        self.register_buffer("_scales", torch.ones(n_blocks, dtype=torch.float32))
        self.register_buffer("_nf4_table", NF4_TABLE.clone())

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_linear(cls, linear: nn.Linear, block_size: int = 64) -> "NF4Linear":
        """Quantize an existing ``nn.Linear`` to NF4."""
        nf4 = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            block_size=block_size,
        )
        # Buffers are allocated on CPU in __init__; move them to the source
        # weight's device BEFORE quantizing so a CUDA-resident model doesn't
        # end up with CPU NF4 buffers that crash on the next forward.
        nf4.to(linear.weight.device)
        nf4._quantize(linear.weight.detach())
        if linear.bias is not None:
            nf4.bias = nn.Parameter(linear.bias.detach().clone())
        return nf4

    # ------------------------------------------------------------------
    # Quantize / Dequantize
    # ------------------------------------------------------------------

    def _quantize(self, weight: torch.Tensor) -> None:
        """Quantize *weight* ``[out, in]`` into packed NF4 + scales."""
        flat = weight.reshape(-1).float()
        n = flat.numel()
        bs = self.block_size

        # Per-block absmax scaling → normalize to [-1, 1]
        n_blocks = self._scales.shape[0]
        padded = F.pad(flat, (0, n_blocks * bs - n))
        blocks = padded.reshape(n_blocks, bs)
        scales = blocks.abs().amax(dim=1).clamp(min=1e-8)
        self._scales.copy_(scales)

        normalized = (blocks / scales.unsqueeze(1)).reshape(-1)[:n]

        # Map to nearest NF4 index via searchsorted on boundaries
        boundaries = NF4_BOUNDARIES.to(flat.device)
        indices = torch.searchsorted(boundaries, normalized).clamp(0, 15)

        # Pack pairs of 4-bit indices into uint8
        if n % 2 == 1:
            indices = F.pad(indices, (0, 1))
        packed = ((indices[0::2] << 4) | indices[1::2]).to(torch.uint8)
        self._packed.copy_(packed)

    def _dequantize(self) -> torch.Tensor:
        """Dequantize NF4 weights to float tensor ``[out, in]``."""
        n = self.in_features * self.out_features
        bs = self.block_size

        # Unpack uint8 → two 4-bit indices
        high = (self._packed >> 4).to(torch.long)
        low = (self._packed & 0x0F).to(torch.long)
        indices = torch.stack([high, low], dim=1).reshape(-1)[:n]

        # Lookup + apply block scales
        values = self._nf4_table[indices]
        n_blocks = self._scales.shape[0]
        padded = F.pad(values, (0, n_blocks * bs - n))
        weight = padded.reshape(n_blocks, bs) * self._scales.unsqueeze(1)
        return weight.reshape(-1)[:n].reshape(self.out_features, self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self._dequantize().to(x.dtype)
        return F.linear(x, weight, self.bias)

    def memory_savings_ratio(self) -> float:
        """Return approximate memory ratio vs fp16 (lower is better)."""
        fp16_bytes = self.in_features * self.out_features * 2
        nf4_bytes = self._packed.numel() + self._scales.numel() * 4
        return nf4_bytes / max(fp16_bytes, 1)


def quantize_linear_nf4(
    model: nn.Module,
    block_size: int = 64,
    skip_names: set[str] | None = None,
) -> int:
    """Replace all ``nn.Linear`` layers in *model* with ``NF4Linear``.

    Layers are frozen (``requires_grad=False``) after quantization.
    Returns the number of layers quantized.

    Args:
        model: Model to quantize in-place.
        block_size: NF4 block size.
        skip_names: Module names to leave unquantized (e.g. ``{'output'}``).
    """
    skip = skip_names or set()
    count = 0
    replacements: list[tuple[nn.Module, str, NF4Linear]] = []

    for name, module in model.named_modules():
        if name in skip:
            continue
        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            if full_name in skip:
                continue
            if isinstance(child, nn.Linear):
                nf4 = NF4Linear.from_linear(child, block_size=block_size)
                nf4.requires_grad_(False)
                replacements.append((module, child_name, nf4))
                count += 1

    for parent, child_name, nf4_layer in replacements:
        setattr(parent, child_name, nf4_layer)

    if count > 0:
        logger.info("Quantized %d Linear layers to NF4 (block_size=%d)", count, block_size)
    return count
