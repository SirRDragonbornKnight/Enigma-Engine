"""
Vision Encoder — ViT with from-scratch or pretrained (timm) backends.

Three modes:
    1. From scratch (default): trains on user data, no external downloads.
    2. Hybrid CNN+ViT: CNN stem for spatial features + transformer.
    3. Pretrained (via timm): ImageNet-trained ViT weights for massive quality
       jump. Requires ``pip install timm``.

Architecture (from scratch):
    Image → PatchEmbedding → + Position Embeddings → N TransformerBlocks → RMSNorm → features

Architecture (pretrained):
    Image → timm backbone → optional dim projection → features

Input:  [batch, 3, image_size, image_size] image tensor
Output: [batch, num_patches, dim] feature tensor

Size presets:
    tiny            — 2 layers, 128-dim (~500K params, from scratch)
    small           — 4 layers, 256-dim (~4M params, from scratch)  [default]
    medium          — 6 layers, 512-dim (~25M params, from scratch)
    pretrained_tiny — ViT-Ti/16, 192-dim (~5.7M params, ImageNet pretrained)
    pretrained_small— ViT-S/16, 384-dim (~22M params, ImageNet pretrained)
    pretrained_base — ViT-B/16, 768-dim (~86M params, ImageNet pretrained)

Usage:
    from enigma_engine.core.vision_encoder import (
        VisionEncoder, VisionEncoderConfig, VISION_PRESETS,
        preprocess_image, encode_image,
    )

    # From scratch
    encoder = VisionEncoder(VISION_PRESETS["small"])
    features = encode_image(encoder, "photo.png")

    # Pretrained (requires timm)
    encoder = VisionEncoder(VISION_PRESETS["pretrained_small"])
    features = encode_image(encoder, "photo.png")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from enigma_engine.core.model_components import RMSNorm

if TYPE_CHECKING:
    from PIL import Image as _PILImage

logger = logging.getLogger(__name__)

# ImageNet normalization used by all standard pretrained vision models
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class VisionEncoderConfig:
    """
    Configuration for the Vision Encoder.

    Fields:
        image_size:  Input image resolution (square). Images are resized to this.
        patch_size:  Size of each image patch. image_size must be divisible by this.
        channels:    Number of input channels (3 for RGB).
        dim:         Hidden dimension of the transformer.
        n_layers:    Number of transformer blocks.
        n_heads:     Number of attention heads.
        dropout:     Dropout rate during training.
        use_rms_norm: Use RMSNorm (True) or LayerNorm (False).
        use_cnn_stem: Use hybrid CNN+ViT (V-G). When True, a CNN stem
            extracts spatial features (edges, textures, shapes) before
            the transformer processes global relationships. This gives
            ViTs spatial inductive bias that pure attention must learn
            from data, making training more efficient on small datasets.
    """
    image_size: int = 224
    patch_size: int = 16
    channels: int = 3
    dim: int = 256
    n_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.1
    use_rms_norm: bool = True
    use_cnn_stem: bool = False
    use_pretrained: bool = False
    pretrained_model: str = "vit_small_patch16_224"
    freeze_backbone: bool = True

    def __post_init__(self):
        if self.patch_size < 1:
            raise ValueError(f"patch_size must be >= 1, got {self.patch_size}")

    @property
    def num_patches(self) -> int:
        """Number of patches the image gets split into."""
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size ({self.image_size}) must be divisible by "
                f"patch_size ({self.patch_size})"
            )
        return (self.image_size // self.patch_size) ** 2

    def to_dict(self) -> dict:
        """Convert config to a plain dict for serialization."""
        return {
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "channels": self.channels,
            "dim": self.dim,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "dropout": self.dropout,
            "use_rms_norm": self.use_rms_norm,
            "use_cnn_stem": self.use_cnn_stem,
            "use_pretrained": self.use_pretrained,
            "pretrained_model": self.pretrained_model,
            "freeze_backbone": self.freeze_backbone,
        }


# =============================================================================
# SIZE PRESETS
# =============================================================================

VISION_PRESETS: dict[str, VisionEncoderConfig] = {
    "tiny": VisionEncoderConfig(
        image_size=224, patch_size=16, dim=128, n_layers=2, n_heads=4,
    ),
    "small": VisionEncoderConfig(
        image_size=224, patch_size=16, dim=256, n_layers=4, n_heads=4,
    ),
    "medium": VisionEncoderConfig(
        image_size=224, patch_size=16, dim=512, n_layers=6, n_heads=8,
    ),
    # V-G: Hybrid CNN+ViT presets — CNN stem provides spatial features
    "hybrid_small": VisionEncoderConfig(
        image_size=224, patch_size=16, dim=256, n_layers=4, n_heads=4,
        use_cnn_stem=True,
    ),
    "hybrid_medium": VisionEncoderConfig(
        image_size=224, patch_size=16, dim=512, n_layers=6, n_heads=8,
        use_cnn_stem=True,
    ),
    # Pretrained ViT presets — require timm. Massive quality jump over from-scratch.
    "pretrained_tiny": VisionEncoderConfig(
        image_size=224, patch_size=16, dim=192, n_layers=12, n_heads=3,
        use_pretrained=True, pretrained_model="vit_tiny_patch16_224",
    ),
    "pretrained_small": VisionEncoderConfig(
        image_size=224, patch_size=16, dim=384, n_layers=12, n_heads=6,
        use_pretrained=True, pretrained_model="vit_small_patch16_224",
    ),
    "pretrained_base": VisionEncoderConfig(
        image_size=224, patch_size=16, dim=768, n_layers=12, n_heads=12,
        use_pretrained=True, pretrained_model="vit_base_patch16_224",
    ),
}


# =============================================================================
# COMPONENTS
# =============================================================================

class CNNStem(nn.Module):
    """CNN stem for hybrid CNN+ViT architecture (V-G).

    Provides spatial inductive bias that pure ViTs lack — CNN layers
    learn edge/texture/shape hierarchies from their convolutional
    structure, while transformers excel at global reasoning.

    Architecture (3 stages):
        [B, 3, H, W] → Conv1(3→C1, k=3, s=2) → BN → GELU → pool
                      → Conv2(C1→C2, k=3, s=1) → BN → GELU → pool
                      → Conv3(C2→dim, k=3, s=1) → BN → GELU
                      → flatten → [B, num_patches, dim]

    The CNN replaces direct patch embedding, giving the ViT access
    to pre-processed spatial features instead of raw pixel patches.
    This provides the spatial hierarchy (edges → textures → shapes)
    that pure attention must learn from data.
    """

    def __init__(self, channels: int, dim: int) -> None:
        super().__init__()
        # Progressive channel expansion: 3 → dim//4 → dim//2 → dim
        c1 = max(32, dim // 4)
        c2 = max(64, dim // 2)

        self.stage1 = nn.Sequential(
            nn.Conv2d(channels, c1, kernel_size=3, stride=2,
                      padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(c1, c2, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(c2, dim, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process image through CNN stages.

        Args:
            x: [B, C, H, W] image tensor.

        Returns:
            [B, num_patches, dim] feature tensor (spatial dims flattened).
        """
        x = self.stage1(x)   # [B, c1, H/4, W/4]
        x = self.stage2(x)   # [B, c2, H/16, W/16]
        x = self.stage3(x)   # [B, dim, H/16, W/16]
        # Flatten spatial dims: [B, dim, H', W'] → [B, H'*W', dim]
        x = x.flatten(2).transpose(1, 2)
        return x


class PatchEmbedding(nn.Module):
    """
    Convert an image into a sequence of patch embeddings.

    Uses a single Conv2d with kernel_size=patch_size and stride=patch_size
    to chop the image into non-overlapping patches and project each patch
    to the transformer hidden dimension.

    Input:  [batch, channels, H, W]
    Output: [batch, num_patches, dim]
    """

    def __init__(self, patch_size: int, channels: int, dim: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            channels, dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] → conv: [B, dim, H/P, W/P] → flatten: [B, num_patches, dim]
        h, w = x.shape[2], x.shape[3]
        if h % self.patch_size != 0 or w % self.patch_size != 0:
            logger.warning(
                "Image dimensions (%d, %d) not evenly divisible "
                "by patch_size %d -- edge pixels will be discarded",
                h, w, self.patch_size)
        x = self.proj(x)                        # [B, dim, H/P, W/P]
        x = x.flatten(2)                         # [B, dim, num_patches]
        x = x.transpose(1, 2)                    # [B, num_patches, dim]
        return x


class _VisionAttention(nn.Module):
    """
    Multi-head self-attention for vision patches.

    Simpler than the text Attention — no KV cache, no RoPE, no GQA.
    Vision patches don't need positional rotation (we use learned position
    embeddings instead) and there's no autoregressive generation.
    """

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # Project to Q, K, V in one shot
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)       # [3, B, heads, N, head_dim]
        q, k, v = qkv.unbind(0)                  # each: [B, heads, N, head_dim]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, N, N]
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # Apply attention to values
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.out_proj(out)


class _VisionFeedForward(nn.Module):
    """
    Feed-forward network for vision transformer blocks.

    Uses SwiGLU activation (same as text model) for better learning:
    FFN(x) = (SiLU(W1·x) * W3·x) · W2
    """

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        hidden = int(dim * 4 * 2 / 3)  # SwiGLU convention: 8/3 * dim
        # Round to nearest multiple of 8 for hardware efficiency
        hidden = ((hidden + 7) // 8) * 8
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class _VisionBlock(nn.Module):
    """
    Single vision transformer block: Norm → Attention → + → Norm → FFN → +

    Same pre-norm architecture as the text transformer blocks.
    """

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1,
                 use_rms_norm: bool = True) -> None:
        super().__init__()
        Norm = RMSNorm if use_rms_norm else nn.LayerNorm
        self.norm1 = Norm(dim)
        self.attn = _VisionAttention(dim, n_heads, dropout)
        self.norm2 = Norm(dim)
        self.ffn = _VisionFeedForward(dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm with residual connections
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# =============================================================================
# MAIN ENCODER
# =============================================================================

class VisionEncoder(nn.Module):
    """
    Vision Transformer (ViT) encoder.

    Three modes:
        1. Pure ViT (default): PatchEmbedding → Transformer → features
        2. Hybrid CNN+ViT (use_cnn_stem=True): CNNStem → Transformer → features
        3. Pretrained (use_pretrained=True): timm backbone → optional projection

    Pretrained mode loads ImageNet-trained ViT weights via timm for a massive
    quality jump over training from scratch. The backbone can be frozen
    (freeze_backbone=True) to train only the projection layer, or unfrozen
    for full fine-tuning.

    Output is always [batch, num_patches, dim] regardless of mode.

    Args:
        config: VisionEncoderConfig with architecture settings.
    """

    def __init__(self, config: VisionEncoderConfig) -> None:
        super().__init__()
        self.config = config

        if config.use_pretrained:
            self._init_pretrained(config)
        elif config.use_cnn_stem:
            self._init_hybrid(config)
        else:
            self._init_from_scratch(config)

    def _init_pretrained(self, config: VisionEncoderConfig) -> None:
        """Initialize with a pretrained timm backbone."""
        try:
            _ensure_timm()
        except ImportError:
            logger.warning(
                "timm not installed -- falling back to from-scratch "
                "vision encoder")
            config.use_pretrained = False
            self._init_from_scratch(config)
            return
        import timm

        self.backbone = timm.create_model(
            config.pretrained_model,
            pretrained=True,
            num_classes=0,  # Remove classification head
        )
        backbone_dim = self.backbone.embed_dim

        # Projection to config.dim if backbone output dim differs
        if backbone_dim != config.dim:
            self.backbone_proj = nn.Linear(backbone_dim, config.dim, bias=False)
            nn.init.trunc_normal_(self.backbone_proj.weight, std=0.02)
        else:
            self.backbone_proj = None

        # Null out from-scratch components (not used in pretrained mode)
        self.cnn_stem = None
        self.patch_embed = None
        self.pos_embed = None
        self.blocks = nn.ModuleList()
        self.norm = nn.Identity()

        if config.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        logger.info(
            "VisionEncoder: pretrained %s (dim=%d%s, %s)",
            config.pretrained_model,
            backbone_dim,
            f" -> {config.dim}" if backbone_dim != config.dim else "",
            "frozen" if config.freeze_backbone else "trainable",
        )

    def _init_hybrid(self, config: VisionEncoderConfig) -> None:
        """Initialize hybrid CNN+ViT from scratch."""
        self.backbone = None
        self.backbone_proj = None

        self.cnn_stem = CNNStem(
            channels=config.channels,
            dim=config.dim,
        )
        self.patch_embed = None
        # CNN stem output spatial size: image_size / 8
        # (2× from stride-2 conv × 2× from pool1 × 2× from pool2)
        cnn_patches = (config.image_size // 8) ** 2
        self.pos_embed = nn.Parameter(
            torch.randn(1, cnn_patches, config.dim) * 0.02
        )
        logger.info(
            "VisionEncoder: hybrid CNN+ViT mode "
            "(CNN stem -> %d patches -> %d transformer blocks)",
            cnn_patches, config.n_layers)

        self._init_blocks_and_norm(config)
        self._init_weights()

    def _init_from_scratch(self, config: VisionEncoderConfig) -> None:
        """Initialize pure ViT from scratch."""
        self.backbone = None
        self.backbone_proj = None
        self.cnn_stem = None

        self.patch_embed = PatchEmbedding(
            patch_size=config.patch_size,
            channels=config.channels,
            dim=config.dim,
        )
        self.pos_embed = nn.Parameter(
            torch.randn(1, config.num_patches, config.dim) * 0.02
        )

        self._init_blocks_and_norm(config)
        self._init_weights()

    def _init_blocks_and_norm(self, config: VisionEncoderConfig) -> None:
        """Shared init for transformer blocks and final norm."""
        self.blocks = nn.ModuleList([
            _VisionBlock(
                dim=config.dim,
                n_heads=config.n_heads,
                dropout=config.dropout,
                use_rms_norm=config.use_rms_norm,
            )
            for _ in range(config.n_layers)
        ])
        Norm = RMSNorm if config.use_rms_norm else nn.LayerNorm
        self.norm = Norm(config.dim)

    def _init_weights(self) -> None:
        """Initialize weights with small values for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode an image tensor into patch features.

        Args:
            x: Image tensor [batch, channels, image_size, image_size].
               Use [-1, 1] normalization for from-scratch models,
               ImageNet normalization for pretrained models.

        Returns:
            Feature tensor [batch, num_patches, dim]
        """
        if x.ndim != 4:
            raise ValueError(
                f"VisionEncoder expects 4D input [B, C, H, W], "
                f"got {x.ndim}D shape {tuple(x.shape)}")
        # Pretrained: use timm backbone
        if self.backbone is not None:
            features = self.backbone.forward_features(x)
            # Strip prefix tokens (CLS, registers) to get patch tokens only
            n_prefix = getattr(self.backbone, "num_prefix_tokens", 0)
            if n_prefix > 0:
                features = features[:, n_prefix:]
            if self.backbone_proj is not None:
                features = self.backbone_proj(features)
            return features

        # From-scratch: CNN stem or patch embedding
        if self.cnn_stem is not None:
            x = self.cnn_stem(x)         # [B, cnn_patches, dim]
        else:
            x = self.patch_embed(x)      # [B, num_patches, dim]

        # Add position embeddings
        x = x + self.pos_embed

        # Transform through blocks
        for block in self.blocks:
            x = block(x)

        # Final normalization
        x = self.norm(x)

        return x

    def param_count(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# TEMPORAL MODELING (VIDEO)
# =============================================================================

class TemporalConv1d(nn.Module):
    """Lightweight 1D temporal convolution for video frame features.

    Applies convolution across the time (frame) dimension so that
    adjacent frame features can exchange information. This enables
    the model to capture motion and temporal context that per-frame
    encoding misses entirely.

    Input:  list of N frame features [1, num_patches, dim]
    Output: list of N frame features [1, num_patches, dim] (same shape)

    Inspired by openpilot's temporal conv approach — much lighter
    than full temporal attention (TimeSformer) while still providing
    cross-frame information flow.
    """

    def __init__(self, dim: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2  # same-padding
        self.conv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=padding)
        self.norm = RMSNorm(dim)

    def forward(self, frame_features: list[torch.Tensor]) -> list[torch.Tensor]:
        """Apply temporal convolution across frames.

        Args:
            frame_features: List of N tensors, each [1, num_patches, dim].

        Returns:
            List of N tensors with temporal context mixed in.
        """
        if len(frame_features) < 2:
            return frame_features

        # Average-pool each frame to get one vector per frame: [N, dim]
        frame_summaries = torch.stack(
            [f.squeeze(0).mean(dim=0) for f in frame_features], dim=0
        )  # [N, dim]

        # Conv1d expects [batch, channels, seq_len]
        x = frame_summaries.unsqueeze(0).permute(0, 2, 1)  # [1, dim, N]
        x = self.conv(x)  # [1, dim, N]
        x = x.permute(0, 2, 1).squeeze(0)  # [N, dim]
        x = self.norm(x)

        # Add temporal context back to each frame's features via broadcast
        result = []
        for i, feat in enumerate(frame_features):
            # feat: [1, num_patches, dim], x[i]: [dim]
            result.append(feat + x[i].unsqueeze(0).unsqueeze(0))

        return result


# =============================================================================
# IMAGE PREPROCESSING
# =============================================================================

def _ensure_pil() -> None:
    """Ensure PIL is available, raise ImportError with helpful message if not."""
    try:
        import PIL  # noqa: F401
    except ImportError:
        raise ImportError(
            "Pillow is required for image processing. "
            "Install with: pip install Pillow"
        ) from None


def _ensure_timm() -> None:
    """Ensure timm is available for pretrained vision models."""
    try:
        import timm  # noqa: F401
    except ImportError:
        raise ImportError(
            "timm is required for pretrained vision models. "
            "Install with: pip install timm"
        ) from None


def preprocess_image(
    image: Union[str, Path, "_PILImage"],
    image_size: int = 224,
    imagenet_normalize: bool = False,
) -> torch.Tensor:
    """
    Preprocess an image for the vision encoder.

    Handles:
    - File paths (str or Path) → loads via PIL
    - PIL Image objects → converts directly
    - Grayscale/RGBA → converts to RGB
    - Any size → resizes to image_size × image_size
    - Normalizes pixel values to [-1, 1] (default) or ImageNet stats

    Args:
        image: PIL Image, file path string, or Path object.
        image_size: Target square size (default 224).
        imagenet_normalize: If True, normalize with ImageNet mean/std
            (required for pretrained models). If False, normalize to [-1, 1].

    Returns:
        Tensor of shape [1, 3, image_size, image_size].
    """
    _ensure_pil()
    from PIL import Image

    # Load from path if needed
    if isinstance(image, (str, Path)):
        image = Image.open(str(image))

    # Convert to RGB (handles grayscale, RGBA, palette, etc.)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize to target size
    image = image.resize((image_size, image_size), Image.BILINEAR)

    # Convert to tensor: HWC uint8 → CHW float
    import numpy as np
    arr = np.array(image, dtype=np.float32)      # [H, W, 3] in [0, 255]

    if imagenet_normalize:
        # ImageNet normalization: (pixel/255 - mean) / std
        arr = arr / 255.0
        mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        std = np.array(IMAGENET_STD, dtype=np.float32)
        arr = (arr - mean) / std
    else:
        arr = arr / 127.5 - 1.0                    # normalize to [-1, 1]

    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]

    return tensor.unsqueeze(0)  # [1, 3, H, W]


def augment_vision_tensor(img: torch.Tensor) -> torch.Tensor:
    """Apply random augmentation to a pre-processed image tensor.

    Designed for training-time data augmentation.  All transforms operate
    directly on tensors in [-1, 1] range, no extra dependencies needed.

    Augmentations applied (each with independent probability):
    - Random horizontal flip (50%)
    - Brightness jitter (±0.1)
    - Contrast jitter (0.8× – 1.2×)
    - Saturation jitter (50% chance, 0.6× – 1.4×)

    Args:
        img: Tensor [1, 3, H, W] in [-1, 1] range.

    Returns:
        Augmented tensor, same shape and range.
    """
    import random as _rand

    # Random horizontal flip
    if _rand.random() < 0.5:
        img = torch.flip(img, dims=[-1])

    # Brightness jitter
    brightness = (_rand.random() - 0.5) * 0.2
    img = (img + brightness).clamp(-1, 1)

    # Contrast jitter
    contrast = 0.8 + _rand.random() * 0.4
    mean = img.mean()
    img = ((img - mean) * contrast + mean).clamp(-1, 1)

    # Saturation jitter
    if _rand.random() < 0.5:
        gray = img.mean(dim=-3, keepdim=True)
        saturation = 0.6 + _rand.random() * 0.8
        img = (saturation * img + (1 - saturation) * gray).clamp(-1, 1)

    return img


# =============================================================================
# CONVENIENCE ENCODE FUNCTIONS
# =============================================================================

@torch.no_grad()
def encode_image(
    encoder: VisionEncoder,
    image: Union[str, Path, "_PILImage"],
) -> torch.Tensor:
    """
    Preprocess and encode a single image.

    Args:
        encoder: Trained VisionEncoder instance.
        image: PIL Image, file path, or Path.

    Returns:
        Feature tensor [1, num_patches, dim].
    """
    tensor = preprocess_image(
        image,
        image_size=encoder.config.image_size,
        imagenet_normalize=encoder.config.use_pretrained,
    )
    tensor = tensor.to(next(encoder.parameters()).device)
    encoder.eval()
    return encoder(tensor)


@torch.no_grad()
def encode_video_frames(
    encoder: VisionEncoder,
    video_path: Union[str, Path],
    max_frames: int = 8,
    max_visual_tokens: int = 0,
    dedup_threshold: float = 0.95,
    temporal_conv: TemporalConv1d | None = None,
) -> torch.Tensor:
    """
    Sample frames from a video and encode each one.

    Requires OpenCV (cv2). Samples evenly-spaced frames, drops
    near-duplicate frames (cosine similarity > dedup_threshold),
    optionally applies temporal convolution for cross-frame context,
    and truncates to max_visual_tokens.

    Args:
        encoder: Trained VisionEncoder instance.
        video_path: Path to video file.
        max_frames: Maximum number of frames to sample.
        max_visual_tokens: Maximum total tokens (patches) to return.
            0 means no limit.
        dedup_threshold: Cosine similarity threshold for dropping
            duplicate frames. Set to 1.0 to disable dedup.
        temporal_conv: Optional TemporalConv1d module. When provided,
            applies 1D convolution across frames so each frame can
            see temporal context from its neighbors.

    Returns:
        Feature tensor [1, N*num_patches, dim] (concatenated frames).
    """
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "OpenCV is required for video encoding. "
            "Install with: pip install opencv-python"
        ) from None

    _ensure_pil()
    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Video has no frames: {video_path}")

    # Sample evenly-spaced frame indices
    n_sample = min(max_frames, total_frames)
    indices = [int(i * total_frames / n_sample) for i in range(n_sample)]

    device = next(encoder.parameters()).device
    encoder.eval()
    all_features = []

    try:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to read video frame %d", idx)
                continue
            # OpenCV BGR → RGB → PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            tensor = preprocess_image(
                pil_img,
                image_size=encoder.config.image_size,
                imagenet_normalize=encoder.config.use_pretrained,
            )
            tensor = tensor.to(device)
            features = encoder(tensor)  # [1, num_patches, dim]
            all_features.append(features)
    finally:
        cap.release()

    if not all_features:
        raise ValueError(f"Could not read any frames from: {video_path}")

    # Drop near-duplicate consecutive frames (cosine similarity)
    if dedup_threshold < 1.0 and len(all_features) > 1:
        unique: list[torch.Tensor] = [all_features[0]]
        for feat in all_features[1:]:
            prev = unique[-1].reshape(-1)
            curr = feat.reshape(-1)
            cos_sim = torch.nn.functional.cosine_similarity(
                prev.unsqueeze(0), curr.unsqueeze(0)).item()
            if cos_sim < dedup_threshold:
                unique.append(feat)
        all_features = unique

    # Apply temporal convolution if provided
    if temporal_conv is not None and len(all_features) > 1:
        all_features = temporal_conv(all_features)

    # Concatenate all frame features along the sequence dimension
    combined = torch.cat(all_features, dim=1)  # [1, N*num_patches, dim]

    # Truncate to max_visual_tokens if specified
    if max_visual_tokens > 0 and combined.shape[1] > max_visual_tokens:
        combined = combined[:, :max_visual_tokens, :]

    return combined


@torch.no_grad()
def encode_screen(encoder: VisionEncoder) -> torch.Tensor:
    """
    Capture the screen and encode it.

    Uses PIL.ImageGrab for screen capture (works on Windows/macOS).

    Args:
        encoder: Trained VisionEncoder instance.

    Returns:
        Feature tensor [1, num_patches, dim].
    """
    _ensure_pil()
    from PIL import ImageGrab

    screenshot = ImageGrab.grab()
    return encode_image(encoder, screenshot)


@torch.no_grad()
def encode_camera(
    encoder: VisionEncoder,
    device_id: int = 0,
) -> torch.Tensor:
    """
    Capture a frame from the webcam and encode it.

    Requires OpenCV (cv2).

    Args:
        encoder: Trained VisionEncoder instance.
        device_id: Camera device index (default 0).

    Returns:
        Feature tensor [1, num_patches, dim].
    """
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "OpenCV is required for camera capture. "
            "Install with: pip install opencv-python"
        ) from None

    _ensure_pil()
    from PIL import Image

    cap = cv2.VideoCapture(device_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera device {device_id}")

    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Failed to capture frame from camera {device_id}")

    # BGR → RGB → PIL → encode
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    return encode_image(encoder, pil_img)
