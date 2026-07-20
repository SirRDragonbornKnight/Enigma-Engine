"""
Audio Encoder — Whisper-style Conv1d + Transformer for audio understanding.

Architecture (from OpenAI Whisper):
    Waveform → Mel Spectrogram → Conv1d(n_mels→dim) → GELU
        → Conv1d(dim→dim, stride=2) → GELU → + Sinusoidal Pos Embed
        → N TransformerBlocks → RMSNorm → features

Input:  mel spectrogram [batch, n_mels, n_frames]
Output: [batch, n_frames // 2, dim] feature tensor

The stride-2 second convolution halves the temporal dimension,
reducing computational cost in the transformer blocks.

Size presets (matching Whisper naming):
    tiny  — 4 layers, 384-dim, 6 heads  (~10M params)
    base  — 6 layers, 512-dim, 8 heads  (~25M params)
    small — 12 layers, 768-dim, 12 heads (~95M params)

Audio preprocessing:
    - ``load_audio()`` reads WAV files via stdlib (no external deps)
    - ``mel_filterbank()`` computes mel filter matrix using torch
    - ``log_mel_spectrogram()`` converts waveform → log-mel via torch.stft
    - ``preprocess_audio()`` convenience: file path or tensor → mel tensor

Usage:
    from enigma_engine.core.audio_encoder import (
        AudioEncoder, AudioEncoderConfig, AUDIO_PRESETS,
        preprocess_audio, log_mel_spectrogram,
    )

    encoder = AudioEncoder(AUDIO_PRESETS["tiny"])
    mel = preprocess_audio("speech.wav")
    features = encoder(mel)  # [1, n_frames/2, 384]
"""
from __future__ import annotations

import logging
import math
import struct
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from enigma_engine.core.model_components import RMSNorm

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AudioEncoderConfig:
    """
    Configuration for the Audio Encoder.

    Fields:
        n_mels:        Number of mel frequency bins (Whisper uses 80).
        dim:           Hidden dimension of the transformer.
        n_layers:      Number of transformer blocks.
        n_heads:       Number of attention heads.
        dropout:       Dropout rate during training.
        max_audio_len: Max mel frames after stride-2 conv (for positional
                       embeddings). 1500 = ~30s at 16kHz with hop=160.
        sample_rate:   Expected audio sample rate in Hz.
        n_fft:         FFT window size for STFT (400 = 25ms at 16kHz).
        hop_length:    STFT hop size (160 = 10ms at 16kHz).
    """
    n_mels: int = 80
    dim: int = 384
    n_layers: int = 4
    n_heads: int = 6
    dropout: float = 0.1
    max_audio_len: int = 1500
    sample_rate: int = 16000
    n_fft: int = 400
    hop_length: int = 160
    use_conformer: bool = False

    def to_dict(self) -> dict:
        """Convert config to a plain dict for serialization."""
        return {
            "n_mels": self.n_mels,
            "dim": self.dim,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "dropout": self.dropout,
            "max_audio_len": self.max_audio_len,
            "sample_rate": self.sample_rate,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "use_conformer": self.use_conformer,
        }


# =============================================================================
# SIZE PRESETS
# =============================================================================

AUDIO_PRESETS: dict[str, AudioEncoderConfig] = {
    "tiny": AudioEncoderConfig(
        n_mels=80, dim=384, n_layers=4, n_heads=6,
    ),
    "base": AudioEncoderConfig(
        n_mels=80, dim=512, n_layers=6, n_heads=8,
    ),
    "small": AudioEncoderConfig(
        n_mels=80, dim=768, n_layers=12, n_heads=12,
    ),
}


# =============================================================================
# COMPONENTS
# =============================================================================

class _AudioAttention(nn.Module):
    """Multi-head self-attention for audio frames.

    Same structure as vision attention — no KV cache, no RoPE, no GQA.
    Audio uses sinusoidal positional embeddings added before the blocks.
    """

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """key_padding_mask: [B, N] bool, True = PADDING (key masked out).

        Padded batches (ragged mel lengths) need this: attention is
        global, so without it every real frame would attend to garbage
        pad frames -- the exact silent corruption train_audio's old
        batch_size=1 refusal existed to prevent.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if key_padding_mask is not None:
            # [B, N] -> [B, 1, 1, N]; -inf on pad KEYS. Pad QUERY rows
            # softmax over all -inf... would be NaN, so pad queries keep
            # one finite entry (themselves) -- their outputs are garbage
            # by design and are zeroed by the encoder after every block.
            neg = torch.finfo(attn.dtype).min
            km = key_padding_mask[:, None, None, :]
            attn = attn.masked_fill(km & ~torch.eye(N, dtype=torch.bool, device=x.device), neg)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.out_proj(out)


class _AudioFeedForward(nn.Module):
    """Feed-forward with SwiGLU activation (same as text/vision models)."""

    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        hidden = int(dim * 4 * 2 / 3)
        hidden = ((hidden + 7) // 8) * 8  # round to 8 for hardware
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class _ConformerConv(nn.Module):
    """Conformer-style depthwise convolution module (espnet pattern).

    Inserted between attention and FFN in each audio block when
    ``use_conformer=True``.  Captures local acoustic patterns that
    pure attention misses.

    Architecture: LayerNorm → Pointwise(dim→2*dim) → GLU → DepthwiseConv1d
                  → BatchNorm → SwiGLU activation → Pointwise(dim→dim) → Dropout
    """

    def __init__(self, dim: int, kernel_size: int = 31,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = RMSNorm(dim)
        # Pointwise expansion (GLU halves, so expand 2x)
        self.pw1 = nn.Linear(dim, dim * 2, bias=False)
        # Depthwise conv: groups=dim means each channel is convolved separately
        padding = (kernel_size - 1) // 2
        self.dw_conv = nn.Conv1d(
            dim, dim, kernel_size, padding=padding, groups=dim, bias=False)
        self.bn = nn.BatchNorm1d(dim)
        self.pw2 = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, dim] → [B, T, dim]"""
        residual = x
        x = self.norm(x)
        # Pointwise + GLU gate
        x = self.pw1(x)
        x, gate = x.chunk(2, dim=-1)
        x = x * torch.sigmoid(gate)
        # Depthwise conv: [B, T, dim] → [B, dim, T] → conv → [B, T, dim]
        x = x.transpose(1, 2)
        x = self.bn(F.silu(self.dw_conv(x)))
        x = x.transpose(1, 2)
        # Pointwise back to dim
        x = self.dropout(self.pw2(x))
        return residual + x


class _AudioBlock(nn.Module):
    """Single audio transformer block.

    Standard: Norm → Attn → + → Norm → FFN → +
    Conformer: Norm → Attn → + → ConvModule → + → Norm → FFN → +
    """

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1,
                 use_conformer: bool = False) -> None:
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = _AudioAttention(dim, n_heads, dropout)
        self.conv_module = _ConformerConv(dim, dropout=dropout) if use_conformer else None
        self.norm2 = RMSNorm(dim)
        self.ffn = _AudioFeedForward(dim, dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), key_padding_mask)
        if self.conv_module is not None:
            x = self.conv_module(x)
        x = x + self.ffn(self.norm2(x))
        return x


def _sinusoidal_embed(max_len: int, dim: int) -> torch.Tensor:
    """Compute sinusoidal positional embeddings (Whisper-style, not learned).

    Returns:
        [1, max_len, dim] tensor of positional embeddings.
    """
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    dim_idx = torch.arange(0, dim, 2, dtype=torch.float32)
    div_term = torch.exp(dim_idx * -(math.log(10000.0) / dim))
    pe = torch.zeros(max_len, dim)
    pe[:, 0::2] = torch.sin(pos * div_term)
    pe[:, 1::2] = torch.cos(pos * div_term[: dim // 2])
    return pe.unsqueeze(0)  # [1, max_len, dim]


# =============================================================================
# MAIN ENCODER
# =============================================================================

class AudioEncoder(nn.Module):
    """
    Whisper-style audio encoder.

    Architecture:
        Conv1d(n_mels -> dim, k=3, p=1) -> GELU
        -> Conv1d(dim -> dim, k=3, stride=2, p=1) -> GELU
        -> + sinusoidal positional embeddings
        -> N transformer blocks
        -> RMSNorm
        -> [batch, n_frames/2, dim]

    The stride-2 second convolution halves temporal resolution,
    reducing compute for the transformer while preserving information.

    Args:
        config: AudioEncoderConfig with architecture settings.
    """

    def __init__(self, config: AudioEncoderConfig) -> None:
        super().__init__()
        self.config = config

        # Whisper-style convolutional front-end
        self.conv1 = nn.Conv1d(config.n_mels, config.dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(config.dim, config.dim, kernel_size=3, stride=2, padding=1)

        # Sinusoidal positional embeddings (not learned, like Whisper)
        self.register_buffer(
            "pos_embed",
            _sinusoidal_embed(config.max_audio_len, config.dim),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            _AudioBlock(
                dim=config.dim,
                n_heads=config.n_heads,
                dropout=config.dropout,
                use_conformer=config.use_conformer,
            )
            for _ in range(config.n_layers)
        ])

        self.norm = RMSNorm(config.dim)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with small values for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @staticmethod
    def feature_lengths(mel_lengths: torch.Tensor) -> torch.Tensor:
        """Mel-frame counts -> feature-frame counts after the stride-2
        conv2 (k=3, p=1): out = floor((T - 1) / 2) + 1 = ceil(T / 2)."""
        return (mel_lengths + 1) // 2

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """
        Encode a mel spectrogram into audio features.

        Args:
            x: Mel spectrogram [batch, n_mels, n_frames]. Ragged batches
                are padded along n_frames (pad value irrelevant -- see
                lengths).
            lengths: Optional [batch] int tensor of VALID mel-frame
                counts per sample. When given, pad frames are excluded
                from attention and their features zeroed after every
                stage, so a padded batch's valid rows match the
                unbatched forward. When None: exact legacy behaviour
                (every frame is real -- the batch_size=1 align path and
                the fixed 30 s distill windows).

        Returns:
            Feature tensor [batch, ceil(n_frames / 2), dim]; rows at and
            beyond a sample's feature_lengths entry are exact zeros.
        """
        pad_mask = None  # [B, T_feat] bool, True = padding
        if lengths is not None:
            if self.config.use_conformer:
                # ConformerConv's BatchNorm1d computes batch statistics
                # over every time step; pad frames would poison them
                # (and train/eval drift would hide it). Refuse loudly
                # until a masked-stats conv module exists.
                raise NotImplementedError(
                    "padding mask with use_conformer=True: BatchNorm1d would "
                    "mix pad frames into batch statistics -- train conformer "
                    "encoders at batch_size=1 or implement masked BN first"
                )
            t_feat = (x.shape[-1] + 1) // 2
            feat_lens = self.feature_lengths(lengths.to(x.device))
            pad_mask = (
                torch.arange(t_feat, device=x.device)[None, :] >= feat_lens[:, None]
            )
            # Zero the INPUT pad region: the k=3 convs read one frame
            # across the valid/pad edge, and conv's own padding=1 already
            # zero-pads beyond the tensor edge -- so zeroed pads make the
            # batched boundary arithmetic IDENTICAL to the unbatched one.
            in_pad = (
                torch.arange(x.shape[-1], device=x.device)[None, :]
                >= lengths.to(x.device)[:, None]
            )
            x = x.masked_fill(in_pad[:, None, :], 0.0)

        # Conv front-end: [B, n_mels, T] -> [B, dim, T] -> [B, dim, T//2]
        x = F.gelu(self.conv1(x))
        if lengths is not None:
            # conv1's output at the first pad position absorbs real
            # neighbours through the k=3 kernel edge (and gelu(bias)
            # elsewhere in the pad); unbatched, conv2 sees implicit
            # ZEROS there. Zeroing makes both paths read identically.
            x = x.masked_fill(in_pad[:, None, :], 0.0)
        x = F.gelu(self.conv2(x))

        # Transpose to [B, T//2, dim] for transformer
        x = x.transpose(1, 2)

        # Add sinusoidal positional embeddings (truncate to actual length)
        T = x.shape[1]
        x = x + self.pos_embed[:, :T, :]

        if pad_mask is not None:
            # Kill conv bias bleed into the pad region before attention.
            x = x.masked_fill(pad_mask[:, :, None], 0.0)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, pad_mask)
            if pad_mask is not None:
                # Pad-query outputs are garbage by design (they only see
                # themselves); re-zero so residuals never accumulate.
                x = x.masked_fill(pad_mask[:, :, None], 0.0)

        # Final norm
        x = self.norm(x)

        if pad_mask is not None:
            # RMSNorm of zeros is zeros only in exact math; guarantee it.
            x = x.masked_fill(pad_mask[:, :, None], 0.0)

        return x

    def param_count(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# MEL SPECTROGRAM UTILITIES (torch-only, no external deps)
# =============================================================================

def _hz_to_mel(hz: float) -> float:
    """Convert frequency in Hz to mel scale."""
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    """Convert mel scale to frequency in Hz."""
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(
    sr: int = 16000,
    n_fft: int = 400,
    n_mels: int = 80,
) -> torch.Tensor:
    """
    Compute a mel filterbank matrix.

    Follows the HTK mel scale (same as librosa and Whisper).

    Args:
        sr: Sample rate in Hz.
        n_fft: FFT window size.
        n_mels: Number of mel frequency bins.

    Returns:
        Mel filterbank [n_mels, n_fft // 2 + 1].
    """
    if sr <= 0:
        raise ValueError(f"Sample rate must be positive, got {sr}")
    n_freqs = n_fft // 2 + 1
    # Mel-spaced center frequencies
    mel_low = _hz_to_mel(0.0)
    mel_high = _hz_to_mel(sr / 2.0)
    mel_points = torch.linspace(mel_low, mel_high, n_mels + 2)
    hz_points = torch.tensor([_mel_to_hz(m.item()) for m in mel_points])

    # Convert Hz to FFT bin indices
    bins = (hz_points * n_fft / sr).long()

    fb = torch.zeros(n_mels, n_freqs)
    for i in range(n_mels):
        left = bins[i]
        center = bins[i + 1]
        right = bins[i + 2]

        # Rising slope
        for j in range(left, center):
            if j < n_freqs and center > left:
                fb[i, j] = (j - left).float() / (center - left).float()

        # Falling slope
        for j in range(center, right):
            if j < n_freqs and right > center:
                fb[i, j] = (right - j).float() / (right - center).float()

    return fb


def log_mel_spectrogram(
    waveform: torch.Tensor,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 80,
    sr: int = 16000,
) -> torch.Tensor:
    """
    Compute log-mel spectrogram from a waveform using torch.

    Args:
        waveform: 1D audio tensor (mono, float32, [-1, 1] range).
        n_fft: FFT window size.
        hop_length: STFT hop size.
        n_mels: Number of mel frequency bins.
        sr: Sample rate (for mel filterbank computation).

    Returns:
        Log-mel spectrogram [1, n_mels, n_frames].
    """
    if waveform.ndim > 1:
        waveform = waveform.squeeze()
    if waveform.ndim != 1:
        raise ValueError(f"Expected 1D waveform, got shape {waveform.shape}")

    # STFT
    window = torch.hann_window(n_fft, device=waveform.device)
    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=True,
    )
    # Magnitude spectrogram [n_freqs, n_frames]
    magnitudes = stft.abs() ** 2

    # Mel filterbank [n_mels, n_freqs]
    fb = mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels).to(waveform.device)

    # Apply filterbank: [n_mels, n_freqs] @ [n_freqs, n_frames] = [n_mels, n_frames]
    mel_spec = fb @ magnitudes

    # Log scale with floor for numerical stability
    log_mel = torch.clamp(mel_spec, min=1e-10).log10()
    log_mel = torch.maximum(log_mel, log_mel.max() - 8.0)
    log_mel = (log_mel + 4.0) / 4.0  # Normalize to roughly [-1, 1]

    return log_mel.unsqueeze(0)  # [1, n_mels, n_frames]


# =============================================================================
# AUDIO LOADING AND PREPROCESSING
# =============================================================================

def load_audio(path: Union[str, Path], target_sr: int = 16000) -> torch.Tensor:
    """
    Load an audio file and return a float32 waveform tensor.

    Supports WAV files via stdlib. For other formats, attempts to use
    soundfile if available.

    Args:
        path: Path to audio file.
        target_sr: Target sample rate (resamples if needed).

    Returns:
        1D float32 tensor normalized to [-1, 1].
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".wav":
        return _load_wav(path, target_sr)

    # Try soundfile for other formats
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError(
            f"Cannot load {suffix} files without soundfile. "
            f"Install with: pip install soundfile\n"
            f"Or convert to WAV format."
        ) from None

    data, sr = sf.read(str(path), dtype="float32")
    waveform = torch.from_numpy(data)

    # Convert stereo to mono
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=-1)

    # Resample if needed
    if sr != target_sr:
        waveform = _resample_linear(waveform, sr, target_sr)

    return waveform


def _load_wav(path: Path, target_sr: int) -> torch.Tensor:
    """Load WAV file using stdlib wave module."""
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        if n_channels <= 0 or n_frames <= 0:
            raise ValueError(
                f"Invalid WAV: {n_frames} frames, {n_channels} channels")
        raw = wf.readframes(n_frames)

    # Decode raw bytes to float32
    expected_bytes = n_frames * n_channels * sample_width
    if len(raw) < expected_bytes:
        raise ValueError(
            f"Truncated WAV: expected {expected_bytes} bytes, "
            f"got {len(raw)}"
        )
    if sample_width == 2:
        fmt = f"<{n_frames * n_channels}h"
        samples = struct.unpack(fmt, raw)
        waveform = torch.tensor(samples, dtype=torch.float32) / 32768.0
    elif sample_width == 4:
        fmt = f"<{n_frames * n_channels}i"
        samples = struct.unpack(fmt, raw)
        waveform = torch.tensor(samples, dtype=torch.float32) / 2147483648.0
    elif sample_width == 3:
        # 24-bit WAV: unpack 3-byte little-endian signed integers
        n_samples = n_frames * n_channels
        samples = []
        for i in range(n_samples):
            b = raw[i * 3: i * 3 + 3]
            val = int.from_bytes(b, byteorder="little", signed=True)
            samples.append(val)
        waveform = torch.tensor(samples, dtype=torch.float32) / 8388608.0
    elif sample_width == 1:
        samples = [b - 128 for b in raw]
        waveform = torch.tensor(samples, dtype=torch.float32) / 128.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")

    # Convert stereo to mono
    if n_channels > 1:
        waveform = waveform.view(-1, n_channels).mean(dim=-1)

    # Resample if needed
    if sr != target_sr:
        waveform = _resample_linear(waveform, sr, target_sr)

    return waveform


def _resample_linear(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    """Simple linear interpolation resampling (no external deps)."""
    ratio = target_sr / orig_sr
    new_len = int(len(waveform) * ratio)
    if new_len == 0:
        return torch.zeros_like(waveform[:1])
    return F.interpolate(
        waveform.unsqueeze(0).unsqueeze(0),
        size=new_len,
        mode="linear",
        align_corners=False,
    ).squeeze()


def preprocess_audio(
    audio: Union[str, Path, torch.Tensor],
    config: AudioEncoderConfig | None = None,
) -> torch.Tensor:
    """
    Preprocess audio for the encoder.

    Accepts a file path (loads and converts to mel) or a raw waveform tensor.

    Args:
        audio: File path string/Path, or 1D waveform tensor.
        config: AudioEncoderConfig for spectrogram params. Uses defaults if None.

    Returns:
        Mel spectrogram tensor [1, n_mels, n_frames].
    """
    if config is None:
        config = AudioEncoderConfig()

    # Load from file if path
    if isinstance(audio, (str, Path)):
        waveform = load_audio(audio, target_sr=config.sample_rate)
    elif isinstance(audio, torch.Tensor):
        waveform = audio.squeeze()
        if waveform.ndim != 1:
            raise ValueError(f"Expected 1D waveform tensor, got shape {audio.shape}")
    else:
        raise TypeError(f"Expected str, Path, or Tensor, got {type(audio)}")

    return log_mel_spectrogram(
        waveform,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        sr=config.sample_rate,
    )


def spec_augment(
    mel: torch.Tensor,
    freq_mask_count: int = 2,
    freq_mask_width: int = 10,
    time_mask_count: int = 2,
    time_mask_width: int = 20,
) -> torch.Tensor:
    """Apply SpecAugment to a mel spectrogram (training-time augmentation).

    Masks random contiguous bands in frequency and time dimensions
    with zeros. This forces the audio encoder to learn from
    incomplete spectral information, improving generalization.

    Reference: Park et al., "SpecAugment: A Simple Data Augmentation
    Method for Automatic Speech Recognition" (2019).

    Args:
        mel: Mel spectrogram tensor [1, n_mels, n_frames] or [n_mels, n_frames].
        freq_mask_count: Number of frequency masks to apply.
        freq_mask_width: Maximum width of each frequency mask (in mel bins).
        time_mask_count: Number of time masks to apply.
        time_mask_width: Maximum width of each time mask (in frames).

    Returns:
        Augmented mel spectrogram, same shape and dtype.
    """
    squeezed = mel.ndim == 2
    if squeezed:
        mel = mel.unsqueeze(0)

    _, n_mels_dim, n_frames = mel.shape
    mel = mel.clone()

    # Frequency masking
    for _ in range(freq_mask_count):
        width = torch.randint(0, min(freq_mask_width, n_mels_dim) + 1, (1,)).item()
        if width == 0:
            continue
        start = torch.randint(0, max(n_mels_dim - width, 1), (1,)).item()
        mel[:, start:start + width, :] = 0.0

    # Time masking
    for _ in range(time_mask_count):
        width = torch.randint(0, min(time_mask_width, n_frames) + 1, (1,)).item()
        if width == 0:
            continue
        start = torch.randint(0, max(n_frames - width, 1), (1,)).item()
        mel[:, :, start:start + width] = 0.0

    if squeezed:
        mel = mel.squeeze(0)

    return mel
