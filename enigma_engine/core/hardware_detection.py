"""
Hardware Detection Module

Detects hardware capabilities and recommends optimal model configurations.
"""

import logging
import platform
import threading
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class HardwareProfile:
    """Hardware profile with system capabilities."""

    device: str = "cpu"
    cpu_cores: int = 1
    cpu_threads: int = 1
    ram_gb: float = 4.0
    total_ram_gb: float = 4.0  # Alias for ram_gb
    available_ram_gb: float = 4.0  # Currently available RAM
    gpu_available: bool = False
    gpu_name: str = ""
    gpu_vram_gb: float = 0.0
    is_apple_silicon: bool = False
    is_raspberry_pi: bool = False
    pi_model: str = ""
    cuda_version: str = ""
    hardware_type: str = "unknown"  # "raspberry_pi", "desktop", "server", etc.
    # Convenience attributes
    has_cuda: bool = False
    has_mps: bool = False
    is_arm: bool = False

    def __str__(self) -> str:
        if self.gpu_available:
            return f"{self.gpu_name} ({self.gpu_vram_gb:.1f}GB VRAM)"
        return f"{self.cpu_cores} CPU cores, {self.ram_gb:.1f}GB RAM"

    def to_dict(self) -> dict[str, Any]:
        """Convert profile to dictionary."""
        import dataclasses

        return dataclasses.asdict(self)


_cached_profile: Optional[HardwareProfile] = None
_profile_lock = threading.Lock()


def detect_hardware() -> HardwareProfile:
    """Detect hardware capabilities of the current system."""
    global _cached_profile
    with _profile_lock:
        if _cached_profile is not None:
            return _cached_profile

    profile = HardwareProfile()

    # CPU info
    import os

    profile.cpu_cores = os.cpu_count() or 1
    profile.cpu_threads = profile.cpu_cores

    # Check for ARM architecture
    machine = platform.machine().lower()
    profile.is_arm = machine in ("arm64", "aarch64", "armv7l", "armv8")

    # RAM info
    try:
        import psutil

        mem = psutil.virtual_memory()
        profile.ram_gb = mem.total / (1024**3)
        profile.available_ram_gb = mem.available / (1024**3)
    except ImportError:
        logger.warning("psutil not installed -- defaulting to 8 GB RAM")
        profile.ram_gb = 8.0  # Default assumption
        profile.available_ram_gb = profile.ram_gb * 0.5
    profile.total_ram_gb = profile.ram_gb

    # Check for Raspberry Pi
    try:
        with open("/proc/device-tree/model", "r", encoding="utf-8") as f:
            model = f.read()
            if "Raspberry Pi" in model:
                profile.is_raspberry_pi = True
                profile.pi_model = model.strip().rstrip("\x00")
                profile.hardware_type = "raspberry_pi"
    except (FileNotFoundError, OSError):
        pass

    # Check for Apple Silicon
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        profile.is_apple_silicon = True
        profile.hardware_type = "apple_silicon"

    # GPU detection
    try:
        import torch

        if torch.cuda.is_available():
            profile.gpu_available = True
            profile.has_cuda = True
            profile.device = "cuda"
            profile.gpu_name = torch.cuda.get_device_name(0)
            profile.gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            profile.cuda_version = torch.version.cuda or ""
            profile.hardware_type = "desktop_gpu"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            profile.gpu_available = True
            profile.has_mps = True
            profile.device = "mps"
            profile.gpu_name = "Apple Silicon GPU"
            # MPS doesn't report VRAM, estimate from unified memory
            profile.gpu_vram_gb = profile.ram_gb * 0.75
    except ImportError:
        pass

    # Set default hardware type if not set
    if profile.hardware_type == "unknown":
        if profile.gpu_available:
            profile.hardware_type = "desktop_gpu"
        elif profile.ram_gb >= 32:
            profile.hardware_type = "server"
        else:
            profile.hardware_type = "desktop_cpu"

    with _profile_lock:
        _cached_profile = profile
    return profile


def get_cached_profile() -> Optional[HardwareProfile]:
    """Get cached hardware profile if available."""
    with _profile_lock:
        return _cached_profile


def clear_cached_profile() -> None:
    """Clear cached hardware profile."""
    global _cached_profile
    with _profile_lock:
        _cached_profile = None


def recommend_model_size(profile: Optional[HardwareProfile] = None) -> str:
    """
    Recommend optimal model size based on hardware.

    Returns:
        Model size string: 'pi_zero', 'nano', 'tiny', 'small', 'medium', 'large'
    """
    if profile is None:
        profile = detect_hardware()

    # Raspberry Pi or very low memory
    if profile.is_raspberry_pi or profile.ram_gb < 2:
        return "pi_zero"

    # Low memory systems
    if profile.ram_gb < 4:
        return "nano"

    # No GPU - use smaller models
    if not profile.gpu_available:
        if profile.ram_gb < 8:
            return "tiny"
        elif profile.ram_gb < 16:
            return "small"
        else:
            return "medium"

    # GPU available - use larger models
    if profile.gpu_vram_gb >= 24:
        return "large"
    elif profile.gpu_vram_gb >= 12:
        return "medium"
    elif profile.gpu_vram_gb >= 6:
        return "small"
    elif profile.gpu_vram_gb >= 4:
        return "tiny"
    else:
        return "nano"


def get_optimal_config(profile: Optional[HardwareProfile] = None) -> dict[str, Any]:
    """
    Get optimal configuration based on hardware.

    Returns:
        Dict with recommended settings: model_size, use_half, batch_size, etc.
    """
    if profile is None:
        profile = detect_hardware()

    model_size = recommend_model_size(profile)

    config = {
        "model_size": model_size,
        "device": profile.device,
        "use_half": profile.gpu_available and profile.device == "cuda",
        "precision": "auto",
        "batch_size": 1,
        "max_seq_len": 512,
    }

    # BF16 detection for Blackwell / Ampere+ GPUs
    if profile.gpu_available and profile.device == "cuda":
        try:
            import torch

            if torch.cuda.is_bf16_supported():
                config["precision"] = "bfloat16"
            else:
                config["precision"] = "float16"
        except Exception:
            config["precision"] = "float16"
    elif not profile.gpu_available:
        config["precision"] = "float32"

    # Adjust based on VRAM/RAM — scaled via InferenceMemoryBudget (S802)
    budget = InferenceMemoryBudget.from_profile(profile)
    config["batch_size"] = budget.inference_batch_size
    config["max_seq_len"] = budget.inference_max_seq_len

    return config


def estimate_memory_usage(
    model_size: str, batch_size: int = 1, seq_len: int = 512, use_half: bool = False
) -> dict[str, float]:
    """
    Estimate memory usage for a given configuration.

    Reads dim/n_layers from MODEL_PRESETS so estimates stay in
    sync with actual model definitions.

    Returns:
        Dict with estimated memory in GB: model_memory, kv_cache, total
    """
    # Pull real config from presets when available
    dim: int = 512
    n_layers: int = 12
    vocab_size: int = 32000
    try:
        from enigma_engine.core.model_presets import MODEL_PRESETS

        cfg = MODEL_PRESETS.get(model_size)
        if cfg is not None:
            dim = cfg.dim
            n_layers = cfg.n_layers
            vocab_size = cfg.vocab_size
    except ImportError:
        pass

    # Estimate parameter count from architecture:
    # embeddings + output + per-layer (QKV + out + FFN gate/up/down)
    ffn_dim = int(dim * 8 / 3)  # SwiGLU default
    per_layer = (
        3 * dim * dim  # Q, K, V projections
        + dim * dim  # output projection
        + 3 * dim * ffn_dim  # gate + up + down
    )
    params = 2 * vocab_size * dim + n_layers * per_layer

    bytes_per_param = 2 if use_half else 4

    # Model weight memory
    model_gb = (params * bytes_per_param) / (1024**3)

    # KV-cache estimate (rough)
    # For transformer: 2 * layers * 2 * hidden_dim * seq_len * batch_size
    kv_bytes = 2 * n_layers * 2 * dim * seq_len * batch_size * bytes_per_param
    kv_gb = kv_bytes / (1024**3)

    return {
        "model_memory": model_gb,
        "kv_cache": kv_gb,
        "total": model_gb + kv_gb,
    }


def recommend_training_batch_size(
    profile: Optional[HardwareProfile] = None,
) -> int:
    """Recommend a safe training batch size based on detected VRAM.

    Uses simple VRAM tiers rather than formula estimation.
    Conservative defaults to avoid OOM during training (training
    uses ~3x model memory due to gradients + optimizer states).

    Args:
        profile: Hardware profile.  Detected automatically if None.

    Returns:
        Recommended batch size (1, 2, 4, 8, or 16).
    """
    if profile is None:
        profile = detect_hardware()

    if not profile.gpu_available:
        # CPU-only: always use batch size 1
        if profile.ram_gb >= 32:
            return 2
        return 1

    vram = profile.gpu_vram_gb
    if vram >= 48:
        return 16
    if vram >= 24:
        return 8
    if vram >= 12:
        return 4
    if vram >= 6:
        return 2
    return 1


# Backward compatibility - get_hardware alias
def get_hardware() -> HardwareProfile:
    """Alias for detect_hardware()."""
    return detect_hardware()


# =====================================================================
# Training Memory Budget — adaptive constants from hardware profile
# =====================================================================


@dataclass
class TrainingMemoryBudget:
    """Compute training-related constants from available hardware.

    All values are derived from *ram_gb* (system RAM) and *vram_gb*
    (GPU VRAM).  Pass ``0`` for either to auto-detect at construction
    time.  Every constant that used to be a module-level literal is
    now a property of this class, scaled to the machine running the
    code — from a Raspberry Pi 5 (8 GB) to a workstation (128 GB+).

    Usage::

        budget = TrainingMemoryBudget()           # auto-detect
        budget = TrainingMemoryBudget(ram_gb=8)    # explicit Pi 5
        budget = TrainingMemoryBudget.from_profile(hw)  # from HardwareProfile
    """

    ram_gb: float = 0.0
    vram_gb: float = 0.0

    def __post_init__(self) -> None:
        if self.ram_gb <= 0:
            try:
                import psutil

                self.ram_gb = psutil.virtual_memory().total / (1024**3)
            except ImportError:
                self.ram_gb = 8.0
        if self.vram_gb <= 0:
            try:
                import torch

                if torch.cuda.is_available():
                    self.vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                else:
                    self.vram_gb = 0.0
            except ImportError:
                self.vram_gb = 0.0

    @classmethod
    def from_profile(cls, profile: HardwareProfile) -> "TrainingMemoryBudget":
        """Create a budget from an existing HardwareProfile."""
        vram = profile.gpu_vram_gb if profile.gpu_available else 0.0
        return cls(ram_gb=profile.ram_gb, vram_gb=vram)

    # -- RAM-scaled properties ----------------------------------------

    @property
    def streaming_threshold(self) -> int:
        """Sequence count above which training switches to disk-backed
        streaming.  More RAM → higher threshold → fewer disk pauses.

        Scale: ~3K sequences per GB of RAM (each seq ≈ few KB of text).
        Clamped to [5_000, 500_000].
        """
        return max(5_000, min(int(self.ram_gb * 3_000), 500_000))

    @property
    def streaming_window(self) -> int:
        """Sequences tokenized + packed per streaming window.
        Larger window = better shuffling + fewer disk reads.

        Scale: ~750 sequences per GB of RAM (packed tensors are bigger
        than raw text).  Clamped to [500, 100_000].
        """
        return max(500, min(int(self.ram_gb * 750), 100_000))

    @property
    def minhash_limit(self) -> int:
        """Max sequence count for MinHash near-duplicate detection.
        O(n²) pairwise — needs RAM for shingle sets.

        Scale: ~2.5K per GB.  Clamped to [5_000, 500_000].
        """
        return max(5_000, min(int(self.ram_gb * 2_500), 500_000))

    @property
    def curriculum_limit(self) -> int:
        """Max sequences to score for curriculum learning.
        Each needs one forward pass — bounded by VRAM+time.

        Uses VRAM if available, else RAM.  Clamped to [5_000, 500_000].
        """
        budget_gb = self.vram_gb if self.vram_gb > 0 else self.ram_gb
        return max(5_000, min(int(budget_gb * 5_000), 500_000))

    @property
    def tok_sample_cap(self) -> int:
        """Max characters collected for BPE tokenizer training.

        BPE quality plateaus around 1–2 GB of text for 32K vocab.
        Beyond that, unique word count explodes (6M+ words) and
        merge iterations become O(hours) with no quality gain.

        Scale: ~25 MB per GB of RAM.  Clamped to [100 MB, 2 GB].
        """
        cap = int(self.ram_gb * 25_000_000)
        return max(100_000_000, min(cap, 2_000_000_000))

    @property
    def dedup_capacity(self) -> int:
        """Max entries in paragraph-level dedup hash table.
        ~41 bytes per entry.

        Scale: use up to 5% of total RAM for the table.
        Clamped to [1_000_000, 200_000_000].
        """
        bytes_budget = self.ram_gb * (1024**3) * 0.05
        entries = int(bytes_budget / 41)
        return max(1_000_000, min(entries, 200_000_000))

    # -- VRAM-scaled properties ---------------------------------------

    @property
    def ce_chunk_size(self) -> int:
        """Cut cross-entropy vocab chunk size.  Higher = faster but
        uses more VRAM (each chunk materializes [chunk, vocab] logits).

        Scale: base 2048 + 512 per GB VRAM.  Clamped to [1024, 16384].
        """
        vram = self.vram_gb if self.vram_gb > 0 else self.ram_gb * 0.25
        return max(1024, min(int(2048 + vram * 512), 16384))

    @property
    def batch_size_cap(self) -> int:
        """Hard ceiling for auto batch-size estimation.

        Scale: 8 base + 4 per 8 GB VRAM.  Clamped to [8, 256].
        """
        vram = self.vram_gb if self.vram_gb > 0 else self.ram_gb * 0.25
        return max(8, min(int(8 + (vram / 8) * 4), 256))

    @property
    def cpu_batch_size(self) -> int:
        """Fallback batch size when no GPU is available.

        Scale: 1 per 8 GB RAM.  Clamped to [1, 16].
        """
        return max(1, min(int(self.ram_gb / 8), 16))

    @property
    def replay_capacity(self) -> int:
        """RL replay buffer capacity.  Stored on CPU.

        Scale: 32 per GB RAM.  Clamped to [64, 4096].
        """
        return max(64, min(int(self.ram_gb * 32), 4096))


@dataclass
class InferenceMemoryBudget:
    """Hardware-scaled constants for inference, caching, and data loading.

    Mirrors :class:`TrainingMemoryBudget` for the inference side.
    Pass ``ram_gb=0`` / ``vram_gb=0`` (the defaults) to auto-detect.

    Covers S801–S807 sustainability items::

        budget = InferenceMemoryBudget()               # auto-detect
        budget = InferenceMemoryBudget(ram_gb=8)        # explicit Pi 5
        budget = InferenceMemoryBudget.from_profile(hw) # from HardwareProfile
    """

    ram_gb: float = 0.0
    vram_gb: float = 0.0

    def __post_init__(self) -> None:
        if self.ram_gb <= 0:
            try:
                import psutil

                self.ram_gb = psutil.virtual_memory().total / (1024**3)
            except ImportError:
                self.ram_gb = 8.0
        if self.vram_gb <= 0:
            try:
                import torch

                if torch.cuda.is_available():
                    self.vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                else:
                    self.vram_gb = 0.0
            except ImportError:
                self.vram_gb = 0.0

    @classmethod
    def from_profile(cls, profile: HardwareProfile) -> "InferenceMemoryBudget":
        """Create a budget from an existing HardwareProfile."""
        vram = profile.gpu_vram_gb if profile.gpu_available else 0.0
        return cls(ram_gb=profile.ram_gb, vram_gb=vram)

    # -- VRAM-scaled properties (S801) --------------------------------

    @property
    def gguf_context_length(self) -> int:
        """GGUF context window scaled to VRAM.

        More granular than the old 5-tier ladder.  A 48 GB GPU now gets
        64K context instead of the same 32K as 24 GB.
        """
        if self.vram_gb >= 48:
            return 65536
        if self.vram_gb >= 24:
            return 32768
        if self.vram_gb >= 16:
            return 16384
        if self.vram_gb >= 12:
            return 12288
        if self.vram_gb >= 8:
            return 8192
        if self.vram_gb >= 4:
            return 4096
        if self.vram_gb >= 2:
            return 2048
        return 2048

    @property
    def gguf_gpu_layers(self) -> int:
        """GGUF GPU layer offload hint.  -1 = all layers."""
        if self.vram_gb >= 8:
            return -1  # full offload
        if self.vram_gb >= 4:
            return 20
        if self.vram_gb >= 2:
            return 10
        return 0  # CPU only

    # -- VRAM-scaled properties (S802) --------------------------------

    @property
    def inference_batch_size(self) -> int:
        """Inference batch size scaled to VRAM/RAM.

        A 32 GB GPU now gets batch=8 instead of the same 4 as 8 GB.
        """
        if self.vram_gb >= 24:
            return 8
        if self.vram_gb >= 16:
            return 6
        if self.vram_gb >= 8:
            return 4
        if self.vram_gb >= 4:
            return 2
        if self.ram_gb >= 16:
            return 2
        return 1

    @property
    def inference_max_seq_len(self) -> int:
        """Inference max sequence length scaled to VRAM/RAM."""
        if self.vram_gb >= 24:
            return 2048
        if self.vram_gb >= 16:
            return 1536
        if self.vram_gb >= 8:
            return 1024
        if self.vram_gb >= 4:
            return 512
        if self.ram_gb >= 16:
            return 512
        return 256

    # -- RAM-scaled properties (S803–S805) ----------------------------

    @property
    def token_count_cache_cap(self) -> int:
        """Token count cache capacity scaled to RAM (S803).

        ~100 bytes per entry.  Base: 4096 at 8 GB, scale linearly.
        Clamped to [1024, 65536].
        """
        cap = int(self.ram_gb * 512)
        return max(1024, min(cap, 65536))

    @property
    def bpe_cache_cap(self) -> int:
        """BPE tokenizer cache capacity scaled to RAM (S804).

        ~200 bytes per entry.  Base: 10000 at 8 GB, scale linearly.
        Clamped to [2000, 100000].
        """
        cap = int(self.ram_gb * 1250)
        return max(2000, min(cap, 100000))

    @property
    def advanced_tok_cache_cap(self) -> int:
        """Advanced tokenizer cache capacity scaled to RAM (S805).

        Same profile as BPE cache.
        Clamped to [2000, 100000].
        """
        cap = int(self.ram_gb * 1250)
        return max(2000, min(cap, 100000))

    # -- RAM-scaled properties (S806) ---------------------------------

    @property
    def dataset_chunk_chars(self) -> int:
        """Dataset chunk read size in chars, scaled to RAM (S806).

        ~2 bytes/char in memory.  Target ~5 %% of RAM for the chunk
        buffer.  Base: 200M at 8 GB, scale linearly.
        Clamped to [50_000_000, 500_000_000].
        """
        chars = int(self.ram_gb * 25_000_000)
        return max(50_000_000, min(chars, 500_000_000))

    @property
    def dataset_stream_threshold(self) -> int:
        """File size above which reading switches to chunked streaming.

        ~10 %% of RAM.  Base: 500M at 8 GB.
        Clamped to [100_000_000, 2_000_000_000].
        """
        threshold = int(self.ram_gb * 62_500_000)
        return max(100_000_000, min(threshold, 2_000_000_000))

    # -- VRAM-scaled properties (S807) --------------------------------

    @property
    def api_max_tokens(self) -> int:
        """API max_tokens upper bound scaled to VRAM.

        Larger VRAM supports longer generation without OOM risk.
        """
        if self.vram_gb >= 24:
            return 16384
        if self.vram_gb >= 16:
            return 8192
        if self.vram_gb >= 8:
            return 4096
        return 2048


__all__ = [
    "HardwareProfile",
    "InferenceMemoryBudget",
    "TrainingMemoryBudget",
    "clear_cached_profile",
    "detect_hardware",
    "estimate_memory_usage",
    "get_cached_profile",
    "get_hardware",
    "get_optimal_config",
    "recommend_model_size",
    "recommend_training_batch_size",
]
