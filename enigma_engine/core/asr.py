"""Speech-to-text primitive -- the ears organ.

Counterpart of core/tts.py: capabilities live as services the model and its
clients reach through the hub, not as weights in the 182M decoder. The native
audio path (core/audio_encoder.py + audio_projection) stays the later in-model
road; this organ gives her working ears TODAY via /v1/audio/transcriptions.

Backend is faster-whisper (CTranslate2 ships wheels, so no C++ toolchain --
the repo rule). The import happens lazily inside Ears so this module always
imports cleanly; a missing backend fails honestly as ASRError. Weights
download from HuggingFace on first construction (base ~145 MB).
"""

from __future__ import annotations

import threading
from pathlib import Path

# base is the accuracy/size sweet spot for command-and-chat audio; callers can
# pass small/medium when transcription quality starts to matter more than VRAM.
DEFAULT_MODEL_SIZE = "base"


class ASRError(Exception):
    """The ASR backend is unavailable or a transcription failed."""


class AudioDecodeError(ASRError):
    """The AUDIO could not be decoded -- the fault is in the bytes handed in,
    not in the organ.

    Only the raise site knows which of the two a failure is, and the seam that
    took the bytes off a client cannot tell them apart from the message. A
    junk upload is a client error (400); a whisper model that will not load,
    or a file the server itself failed to write, is the server's (500)."""


class Ears:
    """One whisper model, one transcription at a time (lock serializes)."""

    def __init__(self, model_size: str = DEFAULT_MODEL_SIZE, device: str = "auto", model_factory=None):
        self._lock = threading.Lock()
        self.model_size = model_size
        self.device = device
        if model_factory is not None:  # tests inject a fake
            self._model = model_factory()
            return
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise ASRError("faster-whisper not installed -- pip install 'enigma-engine[ears]'") from exc

        # cuda/float16 first when present, cpu/int8 as the honest fallback --
        # CTranslate2 occasionally trails torch on brand-new GPU architectures,
        # and ears that fall back beat ears that crash the server.
        candidates = []
        if device in ("auto", "cuda"):
            try:
                import torch

                if torch.cuda.is_available():
                    candidates.append(("cuda", "float16"))
            except ImportError:
                pass
        if device in ("auto", "cpu") or not candidates:
            candidates.append(("cpu", "int8"))

        last: Exception | None = None
        for dev, compute in candidates:
            try:
                self._model = WhisperModel(model_size, device=dev, compute_type=compute)
                if dev != candidates[0][0]:
                    print(f"  WARN: ears fell back to {dev}/{compute}: {last}", flush=True)
                self.device = dev
                return
            except Exception as exc:
                last = exc
        raise ASRError(f"could not load whisper '{model_size}': {last}") from last

    def transcribe(self, audio_path: str | Path) -> dict:
        """Return {text, language, duration} for an audio file."""
        path = Path(audio_path)
        if not path.exists():
            raise ASRError(f"audio file not found: {path}")
        with self._lock:
            try:
                segments, info = self._model.transcribe(str(path))
                text = " ".join(s.text.strip() for s in segments).strip()
            except Exception as exc:
                # Whatever whisper choked on, it choked on THIS audio: a
                # decode error, an empty file, a container it cannot read.
                raise AudioDecodeError(f"transcription failed: {exc}") from exc
        return {
            "text": text,
            "language": getattr(info, "language", None),
            "duration": round(float(getattr(info, "duration", 0.0)), 2),
        }
