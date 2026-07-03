"""
Transcriber Mod — Local audio/video transcription via faster-whisper.

Engine: faster-whisper (CTranslate2 backend, MIT licence).
- Local-only. Zero cloud calls. Replaces the prior cloud-API exfiltration path
  (REALIGN 2.1-transcriber, May 25 2026) that sent raw audio to a third-party
  Speech Recognition Web API.
- File modes: any format ffmpeg can decode (WAV / MP3 / FLAC / OGG / MP4 ...).
- Mic mode: 16 kHz mono PCM captured via `sounddevice`, fed to the same model.
- Settings (`mod.json`):
    whisper_model:   model size or HF id (default "base" ~140 MB)
    whisper_device:  "auto" | "cuda" | "cpu" (default "auto")
    language:        ISO 639-1 code, "" / "auto" for auto-detect
    mic_sample_rate: capture rate Hz (default 16000)
    mic_chunk_sec:   seconds per inference chunk (default 5.0)
"""

import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict

from mod_base import ModClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TranscriberMod(ModClient):
    """Local audio/video transcription mod (faster-whisper)."""

    def __init__(self):
        super().__init__()
        self._listening = False
        self._listen_thread: threading.Thread | None = None
        self._model = None  # faster_whisper.WhisperModel

    def _ensure_imports(self) -> bool:
        """Lazy-load faster-whisper model. Returns False with a logged hint on failure."""
        if self._model is not None:
            return True
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            logger.error("faster-whisper not installed. pip install faster-whisper")
            return False

        settings = self.config.get("settings", {})
        model_name = settings.get("whisper_model", "base")
        device_pref = settings.get("whisper_device", "auto")

        device, compute_type = self._resolve_device(device_pref)

        try:
            self._model = WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type,
            )
            logger.info(
                "faster-whisper loaded: model=%s device=%s compute_type=%s",
                model_name,
                device,
                compute_type,
            )
            return True
        except Exception as exc:
            logger.error(
                "faster-whisper failed to load model %r on device %r: %s",
                model_name,
                device,
                exc,
            )
            self._model = None
            return False

    @staticmethod
    def _resolve_device(pref: str) -> tuple[str, str]:
        """Pick (device, compute_type) honouring user preference + CUDA availability."""
        pref = (pref or "auto").lower()
        if pref == "cpu":
            return "cpu", "int8"
        if pref == "cuda":
            return "cuda", "float16"
        # auto
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda", "float16"
        except ImportError:
            pass
        return "cpu", "int8"

    # =========================================================================
    # COMMANDS
    # =========================================================================

    def cmd_transcribe(self, args: Dict[str, Any]) -> str:
        """Transcribe an audio or video file to text."""
        if not self._ensure_imports():
            return "Error: faster-whisper not loaded (pip install faster-whisper)"

        file_path = args.get("file_path", "").strip()
        if not file_path:
            return "Error: No file path provided"

        path = Path(file_path)
        if not path.exists():
            return f"Error: File not found: {file_path}"

        language = self.config.get("settings", {}).get("language", "")
        lang = language if language and language.lower() != "auto" else None

        try:
            segments, _info = self._model.transcribe(str(path), language=lang)
            text = "".join(seg.text for seg in segments).strip()
            return f"Transcription:\n{text}" if text else "Transcription: (no speech detected)"
        except Exception as exc:
            return f"Error transcribing file: {exc}"

    def cmd_listen(self, args: Dict[str, Any]) -> str:
        """Start live microphone transcription."""
        if not self._ensure_imports():
            return "Error: faster-whisper not loaded (pip install faster-whisper)"

        if self._listening:
            return "Already listening"

        duration = float(args.get("duration", 0) or 0)
        self._listening = True
        self._listen_thread = threading.Thread(
            target=self._listen_loop,
            args=(duration,),
            daemon=True,
        )
        self._listen_thread.start()
        return "Live transcription started"

    def cmd_stop_listen(self, args: Dict[str, Any]) -> str:
        """Stop live transcription."""
        if not self._listening:
            return "Not currently listening"
        self._listening = False
        return "Live transcription stopped"

    def cmd_status(self, args: Dict[str, Any]) -> str:
        """Get transcriber status."""
        status = "listening" if self._listening else "idle"
        return f"Transcriber status: {status}"

    # =========================================================================
    # INTERNAL
    # =========================================================================

    def _listen_loop(self, duration: float) -> None:
        """Background loop for live mic transcription via sounddevice + faster-whisper."""
        try:
            import sounddevice as sd
            import numpy as np
        except ImportError:
            logger.error("Live transcription requires sounddevice + numpy. pip install sounddevice numpy")
            self._listening = False
            return

        settings = self.config.get("settings", {})
        sample_rate = int(settings.get("mic_sample_rate", 16000))
        chunk_sec = float(settings.get("mic_chunk_sec", 5.0))
        language = settings.get("language", "")
        lang = language if language and language.lower() != "auto" else None
        chunk_frames = int(sample_rate * chunk_sec)
        start = time.monotonic()

        while self._listening:
            if duration > 0 and (time.monotonic() - start) > duration:
                break
            try:
                recording = sd.rec(
                    chunk_frames,
                    samplerate=sample_rate,
                    channels=1,
                    dtype="float32",
                )
                sd.wait()
                if not self._listening:
                    break
                audio = np.squeeze(recording)
                segments, _info = self._model.transcribe(audio, language=lang)
                text = "".join(seg.text for seg in segments).strip()
                if text:
                    self.send_message(
                        {
                            "type": "output",
                            "text": f"[Transcribed] {text}",
                        }
                    )
            except Exception as exc:
                logger.debug("Listen error: %s", exc)
                continue

        self._listening = False


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    mod = TranscriberMod()
    if mod.connect():
        mod.register()
        mod.run()
