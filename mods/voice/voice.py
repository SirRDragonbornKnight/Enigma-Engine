"""Voice service - unified STT + TTS surface.

2.1-voice slice (May 26 2026): replaced the prior cloud STT path with
faster-whisper (mirrors mods/transcriber/main.py), and dropped pyttsx3
in favour of Kokoro-82M for TTS (mirrors mods/audiogen/audiogen.py
LocalTTS). All processing is now 100% local - no cloud calls anywhere
in this module.

    STT -> faster-whisper  (mirror mods/transcriber/main.py)
    TTS -> Kokoro-82M      (mirror mods/audiogen/audiogen.py LocalTTS)
    Fallback TTS -> OS-native SAPI / say / espeak (SystemTTS, opt-in)

Voice's unique value over transcriber + audiogen is composition: one
socket that listens, transcribes, AND speaks - useful for hands-free
flows where the GUI talks to a single mod instead of two.

Self-contained socket service (not a ModClient subclass) so it can run
standalone or attach to the router via raw JSON messages on port 9907.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, List, Dict

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("voice")

OUTPUT_DIR = Path("outputs/voice")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Wire protocol (raw socket, matches sibling raw-socket mods)
# ---------------------------------------------------------------------------
class MessageType(str, Enum):
    COMMAND = "command"
    RESPONSE = "response"
    EVENT = "event"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    REGISTER = "register"


@dataclass
class Message:
    type: MessageType
    payload: Dict[str, Any] = field(default_factory=dict)
    id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)

    def to_bytes(self) -> bytes:
        data = {
            "type": self.type.value,
            "payload": self.payload,
            "id": self.id,
            "timestamp": self.timestamp,
        }
        return json.dumps(data).encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> "Message":
        obj = json.loads(data.decode("utf-8"))
        return cls(
            type=MessageType(obj["type"]),
            payload=obj.get("payload", {}),
            id=obj.get("id"),
            timestamp=obj.get("timestamp", time.time()),
        )


# ===========================================================================
# STT - faster-whisper (CTranslate2 backend, local, MIT licence)
# ===========================================================================
class WhisperSTT:
    """Local speech-to-text via faster-whisper.

    Mirrors mods/transcriber/main.py: lazy imports, device auto-select,
    INT8 on CPU / FP16 on CUDA, segments-join transcribe. No cloud calls.
    """

    def __init__(
        self,
        model_name: str = "base",
        device: str = "auto",
        sample_rate: int = 16000,
        mic_chunk_sec: float = 5.0,
    ) -> None:
        self.model_name = model_name
        self.device_pref = device
        self.sample_rate = int(sample_rate)
        self.mic_chunk_sec = float(mic_chunk_sec)
        self._model: Any = None
        self._WhisperModel: Any = None
        self._sd: Any = None
        self._np: Any = None

    # -- lazy imports -------------------------------------------------------
    def _ensure_imports(self) -> bool:
        if self._WhisperModel is not None:
            return True
        try:
            from faster_whisper import WhisperModel

            self._WhisperModel = WhisperModel
            return True
        except Exception as e:
            logger.error(f"faster-whisper not available: {e}")
            return False

    def _ensure_mic_imports(self) -> bool:
        if self._sd is not None and self._np is not None:
            return True
        try:
            import sounddevice as sd
            import numpy as np

            self._sd = sd
            self._np = np
            return True
        except Exception as e:
            logger.error(f"sounddevice/numpy not available for microphone: {e}")
            return False

    def _resolve_device(self) -> tuple[str, str]:
        """Return (device, compute_type) tuple for WhisperModel."""
        pref = (self.device_pref or "auto").lower()
        if pref == "cpu":
            return "cpu", "int8"
        if pref in ("cuda", "gpu"):
            return "cuda", "float16"
        # auto
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda", "float16"
        except Exception:
            pass
        return "cpu", "int8"

    # -- public API ---------------------------------------------------------
    def load(self) -> bool:
        if not self._ensure_imports():
            return False
        try:
            device, compute_type = self._resolve_device()
            logger.info(f"Loading faster-whisper model='{self.model_name}' device={device} compute={compute_type}")
            self._model = self._WhisperModel(self.model_name, device=device, compute_type=compute_type)
            return True
        except Exception as e:
            logger.error(f"faster-whisper load failed: {e}")
            return False

    def transcribe_file(self, path: str, language: Optional[str] = None) -> str:
        if self._model is None and not self.load():
            return ""
        try:
            segments, _info = self._model.transcribe(str(path), language=language)
            return "".join(seg.text for seg in segments).strip()
        except Exception as e:
            logger.error(f"transcribe_file failed: {e}")
            return ""

    def listen_microphone(self, duration: float = 5.0, language: Optional[str] = None) -> str:
        """Record `duration` seconds from default mic, transcribe locally."""
        if not self._ensure_mic_imports():
            return ""
        if self._model is None and not self.load():
            return ""
        try:
            seconds = max(0.5, float(duration))
            logger.info(f"Listening for {seconds:.1f}s @ {self.sample_rate} Hz")
            audio = self._sd.rec(
                int(seconds * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
            )
            self._sd.wait()
            audio = self._np.squeeze(audio)
            segments, _info = self._model.transcribe(audio, language=language)
            return "".join(seg.text for seg in segments).strip()
        except Exception as e:
            logger.error(f"listen_microphone failed: {e}")
            return ""

    def unload(self) -> None:
        self._model = None


# ===========================================================================
# TTS - Kokoro-82M (local, Apache 2.0)
# ===========================================================================
DEFAULT_KOKORO_VOICES = (
    "af_heart",
    "af_bella",
    "af_sarah",
    "am_adam",
    "am_michael",
    "bf_emma",
    "bf_isabella",
    "bm_george",
    "bm_lewis",
)


class LocalTTS:
    """Local TTS via Kokoro-82M. Mirrors mods/audiogen/audiogen.py LocalTTS."""

    def __init__(
        self,
        voice: str = "af_heart",
        lang_code: str = "a",
        speed: float = 1.0,
    ) -> None:
        self.voice = voice
        self.lang_code = lang_code
        self.speed = float(speed)
        self.volume = 1.0  # accepted for API parity; applied at playback
        self._pipeline: Any = None
        self._sf: Any = None
        self._sd: Any = None

    def _ensure_imports(self) -> bool:
        if self._pipeline is not None:
            return True
        try:
            from kokoro import KPipeline

            self._pipeline = KPipeline(lang_code=self.lang_code)
            return True
        except Exception as e:
            logger.error(f"kokoro not available: {e}")
            return False

    def _ensure_io(self) -> bool:
        try:
            if self._sf is None:
                import soundfile as sf

                self._sf = sf
            if self._sd is None:
                import sounddevice as sd

                self._sd = sd
            return True
        except Exception as e:
            logger.error(f"soundfile/sounddevice not available: {e}")
            return False

    def load(self) -> bool:
        return self._ensure_imports()

    def speak(self, text: str) -> bool:
        if not self._ensure_imports() or not self._ensure_io():
            return False
        try:
            generator = self._pipeline(text, voice=self.voice, speed=self.speed)
            for _i, _ps, audio in generator:
                # audio is a 1-D numpy / torch tensor at 24 kHz
                try:
                    arr = audio.numpy() if hasattr(audio, "numpy") else audio
                except Exception:
                    arr = audio
                if self.volume != 1.0:
                    try:
                        arr = arr * float(self.volume)
                    except Exception:
                        pass
                self._sd.play(arr, 24000)
                self._sd.wait()
            return True
        except Exception as e:
            logger.error(f"LocalTTS.speak failed: {e}")
            return False

    def generate_to_file(self, text: str, out_path: Optional[Path] = None) -> Optional[Path]:
        if not self._ensure_imports() or not self._ensure_io():
            return None
        try:
            if out_path is None:
                ts = int(time.time() * 1000)
                out_path = OUTPUT_DIR / f"speech_{ts}.wav"
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            chunks: List[Any] = []
            generator = self._pipeline(text, voice=self.voice, speed=self.speed)
            for _i, _ps, audio in generator:
                arr = audio.numpy() if hasattr(audio, "numpy") else audio
                chunks.append(arr)
            if not chunks:
                return None
            import numpy as np

            data = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
            self._sf.write(str(out_path), data, 24000)
            return out_path
        except Exception as e:
            logger.error(f"LocalTTS.generate_to_file failed: {e}")
            return None

    def set_voice(self, voice: str) -> bool:
        self.voice = voice
        return True

    def set_rate(self, rate: float) -> bool:
        # Kokoro 'speed' approximates rate; clamp to a sane band.
        try:
            self.speed = max(0.5, min(2.0, float(rate)))
            return True
        except Exception:
            return False

    def set_volume(self, volume: float) -> bool:
        try:
            self.volume = max(0.0, min(1.0, float(volume)))
            return True
        except Exception:
            return False

    def get_voices(self) -> List[str]:
        return list(DEFAULT_KOKORO_VOICES)

    def unload(self) -> None:
        self._pipeline = None


# ===========================================================================
# TTS - OS-native fallback (no cloud, no pyttsx3)
# ===========================================================================
class SystemTTS:
    """OS-native speech synthesis. Windows: PowerShell SAPI. macOS: `say`.

    Opt-in fallback when Kokoro can't load. NOT a silent default - the
    pipeline names the failure if `local` (Kokoro) drops out.
    """

    def __init__(self) -> None:
        self.platform = sys.platform
        self._loaded = False

    def load(self) -> bool:
        self._loaded = True
        return True

    def speak(self, text: str) -> bool:
        try:
            text_safe = (text or "").replace('"', "'")
            if self.platform == "win32":
                ps = (
                    "Add-Type -AssemblyName System.Speech; "
                    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                    f'$s.Speak("{text_safe}")'
                )
                os.system(f'powershell -Command "{ps}"')
            elif self.platform == "darwin":
                os.system(f'say "{text_safe}"')
            else:
                # Linux: try espeak / spd-say if installed; else log.
                code = os.system(f'espeak "{text_safe}" 2>/dev/null')
                if code != 0:
                    logger.warning("SystemTTS: no espeak on Linux; install espeak-ng or use Kokoro provider")
                    return False
            return True
        except Exception as e:
            logger.error(f"SystemTTS.speak failed: {e}")
            return False

    def generate_to_file(self, text: str, out_path: Optional[Path] = None) -> Optional[Path]:
        # System voices vary by OS - file capture is platform-specific and
        # not the canonical path. Callers wanting a file should use LocalTTS.
        logger.warning("SystemTTS.generate_to_file not supported; use LocalTTS")
        return None

    def set_voice(self, voice: str) -> bool:
        return False  # OS-native voice selection not exposed

    def set_rate(self, rate: float) -> bool:
        return False

    def set_volume(self, volume: float) -> bool:
        return False

    def get_voices(self) -> List[str]:
        return ["system_default"]

    def unload(self) -> None:
        self._loaded = False


# ===========================================================================
# Pipeline orchestrator - owns one STT + one TTS at a time
# ===========================================================================
class VoicePipeline:
    """Composes a single STT + TTS pair behind a stable surface."""

    def __init__(
        self,
        stt_provider: str = "whisper",
        tts_provider: str = "local",
        whisper_model: str = "base",
        whisper_device: str = "auto",
        kokoro_voice: str = "af_heart",
        kokoro_lang_code: str = "a",
    ) -> None:
        self.stt_provider = stt_provider
        self.tts_provider = tts_provider
        self.whisper_model = whisper_model
        self.whisper_device = whisper_device
        self.kokoro_voice = kokoro_voice
        self.kokoro_lang_code = kokoro_lang_code

        self.stt: Optional[Any] = None
        self.tts: Optional[Any] = None
        self._loaded = False

    # -- factories ---------------------------------------------------------
    def _make_stt(self, name: str) -> Optional[Any]:
        if name == "whisper":
            return WhisperSTT(model_name=self.whisper_model, device=self.whisper_device)
        return None  # no cloud STT supported

    def _make_tts(self, name: str) -> Optional[Any]:
        if name == "local":
            return LocalTTS(voice=self.kokoro_voice, lang_code=self.kokoro_lang_code)
        if name == "system":
            return SystemTTS()
        return None

    # -- lifecycle ---------------------------------------------------------
    def load(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "success": True,
            "stt": {"provider": self.stt_provider, "loaded": False},
            "tts": {"provider": self.tts_provider, "loaded": False},
            "errors": [],
        }

        self.stt = self._make_stt(self.stt_provider)
        if self.stt is None:
            result["success"] = False
            result["errors"].append(
                f"unknown stt provider '{self.stt_provider}'; supported: ['whisper'] (faster-whisper)"
            )
        elif not self.stt.load():
            result["success"] = False
            result["errors"].append(
                f"stt provider '{self.stt_provider}' load failed - "
                "is faster-whisper installed? (pip install faster-whisper)"
            )
        else:
            result["stt"]["loaded"] = True

        self.tts = self._make_tts(self.tts_provider)
        if self.tts is None:
            result["success"] = False
            result["errors"].append(
                f"unknown tts provider '{self.tts_provider}'; supported: ['local' (kokoro), 'system']"
            )
        elif not self.tts.load():
            result["success"] = False
            err = (
                "kokoro not installed (pip install kokoro)"
                if self.tts_provider == "local"
                else "OS-native TTS unavailable"
            )
            result["errors"].append(f"tts provider '{self.tts_provider}' load failed - {err}")
        else:
            result["tts"]["loaded"] = True

        self._loaded = result["success"]
        return result

    def unload(self) -> None:
        if self.stt is not None:
            try:
                self.stt.unload()
            except Exception:
                pass
        if self.tts is not None:
            try:
                self.tts.unload()
            except Exception:
                pass
        self.stt = None
        self.tts = None
        self._loaded = False

    # -- introspection -----------------------------------------------------
    def list_stt_providers(self) -> List[str]:
        return ["whisper"]

    def list_tts_providers(self) -> List[str]:
        return ["local", "system"]


# ===========================================================================
# Voice service - command dispatch + socket server
# ===========================================================================
class Voice:
    """Unified STT+TTS service. Dispatches commands over a raw JSON socket."""

    def __init__(
        self,
        stt_provider: str = "whisper",
        tts_provider: str = "local",
        whisper_model: str = "base",
        whisper_device: str = "auto",
        kokoro_voice: str = "af_heart",
        kokoro_lang_code: str = "a",
    ) -> None:
        self.pipeline = VoicePipeline(
            stt_provider=stt_provider,
            tts_provider=tts_provider,
            whisper_model=whisper_model,
            whisper_device=whisper_device,
            kokoro_voice=kokoro_voice,
            kokoro_lang_code=kokoro_lang_code,
        )
        self._running = False
        self._continuous_listen = False
        self._continuous_thread: Optional[threading.Thread] = None
        self._last_transcript: str = ""

        self._commands = {
            "listen": self._cmd_listen,
            "speak": self._cmd_speak,
            "transcribe": self._cmd_transcribe,
            "generate_audio": self._cmd_generate_audio,
            "generate": self._cmd_generate_audio,  # alias for sibling parity
            "start_continuous": self._cmd_start_continuous,
            "stop_continuous": self._cmd_stop_continuous,
            "set_voice": self._cmd_set_voice,
            "set_rate": self._cmd_set_rate,
            "set_volume": self._cmd_set_volume,
            "list_voices": self._cmd_list_voices,
            "load_provider": self._cmd_load_provider,
            "unload_provider": self._cmd_unload_provider,
            "list_providers": self._cmd_list_providers,
            "set_default": self._cmd_set_default,
            "load": self._cmd_load,
            "status": self._cmd_status,
        }

    # -- dispatch ----------------------------------------------------------
    def handle_command(self, command: str, params: Dict[str, Any]) -> Dict[str, Any]:
        fn = self._commands.get(command)
        if fn is None:
            return {
                "success": False,
                "error": f"unknown command '{command}'",
                "available": list(self._commands.keys()),
            }
        try:
            return fn(params or {})
        except Exception as e:
            logger.exception(f"command '{command}' raised")
            return {"success": False, "error": str(e)}

    # -- lifecycle commands ------------------------------------------------
    def _cmd_load(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return self.pipeline.load()

    def _cmd_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "success": True,
            "service": "voice",
            "stt_provider": self.pipeline.stt_provider,
            "tts_provider": self.pipeline.tts_provider,
            "stt_loaded": self.pipeline.stt is not None,
            "tts_loaded": self.pipeline.tts is not None,
            "continuous_listen": self._continuous_listen,
            "last_transcript": self._last_transcript,
        }

    def _ensure_loaded(self) -> Optional[Dict[str, Any]]:
        if self.pipeline.stt is None or self.pipeline.tts is None:
            result = self.pipeline.load()
            if not result.get("success"):
                return {
                    "success": False,
                    "error": "pipeline load failed",
                    "details": result,
                }
        return None

    # -- STT commands ------------------------------------------------------
    def _cmd_listen(self, params: Dict[str, Any]) -> Dict[str, Any]:
        err = self._ensure_loaded()
        if err is not None:
            return err
        duration = float(params.get("duration", 5.0))
        language = params.get("language")
        text = self.pipeline.stt.listen_microphone(  # type: ignore[union-attr]
            duration=duration, language=language
        )
        self._last_transcript = text
        return {"success": bool(text), "text": text, "duration": duration}

    def _cmd_transcribe(self, params: Dict[str, Any]) -> Dict[str, Any]:
        err = self._ensure_loaded()
        if err is not None:
            return err
        audio_path = params.get("audio_path") or params.get("file_path")
        if not audio_path:
            return {"success": False, "error": "audio_path required"}
        if not Path(audio_path).exists():
            return {
                "success": False,
                "error": f"audio file not found: {audio_path}",
            }
        language = params.get("language")
        text = self.pipeline.stt.transcribe_file(  # type: ignore[union-attr]
            audio_path, language=language
        )
        self._last_transcript = text
        return {"success": bool(text), "text": text, "audio_path": audio_path}

    # -- TTS commands ------------------------------------------------------
    def _cmd_speak(self, params: Dict[str, Any]) -> Dict[str, Any]:
        err = self._ensure_loaded()
        if err is not None:
            return err
        text = params.get("text")
        if not text:
            return {"success": False, "error": "text required"}
        ok = self.pipeline.tts.speak(text)  # type: ignore[union-attr]
        return {"success": bool(ok), "text": text}

    def _cmd_generate_audio(self, params: Dict[str, Any]) -> Dict[str, Any]:
        err = self._ensure_loaded()
        if err is not None:
            return err
        text = params.get("text")
        if not text:
            return {"success": False, "error": "text required"}
        out_param = params.get("out_path") or params.get("output_path")
        out_path = Path(out_param) if out_param else None
        result = self.pipeline.tts.generate_to_file(text, out_path=out_path)  # type: ignore[union-attr]
        if result is None:
            return {
                "success": False,
                "error": (
                    "generate_to_file failed - 'system' provider does not "
                    "support file output; switch to 'local' (kokoro)"
                ),
            }
        return {"success": True, "path": str(result), "text": text}

    # -- continuous listen --------------------------------------------------
    def _cmd_start_continuous(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if self._continuous_listen:
            return {"success": False, "error": "continuous listen already active"}
        err = self._ensure_loaded()
        if err is not None:
            return err
        chunk_sec = float(params.get("chunk_sec", 5.0))
        language = params.get("language")
        self._continuous_listen = True
        self._continuous_thread = threading.Thread(
            target=self._continuous_loop,
            args=(chunk_sec, language),
            daemon=True,
        )
        self._continuous_thread.start()
        return {"success": True, "chunk_sec": chunk_sec}

    def _cmd_stop_continuous(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self._continuous_listen = False
        if self._continuous_thread is not None:
            self._continuous_thread.join(timeout=2.0)
            self._continuous_thread = None
        return {"success": True}

    def _continuous_loop(self, chunk_sec: float, language: Optional[str]) -> None:
        while self._continuous_listen:
            try:
                text = self.pipeline.stt.listen_microphone(  # type: ignore[union-attr]
                    duration=chunk_sec, language=language
                )
                if text:
                    self._last_transcript = text
                    logger.info(f"[continuous] {text}")
            except Exception as e:
                logger.error(f"continuous loop error: {e}")
                time.sleep(0.5)

    # -- voice/rate/volume --------------------------------------------------
    def _cmd_set_voice(self, params: Dict[str, Any]) -> Dict[str, Any]:
        err = self._ensure_loaded()
        if err is not None:
            return err
        voice = params.get("voice")
        if not voice:
            return {"success": False, "error": "voice required"}
        ok = self.pipeline.tts.set_voice(voice)  # type: ignore[union-attr]
        return {"success": bool(ok), "voice": voice}

    def _cmd_set_rate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        err = self._ensure_loaded()
        if err is not None:
            return err
        rate = params.get("rate")
        if rate is None:
            return {"success": False, "error": "rate required"}
        ok = self.pipeline.tts.set_rate(float(rate))  # type: ignore[union-attr]
        return {"success": bool(ok), "rate": rate}

    def _cmd_set_volume(self, params: Dict[str, Any]) -> Dict[str, Any]:
        err = self._ensure_loaded()
        if err is not None:
            return err
        volume = params.get("volume")
        if volume is None:
            return {"success": False, "error": "volume required"}
        ok = self.pipeline.tts.set_volume(float(volume))  # type: ignore[union-attr]
        return {"success": bool(ok), "volume": volume}

    def _cmd_list_voices(self, params: Dict[str, Any]) -> Dict[str, Any]:
        err = self._ensure_loaded()
        if err is not None:
            return err
        voices = self.pipeline.tts.get_voices()  # type: ignore[union-attr]
        return {"success": True, "voices": voices}

    # -- provider management -----------------------------------------------
    def _cmd_load_provider(self, params: Dict[str, Any]) -> Dict[str, Any]:
        kind = params.get("kind")  # 'stt' or 'tts'
        name = params.get("name")
        if kind not in ("stt", "tts"):
            return {"success": False, "error": "kind must be 'stt' or 'tts'"}
        if not name:
            return {"success": False, "error": "name required"}

        if kind == "stt":
            new_stt = self.pipeline._make_stt(name)
            if new_stt is None:
                return {
                    "success": False,
                    "error": f"unknown stt provider '{name}'",
                    "available": self.pipeline.list_stt_providers(),
                }
            if not new_stt.load():
                return {
                    "success": False,
                    "error": f"stt provider '{name}' load failed",
                }
            if self.pipeline.stt is not None:
                self.pipeline.stt.unload()
            self.pipeline.stt = new_stt
            self.pipeline.stt_provider = name
            return {"success": True, "kind": "stt", "name": name}

        # tts
        new_tts = self.pipeline._make_tts(name)
        if new_tts is None:
            return {
                "success": False,
                "error": f"unknown tts provider '{name}'",
                "available": self.pipeline.list_tts_providers(),
            }
        if not new_tts.load():
            return {"success": False, "error": f"tts provider '{name}' load failed"}
        if self.pipeline.tts is not None:
            self.pipeline.tts.unload()
        self.pipeline.tts = new_tts
        self.pipeline.tts_provider = name
        return {"success": True, "kind": "tts", "name": name}

    def _cmd_unload_provider(self, params: Dict[str, Any]) -> Dict[str, Any]:
        kind = params.get("kind")
        if kind == "stt" and self.pipeline.stt is not None:
            self.pipeline.stt.unload()
            self.pipeline.stt = None
            return {"success": True, "kind": "stt"}
        if kind == "tts" and self.pipeline.tts is not None:
            self.pipeline.tts.unload()
            self.pipeline.tts = None
            return {"success": True, "kind": "tts"}
        return {"success": False, "error": "kind must be 'stt' or 'tts'"}

    def _cmd_list_providers(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "success": True,
            "stt": self.pipeline.list_stt_providers(),
            "tts": self.pipeline.list_tts_providers(),
            "active_stt": self.pipeline.stt_provider,
            "active_tts": self.pipeline.tts_provider,
        }

    def _cmd_set_default(self, params: Dict[str, Any]) -> Dict[str, Any]:
        kind = params.get("kind")
        name = params.get("name")
        if kind == "stt":
            valid = ("whisper",)
            if name not in valid:
                return {
                    "success": False,
                    "error": f"stt default must be one of {valid}",
                }
            self.pipeline.stt_provider = name
            return {"success": True, "kind": "stt", "default": name}
        if kind == "tts":
            valid = ("local", "system")
            if name not in valid:
                return {
                    "success": False,
                    "error": f"tts default must be one of {valid}",
                }
            self.pipeline.tts_provider = name
            return {"success": True, "kind": "tts", "default": name}
        return {"success": False, "error": "kind must be 'stt' or 'tts'"}

    # -- networking --------------------------------------------------------
    def connect_to_router(self, host: str, port: int = 9900) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        logger.info(f"Connected to router at {host}:{port}")

        register = Message(
            type=MessageType.REGISTER,
            payload={
                "service": "voice",
                "capabilities": [
                    "stt",
                    "tts",
                    "listen",
                    "speak",
                    "transcribe",
                    "audio",
                    "generate_audio",
                ],
                "commands": list(self._commands.keys()),
            },
        )
        data = register.to_bytes()
        sock.sendall(struct.pack(">I", len(data)) + data)

        self._running = True
        while self._running:
            try:
                len_data = sock.recv(4)
                if not len_data:
                    break
                msg_len = struct.unpack(">I", len_data)[0]
                msg_data = b""
                while len(msg_data) < msg_len:
                    chunk = sock.recv(min(4096, msg_len - len(msg_data)))
                    if not chunk:
                        break
                    msg_data += chunk
                msg = Message.from_bytes(msg_data)
                if msg.type == MessageType.COMMAND:
                    result = self.handle_command(
                        msg.payload.get("command", ""),
                        msg.payload.get("params", {}),
                    )
                    resp = Message(type=MessageType.RESPONSE, payload=result, id=msg.id)
                    resp_data = resp.to_bytes()
                    sock.sendall(struct.pack(">I", len(resp_data)) + resp_data)
            except Exception as e:
                logger.error(f"Router connection error: {e}")
                break
        sock.close()

    def run_standalone(self, port: int = 9907) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))
        sock.listen(5)
        logger.info(f"Voice service listening on port {port}")

        self._running = True
        while self._running:
            try:
                client, _addr = sock.accept()
                threading.Thread(target=self._handle_client, args=(client,), daemon=True).start()
            except Exception as e:
                logger.error(f"Accept error: {e}")
                break
        sock.close()

    def _handle_client(self, client: socket.socket) -> None:
        try:
            while self._running:
                len_data = client.recv(4)
                if not len_data:
                    break
                msg_len = struct.unpack(">I", len_data)[0]
                msg_data = b""
                while len(msg_data) < msg_len:
                    chunk = client.recv(min(4096, msg_len - len(msg_data)))
                    if not chunk:
                        break
                    msg_data += chunk
                msg = Message.from_bytes(msg_data)
                if msg.type == MessageType.COMMAND:
                    result = self.handle_command(
                        msg.payload.get("command", ""),
                        msg.payload.get("params", {}),
                    )
                    resp = Message(type=MessageType.RESPONSE, payload=result, id=msg.id)
                    client.sendall(resp.to_bytes())
        except Exception as e:
            logger.error(f"Client error: {e}")
        finally:
            client.close()

    def shutdown(self) -> None:
        self._running = False
        self._continuous_listen = False
        self.pipeline.unload()


# ===========================================================================
# CLI entry-point
# ===========================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Voice Service")
    parser.add_argument("--port", type=int, default=9907)
    parser.add_argument("--router", type=str)
    parser.add_argument("--listen", action="store_true", help="Listen for speech")
    parser.add_argument("--speak", type=str, help="Speak text")
    parser.add_argument("--transcribe", type=str, help="Transcribe audio file")
    parser.add_argument(
        "--stt",
        type=str,
        default="whisper",
        choices=["whisper"],
        help="STT provider (faster-whisper, local-only)",
    )
    parser.add_argument(
        "--tts",
        type=str,
        default="local",
        choices=["local", "system"],
        help="TTS provider: 'local' (Kokoro) or 'system' (OS-native fallback)",
    )

    args = parser.parse_args()
    service = Voice(stt_provider=args.stt, tts_provider=args.tts)

    if args.listen:
        result = service.handle_command("listen", {})
        print(json.dumps(result, indent=2))
        return
    if args.speak:
        result = service.handle_command("speak", {"text": args.speak})
        print(json.dumps(result, indent=2))
        return
    if args.transcribe:
        result = service.handle_command("transcribe", {"audio_path": args.transcribe})
        print(json.dumps(result, indent=2))
        return

    try:
        if args.router:
            host, port = args.router.split(":")
            service.connect_to_router(host, int(port))
        else:
            service.run_standalone(args.port)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        service.shutdown()


if __name__ == "__main__":
    main()
