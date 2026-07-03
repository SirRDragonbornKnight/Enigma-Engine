#!/usr/bin/env python3
"""
Audio Generation - Standalone Text-to-Speech Service

Text-to-speech and audio generation using:
- Kokoro-82M (local, Apache 2.0, ~300MB) - current best open TTS for <1GB VRAM
- System speech (OS SAPI / say / espeak) as explicit opt-in fallback

Usage:
    python audiogen.py                              # Start service
    python audiogen.py --port 9904                 # Custom port
    python audiogen.py --speak "Hello world"      # Speak text
    python audiogen.py --generate "Hello" --save  # Save to file
"""

import argparse
import json
import logging
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs/audio")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Protocol
# =============================================================================


class MessageType(Enum):
    REGISTER = "register"
    COMMAND = "command"
    RESPONSE = "response"
    EVENT = "event"
    HEARTBEAT = "heartbeat"
    SHUTDOWN = "shutdown"


@dataclass
class Message:
    type: MessageType
    payload: Dict[str, Any] = field(default_factory=dict)
    id: str = ""

    def to_bytes(self) -> bytes:
        data = json.dumps({"type": self.type.value, "payload": self.payload, "id": self.id}).encode("utf-8")
        return struct.pack(">I", len(data)) + data

    @classmethod
    def from_bytes(cls, data: bytes) -> "Message":
        obj = json.loads(data.decode("utf-8"))
        return cls(type=MessageType(obj["type"]), payload=obj.get("payload", {}), id=obj.get("id", ""))


# =============================================================================
# Audio/TTS Providers
# =============================================================================


class LocalTTS:
    """Neural text-to-speech using Kokoro-82M (local, Apache 2.0, ~300MB).

    Engine pick (2.1-audiogen, May 25 2026): Kokoro-82M beats Bark / MeloTTS
    on the open TTS Arena leaderboard at a tiny VRAM footprint. Coqui XTTS-v2
    and F5-TTS are non-commercial licenses; Piper is CPU-only with weaker
    prosody. Kokoro wins on quality-per-VRAM AND license.
    """

    # Canonical Kokoro voices shipped with the model (af=american female,
    # am=american male, bf=british female, bm=british male).
    DEFAULT_VOICES = (
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

    def __init__(self, voice: str = "af_heart", lang_code: str = "a"):
        self.pipeline = None
        self.is_loaded = False
        self.voice = voice
        self.lang_code = lang_code
        # Kokoro speed multiplier; we expose pyttsx3-style 50..300 via set_rate
        # for caller compatibility but store as 0.5..2.0 internally.
        self.rate = 1.0
        self.volume = 1.0
        self.sample_rate = 24000

    def load(self) -> bool:
        try:
            from kokoro import KPipeline

            self.pipeline = KPipeline(lang_code=self.lang_code)
            self.is_loaded = True
            logger.info(f"Kokoro TTS loaded (voice={self.voice}, lang={self.lang_code})")
            return True
        except ImportError:
            logger.error("Kokoro not available: install 'kokoro>=0.3' + 'soundfile' + 'torch'")
        except Exception as e:
            logger.error(f"Kokoro load failed: {e}")
        return False

    def unload(self):
        self.pipeline = None
        self.is_loaded = False

    def get_voices(self) -> List[str]:
        return list(self.DEFAULT_VOICES)

    def set_voice(self, index_or_name):
        voices = self.get_voices()
        if isinstance(index_or_name, int) and 0 <= index_or_name < len(voices):
            self.voice = voices[index_or_name]
        elif isinstance(index_or_name, str) and index_or_name in voices:
            self.voice = index_or_name

    def set_rate(self, rate):
        # Accept pyttsx3-style words/min (~150 default) or kokoro speed (0.5..2.0).
        try:
            r = float(rate)
        except (TypeError, ValueError):
            return
        if r > 5:  # pyttsx3 range
            r = r / 150.0
        self.rate = max(0.5, min(2.0, r))

    def set_volume(self, volume: float):
        try:
            self.volume = max(0.0, min(1.0, float(volume)))
        except (TypeError, ValueError):
            pass

    def speak(self, text: str) -> Dict[str, Any]:
        """Generate audio then play via sounddevice (optional dep)."""
        result = self.generate(text)
        if not result.get("success"):
            return result
        try:
            import soundfile as sf
            import sounddevice as sd

            audio, sr = sf.read(result["path"])
            sd.play(audio, sr)
            sd.wait()
            return {"success": True, "path": result["path"]}
        except ImportError:
            return {
                "success": True,
                "path": result["path"],
                "note": "Generated to file; install 'sounddevice' for direct playback",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def generate(self, text: str, **kwargs) -> Dict[str, Any]:
        """Generate speech and save to a .wav file."""
        if not self.is_loaded or self.pipeline is None:
            return {"success": False, "error": "Local Kokoro provider not loaded"}
        try:
            import numpy as np
            import soundfile as sf
        except ImportError:
            return {
                "success": False,
                "error": "Install 'soundfile' + 'numpy' to write audio output",
            }
        try:
            start = time.time()
            timestamp = int(time.time())
            filepath = OUTPUT_DIR / f"speech_{timestamp}.wav"

            voice = kwargs.get("voice", self.voice)
            speed = float(kwargs.get("speed", self.rate))

            chunks = []
            for _gs, _ps, audio in self.pipeline(text, voice=voice, speed=speed):
                # Kokoro yields torch tensors or numpy arrays; coerce uniformly.
                arr = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
                chunks.append(arr)

            if not chunks:
                return {"success": False, "error": "Kokoro produced no audio"}

            combined = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
            if self.volume != 1.0:
                combined = combined * self.volume
            sf.write(str(filepath), combined, self.sample_rate)

            return {
                "success": True,
                "path": str(filepath),
                "duration": time.time() - start,
                "voice": voice,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


class SystemTTS:
    """System speech fallback using OS features."""

    def __init__(self):
        self.is_loaded = False
        self._platform = None

    def load(self) -> bool:
        import platform

        self._platform = platform.system()
        self.is_loaded = True
        logger.info(f"System TTS ready ({self._platform})")
        return True

    def unload(self):
        self.is_loaded = False

    def get_voices(self) -> List[str]:
        return ["System Default"]

    def speak(self, text: str) -> Dict[str, Any]:
        if not self.is_loaded:
            return {"success": False, "error": "Not loaded"}

        try:
            import subprocess

            if self._platform == "Windows":
                # Use Windows SAPI
                script = f'Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Speak("{text}")'
                subprocess.run(["powershell", "-Command", script], capture_output=True)
            elif self._platform == "Darwin":
                subprocess.run(["say", text])
            else:
                # Linux - try espeak
                subprocess.run(["espeak", text])

            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def generate(self, text: str, **kwargs) -> Dict[str, Any]:
        """Generate using system tools."""
        try:
            import subprocess

            start = time.time()
            timestamp = int(time.time())
            filepath = OUTPUT_DIR / f"system_{timestamp}.wav"

            if self._platform == "Windows":
                script = f'''
                Add-Type -AssemblyName System.Speech
                $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
                $synth.SetOutputToWaveFile("{filepath}")
                $synth.Speak("{text}")
                $synth.Dispose()
                '''
                subprocess.run(["powershell", "-Command", script], capture_output=True)
            elif self._platform == "Darwin":
                subprocess.run(["say", "-o", str(filepath), "--data-format=LEF32@22050", text])
            else:
                subprocess.run(["espeak", "-w", str(filepath), text])

            if filepath.exists():
                return {"success": True, "path": str(filepath), "duration": time.time() - start}
            return {"success": False, "error": "Failed to generate audio"}
        except Exception as e:
            return {"success": False, "error": str(e)}


# =============================================================================
# AudioGen Service
# =============================================================================


class AudioGen:
    """Audio Generation Service."""

    PROVIDERS = {
        "local": LocalTTS,
        "system": SystemTTS,
    }

    # 2.1-audiogen (May 25 2026): default flipped "system" -> "local" so the
    # out-of-box behaviour is the neural Kokoro engine, matching the
    # imagegen/threed/videogen pattern. "system" remains opt-in for callers
    # who want the zero-dependency OS speech path.
    def __init__(self, default_provider: str = "local"):
        self.providers: Dict[str, Any] = {}
        self.default_provider = default_provider
        self._running = False
        self._socket: Optional[socket.socket] = None

        self.commands = {
            "speak": self._cmd_speak,
            "generate": self._cmd_generate,
            "list_voices": self._cmd_list_voices,
            "set_voice": self._cmd_set_voice,
            "set_rate": self._cmd_set_rate,
            "set_volume": self._cmd_set_volume,
            "load_provider": self._cmd_load_provider,
            "unload_provider": self._cmd_unload_provider,
            "list_providers": self._cmd_list_providers,
            "set_default": self._cmd_set_default,
            "status": self._cmd_status,
        }

    def get_provider(self, name: str):
        if name not in self.providers:
            if name not in self.PROVIDERS:
                return None
            self.providers[name] = self.PROVIDERS[name]()
        return self.providers[name]

    def _cmd_speak(self, params: Dict[str, Any]) -> Dict[str, Any]:
        text = params.get("text", "")
        provider_name = params.get("provider", self.default_provider)

        provider = self.get_provider(provider_name)
        if not provider:
            return {"success": False, "error": f"Unknown provider: {provider_name}"}

        if not provider.is_loaded:
            if not provider.load():
                return {"success": False, "error": f"Failed to load {provider_name}"}

        return provider.speak(text)

    def _cmd_generate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        text = params.get("text", "")
        provider_name = params.get("provider", self.default_provider)

        provider = self.get_provider(provider_name)
        if not provider:
            return {"success": False, "error": f"Unknown provider: {provider_name}"}

        if not provider.is_loaded:
            if not provider.load():
                return {"success": False, "error": f"Failed to load {provider_name}"}

        return provider.generate(text, **params)

    def _cmd_list_voices(self, params: Dict[str, Any]) -> Dict[str, Any]:
        provider_name = params.get("provider", self.default_provider)
        provider = self.get_provider(provider_name)
        if not provider or not provider.is_loaded:
            return {"success": False, "error": "Provider not loaded"}
        return {"success": True, "voices": provider.get_voices()}

    def _cmd_set_voice(self, params: Dict[str, Any]) -> Dict[str, Any]:
        index = params.get("index", 0)
        provider_name = params.get("provider", self.default_provider)
        provider = self.get_provider(provider_name)
        if provider and provider.is_loaded:
            provider.set_voice(index)
        return {"success": True}

    def _cmd_set_rate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        rate = params.get("rate", 150)
        provider = self.get_provider(self.default_provider)
        if provider and hasattr(provider, "set_rate"):
            provider.set_rate(rate)
        return {"success": True}

    def _cmd_set_volume(self, params: Dict[str, Any]) -> Dict[str, Any]:
        volume = params.get("volume", 1.0)
        provider = self.get_provider(self.default_provider)
        if provider and hasattr(provider, "set_volume"):
            provider.set_volume(volume)
        return {"success": True}

    def _cmd_load_provider(self, params: Dict[str, Any]) -> Dict[str, Any]:
        name = params.get("provider", self.default_provider)
        provider = self.get_provider(name)
        if not provider:
            return {"success": False, "error": f"Unknown: {name}"}
        success = provider.load() if not provider.is_loaded else True
        return {"success": success, "provider": name}

    def _cmd_unload_provider(self, params: Dict[str, Any]) -> Dict[str, Any]:
        name = params.get("provider", self.default_provider)
        provider = self.providers.get(name)
        if provider:
            provider.unload()
        return {"success": True}

    def _cmd_list_providers(self, params: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        for name, cls in self.PROVIDERS.items():
            provider = self.providers.get(name)
            result[name] = {"loaded": provider.is_loaded if provider else False}
        return {"success": True, "providers": result, "default": self.default_provider}

    def _cmd_set_default(self, params: Dict[str, Any]) -> Dict[str, Any]:
        name = params.get("provider")
        if name not in self.PROVIDERS:
            return {"success": False, "error": f"Unknown: {name}"}
        self.default_provider = name
        return {"success": True, "default": name}

    def _cmd_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        loaded = [n for n, p in self.providers.items() if p.is_loaded]
        return {"success": True, "service": "audiogen", "loaded_providers": loaded}

    def handle_command(self, cmd: str, params: Dict[str, Any]) -> Dict[str, Any]:
        handler = self.commands.get(cmd)
        if not handler:
            return {"success": False, "error": f"Unknown command: {cmd}"}
        try:
            return handler(params)
        except Exception as e:
            return {"success": False, "error": str(e)}

    def connect_to_router(self, host: str = "localhost", port: int = 9900):
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.connect((host, port))

            reg_msg = Message(
                type=MessageType.REGISTER,
                payload={
                    "name": "audiogen",
                    "capabilities": ["tts", "audio", "speak"],
                    "commands": list(self.commands.keys()),
                },
            )
            self._socket.sendall(reg_msg.to_bytes())
            self._running = True
            logger.info("Connected to router")

            while self._running:
                try:
                    len_data = self._socket.recv(4)
                    if not len_data:
                        break
                    msg_len = struct.unpack(">I", len_data)[0]
                    msg_data = b""
                    while len(msg_data) < msg_len:
                        chunk = self._socket.recv(min(4096, msg_len - len(msg_data)))
                        if not chunk:
                            break
                        msg_data += chunk

                    msg = Message.from_bytes(msg_data)
                    if msg.type == MessageType.COMMAND:
                        result = self.handle_command(msg.payload.get("command", ""), msg.payload.get("params", {}))
                        resp = Message(type=MessageType.RESPONSE, payload=result, id=msg.id)
                        self._socket.sendall(resp.to_bytes())
                    elif msg.type == MessageType.SHUTDOWN:
                        self._running = False
                except socket.timeout:
                    continue
                except Exception as e:
                    logger.error(f"Error: {e}")
                    break
        except Exception as e:
            logger.error(f"Connection error: {e}")
        finally:
            if self._socket:
                self._socket.close()

    def run_standalone(self, port: int = 9904):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("localhost", port))
        server.listen(5)
        logger.info(f"AudioGen server listening on port {port}")
        self._running = True

        while self._running:
            try:
                server.settimeout(1.0)
                try:
                    client, _ = server.accept()
                except socket.timeout:
                    continue
                threading.Thread(target=self._handle_client, args=(client,), daemon=True).start()
            except Exception as e:
                logger.error(f"Server error: {e}")
        server.close()

    def _handle_client(self, client: socket.socket):
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
                    result = self.handle_command(msg.payload.get("command", ""), msg.payload.get("params", {}))
                    resp = Message(type=MessageType.RESPONSE, payload=result, id=msg.id)
                    client.sendall(resp.to_bytes())
        except Exception as e:
            logger.error(f"Client error: {e}")
        finally:
            client.close()

    def shutdown(self):
        self._running = False
        for p in self.providers.values():
            if p.is_loaded:
                p.unload()


def main():
    parser = argparse.ArgumentParser(description="Audio Generation Service")
    parser.add_argument("--port", type=int, default=9904)
    parser.add_argument("--router", type=str)
    parser.add_argument(
        "--provider",
        type=str,
        default="local",
        choices=["local", "system"],
        help="Default provider (local=Kokoro-82M neural TTS, system=OS SAPI/say/espeak)",
    )
    parser.add_argument("--speak", type=str, help="Speak text")
    parser.add_argument("--generate", type=str, help="Generate to file")

    args = parser.parse_args()
    service = AudioGen(default_provider=args.provider)

    if args.speak:
        result = service.handle_command("speak", {"text": args.speak, "provider": args.provider})
        print(json.dumps(result, indent=2))
        return

    if args.generate:
        result = service.handle_command("generate", {"text": args.generate, "provider": args.provider})
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
