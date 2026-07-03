#!/usr/bin/env python3
"""
Video Generation - Standalone AI Video Generator

Creates videos from text prompts using:
- CogVideoX-5B (local, THUDM) - current best text-to-video that fits 16GB VRAM
- Built-in animated GIF (explicit opt-in fallback only)

Usage:
    python videogen.py                           # Start service
    python videogen.py --port 9903              # Custom port
    python videogen.py --generate "ocean waves" # Generate video
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
from typing import Any, Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs/videos")
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
# Video Generation Providers
# =============================================================================


class BuiltinVideo:
    """Built-in animated GIF generator."""

    def __init__(self):
        self.is_loaded = False

    def load(self) -> bool:
        self.is_loaded = True
        return True

    def unload(self):
        self.is_loaded = False

    def generate(
        self, prompt: str, frames: int = 16, duration: float = 2.0, width: int = 256, height: int = 256, **kwargs
    ) -> Dict[str, Any]:
        if not self.is_loaded:
            return {"success": False, "error": "Not loaded"}

        try:
            from PIL import Image, ImageDraw

            start = time.time()
            images = []

            # Parse prompt for colors/style
            colors = self._parse_colors(prompt)

            for i in range(frames):
                img = Image.new("RGB", (width, height), colors["bg"])
                draw = ImageDraw.Draw(img)

                # Animate based on prompt keywords
                t = i / frames

                if any(w in prompt.lower() for w in ["wave", "ocean", "water"]):
                    self._draw_waves(draw, width, height, t, colors)
                elif any(w in prompt.lower() for w in ["fire", "flame", "burn"]):
                    self._draw_fire(draw, width, height, t, colors)
                elif any(w in prompt.lower() for w in ["spin", "rotate", "circle"]):
                    self._draw_spinner(draw, width, height, t, colors)
                elif any(w in prompt.lower() for w in ["bounce", "ball", "jump"]):
                    self._draw_bounce(draw, width, height, t, colors)
                else:
                    self._draw_pulse(draw, width, height, t, colors)

                images.append(img)

            timestamp = int(time.time())
            filepath = OUTPUT_DIR / f"video_{timestamp}.gif"

            images[0].save(
                str(filepath), save_all=True, append_images=images[1:], duration=int(duration * 1000 / frames), loop=0
            )

            return {"success": True, "path": str(filepath), "frames": frames, "duration": time.time() - start}
        except ImportError:
            return {"success": False, "error": "PIL not installed: pip install pillow"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _parse_colors(self, prompt: str) -> Dict[str, tuple]:
        p = prompt.lower()
        if "ocean" in p or "water" in p or "blue" in p:
            return {"bg": (20, 40, 80), "fg": (100, 180, 255), "accent": (255, 255, 255)}
        elif "fire" in p or "flame" in p or "red" in p:
            return {"bg": (40, 20, 20), "fg": (255, 100, 50), "accent": (255, 200, 100)}
        elif "forest" in p or "green" in p or "nature" in p:
            return {"bg": (20, 40, 30), "fg": (100, 200, 100), "accent": (200, 255, 150)}
        else:
            return {"bg": (30, 30, 50), "fg": (100, 150, 255), "accent": (255, 200, 100)}

    def _draw_waves(self, draw, w, h, t, colors):
        import math

        for y in range(0, h, 4):
            offset = int(20 * math.sin(y / 20 + t * 2 * math.pi))
            alpha = int(255 * (1 - y / h))
            color = tuple(int(c * alpha / 255) for c in colors["fg"])
            draw.line([(0, y + offset), (w, y + offset)], fill=color, width=2)

    def _draw_fire(self, draw, w, h, t, colors):
        import random

        random.seed(int(t * 100))
        for _ in range(50):
            x = random.randint(w // 4, 3 * w // 4)
            base_y = h - random.randint(10, 30)
            height_var = random.randint(50, h // 2)
            y = base_y - int(height_var * (1 - ((t + random.random()) % 1)))
            size = random.randint(3, 8)
            draw.ellipse([x - size, y - size, x + size, y + size], fill=colors["fg"])

    def _draw_spinner(self, draw, w, h, t, colors):
        import math

        cx, cy = w // 2, h // 2
        radius = min(w, h) // 3
        angle = t * 2 * math.pi
        for i in range(8):
            a = angle + i * math.pi / 4
            x = cx + int(radius * math.cos(a))
            y = cy + int(radius * math.sin(a))
            size = 10 + int(5 * math.sin(a + t * 4 * math.pi))
            draw.ellipse([x - size, y - size, x + size, y + size], fill=colors["fg"])

    def _draw_bounce(self, draw, w, h, t, colors):
        import math

        x = w // 2
        y = int(h * 0.3 + h * 0.4 * abs(math.sin(t * 2 * math.pi)))
        draw.ellipse([x - 30, y - 30, x + 30, y + 30], fill=colors["fg"])
        draw.ellipse([x - 25, y - 25, x + 25, y + 25], fill=colors["accent"])

    def _draw_pulse(self, draw, w, h, t, colors):
        cx, cy = w // 2, h // 2
        for i in range(5):
            phase = (t + i * 0.2) % 1
            radius = int(min(w, h) * 0.4 * phase)
            if radius > 0:
                draw.ellipse(
                    [cx - radius, cy - radius, cx + radius, cy + radius], outline=(*colors["fg"][:3],), width=2
                )


class LocalVideo:
    """Local text-to-video using CogVideoX-5B (THUDM).

    Replaces the legacy AnimateDiff path. CogVideoX-5B is the current best
    open-weights text-to-video model that fits a 16GB VRAM budget with bf16 +
    sequential CPU offload + VAE tiling. Quality is markedly higher than the
    SD1.5 motion-adapter approach for the same envelope, and the prompt->video
    contract matches imagegen/threed (no input image required, unlike SVD).

    Slice 2.1-videogen (May 25 2026): no silent fallback to BuiltinVideo.
    ImportError and load failures return False with logger.error so the
    dispatcher can surface 'Failed to load local' instead of masquerading
    placeholder GIFs as a successful CogVideoX run.
    """

    def __init__(self, model_id: str = "THUDM/CogVideoX-5b"):
        self.model_id = model_id
        self.pipe = None
        self.is_loaded = False

    def load(self) -> bool:
        try:
            import torch
            from diffusers import CogVideoXPipeline

            dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
            self.pipe = CogVideoXPipeline.from_pretrained(self.model_id, torch_dtype=dtype)

            if torch.cuda.is_available():
                # 16GB AI budget: prefer sequential CPU offload over plain .to('cuda').
                # Falls back to direct device move if accelerate is missing.
                try:
                    self.pipe.enable_sequential_cpu_offload()
                except Exception:
                    self.pipe = self.pipe.to("cuda")
                # VAE tiling/slicing reduces peak VRAM on the decode pass.
                try:
                    self.pipe.vae.enable_tiling()
                    self.pipe.vae.enable_slicing()
                except Exception:
                    pass

            self.is_loaded = True
            logger.info("CogVideoX-5B loaded")
            return True
        except ImportError:
            logger.error("CogVideoX not available: install 'diffusers>=0.30' + 'transformers' + 'torch' + 'accelerate'")
            return False
        except Exception as e:
            logger.error(f"CogVideoX load failed: {e}")
            return False

    def unload(self):
        if self.pipe:
            del self.pipe
            self.pipe = None
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
        self.is_loaded = False

    def generate(self, prompt: str, duration: float = 2.0, fps: int = 8, **kwargs) -> Dict[str, Any]:
        if not self.is_loaded or not self.pipe:
            return {"success": False, "error": "Local CogVideoX provider not loaded"}

        prompt = str(prompt).strip() if prompt else ""
        if not prompt:
            return {"success": False, "error": "Prompt cannot be empty"}

        try:
            start = time.time()
            # CogVideoX prefers frame counts that are multiples of 8.
            num_frames = max(8, int(duration * fps))
            num_frames = ((num_frames + 7) // 8) * 8

            output = self.pipe(
                prompt=prompt,
                num_frames=num_frames,
                guidance_scale=kwargs.get("guidance_scale", 6.0),
                num_inference_steps=kwargs.get("num_inference_steps", 50),
            )
            frames = output.frames[0]

            timestamp = int(time.time())
            try:
                from diffusers.utils import export_to_video

                filepath = OUTPUT_DIR / f"video_{timestamp}.mp4"
                export_to_video(frames, str(filepath), fps=fps)
            except Exception:
                # Fallback only for the on-disk encoder; the model itself ran.
                filepath = OUTPUT_DIR / f"video_{timestamp}.gif"
                frames[0].save(
                    str(filepath),
                    save_all=True,
                    append_images=frames[1:],
                    duration=int(1000 / fps),
                    loop=0,
                )

            return {
                "success": True,
                "path": str(filepath),
                "duration": time.time() - start,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}


# =============================================================================
# VideoGen Service
# =============================================================================


class VideoGen:
    """Video Generation Service."""

    PROVIDERS = {
        "builtin": BuiltinVideo,
        "local": LocalVideo,
    }

    def __init__(self, default_provider: str = "local"):
        # Slice 2.1-videogen (May 25 2026): default flipped builtin -> local so
        # CogVideoX-5B is the out-of-box behaviour. Builtin GIF stays available
        # as an explicit opt-in fallback, never an automatic substitute. Load
        # failures surface 'Failed to load local' via _cmd_generate per the
        # loud-on-real-issue principle.
        self.providers: Dict[str, Any] = {}
        self.default_provider = default_provider
        self._running = False
        self._socket: Optional[socket.socket] = None

        self.commands = {
            "generate": self._cmd_generate,
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

    def _cmd_generate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        prompt = params.get("prompt", "")
        provider_name = params.get("provider", self.default_provider)

        provider = self.get_provider(provider_name)
        if not provider:
            return {"success": False, "error": f"Unknown provider: {provider_name}"}

        if not provider.is_loaded:
            if not provider.load():
                return {"success": False, "error": f"Failed to load {provider_name}"}

        return provider.generate(prompt, **params)

    def _cmd_load_provider(self, params: Dict[str, Any]) -> Dict[str, Any]:
        name = params.get("provider", self.default_provider)
        provider = self.get_provider(name)
        if not provider:
            return {"success": False, "error": f"Unknown provider: {name}"}
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
            return {"success": False, "error": f"Unknown provider: {name}"}
        self.default_provider = name
        return {"success": True, "default": name}

    def _cmd_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        loaded = [n for n, p in self.providers.items() if p.is_loaded]
        return {"success": True, "service": "videogen", "loaded_providers": loaded}

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
                    "name": "videogen",
                    "capabilities": ["generate_video", "video"],
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

    def run_standalone(self, port: int = 9903):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("localhost", port))
        server.listen(5)
        logger.info(f"VideoGen server listening on port {port}")
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
    parser = argparse.ArgumentParser(description="Video Generation Service")
    parser.add_argument("--port", type=int, default=9903)
    parser.add_argument("--router", type=str)
    parser.add_argument(
        "--provider",
        type=str,
        default="local",
        choices=["builtin", "local"],
        help="Default provider (local=CogVideoX-5B, builtin=animated GIF placeholder)",
    )
    parser.add_argument("--generate", type=str)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--fps", type=int, default=8)

    args = parser.parse_args()
    service = VideoGen(default_provider=args.provider)

    if args.generate:
        result = service.handle_command(
            "generate", {"prompt": args.generate, "duration": args.duration, "fps": args.fps, "provider": args.provider}
        )
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
