#!/usr/bin/env python3
"""
Image Generation - Standalone AI Image Generator

Creates images from text prompts using local providers:
- Placeholder (no deps)
- Stable Diffusion (local GPU)

Usage:
    python imagegen.py                           # Start service on default port
    python imagegen.py --port 9901              # Start on custom port
    python imagegen.py --generate "a sunset"   # Generate single image
    python imagegen.py --provider local        # Use specific provider
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

# Output directory
OUTPUT_DIR = Path("outputs/images")
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
# Image Generation Providers
# =============================================================================


class PlaceholderImage:
    """Built-in image generator with no external dependencies."""

    def __init__(self):
        self.is_loaded = False

    def load(self) -> bool:
        self.is_loaded = True
        return True

    def unload(self):
        self.is_loaded = False

    def generate(self, prompt: str, width: int = 512, height: int = 512, **kwargs) -> Dict[str, Any]:
        try:
            start = time.time()
            timestamp = int(time.time())
            filename = f"generated_{timestamp}.png"
            filepath = OUTPUT_DIR / filename

            try:
                from PIL import Image, ImageDraw, ImageFont

                img = Image.new("RGB", (width, height))
                for y in range(height):
                    r = int(40 + (y / height) * 60)
                    g = int(60 + (y / height) * 40)
                    b = int(100 + (y / height) * 80)
                    for x in range(width):
                        img.putpixel((x, y), (r, g, b))

                draw = ImageDraw.Draw(img)
                try:
                    font = ImageFont.truetype("arial.ttf", 20)
                except Exception:
                    font = ImageFont.load_default()

                words = prompt.split()
                lines = []
                current = ""
                for word in words:
                    test = current + " " + word if current else word
                    if len(test) < 40:
                        current = test
                    else:
                        lines.append(current)
                        current = word
                if current:
                    lines.append(current)

                y_pos = height // 2 - len(lines) * 15
                for line in lines[:5]:
                    bbox = draw.textbbox((0, 0), line, font=font)
                    text_width = bbox[2] - bbox[0]
                    x_pos = (width - text_width) // 2
                    draw.text((x_pos, y_pos), line, fill=(255, 255, 255), font=font)
                    y_pos += 30

                draw.text((10, height - 30), "PLACEHOLDER", fill=(150, 150, 150), font=font)
                img.save(str(filepath))

            except ImportError:
                # Minimal fallback with pure Python
                with open(filepath, "wb") as f:
                    f.write(self._create_minimal_png(width, height, prompt))

            return {"success": True, "path": str(filepath), "duration": time.time() - start, "is_placeholder": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _create_minimal_png(self, w: int, h: int, text: str) -> bytes:
        """Create minimal PNG without PIL."""
        import zlib

        def crc32(data):
            return zlib.crc32(data) & 0xFFFFFFFF

        def chunk(ctype, data):
            return struct.pack(">I", len(data)) + ctype + data + struct.pack(">I", crc32(ctype + data))

        raw = b""
        for y in range(h):
            raw += b"\x00"  # filter byte
            for x in range(w):
                r = int(40 + (y / h) * 60)
                g = int(60 + (y / h) * 40)
                b = int(100 + (y / h) * 80)
                raw += bytes([r, g, b])

        png = b"\x89PNG\r\n\x1a\n"
        png += chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        png += chunk(b"IDAT", zlib.compress(raw))
        png += chunk(b"IEND", b"")
        return png


class StableDiffusionLocal:
    """Local Stable Diffusion image generation."""

    def __init__(self, model_id: str = "nota-ai/bk-sdm-small"):
        self.model_id = model_id
        self.pipe = None
        self.is_loaded = False

    def load(self) -> bool:
        try:
            import torch
            from diffusers import StableDiffusionPipeline

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32

            logger.info(f"Loading Stable Diffusion from {self.model_id}...")
            logger.info(f"Device: {device}, dtype: {dtype}")

            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False,
            ).to(device)

            if device == "cuda":
                try:
                    self.pipe.enable_attention_slicing()
                except Exception:
                    pass

            self.is_loaded = True
            logger.info("Stable Diffusion loaded!")
            return True
        except ImportError:
            logger.error("Install: pip install diffusers transformers accelerate torch")
            return False
        except Exception as e:
            logger.error(f"Failed to load: {e}")
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

    def _set_scheduler(self, name: str) -> None:
        """Swap the pipeline scheduler by name."""
        name_lower = name.lower().replace("-", "").replace("_", "")
        try:
            if "dpmsolver" in name_lower or "dpm" in name_lower:
                from diffusers import DPMSolverMultistepScheduler

                self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
            elif "euler" in name_lower:
                from diffusers import EulerDiscreteScheduler

                self.pipe.scheduler = EulerDiscreteScheduler.from_config(self.pipe.scheduler.config)
            else:
                logger.warning(f"Unknown scheduler '{name}', keeping default")
        except Exception as exc:
            logger.warning(f"Failed to set scheduler '{name}': {exc}")

    def generate(
        self,
        prompt: str,
        width: int = 512,
        height: int = 512,
        steps: int = 30,
        guidance: float = 7.5,
        negative_prompt: str = "",
        scheduler: str = "",
        **kwargs,
    ) -> Dict[str, Any]:
        if not self.is_loaded:
            return {"success": False, "error": "Model not loaded"}

        try:
            start = time.time()
            prompt = str(prompt).strip() if prompt else ""
            if not prompt:
                return {"success": False, "error": "Prompt cannot be empty"}

            # Swap scheduler if requested
            if scheduler:
                self._set_scheduler(scheduler)

            neg_prompt = str(negative_prompt).strip() if negative_prompt else None
            if neg_prompt == "":
                neg_prompt = None

            result = self.pipe(
                prompt,
                negative_prompt=neg_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance,
            )

            image = result.images[0]
            timestamp = int(time.time())
            filename = f"sd_{timestamp}.png"
            filepath = OUTPUT_DIR / filename
            image.save(str(filepath))

            return {"success": True, "path": str(filepath), "duration": time.time() - start}
        except Exception as e:
            return {"success": False, "error": str(e)}


# =============================================================================
# ImageGen Service
# =============================================================================


class ImageGen:
    """Image Generation Service with router protocol support."""

    PROVIDERS = {
        "placeholder": PlaceholderImage,
        "local": StableDiffusionLocal,
    }

    def __init__(self, default_provider: str = "local"):
        # 2.1-imagegen slice (May 25 2026): default flipped placeholder -> local.
        # StableDiffusionLocal.load() is loud-on-real-issue (logger.error on
        # ImportError + Exception, returns False); _cmd_generate surfaces the
        # failure as {"success": False, "error": "Failed to load local"}
        # rather than silently falling back to placeholder. Honors §4
        # "loud-on-real-issue, silent-on-normal-path."
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
        """Get or create provider instance."""
        if name not in self.providers:
            if name not in self.PROVIDERS:
                return None
            self.providers[name] = self.PROVIDERS[name]()
        return self.providers[name]

    def _cmd_generate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an image."""
        prompt = params.get("prompt", "")
        width = params.get("width", 512)
        height = params.get("height", 512)
        provider_name = params.get("provider", self.default_provider)

        provider = self.get_provider(provider_name)
        if not provider:
            return {"success": False, "error": f"Unknown provider: {provider_name}"}

        if not provider.is_loaded:
            if not provider.load():
                return {"success": False, "error": f"Failed to load {provider_name}"}

        return provider.generate(prompt, width=width, height=height, **params)

    def _cmd_load_provider(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Load a provider."""
        name = params.get("provider", self.default_provider)
        provider = self.get_provider(name)
        if not provider:
            return {"success": False, "error": f"Unknown provider: {name}"}

        if provider.is_loaded:
            return {"success": True, "message": f"{name} already loaded"}

        success = provider.load()
        return {"success": success, "provider": name}

    def _cmd_unload_provider(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Unload a provider."""
        name = params.get("provider", self.default_provider)
        provider = self.providers.get(name)
        if provider:
            provider.unload()
        return {"success": True, "provider": name}

    def _cmd_list_providers(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List available providers."""
        result = {}
        for name, cls in self.PROVIDERS.items():
            provider = self.providers.get(name)
            result[name] = {"loaded": provider.is_loaded if provider else False, "class": cls.__name__}
        return {"success": True, "providers": result, "default": self.default_provider}

    def _cmd_set_default(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Set the default provider."""
        name = params.get("provider")
        if name not in self.PROVIDERS:
            return {"success": False, "error": f"Unknown provider: {name}"}
        self.default_provider = name
        return {"success": True, "default": name}

    def _cmd_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get service status."""
        loaded = []
        for name, provider in self.providers.items():
            if provider.is_loaded:
                loaded.append(name)
        return {"success": True, "service": "imagegen", "loaded_providers": loaded, "default": self.default_provider}

    def handle_command(self, cmd: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a command."""
        handler = self.commands.get(cmd)
        if not handler:
            return {"success": False, "error": f"Unknown command: {cmd}"}
        try:
            return handler(params)
        except Exception as e:
            return {"success": False, "error": str(e)}

    def connect_to_router(self, host: str = "localhost", port: int = 9900):
        """Connect to the router and register this service."""
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.connect((host, port))

            # Register
            reg_msg = Message(
                type=MessageType.REGISTER,
                payload={
                    "name": "imagegen",
                    "capabilities": ["generate_image", "image"],
                    "commands": list(self.commands.keys()),
                },
            )
            self._socket.sendall(reg_msg.to_bytes())

            self._running = True
            logger.info(f"Connected to router at {host}:{port}")

            # Listen for commands
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
                        cmd = msg.payload.get("command", "")
                        params = msg.payload.get("params", {})
                        result = self.handle_command(cmd, params)

                        resp = Message(type=MessageType.RESPONSE, payload=result, id=msg.id)
                        self._socket.sendall(resp.to_bytes())

                    elif msg.type == MessageType.SHUTDOWN:
                        logger.info("Shutdown requested")
                        self._running = False

                except socket.timeout:
                    continue
                except Exception as e:
                    logger.error(f"Error: {e}")
                    break

        except ConnectionRefusedError:
            logger.warning(f"Could not connect to router at {host}:{port}")
        except Exception as e:
            logger.error(f"Connection error: {e}")
        finally:
            if self._socket:
                self._socket.close()
                self._socket = None

    def run_standalone(self, port: int = 9901):
        """Run as standalone server (without router)."""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("localhost", port))
        server.listen(5)

        logger.info(f"ImageGen server listening on port {port}")
        self._running = True

        while self._running:
            try:
                server.settimeout(1.0)
                try:
                    client, addr = server.accept()
                except socket.timeout:
                    continue

                threading.Thread(target=self._handle_client, args=(client,), daemon=True).start()

            except Exception as e:
                logger.error(f"Server error: {e}")

        server.close()

    def _handle_client(self, client: socket.socket):
        """Handle a client connection."""
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
                    cmd = msg.payload.get("command", "")
                    params = msg.payload.get("params", {})
                    result = self.handle_command(cmd, params)

                    resp = Message(type=MessageType.RESPONSE, payload=result, id=msg.id)
                    client.sendall(resp.to_bytes())

        except Exception as e:
            logger.error(f"Client error: {e}")
        finally:
            client.close()

    def shutdown(self):
        """Shutdown the service."""
        self._running = False
        for provider in self.providers.values():
            if provider.is_loaded:
                provider.unload()


def main():
    parser = argparse.ArgumentParser(description="Image Generation Service")
    parser.add_argument("--port", type=int, default=9901, help="Server port")
    parser.add_argument("--router", type=str, help="Router address (host:port)")
    parser.add_argument(
        "--provider",
        type=str,
        default="local",
        choices=["placeholder", "local"],
        help="Default provider (local = Stable Diffusion; placeholder = no-deps fallback)",
    )
    parser.add_argument("--generate", type=str, help="Generate single image")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)

    args = parser.parse_args()

    service = ImageGen(default_provider=args.provider)

    # Single generation mode
    if args.generate:
        result = service.handle_command(
            "generate", {"prompt": args.generate, "width": args.width, "height": args.height, "provider": args.provider}
        )
        print(json.dumps(result, indent=2))
        return

    # Service mode
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
