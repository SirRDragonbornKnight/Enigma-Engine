#!/usr/bin/env python3
"""
3D Generation - Standalone AI 3D Model Generator

Creates 3D models from text prompts using:
- Shap-E (local)
- Built-in geometric primitives

Usage:
    python threed.py                           # Start service
    python threed.py --port 9905              # Custom port
    python threed.py --generate "a chair"    # Generate 3D model
"""

import argparse
import json
import logging
import math
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

OUTPUT_DIR = Path("outputs/3d")
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
# 3D Generation Providers
# =============================================================================


class Builtin3DGen:
    """Built-in 3D generator using geometric primitives."""

    def __init__(self):
        self.is_loaded = False

    def load(self) -> bool:
        self.is_loaded = True
        return True

    def unload(self):
        self.is_loaded = False

    def generate(self, prompt: str, **kwargs) -> Dict[str, Any]:
        if not self.is_loaded:
            return {"success": False, "error": "Not loaded"}

        try:
            start = time.time()
            prompt_lower = prompt.lower()

            # Choose shape based on keywords
            if any(w in prompt_lower for w in ["cube", "box", "square"]):
                obj_data = self._generate_cube()
            elif any(w in prompt_lower for w in ["sphere", "ball", "round"]):
                obj_data = self._generate_sphere()
            elif any(w in prompt_lower for w in ["cylinder", "tube", "pillar"]):
                obj_data = self._generate_cylinder()
            elif any(w in prompt_lower for w in ["cone", "pyramid", "point"]):
                obj_data = self._generate_cone()
            elif any(w in prompt_lower for w in ["torus", "donut", "ring"]):
                obj_data = self._generate_torus()
            else:
                # Default: combine primitives
                obj_data = self._generate_complex(prompt_lower)

            timestamp = int(time.time())
            filepath = OUTPUT_DIR / f"3d_{timestamp}.obj"
            filepath.write_text(obj_data)

            return {"success": True, "path": str(filepath), "format": "obj", "duration": time.time() - start}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _generate_cube(self, size: float = 1.0) -> str:
        s = size / 2
        vertices = [(-s, -s, -s), (s, -s, -s), (s, s, -s), (-s, s, -s), (-s, -s, s), (s, -s, s), (s, s, s), (-s, s, s)]
        faces = [(1, 2, 3, 4), (5, 8, 7, 6), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 8, 4), (5, 1, 4, 8)]
        return self._make_obj(vertices, faces)

    def _generate_sphere(self, radius: float = 1.0, segments: int = 16) -> str:
        vertices = []
        faces = []

        for i in range(segments + 1):
            lat = math.pi * i / segments - math.pi / 2
            for j in range(segments):
                lon = 2 * math.pi * j / segments
                x = radius * math.cos(lat) * math.cos(lon)
                y = radius * math.sin(lat)
                z = radius * math.cos(lat) * math.sin(lon)
                vertices.append((x, y, z))

        for i in range(segments):
            for j in range(segments):
                p1 = i * segments + j + 1
                p2 = p1 + 1 if j < segments - 1 else i * segments + 1
                p3 = p2 + segments
                p4 = p1 + segments
                if i < segments:
                    faces.append((p1, p2, p3, p4))

        return self._make_obj(vertices, faces)

    def _generate_cylinder(self, radius: float = 0.5, height: float = 2.0, segments: int = 16) -> str:
        vertices = []
        faces = []

        # Top and bottom centers
        vertices.append((0, height / 2, 0))
        vertices.append((0, -height / 2, 0))

        # Side vertices
        for y in [height / 2, -height / 2]:
            for i in range(segments):
                angle = 2 * math.pi * i / segments
                x = radius * math.cos(angle)
                z = radius * math.sin(angle)
                vertices.append((x, y, z))

        # Top cap
        for i in range(segments):
            next_i = (i + 1) % segments
            faces.append((1, 3 + i, 3 + next_i))

        # Bottom cap
        for i in range(segments):
            next_i = (i + 1) % segments
            faces.append((2, 3 + segments + next_i, 3 + segments + i))

        # Sides
        for i in range(segments):
            next_i = (i + 1) % segments
            top1 = 3 + i
            top2 = 3 + next_i
            bot1 = 3 + segments + i
            bot2 = 3 + segments + next_i
            faces.append((top1, top2, bot2, bot1))

        return self._make_obj(vertices, faces)

    def _generate_cone(self, radius: float = 1.0, height: float = 2.0, segments: int = 16) -> str:
        vertices = []
        faces = []

        # Apex and base center
        vertices.append((0, height / 2, 0))
        vertices.append((0, -height / 2, 0))

        # Base vertices
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            x = radius * math.cos(angle)
            z = radius * math.sin(angle)
            vertices.append((x, -height / 2, z))

        # Side faces
        for i in range(segments):
            next_i = (i + 1) % segments
            faces.append((1, 3 + next_i, 3 + i))

        # Base
        for i in range(segments):
            next_i = (i + 1) % segments
            faces.append((2, 3 + i, 3 + next_i))

        return self._make_obj(vertices, faces)

    def _generate_torus(
        self, major_r: float = 1.0, minor_r: float = 0.3, major_seg: int = 16, minor_seg: int = 8
    ) -> str:
        vertices = []
        faces = []

        for i in range(major_seg):
            theta = 2 * math.pi * i / major_seg
            for j in range(minor_seg):
                phi = 2 * math.pi * j / minor_seg
                x = (major_r + minor_r * math.cos(phi)) * math.cos(theta)
                y = minor_r * math.sin(phi)
                z = (major_r + minor_r * math.cos(phi)) * math.sin(theta)
                vertices.append((x, y, z))

        for i in range(major_seg):
            next_i = (i + 1) % major_seg
            for j in range(minor_seg):
                next_j = (j + 1) % minor_seg
                p1 = i * minor_seg + j + 1
                p2 = i * minor_seg + next_j + 1
                p3 = next_i * minor_seg + next_j + 1
                p4 = next_i * minor_seg + j + 1
                faces.append((p1, p2, p3, p4))

        return self._make_obj(vertices, faces)

    def _generate_complex(self, prompt: str) -> str:
        """Generate a complex shape combining primitives."""
        # Default to a simple object
        if "chair" in prompt:
            return self._chair()
        elif "table" in prompt:
            return self._table()
        else:
            return self._generate_cube()

    def _chair(self) -> str:
        """Generate a simple chair."""
        lines = ["# Chair\n"]
        v_offset = 0

        # Seat (flat cube)
        seat = self._cube_at(0, 0.45, 0, 0.5, 0.05, 0.5)
        lines.append(seat)
        v_offset += 8

        # Back (tall flat cube)
        back = self._cube_at(0, 0.85, -0.2, 0.5, 0.4, 0.05, v_offset)
        lines.append(back)
        v_offset += 8

        # Legs (4 cylinders approximated as thin cubes)
        positions = [(-0.2, 0.2, -0.2), (0.2, 0.2, -0.2), (-0.2, 0.2, 0.2), (0.2, 0.2, 0.2)]
        for x, y, z in positions:
            leg = self._cube_at(x, y, z, 0.05, 0.4, 0.05, v_offset)
            lines.append(leg)
            v_offset += 8

        return "".join(lines)

    def _table(self) -> str:
        """Generate a simple table."""
        lines = ["# Table\n"]
        v_offset = 0

        # Top
        top = self._cube_at(0, 0.75, 0, 1.0, 0.05, 0.6)
        lines.append(top)
        v_offset += 8

        # Legs
        positions = [(-0.4, 0.35, -0.25), (0.4, 0.35, -0.25), (-0.4, 0.35, 0.25), (0.4, 0.35, 0.25)]
        for x, y, z in positions:
            leg = self._cube_at(x, y, z, 0.06, 0.7, 0.06, v_offset)
            lines.append(leg)
            v_offset += 8

        return "".join(lines)

    def _cube_at(self, cx, cy, cz, sx, sy, sz, v_offset=0) -> str:
        """Generate a cube at position with size."""
        vertices = []
        for dx in [-sx / 2, sx / 2]:
            for dy in [-sy / 2, sy / 2]:
                for dz in [-sz / 2, sz / 2]:
                    vertices.append((cx + dx, cy + dy, cz + dz))

        lines = []
        for v in vertices:
            lines.append(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")

        # Faces (order matters for normals)
        o = v_offset
        faces = [
            (1 + o, 3 + o, 7 + o, 5 + o),
            (2 + o, 6 + o, 8 + o, 4 + o),
            (1 + o, 2 + o, 4 + o, 3 + o),
            (5 + o, 7 + o, 8 + o, 6 + o),
            (1 + o, 5 + o, 6 + o, 2 + o),
            (3 + o, 4 + o, 8 + o, 7 + o),
        ]
        for f in faces:
            lines.append(f"f {f[0]} {f[1]} {f[2]} {f[3]}\n")

        return "".join(lines)

    def _make_obj(self, vertices, faces) -> str:
        lines = ["# Generated 3D Object\n"]
        for v in vertices:
            lines.append(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        for f in faces:
            lines.append("f " + " ".join(str(i) for i in f) + "\n")
        return "".join(lines)


class Local3DGen:
    """Local 3D generation using Shap-E.

    2.1-threed slice (May 25 2026): removed silent fallback to Builtin3DGen.
    Previously load() would import Shap-E, swallow ImportError/Exception, and
    masquerade as loaded by promoting the builtin placeholder. Callers saw
    'local' loaded successfully while actually receiving placeholder output.
    Honors §4 'loud-on-real-issue, silent-on-normal-path'.
    """

    def __init__(self, use_cpu: bool = False):
        self.pipe = None
        self.is_loaded = False
        self.use_cpu = use_cpu

    def load(self) -> bool:
        try:
            import torch
            from diffusers import ShapEPipeline

            if torch.cuda.is_available() and not self.use_cpu:
                free_mem = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / (
                    1024**3
                )
                if free_mem < 3.5:
                    self.use_cpu = True
                    logger.warning("Low GPU memory, using CPU")

            dtype = torch.float32 if self.use_cpu else torch.float16
            device = "cpu" if self.use_cpu else "cuda"

            logger.info(f"Loading Shap-E ({device})...")
            self.pipe = ShapEPipeline.from_pretrained("openai/shap-e", torch_dtype=dtype, low_cpu_mem_usage=True)

            if not self.use_cpu and torch.cuda.is_available():
                self.pipe = self.pipe.to("cuda")
                try:
                    self.pipe.enable_attention_slicing(1)
                except Exception:
                    pass

            self.is_loaded = True
            logger.info("Shap-E loaded")
            return True
        except ImportError:
            logger.error("Shap-E not available: install 'diffusers' + 'transformers' + 'torch'")
            return False
        except Exception as e:
            logger.error(f"Shap-E load failed: {e}")
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

    def generate(
        self, prompt: str, guidance_scale: float = 15.0, num_inference_steps: int = 64, **kwargs
    ) -> Dict[str, Any]:
        if not self.is_loaded or not self.pipe:
            return {"success": False, "error": "Local Shap-E provider not loaded"}

        try:
            start = time.time()
            prompt = str(prompt).strip() if prompt else ""
            if not prompt:
                return {"success": False, "error": "Prompt cannot be empty"}

            output = self.pipe(
                prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
            )

            timestamp = int(time.time())
            mesh = output.images[0]

            try:
                import trimesh

                filepath = OUTPUT_DIR / f"3d_{timestamp}.ply"
                if hasattr(mesh, "export"):
                    mesh.export(str(filepath))
                else:
                    tmesh = trimesh.Trimesh(
                        vertices=mesh.verts.cpu().numpy() if hasattr(mesh, "verts") else mesh,
                        faces=mesh.faces.cpu().numpy() if hasattr(mesh, "faces") else None,
                    )
                    tmesh.export(str(filepath))
                return {"success": True, "path": str(filepath), "format": "ply", "duration": time.time() - start}
            except ImportError:
                import pickle

                filepath = OUTPUT_DIR / f"3d_{timestamp}.pkl"
                with open(filepath, "wb") as f:
                    pickle.dump(mesh, f)
                return {"success": True, "path": str(filepath), "format": "pkl", "duration": time.time() - start}
        except Exception as e:
            return {"success": False, "error": str(e)}


# =============================================================================
# ThreeD Service
# =============================================================================


class ThreeD:
    """3D Generation Service."""

    PROVIDERS = {
        "builtin": Builtin3DGen,
        "local": Local3DGen,
    }

    def __init__(self, default_provider: str = "local"):
        # 2.1-threed slice (May 25 2026): default flipped builtin -> local.
        # Local3DGen.load() now fails loud (logger.error + return False) when
        # Shap-E weights/deps are missing; _cmd_generate surfaces 'Failed to
        # load local' instead of silently producing builtin placeholder cubes.
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
            return {"success": False, "error": f"Unknown: {name}"}
        success = provider.load() if not provider.is_loaded else True
        return {"success": success, "provider": name}

    def _cmd_unload_provider(self, params: Dict[str, Any]) -> Dict[str, Any]:
        name = params.get("provider")
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
        return {"success": True, "service": "threed", "loaded_providers": loaded}

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
                    "name": "threed",
                    "capabilities": ["generate_3d", "3d", "model"],
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

    def run_standalone(self, port: int = 9905):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("localhost", port))
        server.listen(5)
        logger.info(f"ThreeD server listening on port {port}")
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
    parser = argparse.ArgumentParser(description="3D Generation Service")
    parser.add_argument("--port", type=int, default=9905)
    parser.add_argument("--router", type=str)
    parser.add_argument("--provider", type=str, default="local", choices=["builtin", "local"])
    parser.add_argument("--generate", type=str)

    args = parser.parse_args()
    service = ThreeD(default_provider=args.provider)

    if args.generate:
        result = service.handle_command("generate", {"prompt": args.generate, "provider": args.provider})
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
