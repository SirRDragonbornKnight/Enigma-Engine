#!/usr/bin/env python3
"""
Vision - Standalone Screen/Image Analysis Service

Screen capture and image analysis using:
- PIL for screenshots
- OCR (tesseract or built-in)
- AI image analysis

Usage:
    python vision.py                        # Start service
    python vision.py --port 9906           # Custom port
    python vision.py --capture             # Capture screen
    python vision.py --analyze image.png  # Analyze image
"""

import argparse
import base64
import json
import logging
import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs/vision")
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
# Vision Components
# =============================================================================


class ScreenCapture:
    """Screen capture functionality."""

    def __init__(self):
        self.is_loaded = False
        self._method = None

    def load(self) -> bool:
        # Try different capture methods
        try:
            import mss  # noqa: F401  -- availability probe (import success = capability present)

            self._method = "mss"
            self.is_loaded = True
            logger.info("Using mss for screen capture")
            return True
        except ImportError:
            pass

        try:
            from PIL import ImageGrab  # noqa: F401  -- availability probe (import success = capability present)

            self._method = "pil"
            self.is_loaded = True
            logger.info("Using PIL for screen capture")
            return True
        except ImportError:
            pass

        try:
            import pyautogui  # noqa: F401  -- availability probe (import success = capability present)

            self._method = "pyautogui"
            self.is_loaded = True
            logger.info("Using pyautogui for screen capture")
            return True
        except ImportError:
            pass

        logger.error("No screen capture available. Install: pip install mss pillow pyautogui")
        return False

    def unload(self):
        self.is_loaded = False

    def capture(self, region: Optional[tuple] = None, save: bool = True) -> Dict[str, Any]:
        """Capture screen or region."""
        if not self.is_loaded:
            return {"success": False, "error": "Not loaded"}

        try:
            start = time.time()
            image = None

            if self._method == "mss":
                import mss
                from PIL import Image

                with mss.mss() as sct:
                    monitor = sct.monitors[1]  # Primary monitor
                    if region:
                        monitor = {"left": region[0], "top": region[1], "width": region[2], "height": region[3]}
                    screenshot = sct.grab(monitor)
                    image = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")

            elif self._method == "pil":
                from PIL import ImageGrab

                bbox = region if region else None
                image = ImageGrab.grab(bbox=bbox)

            elif self._method == "pyautogui":
                import pyautogui

                if region:
                    image = pyautogui.screenshot(region=region)
                else:
                    image = pyautogui.screenshot()

            if not image:
                return {"success": False, "error": "Capture failed"}

            result = {"success": True, "width": image.width, "height": image.height, "duration": time.time() - start}

            if save:
                timestamp = int(time.time())
                filepath = OUTPUT_DIR / f"capture_{timestamp}.png"
                image.save(str(filepath))
                result["path"] = str(filepath)

            # Also include base64 for remote access
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            result["data"] = base64.b64encode(buffer.getvalue()).decode()

            return result
        except Exception as e:
            return {"success": False, "error": str(e)}


class OCR:
    """Optical Character Recognition."""

    def __init__(self):
        self.is_loaded = False
        self._method = None

    def load(self) -> bool:
        # Try Tesseract
        try:
            import pytesseract

            # Test if tesseract is available
            pytesseract.get_tesseract_version()
            self._method = "tesseract"
            self.is_loaded = True
            logger.info("Using Tesseract OCR")
            return True
        except Exception:
            pass

        # Try EasyOCR. Force CPU: screen OCR is fast enough on CPU (~1-2s) and
        # must NOT contend for VRAM with the resident Enigma model (~9 GB).
        try:
            import easyocr

            self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            self._method = "easyocr"
            self.is_loaded = True
            logger.info("Using EasyOCR")
            return True
        except ImportError:
            pass

        # Built-in fallback (limited)
        self._method = "builtin"
        self.is_loaded = True
        logger.info("Using basic OCR (limited)")
        return True

    def unload(self):
        self.is_loaded = False
        if hasattr(self, "_reader"):
            del self._reader

    def extract_text(self, image_path: str) -> Dict[str, Any]:
        """Extract text from image."""
        if not self.is_loaded:
            return {"success": False, "error": "Not loaded"}

        try:
            from PIL import Image

            image = Image.open(image_path)

            if self._method == "tesseract":
                import pytesseract

                text = pytesseract.image_to_string(image)
                return {"success": True, "text": text.strip()}

            elif self._method == "easyocr":
                result = self._reader.readtext(image_path)
                text = " ".join([r[1] for r in result])
                return {
                    "success": True,
                    "text": text.strip(),
                    "boxes": [{"text": r[1], "confidence": r[2]} for r in result],
                }

            else:
                # Basic analysis - just report dimensions
                return {
                    "success": True,
                    "text": "",
                    "note": "Install pytesseract or easyocr for OCR",
                    "image_size": f"{image.width}x{image.height}",
                }
        except Exception as e:
            return {"success": False, "error": str(e)}


class ImageAnalyzer:
    """AI-powered image analysis."""

    def __init__(self):
        self.is_loaded = False
        self._method = None
        self._model = None

    def load(self) -> bool:
        # Try BLIP for captioning
        try:
            from transformers import BlipProcessor, BlipForConditionalGeneration
            import torch

            self._processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self._model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

            if torch.cuda.is_available():
                self._model = self._model.to("cuda")

            self._method = "blip"
            self.is_loaded = True
            logger.info("Using BLIP for image analysis")
            return True
        except ImportError:
            pass
        except Exception as e:
            logger.debug(f"BLIP failed: {e}")

        # Fallback to basic analysis
        self._method = "basic"
        self.is_loaded = True
        logger.info("Using basic image analysis")
        return True

    def unload(self):
        if self._model:
            del self._model
            self._model = None
        if hasattr(self, "_processor"):
            del self._processor
        self.is_loaded = False

    def analyze(self, image_path: str, prompt: Optional[str] = None) -> Dict[str, Any]:
        """Analyze an image."""
        if not self.is_loaded:
            return {"success": False, "error": "Not loaded"}

        try:
            from PIL import Image

            image = Image.open(image_path)

            result = {
                "success": True,
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "format": image.format,
            }

            if self._method == "blip":
                import torch

                if prompt:
                    inputs = self._processor(image, prompt, return_tensors="pt")
                else:
                    inputs = self._processor(image, return_tensors="pt")

                if torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}

                output = self._model.generate(**inputs, max_new_tokens=50)
                caption = self._processor.decode(output[0], skip_special_tokens=True)

                result["caption"] = caption
                result["analysis"] = caption
            else:
                # Basic analysis
                result["analysis"] = f"Image: {image.width}x{image.height} {image.mode}"

                # Color analysis
                if image.mode in ("RGB", "RGBA"):
                    colors = image.getcolors(maxcolors=10000)
                    if colors:
                        colors.sort(key=lambda x: -x[0])
                        result["dominant_colors"] = [
                            {"count": c[0], "rgb": c[1][:3] if len(c[1]) > 3 else c[1]} for c in colors[:5]
                        ]

            return result
        except Exception as e:
            return {"success": False, "error": str(e)}


# =============================================================================
# Vision Service
# =============================================================================


class Vision:
    """Vision Service - screen capture and image analysis."""

    def __init__(self):
        self.capture = ScreenCapture()
        self.ocr = OCR()
        self.analyzer = ImageAnalyzer()
        self._running = False
        self._socket: Optional[socket.socket] = None
        self._watching = False
        self._watch_thread = None

        self.commands = {
            "capture": self._cmd_capture,
            "analyze": self._cmd_analyze,
            "ocr": self._cmd_ocr,
            "start_watch": self._cmd_start_watch,
            "stop_watch": self._cmd_stop_watch,
            "load": self._cmd_load,
            "status": self._cmd_status,
        }

    def _cmd_capture(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.capture.is_loaded:
            self.capture.load()

        region = params.get("region")
        if region and isinstance(region, list) and len(region) == 4:
            region = tuple(region)
        else:
            region = None

        return self.capture.capture(region=region, save=params.get("save", True))

    def _cmd_analyze(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.analyzer.is_loaded:
            self.analyzer.load()

        image_path = params.get("image_path")
        if not image_path:
            # Capture first
            cap_result = self._cmd_capture({"save": True})
            if not cap_result.get("success"):
                return cap_result
            image_path = cap_result.get("path")

        return self.analyzer.analyze(image_path, params.get("prompt"))

    def _cmd_ocr(self, params: Dict[str, Any]) -> Dict[str, Any]:
        if not self.ocr.is_loaded:
            self.ocr.load()

        image_path = params.get("image_path")
        if not image_path:
            cap_result = self._cmd_capture({"save": True})
            if not cap_result.get("success"):
                return cap_result
            image_path = cap_result.get("path")

        return self.ocr.extract_text(image_path)

    def _cmd_start_watch(self, params: Dict[str, Any]) -> Dict[str, Any]:
        interval = params.get("interval", 5)
        if self._watching:
            return {"success": True, "message": "Already watching"}

        self._watching = True
        self._watch_thread = threading.Thread(target=self._watch_loop, args=(interval,), daemon=True)
        self._watch_thread.start()
        return {"success": True, "message": f"Watching every {interval}s"}

    def _cmd_stop_watch(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self._watching = False
        return {"success": True, "message": "Watch stopped"}

    def _watch_loop(self, interval: float):
        while self._watching and self._running:
            result = self._cmd_capture({"save": True})
            if result.get("success"):
                logger.info(f"Captured: {result.get('path')}")
            time.sleep(interval)

    def _cmd_load(self, params: Dict[str, Any]) -> Dict[str, Any]:
        components = params.get("components", ["capture", "ocr", "analyzer"])
        results = {}
        if "capture" in components:
            results["capture"] = self.capture.load()
        if "ocr" in components:
            results["ocr"] = self.ocr.load()
        if "analyzer" in components:
            results["analyzer"] = self.analyzer.load()
        return {"success": True, "loaded": results}

    def _cmd_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "success": True,
            "service": "vision",
            "capture_loaded": self.capture.is_loaded,
            "ocr_loaded": self.ocr.is_loaded,
            "analyzer_loaded": self.analyzer.is_loaded,
            "watching": self._watching,
        }

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
                    "name": "vision",
                    "capabilities": ["capture", "screenshot", "ocr", "analyze_image"],
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

    def run_standalone(self, port: int = 9906):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("localhost", port))
        server.listen(5)
        logger.info(f"Vision server listening on port {port}")
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
        self._watching = False
        self.capture.unload()
        self.ocr.unload()
        self.analyzer.unload()


def main():
    parser = argparse.ArgumentParser(description="Vision Service")
    parser.add_argument("--port", type=int, default=9906)
    parser.add_argument("--router", type=str)
    parser.add_argument("--capture", action="store_true", help="Capture screen")
    parser.add_argument("--analyze", type=str, help="Analyze image")
    parser.add_argument("--ocr", type=str, help="Extract text from image")

    args = parser.parse_args()
    service = Vision()

    if args.capture:
        result = service.handle_command("capture", {})
        print(json.dumps({k: v for k, v in result.items() if k != "data"}, indent=2))
        return

    if args.analyze:
        result = service.handle_command("analyze", {"image_path": args.analyze})
        print(json.dumps(result, indent=2))
        return

    if args.ocr:
        result = service.handle_command("ocr", {"image_path": args.ocr})
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
