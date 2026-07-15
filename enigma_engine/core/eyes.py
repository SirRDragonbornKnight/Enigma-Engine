"""Image understanding primitive -- the eyes organ.

A local BLIP captioner (transformers pipeline) gives her working eyes TODAY:
OpenAI-style image messages are captioned into text she can read, and
/v1/images/describe captions uploads directly. The native path
(core/vision_encoder.py + vision_projection, trained projectors) stays the
later in-model road for real-time vision -- this organ does not replace it.

flatten_image_content() is the pure ingestion half: it rewrites ONE message's
OpenAI multimodal content list into plain text, honestly marking anything the
server cannot see, so serve stays a thin caller and the logic stays testable.
Weights download from HuggingFace on first construction (~1 GB).
"""

from __future__ import annotations

import base64
import io
import threading
from pathlib import Path

DEFAULT_MODEL_ID = "Salesforce/blip-image-captioning-base"


class EyesError(Exception):
    """The captioning backend is unavailable or a caption failed."""


class Eyes:
    """One captioner, one image at a time (lock serializes GPU use)."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device=None, captioner_factory=None):
        self._lock = threading.Lock()
        self.model_id = model_id
        if captioner_factory is not None:  # tests inject a fake: PIL.Image -> str
            self._caption = captioner_factory()
            return
        # Direct BLIP classes, NOT pipeline("image-to-text"): transformers 5.x
        # removed that pipeline task (measured 2026-07-14); the model classes
        # are the stable surface.
        try:
            from transformers import BlipForConditionalGeneration, BlipProcessor
        except ImportError as exc:
            raise EyesError("transformers not installed -- pip install 'enigma-engine[eyes]'") from exc
        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            processor = BlipProcessor.from_pretrained(model_id)
            model = BlipForConditionalGeneration.from_pretrained(model_id).to(device)
            model.eval()
        except Exception as exc:
            raise EyesError(f"could not load captioner '{model_id}': {exc}") from exc

        def _caption(img) -> str:
            inputs = processor(images=img, return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=40)
            return processor.decode(out[0], skip_special_tokens=True)

        self._caption = _caption

    def describe(self, image) -> str:
        """Caption an image given as bytes, a path, or a PIL.Image."""
        try:
            from PIL import Image
        except ImportError as exc:
            raise EyesError("Pillow not installed -- pip install 'enigma-engine[eyes]'") from exc
        try:
            if isinstance(image, (bytes, bytearray)):
                img = Image.open(io.BytesIO(bytes(image)))
            elif isinstance(image, (str, Path)):
                img = Image.open(image)
            else:
                img = image
            img = img.convert("RGB")
        except EyesError:
            raise
        except Exception as exc:
            raise EyesError(f"could not read image: {exc}") from exc
        with self._lock:
            try:
                caption = (self._caption(img) or "").strip()
            except Exception as exc:
                raise EyesError(f"captioning failed: {exc}") from exc
        if not caption:
            raise EyesError("captioner returned nothing")
        return caption


def flatten_image_content(content: list, describe=None) -> str:
    """Rewrite one message's OpenAI multimodal content list into plain text.

    describe: bytes -> caption, or None when the eyes organ is off. Every part
    the model cannot see becomes an HONEST inline marker instead of silently
    vanishing -- she should never answer as if she saw an image she didn't.
    Only data: URLs are accepted; the server never fetches remote images.
    """
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
            continue
        if not isinstance(part, dict):
            parts.append(f"[unsupported content part: {type(part).__name__}]")
            continue
        ptype = part.get("type")
        if ptype == "text":
            parts.append(str(part.get("text") or ""))
        elif ptype == "image_url":
            url = str((part.get("image_url") or {}).get("url") or "")
            if describe is None:
                parts.append("[image ignored -- eyes disabled (start serve with --eyes)]")
            elif url.startswith("data:"):
                try:
                    payload = base64.b64decode(url.split(",", 1)[1])
                    parts.append(f"[image: {describe(payload)}]")
                except Exception as exc:
                    parts.append(f"[image error: {exc}]")
            else:
                parts.append("[image ignored -- only data: URLs are supported]")
        else:
            parts.append(f"[unsupported content part: {ptype!r}]")
    return "\n".join(p for p in parts if p).strip()
