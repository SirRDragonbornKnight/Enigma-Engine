"""Image understanding primitive -- her OWN eyes.

Phase 4.5 step 5 ("wire and retire", 2026-07-17): the BLIP scaffold this
module used to wrap is RETIRED. Captions now come from her own weights end
to end -- the distilled-and-aligned VisionEncoder + the vision_projection
trained by align_vision.py + the served frozen text model. Generation runs
in exactly the align-trained shape: [projected patches][BOS -> caption ->
EOS], greedy, no KV cache (a ~250-position forward on a 182M model is
millisecond-scale; recomputing per token keeps this free of cache surgery).

flatten_image_content() is the pure ingestion half: it rewrites ONE message's
OpenAI multimodal content list into plain text, honestly marking anything the
server cannot see, so serve stays a thin caller and the logic stays testable.
The caption enters chat as the "[image: ...]" marker shape the adopted model
was SFT-trained on -- token-level image injection into templated chat is the
NEXT training cycle's work (she has no trained image-delimiter tokens yet).
"""

from __future__ import annotations

import base64
import io
import threading
from pathlib import Path


class EyesError(Exception):
    """The captioning backend is unavailable or a caption failed."""


class Eyes:
    """One captioner, one image at a time (the shared lock serializes GPU use
    with text generation -- captioning borrows the served model)."""

    def __init__(
        self,
        model=None,
        tokenizer=None,
        encoder=None,
        gen_lock: threading.Lock | None = None,
        max_caption_tokens: int = 48,
        captioner_factory=None,
    ):
        self._lock = gen_lock if gen_lock is not None else threading.Lock()
        if captioner_factory is not None:  # tests inject a fake: PIL.Image -> str
            self._caption = captioner_factory()
            return
        if model is None or tokenizer is None or encoder is None:
            raise EyesError(
                "native eyes need the served model + tokenizer + her aligned "
                "vision encoder (train one with distill_vision_encoder.py then "
                "align_vision.py)"
            )
        if getattr(model, "vision_projection", None) is None:
            raise EyesError(
                "the served model has no vision_projection -- serve must open "
                "the vision port and graft the aligned projection weights"
            )
        import torch

        device = next(model.parameters()).device
        encoder = encoder.to(device).eval()
        image_size = getattr(getattr(encoder, "config", None), "image_size", 224)
        eos = getattr(tokenizer, "eos_token_id", 2)
        bos = getattr(tokenizer, "bos_token_id", 1)

        def _caption(img) -> str:
            from enigma_engine.core.vision_encoder import preprocess_image

            # Her native from-scratch normalization ([-1, 1]) -- the SAME
            # preprocessing distillation and alignment trained under.
            x = preprocess_image(img, image_size=image_size, imagenet_normalize=False).to(device)
            autocast = torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=(device.type == "cuda")
            )
            with torch.no_grad(), autocast:
                feats = encoder(x)
                ids = [bos]
                for _ in range(max_caption_tokens):
                    logits = model.forward_multimodal(
                        input_ids=torch.tensor([ids], dtype=torch.long, device=device),
                        vision_features=feats,
                    )
                    nxt = int(logits[0, -1].argmax())
                    if nxt == eos:
                        break
                    ids.append(nxt)
            return tokenizer.decode(ids[1:], skip_special_tokens=True).strip()

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
