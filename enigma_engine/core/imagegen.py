"""Image generation primitive -- the imagination organ.

Stable Diffusion via diffusers, behind the `imagine` built-in tool and
/v1/images/generations. Generation is a separate model family by design
(CLAUDE.md): the 182M decoder CONDUCTS -- it writes the prompt and calls the
tool -- and this organ paints the pixels.

Default model is sd-turbo: distilled SD 2.1, one denoising step, seconds per
image on GPU (~2.5 GB download from HuggingFace on first construction).
"""

from __future__ import annotations

import threading
from pathlib import Path

DEFAULT_MODEL_ID = "stabilityai/sd-turbo"


class ImageGenError(Exception):
    """The diffusion backend is unavailable or a generation failed."""


class Painter:
    """One pipeline, one generation at a time (lock serializes VRAM use)."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device=None, pipe_factory=None):
        self._lock = threading.Lock()
        self.model_id = model_id
        # Turbo-family models are distilled for 1-4 steps WITHOUT guidance;
        # running them at normal step counts wastes seconds for worse images.
        self._turbo = "turbo" in model_id.lower()
        if pipe_factory is not None:  # tests inject a fake
            self._pipe = pipe_factory()
            return
        try:
            # DiffusionPipeline, NOT AutoPipelineForText2Image: AutoPipeline
            # eagerly imports EVERY pipeline family, and under transformers 5.x
            # one of them (HunyuanDiT -> MT5Tokenizer) no longer imports.
            # DiffusionPipeline reads model_index.json and loads only the class
            # this model actually uses (measured 2026-07-14).
            from diffusers import DiffusionPipeline
        except ImportError as exc:
            raise ImageGenError("diffusers not installed -- pip install 'enigma-engine[imagegen]'") from exc
        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            dtype = torch.float16 if device == "cuda" else torch.float32
            self._pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
            self._pipe.to(device)
            # tqdm bars draw unicode blocks -- the cp1252 console rule.
            self._pipe.set_progress_bar_config(disable=True)
        except Exception as exc:
            raise ImageGenError(f"could not load '{model_id}': {exc}") from exc

    def generate(
        self,
        prompt: str,
        out_path: str | Path,
        steps: int | None = None,
        width: int = 512,
        height: int = 512,
    ) -> Path:
        """Render prompt to a PNG at out_path and return that path."""
        prompt = str(prompt).strip()
        if not prompt:
            raise ImageGenError("empty prompt")
        if steps is None:
            steps = 1 if self._turbo else 30
        guidance = 0.0 if self._turbo else 7.5
        with self._lock:
            try:
                image = self._pipe(
                    prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance,
                    width=width,
                    height=height,
                ).images[0]
            except Exception as exc:
                raise ImageGenError(f"generation failed: {exc}") from exc
        out_path = Path(out_path)
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(out_path)
        except OSError as exc:
            # A full disk / unwritable home leaked a bare OSError past the
            # organ's error type -- serve's narrow except turned an `imagine`
            # into a 500 mid-conversation, the one thing the builtin contract
            # promises never happens (review 2026-08-13).
            raise ImageGenError(f"could not write the image: {exc}") from exc
        return out_path
