"""
Image Generation Mod for Enigma AI Engine

Generate images from text prompts using various backends:
- Stable Diffusion WebUI (AUTOMATIC1111) API
- ComfyUI API
- Local diffusers (requires GPU + dependencies)

Threading model:
    All handlers are synchronous. Each mod runs in its own subprocess,
    so blocking HTTP requests and heavy computation are fine.

Usage:
    python -m mods.imagegen.main

The mod automatically detects available backends and uses the first one found.
"""

import base64
import json
import logging
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict

# Import the base class — handles connection, protocol, registration
from mod_base import ModClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional imports
try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    logger.warning("requests not installed - HTTP backends disabled")

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    logger.warning("PIL not installed - image saving disabled")


class ImageGenMod(ModClient):
    """Image generation mod with multiple backend support.

    Inherits connection/protocol from ModClient.
    Only adds backend detection and cmd_* handlers.

    Supported pipelines (diffusers backend):
        - txt2img: Text to image generation (default)
        - img2img: Image-to-image transformation
        - inpainting: Paint into masked regions
        - controlnet: Structure-guided generation (edge, depth, pose)
        - SDXL: Higher quality base model (1024x1024)

    Additional features:
        - LoRA loading for community fine-tuned styles
        - Scheduler selection (DPM-Solver, Euler, PNDM, etc.)
        - Pipeline caching (avoids reloading on every call)
    """

    def __init__(self, config_path: Path = None):
        """Initialize mod from config."""
        super().__init__(config_path)

        self.settings = self.config.get("settings", {})

        # Output directory
        self.output_dir = Path(self.settings.get("output_dir", "outputs/images"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Detect available backend
        self.backend = None
        self._backend_info: Dict[str, Any] = {}
        self._detect_backend()

        # Pipeline cache for diffusers (avoids reloading each call)
        self._pipe_cache: Dict[str, Any] = {}
        self._active_loras: list[str] = []

        logger.info(f"Active backend: {self.backend or 'none'}")

    def _detect_backend(self):
        """Detect which backend is available."""
        backend_pref = self.settings.get("backend", "auto")
        backends = self.settings.get("backends", {})

        # If specific backend requested
        if backend_pref != "auto" and backend_pref in backends:
            if self._check_backend(backend_pref, backends[backend_pref]):
                self.backend = backend_pref
                return

        # Auto-detect: try each enabled backend
        for name, config in backends.items():
            if config.get("enabled", False):
                if self._check_backend(name, config):
                    self.backend = name
                    return

        logger.warning("No image generation backend available")

    def _check_backend(self, name: str, config: Dict) -> bool:
        """Check if a backend is available."""
        if not HAS_REQUESTS:
            return False

        try:
            if name == "sdwebui":
                url = config.get("url", "http://127.0.0.1:7860")
                r = requests.get(f"{url}/sdapi/v1/sd-models", timeout=2)
                if r.status_code == 200:
                    self._backend_info["models"] = [m["title"] for m in r.json()]
                    self._backend_info["url"] = url
                    logger.info(f"SD WebUI available at {url}")
                    return True

            elif name == "comfyui":
                url = config.get("url", "http://127.0.0.1:8188")
                r = requests.get(f"{url}/system_stats", timeout=2)
                if r.status_code == 200:
                    self._backend_info["url"] = url
                    logger.info(f"ComfyUI available at {url}")
                    return True

            elif name == "diffusers":
                # Check if diffusers is installed
                try:
                    from diffusers import StableDiffusionPipeline  # noqa: F401

                    self._backend_info["model"] = config.get("model", "runwayml/stable-diffusion-v1-5")
                    self._backend_info["device"] = config.get("device", "auto")
                    self._backend_info["scheduler"] = config.get("scheduler", "default")
                    return True
                except ImportError:
                    return False

        except Exception as e:
            logger.debug(f"Backend {name} not available: {e}")

        return False

    def cmd_generate(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an image from a text prompt.

        Supports multiple generation modes via the ``mode`` arg:
            - txt2img (default): text prompt → image
            - img2img: input image + prompt → modified image
            - inpainting: input image + mask + prompt → inpainted image
        """
        prompt = args.get("prompt", "")
        if not prompt:
            raise ValueError("Prompt is required")

        negative_prompt = args.get("negative_prompt", "")
        width = args.get("width", 512)
        height = args.get("height", 512)
        steps = args.get("steps", 20)
        cfg_scale = args.get("cfg_scale", 7.5)
        seed = args.get("seed", -1)
        mode = args.get("mode", "txt2img")
        scheduler = args.get("scheduler", "default")

        if not self.backend:
            raise RuntimeError(
                "No image generation backend available. Please start Stable Diffusion WebUI or install diffusers."
            )

        # Generate based on backend
        if self.backend == "sdwebui":
            return self._generate_sdwebui(prompt, negative_prompt, width, height, steps, cfg_scale, seed)
        elif self.backend == "comfyui":
            return self._generate_comfyui(prompt, negative_prompt, width, height, steps, cfg_scale, seed)
        elif self.backend == "diffusers":
            return self._generate_diffusers(
                prompt,
                negative_prompt,
                width,
                height,
                steps,
                cfg_scale,
                seed,
                mode=mode,
                scheduler=scheduler,
                init_image=args.get("init_image"),
                mask_image=args.get("mask_image"),
                strength=args.get("strength", 0.75),
                controlnet_image=args.get("controlnet_image"),
                controlnet_model=args.get("controlnet_model"),
            )

        raise RuntimeError(f"Unknown backend: {self.backend}")

    def _generate_sdwebui(
        self, prompt: str, negative_prompt: str, width: int, height: int, steps: int, cfg_scale: float, seed: int
    ) -> Dict[str, Any]:
        """Generate using Stable Diffusion WebUI API."""
        url = self._backend_info.get("url", "http://127.0.0.1:7860")

        payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "cfg_scale": cfg_scale,
            "seed": seed,
        }

        response = requests.post(f"{url}/sdapi/v1/txt2img", json=payload, timeout=120)

        if response.status_code != 200:
            raise RuntimeError(f"SD WebUI error: {response.status_code}")

        result = response.json()
        images = result.get("images", [])

        if not images:
            raise RuntimeError("No images returned")

        # Save image
        saved_path = None
        if self.settings.get("save_images", True) and HAS_PIL:
            img_data = base64.b64decode(images[0])
            img = Image.open(BytesIO(img_data))

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"img_{timestamp}.png"
            saved_path = str(self.output_dir / filename)
            img.save(saved_path)
            logger.info(f"Saved image: {saved_path}")

        # Get actual seed used
        info = json.loads(result.get("info", "{}"))
        actual_seed = info.get("seed", seed)

        return {"image_base64": images[0], "saved_path": saved_path, "seed": actual_seed, "backend": "sdwebui"}

    def _generate_comfyui(
        self, prompt: str, negative_prompt: str, width: int, height: int, steps: int, cfg_scale: float, seed: int
    ) -> Dict[str, Any]:
        """Generate using ComfyUI API."""
        url = self._backend_info.get("url", "http://127.0.0.1:8188")

        # Basic txt2img workflow
        workflow = {
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "cfg": cfg_scale,
                    "denoise": 1,
                    "latent_image": ["5", 0],
                    "model": ["4", 0],
                    "negative": ["7", 0],
                    "positive": ["6", 0],
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "seed": seed if seed > 0 else 42,
                    "steps": steps,
                },
            },
            "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "v1-5-pruned-emaonly.safetensors"}},
            "5": {"class_type": "EmptyLatentImage", "inputs": {"batch_size": 1, "height": height, "width": width}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": prompt}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": negative_prompt}},
            "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
            "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "enigma", "images": ["8", 0]}},
        }

        # Queue prompt
        response = requests.post(f"{url}/prompt", json={"prompt": workflow}, timeout=120)

        if response.status_code != 200:
            raise RuntimeError(f"ComfyUI error: {response.status_code}")

        result = response.json()
        prompt_id = result.get("prompt_id")

        # Poll for completion
        for _ in range(60):
            time.sleep(2)
            history = requests.get(f"{url}/history/{prompt_id}", timeout=5)
            if history.status_code == 200:
                hist_data = history.json()
                if prompt_id in hist_data:
                    # Get output image path
                    outputs = hist_data[prompt_id].get("outputs", {})
                    if "9" in outputs and outputs["9"].get("images"):
                        img_info = outputs["9"]["images"][0]
                        return {
                            "saved_path": img_info.get("filename"),
                            "backend": "comfyui",
                            "subfolder": img_info.get("subfolder", ""),
                        }

        raise RuntimeError("ComfyUI generation timeout")

    def _generate_diffusers(
        self,
        prompt: str,
        negative_prompt: str,
        width: int,
        height: int,
        steps: int,
        cfg_scale: float,
        seed: int,
        *,
        mode: str = "txt2img",
        scheduler: str = "default",
        init_image: str | None = None,
        mask_image: str | None = None,
        strength: float = 0.75,
        controlnet_image: str | None = None,
        controlnet_model: str | None = None,
    ) -> Dict[str, Any]:
        """Generate using local diffusers library.

        Supports txt2img, img2img, inpainting, and ControlNet pipelines.
        Pipelines are cached after first load to avoid reloading on every call.
        """
        import torch

        model = self._backend_info.get("model", "runwayml/stable-diffusion-v1-5")
        device_pref = self._backend_info.get("device", "auto")

        if device_pref == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = device_pref

        dtype = torch.float16 if device == "cuda" else torch.float32

        # Select and load the right pipeline for the requested mode
        pipe = self._get_or_load_pipeline(mode, model, device, dtype, controlnet_model=controlnet_model)

        # Apply scheduler if requested
        if scheduler != "default":
            self._set_scheduler(pipe, scheduler)

        # Seed
        generator = None
        if seed > 0:
            generator = torch.Generator(device).manual_seed(seed)

        # Load input images if provided (img2img / inpainting / controlnet)
        pil_init = self._load_pil_image(init_image) if init_image else None
        pil_mask = self._load_pil_image(mask_image) if mask_image else None
        pil_control = self._load_pil_image(controlnet_image) if controlnet_image else None

        # Generate based on mode
        if mode == "img2img":
            if pil_init is None:
                raise ValueError("img2img mode requires 'init_image' path")
            image = pipe(
                prompt,
                image=pil_init,
                negative_prompt=negative_prompt or None,
                strength=strength,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                generator=generator,
            ).images[0]

        elif mode == "inpainting":
            if pil_init is None or pil_mask is None:
                raise ValueError("inpainting mode requires 'init_image' and 'mask_image' paths")
            image = pipe(
                prompt,
                image=pil_init,
                mask_image=pil_mask,
                negative_prompt=negative_prompt or None,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                generator=generator,
            ).images[0]

        elif mode == "controlnet":
            if pil_control is None:
                raise ValueError("controlnet mode requires 'controlnet_image' path")
            image = pipe(
                prompt,
                image=pil_control,
                negative_prompt=negative_prompt or None,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                generator=generator,
            ).images[0]

        else:
            # Default: txt2img
            image = pipe(
                prompt,
                negative_prompt=negative_prompt or None,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                generator=generator,
            ).images[0]

        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"img_{timestamp}.png"
        saved_path = str(self.output_dir / filename)
        image.save(saved_path)

        # Encode to base64
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode()

        return {
            "image_base64": img_base64,
            "saved_path": saved_path,
            "seed": seed,
            "mode": mode,
            "backend": "diffusers",
        }

    # -----------------------------------------------------------------
    # Pipeline management
    # -----------------------------------------------------------------

    def _get_or_load_pipeline(self, mode: str, model: str, device: str, dtype, *, controlnet_model: str | None = None):
        """Load or retrieve a cached diffusers pipeline.

        Pipelines keyed by (mode, model) to avoid redundant loading.
        SDXL models detected automatically by model name.
        """
        cache_key = f"{mode}:{model}:{controlnet_model or ''}"
        if cache_key in self._pipe_cache:
            return self._pipe_cache[cache_key]

        is_sdxl = "sdxl" in model.lower() or "xl" in model.lower()

        if mode == "img2img":
            pipe = self._load_img2img_pipeline(model, device, dtype, is_sdxl)
        elif mode == "inpainting":
            pipe = self._load_inpainting_pipeline(model, device, dtype, is_sdxl)
        elif mode == "controlnet":
            pipe = self._load_controlnet_pipeline(model, device, dtype, controlnet_model, is_sdxl)
        else:
            pipe = self._load_txt2img_pipeline(model, device, dtype, is_sdxl)

        self._pipe_cache[cache_key] = pipe
        return pipe

    def _load_txt2img_pipeline(self, model: str, device: str, dtype, is_sdxl: bool):
        """Load txt2img pipeline (SD 1.5 or SDXL)."""
        if is_sdxl:
            from diffusers import StableDiffusionXLPipeline

            pipe = StableDiffusionXLPipeline.from_pretrained(model, torch_dtype=dtype)
        else:
            from diffusers import StableDiffusionPipeline

            pipe = StableDiffusionPipeline.from_pretrained(model, torch_dtype=dtype)
        pipe = pipe.to(device)
        logger.info(f"Loaded txt2img pipeline: {model} ({device})")
        return pipe

    def _load_img2img_pipeline(self, model: str, device: str, dtype, is_sdxl: bool):
        """Load img2img pipeline."""
        if is_sdxl:
            from diffusers import StableDiffusionXLImg2ImgPipeline

            pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(model, torch_dtype=dtype)
        else:
            from diffusers import StableDiffusionImg2ImgPipeline

            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model, torch_dtype=dtype)
        pipe = pipe.to(device)
        logger.info(f"Loaded img2img pipeline: {model} ({device})")
        return pipe

    def _load_inpainting_pipeline(self, model: str, device: str, dtype, is_sdxl: bool):
        """Load inpainting pipeline."""
        if is_sdxl:
            from diffusers import StableDiffusionXLInpaintPipeline

            pipe = StableDiffusionXLInpaintPipeline.from_pretrained(model, torch_dtype=dtype)
        else:
            from diffusers import StableDiffusionInpaintPipeline

            pipe = StableDiffusionInpaintPipeline.from_pretrained(model, torch_dtype=dtype)
        pipe = pipe.to(device)
        logger.info(f"Loaded inpainting pipeline: {model} ({device})")
        return pipe

    def _load_controlnet_pipeline(self, model: str, device: str, dtype, controlnet_model: str | None, is_sdxl: bool):
        """Load ControlNet pipeline."""
        from diffusers import ControlNetModel

        cn_model_id = controlnet_model or "lllyasviel/sd-controlnet-canny"
        controlnet = ControlNetModel.from_pretrained(cn_model_id, torch_dtype=dtype)

        if is_sdxl:
            from diffusers import StableDiffusionXLControlNetPipeline

            pipe = StableDiffusionXLControlNetPipeline.from_pretrained(model, controlnet=controlnet, torch_dtype=dtype)
        else:
            from diffusers import StableDiffusionControlNetPipeline

            pipe = StableDiffusionControlNetPipeline.from_pretrained(model, controlnet=controlnet, torch_dtype=dtype)
        pipe = pipe.to(device)
        logger.info(f"Loaded controlnet pipeline: {model} + {cn_model_id} ({device})")
        return pipe

    @staticmethod
    def _load_pil_image(path: str):
        """Load a PIL Image from a file path."""
        if not HAS_PIL:
            raise RuntimeError("PIL is required for image loading")
        return Image.open(path).convert("RGB")

    # -----------------------------------------------------------------
    # Scheduler selection
    # -----------------------------------------------------------------

    @staticmethod
    def _set_scheduler(pipe, scheduler_name: str) -> None:
        """Swap the pipeline's scheduler.

        Supported: dpm, euler, euler_a, pndm, ddim, lms.
        """
        sched_config = pipe.scheduler.config
        name = scheduler_name.lower()

        if name == "dpm":
            from diffusers import DPMSolverMultistepScheduler

            pipe.scheduler = DPMSolverMultistepScheduler.from_config(sched_config)
        elif name == "euler":
            from diffusers import EulerDiscreteScheduler

            pipe.scheduler = EulerDiscreteScheduler.from_config(sched_config)
        elif name == "euler_a":
            from diffusers import EulerAncestralDiscreteScheduler

            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(sched_config)
        elif name == "pndm":
            from diffusers import PNDMScheduler

            pipe.scheduler = PNDMScheduler.from_config(sched_config)
        elif name == "ddim":
            from diffusers import DDIMScheduler

            pipe.scheduler = DDIMScheduler.from_config(sched_config)
        elif name == "lms":
            from diffusers import LMSDiscreteScheduler

            pipe.scheduler = LMSDiscreteScheduler.from_config(sched_config)
        else:
            logger.warning(f"Unknown scheduler '{scheduler_name}', keeping default")

    # -----------------------------------------------------------------
    # LoRA management
    # -----------------------------------------------------------------

    def cmd_load_lora(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Load a LoRA adapter into the active diffusers pipeline.

        Requires a cached txt2img pipeline (run generate at least once
        with diffusers backend first).
        """
        if self.backend != "diffusers":
            raise RuntimeError("LoRA loading only supported with diffusers backend")

        lora_path = args.get("path", "")
        weight = args.get("weight", 1.0)
        if not lora_path:
            raise ValueError("LoRA path is required")

        # Find active txt2img pipeline
        pipe = None
        for key, cached in self._pipe_cache.items():
            if key.startswith("txt2img:"):
                pipe = cached
                break

        if pipe is None:
            raise RuntimeError("No active pipeline. Generate an image first to load the pipeline.")

        pipe.load_lora_weights(lora_path)
        if weight != 1.0:
            pipe.fuse_lora(lora_scale=weight)

        self._active_loras.append(lora_path)
        logger.info(f"Loaded LoRA: {lora_path} (weight={weight})")

        return {
            "loaded": lora_path,
            "weight": weight,
            "active_loras": self._active_loras,
        }

    def cmd_unload_lora(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Unload all LoRA adapters from the active pipeline."""
        if self.backend != "diffusers":
            raise RuntimeError("LoRA management only supported with diffusers backend")

        for key, pipe in self._pipe_cache.items():
            if key.startswith("txt2img:"):
                try:
                    pipe.unfuse_lora()
                    pipe.unload_lora_weights()
                except Exception as exc:
                    logger.warning(f"Error unloading LoRA from {key}: {exc}")

        cleared = list(self._active_loras)
        self._active_loras.clear()
        logger.info("Unloaded all LoRAs")

        return {"unloaded": cleared}

    def cmd_list_schedulers(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List available scheduler names."""
        return {
            "schedulers": ["default", "dpm", "euler", "euler_a", "pndm", "ddim", "lms"],
            "current": self._backend_info.get("scheduler", "default"),
        }

    def cmd_status(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Return mod status."""
        return {
            "name": self.name,
            "running": self.running,
            "backend": self.backend,
            "backend_info": self._backend_info,
            "output_dir": str(self.output_dir),
            "cached_pipelines": list(self._pipe_cache.keys()),
            "active_loras": list(self._active_loras),
            "supported_modes": ["txt2img", "img2img", "inpainting", "controlnet"],
            "supported_schedulers": ["default", "dpm", "euler", "euler_a", "pndm", "ddim", "lms"],
        }

    def cmd_stop(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Stop the mod."""
        self.running = False
        return {"stopped": True}

    def cmd_list_models(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """List available models."""
        models = []

        if self.backend == "sdwebui" and "models" in self._backend_info:
            models = self._backend_info["models"]
        elif self.backend == "diffusers":
            models = [self._backend_info.get("model", "unknown")]

        return {"models": models, "backend": self.backend}

    def cmd_set_model(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Set the active model."""
        model = args.get("model", "")
        if not model:
            raise ValueError("Model name required")

        if self.backend == "sdwebui":
            url = self._backend_info.get("url", "http://127.0.0.1:7860")

            response = requests.post(
                f"{url}/sdapi/v1/options",
                json={"sd_model_checkpoint": model},
                timeout=60,
            )

            if response.status_code != 200:
                raise RuntimeError(f"Failed to set model: {response.status_code}")

            return {"model": model, "status": "changed"}

        return {"error": "Model switching not supported for this backend"}


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    mod = ImageGenMod()
    try:
        mod.run()
    except KeyboardInterrupt:
        logger.info("Mod stopped by user")
