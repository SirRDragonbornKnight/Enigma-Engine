# Image Generation Brick

Generate images from text prompts using Stable Diffusion,
ComfyUI, or the diffusers library.

## Commands

| Command | Description |
|---------|-------------|
| generate | Generate an image from a text prompt |
| status | Get brick status and available backends |
| stop | Stop the brick |
| list_models | List available image generation models |
| set_model | Select which model to use for generation |

## Backends

The imagegen brick supports multiple backends:

- **Stable Diffusion WebUI** — automatic1111 or forge
- **ComfyUI** — node-based workflow
- **Diffusers** — HuggingFace diffusers library

## Usage

```
generate prompt="a cat sitting on a cloud" steps=20 width=512
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| prompt | (required) | Text description of the image |
| negative_prompt | "" | What to avoid |
| width | 512 | Image width in pixels |
| height | 512 | Image height in pixels |
| steps | 20 | Denoising steps (more = better but slower) |
| cfg_scale | 7.5 | How closely to follow the prompt |
| seed | -1 | Random seed (-1 for random) |

## Output

Generated images are saved to `outputs/images/`.
