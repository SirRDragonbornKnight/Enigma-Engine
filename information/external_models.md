# External Models -- What Works and What Doesn't

Short version: **the engine loads only its own `.pth` checkpoints.**
Enigma is a from-scratch model, not a host for other people's weights.

---

## Loading external formats: not supported

There are no import loaders. The constructors exist but refuse honestly:

| Method | Result |
|--------|--------|
| `Enigma.from_huggingface(...)` | raises `NotImplementedError` |
| `Enigma.from_gguf(...)` | raises `NotImplementedError` |
| `Enigma.from_onnx(...)` | raises `NotImplementedError` |

If you want to run a HuggingFace/GGUF/ONNX/Ollama model, run it under
its own server (Ollama, llama-server, vLLM, ...) -- they all speak the
same OpenAI API that `serve_enigma.py` does, so clients (including the
Enigma Avatar bus) can point at either.

---

## GGUF export: supported (one-way)

`enigma_engine/core/gguf.py` provides `export_to_gguf()`, which writes
an Enigma checkpoint out as a GGUF file so it can run under
**llama-server** (useful on hardware or platforms where the PyTorch
serve path is inconvenient). This is export only -- the engine never
reads GGUF back in.

RULED 2026-07-24: serving Enigma THROUGH llama.cpp was considered and
REJECTED -- her serving path stays from-scratch, our own code. Export
remains available for taking a checkpoint elsewhere, but it is not a
serving option here. The vendored `enigma_engine/bin/llama-server/`
binary (~1 GB) was DELETED 2026-07-25 (it was gitignored and never
committed; nothing to recover). Note also that the qwen3 auto-flip in
gguf.py is math-wrong for the v1 architecture (norms before rope,
missing NEOX permute).

---

## Native checkpoints

- Format: `.pth` containing `model_state_dict` + architecture config,
  written atomically (`.tmp` then rename).
- Served checkpoint of record: `models/enigma_dpo/model.pth`.
- All training scripts (`pretrain_enigma.py`, `finetune_enigma.py`,
  `dpo_enigma.py`) read and write this format and nothing else.

---

## Where external models DO appear

Some **organs** use external pretrained models as local backends -- that
is separate from the LLM itself:

| Organ | Backend |
|-------|---------|
| Eyes (`--eyes`) | **her own distilled ViT** (~19M, distilled from DINOv2-S -- no external model at runtime) |
| Ears (`--ears`) | faster-whisper (ASR) |
| Voice (`--voice`) | Kokoro-82M (TTS) |
| Imagination (`--image-gen`) | Stable Diffusion sd-turbo |

These are services behind serve flags; the language model in the loop
is always Enigma's own weights.
