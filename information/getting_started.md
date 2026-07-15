# Getting Started

Enigma is a from-scratch 182M decoder-only LLM -- her own architecture,
her own BPE tokenizer (base vocab 4718), her own weights. Everything runs
locally. This page gets you from a fresh checkout to chatting with her.

---

## Install

```bash
pip install -e ".[server,huggingface]"
```

The `server` extra pulls torch + FastAPI/uvicorn (required to serve).
Other extras enable organs: `voice`, `ears`, `eyes`, `imagegen`
(see `pyproject.toml`).

---

## Serve

```bash
python serve_enigma.py --model models/enigma_dpo/model.pth
```

(or just `enigma ...` -- the console script installed by pip.)

This starts an OpenAI-compatible API at `http://127.0.0.1:8000/v1`.
`models/enigma_dpo/model.pth` is the checkpoint of record; without
`--model`, the flag defaults to the raw pretrain checkpoint
`models/enigma_pretrain_large/latest.pth`.

| Flag | What it does |
|------|-------------|
| `--model PATH` | Enigma checkpoint (.pth) to serve |
| `--host` / `--port` | Bind address (default 127.0.0.1:8000) |
| `--max-context N` | Context window in tokens (default 1024) |
| `--memory-dir DIR` | Enable the memory store (JSONL + BM25); memories are injected into her context |
| `--voice` | Voice organ: `speak` tool + `/v1/audio/speech` (local pyttsx3/SAPI TTS) |
| `--ears` | Ears organ: `/v1/audio/transcriptions` (local faster-whisper) |
| `--eyes` | Eyes organ: image messages captioned into her context + `/v1/images/describe` (local BLIP) |
| `--image-gen` | Imagination organ: `imagine` tool + `/v1/images/generations` (local Stable Diffusion) |

## Chat

Any OpenAI client works -- point it at the server:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="local")
r = client.chat.completions.create(model="enigma", messages=[{"role": "user", "content": "Hi!"}])
print(r.choices[0].message.content)
```

---

## Where things live

| Path | Contents |
|------|----------|
| `models/` | Checkpoints (`enigma_dpo/model.pth` is the served one) |
| `data/` | Training data (`pretrain/`, `sft/`), memory stores |
| `enigma_engine/` | The package: model, tokenizer, chat format, organs |
| `~/.enigma_engine/images/` | PNGs from the `imagine` tool |

## Train your own

The full pipeline (pretrain -> SFT -> DPO -> eval) is in
[training_guide.md](training_guide.md); the command list is in
[quick_commands.md](quick_commands.md).
