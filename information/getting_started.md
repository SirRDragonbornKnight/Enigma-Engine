# Getting Started

Enigma is a from-scratch 238M-class decoder-only LLM -- her own architecture,
her own BPE tokenizer (live lineage vocab 16,366; the v1 base table was
4,718), her own weights. Everything runs locally. This page gets you from a
fresh checkout to chatting with her.

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
python serve_enigma.py --model models/enigma_v2_sft2/model.pth
```

(or just `enigma ...` -- the console script installed by pip.)

This starts an OpenAI-compatible API at `http://127.0.0.1:8000/v1`.
`models/enigma_v2_sft2/model.pth` is the checkpoint of record (the v2
lineage, adopted at Gate D on 2026-08-09) and is also the `--model`
default -- a bare serve command serves the adopted model. The previous
adopted model (v8) stays at `models/enigma_dpo/model.pth` as the
rollback; serve it by passing `--model` explicitly.

| Flag | What it does |
|------|-------------|
| `--model PATH` | Enigma checkpoint (.pth) to serve |
| `--host` / `--port` | Bind address (default 127.0.0.1:8000) |
| `--max-context N` | Context window in tokens (default 2048; capped down to the model's trained context with a WARN) |
| `--memory-dir DIR` | Enable the memory store (JSONL + BM25); memories are injected into her context |
| `--voice` | Voice organ: `speak` tool + `/v1/audio/speech` (local Kokoro-82M TTS) |
| `--ears` | Ears organ: `/v1/audio/transcriptions` (local faster-whisper) |
| `--eyes` | Eyes organ: image messages captioned into her context + `/v1/images/describe` (her own distilled ViT) |
| `--image-gen` | Imagination organ: `imagine` tool + `/v1/images/generations` (local Stable Diffusion) |
| `--search` | Search organ: a `<search>query</search>` span in her output runs a lookup through this machine's own SearXNG (WSL2 docker at 127.0.0.1:8888; `--search-url` to point elsewhere) and the results return to her context. Needs the v2 vocab that carves the tags; reachability is per-query, never a boot gate |
| `--allow-downloads` | Permit the ONE-TIME organ weight download from HuggingFace. Without it the server is fully offline (organs load from local cache only) -- first-ever use of an organ on a machine needs this flag once |
| `--device {auto,cuda,cpu}` | Device for the MODEL (default `auto` = cuda when available). `--device cuda` without CUDA REFUSES rather than silently running on the CPU |
| `--dry-multiplier F` | DRY sampling strength for requests that don't set their own (default 0 = off; 0.8 is the working value). Attacks verbatim repetition loops |
| `--tool-span-constrain` | Force valid JSON inside a tool-call span (xgrammar). Degrades to a WARN and unchanged decoding if the wheel or tokenizer parity is missing |
| `--state-reinject` | Re-inject the numeric facts the user stated earlier in THIS conversation as a prefix on the final user turn. Conversation-local; never reads the memory store |
| `--wake --wake-watch DIR` | Let her speak unprompted about NEW files in DIR (OFF by default; files already there at boot are never announced). `--wake-interval` / `--wake-cooldown` / `--wake-quiet H-H` tune it; `GET /v1/wake/recent` is the feed. **Without `--wake-watch` nothing can wake her** -- a bare timer tick does not call the model |

The three conversation levers and the organs are what `Start-Enigma.ps1` (and
`Talk to Enigma.bat` / `Enigma Tray.bat` / `Enigma HUD.bat`) turn on for daily use;
a bare `serve_enigma.py` leaves all three OFF.

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
| `models/` | Checkpoints (`enigma_v2_sft2/model.pth` is the served one; `enigma_dpo/model.pth` is the v8 rollback) |
| `data/` | Training data (`pretrain/`, `sft/`), memory stores |
| `enigma_engine/` | The package: model, tokenizer, chat format, organs |
| `~/.enigma_engine/images/` | PNGs from the `imagine` tool |

## Train your own

The full pipeline (pretrain -> SFT -> DPO -> eval) is in
[training_guide.md](training_guide.md); the command list is in
[quick_commands.md](quick_commands.md).
