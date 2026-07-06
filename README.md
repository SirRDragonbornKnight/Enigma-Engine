# Enigma Engine

This repository is **Enigma** — a from-scratch LLM. Its companion **Enigma Avatar** (the desktop
overlay an LLM can drive) was split into its **own repo** on 2026-06-28; the two are now separate
codebases linked only by a local message bus. Keep them straight:

| | What it is | Lives in |
|---|---|---|
| **Enigma** — the engine | A **from-scratch decoder-only LLM**: its own architecture, BPE tokenizer (base vocab 4718), and weights. NOT a wrapper around another model. Pipeline: **pretrain -> SFT -> serve**. | THIS repo: `enigma_engine/` + the root pipeline scripts (`pretrain_enigma.py`, `make_sft_data.py`, `finetune_enigma.py`, `serve_enigma.py`) |
| **Enigma Avatar** — the overlay | An **Electron desktop overlay** that floats a rigged 3D model (any `.glb`/`.gltf`/`.vrm`/`.fbx`) and animates it with **pure procedural motion** + spring physics + lip-sync. An optional *body* an LLM can drive. | **its own repo:** `C:\Users\SirKn\Enigma Avatar\` |

A **third location lives OUTSIDE this repo** and belongs to the avatar:

- `C:\Users\SirKn\3d Avatar\` — the avatar's **GLB models** (`Avatars/`) and the original
  **design spec** (`The project is to make a 3d model t.txt`). The avatar repo's `models/` dir
  is gitignored (large + non-redistributable); models are sourced from here.

## How the two relate

Enigma is the **brain**; the avatar is an optional **body**. They meet only at a local
WebSocket bus (`ws://127.0.0.1:8765`, see the avatar repo's `bus.py`): a served LLM — Enigma, or
Odysseus, or any OpenAI-compatible model — sends speech + motion commands and the overlay
renders them. **Either runs without the other**: you can pretrain/serve Enigma with no
avatar, and run the avatar against any LLM.

## Where to look

- **The LLM** — [`CLAUDE.md`](CLAUDE.md) is the authoritative guide: setup, the
  `pretrain -> SFT -> serve` pipeline, checkpointing, and guardrails.
- **The avatar** — its own repo at `C:\Users\SirKn\Enigma Avatar\` (`STATUS.md` = what works + how
  to launch it; `TODO.md` = backlog / audit log; `CLAUDE.md` = its working rules).
- **Project state** — `CLEANUP_TRACKER.md`, `CODE_REVIEW.md`, `KNOWN_ISSUES.md`, `SUGGESTIONS.md`.

## A note on older docs ("Modkit")

This repo began as **"Enigma AI Engine"**, briefly became **"Modkit"**
(a *Forge* that LoRA-fine-tuned Qwen, plus an MCP layer for Odysseus), and is now the
**from-scratch Enigma LLM** described in `CLAUDE.md`. **If a doc still says "Modkit" or
references Modkit-era entry points (`forge.py`, `train_enigma_lora.py`,
`make_enigma_local.py`, `modkit_mcp.py`), it predates the pivot —
`CLAUDE.md` is the current source of truth.**

## License

MIT (see [`LICENSE`](LICENSE)).
