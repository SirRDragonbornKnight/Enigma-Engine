# Enigma Engine

This repository is **Enigma** — a from-scratch LLM. Its companion **Enigma Avatar** (the desktop
overlay an LLM can drive) was split into its **own repo** on 2026-06-28; the two are now separate
codebases linked only by a local message bus. Keep them straight:

| | What it is | Lives in |
|---|---|---|
| **Enigma** — the engine | A **from-scratch decoder-only LLM**: its own architecture, BPE tokenizer (base vocab 4718), and weights. NOT a wrapper around another model. Pipeline: **pretrain -> (facts continued-pretrain, optional) -> SFT -> DPO -> serve**. | THIS repo: `enigma_engine/` + the root pipeline scripts (`pretrain_enigma.py`, `make_sft_data.py`, `finetune_enigma.py`, `make_dpo_data.py`, `dpo_enigma.py`, `serve_enigma.py`) |
| **Enigma Avatar** — the overlay | A **Unity 6 desktop avatar** (ground-up rebuild) that renders a rigged 3D model an LLM can drive — an optional *body*. The original Electron overlay is maintenance-only at `C:\Users\SirKn\Enigma Avatar window\`. | **its own repo:** `C:\Users\SirKn\Enigma Avatar\` |

A **third location lives OUTSIDE this repo** and belongs to the avatar:

- `C:\Users\SirKn\Desktop\Avatars\` — the avatar's live **GLB model zoo** (the dir in active
  use; ~40 models). `C:\Users\SirKn\3d Avatar\` keeps the original **design spec**
  (`The project is to make a 3d model t.txt`) and the earlier model copies. The avatar repo's
  `models/` dir is gitignored (large + non-redistributable); models are sourced from the zoo.

## How the two relate

Enigma is the **brain**; the avatar is an optional **body**. They meet only at a local
WebSocket bus (`ws://127.0.0.1:8765`, see the avatar repo's `bus.py`): a served LLM — Enigma, or
Odysseus, or any OpenAI-compatible model — sends speech + motion commands and the overlay
renders them. **Either runs without the other**: you can pretrain/serve Enigma with no
avatar, and run the avatar against any LLM.

## What she can do today

- Serve locally with organs as flags: `--voice` (Kokoro TTS), `--ears` (ASR),
  `--eyes` (her own distilled ViT captions images), `--image-gen` -- all local services,
  nothing leaves the machine.
- A sealed behavior gate (`eval_behavior.py` against the locked holdout) decides every
  checkpoint adoption; scorecards live in `EVAL_REDESIGN.md`.
- The v2 tokenizer + retokenized corpus are built (`TOKENIZER_V2_SPEC.md`); the v2
  pretrain is the next GPU spend.
- User-facing launchers: `Talk to Enigma.bat` / `Enigma Tray.bat`.

## Where to look

- **The LLM** — [`CLAUDE.md`](CLAUDE.md) is the authoritative guide: setup, the
  `pretrain -> (facts continued-pretrain, optional) -> SFT -> DPO -> serve` pipeline,
  checkpointing, and guardrails.
- **The avatar** — its own repo at `C:\Users\SirKn\Enigma Avatar\` (`CLAUDE.md` = its working
  rules + how to launch it; `TODO.md` = backlog / audit log).
- **Project state** — `BACKLOG.md` (the consolidated open-work ledger), `ROADMAP.md`,
  `CLEANUP_TRACKER.md`, `KNOWN_ISSUES.md`, `SUGGESTIONS.md`. Closed-bug ledgers and
  point-in-time review dumps live in `_archive/`.
- **Specs + eval state** -- `EVAL_REDESIGN.md` (the eval de-contamination design, dev-set
  count owner, and the sealed-gate scorecards), `TOKENIZER_V2_SPEC.md` (v2 vocab + corpus,
  landed), `VISION_QUALITY_SPEC.md` (the next vision align cycle, design only).

## A note on older docs ("Modkit")

This repo began as **"Enigma AI Engine"**, briefly became **"Modkit"**
(a *Forge* that LoRA-fine-tuned Qwen, plus an MCP layer for Odysseus), and is now the
**from-scratch Enigma LLM** described in `CLAUDE.md`. **If a doc still says "Modkit" or
references Modkit-era entry points (`forge.py`, `train_enigma_lora.py`,
`make_enigma_local.py`, `modkit_mcp.py`), it predates the pivot —
`CLAUDE.md` is the current source of truth.**

## License

MIT (see [`LICENSE`](LICENSE)).
