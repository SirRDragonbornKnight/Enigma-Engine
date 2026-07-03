# CLAUDE.md — Enigma Engine

Guidance for Claude Code working in this repo. Keep this file short (target <200 lines).
Harness-enforced rules (permissions, hooks, model) belong in `.claude/settings.json`, NOT here.

## What this is
This repo is **Enigma** — a **from-scratch** decoder-only LLM (its own architecture, BPE tokenizer base
vocab 4718, and weights; NOT a wrapper). Python. Pipeline is **pretrain → SFT → serve**; train + serve
share one chat renderer (`enigma_engine/core/chat_format.py`) so the prompt format can't drift.

**Enigma Avatar** (the Electron desktop overlay an LLM can drive) was split into a **separate sibling
repo** at `C:\Users\SirKn\Enigma Avatar\` on 2026-06-28 (full history preserved). The two meet only at
the local WebSocket bus (`ws://127.0.0.1:8765`). Work on the avatar in THAT repo, not here.

## Setup / build / test — run these first
- Python 3.12 (`C:\Users\SirKn\AppData\Local\Programs\Python\Python312\python.exe`).
- Enigma tests: `python -m pytest tests/ -q`   ·   Lint: `ruff check`
- (Avatar tests live in the separate **Enigma Avatar** repo now — run `node --test` there.)
- If a fresh session can't run the tests from this section, fix THIS section first.

## The pipeline
- **Pretrain** — `python pretrain_enigma.py` (trains from scratch on the memmapped token corpus).
  Resume the live run with `resume_training.ps1`; watch it with `tail_training_log.ps1`.
- **SFT data** — `python make_sft_data.py` → writes `data/sft/{tool_calls,identity,mix}.jsonl`.
- **Finetune (SFT)** — `python finetune_enigma.py` (base checkpoint → instruct/tool model;
  imports the optimizer/LR "arsenal" from `pretrain_enigma.py`).
- **Serve** — `python serve_enigma.py` (OpenAI-compatible FastAPI server; loads the `.pth`
  checkpoint directly). Run with `--help` for flags.

## Conventions / guardrails
- **Console output must be ASCII** — the Windows cp1252 console can't print `→`, `—`, etc.
  Use ASCII in any script that prints.
- **Do not change the live pretrain defaults** (`--optimizer adamw --schedule cosine`) — they are
  asserted bit-identical to the live training lineage. Muon / WSD are future-run-only, behind flags.
- Checkpoints rotate `latest.pth` → `prev.pth` atomically with a finite-loss guard; resume rebuilds
  config from the checkpoint and hard-fails on any arch/optimizer mismatch.
- Pretraining **DONE 2026-07-03**: full 287,882 steps / 56.6B tokens, val ppl 3.5 (`models/enigma_pretrain_large/model.pth`, SHA256-backed). Lineage is immutable; forward plan is `ROADMAP.md`. Bottleneck is now SFT data, not compute.
- From-scratch ethos: prefer fresh, correct code; engines should fail honestly ("feature absent")
  rather than guess.

## Enigma Avatar — now a separate repo
The desktop overlay moved to its own repository at `C:\Users\SirKn\Enigma Avatar\` (2026-06-28, full
history preserved). Its working guidance — the load-bearing **fail-safe click-through / panic-latch**
safety rules, the Origin-gated **bus** protocol, the per-engine **guard-at-the-boundary** ethos, the
**generic-only** rule, and the external spec/model locations under `C:\Users\SirKn\3d Avatar\` — lives
in THAT repo's `CLAUDE.md` + `STATUS.md`. From here, the only coupling is the WebSocket bus protocol.

## Working style
- "Make a plan first" means present the plan and **stop for approval** — don't build it in the same pass.
- Scope to exactly what's asked; deliver small, verify, then continue.
- **Fix in place, don't compensate.** When code already in the program is wrong or needs to change,
  CHANGE that code — don't bolt on new code (shims, wrapper layers, fallback branches, parallel
  implementations) to work around it. Adding compensating code to dodge a real fix leaves two versions
  of the truth to drift apart and grows the surface to maintain. Edit the source of the problem.

## Gotchas (mistakes made here — don't repeat them)
- **No C++ build toolchain on this box.** Only adopt npm/native deps that ship PREBUILT binaries —
  verify before installing. (`koffi`, a prebuilt FFI, works and is how the overlay calls Win32;
  `node-window-manager` needed a compiler and failed. Wasted a round-trip installing it.)
- **One-off Electron/Node probes** belong in the **Enigma Avatar** repo now (so `node_modules` resolves); write
  the result to a file and `process.exit()`, then delete the probe. Running from a dir without
  `node_modules` (e.g. the scratchpad), piping stdout through another command, or relying on
  `app.quit()` makes Electron HANG or pop a blocking GUI error dialog in this non-interactive shell —
  this happened twice and landed an error dialog on the user.
- **Verify load-bearing numbers/line-refs with a direct tool call BEFORE relaying them** — never trust
  subagent audit output. Reports here claimed a "1600-char" line (the real max was 702) and line
  numbers off by ~100, and inflated an ASCII-rule count by conflating comments + on-screen text with
  actual terminal output. Measure, show the receipt. (See also: ground every load-bearing number.)

## Project state docs
`CLEANUP_TRACKER.md`, `CODE_REVIEW.md`, `KNOWN_ISSUES.md`, `SUGGESTIONS.md`.
