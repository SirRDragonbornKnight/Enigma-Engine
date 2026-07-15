# CLAUDE.md — Enigma Engine

Guidance for Claude Code working in this repo. Keep this file short (target <200 lines).
Harness-enforced rules (permissions, hooks, model) belong in `.claude/settings.json`, NOT here.

## What this is
This repo is **Enigma** — a **from-scratch** decoder-only LLM (its own architecture, BPE tokenizer base
vocab 4718, and weights; NOT a wrapper). Python. Pipeline is **pretrain → SFT → serve**; train + serve
share one chat renderer (`enigma_engine/core/chat_format.py`) so the prompt format can't drift.

**Multimodal state (measured 2026-07-14):** Enigma is a TEXT decoder that will PERCEIVE
(image/audio INPUT) in-model. GENERATION (image/video/speech-audio) is a separate model family —
NOT this LLM's job; if wanted in-repo it is a bundled service, not the model painting pixels.
Perception is HALF-BUILT; text-only ships today:
- Ready: `forward_multimodal` + `vision_projection`/`audio_projection` (`core/model.py`), encoders
  `core/vision_encoder.py` + `core/audio_encoder.py`, Forge modes `train_vision`/`train_audio`.
- NOT wired / NOT trained (verified): the shipped `enigma_dpo` checkpoint has NO projection weights
  and `use_vision`/`use_audio` are off; `chat_format.py` has NO image/audio tokens; `serve_enigma.py`
  has NO multimodal path; `collect_audio_data.py` does NOT exist (vision has a collector).
- To finish seeing/hearing: enable config flags -> paired image/audio<->text data -> train projectors
  -> add image/audio tokens to `chat_format` -> wire `serve`. (Encoder trains FROZEN; only the
  projector updates, and it lives in `self.model` so it IS saved.) Restore history: `KNOWN_ISSUES.md` #11.

The Modkit-era `mods/` + `plugins/` subsystem and its `commands`/`mod_tools`/`plugin_loader` registry
were REMOVED 2026-07-13 (never loaded by `serve_enigma.py`; superseded). Pip distribution renamed
`modkit` -> `enigma-engine`. Capabilities that are genuinely separate models (image gen, TTS, ASR,
search) return as hub/tool services, NOT in-repo mods.

**Enigma Avatar** (the desktop overlay an LLM can drive) was split into a **separate sibling
repo** at `C:\Users\SirKn\Enigma Avatar\` on 2026-06-28 (full history preserved) and is now a
**Unity 6 rebuild** — the Electron+three.js predecessor lives at `C:\Users\SirKn\Enigma Avatar
window\` (still runnable, maintenance-only). The two meet only at the local WebSocket bus
(`ws://127.0.0.1:8765`). Work on the avatar in THAT repo, not here.

## Setup / build / test — run these first
- Python 3.12 (`C:\Users\SirKn\AppData\Local\Programs\Python\Python312\python.exe`).
- Enigma tests: `python -m pytest tests/ -q`   ·   Lint: `ruff check` — use the system Python
  above or `venv\Scripts\python.exe`; NOT `.venv\` (it has no pytest/ruff installed).
- (Avatar tests live in the **Enigma Avatar** repo — the gate there is
  `powershell -File tools\verify.ps1` plus `python -m pytest python/tests`.
  `node --test` belongs to the Electron predecessor in `Enigma Avatar window\`.)
- If a fresh session can't run the tests from this section, fix THIS section first.

## The pipeline
- **Pretrain** — `python pretrain_enigma.py` (trains from scratch on the memmapped token corpus).
  Resume the live run with `resume_training.ps1`; watch it with `tail_training_log.ps1`.
- **SFT data** — `python make_sft_data.py` → writes `data/sft/{tool_calls,identity,mix}.jsonl`.
- **Finetune (SFT)** — `python finetune_enigma.py` (base checkpoint → instruct/tool model;
  imports the optimizer/LR "arsenal" from `enigma_engine.core.optim`, shared with pretrain).
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
- SFT+DPO **ADOPTED 2026-07-06**: all three launchers (`Start-Enigma.ps1`/`.bat`, `Launch Enigma.bat`)
  serve `models/enigma_dpo/model.pth`; revert target is `models/enigma_sft`. Backed up with SHA256
  receipts at `Enigma Backups\enigma_dpo_adopted\` (2026-07-13).
- From-scratch ethos: prefer fresh, correct code; engines should fail honestly ("feature absent")
  rather than guess.

## Enigma Avatar — now a separate repo
The desktop overlay moved to its own repository at `C:\Users\SirKn\Enigma Avatar\` (2026-06-28, full
history preserved) and was rebuilt in **Unity 6**; the Electron predecessor is
`C:\Users\SirKn\Enigma Avatar window\` (maintenance-only, do not break it). Its working guidance —
the fail-safe click-through safety rules, the bus protocol, the **generic-only** rule (no per-model
salvage) — lives in THAT repo's `CLAUDE.md` + `TODO.md` + `Docs/DESIGN-*.md` (it has NO `STATUS.md`).
Model zoo: `Desktop\Avatars\`. Original design spec: under `C:\Users\SirKn\3d Avatar\`. From here,
the only coupling is the WebSocket bus protocol.

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
- **One-off Electron/Node probes** belong in the **Enigma Avatar window** repo (the Electron
  predecessor — the Unity repo has no `node_modules`); write
  the result to a file and `process.exit()`, then delete the probe. Running from a dir without
  `node_modules` (e.g. the scratchpad), piping stdout through another command, or relying on
  `app.quit()` makes Electron HANG or pop a blocking GUI error dialog in this non-interactive shell —
  this happened twice and landed an error dialog on the user.
- **Verify load-bearing numbers/line-refs with a direct tool call BEFORE relaying them** — never trust
  subagent audit output. Reports here claimed a "1600-char" line (the real max was 702) and line
  numbers off by ~100, and inflated an ASCII-rule count by conflating comments + on-screen text with
  actual terminal output. Measure, show the receipt. (See also: ground every load-bearing number.)
- **A refactor that deletes a module must also delete or guard its callers.** The Modkit-era
  "dissolve the monolith" refactor deleted 6 modules (vision/audio encoders, gguf, reasoning,
  sentiment, inference) but kept code importing them — 4 were crash-on-use landmines found only
  on 2026-07-13. `tests/test_import_integrity.py` now gates this; keep its allowlist honest.

## Project state docs
`CLEANUP_TRACKER.md`, `CODE_REVIEW.md`, `KNOWN_ISSUES.md`, `SUGGESTIONS.md`,
`AUDIT_2026-07-13.md` (trust-nothing audit: gates, fixes, doc-drift, majors triage, modkit
removal), plus `ULTRAREVIEW_2026-07-12.md` (whole-repo review: 80 verified findings). All 3 CRITICALS
FIXED 2026-07-13 (tests + ruff green): DPO/SimPO/KTO/ORPO prompt-mask off-by-one — the four
preference encoders now share `_encoded_prompt_len`, which drops the trailing EOS so the first
completion token is no longer masked out of the gradient; `extend_length.ps1` now RESUMES an
existing block-2048 run instead of warm-starting from step 0 and clobbering it (and
`pretrain_enigma.py` resume writes back to the checkpoint's own directory, not
`enigma_pretrain_base`); LoRA `cpu=` now tracks GPU presence instead of `offload_optimizer`,
so the default config no longer forces the model to CPU while batches go to CUDA. Majors and
minors from the review remain open — see the file.
2026-07-14 training-path fixes: `make_sft_data.py` no longer explodes a string-valued
`questions`/`answers` field into per-character records; the Trainer online-DPO `random.sample`
no longer sizes `k` off the unfiltered list (was a crash). Bottleneck stays SFT DATA at scale
(tool corpus ~29, identity 360, broad recall ~50%).
Reading rules: `CODE_REVIEW.md` is a closed-bug LEDGER — its present-tense entries are history,
not current state. (Removed 2026-07-14 as dead cruft: `FORGE_TEST_GUIDE.md`, `ENIGMA_QUANTIZE_PLAN.md`,
`AA code maker.md` (all Qwen-8B/Forge-GUI era), and `information/commands_reference.md` (documented the
deleted `[CMD]`/modkit command surface). `information/` still holds live docs + the code-referenced
`information/trainer/` prompts.)
