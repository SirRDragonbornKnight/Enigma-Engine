# CLAUDE.md — Enigma Engine

Guidance for Claude Code working in this repo. Keep this file short (target <200 lines).
Harness-enforced rules (permissions, hooks, model) belong in `.claude/settings.local.json`, NOT here.

## What this is
This repo is **Enigma** — a **from-scratch** decoder-only LLM (own architecture, own BPE tokenizer
at base vocab 4718, own weights; NOT a wrapper). Python. Pipeline: **pretrain → (facts
continued-pretrain, optional) → SFT → DPO → serve**; train and serve share one chat renderer
(`enigma_engine/core/chat_format.py`) so the prompt format cannot drift.

**Identity ruling (user, 2026-07-16):** Enigma Engine IS Enigma — one AI, her own model, and the
machinery that trains and serves HER. The Forge-era framing (a generic engine to spawn different
AIs off copies of itself) is RETIRED — **never describe this repo as a "model factory" or a
framework**; the AI we made is the thing being worked on. Surviving names (`ForgeConfig`,
`forge_config.json`, `models/enigma_forge_tiny`) are identifiers, not identity — renaming is
optional and the git history STAYS (honest archaeology). The dormant FORGE trainer itself was
DELETED in the 2026-07-18 compression pass (~19k lines; see `CLEANUP_TRACKER.md`).
(`serve --persona` serves a DIFFERENT AI from a persona pack with her own data home —
that is the trainer molding another mind, not Enigma changing hers.)

**Multimodal state:** Enigma is a TEXT decoder that PERCEIVES (image/audio INPUT) in-model;
GENERATION is a separate model family — a bundled service, never this LLM painting pixels.
Vision perception is LIVE: `serve --eyes` grafts the align checkpoint's encoder+projection
onto the served weights at boot (missing checkpoint = WARN + text-only); the SHIPPED align
checkpoint predates encoder-config persistence, so it loads through the `--eyes-preset`
fallback (default `medium`) until the next align run rewrites it. Native audio input is IN
PROGRESS — LibriSpeech collected, the distill teacher download is the blocker. Encoders and
the shared align trainer: `core/vision_encoder.py`, `core/audio_encoder.py`,
`training/encoder_align.py` (entry points `align_vision.py` / `align_audio.py`). Step order
and status: `ROADMAP.md` Phase 4.5. Vision image DOMAIN is the user's open decision
(`VISION_QUALITY_SPEC.md` §4 — NOT the everyday-LLaVA diet).
**Training runs are PERMITTED but SEQUENCED** (ruled 2026-07-24): everything that trains
waits and runs as one consolidated block in the order `BACKLOG.md` §7.95 sets. So "may I
train?" = yes-when-its-turn-comes. Ask-before-hot-runs courtesy applies while the user is
connected/gaming.

The Modkit-era `mods/`+`plugins/` subsystem and its registry were REMOVED 2026-07-14 (never
loaded by serve; pip name is `enigma-engine` 2.0.0). Capabilities that are genuinely separate
models (image gen, TTS, ASR, search) return as organ/tool services below, NOT in-repo mods.

**Organs (live 2026-07-14; COMPLETE-BY-DEFAULT ruled 2026-07-27):** capabilities as local
services behind serve flags; each is a `core/` primitive with an injectable factory (tests
never download models or touch hardware), loaded eagerly at startup (a broken organ WARNs
and text serving continues). The user-facing launcher boots ALL of them — "the old way of
having the tools separate was to save space; add them in so she is complete" (user, verbatim).
Flags still exist for scratch/eval servers, which stay lean on purpose.
- `--voice` → `core/tts.py` (**Kokoro-82M**): intent-gated `speak` built-in + the
  `/v1/audio/*` endpoints; voices are style tensors that BLEND by weighted sum; the active
  recipe persists to `~/.enigma_engine/voice.json`. **The server must run under the repo
  `venv/`** — it is the only interpreter with kokoro installed, and Start-Enigma points
  there. Talk-mode PERSISTS in `data/talk_mode.json` and is re-read at boot: turn narration
  on once and every later launch narrates — no launcher resets it.
- `--ears` → `core/asr.py` (faster-whisper, cuda→cpu fallback): `/v1/audio/transcriptions`.
- `--eyes` → `core/eyes.py` (**her OWN captioner** — aligned encoder + grafted projection +
  the served model): OpenAI image_url content in chat is captioned to `[image: ...]` text
  before gates/memory/render (data: URLs only, honest markers when it can't see), plus
  `/v1/images/describe`. The chat page can upload a picture to her.
- `--image-gen` → `core/imagegen.py` (diffusers sd-turbo, 1-step): intent-gated `imagine`
  built-in (PNGs land in `~/.enigma_engine/images/`) + `/v1/images/generations`.
Verified end-to-end vs served weights: speak → Kokoro audio; voice→wav→ears and
imagine→png→eyes loops pass. `/v1/capabilities` reports which organs a server booted with.

## Setup / build / test — run these first
- Python 3.12 (`C:\Users\SirKn\AppData\Local\Programs\Python\Python312\python.exe`).
- Enigma tests: `python -m pytest tests/ -q` — use the system Python above or
  `venv\Scripts\python.exe`; NOT `.venv\` (a stub; no pytest). Suite baseline lives in
  `CLEANUP_TRACKER.md`.
- **NO ruff (user ruling 2026-07-18): do not run ruff or make ruff-appeasement edits.**
- The 2026-07-18 compression pass is COMMITTED (`b02bc297`; record in `CLEANUP_TRACKER.md`).
  Do not "restore" deleted modules on the assumption their removal was an accident, and do
  not commit unbidden.
- If a fresh session can't run the tests from this section, fix THIS section first.

## The pipeline
- **Pretrain** — `python pretrain_enigma.py` (trains from scratch on the memmapped token
  corpus). Resume with `resume_training.ps1`; watch with `tail_training_log.ps1`.
- **SFT data** — `python make_sft_data.py` → writes `data/sft/{tool_calls,identity,mix}.jsonl`.
- **Finetune (SFT)** — `python finetune_enigma.py` (base checkpoint → instruct/tool model;
  imports the optimizer/LR "arsenal" from `enigma_engine.core.optim`, shared with pretrain).
- **DPO** — `python make_dpo_data.py` → `data/sft/dpo_pairs.jsonl`, then `python dpo_enigma.py`
  (policy + frozen reference; masks via `chat_format.render_training`). Defaults are the
  adopted-safe ones (`--lr 5e-7 --epochs 1`) — at 182M, lr 2e-6 x2 epochs measurably WRECKED
  her (identity 83→50%, factual 50→0%). DPO here is a nudge or a wrecking ball; do not raise
  the lr without a scorecard.
- **Eval** — `python eval_behavior.py` = the behavior gate, run against a RUNNING server.
  Probe counts, seal history, and every scorecard number live in `EVAL_REDESIGN.md` — that
  file owns the numbers; do not restate them here. Operational rules: `--base-url` defaults
  to the SCRATCH port 8123 (deliberately not the live 8000) and `--temperature` to 0.0; the
  run CLEARS the target server's memory store, so it REFUSES any target off the scratch port
  unless `--allow-live-server` says the target is disposable. Dev probes:
  `data/eval/behavior_probes.jsonl`; the LOCKED set is SEALED — re-sealed at run start, an
  edited holdout is refused rather than scored.
- **Teach** — `python teach_enigma.py` chats against a running serve; `/fix` bakes a
  correction into `teachings.jsonl` + `teach_pairs.jsonl` (both gitignored — personal), with
  confirm-before-bake augmentation. `teachings.jsonl` is the USER's authoring channel: never
  delete it; it is still the untouched example file until they write in it.
- **Serve** — `python serve_enigma.py` (OpenAI-compatible FastAPI server; loads the `.pth`
  checkpoint directly). Generation runs **bf16 autocast + TF32** on CUDA (`--fp32` = escape
  hatch; parity receipt 2026-07-17: the then-current 90-probe dev gate scored identically
  under both). Import-safe since 2026-07-18: startup lives in `boot()`; a mounted-but-
  unbooted app answers 503, and `tests/test_serve_enigma.py` covers the live paths.

## Conventions / guardrails
- **Console output must be ASCII** — the Windows cp1252 console hard-crashes on unmapped
  chars (`→`); the rule is ZERO non-ASCII in console-bound strings
  (print/logger/raise/argparse/_emit_progress). TEST-GATED by `tests/test_repo_hygiene.py`
  (AST sweep over the console sinks; comments/docstrings stay out of scope).
- **Do not change the live pretrain defaults** (`--optimizer adamw --schedule cosine`) — they
  are asserted bit-identical to the live training lineage. Muon / WSD are future-run-only,
  behind flags.
- Checkpoints rotate `latest.pth` → `prev.pth` atomically with a finite-loss guard; resume
  rebuilds config from the checkpoint and hard-fails on any arch/optimizer mismatch.
- Pretraining **DONE 2026-07-03**: full 287,882 steps / 56.6B tokens, val ppl 3.5
  (`models/enigma_pretrain_large/model.pth`; SHA256 receipts + the training logs live in
  `Enigma Backups\`, not beside the checkpoint). Lineage is immutable — the v1 corpus
  (`data/pretrain/tokens.bin` + sidecar) is filesystem read-only AND pretokenize refuses to
  write that path. Forward plan is `ROADMAP.md`; bottleneck is SFT data, not compute.
- SFT+DPO **ADOPTED**: every entry point serves `models/enigma_dpo/model.pth`; the launcher
  chain adds `--memory-dir data\memory` (bare console scripts default memory OFF). The
  user-facing chain is **Talk to Enigma.bat / Enigma Tray.bat / Stop Enigma.bat** (Desktop
  wrappers → repo scripts → `Start-Enigma.ps1` → serve; her window is `enigma_window.py`).
  Current adopted weights = **v8** (2026-07-16); the live gate number is the sealed-set
  baseline in `EVAL_REDESIGN.md`. Adoption receipt, backups and revert targets: `BACKLOG.md`.
- From-scratch ethos: prefer fresh, correct code; engines should fail honestly ("feature
  absent") rather than guess.

## Enigma Avatar — now a separate repo
The desktop overlay is its own repo at `C:\Users\SirKn\Enigma Avatar\` (2026-06-28, history
preserved), rebuilt in **Unity 6**; the Electron predecessor at `C:\Users\SirKn\Enigma Avatar
window\` is maintenance-only (do not break it). Work on the avatar THERE, not here — its guidance
(fail-safe click-through rules, bus protocol, the **generic-only** rule) lives in that repo's
`CLAUDE.md` + `TODO.md` + `Docs/DESIGN-*.md` (it has NO `STATUS.md`). Model zoo: `Desktop\Avatars\`.
From here the ONLY coupling is the WebSocket bus (`ws://127.0.0.1:8765`).

## Working style
- "Make a plan first" means present the plan and **stop for approval** — don't build it in the same pass.
- Scope to exactly what's asked; deliver small, verify, then continue.
- **Fix in place, don't compensate.** When existing code is wrong, CHANGE it — don't bolt on shims,
  wrapper layers, fallback branches or parallel implementations to work around it. Compensating code
  leaves two versions of the truth to drift apart. Edit the source of the problem.

## Gotchas (mistakes made here — don't repeat them)
- **No C++ build toolchain on this box.** Only adopt deps that ship PREBUILT wheels/binaries, and
  verify BEFORE installing — a source build will fail here.
- **Verify load-bearing numbers/line-refs with a direct tool call BEFORE relaying them** — never trust
  subagent audit output. Reports here claimed a "1600-char" line (the real max was 702) and line
  numbers off by ~100, and inflated an ASCII-rule count by conflating comments + on-screen text with
  actual terminal output. Measure, show the receipt.
- **Launcher/PowerShell trio (all cost a round-trip on 2026-07-16).** PS 5.1 `Start-Process
  -ArgumentList` does NOT quote args — the space in "Enigma Engine" split a `-File` path and the
  tray silently failed; pass one pre-quoted string or go through the .bat wrappers. A process
  search can match YOUR OWN shell — exclude `$PID` or match the exact `-File` path. And launch
  user-facing long-lived processes DETACHED, never as harness background tasks. ALSO: PS 5.1
  `Set-Content/Out-File -Encoding utf8` writes a BOM — shell string round-trips BOM-stamped four
  source files during mutation testing (2026-07-26, caught by the hygiene gates); edit files with
  byte-clean tools, never shell read-modify-write.
- **Runtime state files must be CWD-independent; helper threads handed to `webview.start` are
  NON-daemon.** A relative `Path("data")/mute_state.json` silently broke mute persistence for
  console-script servers started elsewhere, and a never-give-up poll loop leaked a ghost
  `pythonw.exe`. Anchor state at the repo dir or `Path.home()/".enigma_engine"`; daemonize or
  bound any loop that outlives its window.
- **A refactor that deletes a module must also delete or guard its callers.** The Modkit-era
  refactor deleted 6 modules but kept code importing them — 4 were crash-on-use landmines found
  only on 2026-07-13. `tests/test_import_integrity.py` gates this (AST-based; relative imports
  are a known blind spot); keep its allowlist honest.
- **Every fix pass gets its own adversarial re-audit, and every new regression test gets
  mutation-verified** (reintroduce the bug, watch the test fail, revert). The 2026-07-17/18
  test-suite audit ran 5 rounds to convergence; every round's fixes shipped smaller defects
  than they fixed — the pattern has held in every arc since.

## Project state docs — the complete index
- `VISION.md` — the destination (all-purpose companion, own runtime, research mode,
  devices) and the order it lands in; written from the user's own words 2026-07-27.
- `BACKLOG.md` — consolidated worklist; §7.95 owns the training-block execution order.
- `ROADMAP.md` — phase plan; Phase 4.5 organ steps; Phase 7 verdicts.
- `EVAL_REDESIGN.md` — eval design, seal history, and ALL scorecard/probe numbers.
- `PHASE7_GATE.md` — measured ceilings + the remake charter.
- `TOKENIZER_V2_SPEC.md` — v2 vocab + corpus receipts.
- `VISION_QUALITY_SPEC.md` — vision data quality; §4 image domain is an open user decision.
- `KNOWN_ISSUES.md` — honest-gap ledger.
- `SUGGESTIONS.md` — 2026 landscape research + principles.
- `CLEANUP_TRACKER.md` — the tree as it stands + suite baseline + deletion records.
- `_archive/` — closed ledgers (present-tense entries there are history, not current state).

**Data state:** bottleneck stays SFT DATA at scale — tool corpus 530 records, identity 422,
mix 114,244, dpo_pairs 240, general diet `data/finetune/combined_finetune.jsonl` 105,203
short pairs (counts re-verified 2026-07-28 after the reseal-#7 rebuild;
`EVAL_REDESIGN.md` owns them). Recall strategy since 2026-07-15: facts INSTALL
via a continued-pretrain pass (`make_facts_pretrain_data.py` → `pretrain_enigma.py
--tokens-bin`), SFT only SURFACES them — measured factual 13/20 → 19/20. The eval leak
guard's enforcement design (what refuses vs what reports, at build and at consume time) is
`EVAL_REDESIGN.md`'s to explain; the one path with no consume-time guard is
`pretokenize_data.py`, which is why `make_pretrain_curated.py` screens at build time.
