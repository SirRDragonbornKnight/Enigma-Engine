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

**Multimodal state:** Enigma is a TEXT decoder that PERCEIVES (image/audio INPUT) in-model;
GENERATION is a separate model family — a bundled service, never this LLM painting pixels.
Vision perception is LIVE (`serve --eyes`, 2026-07-20); native audio is in progress:
- Ready: `forward_multimodal` + `vision_projection`/`audio_projection` (`core/model.py`), encoders
  `core/vision_encoder.py` + `core/audio_encoder.py`, and the vision-align trainer
  (`enigma_engine/training/encoder_align.py` — one hardened `_train_encoder` core with
  `train_vision` + `train_audio` wrappers; carved from the deleted Forge trainer 2026-07-18,
  hardened + generalized 2026-07-19; entry points `align_vision.py` / `align_audio.py`).
- The shipped `enigma_dpo` checkpoint itself has NO projection weights (`use_vision`/`use_audio`
  off; `chat_format.py` has no image/audio tokens) — eyes work by GRAFTING the align
  checkpoint's encoder+projection at serve time; her text weights are untouched.
- Plan is ROADMAP **Phase 4.5** (distill-then-align). Vision state 2026-07-17: distill DONE
  (`models/enigma_vision_distill/` — DINOv2-S -> her ViT-medium; student sees [-1,1], teacher
  ImageNet norm; the [-1,1] contract is TEST-PINNED in `tests/test_vision_normalization.py`);
  `align_vision.py` run COMPLETE 2026-07-20 (val 1.4884). Align checkpoints persist the
  encoder config since `d15bc6c`, but the SHIPPED checkpoint predates that commit and has
  no `vision_encoder_config` key -- `serve --eyes` loads it through the `--eyes-preset`
  fallback (default `medium`) until the next align run rewrites it.
  `serve --eyes` grafts the align checkpoint's encoder+projection onto served weights (missing
  ckpt = WARN + text-only). Audio: LibriSpeech collected (28,539 pairs), `distill_audio_encoder.py`
  ready but NOT launched (gated on downloading the `openai/whisper-base` teacher — the cached
  Systran/faster-whisper-base is the ASR organ, NOT the teacher); the audio ALIGN trainer runs
  on the shared encoder-align core (`train_audio` + `align_audio.py`; mask-aware encoder since
  2026-07-20, `--batch-size 8` works — padded-batch==unbatched at 3.6e-7). Encoder persistence FIXED
  `f9ec5184`, re-locked for audio 2026-07-19. History: `KNOWN_ISSUES.md` #11.
- **TRAINING-LAST ruling LIFTED by user 2026-07-20** ("gpu usage is fine") — training runs are
  allowed again; ask-before-hot-runs courtesy still applies while the user is connected/gaming.
  Vision image DOMAIN is still the user's open decision (`VISION_QUALITY_SPEC.md` §4 — NOT the
  everyday-LLaVA diet).

The Modkit-era `mods/`+`plugins/` subsystem and its `commands`/`mod_tools`/`plugin_loader` registry
were REMOVED 2026-07-14 (never loaded by serve; pip name is now `enigma-engine` 2.0.0). Capabilities
that are genuinely separate models (image gen, TTS, ASR, search) return as hub/tool services below,
NOT in-repo mods.

**Organs (live 2026-07-14):** capabilities as local services behind serve flags; each is a
`core/` primitive with an injectable factory (tests never download models or touch hardware),
loaded eagerly at startup (a broken organ WARNs and text serving continues).
- `--voice` -> `core/tts.py` (**Kokoro-82M since 2026-07-23**; one worker thread serializes
  synth/play/recipe, playback is interruptible sentence-by-sentence): intent-gated `speak`
  built-in + `/v1/audio/speech` (WAV) + `/v1/audio/voices` + `/v1/audio/stop`,
  `/v1/audio/talk-mode`, `/v1/audio/status`, `/v1/audio/voice`. `speak` = server speakers;
  `avatar_say` stays a CLIENT tool. Voices are style tensors that BLEND by weighted sum; the
  active recipe persists to `~/.enigma_engine/voice.json`. **The server must run under the repo
  `venv/`** — it is the only interpreter with kokoro installed, and Start-Enigma points there.
  Talk-mode PERSISTS in `data/talk_mode.json` and is re-read at boot: she starts silent
  today only because that file does not exist yet. Turn narration on once and every later
  launch narrates -- no launcher resets it.
- `--ears` -> `core/asr.py` (faster-whisper, cuda->cpu fallback): `/v1/audio/transcriptions`.
- `--eyes` -> `core/eyes.py` (**her OWN captioner since 2026-07-17** — aligned encoder + grafted
  projection + the served model; BLIP DELETED): OpenAI image_url content in chat is captioned to
  `[image: ...]` text before gates/memory/render (`flatten_image_content`, data: URLs only,
  honest markers when it can't see) + `/v1/images/describe`. Eyes verified LIVE 2026-07-20
  ("eyes: on" at boot); degrades text-only only if the align checkpoint is missing.
- `--image-gen` -> `core/imagegen.py` (diffusers sd-turbo, 1-step): intent-gated `imagine`
  built-in (PNGs land in `~/.enigma_engine/images/`) + `/v1/images/generations` (b64_json).
Verified end-to-end vs served `enigma_dpo`: "Say hello out loud." -> speak call -> Kokoro audio;
voice->wav->ears and imagine->png->eyes loops pass on real weights.

## Setup / build / test — run these first
- Python 3.12 (`C:\Users\SirKn\AppData\Local\Programs\Python\Python312\python.exe`).
- Enigma tests: `python -m pytest tests/ -q` — use the system Python above or
  `venv\Scripts\python.exe`; NOT `.venv\` (no pytest installed there).
- **NO ruff (user ruling 2026-07-18): do not run ruff or make ruff-appeasement edits.**
- **The 2026-07-18 compression pass is COMMITTED** (`b02bc297`, on user order 2026-07-19,
  together with the vision-align checkpoint-safety arc). The suite dropping **574 → 349** was
  that pass deleting the dormant stack's own tests, NOT lost coverage (**555** as of
  2026-07-20 after the v2-prep arcs); the served checkpoint was verified byte-identical
  before/after. Do not "restore" deleted modules on the assumption their removal was an
  accident, and do not commit unbidden.
- (Avatar tests live in the Enigma Avatar repo and are gated there — see that section below.)
- If a fresh session can't run the tests from this section, fix THIS section first.

## The pipeline
- **Pretrain** — `python pretrain_enigma.py` (trains from scratch on the memmapped token corpus).
  Resume the live run with `resume_training.ps1`; watch it with `tail_training_log.ps1`.
- **SFT data** — `python make_sft_data.py` → writes `data/sft/{tool_calls,identity,mix}.jsonl`.
- **Finetune (SFT)** — `python finetune_enigma.py` (base checkpoint → instruct/tool model;
  imports the optimizer/LR "arsenal" from `enigma_engine.core.optim`, shared with pretrain).
- **DPO** — `python make_dpo_data.py` → `data/sft/dpo_pairs.jsonl`, then `python dpo_enigma.py`
  (policy + frozen reference; masks via `chat_format.render_training`). This STANDALONE script is
  the real DPO path. Defaults are now the adopted-safe ones (`--lr 5e-7 --epochs 1`) — at 182M,
  lr 2e-6 x2 epochs measurably WRECKED her (identity 83→50%, factual 50→0%). DPO here is a nudge
  or a wrecking ball; do not raise the lr without a scorecard.
- **Eval** — `python eval_behavior.py` = the behavior gate, run against a RUNNING server
  (dev set widened to 113 probes 2026-07-20; `--probes` selects a probe file; the scorecard
  prints probe file + decode config). `--base-url` defaults to `http://127.0.0.1:8123` (a
  SCRATCH port, deliberately not the live 8000) and `--temperature` to 0.0 (true greedy,
  reproducible). Dev probes: `data/eval/behavior_probes.jsonl`; the LOCKED set is the user's
  to author blind (`data/eval/LOCKED_PROBES_AUTHORING.md`), still absent by design.
- **Teach** — `python teach_enigma.py` chats against a running serve; `/fix` bakes a correction
  into `teachings.jsonl` + `teach_pairs.jsonl` (both gitignored — personal), with
  confirm-before-bake augmentation. `teachings.jsonl` is the USER's authoring channel: never
  delete it, and it is still the untouched example file until they write in it.
- **Serve** — `python serve_enigma.py` (OpenAI-compatible FastAPI server; loads the `.pth`
  checkpoint directly). Generation runs **bf16 autocast + TF32** on CUDA since 2026-07-17
  (`--fp32` = full-fp32 escape hatch, disables both; 90-probe gate re-measured 79/90 under
  bf16, same as fp32). Run with `--help` for flags. Since 2026-07-18 the module is
  **import-safe**: startup lives in `boot()` (called by `main()`); a mounted-but-unbooted app
  answers 503 via middleware, and `tests/test_serve_enigma.py` covers the live paths
  (stream parity, mute, intent gates, train/serve system-shape byte-parity).

## Conventions / guardrails
- **Console output must be ASCII** — the Windows cp1252 console hard-crashes on unmapped
  chars (`→`); em dashes happen to map but the rule is ZERO non-ASCII in console-bound
  strings (print/logger/raise/argparse/_emit_progress). Since 2026-07-18 this is TEST-GATED
  (`tests/test_repo_hygiene.py` AST sweep over the console sinks — its first run caught 9
  arrow lines three manual sweeps had missed; comments/docstrings stay out of scope).
- **Do not change the live pretrain defaults** (`--optimizer adamw --schedule cosine`) — they are
  asserted bit-identical to the live training lineage. Muon / WSD are future-run-only, behind flags.
- Checkpoints rotate `latest.pth` → `prev.pth` atomically with a finite-loss guard; resume rebuilds
  config from the checkpoint and hard-fails on any arch/optimizer mismatch.
- Pretraining **DONE 2026-07-03**: full 287,882 steps / 56.6B tokens, val ppl 3.5
  (`models/enigma_pretrain_large/model.pth`; SHA256 receipts live in
  `Enigma Backups\enigma_pretrain_large_final\`, not beside the checkpoint). Lineage is
  immutable; forward plan is `ROADMAP.md`. Bottleneck is now SFT data, not compute.
- SFT+DPO **ADOPTED**: every entry point serves `models/enigma_dpo/model.pth`; the launcher
  chain adds `--memory-dir data\memory` (bare `enigma`/`enigma-ai` console scripts default
  memory OFF — `--memory-dir` default is None). The user-facing chain (since 2026-07-16) is
  **Talk to Enigma.bat / Enigma Tray.bat / Stop Enigma.bat** (Desktop wrappers -> repo
  scripts -> `Start-Enigma.ps1` -> serve; her window is `enigma_window.py`).
  Current adopted weights = **v8** (2026-07-16, SHA256 verified 2026-07-19 as
  `A11DB8F04CC63C74...`, 79/90 on the 90-probe gate — first to pass all 7 categories); backup
  at `Enigma Backups\enigma_dpo_v8_adopted\`. Revert targets: v5 at `enigma_dpo_v5_adopted\`,
  older v1 at `enigma_dpo_adopted\`, or `models/enigma_sft`. Backups hold model+config+vocab .sha256.
- From-scratch ethos: prefer fresh, correct code; engines should fail honestly ("feature absent")
  rather than guess.

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
  verify BEFORE installing — a source build will fail here. (Node/Electron probe lessons moved to
  the avatar repos, which own that surface.)
- **Verify load-bearing numbers/line-refs with a direct tool call BEFORE relaying them** — never trust
  subagent audit output. Reports here claimed a "1600-char" line (the real max was 702) and line
  numbers off by ~100, and inflated an ASCII-rule count by conflating comments + on-screen text with
  actual terminal output. Measure, show the receipt. (See also: ground every load-bearing number.)
- **Launcher/PowerShell trio (all cost a round-trip on 2026-07-16).** PS 5.1 `Start-Process
  -ArgumentList` does NOT quote args — the space in "Enigma Engine" split a `-File` path and the
  tray silently failed; pass one pre-quoted string or go through the .bat wrappers. A process
  search can match YOUR OWN shell (`CommandLine -like '*Enigma-Tray*'` matched the diagnostic
  PowerShell carrying the string) — exclude `$PID` or match the exact `-File` path. And launch
  user-facing long-lived processes DETACHED, never as harness background tasks, or they tether to
  the session and read to the user as a stuck task.
- **Runtime state files must be CWD-independent; helper threads handed to `webview.start` are
  NON-daemon.** A relative `Path("data")/mute_state.json` silently broke mute persistence for
  console-script servers started elsewhere, and a never-give-up poll loop leaked a ghost
  `pythonw.exe`. Anchor state at the repo dir or `Path.home()/".enigma_engine"`; daemonize or
  bound any loop that outlives its window.
- **A refactor that deletes a module must also delete or guard its callers.** The Modkit-era
  "dissolve the monolith" refactor deleted 6 modules (vision/audio encoders, gguf, reasoning,
  sentiment, inference) but kept code importing them — 4 were crash-on-use landmines found only
  on 2026-07-13. `tests/test_import_integrity.py` gates this (AST-based since 2026-07-18, incl.
  from-lists and exec()-string imports; relative imports are a known blind spot); keep its
  allowlist honest.
- **Every fix pass gets its own adversarial re-audit, and every new regression test gets
  mutation-verified** (reintroduce the bug, watch the test fail, revert). The 2026-07-17/18
  test-suite audit ran 5 rounds to convergence; every round's fixes shipped smaller defects
  than they fixed — the pattern held all five times.

## Project state docs
`CLEANUP_TRACKER.md` (incl. the 2026-07-18 compression-pass record), `KNOWN_ISSUES.md`,
`SUGGESTIONS.md`, `BACKLOG.md`. Point-in-time review dumps live in `_archive/`
(`ULTRAREVIEW_2026-07-12.md`, `AUDIT_2026-07-13.md`, `CODE_REVIEW.md` — closed-bug LEDGERS;
present-tense entries there are history, not current state). The old ultrareview's dormant-arsenal
majors were RESOLVED BY DELETION 2026-07-18 (the code they were wrong in is gone — see
`BACKLOG.md` §2). #14 (non-SDPA rectangular-decode mask) was the one that survived as real
code, and it was FIXED the same day — it turned out to be a loud broadcast crash, not silent
corruption, unreachable from the live serve loop; regression tests in
`tests/test_cpu_rectangular_decode.py`.

**Data state (counted 2026-07-19, all verified):** bottleneck stays SFT DATA at scale — tool
corpus 544 records (incl. speak/imagine), identity 426, mix 114,316, dpo_pairs 242, general diet
`data/finetune/combined_finetune.jsonl` 105,203 short pairs (per-source length caps DIFFER — see
`information/trainer/training_guide.md` Stage 2). Recall strategy since 2026-07-15: facts INSTALL
via a continued-pretrain pass (`make_facts_pretrain_data.py` → `pretrain_enigma.py --tokens-bin`
→ `models/enigma_pretrain_facts`; SFT inits from it, and `knowledge_corpus.py` x5 only SURFACES
them) — measured factual 13/20 → 19/20 on the 90-probe suite. The eval leak guard
(`eval_leak_guard.py`) is wired into `make_sft_data` but stays a NO-OP until a human authors
`data/eval/locked_probes.jsonl` and seals it — still absent, by design (separation of powers).
