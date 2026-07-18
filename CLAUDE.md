# CLAUDE.md — Enigma Engine

Guidance for Claude Code working in this repo. Keep this file short (target <200 lines).
Harness-enforced rules (permissions, hooks, model) belong in `.claude/settings.local.json`, NOT here.

## What this is
This repo is **Enigma** — a **from-scratch** decoder-only LLM (its own architecture, BPE tokenizer base
vocab 4718, and weights; NOT a wrapper). Python. Pipeline is **pretrain → (facts continued-pretrain,
optional) → SFT → DPO → serve**; train + serve share one chat renderer
(`enigma_engine/core/chat_format.py`) so the prompt format can't drift.

**Identity ruling (user, 2026-07-16):** Enigma Engine IS Enigma — one AI, her own model, and the
machinery that trains and serves HER. The Forge-era framing (a generic engine to spawn different
AIs off copies of itself) is RETIRED — never describe this repo as a "model factory" or a
framework; the AI we made is the thing being worked on. The name arc stays visible in history
(ForgeEngine → EnigmaEngine `7ef9068f`; a Modkit refocus, dissolved 2026-07-14 `5bbb4b44`) and in
surviving identifiers (`ForgeConfig`, the FORGE trainer, `forge_config.json`,
`models/enigma_forge_tiny`) — those are identifiers, not identity; renaming them is optional
cleanup, and the git history STAYS (no rewrites — it is honest archaeology).

**Multimodal state (measured 2026-07-14):** Enigma is a TEXT decoder that will PERCEIVE
(image/audio INPUT) in-model. GENERATION (image/video/speech-audio) is a separate model family —
NOT this LLM's job; if wanted in-repo it is a bundled service, not the model painting pixels.
Perception is HALF-BUILT; text-only ships today:
- Ready: `forward_multimodal` + `vision_projection`/`audio_projection` (`core/model.py`), encoders
  `core/vision_encoder.py` + `core/audio_encoder.py`, Forge modes `train_vision`/`train_audio`.
- NOT trained yet (verified): the shipped `enigma_dpo` checkpoint has NO projection weights and
  `use_vision`/`use_audio` are off; `chat_format.py` has NO image/audio tokens.
- Plan is ROADMAP **Phase 4.5** (distill-then-align). Vision state 2026-07-17: distill DONE
  (`models/enigma_vision_distill/` — DINOv2-S -> her ViT-medium; student sees [-1,1], teacher
  ImageNet norm; the [-1,1] contract is TEST-PINNED in `tests/test_vision_normalization.py`);
  `align_vision.py` built, data staged (558k pairs), run PARKED by the training-last ruling.
  `serve --eyes` grafts the align checkpoint's encoder+projection onto served weights (missing
  ckpt = WARN + text-only). Audio: `collect_audio_data.py` collected LibriSpeech (28,539 pairs);
  `distill_audio_encoder.py` ready, not launched. Encoder persistence FIXED `f9ec5184`
  (checkpoints carry encoder state; resume refuses text-only ckpts). History: `KNOWN_ISSUES.md` #11.
- **TRAINING-LAST ruling (user, 2026-07-17):** all training runs (vision align, audio distill,
  any SFT/DPO cycle) are deferred to the END of the current arc. Vision image DOMAIN is the
  user's open decision (`VISION_QUALITY_SPEC.md` §4 — NOT the everyday-LLaVA diet).

The Modkit-era `mods/` + `plugins/` subsystem and its `commands`/`mod_tools`/`plugin_loader` registry
were REMOVED 2026-07-14 (never loaded by `serve_enigma.py`; superseded). Pip distribution renamed
`modkit` -> `enigma-engine`. Capabilities that are genuinely separate models (image gen, TTS, ASR,
search) return as hub/tool services, NOT in-repo mods.

**Organs (live 2026-07-14):** capabilities as local services behind serve flags; each is a
`core/` primitive with an injectable factory (tests never download models or touch hardware),
loaded eagerly at startup (a broken organ WARNs and text serving continues).
- `--voice` -> `core/tts.py` (pyttsx3/SAPI; one engine per JOB — say-then-save on one engine
  deadlocks, see module docstring): intent-gated `speak` built-in + `/v1/audio/speech` (WAV) +
  `/v1/audio/voices`. `speak` = server speakers; `avatar_say` stays a CLIENT tool.
  **Voice PARKED by user ruling (2026-07-16): launchers start voiceless** — the chat page
  degrades to "voice: off"; revisit "later when it matters." When it lifts: `Start-Enigma.ps1
  -Voice` serves `--voice-name zira` (user dislikes the stock David voice), and the wanted
  real fix is the Kokoro-82M swap (BACKLOG §5, ~330 MB download, needs the user's go-ahead).
- `--ears` -> `core/asr.py` (faster-whisper, cuda->cpu fallback): `/v1/audio/transcriptions`.
- `--eyes` -> `core/eyes.py` (**her OWN captioner since 2026-07-17** — aligned encoder + grafted
  projection + the served model; BLIP DELETED): OpenAI image_url content in chat is captioned to
  `[image: ...]` text before gates/memory/render (`flatten_image_content`, data: URLs only,
  honest markers when it can't see) + `/v1/images/describe`. Degrades text-only until the
  align run produces a checkpoint (training-last).
- `--image-gen` -> `core/imagegen.py` (diffusers sd-turbo, 1-step): intent-gated `imagine`
  built-in (PNGs land in `~/.enigma_engine/images/`) + `/v1/images/generations` (b64_json).
Verified end-to-end vs served `enigma_dpo`: "Say hello out loud." -> speak call -> SAPI audio;
voice->wav->ears and imagine->png->eyes loops pass on real weights.

**Enigma Avatar** (the desktop overlay an LLM can drive) was split into a **separate sibling
repo** at `C:\Users\SirKn\Enigma Avatar\` on 2026-06-28 (full history preserved) and is now a
**Unity 6 rebuild** — the Electron+three.js predecessor lives at `C:\Users\SirKn\Enigma Avatar
window\` (still runnable, maintenance-only). The two meet only at the local WebSocket bus
(`ws://127.0.0.1:8765`). Work on the avatar in THAT repo, not here.

## Setup / build / test — run these first
- Python 3.12 (`C:\Users\SirKn\AppData\Local\Programs\Python\Python312\python.exe`).
- Enigma tests: `python -m pytest tests/ -q` — use the system Python above or
  `venv\Scripts\python.exe`; NOT `.venv\` (no pytest installed there).
- **NO ruff (user ruling 2026-07-18): do not run ruff or make ruff-appeasement edits.**
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
  (`Start-Enigma.bat` and `Launch Enigma.bat` were deleted 2026-07-17 as superseded;
  the `enigma`/`enigma-ai` console scripts now default to the adopted DPO model too.)
  Current adopted weights = **v8** (2026-07-16, SHA256
  `A11DB8F0...`, 79/90 on the 90-probe gate — first to pass all 7 categories); receipted backup
  at `Enigma Backups\enigma_dpo_v8_adopted\`. Revert targets: v5 at `enigma_dpo_v5_adopted\`,
  older v1 at `enigma_dpo_adopted\`, or `models/enigma_sft`. Backups hold model+config+vocab .sha256.
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
- **PS 5.1 `Start-Process -ArgumentList` does NOT quote arguments** — the space in "Enigma
  Engine" split a `-File` path and the tray silently failed to launch (2026-07-16). Pass one
  pre-quoted string (backtick-escaped quotes), or launch through the .bat wrappers, which quote
  correctly.
- **A process-search can match your own shell** — `Get-CimInstance ... CommandLine -like
  '*Enigma-Tray*'` matched the diagnostic PowerShell carrying the search string; two "tray pids"
  were self-matches (2026-07-16). Exclude `$PID` or match on the exact `-File` path.
- **Launch user-facing long-lived processes DETACHED** (Start-Process), never as harness
  background tasks — a chat window launched as a background task was tethered to the Claude
  session and read to the user as a stuck task (2026-07-16).
- **Runtime state files must be CWD-independent, and helper threads handed to
  `webview.start` are NON-daemon.** A relative `Path("data")/mute_state.json` silently
  breaks mute persistence for console-script servers started elsewhere, and a
  never-give-up poll loop in the window shim leaks a ghost `pythonw.exe` when the window
  closes first (both found by the 2026-07-17 adversarial audit of the previous day's
  fixes). Anchor state at the repo dir or `Path.home()/".enigma_engine"`; daemonize or
  bound loops that outlive their window.
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
(tool corpus 544 generated records incl. speak/imagine, identity 426). Recall strategy changed
2026-07-15: facts INSTALL via a continued-pretrain pass (`make_facts_pretrain_data.py` ->
`pretrain_enigma.py --tokens-bin` -> `models/enigma_pretrain_facts`; SFT inits from it and
`knowledge_corpus.py` x5 only SURFACES them) — measured factual 13/20 -> 19/20 on the 90-probe
suite. The general diet is 105,203 short pairs (count receipted in `BACKLOG.md` §3; produced by
collect_finetuning_data.py — per-source length caps DIFFER, see training_guide.md Stage 2).
Reading rules: `CODE_REVIEW.md` is a closed-bug LEDGER — its present-tense entries are history,
not current state. (Removed 2026-07-14 as dead cruft: `FORGE_TEST_GUIDE.md`, `ENIGMA_QUANTIZE_PLAN.md`,
`AA code maker.md` (all Qwen-8B/Forge-GUI era), and `information/commands_reference.md` (documented the
deleted `[CMD]`/modkit command surface). `information/` still holds live docs + the code-referenced
`information/trainer/` prompts.)
