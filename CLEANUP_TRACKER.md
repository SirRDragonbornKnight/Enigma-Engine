# Cleanup Tracker — reset 2026-06-11; COMPRESSION PASS 2026-07-18; ART PASS 2026-07-26

This tracker describes the tree as it stands today; earlier eras (`gui/`,
`web/`, `services/`, `api/`, the Qwen-era `engine_*`/`inference`/`rag`
modules, and the Forge training arm removed 2026-07-18) live in git history.
Current truth:

## 2026-07-26 cleaning pass (four survey agents, every claim verified)

- **v1 lineage training logs PRESERVED first:** all root training logs were
  untracked AND gitignored, so git was never their archive -- and the three
  `train_resume.out.*` side-logs held the ONLY record of steps ~41k-51k
  (measured: `train_large.log` provably lacks that band). The full set now
  lives at `Enigma Backups\training_logs_v1\` with a SHA256SUMS receipt.
  `train_large.log` STAYS in the root (live consumers: `training_progress.py`,
  `tail_training_log.ps1`); the four backed-up `train_resume.out*` side-logs,
  the abandoned 121M `train_base_v2.log`, the SFT receipts and the pre-v1
  `_merge.log` (a LoRA merge from the era whose code was deleted 2026-07-18)
  were removed locally.
- **Deleted without backup (no value):** the zero-byte `*.err.log` stubs
  (EXCEPT `serve_enigma.err.log`, which stays -- it is Start-Enigma.ps1's
  live -RedirectStandardError sink and empty is its healthy state), the
  4-line `serve_sft.log` banner, the one-line `power_guardian.log`, the 12
  `logs\forge_*.log` from the deleted trainer, 16 zero-byte run-log stubs
  inside `models\enigma_sft\`, six stray `*.bak`, all `__pycache__` (two
  interpreter generations of dead bytecode in `tests\` alone), and 20 empty
  directories (incl. the 8 `models\probe_*` husks of an aborted 07-21 sweep).
- **Untracked from git:** the `orchestration-workflow\` + `.claude` weather-*
  demo scaffold (8 files, 2026-06-24) -- an unrelated demo that was 8 of the
  repo's 189 tracked files; recoverable from history.
- **Deps trued up:** PyYAML dropped (core dep, zero imports tree-wide -- the
  python-dotenv profile); HF `tokenizers` dropped (never imported; the repo
  ships its own BPE stack); `datasets` DECLARED at last (new `collect` extra +
  requirements line -- the collectors imported it while nothing installed it,
  so a clean checkout died at fetch time).
- **The v1 corpus got its filesystem lock:** `tokens.bin` + `tokens.json` are
  now attrib +R -- the pretokenize refusal only ever guarded that one script.
- **Judged ALIVE and now listed below (they were reference-free but are ops
  tools for the coming training block):** `sweep_lr.py`, `bench_generate.py`,
  `extend_length.ps1`, `power_guardian.ps1`.
- ~~Disk-reclaim candidates~~ **RULED 2026-07-29, EXECUTED AND CLOSED
  2026-07-30**: 272.77 GB / 60 targets moved to one staging path, verified,
  then deleted on the user's order -- C: free +262.4 GB measured. Per-item
  record lives in BACKLOG section 9; the surviving manifest (2dp GB per
  target -- the byte-exact original died with the staging dir) is
  `Enigma Backups\s9_manifest_reconstructed_2026-07-30.md`. pi_zero.pth
  HELD and verified present after; tokens.bin / tokens_v2.bin not named,
  still pending their own word.

## 2026-07-18 compression pass (what changed)

Deleted after a three-agent adversarial audit verified every claim
(caller evidence, exec-string probes, launcher/doc references):

- **The dormant Forge training stack (~13k LOC):** `training/training.py`
  (the monolithic Trainer incl. its DPO/APO/SimPO/KTO/ORPO paths, which
  rendered the OLD `User:`/`Assistant:` template and were format-misaligned
  with `chat_format`), `dispatch.py`, `schema.py`, `registry.py`,
  `training_evaluation.py`, `core/rl_training.py`, `core/lora_utils.py`
  (LoRA capability removed from `model.py` with it),
  `core/progressive_growing.py`, `core/reasoning.py`,
  `run_training_diagnostic.py`. The one live piece — `train_vision` —
  was carved verbatim into `training/vision_align.py` FIRST (align_vision.py
  now imports from there; old vision checkpoints resume unchanged).
- **True orphans:** `training/training_queue.py`, `training_monitor.py`
  (zero importers anywhere).
- **Test-only orphans:** `core/weight_mapping.py`, `core/model_merging.py`,
  `core/reward_functions.py`, `core/nf4_linear.py`, `core/curated_dataset.py`,
  `core/dataset.py` (+ their paired tests).
- **Dead experimental flags inside live files (~600 LOC):** MoE (was a stub —
  no expert FFN class ever existed), MoD, MLA, differential attention, ToMe,
  cross-layer KV-share, nGPT weight-norm, multi-token-predict heads,
  early-exit, shifted sparse attention, NEFTune, the flash-attn manual branch
  (SDPA is the path). All were OFF in every shipped config. **Verified: the
  served checkpoint strict-loads and produces byte-identical logits/tokens
  vs the pre-surgery code.** `--no-diff-attn` survived briefly as an accepted
  no-op; removed 2026-07-19 together with its only consumers (the flag lines
  in `extend_length.ps1` / `resume_training.ps1`).
- **Scratch scripts:** `_append_anime.py`, `_collect_anime_ln.py`,
  `_fix_anime_coverage.py`, `_audit_eval.py`.
- **Deps:** python-dotenv (core, zero imports), customtkinter (referenced a
  deleted GUI), peft/accelerate (LoRA stack) removed; bitsandbytes kept for
  the guarded INT4 fallback in `model.py`.
- **Archived:** `ULTRAREVIEW_2026-07-12.md`, `AUDIT_2026-07-13.md`,
  `CODE_REVIEW.md` -> `_archive/` (point-in-time review dumps / closed-bug
  ledgers; their present-tense entries are history, not current state).

## Package state — `enigma_engine/`

- **LIVE — her core:** `model.py`, `model_components.py`, `model_presets.py`,
  `model_utils.py`, `safe_save.py`, `tokenizer.py`, `bpe_tokenizer.py`
  (the vocab TRAINER — needed for tokenizer v2), `advanced_tokenizer.py`
  (the runtime tokenizer), `kv_cache.py` (the one real cache — the unused
  research variants and `config/` were deleted 2026-08-06; serve builds its
  CONFIG from the checkpoint), `model_registry.py`,
  `calculator.py`, `chat_format.py` (ONE template for train+serve),
  `memory_store.py` (BM25/JSONL), `hardware_detection.py`,
  and `optim.py` — the shared pretrain/finetune optimizer+schedule arsenal
  (`build_optimizer`/`get_lr`; also holds the flag-gated Muon and WSD that
  ROADMAP Phase 7 depends on).
  Edits here require the bit-identical fingerprint regime
  (`_verify_ckpt.py`) — the live checkpoint lineage depends on this code.
- **LIVE — organs (served behind flags):** `tts.py` (--voice), `asr.py`
  (--ears), `eyes.py` (--eyes, native), `imagegen.py` (--image-gen),
  `search.py` (--search, the sixth: <search>-span lookups through the local
  SearXNG; v2 vocab only — legacy tables carry no tag ids).
- **LIVE — perception training:** `vision_encoder.py`, `audio_encoder.py`,
  `training/encoder_align.py` (one hardened `_train_encoder` core;
  `train_vision` + `train_audio` wrappers for align_vision.py /
  align_audio.py — renamed from `vision_align.py` when audio joined).
- **LIVE — added since the compression pass:** `core/persona.py` (persona
  packs; `serve --persona`), `core/persona_content.py` (the CONTENT half of
  that seam — anchors, paraphrase intents, denials, self-facts, asides;
  `default_content()` is Enigma, `load_content()` is a pack directory),
  `core/barge_in.py` (mic energy VAD), `core/pretokenize.py` (the rust-backed
  v2 encode path).
- **KEPT DORMANT BY RULING:** `core/gguf.py` (llama-server export route,
  KNOWN_ISSUES; reachable only via `Enigma.export_to_gguf`). The GGUF SERVING
  pivot was REJECTED 2026-07-24, so the only remaining use for this file is
  the quantization idea in BACKLOG §6 — and its qwen3 auto-flip is math-wrong
  for the v1 architecture (norms before rope, missing NEOX permute). Nothing
  should ride it before that is fixed.
- **In git history only (idea-source, not code):** the Forge trainer's
  method variants (SimPO/KTO/ORPO/APO losses, EMA/SWA/LISA/LLRD machinery,
  RL loops) — retrieve from history if an instruct-pass design wants to
  reference them; any rebuild should be a bespoke script in the
  `dpo_enigma.py` pattern, NOT a revival of the monolith.

## Root scripts

- **Live tools:** `pretrain_enigma.py`, `finetune_enigma.py` (SFT),
  `dpo_enigma.py` (the adopted preference path), `teach_enigma.py`,
  `eval_behavior.py`, `eval_leak_guard.py`, `validate_probes.py` (probe-file
  authoring gate), `make_sft_data.py`, `make_dpo_data.py`,
  `make_facts_pretrain_data.py`, `make_pretrain_curated.py` (the T1 curated
  shard), `serve_enigma.py`, `sample_enigma.py`, `pretokenize_data.py`,
  `identity_anchors.py`, `identity_paraphrases.py`, `knowledge_corpus.py`,
  `align_vision.py`, `align_audio.py`, `distill_vision_encoder.py`,
  `distill_audio_encoder.py`, `ema_checkpoints.py` (checkpoint lineage tool),
  `training_progress.py`, `enigma_window.py`,
  `_verify_ckpt.py` (standing checkpoint fingerprint — keep).
- **Ops tools (reference-free by design — their caller is the operator; the
  2026-07-26 pass judged them by CONTENT, not by grep):** `sweep_lr.py`
  (T2/T3 LR grid over complete short runs incl. decay tail),
  `bench_generate.py` (decode ms/token + host-sync counter),
  `extend_length.ps1` (Phase-4 block-2048 launcher — DELETED 2026-08-26:
  dead v1-era recipe, `--init-from models/enigma_pretrain_large`, while v2
  trains natively at 2048; delegated call, git history preserves it),
  `power_guardian.ps1` (UPS watchdog, manually detached).
- **Corpus provenance (keep):** `collect_pretraining_data.py`,
  `collect_finetuning_data.py`, `collect_distill_data.py`,
  `collect_search_data.py`, `collect_vision_data.py`,
  `collect_audio_data.py`, `create_smoke_test_data.py`.

## Rules

1. **Verify importers before deleting** — including exec-string imports and
   launcher/doc references (the 2026-07-18 audit caught both kinds).
2. **Fingerprint before/after** any edit near the live model code. **BOTH
   lineages, since v2 was adopted 2026-08-09 — a v1-only fingerprint would
   pass a change that broke loading the model actually being served:**
   - v2 SERVED (`models/enigma_v2_sft2/model.pth`, step 480):
     PARAMS 238,374,400 / MODEL_KEYS 263 / KEYHASH `d9babe5af0f77dcf`,
     MISSING and UNEXPECTED both empty (measured 2026-08-09, CPU load-only).
   - v1 ROLLBACK (`_verify_ckpt.py`'s hardcoded
     `models/enigma_pretrain_large/latest.pth`):
     PARAMS 182,094,848 / KEYHASH `12edc0bc1ded383d`.
   `_verify_ckpt.py` still points at the v1 path only; point it at a
   checkpoint explicitly when fingerprinting the served lineage.
3. **git is the archive** — keep ideas, not code.
4. Suite baseline: **1626 passed on ENIGMAPC (2026-09-02, paired with the 2026-09-02 autonomous-stretch commit; 17 tests read external inputs — Enigma Backups transcripts plus the gitignored focused corpus, sealed locked-probe plaintext and on-disk checkpoint configs — and SKIP on any other machine; 15 more shell out to powershell.exe for the launcher -DryRun and Resolve-EnigmaPersona runs and skip where it is absent, reading nothing outside the repo)** — THE live number; other docs
   point here, and the commit that changes the count updates this line IN
   THE SAME COMMIT (this rule went stale by 2 within a day of being written;
   a manual step nothing enforces will drift again without the pairing — and
   it did, sitting at 810 across the whole 07-28→08-06 arc while the suite
   grew by 63). 873 → 867 is the kv-cache strip: six tests of the deleted
   research caches (TurboQuant/H2O/StreamingLLM) went with their classes;
   867 → 895 is the T4 regen shapes (`test_sft_regen_shapes.py` 23, plus 5
   teachings-route tests in `test_facts_pretrain_data.py`); 895 → 929 is the
   search organ + epistemics wave (`test_search_organ.py` 22, and the shape
   file grew to 36 with the search/unknown corpora); **929 → 945 is the
   T4→T6 arc in `42ba346b`** (13 structured-output/episodic/image-block
   tests in the drafting wave, plus 3 fix-arc tests for the power-expression,
   dictation-restraint and widened memory-decline shapes); **945 → 961 is the
   Stage-C epistemics wave in `fd88faa0`** (`test_eval_grading.py` grew 4
   decline-then-guess guard tests, and `test_dpo_focused_pairs.py` 12 pins
   the focused corpus: probe-screen clearance, grader-vocabulary coverage,
   counterweight invariants, and the trainer's system-slot render); 961 →
   992 is the review wave + Round-1 hardening + its ordered audit
   (2026-08-13): `test_dpo_trainer.py` 15 (collision detector with
   threshold-honest fixtures, tail coverage, required --out + startup
   artifact refusal incl. file-typo and prev.pth cases, --sanity exempt,
   main-wiring pin, dark-instrument warning, triageable advisory with a
   discriminating 6-hit fixture, save-before-report with uniqueness pins,
   payload shape + call-site pin), `test_facts_output_guard.py` 5
   (write_etok + STARTUP refusal, sidecar guard, .bin shape check),
   `test_eval_grading.py` +1 (the guess-guard cut point matches whole
   words -- constructed case; parity 174 sealed + 126 corpus rows, 0
   diffs), `test_guard_sweep.py` 4 (sealed re-grade + FP + corpus kill +
   the guard-specific decline-then-guess kill that keeps the sweep
   non-vacuous), and `test_finetune_out_guard.py` 6 (--out required +
   refusal, resume exempt ONLY into its own dir, sanity exempt and
   mkdir-after-sanity pinned, main-wiring pin).
   **992 → 1051 is the Round-2 review arc (2026-08-14/15)** -- the
   seven-lens sweep (make_sft_data, knowledge_corpus, identity, the eval
   stack, core, serve, collectors), its five fix waves, the arc's ordered
   audit and the ledger tail. Five new files: `test_safe_save.py` 4 (a
   rotation exception leaves the .tmp on disk instead of eating the save),
   `test_identity_generators.py` 6 (the identity machinery + its collision
   screens), `test_collect_audio_data.py` 3 (transcript mangling + atomic
   write, de-vacuumed), `test_eval_scratch_gate.py` 5 (the memory-wipe
   gate's fail-closed shapes) and `test_sft_writer_guard.py` 3 (the
   artifacts land together after fit_mix). Growth elsewhere:
   `test_chat_format.py` +7 (the parser trio + abandoned-tool-span
   hardening), `test_knowledge_data.py` +6 (cloze guards, degenerate
   census 4 → 0, fail-closed probe screen), `test_sft_regen_shapes.py`
   52 → 61 collected, +9 (seven new defs -- six screens print their
   held-out count, the dev screen reads the whole prompt, tool records
   never take the ASCII fast path -- one dropped verbatim duplicate of
   `test_memory_present_declines_do_not_parrot_the_distractor`, which now
   carries its width assertion, and the widened determinism parametrize
   adds 3 cases, no new def), `test_serve_enigma.py` +4 (renderer-parity
   extraction behind the shadowing 400, persist-failure WARNs, stream
   terminal-frame parity, /v1/memory kinds), `test_eval_transcript.py` +3
   (a zero-n baseline is refused a verdict, a torn write leaves the
   previous transcript intact), `test_memory_store.py` +3 (measure-key
   collapse, forget(None), supersede-persist ordering), and the three
   `test_collect_*.py` +2 each (recorded-diet reproduction, per-source
   manifests, atomic combine). Three more hardened INSIDE existing tests
   with no count change: `test_eval_leak_guard.py` (the FLAT tool-call
   args shape the builder actually emits now reaches the screen),
   `test_search_organ.py` (search is INSTRUCT-gated) and
   `test_eval_grading.py` (three probe phrasings re-cut).
   **1051 → 1059 is Stage-7 wave 1 (2026-08-15)** -- persona convergence.
   `test_serve_enigma.py` +4 (the legacy-state migration trio: a
   default-persona boot copies the repo's mute/talk truth into her home
   ONCE and says both paths, a later boot reads her home rather than
   re-seeding from the stale copy, and a second persona inherits none of
   her state; plus a pack boot adopts no legacy state at all -- the gate
   is the persona, not the order of boots), `test_persona.py` +4
   collected (one new def -- the chat page is titled with whoever is
   served, placeholder never reaching a browser -- and 3 new
   `test_an_unsafe_pack_is_refused` parametrize cases pinning the
   printable-ASCII property: a cp1252-hostile accent, a Cyrillic homoglyph
   of "E", and a tab).
   **1059 → 1103 is Stage-7 wave 2 + its ordered audit's fix wave
   (2026-08-16)** -- persona content interface, curated routing, pack
   content loader and seam hardening.
   **1103 -> 1206 is Stage-7 wave 3 (3a-3d) plus the 2026-08-17 audit-fix
   wave, the micro-cleanup and the verifier closing sweep** -- serve
   self-identification, launcher parameterization, per-AI eval and the pack
   authoring guide. Five new files carry 38 of the 103: `test_launchers.py`
   17, `test_conftest_guard.py` 7, `test_eval_persona_gate.py` 6,
   `test_validate_probes.py` 4 and `test_persona_probes.py` 4. The other 65
   grew in place -- `test_persona.py` +30, `test_persona_content.py` +16,
   `test_pretrain_curated.py` +8, `test_serve_enigma.py` +6,
   `test_enigma_window.py` +4, `test_eval_scratch_gate.py` +1 -- while
   `test_eval_transcript.py` was reworked around the persona-aware gate at
   the same 42 collected.
   **1206 -> 1254 is Stage-7 wave 4 (4a-4d) plus the fix wave from the
   2026-08-17 audit of `202b4863` (2026-08-18)** -- the pretokenize curated
   seam (`--curated-dir`/`--only-curated`), the SFT and DPO persona seams
   with their `--out` refusals, gate selection on both trainers, and the
   authoring doc's smoke-run recipe. Two new files carry 30 of the 48:
   `test_sft_persona.py` 12 and `test_dpo_persona.py` 18; the audit-fix
   wave's 11 and wave 4a's 7 grew in place (the scratch-rule union,
   pack-file encoding refusals, `data_dirname` control chars and the -Port
   reserved-port guard on the fix side; real-bytes curated-walk and refusal
   tests on the 4a side).
   **1254 -> 1256 is the round-2 solution-audit fix wave (2026-08-19)** --
   training prose unified on the em-dash (the voice path renders it and
   deletes the ASCII form), calculate delegation widened to hard arithmetic
   with the percent idiom canonicalized, and the facts-stream purity pin
   narrowed to intent (ASCII + the em-dash, C0/DEL refused). The 2 new
   tests are the dash-convention gate in `test_repo_hygiene.py` and the
   percent-idiom guard in `test_sft_regen_shapes.py`; everything else grew
   or moved in place.
   **1256 -> 1259 is the 2026-08-20 gate-flip audit riders** -- the 3 new
   tests are the `builtin_offering` stamp pin, availability-vs-offering and
   the regime-drift WARN, in `test_serve_enigma.py` and
   `test_eval_transcript.py`.
   **1259 -> 1270 is the 2026-08-22 full-audit fix wave** -- the sealed
   authored-to-clear pin for the math/tool corpora in
   `test_sft_regen_shapes.py`, 2 teach-line forget-precedence tests in
   `test_validate_probes.py` and the missing-capabilities WARN in
   `test_eval_transcript.py`; the other 7 ride the trainer/data fixes --
   DPO system-block screening build+consume (`test_eval_leak_guard.py`
   +2, `test_dpo_focused_pairs.py` +1), the commented-probe-file reader
   (`test_knowledge_data.py`), search url coercion
   (`test_search_organ.py`), the pretokenize sidecar refusal
   (`test_pretokenize_data.py`) and the fit-mix fast-path bound back in
   `test_sft_regen_shapes.py`.
   **1270 -> 1324 is the second 2026-08-22 audit fix wave** -- 15 grading
   regressions pinned (`test_eval_grading.py` +10 including the
   dash-appended-guess known-miss, `test_eval_transcript.py` +4 and
   `test_validate_probes.py` +2, less the one assertion they replaced),
   23 serve tests (12 non-finite-knob and 4 stream-parity in
   `test_serve_enigma.py`, the port probe, 4 caption-cache in
   `test_eyes.py`, and the `test_search_organ.py` rename+extend) and 16
   collector tests (`test_collect_pretraining_data.py` +9 done-marker and
   atomic-write, `test_collect_finetuning_data.py` +4 license/loader and
   `test_serve_enigma.py` +3 VoiceReq).
   **1324 -> 1405 is the third 2026-08-22 audit fix wave** -- 35
   core/tokenizer and serve (surrogate sanitize, special-forgery
   neutralization, the KV-window clean stop, the stream U+FFFD hold-back,
   the multi-family port probe, the chunked-CE contract and the ADV merge
   round-trip: `test_tokenizer_v2.py` +14, `test_chat_format.py` +8,
   `test_model_kv_cache.py` +7, `test_serve_enigma.py` +4,
   `test_audit_regressions.py` +2), 25 organs and memory (TTS worker
   survival plus cancel-on-timeout, the image-size door, the memory cap,
   fsync routing, the painter ValueError, the ears 400/500 split, the
   concurrent-writer detector and caption in-flight sharing:
   `test_serve_enigma.py` +13, `test_tts.py` +4, `test_eyes.py` +4,
   `test_memory_store.py` +2, `test_asr.py` +1, `test_imagegen.py` +1),
   1 net at the grading boundary (the reversal-marker test; the join-shape
   re-catch rewrote the known-miss pin in place) and 20 periphery
   (`test_launchers.py` +6, `test_collect_pretraining_data.py` +6,
   `test_enigma_window.py` +3, `test_teach_tool.py` +2, the new
   `test_sweep_receipts.py` +2 and `test_repo_hygiene.py` +1).
   **1405 -> 1416 is the round-four regression-fix wave** -- the round-three
   arc's adversarial findings: the reversal-void first-person-claim gate and
   the stative-decline exemption (`test_eval_grading.py` +2), the memory
   store's refuse-then-heal concurrent-writer contract
   (`test_memory_store.py` +1) with ConcurrentWriter caught at the tool door
   and all three /v1/memory routes answering 409, plus the completions-path
   U+FFFD hold-back in both modes (`test_serve_enigma.py` +4, one of them
   parametrized terminal/mid-text), Stop's booting-serve matcher gaining port
   discrimination via the ServePortMatch resolve field (`test_launchers.py`
   +3) and the bounded eyes in-flight waiter (`test_eyes.py` +1).
   **1416 -> 1429 is the review-fixes Wave A** -- the audited plan out of the
   2026-08-23 professional review: the pretrain warm-start vocab guard
   (`test_v2_trainer.py` +4) and the DPO block-vs-context guard
   (`test_dpo_trainer.py` +3), the atomic vision-pairs writer
   (`test_collect_vision_data.py` +1), the SFT resume data-sha guard
   (`test_finetune_sft.py` +3), the `[collect]` extras parity pin
   (`test_repo_hygiene.py` +1) and the conftest dot-dir addition guard
   (`test_conftest_guard.py` +1).
   **1429 -> 1461 is the review-fixes Wave B** -- the same plan's second half:
   the serve LAN-bind refusal and the server-side body caps
   (`test_serve_enigma.py` +8, 2 bind-guard tests plus 6 cap tests across the
   capped reader, the image and audio endpoints, speech input and the chat
   text/parts split), and direct tests for the two load-path modules that had
   none (`test_model_registry.py` +11, `test_hardware_detection.py` +13).
   **1461 -> 1467 is the multiturn-arc Wave 1** -- the delegated open-pile
   guards: the pretrain `--out` existing-lineage refusal
   (`test_v2_trainer.py` +3), output guards on all four encoder writers
   (`tests/test_encoder_out_guards.py`, new file, +4), the comparator's
   zero-gated-run refusal and the "everybody" non-value stem
   (`test_eval_grading.py` +2), and the dead `combined_finetune.txt` call
   path's removal (`test_collect_finetuning_data.py` net -3: the dead-path
   assertions went, one adversarial-negative asserting the `.txt` is NOT
   written came in).
   **1467 -> 1504 is the multiturn-arc waves 2+3** -- waves 2 and 3 plus
   tasks 4.1/4.2 of the 2026-08-25 multiturn plan: the reseal #8 eval and
   guard tests, the W3 corpus shape tests, and the per-token loss
   normalization tests.
   **1504 -> 1609 is the 2026-09-01 all-in-one arc** -- 85 from the arc's own
   new files: the serve conversation wave (`test_dry_sampler.py` 13,
   `test_toolspan_constraint.py` 7, `test_state_reinjection.py` 3), the
   CPU/portable lane (`test_strip_serving_ckpt.py` 2, `test_device_flag.py` 8,
   `test_quantize_serving.py` 10), and the wake loop (`test_wake_loop.py` 18,
   `test_wake_serve.py` 24); the remaining 20 are the parallel HUD session's --
   `test_eyes_lineage_guard.py` 7 (new file) plus 13 added to existing files
   (`test_serve_enigma.py` +7, `test_launchers.py` +2,
   `test_repo_hygiene.py` +1, `test_sft_regen_shapes.py` +3).
   **1609 -> 1626 is the 2026-09-02 autonomous stretch** -- the encoder-writer
   guards gained BEHAVIOR tests (`test_encoder_out_guards.py` 4 -> 12): the
   committed source-grep pin survives a guard neutered to a bare `return`
   (verified), so the distills' three rotation names, the aligns' real
   `{stem}_{modality}_best.pt` names, the `.pth`-typo refusal and a
   guard-runs-before-the-first-write ordering check were added; and
   `test_memory_synonyms.py` (new file, 9) covers query-time synonym expansion
   in memory retrieval -- the recorded "What's my job?" / "I work as a nurse"
   miss, four paraphrases, and the dampening that keeps a literal match ahead
   of a synonym one.
   The earlier "measured CPU-only, +3 with the GPU visible" qualifier did
   not reproduce and is retired: on this torch build `is_available()`
   ignores `CUDA_VISIBLE_DEVICES`, so collection is the same either way —
   867 counts the cuda cells of the `DEVICES`-parametrized tests.
   History: 574 before the 2026-07-18 compression pass, 349 after it (the
   delta was the dormant stack's own test mass, every removed test named),
   then steady growth through the v2-prep and audit arcs. Any cleanup that
   drops a test must say so explicitly.
5. **Retired ForgeConfig fields are load-bearing in reverse.** Every
   checkpoint on disk still carries the removed keys (up to 19 per config);
   `from_dict` tolerates them only because it filters against `known`.
   Never "simplify" it to `cls(**d)` — `tests/test_config_compat.py` fails
   loudly if anyone does.
