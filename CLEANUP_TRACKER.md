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
  (the runtime tokenizer), `kv_cache.py`, `model_registry.py`,
  `calculator.py`, `chat_format.py` (ONE template for train+serve),
  `memory_store.py` (BM25/JSONL), `hardware_detection.py`, `config/`,
  and `optim.py` — the shared pretrain/finetune optimizer+schedule arsenal
  (`build_optimizer`/`get_lr`; also holds the flag-gated Muon and WSD that
  ROADMAP Phase 7 depends on).
  Edits here require the bit-identical fingerprint regime
  (`_verify_ckpt.py`) — the live checkpoint lineage depends on this code.
- **LIVE — organs (served behind flags):** `tts.py` (--voice), `asr.py`
  (--ears), `eyes.py` (--eyes, native), `imagegen.py` (--image-gen).
- **LIVE — perception training:** `vision_encoder.py`, `audio_encoder.py`,
  `training/encoder_align.py` (one hardened `_train_encoder` core;
  `train_vision` + `train_audio` wrappers for align_vision.py /
  align_audio.py — renamed from `vision_align.py` when audio joined).
- **LIVE — added since the compression pass:** `core/persona.py` (persona
  packs; `serve --persona`), `core/barge_in.py` (mic energy VAD),
  `core/pretokenize.py` (the rust-backed v2 encode path).
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
  `extend_length.ps1` (Phase-4 block-2048 launcher, dormant),
  `power_guardian.ps1` (UPS watchdog, manually detached).
- **Corpus provenance (keep):** `collect_pretraining_data.py`,
  `collect_finetuning_data.py`, `collect_distill_data.py`,
  `collect_search_data.py`, `collect_vision_data.py`,
  `collect_audio_data.py`, `create_smoke_test_data.py`.

## Rules

1. **Verify importers before deleting** — including exec-string imports and
   launcher/doc references (the 2026-07-18 audit caught both kinds).
2. **Fingerprint before/after** any edit near the live model code
   (`_verify_ckpt.py`: PARAMS 182,094,848 / KEYHASH `12edc0bc1ded383d`).
3. **git is the archive** — keep ideas, not code.
4. Suite baseline: **810 passed (2026-07-27)** — THE live number; other docs
   point here, and the commit that changes the count updates this line IN
   THE SAME COMMIT (this rule went stale by 2 within a day of being written;
   a manual step nothing enforces will drift again without the pairing).
   History: 574 before the 2026-07-18 compression pass, 349 after it (the
   delta was the dormant stack's own test mass, every removed test named),
   then steady growth through the v2-prep and audit arcs. Any cleanup that
   drops a test must say so explicitly.
5. **Retired ForgeConfig fields are load-bearing in reverse.** Every
   checkpoint on disk still carries the removed keys (up to 19 per config);
   `from_dict` tolerates them only because it filters against `known`.
   Never "simplify" it to `cls(**d)` — `tests/test_config_compat.py` fails
   loudly if anyone does.
