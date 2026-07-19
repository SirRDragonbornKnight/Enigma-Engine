# Cleanup Tracker — reset 2026-06-11; COMPRESSION PASS 2026-07-18

This tracker describes the tree as it stands today; earlier eras (`gui/`,
`web/`, `services/`, `api/`, the Qwen-era `engine_*`/`inference`/`rag`
modules, and the Forge training arm removed 2026-07-18) live in git history.
Current truth:

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
  vs the pre-surgery code.** `--no-diff-attn` survives as an accepted no-op
  so proven launch commands keep working.
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
  `training/vision_align.py` (Trainer for align_vision.py).
- **KEPT DORMANT BY RULING:** `core/gguf.py` (llama-server export route,
  KNOWN_ISSUES; reachable only via `Enigma.export_to_gguf`).
- **In git history only (idea-source, not code):** the Forge trainer's
  method variants (SimPO/KTO/ORPO/APO losses, EMA/SWA/LISA/LLRD machinery,
  RL loops) — retrieve from history if an instruct-pass design wants to
  reference them; any rebuild should be a bespoke script in the
  `dpo_enigma.py` pattern, NOT a revival of the monolith.

## Root scripts

- **Live tools:** `pretrain_enigma.py`, `finetune_enigma.py` (SFT),
  `dpo_enigma.py` (the adopted preference path), `teach_enigma.py`,
  `eval_behavior.py`, `eval_leak_guard.py`, `make_sft_data.py`,
  `make_dpo_data.py`, `make_facts_pretrain_data.py`, `serve_enigma.py`,
  `sample_enigma.py`, `pretokenize_data.py`, `identity_anchors.py`,
  `identity_paraphrases.py`, `knowledge_corpus.py`, `align_vision.py`,
  `distill_vision_encoder.py`, `distill_audio_encoder.py`,
  `training_progress.py`, `enigma_window.py`,
  `_verify_ckpt.py` (standing checkpoint fingerprint — keep).
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
4. Suite baseline: **349 passed** (2026-07-18 post-compression: 340 after the
   deletions, +9 from the new `test_config_compat.py` and
   `test_cpu_rectangular_decode.py`; was 574 before the pass — the delta is
   the dormant stack's own test mass, every removed test named in the pass)
   — any cleanup that drops a test must say so explicitly.
5. **Retired ForgeConfig fields are load-bearing in reverse.** Every
   checkpoint on disk still carries the removed keys (up to 19 per config);
   `from_dict` tolerates them only because it filters against `known`.
   Never "simplify" it to `cls(**d)` — `tests/test_config_compat.py` fails
   loudly if anyone does.
