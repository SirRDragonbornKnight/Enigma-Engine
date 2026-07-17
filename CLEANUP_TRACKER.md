# Cleanup Tracker — reset 2026-06-11

This tracker describes the tree as it stands today; earlier eras (`gui/`,
`web/`, `services/`, `api/`, the Qwen-era `engine_*`/`inference`/`rag`
modules) live in git history. Current truth:

## Package state — `enigma_engine/`, 45 files / ~30.7k LOC
<!-- measured 2026-07-16 (find enigma_engine -name "*.py" | wc -l; cat | wc -l) -->


- **LIVE — her core (~7.5k LOC):** `model.py`, `model_components.py`,
  `model_presets.py`, `model_utils.py`, `safe_save.py`, `tokenizer.py`,
  `bpe_tokenizer.py`, `advanced_tokenizer.py`, `kv_cache.py`, `nf4_linear.py`,
  `model_registry.py`, `calculator.py`, and since 06-11 `chat_format.py` (the
  instruct-pass format — ONE template for train+serve) and `memory_store.py`
  (runtime memory, BM25/JSONL). Edits here require the bit-identical fingerprint
  regime (`_verify_ckpt.py`) — the live checkpoint lineage depends on this
  code.
- **LIVE — organs (2026-07-14, served behind flags):** `tts.py` (--voice),
  `asr.py` (--ears), `eyes.py` (--eyes), `imagegen.py` (--image-gen).
- **RESTORED 2026-07-13 (training-side, projectors untrained):**
  `vision_encoder.py`, `audio_encoder.py`, `gguf.py` (export), `reasoning.py`.
- **DORMANT BY RULING (2026-06-11, ~13k LOC):** `training/` package +
  `core/rl_training.py` + `core/lora_utils.py` (`router.py` since REMOVED
  with modkit). Evidence: zero
  HuggingFace imports outside lazy-optional paths in `lora_utils`; the Trainer
  targets the custom `Enigma` class. It is the in-house SFT/LoRA/RLHF arsenal
  — the moat's training arm — and is test-covered (~80 tests). **KEEP** until
  the instruct pass is designed; then either reconnect it or replace with a
  bespoke finetune script (the `pretrain_enigma.py` pattern). Nothing at
  runtime imports it today.
- **TEST-COVERED SUPPORT:** `dataset.py`, `curated_dataset.py`,
  `progressive_growing.py`, `weight_mapping.py`,
  `hardware_detection.py`, `config/`.
  (`commands.py`, `plugin_loader.py`, `mod_tools.py` REMOVED 2026-07-13 with the modkit subsystem.)
- **FULL ORPHAN (audit 2026-07-16):** `adaptive_trainer.py` — its covering
  tests (`test_gui.py`/`test_new_features.py`/`test_training.py`) are long
  deleted, zero importers anywhere, and `training/dispatch.py` explicitly
  rejects the registered "adaptive" mode. Safely deletable per rule 1.
- **In git history only (idea-source, not code):** `core/personality_data.py`'s
  distillation prompts are an idea-source for the values corpus — retrieve
  from git when needed.

## Root scripts

- **Live tools:** `pretrain_enigma.py`, `finetune_enigma.py` (SFT, 06-11),
  `make_sft_data.py` (06-11), `serve_enigma.py`, `sample_enigma.py`,
  `pretokenize_data.py`,
  `identity_anchors.py` (its EXAMPLES feed make_sft_data; renamed from
  `make_enigma_corpus.py` 06-30),
  `_verify_ckpt.py` (standing checkpoint fingerprint — keep).
- **Corpus provenance (keep):** `collect_pretraining_data.py`,
  `collect_finetuning_data.py`, `collect_distill_data.py`,
  `collect_search_data.py`, `collect_vision_data.py`,
  `create_smoke_test_data.py`.
- **Scratch (tracked since 721d25e; delete when stale):** `_append_anime.py`,
  `_collect_anime_ln.py`, `_fix_anime_coverage.py`, `_audit_eval.py`.
- **Muppet-era — RESOLVED at instruct-pass design (2026-06-11):**
  superseded by `finetune_enigma.py` (git is the archive).
  `run_training_diagnostic.py` kept — it travels with the dormant FORGE
  stack.

## Rules

1. **Verify importers before deleting.** (A sub-agent mislabeled `mod_tools`
   as orphaned; grep found ~25 tests using it.)
2. **Fingerprint before/after** any edit near the live model code
   (`_verify_ckpt.py`: PARAMS 182,094,848 / KEYHASH `12edc0bc1ded383d`).
3. **git is the archive** — keep ideas, not code.
4. Suite baseline: **512 passed** (2026-07-17; was 364 at the 06-11 reset) —
   any cleanup that drops a test must say so explicitly.
