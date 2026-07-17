# Enigma Engine — Backlog

> Consolidated open work, newest snapshot 2026-07-15. Sources: the 4-reviewer
> methods audit (verdicts in `ROADMAP.md` "Update 2026-07-15"), the
> ultrareview ledger (`ULTRAREVIEW_2026-07-12.md`), and the dormant-code audit
> (`AUDIT_2026-07-13.md`). Priority = leverage x confidence, not size.
> Status: [ ] open  [~] in progress (uncommitted)  [x] done this arc.

---

## 0. Instrument arc (2026-07-15) — landed in full

- [x] **SFT val-split dedup** (`finetune_enigma.py`) — val fills only from
  records whose ids appear exactly once; duplicates all train (`47f557ae`).
- [x] **Merge `teach_pairs.jsonl` into DPO** (`make_dpo_data.py`) — user `/fix`
  corrections fold into preference data x3, behind the probe holdout (`47f557ae`).
- [x] **DPO val grouping by prompt** (`dpo_enigma.py`) — `group_split` deals
  whole prompt-buckets to val; no (prompt, chosen) twin or x3-taught duplicate
  straddles the sides (`fd2776d1`).
- [x] **Eval grader word-boundary matching** (`eval_behavior.py` + probes) —
  substring grading passed wrong answers ("own" on "known", "no" on "nothing")
  AND a perfect trained hosting answer failed probe 4's key list; fixed both
  directions, locked by `tests/test_eval_grading.py` (`bacc7473`). v5
  re-measured 27/29 on the 29-probe suite — superseded the same day by the
  90-probe suite below.
- [x] **`/undo` really undoes** (`teach_enigma.py`) — retracts the persisted
  records by truncating to pre-append offsets; a second `/fix` replaces the
  first instead of rejecting the user's own correction (`deb7c182`).
- [x] **Eval probes 29 -> 90** (`090e6644`) — every new probe machine-vetted:
  no trained-string collisions, facts only from the knowledge corpus, keys
  aligned with trained answer families. v5's HONEST baseline: 70/90 (78%),
  RESULT FAIL (adversarial 8/12, restraint 9/12, factual 13/20).

---

## 0.5 Eval trust -- de-contamination (2026-07-16 realism audit; see EVAL_REDESIGN.md)

> The behavior gate is partly self-measuring: `knowledge_corpus.py:21` authors
> probe twins on purpose and the leak guard is exact-match only, so
> factual/identity/adversarial/restraint scores conflate recall with
> generalization (math/memory/tool are clean). Full design + status in
> `EVAL_REDESIGN.md`. Tokenizer ceiling spec in `TOKENIZER_V2_SPEC.md`.

- [x] **Grader concession fix** (`eval_behavior.py`) -- adversarial/identity fail
  on an AFFIRMED false origin (`_false_origin_conceded`/`_grade_identity`); true
  greedy default (temp 0). Tests in `tests/test_eval_grading.py`.
- [x] **Locked-probe guard machinery** (`eval_leak_guard.py`) -- sealed
  hashed-shingle manifest + fuzzy Jaccard guard, wired into `make_sft_data`
  (`_held_out`), no-op until a manifest exists. Tests in
  `tests/test_eval_leak_guard.py`. Known limit: verb-swap paraphrases land in
  the 0.5-0.6 review band (flagged, not dropped).
- [ ] **Author the locked probe set** (~60-90, BLIND to the training corpus),
  then `python eval_leak_guard.py seal data/eval/locked_probes.jsonl`. Needs a
  human by design (the separation-of-powers rule).
- [ ] Widen thin eval categories to >=15 probes; re-measure v5/v8 on the locked
  set for the honest baseline.
- [ ] (Optional) second-grader agreement pass; semantic-embedding leak guard to
  close the verb-swap gap.

## 1. Correctness / measurement instruments (high leverage, small)

- [x] Encoder **persistence bug** — FIXED `f9ec5184`: `_save_checkpoint` takes
  encoder/optimizer overrides, vision/audio save their encoder + the LOCAL
  optimizer that actually stepped, `_load_encoder_checkpoint` resumes and
  REFUSES text-only checkpoints; 6 tests. Residual open: serve-side
  native-encoder load path (Phase 4.5 step 5).
- [x] **`--tokens-bin` resume-locked** (`pretrain_enigma.py`; final audit
  2026-07-16 M1) — FIXED 2026-07-16: `tokens_bin` is now recorded in the
  checkpoint schedule, and corpus resolution moved to AFTER the resume/schedule
  restore, so a bare `--resume` recovers the run's own corpus instead of
  silently finishing a facts run on the default 56.6B corpus. An explicit
  `--tokens-bin` still wins; checkpoints written before this fix predate the
  key and must re-pass the flag. (`test_pretrain_warmstart.py` still green.)
- [x] **`group_split` can't empty train / overshoot val** (`dpo_enigma.py`;
  final audit M2) — FIXED 2026-07-16: fewer than two prompt groups (small
  teach_pairs.jsonl) train the whole set with val empty; otherwise deal
  SMALLEST groups to val and never assign the largest group to val, so a giant
  group can neither empty train (`_batchify([])`) nor overshoot val_cap.
  Regression tests in `tests/test_dpo_split.py`.
- [x] **Facts-corpus val contract** (`make_facts_pretrain_data.py`; final
  audit M3) — FIXED 2026-07-16: `val_reserve = max(arg, target // 100)` so the
  pure-replay tail always covers pretrain's n//100 [val] window; a fence stops
  any fact doc from crossing `mixed_end` into that tail; documented command now
  passes `--val-general-end 0`. Regression tests in
  `tests/test_facts_pretrain_data.py`.
- [x] Repetition-penalty scope — was penalizing the prompt, suppressing her own
  primed vocabulary (ultrareview #9). Fixed + regression-tested (`b75ed617`).
- [x] Eval memory-store clear (#30) + golden-eval EOS strip (#12) — fixed (`fe5359a7`).
- [ ] Packing without doc-boundary attention masking — conversations attend
  across packed neighbors. Small effect at 182M/1024; INVESTIGATE only if
  context-bleed shows in chat.
- [ ] Instruct serve omits the trained "You are Enigma..." preamble when the
  client supplies its own system message (tools block joins without it) — a
  train/serve shape she never saw; may weaken tool-calling under custom
  system prompts. LOW CONFIDENCE (audit 2026-07-15); verify against the SFT
  corpus before changing `_with_context`.

## 2. Ultrareview backlog — verified-open correctness majors

> 11 confirmed open 2026-07-14. Mostly the DORMANT training arsenal (LoRA / RL /
> queue) -- but NOT all (corrected 2026-07-16): #6 below is a live train/serve
> shape mismatch, and the unverified tail holds more live-path serve findings
> (#15 tool-call name drop, #31 stream/non-stream divergence, #36 serve runs
> fp32, #45 memory-store fsync, #51 /v1/memory empty-text 500). AUDIT_2026-07-13's
> "~43 of 44 open" counts every category; this list is the verified
> correctness-major subset.

- [ ] #5 training_queue: `start()` after `stop()` can run two jobs concurrently.
- [ ] #6 memory-block + tool-spec never combined in TRAINING but serve combines
  them — the combined system shape is 0% of training data.
- [ ] #7 LoRA + kv_share models crash on grad-checkpointing.
- [ ] #8 online-DPO generates from un-stripped trailing EOS.
- [ ] #10 DPO fp16 fallback has no GradScaler (bf16 path, the one we use, is fine).
- [ ] #11 LoRA `merge_into_base` doubles weights (treats saved values as deltas).
- [ ] #13 LoRA `create()` snapshots the full model as a fake adapter.
- [ ] #14 non-SDPA attention uses a square mask for rectangular cached decode
  (CPU/MPS path; CUDA SDPA path is correct).
- [ ] #33 Trainer preference path uses a bespoke `User:/Assistant:` template, not
  `render_training` — format-misaligned with serve. DECISION: deprecate it
  (the standalone `dpo_enigma.py` is the real path) vs align it.
- [ ] ~63 majors/minors + ~15 appendix items UNVERIFIED — mostly efficiency /
  simplification in the dormant arsenal + ASCII-console nits. Triage on demand.

## 3. Data strategy (the real quality ceiling)

- [x] **Drop OpenThoughts3** — out of `--all`, 58 MB source file deleted,
  regenerable via the explicit flag (`8104e09c`).
- [x] **Rebuild the diet** — collectors for No-Robots / Everyday-Conversations /
  TriviaQA / NQ-Open + SmolTalk2 diet mode (600-char completion cap, 800-char
  prompt cap, think-split skip). combined_finetune.jsonl: 105,203 SHORT pairs
  (`8104e09c`).
- [x] **Facts many-format CONTINUED PRETRAINING** — `make_facts_pretrain_data.py`
  (60M tokens, 2.4% facts, replay-anchored) + `pretrain_enigma.py --tokens-bin`;
  checkpoint `models/enigma_pretrain_facts` (`3b553038`, `701434be`). Measured:
  factual 13/20 -> 19/20 on v6. Val-contract nit open in section 1.
- [x] **knowledge_corpus format mixing** — `gen_knowledge_pretrain_text`: 914
  lines as declarative / QA / key-term-final cloze / in-context (`701434be`).
- [x] **v8 ADOPTED 2026-07-16** — measured **79/90 (88%), ALL SEVEN CATEGORIES
  PASS** — the first checkpoint to clear the full 90-probe gate (identity 15/18,
  adversarial 11/12, tool 12/12, restraint 10/12, math 7/8, memory 7/8, factual
  17/20). Lineage: v5 70/90 FAIL -> v6 76/90 FAIL (diet dilution) -> v7 72/90
  FAIL (repetition != coverage) -> v8 PASS (coverage-widened memory/identity +
  moderate fractions, on the facts continued-pretrain base). `models/enigma_dpo/
  model.pth` now holds v8 (SHA256 `A11DB8F0...`); receipted backup at `Enigma
  Backups\enigma_dpo_v8_adopted\` (model+config+vocab .sha256). v5's backup at
  `enigma_dpo_v5_adopted\` is untouched (revert target). Restart serve to load
  v8. Note: v8's memory score includes the corrected October probe; v6/v7 were
  measured under the old March key.
- [x] **Teach-loop auto-augment** — DONE 2026-07-16 (`teach_enigma.py`): each
  `/fix` now `augment_teaching`s the correction into >=3 deduped question
  phrasings + a declarative statement twin (only for simple `what/who/where is
  X`; behavioral corrections get none), then `review_augmentation` shows them
  for accept / edit / skip / cancel before ANY write (confirm-before-bake).
  Non-interactive stdin auto-accepts so scripted teaching still works. Bake
  weight `TEACHINGS_REPEAT` 8 -> 4 (`make_sft_data.py`) now corrections carry
  their own variety. Regression tests in `tests/test_teach_tool.py`.
- [x] Low-quality gate (URLs/HTML/encoding/loops; profanity NOT filtered per
  ruling) — done (`d0dd527e`).
- [x] Knowledge weight x2 -> x5 — done (`43870254`).

## 4. Phase 4.5 — Owned organs (the huge project; ~1-2 weeks GPU)

> Full ordered plan in `ROADMAP.md` Phase 4.5. Retire borrowed backbones one at
> a time; teachers used OFFLINE during distillation only.

- [x] 1. Encoder persistence — DONE `f9ec5184` (see section 1); the blocker
  is dead. Serve-side encoder loading folds into step 5.
- [ ] 2. Collect LLaVA-Pretrain 558k (`collect_vision_data.py`, ~14 GB).
- [ ] 3. Distill DINOv2-S -> her own ViT-medium (~25M, ~1-2 GPU-days).
- [ ] 4. `train_vision` align on 558k (add real batching first — current loop is batch-1).
- [ ] 5. Image begin/end tokens (ids 4724+ free), serve wiring, delete BLIP.
- [ ] 6. Her ears: write `collect_audio_data.py` (LibriSpeech-clean-100),
  distill whisper encoder, `train_audio`, wire, retire whisper (~3 days).
- [ ] 7. Her voice: train a small TTS on a chosen voice (later).
- [ ] 8. Her imagination: own image generator (much later; SD stays the tool she wields).

## 5. Interim organ upgrades (still borrowed, better scaffolding; pip-only)

> USER RULING 2026-07-16: voice/sound stays OFF for now ("we will work on it
> later when it matters") -- launchers no longer pass -Voice. The zira
> `--voice-name` stopgap and the Kokoro swap below wait for that ruling to lift.

- [ ] TTS SAPI -> **Kokoro-82M** (~330 MB; near-natural, pure-Python G2P).
- [ ] ASR whisper-base -> **large-v3-turbo** (~1.6 GB; ~half the errors).
  VERIFY FIRST: CTranslate2 CUDA works on the 5090 (sm_120) — else it silently
  falls back to slow CPU int8. One-line check: `Ears(device="cuda").device`.
- [ ] Eyes BLIP -> **SmolVLM2** with **question-conditioned VQA** — captioning
  throws the user's question away; VQA answers what was actually asked. Bigger
  win than the model swap alone.
- [ ] Image gen sd-turbo -> **sdxl-turbo** — a one-string change in `Painter`
  (already turbo-aware); higher fidelity, fits VRAM easily.
- [x] Offline-by-default privacy (organs load from cache; `--allow-downloads`
  gates the one first fetch) — done (`6d3cf598`).

## 6. Cost & efficiency (keep/raise quality, lower resource use)

- [ ] **Quantization** — `core/gguf.py` export exists. int8 is near-lossless and
  roughly halves her VRAM/footprint; int4 (~4x smaller) with slight quality
  cost lets her run on far weaker hardware and start faster. She's already free
  to run (local, no API), so this is pure headroom, not a cost cut.
- [ ] **Load organs on-demand** vs eager-at-boot — keeps idle VRAM low when an
  organ isn't in use; matters once eyes/imagination models get bigger.
- [ ] **min_p-only sampling A/B** — drop top_p+top_k, keep min_p 0.05-0.1; the
  min_p literature says it tolerates higher temperature without rambling. One
  eval run decides. (KEEP verdict stands until measured.)
- [ ] The genuine cost tradeoff to be aware of: a bigger brain (Phase 7,
  350-700M) buys quality but costs more VRAM/time. Data quality is the cheaper
  quality lever at this scale — spend there first.

## 7. Housekeeping / dormant code (low priority, low risk)

- [x] `enigma_engine/core/adaptive_trainer.py` — DELETED 2026-07-17 along with
  `adaptive_prompts.json` and the "adaptive" mode registration
  (schema/registry/dispatch); regression assert in `test_training_dispatch.py`.
- [x] Unused deps `SpeechRecognition` + `sounddevice` — dropped from
  `pyproject.toml` (full + voice extras) 2026-07-17.
- [x] `data/sft/math.jsonl` — deleted 2026-07-17.
- [x] `enigma_engine/core/rl_training.py` guarded caller of the deleted
  `sentiment` module — removed 2026-07-17; `test_import_integrity.py`
  ALLOWED_MISSING is now empty (gate fully strict).
- [x] Config naming — `_load_user_config` now also searches
  `~/.enigma_engine/forge_config.json` (legacy `config.json` kept for
  back-compat) 2026-07-17.
- [x] **Scratch checkpoints PRUNED 2026-07-16** (user-approved in chat): ~500 GB
  freed (1.0T -> 1.5T free) across scratch sft/dpo checkpoints and five
  forgotten April `_pretrain_sequences.jsonl` caches. v8 is the adopted DPO
  (`models/enigma_dpo`, receipted backup `enigma_dpo_v8_adopted\`); kept:
  `enigma_sft`/`enigma_dpo`, `sft_v8`/`dpo_v8`, all pretrain runs, the Qwen
  zoo, smoke/trainv4 fixtures.
- [x] Docs: facts continued-pretrain recipe (training_guide.md Stage 1.5 +
  quick_commands.md rows) + diet collector flags documented; CLAUDE.md
  pipeline line mentions the optional facts hop (2026-07-17).
- [x] Teach tool nits (final audit m3/m4) — DONE 2026-07-16 alongside the
  auto-augment work: `/good` now refuses when the exchange already has a saved
  teaching (no double-write; `/undo` first to change it); `retract` only ever
  SHRINKS a file (guards against NUL-padding a hand-edited jsonl when the
  recorded offset is past the current end). Regression test for the shrink
  guard in `tests/test_teach_tool.py`.
- [ ] Memory-read data nit (final audit m9): contradictory color facts can
  co-occur in one distractor block (green vs orange) — de-conflict attribute
  domains when widening further.
- [ ] `teachings.jsonl` still the untouched example template — YOUR channel to
  author (values / personal facts); bakes in at x8.

## 8. Long-term (Phase 7 / embodiment; weeks of GPU)

- [ ] New tokenizer — fix the standalone-space waste (26.6% corpus-wide; 29.5%
  on the 2026-07-16 English-sample re-measure) + digit-splitting
  (the real fix for number recall; word-numbers are a band-aid). Requires
  retokenizing the corpus.
- [ ] Length extension block 1024 -> 2048+ (Phase 4) — unlocks real multi-turn
  tool loops AND native video (frames are token-hungry: 196/frame).
- [ ] Deeper-thinner 350-700M architecture — the 5090 can carry it.
- [ ] Embodiment: tool-executor bridge + avatar bus (`ws://127.0.0.1:8765`) —
  work lives in the Enigma Avatar repo.
- [ ] Training sim -> trajectory logs -> real-time game play (FNAF target).
- [ ] Video: organ-tier (frame-sample -> describe -> summarize) buildable ANY
  time; native video needs Phase 4 first.
