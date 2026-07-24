# Enigma Engine — Backlog

> Consolidated open work, newest snapshot 2026-07-15. Sources: the 4-reviewer
> methods audit (verdicts in `ROADMAP.md` "Update 2026-07-15"), the
> ultrareview ledger (`_archive/ULTRAREVIEW_2026-07-12.md`), and the
> dormant-code audit (`_archive/AUDIT_2026-07-13.md`; both moved to
> `_archive/` in the 2026-07-18 compression pass).
> Priority = leverage x confidence, not size.
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
- [x] **Checkpoint-safety arc 2026-07-19** (`vision_align.py` +
  `align_vision.py`) — two audited rounds, converged; 12 regression tests
  (file now holds 17), each mutation-verified; suite 349 -> 361. Round 1:
  missing `resume_from` REFUSES (was warn-and-restart, which then overwrote
  the prior best), `_load_encoder_checkpoint` writes the `.keep` cleanup
  marker, train_vision body in try/finally (aborts left params frozen +
  encoder in train mode), `save_every_steps` implemented (rolling
  `{stem}_vision_step.pt`; `align_vision.py --save-steps`, default 500).
  Round 2 (adversarial audit of round 1): best tracking split — `best_loss`
  = pure metric (drives early stopping), `best_written` = what reached disk
  (drives retry); a run whose best never persisted ends with `abort_reason`
  set (align_vision SystemExits on it); mid-epoch rolling-checkpoint resume
  winds scheduler+step back to the epoch boundary (`epoch_start_step` is now
  load-bearing); str-path `.keep` fix; remainder-flush steps also fire the
  rolling save; `encoder_key` ValueError raised before the swallow-all try;
  fresh writes delete stale `.keep` markers; guarded finally.
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
  client supplies its own system message (tools block joins without it).
  **VERIFIED AGAINST THE CORPUS 2026-07-19 — the shape mismatch is real, but
  the current behavior is DELIBERATE and test-pinned, so this is a decision,
  not a bug fix.** Evidence: `make_sft_data._system()` ALWAYS emits
  "You are Enigma. You can use tools...\nAvailable tools:...", and the
  memory_tools generator (`make_sft_data.py` ~764/784) shows that even when
  a block PRECEDES the tool spec, the preamble is retained
  (`block + "\n\n" + _system(subset)`). So the model has never seen a tool
  spec whose immediate left context isn't that preamble — which is exactly
  what serve renders when a client system message exists
  (`serve_enigma.py` ~942: the preamble is prepended only when there is NO
  client system message). Counter-argument: honoring the client's system
  message is correct OpenAI-compatible semantics, and
  `tests/test_serve_enigma.py::test_with_context_client_system_message_is_appended_not_preambled`
  pins "You are Enigma" NOT being injected. Middle path if we act: keep the
  client's message as the OPENER (their intent preserved) but restore the
  preamble to the tools block itself — client + "\n\n" + memories + "\n\n" +
  preamble + tools — which matches the trained join shape exactly; that test
  would need its assertion updated (its stated intent still holds).
  Reachability (checked 2026-07-19): LOW — no HTML/JS in the repo builds a
  system-role message, so her own chat page and the launcher chain never hit
  this branch; it only affects external OpenAI-compatible clients that send
  their own system prompt. That argues for leaving it alone until such a
  client actually matters. USER CALL.

## 2. Ultrareview backlog — verified-open correctness majors

> 11 confirmed open 2026-07-14. The LIVE-PATH subset was triaged and CLOSED
> 2026-07-17 (all six verified still present, then fixed + regression-tested;
> serve smoke + full 90-probe eval green): #6 (combined shape, data-side),
> #15 (name-less tool call kept via raw), #31 (stream/non-stream parity),
> #36 (serve bf16 autocast + TF32 — eval re-measured 79/90, same as fp32),
> #45 (memory-store fsync via atomic_write_text), #51 (/v1/memory 400).
> What remained below was the DORMANT training arsenal (LoRA / RL / queue).
> RESOLVED BY DELETION 2026-07-18: the compression pass removed the entire
> dormant Forge stack (training.py, dispatch/schema/registry,
> training_evaluation, rl_training, lora_utils, progressive_growing,
> reasoning, training_queue, training_monitor, run_training_diagnostic),
> so #5/#7/#8/#10/#11/#13/#33 and the ~63 unverified arsenal items no
> longer have code to be wrong in. #33's decision landed as "deprecate":
> `dpo_enigma.py` is the preference path. See CLEANUP_TRACKER.md.

- [x] #5 training_queue double-run — module deleted 2026-07-18.
- [x] #6 memory-block + tool-spec combined shape — FIXED 2026-07-17 (data
  side): `gen_memory_tools_examples` bakes serve's exact join (memories,
  blank line, preamble + tools; both answer-from-memory and still-call-the-
  tool behaviors), x8 in the mix (53 records), locked by
  `tests/test_memory_tools_data.py`. NOTE: the SERVED model only learns the
  shape at the next SFT->DPO cycle; until then serve still renders it.
- [x] #7/#8/#10/#11/#13/#33 — resolved by deletion 2026-07-18 (LoRA stack,
  online-DPO, Trainer preference paths all removed with the Forge bloc).
- [x] #14 non-SDPA attention used a square mask for rectangular cached decode
  (CPU/MPS path; CUDA SDPA path was already correct) — **FIXED 2026-07-18**.
  Characterized by execution first: it was a loud broadcast CRASH
  (`RuntimeError: size of tensor a (9) must match tensor b (3)`), not the
  silent corruption the old wording implied. Unreachable from the live serve
  loop (prefill once, then one token at a time), so it never bit us; any
  chunked prefill or multi-token continuation on CPU died. Fix mirrors the
  SDPA branch: bottom-right aligned `tril(diagonal=T_k - T)`; square prefill
  reduces to the old mask (served logits verified byte-identical).
  Regression tests: `tests/test_cpu_rectangular_decode.py` (5 tests, pins
  crash-freedom AND value-correctness vs no-cache recompute, plus a causality
  guard). MUTATION-VERIFIED against both the original square mask and a
  correctly-shaped-but-top-left-aligned mask.

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
- [x] 2. Collect LLaVA-Pretrain 558k — DONE (data staged; see CLAUDE.md
  multimodal state 2026-07-17).
- [x] 3. Distill DINOv2-S -> her own ViT-medium — DONE 2026-07-17
  (`models/enigma_vision_distill/`, val cosine 0.3469; [-1,1] contract
  test-pinned in `tests/test_vision_normalization.py`).
- [x] 4. `train_vision` align on 558k — DONE 2026-07-20 (val 1.4884;
  `models/enigma_vision_align/`). `serve --eyes` boots "eyes: on" and
  captions live. Captions are primitive at 182M — grounding errors and
  greedy loops (the loop is fixed by the caption repetition penalty,
  `d15bc6c`); quality work belongs to the next align cycle
  (VISION_QUALITY_SPEC: bigger student, pixel-shuffle connector, stage-2
  unfreeze), which is gated on the user's image-domain decision.
- [~] 5. Image begin/end tokens (ids 4724+ free), serve wiring, delete BLIP.
  Serve wiring DONE and BLIP deleted 2026-07-17; her own distilled ViT serves
  live under `serve --eyes`. STILL OPEN, both needing the next training cycle:
  (a) the image begin/end TOKENS were never allocated or trained -- ids
  4724-4735 remain "reserved for future passes" in `chat_format.py`, there is
  no delimiter constant, and captions reach the model as the "[image: ...]"
  text marker instead; (b) captions are question-blind (serve passes
  `EYES.describe` as a bare 1-arg callable, so the pixels are gone before the
  question is asked) even though `model.forward_multimodal` already
  concatenates [vision][text] -- closing it is a stage-2 VQA align plus an
  `Eyes.answer(img, question)` path.
- [~] 6. Her ears: `collect_audio_data.py` DONE, `distill_audio_encoder.py`
  DONE-not-launched (own loop, survived the compression pass). Align
  trainer REBUILT 2026-07-19: `vision_align.py` generalized into
  `enigma_engine/training/encoder_align.py` (one `_train_encoder` core, a
  `_Modality` adapter, `train_vision`/`train_audio` wrappers) so audio
  inherits every hardening + regression pin from day one; `align_audio.py`
  entry point added; 6 audio contract tests re-lock the encoder
  persistence twin. Batched audio LANDED 2026-07-20: the mask-aware
  AudioEncoder makes padded-batch == unbatched at 3.6e-7, so
  `align_audio.py --batch-size 8` is the supported path. REMAINING (GPU):
  the distill is blocked on downloading the `openai/whisper-base` teacher
  (the cached Systran/faster-whisper-base is the ASR organ, NOT the
  teacher); then run the align, serve wiring + retire whisper.
- [ ] 7. Her voice: train a small TTS on a chosen voice (later).
- [ ] 8. Her imagination: own image generator (much later; SD stays the tool she wields).

## 5. Interim organ upgrades (still borrowed, better scaffolding; pip-only)

> USER RULING 2026-07-16: voice/sound stays OFF for now ("we will work on it
> later when it matters") -- launchers no longer pass -Voice.
> RULING LIFTED 2026-07-23: voice work resumed by user order; the launchers
> pass `-Voice` again and boot with talk-mode OFF (she starts silent).

- [x] TTS SAPI -> **Kokoro-82M** (~330 MB; near-natural, pure-Python G2P).
  DONE 2026-07-23: `core/tts.py` runs on Kokoro. Synthesis measured at RTF
  ~0.25x on the 5090 (1.31 s of compute for 5.28 s of audio, first run
  including warmup) -- a session measurement with no committed benchmark, so
  re-measure before relying on it. Voices are style tensors that blend by
  weighted sum (`set_recipe` multiplies and sums whatever `load_voice`
  returns; no shape is asserted anywhere); the
  chosen recipe approximates the Cortana character and persists to
  `~/.enigma_engine/voice.json`. The `[voice]` extra installs kokoro +
  soundfile + sounddevice, and the launcher runs the server under the repo
  `venv/` where kokoro lives.
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
- [x] Memory-read data nit (final audit m9) — FIXED 2026-07-19. Confirmed
  real: `gen_memory_read_examples` drew distractors from every other fact,
  so a block could assert BOTH "favorite color is green" and "likes the
  color orange" while the trained answer named one — the question has two
  valid answers in context, teaching an arbitrary pick instead of
  retrieval. Fix is a general mechanism, not a one-pair patch:
  `_CONFLICTING_FACTS` groups facts that answer the same question and the
  sampler excludes the target's group (add a group when widening with an
  attribute that already has a value). `tests/test_memory_read_data.py`
  pins the contract, that memory_tools inherits it, and — because the
  groups repeat fact strings — that a renamed fact can't silently drop out
  of its group. Both mutation-verified. Takes effect at the next SFT bake;
  the served v8 was trained on the old data.
- [ ] `teachings.jsonl` still the untouched example template — YOUR channel to
  author (values / personal facts); bakes in at x8.

## 7.5 2026-07-19 review — open cleanup/efficiency findings

> From the xhigh compression-pass review (25 verified findings; the
> correctness/latent-bug subset lives in KNOWN_ISSUES #12, the
> checkpoint-safety subset was fixed same day — section 1). All verified
> against the working tree. Efficiency items matter most before the 558k
> align run.

- [x] **Serial PIL decode inside the training step** — FIXED 2026-07-19
  (pre-align batch, round 3): path decodes run on an 8-thread pool with
  prefetch depth 2 while the GPU trains; augmentation stays on the main
  thread in batch order (seeded determinism test-pinned); the `verify()`
  probe pre-pass runs pooled in bounded 512-item chunks; text batches
  build on CPU with one `.to(device)` (train + val). In-memory PIL refs
  decode inline (no pool win; a shared lazy PIL object must not `load()`
  concurrently). Same round: token-weighted val loss; epochs whose metric
  a stop truncated are never ranked while a val pass that COMPLETED
  before the stop still ranks (closes the stop-mid-epoch ranking finding
  both ways); the four lying knob defaults refused;
  `ForgeConfig.from_dict` known-set derived from `dataclasses.fields()`.
  10 new tests; 19-mutation sweep all killed; suite 361 -> 371; audited
  to convergence (4 rounds, severity high -> med -> low -> none).
- [x] **Cleanup batch 2026-07-19 (round 4)** — closed in one audited pass
  (6 new mutation-verified tests, 25-mutation sweep, suite 371 -> 377):
  dead fallback optimizer -> lazy property, old-checkpoint optimizer state
  no longer materialized; `_estimate_batch_size` RETIRED (batch_size >= 1
  required; refusal points at `hardware_detection.
  recommend_training_batch_size`) — also removes the caller-config
  mutation; train/val share `_forward_ce` (drop-policy drift dead);
  `TrainingConfig` slimmed ~25 inert fields with field-derived
  to_dict/from_dict; one-allocation CPU mask; `training/__init__` shim
  and `--no-diff-attn` (+ both .ps1 consumers) removed; MoE/LoRA/
  speculative/MTP/test-prose doc drift grounded; `total_tokens`/
  `dataset_fingerprint` dead checkpoint keys and `_emit_loss`'s dead
  val_loss param removed.
- [ ] **Deliberately DEFERRED (rationale logged 2026-07-19):**
  trainable-subset intermediate best-saves — the best checkpoint is the
  primary resume artifact and must stay full-format; the flagship 1-epoch
  run writes ~one best save total, so the I/O win is small next to the
  resume-compat risk. Revisit only if multi-epoch align runs become the
  norm. Also still open: `_save_checkpoint` stores `config.__dict__`
  instead of canonical `ForgeConfig.to_dict()` and writes dual
  `model_config`+`config` keys (`test_encoder_persistence.py` pins both
  keys — change together; `config` is the live key with 7 readers).

## 7.9 v2 pretrain: measured launch constraints (2026-07-21, on the 5090)

Micro-batch fit search over the deep-thin presets, `--sanity` (one fwd/bwd),
`--no-grad-ckpt`, sdpa cudnn, corpus `tokens_v2.bin`:

| preset | block 2048 | block 8192 |
|---|---|---|
| `v2_deep_186m` | fits, micro-batch 8 | **does not fit at micro-batch 1** |
| `v2_deep_238m` | fits, micro-batch 16 | **does not fit at micro-batch 1** |
| `v2_deep_542m` | fits, micro-batch 8 | **does not fit at micro-batch 1** |

**That table is measured with `--sanity` and is NOT a training-shape receipt.**
`--sanity` runs one fwd/bwd and allocates no optimizer state, so it overstates
what fits: at 186m/2048 it declared micro-batch 8 usable, but the full step
(fwd + bwd + Muon) peaks at 31.68 GB on a 31.84 GB card and thrashes into
shared memory, running **2.6x slower** than micro-batch 6. Measured on the full
step, muon, 186m @ 2048:

| micro-batch | tok/s | peak | days/epoch |
|---|---|---|---|
| 8 | 12,499 | 31.68 GB | 21.9 |
| **6** | **31,894** | 24.21 GB | **8.6** |
| 4 | 27,038 | 16.75 GB | 10.1 |
| 1 | 10,109 | 5.66 GB | 27.1 |

**Full grid, corrected method — THE standing receipt (the micro-batch table above was an earlier partial run of the same method; where they differ by ~2%, this grid supersedes)** (full step incl. Muon, non-power-of-2 micro-batches,
23.69B-token corpus). Best non-thrash config per size:

| preset | config | tok/s | MFU | peak | days/epoch |
|---|---|---|---|---|---|
| `v2_deep_186m` | block 2048, mb 6, no ckpt | 32,639 | 17.4% | 24.2 GB | **8.4** |
| `v2_deep_238m` | block 2048, mb 6, no ckpt | 31,311 | 21.4% | 23.8 GB | **8.8** |
| `v2_deep_542m` | block 2048, mb 16, **ckpt** | 14,294 | 22.2% | 15.3 GB | **19.2** |

- **238m costs only 5% more wall-clock than 186m for 28% more parameters**, and reaches
  higher MFU (21.4% vs 17.4%) because 20L@1024 has better arithmetic intensity than the
  launch-bound 28L@768.
- 542m must use checkpointing: its no-ckpt rows fall off the cliff catastrophically
  (mb 5/6/7 -> 845 / 578 / 503 tok/s at 37-50 GB "allocated", i.e. 325-545 days/epoch).
  With ckpt at mb 16 it is 14,294 tok/s and well-behaved.
- Block 8192 costs 35-40% throughput at every size and buys context the tokenizer already
  provides; 2048 stands.
- mb 7 measures faster than mb 6 at 186m/238m but peaks at 28.1 GB, leaving under 4 GB for
  the val batches, the corpus memmap and allocator fragmentation the synthetic probe does
  not carry. mb 6 is the recommended launch value.

Method rules this produced, for any future capacity search:
- Measure the FULL step including the optimizer, never `--sanity` alone.
- Sweep non-powers-of-two: a halving search cannot land on 6.
- Treat any config peaking above ~85% of VRAM as unusable however fast it
  looks in a fwd/bwd-only probe -- the allocator spills silently and the run
  merely gets slow, with no error to notice.
- Layer isolation on the same shape: forward alone reaches 94.6% MFU, so the
  model is not the problem; throughput collapses only as memory is added
  (fwd+bwd 11.6%, +adamw 5.3%, +muon 1.9% at the thrashing micro-batch).

- **Activation checkpointing is MANDATORY at block 8192** and the no-ckpt
  advice (SUGGESTIONS / the v2 research verdicts) holds only at 2048. Weights,
  grads and optimizer state are just 1.39 / 1.78 / 4.04 GB for the three
  presets -- activations dominate and scale with seq_len x layers.
- **Block 2048 is the recommended launch shape**: under the v2 tokenizer it
  carries ~1444 words (~4,945 v1-token-equivalents), inside the researched
  4k-8k target band, at full speed. Block 8192 overshoots the band (~19.8k
  equivalents) and pays 30-40% for the checkpointing it would then require.
- `--block` defaults to **1024**: a launch that omits it trains at 1024 no
  matter what the preset's `max_seq_len` says.

### Launch commands, BRANCHED on the size call

The flags are not shared across sizes. `--no-grad-ckpt` is right for 186m/238m
and catastrophic for 542m (the cliff above: 503-845 tok/s, 325-545 days/epoch),
and 542m's micro-batch is 16, not 6. Copying one size's line to the other
produces a run roughly 40x slower with nothing in the output to say so. Flag
surface verified against `pretrain_enigma.py --help` 2026-07-23.

238m -- wall-clock optimal (8.8 days/epoch):

    python pretrain_enigma.py --size v2_deep_238m --optimizer muon \
      --schedule wsd_sqrt --sdpa-backend cudnn --no-grad-ckpt \
      --block 2048 --micro-batch 6 \
      --tokens-bin data/pretrain/tokens_v2.bin \
      --out models/enigma_v2_238m --seed <N> --archive-every <N>

542m -- largest the 5090 sanely trains (19.2 days/epoch). Note BOTH changes:
drop `--no-grad-ckpt` (checkpointing is mandatory here) and raise the
micro-batch to 16:

    python pretrain_enigma.py --size v2_deep_542m --optimizer muon \
      --schedule wsd_sqrt --sdpa-backend cudnn \
      --block 2048 --micro-batch 16 \
      --tokens-bin data/pretrain/tokens_v2.bin \
      --out models/enigma_v2_542m --seed <N> --archive-every <N>

- `--tokens-bin` defaults to the v1 `tokens.bin`: pass the v2 corpus
  EXPLICITLY or the run trains the new architecture on the old tokenization.
- Size `--archive-every` so the decay tail leaves ~10 archives; post-hoc EMA
  has nothing to average otherwise, and an EMA checkpoint is `--init-from`
  only (it carries no optimizer state, so `--resume` refuses it).
- There is NO rope-theta flag: theta comes from the preset (all three
  `v2_deep_*` carry 500000). Changing it means editing `model_presets.py`,
  which is a lineage decision, not a launch knob.

## 7.95 THE TRAINING BLOCK — everything that trains, deferred to the end, in order

> RULED 2026-07-24: anything that trains the AI waits as long as possible and
> runs as one consolidated block. Local 5090 only (rental rejected same day).
> Non-training prerequisites run first, whenever convenient.

Prerequisites (not training):
- P1. Seal the locked probes (validate, seal manifest, record drop counts +
  shas + scorecards in EVAL_REDESIGN). User's call on timing.
- P2. v5/v8 locked re-measure (`--port 8123`, throwaway `--memory-dir`,
  `--transcript` OUTSIDE the repo) = the baseline v2 must beat. Runs before
  any vocab adoption. Needs the serve-launch permission line (user's hand).

The block, in execution order:
- T1. Corpus prep (~1 day, mostly CPU): quality-score the raw third
  (edu-classifier or DCLM swap), add code+math (FineMath collector was never
  run), add short conversational register (chat was left entirely to SFT
  last lineage), 5-10 paraphrase variants of every must-know fact IN the
  corpus, decay-tail annealing set (~2-3B best tokens); then the rust
  retokenize (~42 min). Decide doc-boundary attention masking here.
- T2. 10k-step probe pretrain (hours): first val-loss receipt for the v2
  lineage + live shakeout of archive cadence. The only v2 runs on disk are
  throughput probes.
- T3. Full v2 pretrain (5090; size = user's call at launch -- 238m 8.8
  d/epoch or 542m 19.2 d/epoch; commands above, flags BRANCH on size).
- T4. SFT regen riding the bake (data work, minutes-hours of GPU):
  multi-turn, <think> reasoning, math re-enabled (per-digit vocab kills the
  old disable reason), widened knowledge, DPO pairs beyond identity, the
  contradiction/correction shape, teachings.jsonl dual-routed into the facts
  stream, identity anchors rewritten to actual capabilities, the
  ALWAYS-OFFERED built-in block (router gates retire here, ruled
  2026-07-24), the client-system "Available tools:" shape, image turns
  carrying system/tools/memory blocks, URL-bearing records now kept,
  trained-tool-name list pruned to what has a runtime, `--vocab`/`--block`
  passed explicitly, finetune `--block` raised.
- T5. DPO/polish pass (safe recipe: lr 5e-7 x 1 epoch).
- T6. Gate: locked eval vs the P2 baseline -> beat aggregate with no
  category floor regression = adopt + flip the vocab default; ambiguous =
  user's call.
- T7. Post-adoption organ training, in order: vision stage-2 VQA
  (needs the image-domain pick), ears distill + align (needs the
  whisper-base teacher download), then the far-future own TTS / own
  image-gen transplants.

Serving stays FROM-SCRATCH (ruled 2026-07-24): the llama.cpp/GGUF pivot is
REJECTED -- her serving path is our own code. Consequence: the vendored
`enigma_engine/bin/llama-server/` binary (+~1 GB DLLs) is dead weight,
pending a deletion decision; the deferred eager-path optimizations
(enable_gqa, fused RMSNorm, CUDA graphs) are back on the table as
future serving work.

## 8. Long-term (Phase 7 / embodiment; weeks of GPU)

- [x] New tokenizer — DONE 2026-07-20: v2 vocab 16,366 rows kills the
  standalone-space waste (25.5% -> 0.0%) and splits digits per-character;
  2.41x chars/token. Corpus retokenized to `data/pretrain/tokens_v2.bin`
  (23,694,200,666 tokens uint16); v1 `tokens.bin` untouched.
- [ ] Length extension block 1024 -> 2048+ (Phase 4) — **OBSOLETE IF THE v2
  PRETRAIN PROCEEDS** (v2 = 2.41x text per token + native 8192 context in the
  `v2_deep_*` presets). Run only if the v2 go/no-go lands on "no".
- [ ] Deeper-thinner 350-700M architecture — the 5090 can carry it. Presets
  `v2_deep_186m`/`238m`/`542m` are BUILT and opt-in; the open steps are the
  size call and the pretrain launch.
- [ ] Embodiment: tool-executor bridge + avatar bus (`ws://127.0.0.1:8765`) —
  work lives in the Enigma Avatar repo.
- [ ] Training sim -> trajectory logs -> real-time game play (FNAF target).
- [ ] Video: organ-tier (frame-sample -> describe -> summarize) buildable ANY
  time; native video needs Phase 4 first.
