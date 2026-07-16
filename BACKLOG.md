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

## 1. Correctness / measurement instruments (high leverage, small)

- [x] Encoder **persistence bug** — FIXED `f9ec5184`: `_save_checkpoint` takes
  encoder/optimizer overrides, vision/audio save their encoder + the LOCAL
  optimizer that actually stepped, `_load_encoder_checkpoint` resumes and
  REFUSES text-only checkpoints; 6 tests. Residual open: serve-side
  native-encoder load path (Phase 4.5 step 5).
- [ ] **`--tokens-bin` is not resume-locked** (`pretrain_enigma.py`; final
  audit 2026-07-16 M1) — the checkpoint schedule does not record the corpus
  path, so `--resume` without re-passing the flag silently finishes a facts
  run on the default 56.6B corpus. Record tokens_bin in the schedule and
  restore it on resume.
- [ ] **`group_split` can empty train / overshoot val** (`dpo_enigma.py`;
  final audit M2) — a single-prompt dataset (small teach_pairs.jsonl) puts
  everything in val and crashes `_batchify([])`; a giant first group blows
  past val_cap. Guard: never let train go empty; deal groups smallest-first
  or split oversized groups.
- [ ] **Facts-corpus val contract** (`make_facts_pretrain_data.py`; final
  audit M3) — pretrain's val window (n//100 = 600k at the 60M default) is
  larger than the 500k pure-replay tail, and a fact doc can spill past
  mixed_end; [val] reads ~0.4% fact tokens. Fix: `val_reserve =
  max(arg, target // 100)`, stop fact docs at mixed_end - max_doc_len, and
  put `--val-general-end 0` in the documented command.
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

> 11 confirmed open 2026-07-14; all in the DORMANT training arsenal (LoRA / RL /
> queue), so none affect the live serve/SFT/DPO path today. Fix if/when that
> code is reconnected.

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
- [ ] **Retrain adoption (v6/v7/v8)** — the new-diet cycle is measured but NOT
  adopted; `models/enigma_dpo` still serves v5. On 90 probes: v5 70/90,
  v6 76/90 (factual 95%, adversarial 92%, but memory 3/8 + identity 14/18 —
  dilution), v7 72/90 (repetition knobs did not fix it), v8 = wider memory/
  identity coverage at moderate fractions (`8fdaf541`) — trained, EVAL PENDING.
  Decide adoption from the v8 scorecard; if memory/identity still gate-fail,
  next levers are more memory-read surface diversity and a small identity
  boost, not bigger repeats.
- [ ] **Teach-loop auto-augment** — single-phrasing corrections at x8 memorize
  strings (and amplify a WRONG teaching). Auto-expand each `/fix` into 3+
  paraphrases + a statement twin; drop to ~x4; add a confirm-before-bake step.
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

- [ ] `enigma_engine/core/adaptive_trainer.py` — orphaned (registered but
  dispatch rejects it). Keep or delete — user call.
- [ ] Unused deps `SpeechRecognition` + `sounddevice` (nothing imports them).
- [ ] `data/sft/math.jsonl` — orphan file, neither written nor read.
- [ ] `enigma_engine/core/rl_training.py:1873` — guarded caller of the deleted
  `sentiment` module.
- [ ] Config naming: the loader searches `forge_config.json` (exists, repo
  root, and is found) plus a never-created `~/.enigma_engine/config.json` —
  naming inconsistency only, nothing broken.
- [ ] **Scratch checkpoints now span v2-v8** (each sft_v* ~6.6 GB, each dpo_v*
  ~2.2 GB, plus enigma_pretrain_facts ~6.6 GB) — re-measure before pruning;
  v5 is adopted + backed up; v2-v4 (~26 GB) remain prune-safe on your word,
  v6-v8 hold this arc's evidence until an adoption call is made.
- [ ] Docs: `information/training_guide.md` + `quick_commands.md` don't yet
  cover the facts continued-pretrain recipe or the new collector flags
  (--no-robots/--everyday/--triviaqa/--nq-open/--smoltalk2-cap); CLAUDE.md
  pipeline line should mention the optional facts hop (final audit
  2026-07-16, findings 18-20).
- [ ] Teach tool nits (final audit m3/m4): /good after /fix double-writes the
  same teaching; truncate-after-external-edit can NUL-pad a hand-edited
  jsonl. Both edge-case; fix alongside the auto-augment work.
- [ ] Memory-read data nit (final audit m9): contradictory color facts can
  co-occur in one distractor block (green vs orange) — de-conflict attribute
  domains when widening further.
- [ ] `teachings.jsonl` still the untouched example template — YOUR channel to
  author (values / personal facts); bakes in at x8.

## 8. Long-term (Phase 7 / embodiment; weeks of GPU)

- [ ] New tokenizer — fix the 26.6% standalone-space waste + digit-splitting
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
