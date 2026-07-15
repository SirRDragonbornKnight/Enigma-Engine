# Enigma Engine — Backlog

> Consolidated open work, newest snapshot 2026-07-15. Sources: the 4-reviewer
> methods audit (verdicts in `ROADMAP.md` "Update 2026-07-15"), the
> ultrareview ledger (`ULTRAREVIEW_2026-07-12.md`), and the dormant-code audit
> (`AUDIT_2026-07-13.md`). Priority = leverage x confidence, not size.
> Status: [ ] open  [~] in progress (uncommitted)  [x] done this arc.

---

## 0. In flight right now (this session, UNCOMMITTED + UNTESTED)

- [~] **SFT val-split dedup** (`finetune_enigma.py`) — val was drawing from the
  same oversampled duplicate lines as train, so val loss read optimistic. Now
  fills val from records whose ids appear exactly once. NEEDS: suite run, then commit.
- [~] **Merge `teach_pairs.jsonl` into DPO** (`make_dpo_data.py`) — user `/fix`
  corrections now fold into preference data (x3, behind the probe holdout).
  NEEDS: suite run, then commit.
- [ ] **DPO val grouping by prompt** (`dpo_enigma.py`) — "100% val accuracy" is 8
  template-sharing pairs; group the split by prompt so val measures
  generalization. NOT STARTED.
- [ ] **Eval probes 29 -> ~90** (`data/eval/behavior_probes.jsonl`) — categories
  gate on ~4 probes; one flaky answer swings 25 pts. Agent authoring this
  DIED on Fable credits; re-run under a model with budget. NOT DONE.

---

## 1. Correctness / measurement instruments (high leverage, small)

- [ ] Encoder **persistence bug (FATAL, blocks Phase 4.5)** — `train_vision`/
  `train_audio` train the encoders but never SAVE them (checkpoint holds only
  the LM); serve has no load path. A week of native-eye training would
  evaporate on exit. Fix BEFORE any encoder GPU time. `training.py` `_save_checkpoint`.
- [x] Repetition-penalty scope — was penalizing the prompt, suppressing her own
  primed vocabulary (ultrareview #9). Fixed + regression-tested (`b75ed617`).
- [x] Eval memory-store clear (#30) + golden-eval EOS strip (#12) — fixed (`fe5359a7`).
- [ ] Packing without doc-boundary attention masking — conversations attend
  across packed neighbors. Small effect at 182M/1024; INVESTIGATE only if
  context-bleed shows in chat.

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

- [ ] **Drop OpenThoughts3** — 1,000 records, 100% silently dropped at build
  (median completion ~14.5k tokens >> block 1024). Pure waste in
  `combined_finetune.jsonl`.
- [ ] **Rebuild the diet** — Dolly (73% of general) trains extract-from-context,
  not recall. Adopt small-model-native sets: SmolTalk2, Everyday-Conversations,
  No-Robots, + short-answer TriviaQA / Natural-Questions. Cap completions
  ~600 chars so they fit the block. Target 60-100k SHORT records.
- [ ] **Facts need many-format CONTINUED PRETRAINING** — SFT surfaces knowledge,
  it cannot install it (the Jupiter/Saturn phrasing-brittleness). Expand each
  `knowledge_corpus.py` fact into statement + QA + cloze variants and fold into
  a short continued-pretrain pass (new checkpoint dir). Highest-leverage move
  on the factual ceiling.
- [ ] **knowledge_corpus format mixing** — currently Q x rotating-A only; add
  declarative + cloze + fact-in-context per fact.
- [ ] **Teach-loop auto-augment** — single-phrasing corrections at x8 memorize
  strings (and amplify a WRONG teaching). Auto-expand each `/fix` into 3+
  paraphrases + a statement twin; drop to ~x4; add a confirm-before-bake step.
- [x] Low-quality gate (URLs/HTML/encoding/loops; profanity NOT filtered per
  ruling) — done (`d0dd527e`).
- [x] Knowledge weight x2 -> x5 — done (`43870254`).

## 4. Phase 4.5 — Owned organs (the huge project; ~1-2 weeks GPU)

> Full ordered plan in `ROADMAP.md` Phase 4.5. Retire borrowed backbones one at
> a time; teachers used OFFLINE during distillation only.

- [ ] 1. Encoder persistence (see 1 above) — blocks everything.
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

- [ ] `adaptive_trainer.py` — orphaned (registered but dispatch rejects it).
  Keep or delete — user call.
- [ ] Unused deps `SpeechRecognition` + `sounddevice` (nothing imports them).
- [ ] `data/sft/math.jsonl` — orphan file, neither written nor read.
- [ ] `rl_training.py:1873` — guarded caller of the deleted `sentiment` module.
- [ ] Config naming: loader looks for `~/.enigma_engine/config.json` but
  everything else uses `forge_config.json`; that file never existed.
- [ ] **Scratch checkpoints ~17 GB** — `models/enigma_{sft,dpo}_v{2,3,4,5}`.
  v5 is adopted + backed up; v2-v4 are prune-safe on your word.
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
