# Enigma Roadmap — written 2026-07-03, the day pretraining finished

> The full plan, phase by phase. Supersedes the "Roadmap (mouth & hands)" section
> of `SUGGESTIONS.md` (kept for its landscape research + principles, which still
> hold). Status lines here are MEASURED, not asserted — re-measure before
> trusting any number that matters.

## North star

A local, from-scratch, personally-aligned AI — Jarvis-class companion that runs
entirely on this machine, drives the avatar, uses tools, remembers, and provably
won't turn evil because the user authored her values. No cloud, no wrapper, no
Muppet.

## Where she stands (2026-07-03, measured)

- Base model DONE: 182M params, 287,882 steps, 56.6B tokens, val ppl 3.5
  (`models/enigma_pretrain_large/model.pth`; backed up with SHA256 receipts at
  `C:\Users\SirKn\Enigma Backups\enigma_pretrain_large_final\`).
- Pipeline PROVEN end-to-end: pretrain -> SFT -> serve. 10 confirmed bugs fixed
  2026-07-03 (warmup clamp, schedule-lock seed/val_frac, salted hash, final-save
  guards, tool-args echo-back, max_ids hard cap, control-token injection,
  dangling-span flush, raw-call surfacing + honest finish_reason, fsync +
  corrupt-latest fallback). 358 tests green.
- First instruct checkpoint alive (`models/enigma_sft/model.pth`, 32 steps):
  speaks in-format, stops at turn boundaries, serve auto-detects instruct mode.
- CANNOT yet: use tools reliably (29 training examples), hold identity (122
  anchors), converse long (block 1024).
- **The bottleneck is no longer code — it is data.**

## Update 2026-07-15 — user rulings + methods audit (4-reviewer pass)

**User rulings, binding on all future work:**
- **Privacy is absolute.** serve is offline-by-default (`HF_HUB_OFFLINE` etc.;
  `--allow-downloads` exists solely for an explicit first weight fetch).
  Nothing she hears or says leaves this machine. Ever.
- **Her abilities must become HER OWN.** Borrowed organ backbones
  (whisper/sd-turbo/Kokoro — BLIP is already retired for her own ViT) are
  SCAFFOLDING; every organ has a transplant path to own-trained weights —
  see Phase 4.5.
- **No language censorship.** Data gates drop structural junk only, never words.
- **Clarity over character.** Understandable sentences are the bar; "Enigma"
  is a personality, not an excuse for word salad.
- **She is taught by chatting.** `teach_enigma.py` corrections are the primary
  values/facts channel (x8 in every bake).

**Methods-audit verdicts (2026-07-15):**
- KEEP (validated): restart-from-pretrain each SFT cycle; oversample weights;
  regex tool gates; clarity sampling defaults (one min_p-only A/B queued).
  **The "regex tool gates" verdict is CONTESTED as of the 2026-07-22 router
  audit** — the gates were measured defective in both directions (false-miss
  "Draw me a dragon" / word-number math; false-fire "Do not draw anything"),
  and a miss means no offer, no gradient, and nothing the eval can see. The
  proposed replacement is a fixed always-offered built-in block baked into the
  v2 SFT regen. This is a pending user ruling; the line stays until it lands.
- FIX instruments: SFT val split is contaminated by oversample duplicates
  (dedup before split — FIXED `47f557ae`); DPO "100% val" is 8
  template-sharing pairs (group val by prompt — FIXED `fd2776d1`); 29 eval
  probes gate on ~4 per category (FIXED `090e6644` — 90 machine-vetted
  probes). Second-pass audit 2026-07-15: the grader itself matched keywords
  as substrings ("own" passed on "known") and one perfect trained answer
  failed its probe — both FIXED `bacc7473` (word-boundary grading + probe
  keys); v5 re-measured 27/29 on the old suite, then 70/90 (78%) FAIL on the
  expanded suite — THAT is the honest baseline the retrains must beat.
- FIX fatal (code): `train_vision`/`train_audio` never SAVE the trained
  encoders (checkpoint holds only the LM) and serve cannot load them — a
  successful native-eye run evaporates on exit. Encoders train FULLY (the old
  "trains frozen" doc line was wrong). SAVE/RESUME/optimizer FIXED `f9ec5184`
  (2026-07-15); serve-side encoder loading remains Phase 4.5 work.
  (Historical: both methods lived on the Forge Trainer. `train_vision` was
  carved into `enigma_engine/training/vision_align.py` on 2026-07-18 keeping
  this fix intact — smoke-verified: the checkpoint carries 45 encoder tensors
  plus the stepped local optimizer. `train_audio` was NOT carved — see the
  Phase 4.5 step-6 gap note.)
- DATA: OpenThoughts3 (1,000 recs) is 100% dead weight — median completion
  ~14.5k tokens vs block 1024, every record silently dropped at build. Dolly
  (73% of general) trains extract-from-context, not recall. Rebuild the diet
  around SmolTalk2 / Everyday-Conversations / No-Robots + TriviaQA/NQ short
  answers; target 60-100k SHORT records. Facts want many-format exposure
  (statement/QA/cloze) in CONTINUED PRETRAINING — SFT surfaces knowledge, it
  cannot install it (the Jupiter/Saturn phrasing-brittleness receipt).
  ALL LANDED same evening: diet `8104e09c` (105,203 pairs), facts pretrain
  `701434be`+`3b553038` -> `models/enigma_pretrain_facts` (factual 13/20 ->
  19/20 on v6). Retrain candidates measured on the 90-probe gate: v6 76/90,
  v7 72/90 (memory/identity dilution — see make_sft_data's measured
  comments), v8 (coverage-widened) 79/90 — the FIRST to pass all seven
  categories. **v8 ADOPTED 2026-07-16** (`models/enigma_dpo/model.pth`, SHA
  receipt `Enigma Backups\enigma_dpo_v8_adopted\`); the v5 backup
  (`enigma_dpo_v5_adopted`) stays as the revert target.
- TEACH LOOP: auto-augment corrections (paraphrases + statement twin, ~x4 —
  DONE 2026-07-16, `teach_enigma.py` augment_teaching + confirm-before-bake
  review; TEACHINGS_REPEAT 8->4); merge `teach_pairs.jsonl` into DPO behind the
  probe filter (DONE `47f557ae`). Second-pass audit: /undo left records baked on disk and
  a second /fix rejected the user's own correction — both FIXED `deb7c182`.
- ORGAN UPGRADES (interim, still borrowed, all pip-only): TTS -> Kokoro-82M;
  ASR -> whisper large-v3-turbo (verify CTranslate2 on sm_120 first); eyes ->
  SmolVLM2 with question-conditioned VQA (captioning throws the user's
  question away); image gen -> sdxl-turbo (a string change in Painter).

## Update 2026-07-06 (measured, `eval_behavior.py` held-out scorecard)

- **Phase 1 DONE, Phase 2 EXIT CRITERIA MET**: 26/29 (90%) — identity 83%,
  adversarial/tool/restraint 100%, math 100% via the `calculate` built-in,
  memory 4/4 end-to-end via the `remember` built-in (she saves what you tell
  her, supersedes corrections, recalls across conversations). Suite 404.
- User-teaching channels live: `remember` tool (facts from chat, instant) and
  `teachings.jsonl` (user-authored facts baked into weights, ~10-min loop).
- Best checkpoint backed up: `Enigma Backups\enigma_sft_memory_pass` (SHA256).
- Avatar wiring: out of scope by user decision (2026-07-06) — Phase 5 is
  Odysseus + memory growth only.
- Measured ceilings + remake design recorded in `PHASE7_GATE.md`.
- CANNOT yet (receipts in `PHASE7_GATE.md`): converse long (block 1024),
  recall broad facts (~50%, capacity), compute without the tool, stay crisp
  under far-out-of-distribution attacks.

---

## Phase 0 — Lock in today (minutes; do first) — CLOSED 2026-07-16
(#1 and #4 verified DONE: repo fully committed + pushed, SUGGESTIONS updated;
#2/#3 subsumed by the receipted v1/v5/v8 backups and the later audits.)

1. Commit the 2026-07-03 work to `engine-refactor` (10 fixes + training_progress.py
   + data rebuild), PR when ready.
2. Back up `models/enigma_sft/model.pth` beside the base backup.
3. Look up deferred refactor items #5/#7/#8 — the "wait for pretrain" gate is open.
4. Update `SUGGESTIONS.md`: pretrain is DONE (doc still says paused at 63.8%).

## Phase 1 — Feed her: SFT data at scale (highest leverage; days)

The old roadmap said it first: "Before the REAL pass: fatten the tool corpus
(29 seed examples) and curate the values data."

- **1a. Tool corpus 29 -> ~1,000.** Programmatic expansion of `make_sft_data.py`:
  the real Modkit/avatar tool surface (avatar_express, avatar_say, see_screen, ...),
  paraphrase variety, multi-turn chains (call -> result -> follow-up), error
  results, restraint at scale. All synthetic, all local.
- **1b. Chat-shaped general data.** `collect_finetuning_data.py` already downloads
  OASST1 (~80k turns) and Dolly 15k — real SHORT conversations, unlike the current
  distill corpus (74% of completions >985 tokens, measured 2026-07-03). Target:
  10k-30k conversations that fit block 1024.
- **1c. Values/identity corpus — the USER'S authorship** (constitutive alignment:
  rewrite the 8 Qwen-era anchors from-scratch-true, scale to a few hundred). Claude builds
  tooling/QA only.
- **1d. QA gate before any GPU time:** refusal-boilerplate scan (proven clean
  2026-07-03: 1 hit in 647 records, and it was a BBQ invitation), dedupe,
  block-fit report, tool/restraint balance.

## Phase 2 — SFT v2 + a real eval harness (hours)

- Train on the fattened mix (2-4 epochs; `--optimizer muon` is queued for exactly
  this pass per the landscape research — flag exists, SFT runs are cheap to redo).
- Behavior evals AS CODE (extend `eval_behavior.py`, don't duplicate): tool-emission
  rate on tool-appropriate asks, restraint rate, identity accuracy, format
  adherence (stops at <|im_end|>), val ppl. Every SFT run gets compared on the
  same probes.
- **Exit criteria: >80% tool emission and >80% restraint on held-out probes.**

## Phase 3 — Cheap base upgrade: multi-epoch continuation (optional; ~12 GPU-days/epoch)

Data-constrained scaling laws (see SUGGESTIONS landscape): up to ~4 epochs over
the same corpus ~= fresh tokens — "the cheapest capability lever we have."
- PROBE FIRST: ~10k steps continuation, measure the val delta, continue only if
  the curve says yes.
- WSD schedule (flag exists), NEW run directory — the finished lineage is never
  touched.

## Phase 4 — Length extension: block 1024 -> 2048+ (days)

**OBSOLETE IF THE v2 PRETRAIN PROCEEDS** — this is a v1-lineage continuation, and
v2 reaches the same place without it: the v2 tokenizer carries 2.41x more text per
token (block 1024 goes ~300 -> ~722 words) and the `v2_deep_*` presets train at a
native 8192 context. Run this ONLY if the v2 go/no-go lands on "no".

Settled 2025 recipe (already researched in SUGGESTIONS): raise RoPE theta +
continued pretraining on long documents (<10B tokens) + intra-document attention
masking, then re-SFT.
- Unlocks: real multi-turn tool loops, memory-injection headroom — and the 74%
  of the distill corpus that is currently too long becomes usable SFT data.
- **Data receipt (measured 2026-07-06, 3.6% sample of tokens.bin):** docs >=2048
  tokens hold 86.5% of all corpus tokens (~49B) — 5x the <10B the recipe needs.
  Median doc 958 tokens, p90 3,741. The long-doc prerequisite is MET; what
  remains is the training work (theta raise + doc-masked continuation + re-SFT).

## Phase 4.5 — Owned organs: the transplant arc (user-ordered 2026-07-15; ~1-2 weeks of 5090 time)

Goal: every sense runs on HER weights; borrowed backbones retire one at a
time. Teachers may be used OFFLINE during training (distillation) — the
weights that result are hers, on her machine, forever.

1. **Persistence first (hours, blocks everything):** save/load encoder
   state_dicts alongside checkpoints; wire serve to load them. Today a
   trained eye evaporates on process exit (audit 2026-07-15).
2. **Vision data:** `collect_vision_data.py` full LLaVA-Pretrain 558k
   (~14 GB, one manual images.zip download).
3. **Her eyes:** distill DINOv2-S patch features into her own ViT-medium
   (~25M, cosine loss, ~1-2 GPU-days). Contrastive-from-scratch is NOT
   viable solo (needs 10M+ pairs); distillation is the honest shortcut that
   still ends in owned weights.
4. **Align: DONE 2026-07-20.** `align_vision.py` ran batched over the pairs
   (encoder unfrozen, LM frozen via `freeze_text_io` so the projection
   targets v8's embedding space). The optional final pass unfreezing the
   last 2-4 LM layers was not run.
5. **Wire and retire: PARTLY DONE.** Serve ingestion is live and the BLIP path
   was deleted 2026-07-17 — her eyes are hers, and captions enter chat as the
   "[image: ...]" TEXT marker. The image begin/end tokens are NOT done: ids
   4724-4735 are still listed as reserved in `chat_format.py`, no delimiter
   constant exists, and `core/eyes.py` says so outright — she has no trained
   image-delimiter tokens, and token-level image injection is the next
   training cycle's work. Also open: the captions are question-blind
   (see BACKLOG item 5).
6. **Her ears (same shape, ~3 days):** `collect_audio_data.py` DONE
   (LibriSpeech-clean-100, 28,539 pairs); `distill_audio_encoder.py` DONE
   (own loop, unaffected by the compression pass) but NOT launched.
   The 2026-07-18 gap (the align step's `Trainer.train_audio` was deleted
   with the Forge trainer) was CLOSED 2026-07-19: `vision_align.py` was
   generalized into `enigma_engine/training/encoder_align.py` — one shared
   `_train_encoder` core with `train_vision`/`train_audio` wrappers — and
   `align_audio.py` is the entry point, batching at `--batch-size 8`
   through a padding-mask collate. What remains is GPU work gated on
   downloading the `openai/whisper-base` teacher (the cached
   Systran/faster-whisper-base is the ASR organ, NOT the teacher), then the
   align run, serve wiring, and retiring whisper.
7. **Her voice (later):** train a small TTS on a chosen voice — in-house
   project; Kokoro is scaffolding until then.
8. **Her imagination (much later):** an own-trained image generator is
   possible but will trail Stable Diffusion for a long time; SD stays the
   tool she WIELDS until the trade is worth making.

**Video:** 196 tokens/frame means 3-4 frames max at block 1024 — organ-level
video (sample frames -> describe -> summarize) is buildable any time; NATIVE
video needs the Phase 4 length extension first. Real-time game vision
(FNAF) = native eyes + Phase 4 + the training sim.

## Phase 5 — Embodiment (the point of it all; mostly glue)

- Tool EXECUTOR bridge: serve emits `tool_calls` -> a small local executor runs
  them (avatar bus ws://127.0.0.1:8765, screen, timers, ...) and loops results
  back into the conversation. The avatar side follows THAT repo's TODO.md.
- Odysseus as her face (`/setup local http://127.0.0.1:8000/v1`).
- Memory store (BM25 v1, already in serve) grows with use.

## Phase 6 — Alignment polish: DPO / self-play (optional; scaffolding exists)

(The old RewardModel/RLHF/self-play scaffolding was deleted in the 2026-07-18
compression pass — git history holds it; any future RL would be a small
bespoke script.) Realistic at 182M: DPO on format/tone/values preferences.
The "won't turn evil" property comes from Phase 1c authorship more than RL.

**DONE 2026-07-06 (measured):** `make_dpo_data.py` (176 authored-voice vs
measured-failure-mode pairs) + `dpo_enigma.py` (policy + frozen reference,
render_training masks). At lr 2e-6 x2 epochs DPO OVER-OPTIMIZED and damaged
her (identity 83->50%, factual 50->0%) — at 182M DPO is a nudge or a wrecking
ball. At lr 5e-7 x1 epoch: preference accuracy 100% with margin AND the full
scorecard held (26/29, all gates PASS) — ADOPTED then, as v1. Superseded:
**v8 is the adopted DPO since 2026-07-16** (79/90 on the 90-probe gate, first
to pass all seven categories; `models/enigma_dpo` holds v8, receipted backup
`Enigma Backups\enigma_dpo_v8_adopted\`; revert targets = v5/v1 backups or
`models/enigma_sft`).

### Candidate next lever: ON-POLICY DISTILLATION (research 2026-07-18, not started)

The biggest post-training idea since our DPO adoption, and it FITS the
owned-weights rule because the teacher exists only at training time — no
runtime wrapping, her weights stay hers.

- **Mechanism:** sample trajectories from HER (on-policy), have a big local
  teacher (e.g. a Qwen3-class model run locally) grade EVERY token via reverse
  KL, train on that dense signal. Sources: Thinking Machines' write-up +
  Qwen3 report numbers (74.4% AIME'24 at ~1,800 GPU-hr vs RL's 67.6% at
  ~17,920 — roughly 10x cheaper than RL, and it beats plain SFT).
- **Why it matters HERE specifically:** its signature strength is preventing
  catastrophic forgetting — domain SFT that crushed instruction-following was
  recovered while KEEPING the new knowledge. That is exactly our recurring
  v2-v8 failure mode (teach her facts, lose identity/voice; every cycle a
  coin-flip on which category regresses).
- **Honest caveat (fact-checked):** every public demonstration is a 4B+
  student, Qwen-heavy. There is NO published sub-1B result — at 182M we would
  be the replication, not the follower. Prototype small and gate it on the
  locked eval before believing it.
- **Shape if built:** a new bespoke script in the `dpo_enigma.py` pattern
  (~200-300 lines: rollout, teacher-logprob, reverse-KL loss, the shared
  chat_format masks) — NOT a revival of the deleted Forge trainer. Keep a
  light DPO/SimPO pass for style on top.
- Also worth knowing (same research pass): current preference-method consensus
  puts SimPO (reference-free, length-normalized — drops the deepcopy'd
  reference model) and KTO (thumbs-up/down data, which the avatar could
  generate naturally) ahead of vanilla DPO; RLVR/GRPO "sharpens rather than
  expands" and is only worth it for narrow verifiable rewards like tool-call
  JSON validity.

## Phase 7 — The next generation (the big fork; weeks of GPU)

This phase is IN PROGRESS: the v2 prefix landed 2026-07-20 and the v2 pretrain is
the next GPU spend, ahead of Phases 4/5.
- New tokenizer — **DONE**: v2 vocab is 16,366 rows (leading-space merges,
  per-digit numbers), measured 2.41x chars/token over v1, and the corpus is
  retokenized to `data/pretrain/tokens_v2.bin` (23,694,200,666 tokens, uint16).
  v1's `tokens.bin` (227 GB / 211 GiB — the "210GB"/"227 GB" figures in older
  notes are this same file in GiB vs GB) is untouched and still feeds the live
  lineage. Scaling-law grounding: TOKENIZER_V2_SPEC.
- Deeper-thinner architecture, Muon + WSD from step 0 (both implemented in
  `core/optim.py`, flag-gated off for lineage compat), QK-norm BEFORE RoPE,
  optional gated attention, 350-700M params — the 5090 (32GB) can carry it.
  **Presets ready**: `v2_deep_186m` / `v2_deep_238m` / `v2_deep_542m`
  (28L@768 / 20L@1024 / 30L@1280, native 8192 context, Peri-LN,
  QK-norm-before-RoPE, olmo2_flat init), all opt-in with v1 defaults untouched.
- HRM stays a PARKED experiment (heed the ARC Prize critique).
- **Gate: a written list of things Phase 2-5 Enigma provably cannot do —
  that list lives in `PHASE7_GATE.md` (started 2026-07-06, receipts included).
  What actually gates the v2 pretrain now is the locked-probe baseline
  (`data/eval/LOCKED_PROBES_AUTHORING.md`) and the size call — not Phases 4/5.**

---

## Through-line rules

- Every phase has a measurable exit; GPU time is never spent before a data/probe
  receipt.
- The finished base lineage is immutable — new runs get new directories.
- Engines fail honestly; evals are code, not vibes; ground every load-bearing
  number in a fresh measurement.
