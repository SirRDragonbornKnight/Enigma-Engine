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
  (BLIP/whisper/sd-turbo/SAPI) are SCAFFOLDING; every organ has a transplant
  path to own-trained weights — see Phase 4.5.
- **No language censorship.** Data gates drop structural junk only, never words.
- **Clarity over character.** Understandable sentences are the bar; "Enigma"
  is a personality, not an excuse for word salad.
- **She is taught by chatting.** `teach_enigma.py` corrections are the primary
  values/facts channel (x8 in every bake).

**Methods-audit verdicts (2026-07-15):**
- KEEP (validated): restart-from-pretrain each SFT cycle; oversample weights;
  regex tool gates; clarity sampling defaults (one min_p-only A/B queued).
- FIX instruments: SFT val split is contaminated by oversample duplicates
  (dedup before split — val loss reads optimistic today); DPO "100% val" is 8
  template-sharing pairs (group val by prompt); 29 eval probes gate on ~4 per
  category (target ~90 with paraphrase twins).
- FIX fatal (code): `train_vision`/`train_audio` never SAVE the trained
  encoders (checkpoint holds only the LM) and serve cannot load them — a
  successful native-eye run evaporates on exit. Encoders train FULLY (the old
  "trains frozen" doc line was wrong).
- DATA: OpenThoughts3 (1,000 recs) is 100% dead weight — median completion
  ~14.5k tokens vs block 1024, every record silently dropped at build. Dolly
  (73% of general) trains extract-from-context, not recall. Rebuild the diet
  around SmolTalk2 / Everyday-Conversations / No-Robots + TriviaQA/NQ short
  answers; target 60-100k SHORT records. Facts want many-format exposure
  (statement/QA/cloze) in CONTINUED PRETRAINING — SFT surfaces knowledge, it
  cannot install it (the Jupiter/Saturn phrasing-brittleness receipt).
- TEACH LOOP: auto-augment corrections (paraphrases + statement twin, ~x4);
  merge `teach_pairs.jsonl` into DPO behind the probe filter.
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

## Phase 0 — Lock in today (minutes; do first)

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
- Behavior evals AS CODE (extend `_audit_eval.py`, don't duplicate): tool-emission
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
4. **Align:** `train_vision` over the 558k pairs (encoder unfrozen, LM
   frozen; add real batching first — the current loop is batch-1), optional
   final pass unfreezing the last 2-4 LM layers.
5. **Wire and retire:** image begin/end tokens (ids 4724+ reserved, 12
   free), serve ingestion, delete the BLIP path. Her eyes are hers.
6. **Her ears (same shape, ~3 days):** write `collect_audio_data.py`
   (LibriSpeech-clean-100), distill the whisper encoder into her
   AudioEncoder, `train_audio`, wire, retire whisper.
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

`enigma_engine/core/rl_training.py` already implements RewardModel + RLHF (RL-B)
+ self-play (RL-C). Realistic at 182M: DPO on format/tone/values preferences.
The "won't turn evil" property comes from Phase 1c authorship more than RL.

**DONE 2026-07-06 (measured):** `make_dpo_data.py` (176 authored-voice vs
measured-failure-mode pairs) + `dpo_enigma.py` (policy + frozen reference,
render_training masks). At lr 2e-6 x2 epochs DPO OVER-OPTIMIZED and damaged
her (identity 83->50%, factual 50->0%) — at 182M DPO is a nudge or a wrecking
ball. At lr 5e-7 x1 epoch: preference accuracy 100% with margin AND the full
scorecard held (26/29, all gates PASS) — ADOPTED (`models/enigma_dpo`, now
what Start-Enigma serves; revert = point it back at `models/enigma_sft`).

## Phase 7 — The next generation (the big fork; weeks of GPU)

Only when the current lineage hits a measured ceiling:
- New tokenizer (fix the 26.6% standalone-space waste; ~16-32k vocab, GPT-2-style
  leading-space merge) — requires retokenizing the 210GB corpus (`pretokenize_data.py`).
- Deeper-thinner architecture, Muon + WSD from step 0, native block 2048,
  350-700M params — the 5090 (32GB) can carry it.
- HRM stays a PARKED experiment (heed the ARC Prize critique).
- **Gate: a written list of things Phase 2-5 Enigma provably cannot do —
  that list now lives in `PHASE7_GATE.md` (started 2026-07-06, receipts
  included; gate NOT yet open — Phase 4/5 first).**

---

## Through-line rules

- Every phase has a measurable exit; GPU time is never spent before a data/probe
  receipt.
- The finished base lineage is immutable — new runs get new directories.
- Engines fail honestly; evals are code, not vibes; ground every load-bearing
  number in a fresh measurement.
