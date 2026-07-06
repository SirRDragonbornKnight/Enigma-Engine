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

## Update 2026-07-06 (measured, `eval_behavior.py` held-out scorecard)

- **Phase 1 DONE, Phase 2 EXIT CRITERIA MET**: 26/29 (90%) — identity 83%,
  adversarial/tool/restraint 100%, math 100% via the `calculate` built-in,
  memory 4/4 end-to-end via the `remember` built-in (she saves what you tell
  her, supersedes corrections, recalls across conversations). Suite 404.
- User-teaching channels live: `remember` tool (facts from chat, instant) and
  `teachings.jsonl` (user-authored facts baked into weights, ~10-min loop).
- Best checkpoint backed up: `Enigma Backups\enigma_sft_memory_pass` (SHA256).
- Avatar wiring: dropped at user direction (2026-07-06) — Phase 5 is Odysseus
  + memory growth only.
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
  recurate the 8 dropped Qwen-era anchors, scale to a few hundred). Claude builds
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
