# Phase 7 Gate — measured ceilings and the remake charter

**Gate status: OPEN — v2 is IN PROGRESS.** Superseded as a decision
2026-07-20 by ROADMAP's Phase 7 section: the v2 prefix landed and the v2
pretrain is the next GPU spend, ahead of Phases 4/5. The baseline it needed
is measured (56/120 v8, 55/120 v5 on the sealed locked set, 2026-07-27 —
table and receipts in `EVAL_REDESIGN.md`, which owns all scorecard numbers);
T1 corpus prep is DONE (tokens_v2b.bin, 2026-07-28), T2 is CLOSED (LR
sweep, 2026-07-29: 3e-3 measured), and both launch gates are RULED
2026-07-30: **Gate B = `v2_deep_238m`**, **corpus rebuild before T3**
(records at `BACKLOG.md` items 7 and 11). What remains before launch is the
rebuild itself plus the T3 pre-flight.

What stays live in this file: the measured-ceilings ledger (the written list
of things Phase 2-5 Enigma provably cannot do, each with its receipt — the
evidence the gate opened on) and the remake charter the v2 lineage inherits.
The gate's scorecard narration moved to
`_archive/PHASE7_GATE_SCORECARD_2026-07.md` when EVAL_REDESIGN became the one
owner of eval numbers.

Current lineage: 182M params, vocab 4718, block 1024, val ppl 3.5
(pretrain DONE 2026-07-03, 287,882 steps / 56.6B tokens).

## Measured ceilings (cannot fix with more SFT data at 182M)

1. **In-weights arithmetic: 0%.** The tokenizer splits numbers
   inconsistently ('56' -> ['5','6'] but '15' -> ['15'], '100' -> ['1','00']),
   so digit-wise computation is unlearnable. Trained on math she emitted
   confident wrong answers ("7 times 8 is a number in the square root of 2").
   BYPASSED 2026-07-05 with the server-side `calculate` tool (math 100% via
   routing) — but the wall itself stands, and word-number asks ("seven times
   eight", no digits) miss the tool's intent gate.
2. **Factual recall: ~50% and capacity-bound.** "Largest planet" -> Pluto /
   Neptune / Mercury across runs; "days in a week" -> 49 / 29 / "one day, and
   one week". These are knowledge-capacity failures, not format failures —
   the same runs hold identity and tools fine. More SFT rounds did not move
   it (rounds 7-9, 2026-07-05). Partially bypassable per-fact via the
   `remember` tool / teachings.jsonl, not in general.
3. **Novel-attack follow-through rambles.** The trained "No" reflex
   generalizes (sycophantic agreement is gone), but the sentence after the
   denial can wander into salad (Copilot/MapReduce fragments) on attacks far
   from the training distribution. Capability boundary; paraphrase coverage
   helped only partially.
4. **Identity ceiling ~83% on held-out phrasings.** The last ~17% is
   run-to-run noise at this scale (measured: val wobbles 0.687-0.759 across
   identical-recipe runs and single probes flip). Diversity fixed 17% -> 83%;
   nothing data-shaped has moved it past that.
5. **Block 1024.** Multi-turn depth, long tool chains, and memory-injection
   headroom all compete for one small window; 74% of the distill corpus is
   unusable as SFT data. Phase 4 (RoPE theta raise + continued pretrain +
   re-SFT) could retrofit this WITHOUT Phase 7, but ROADMAP marks Phase 4
   OBSOLETE if the v2 pretrain proceeds — v2's tokenizer and native context
   reach the same place, so this is no longer the first thing to attempt.
6. **Plain-value corrections accumulate instead of replacing (RULED
   2026-07-24: coexist by default).** Supersede keys on subject + value-kind;
   namings (by value head or by the `name` attribute), measures, and the
   single-valued verb relations replace, and everything else COEXISTS —
   `My car is red.` + `My car is electric.` are two facts, and the ruling
   chose keeping both over guessing which one a shared coarse kind was
   "correcting" (a wrong supersede destroys a fact; a kept duplicate is
   merely outranked at retrieval). The accepted residual: a corrected mood
   or colour leaves the stale value in the store (`mood is happy` +
   `mood is sad` both retrievable). The lexical fallback still runs at
   `_SUPERSEDE_MIN = 0.75` only for texts the key parser declines. Both
   directions are test-pinned in the convergence battery, including the
   red/electric coexist case. Escaping the residual needs semantics — a
   smarter store or a model big enough to resolve the contradiction itself.

## Remake decisions (design of the next generation, when the gate opens)

- **Tokenizer first — the one provably wrong foundation.** ~16-32k vocab,
  GPT-2-style leading-space merge (26.6% of corpus tokens — 29.5% on the
  2026-07-16 English-sample measure — are standalone
  spaces — a quarter of every context and training FLOP), digit-consistent
  number handling (kills ceiling #1 at the root). Requires retokenizing the
  raw sources with `pretokenize_data.py` (rebuilds the 227 GB / 211 GiB
  tokens.bin).
- **350-700M params, deeper-thinner.** Ceilings #2-#4 are capacity walls;
  the 5090 (32GB) carries this size. HRM stays parked (ARC Prize critique).
- **Native block 2048-4096** from step 0. Intra-document attention masking
  was part of this line as written; RULED 2026-07-27: skipped for the v2
  run (negligible at this block by Meta's own measurement, and not worth
  new hot-path mask plumbing before a days-long run) and REQUIRED at the
  future length-extension anneal, where it genuinely matters.
- **Muon + WSD from step 0** (flags already exist). WSD so a good run can be
  extended without cosine's pre-committed horizon.
- **Pretraining data: include worked arithmetic** (the FineMath collector
  was never run — she read ABOUT math without seeing it done) **and short
  conversational text** (chat register was left entirely to SFT).

## Process rules the remake inherits (learned the expensive way)

- Held-out eval harness BEFORE the first training run — "9/9 identity" once
  meant memorized probe strings, exposed only by held-out paraphrases.
- Paraphrase diversity is a data rule, not a fix (repetition memorizes;
  17% -> 83%).
- Tools-over-weights decided upfront: the model ROUTES what it cannot hold
  (math, memory, lookup) and HOLDS voice/values/judgment. Every serve-side
  injection format gets a matching SFT slice (she ignored "Things you
  remember:" until trained on it). The per-built-in intent gates were
  OVERTURNED 2026-07-24 — they retire at the v2 regen in favor of the
  always-offered built-in block (verdict in ROADMAP; execution owned by
  `BACKLOG.md` §7.95 T4).
- Run-stamped checkpoint directories with SHA256 receipts — rounds
  overwrote model.pth all week; the peaks survived by manual backup only.
- QA gates from day 0: foreign-identity purge, boilerplate filter,
  eval-leak guard (each was retrofitted after it bit).

## What does NOT change

From-scratch ethos (own architecture, tokenizer, weights — provably nobody's
fine-tune). One chat renderer shared by train and serve. Evals as code,
gating every run. The finished lineage stays immutable; new runs get new
directories. Small-model-plus-tools as the shape of the thing.
