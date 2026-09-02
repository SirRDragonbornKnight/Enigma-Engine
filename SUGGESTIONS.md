# Landscape research + principles — current as of 2026-07-26

> What survives of the 2026-06-11 strategy doc. Its "Roadmap (mouth & hands)"
> section was superseded by `ROADMAP.md` (which says so in its own header) and
> its "shape" section duplicated the repo identity that `README.md` and
> `CLAUDE.md` own — both were cut 2026-07-26; git history keeps them. What
> remains is the piece other docs still cite: the 2026 landscape verdicts
> (BACKLOG's block-size and optimizer decisions lean on them) and the
> principles.

## 2026 landscape check (researched 2026-06-11)

Verdict: nothing in the live run is wrong. The stack matches current
small-model practice (GQA + qk-norm + SwiGLU at ~8/3·dim + RMSNorm + tied
embeddings + RoPE; bf16 autocast + torch.compile; AdamW 0.9/0.95 with decay
only on ≥2-D tensors), and the frozen-weights + external-memory/tools learning
model is the 2026 consensus, not a compromise. The advances below are queued
for FUTURE decisions — none justify touching the paused run mid-schedule:

- **Muon optimizer** — production-proven in 2025-26 (Kimi K2 1T, GLM,
  Megatron support; ~1.3-2× data/compute efficiency vs AdamW). Candidate for
  the instruct pass and any next pretrain. Never mid-run.
  **BUILT 06-11:** `pretrain_enigma.py --optimizer muon`, flag-gated, default
  `adamw` = the live path; resume optimizer-mismatch fails loudly.
- **WSD / decay-to-zero schedules** beat fixed-budget cosine when training
  might continue past the planned budget. Adopt for the NEXT run; the current
  cosine run keeps its recorded schedule.
  **BUILT 06-11:** `--schedule wsd --wsd-decay-frac 0.1`, flag-gated, default
  `cosine` bit-identical to the live formula (regression-tested).
- **Multi-epoch data** (data-constrained scaling laws): up to ~4 epochs of
  the same corpus ≈ fresh tokens; meaningful gains decay around 16. Our run
  is single-epoch (56.6B) — after step 287,882 a continuation over the same
  corpus is legitimate, modern, and the cheapest capability lever we have.
- **Depth vs width:** 2026 small models run deeper-thinner (SmolLM2-135M is
  30 layers; ours is 16×1024). A next-architecture consideration, not an
  error — wider buys throughput on a single consumer GPU.
- **Intra-document attention masking** (Llama 3): negligible effect at block
  1024 by Meta's own measurement; becomes important IF we do the
  length-extension anneal. The 2025 extension recipe is settled: raise RoPE θ
  + continued pretraining on long documents (<10B tokens) — fold both into
  that decision.
- **min-p sampling** is now in every major serving stack (llama.cpp defaults
  it at 0.1); cheap optional add to serve/sample for high-temperature
  stability. Measured benefit is contested — nicety, not a need.
  **BUILT 06-11:** plumbed through generate/generate_stream + the server's
  request models (`min_p`, default 0 = off; the filter already lived in
  `sample_next_token`).
- **Tokenizer:** small vocab is *defensible* at our compute scale (vocab
  scaling laws: embedding FLOPs saved fund longer training; big vocabs
  underfit rare tokens). The standalone-space token (26.6% of the stream;
  29.5% on the 2026-07-16 English-sample re-measure)
  remains our real inefficiency — a next-generation tokenizer should merge
  leading spaces GPT-2-style before any re-pretrain. Not fixable for this
  lineage: retokenizing means a new run. (The v2 vocab that does exactly this
  landed 2026-07-20 — `TOKENIZER_V2_SPEC.md`.)

## Principles

- **Black box:** local-first; nothing private leaves this PC. Privacy is the
  invariant, not zero egress — search queries go out via local SearXNG.
- **Ships to other machines:** no hardcoded user paths; degrade gracefully
  with NO model present.
- **Keep ideas, not code** — git is the archive; verify before delete.
- **The training arm is the moat** — meaning the LIVE bespoke scripts
  (pretrain/finetune/dpo/teach + the data pipeline), not bulk. The dormant
  Forge monolith was deleted 2026-07-18; future methods land as small
  bespoke scripts in the `dpo_enigma.py` pattern.
