# Tokenizer v2 redo — spec + cost (2026-07-16)

> Status: SPEC, not started. The Phase-7 foundational fix. Do NOT start GPU work
> until the eval is trustworthy (see `EVAL_REDESIGN.md`) -- without it you can't
> tell a better base from a differently-overfit one. Grounded 2026-07-16.

## Why (measured, not asserted)

Empirical, current tokenizer on a sample of English:
- **1.55 chars/token** (a normal byte-level BPE gets ~3.5-4).
- **29.5% of tokens are bare standalone spaces** (never merged into the
  following word).
- **Digits fragment**: "1000" -> "1","00","0" (inconsistent), which is the root
  of number-recall brittleness (word-numbers are a band-aid).
- **0% leading-space merges**; **vocab only 4718**.

Consequences that everything else works around:
- block-1024 context ~= 250-300 words.
- "56.6B training tokens" ~= only ~15B tokens of real text by a normal
  tokenizer's measure.
- Facts are phrasing-brittle; recall needs a continued-pretrain workaround.

## What changes

1. **Train the new tokenizer.** Machinery already exists (`tokenizer.py:871`,
   default vocab 32000). Requirements:
   - Leading-space merges (GPT-2 "Gdot" convention): space attaches to the
     following word -> kills the 29.5% waste.
   - Consistent digit handling: **per-digit tokens** (predictable arithmetic).
     Grounding (2026-07-18): the small-model evidence favors single-digit
     (nanoGPT-scale ablation arxiv 2510.06824 finds it best, if costlier); the
     often-cited 2402.14903 is GPT-3.5/4-eval-only and does NOT train from
     scratch, so its "3-digit R2L beats L2R" result is about chunking DIRECTION,
     not a case against single-digit. Per-digit is the pick; real math rides the
     calculator tool anyway. The current inconsistent split is the worst option.
   - Vocab: **16k is the better-supported target; 32k only with a size bump.**
     External-research grounding (2026-07-18, adversarially fact-checked):
     "Scaling Laws with Vocabulary" (arxiv 2407.13623) gives ~13-16k as the
     COMPUTE-OPTIMAL vocab at ~200M non-embedding params; 24-32k is defensible
     only via the paper's overtraining adjustment (our ~23B-token pass IS
     heavily over-trained, which nudges it up) — but 16k is the safe pick and
     32k is right only if the model also grows to 350-700M. Embedding cost at
     dim 1024: 16k ~= 16M params, 32k ~= 33M vs ~4.8M now. The 32k bump pairs
     naturally with a size bump; at a flat 182M prefer 16k.
   - **SuperBPE (arxiv 2503.13423) is worth a controlled A/B, not a blind
     adopt.** Superword tokens that cross whitespace directly target our 29.5%
     bare-space waste (their headline: up to 33% fewer tokens, +4% avg / +8.2%
     MMLU) — BUT every published result is at 200k vocab and 8B scale with NO
     independent sub-1B replication, and it regressed LAMBADA/HumanEval. Treat
     it as: implement leading-space merges first (the proven win), then A/B a
     SuperBPE-style pass against it on the locked eval before committing.

2. **Retokenize the corpus.** `pretokenize_data.py` over the raw sources -> new
   `tokens.bin`. New token count ~= 56.7B x (1.55/3.8) ~= **~23B tokens** (56.7B
   and 1.55 are measured; the 3.8 ratio is an estimate -- anime/fandom-heavy
   corpus may compress a bit worse). Output shrinks 227 GB -> ~90-100 GB. CPU/IO
   only, no GPU.

3. **Fresh pretrain.** New vocab = new embedding = NOT resumable; step 0. New
   lineage, new directory; the 182M lineage stays immutable (revert intact).
   - Chinchilla-optimal for 182M ~= 3.6B tokens; ~23B -> heavily over-trained
     (good for a small model). At 350-700M, optimal ~7-14B; ~23B still ample.
   - **Turn ON the levers that already exist but were frozen off for lineage
     compat** (2026-07-18 research, fact-checked): `--optimizer muon` (hybrid
     Muon-on-matrices / AdamW-on-embeddings+head — real ~1.3-1.4x speedup at
     0.1-0.5B per arxiv 2509.02046, NOT the inflated 2x; contested by
     Moonshot's ~2x claim, so measure) and `--schedule wsd` with a decay-phase
     anneal loaded with the highest-quality data (knowledge_corpus + tool
     traces + persona in the final ~10%). WSD ≈ cosine on loss ALONE — the win
     is the data curriculum the decay phase enables, so stage the data, don't
     just flip the schedule.
   - **Fold in the facts-diversity lesson at pretrain time** (Physics of LMs
     3.1/3.3, arxiv 2404.05405 — the principled version of the KNOWLEDGE_REPEAT
     workaround): any must-know fact (identity, core domain) needs 5-10
     paraphrase variants IN the pretrain corpus to be extractable, not just
     surfaced at SFT. The ~2 bits/param ceiling (~45MB of facts at 182M) means
     the model stays a REASONER+TOOLS+MEMORY router, not a knowledge store —
     spend fact-space on identity/core only.
   - **QK-norm ordering nit to fix in the v2 arch** (confirmed standard
     2026-07-18): apply RMSNorm to Q/K BEFORE RoPE, not after — Qwen3/Gemma3/
     modded-nanogpt all norm-then-RoPE. Near-equivalent math (RoPE preserves
     RMS) so it's a convention fix, cheap to do in a fresh lineage. Also drop
     RoPE theta from 500000 -> 10k-100k (500k is a long-context setting wasted
     at 1024 ctx). Optional cheap adds: gated attention (Qwen3-Next, NeurIPS
     2025 best paper — one sigmoid gate/layer, kills attention-sink spikes) and
     a deeper-thinner reshape (MobileLLM: at sub-1B more layers beat more width;
     16x1024 is on the wide side). All three UNPROVEN at sub-1B specifically —
     you'd be the replication; A/B on the locked eval.

## Cost estimate (anchored on ROADMAP's ~12 GPU-days/epoch)

- Retokenization: hours-to-a-day, no GPU.
- Pretrain 182M over ~23B tokens ~= **~5 GPU-days** (fewer tokens than the
  original 56.7B pass at the same throughput). OR spend the budget on size:
  **350-700M over ~23B tokens ~= ~12-20 GPU-days**. Wall-clock **~1-3 weeks**
  depending on the size call.
- Throughput is NOT independently measured here -- it leans on ROADMAP's own
  figure. Run a 10k-step probe first to firm it up.
- Downstream re-do is cheap: SFT/DPO re-run in minutes-hours; any encoders
  retrain against new hidden states.

## Risk / decision framing

- Low destruction risk: new lineage in a new directory; nothing destroyed.
- Real risk: weeks of GPU on a bet. Hence the eval-first ordering.
- DO this if the goal is a materially more capable companion (context, number
  recall, knowledge density).
- DON'T if you're content with the 182M's narrow envelope -- but then stop
  investing in Phase-4.5 owned-organs as if the ceiling isn't there; wire the
  borrowed organs to the avatar and ship.

## Recommended sequence

1. `EVAL_REDESIGN.md` (days, no GPU).
2. Re-measure v5/v8 on the locked set -> honest yardstick.
3. 10k-step tokenizer-probe pretrain to firm up throughput.
4. Commit to the full redo with real numbers on a trustworthy eval.
