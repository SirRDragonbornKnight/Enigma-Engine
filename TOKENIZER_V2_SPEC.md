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
   - Consistent digit handling: per-digit tokens (predictable arithmetic) OR a
     number regex -- pick one; the current inconsistent split is the worst
     option. Recommend per-digit.
   - Vocab: 16k conservative / 32k if the corpus supports it (56.7B tokens
     does; measured tokens.json total_tokens 56,708,655,637 -- the 227 GB is
     tokens.bin at uint32, the same file older notes call "210GB" in GiB).
     Embedding cost at dim 1024: 32k vocab ~= 33M embedding params vs ~4.8M now
     (+28M net if tied) -- +15% on a 182M model, negligible on 350-700M. The
     vocab bump pairs naturally with a size bump.

2. **Retokenize the corpus.** `pretokenize_data.py` over the raw sources -> new
   `tokens.bin`. New token count ~= 56.7B x (1.55/3.8) ~= **~23B tokens** (56.7B
   and 1.55 are measured; the 3.8 ratio is an estimate -- anime/fandom-heavy
   corpus may compress a bit worse). Output shrinks 227 GB -> ~90-100 GB. CPU/IO
   only, no GPU.

3. **Fresh pretrain.** New vocab = new embedding = NOT resumable; step 0. New
   lineage, new directory; the 182M lineage stays immutable (revert intact).
   - Chinchilla-optimal for 182M ~= 3.6B tokens; ~23B -> heavily over-trained
     (good for a small model). At 350-700M, optimal ~7-14B; ~23B still ample.

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
