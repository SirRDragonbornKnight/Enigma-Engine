# Tokenizer v2 redo — spec + cost (2026-07-16, MEASURED 2026-07-19)

> Status: **CPU PREFIX BUILT AND MEASURED 2026-07-19** — the pre-tokenizer,
> vocab versioning, and both encode paths are implemented and test-pinned
> (`enigma_engine/core/pretokenize.py`, `tests/test_tokenizer_v2.py`), and a
> real 16k vocab was trained on a stratified corpus sample to replace this
> doc's estimates with measurements (below). NOT yet done: the production
> vocab train on a larger slice, the full corpus retokenize (~5 CPU-hours),
> and the pretrain itself. Do NOT start GPU work until the eval is
> trustworthy (see `EVAL_REDESIGN.md`) -- without it you can't tell a better
> base from a differently-overfit one.
>
> **v1 is untouched and provably so**: the vocab file carries no version key,
> absence means v1, and `tests/test_tokenizer_v2.py` pins the live vocab's
> sha256 + exact encode IDs. The served v8 model is unaffected.

## MEASURED 2026-07-19 (supersedes the estimates below)

Method: 64 MB stratified sample of `data/pretrain/combined.txt` (256 chunks
evenly spaced across all 88.59 GB, so no single source dominates); v2 vocab
trained on 56 MB of it; both tokenizers measured on the SAME held-out 2 MB
the training never saw.

| metric | v1 (live, vocab 4718) | v2 (vocab 16384) |
|---|---|---|
| chars/token (held-out) | **1.6322** | **4.1113** (2.52x) |
| bare-space tokens | 25.5% | **0.0%** |
| leading-space tokens | 0% | 67.9% |
| digit consistency | `1000`->`1,00,0` but `2500`->`2,5,00` | per-digit always |
| round-trip lossless | yes | yes |

Apples-to-apples at matched vocab size (v2 @ 4000 vs v1 @ 4718): **3.06 vs
1.63 chars/token**, so ~1.9x comes from the pre-tokenizer alone and the rest
from the larger vocab.

- **Sample is representative**: projecting v1's measured 1.6322 over the full
  corpus gives 58.3B tokens / 217 GB vs the actual receipt of 56.7B / 211 GB
  — within 3%.
- **Corpus projection at v2's measured 4.1113**: **~23.1B tokens** — which
  CONFIRMS this doc's original estimate of ~23B (it guessed 3.8 chars/token;
  the real figure is slightly better). On disk that is ~86 GB as uint32 —
  but a 16,384 vocab fits **uint16**, so the v2 corpus should be written
  2-byte: **~43 GB, down from today's 211 GB (4.9x smaller)**. The current
  `pretokenize_data.py` hardcodes uint32; switching the v2 write path to
  uint16 is a required step of the retokenize (and pretrain's memmap reader
  must match). Vocab must stay under 65,536 to keep this.
- 16,384 is deliberately a multiple of 1024 (a Triton fused-CE bug class
  produces silently wrong results otherwise).
- **Context, for free**: block 1024 goes from ~293 words (v1) to ~739 words
  (v2) — a 2.5x effective context increase with no architecture change.
- Retokenize cost, MEASURED with the v2 tokenizer (2026-07-19). v2 encodes
  slower per MB than v1 (more merge work per unit) but produces 2.5x fewer
  tokens; single-threaded it is 4.4 MB/s = **5.7 hours** for 88.59 GB.
  `pretokenize_data.py` has NO parallelism today, and the work is embarrassingly
  parallel (chunk on line boundaries, one tokenizer per worker). Measured
  multiprocessing scaling on this box (16 logical cores):

  | workers | MB/s | speedup | full corpus |
  |---|---|---|---|
  | 1 | 4.42 | 1.0x | 5.7 h |
  | 2 | 8.49 | 1.9x | 3.0 h |
  | 4 | 15.69 | 3.5x | 1.6 h |
  | 8 | 28.45 | 6.4x | 0.9 h |
  | 12 | 37.34 | 8.4x | **0.7 h** |

  One-time pool init is ~3 s (negligible at corpus scale). ~70% parallel
  efficiency at 12 workers. Note these were taken while the machine was in
  use, so treat them as a floor. **4 workers is the "run it while gaming"
  setting** (1.6 h, 12 cores left free); 12 workers only when the box is idle.
  A first benchmark attempt at 4 MB reported a SLOWDOWN — the workload was
  smaller than Windows spawn + per-worker init, measuring startup rather than
  throughput. Size any re-benchmark so per-worker work >> init.
- Vocab quality spot-check: longest learned units are ` responsibilities`,
  ` infrastructure`, ` implementation`; 10,603 of 16,384 entries are
  leading-space word tokens; 28 tokens contain a digit and NONE contain two.

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

## Where the CPU prefix stands (2026-07-19)

DONE (committed, test-pinned, v1 provably untouched):
- `enigma_engine/core/pretokenize.py` — the v2 pattern: leading-space
  attachment, ONE digit per token, punctuation/underscore safe, lossless
  concat. Single source of truth used by BOTH the trainer and the runtime
  encoder (v1's trainer/runtime pre-tokenizers silently disagree — that wart
  is left alone in v1 but deliberately not reproduced in v2).
- Vocab files now declare `"pretokenizer"`; ABSENCE MEANS V1, so the live
  vocab keeps v1 semantics with zero file changes and no way to apply v2
  rules to it by accident.
- `</w>` is not emitted under v2 (the leading space already carries the word
  boundary, and v1's `</w>`->space decode would re-insert whitespace v2 must
  not add). Rust backend is bypassed under v2 (it hardcodes v1 splitting).
- 53 tests in `tests/test_tokenizer_v2.py`, including a ~13k-codepoint
  losslessness sweep that caught a real hole in the first pattern draft
  (underscores were being silently dropped by `re.findall`).

NEXT, in order:
1. Train the PRODUCTION v2 vocab on a larger slice (the measured vocab used
   56 MB; training is cheap — 0.9 min — so a multi-GB slice is affordable and
   will sharpen the merge statistics).
2. Full corpus retokenize -> new `tokens.bin` (~43 GB as uint16; 1.4 TB free
   so headroom is fine). Needs a PARALLEL v2 write path, which does not exist
   yet — `pretokenize_data.py` is single-threaded and hardcodes uint32, and it
   is the script that produced the immutable v1 corpus, so v2 should get its
   own script rather than growing a mode flag. Requirements for it:
   - `--workers` (default: leave headroom, e.g. cpu_count//2); chunk on line
     boundaries so no word is torn; one tokenizer per worker via a Pool
     initializer (Windows spawn-safe).
   - uint16 output AND a matching dtype in pretrain's memmap reader — change
     them together or the corpus reads as garbage.
   - New lineage, NEW directory: never overwrite the v1 `tokens.bin`.
   - Keep v1's guards: eos-id bounds check, refuse-on-vocab-mismatch, and the
     paragraph dedup.
   Wall clock once it exists: ~1.6 h at 4 workers (safe while gaming) or
   ~0.7 h at 12 (idle box).
3. Then, and only then, the GPU decision per the framing above.

Open A/B questions unchanged: SuperBPE-style superword pass, 16k vs 32k
(16k stands at a flat 182M), and the v2-arch nits (QK-norm before RoPE, RoPE
theta 500k -> 10-100k).
