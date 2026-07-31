# Tokenizer v2 redo — spec + cost (2026-07-16, MEASURED 2026-07-19)

> Status: **VOCAB DONE 2026-07-20, CORPUS v2b DONE 2026-07-28** -- the
> production v2 vocab (16,366 rows) is trained, and the live corpus is
> `data/pretrain/tokens_v2b.bin` (receipts in "CORPUS v2b LANDED 2026-07-28";
> the earlier "CORPUS LANDED 2026-07-20" section describes `tokens_v2.bin`,
> which is now the receipted rollback). Only the pretrain itself remains, queued as
> `BACKLOG.md` T2/T3 inside the training block. The eval-trustworthiness
> precondition this doc named is MET: the locked gate is sealed and the
> 56/120 (v8) / 55/120 (v5) baseline is measured (see `EVAL_REDESIGN.md`).
>
> **2026-07-20: all 9 audit findings FIXED** (see the BLOCKERS section, now a
> resolution ledger). Suite 475 green at the time; the 9 new/changed
> contracts are each mutation-verified.
>
> **v1 is untouched** — verified directly: the live vocab's git blob oid is
> identical from session start to HEAD and no commit touches it. But see
> BLOCKERS below: the claim that tests "pin its sha256" was FALSE (no test
> hashes it), and the v1-immutability net is thinner than advertised.

## BLOCKERS — ALL 9 FIXED 2026-07-20 (arc-1; suite 475 green, 9/9 mutation-killed)

All nine findings below are RESOLVED. The two HIGH blockers no longer stand
between here and a v2 pretrain. Summary of the fixes (files: `pretokenize.py`,
`bpe_tokenizer.py`, `advanced_tokenizer.py`, `chat_format.py`, `tokenizer.py`,
`finetune_enigma.py`, `serve_enigma.py`; tests in `test_tokenizer_v2.py` +
`test_chat_format.py`):

1. **HIGH-1 FIXED** — one shared carve-out. `pretokenize_v2_with_specials(text,
   special_tokens)` is the single source for both classes; each passes its OWN
   `special_tokens`, and the runtime now ADOPTS the file's full special set on
   v2 load (the __init__ subset was why the runtime carved fewer tags than the
   trainer). `'a </s> b'` -> identical ids on both paths. Parity pinned across
   5 cases incl. the audit's own reproducer.
2. **HIGH-2 FIXED** — chat-token ids are DERIVED, not hardcoded.
   `attach_chat_tokens` computes base = first row past the real vocab (== 4718
   on the live v1 vocab, so v1 is byte-unchanged), refuses to alias an occupied
   row, refuses a non-contiguous vocab, and adopts fully-baked chat tokens if a
   future vocab carries them. `render_*`/`parse_assistant_ids`/serve read ids
   off the instance via `chat_token_ids`/`think_token_ids`; `reinit_chat_rows`
   derives its rows + mean slice too. Verified on a synthetic 5,996-row vocab:
   no real row overwritten, and no v1 constant leaks into a render.
3. **MED-3 FIXED** — `test_live_vocab_sha256_is_pinned` hashes the file bytes
   against `83510aef…bf03`.
4. **MED-4 FIXED** — net widened: 5 more encode pins, a full chat-render pin,
   and `test_v1_pins_detect_a_splitter_swap` (proves the digit-bearing pins
   move if v1 silently routed through the v2 splitter).
5. **MED-5 FIXED** — `get_tokenizer` cache key includes the vocab file
   fingerprint `(mtime_ns, size)`; an in-place swap now misses the cache.
6. **MED-6 FIXED (documented)** — decode of a special-token LITERAL at the
   serving default is now a PINNED property (`'a </s> b'` -> `'a  b'` with
   skip=True, exact only with skip=False), not a silent surprise.
7. **LOW-7 FIXED** — `V2_PATTERN` contraction alt is `(?i:…)`, matching cl100k.
8. **LOW-8 FIXED (documented)** — the orphaned pre-tag space is a pinned unit;
   bare-space rates must be quoted on tag-bearing text.
9. **LOW-9 FIXED** — `train_tokenizer` requires an explicit `output_path`
   (no more live-vocab default) and takes a `pretokenizer=` arg.

### Original audit text (2026-07-19) — kept for the record

The v2 machinery is sound in the small (20/20 mutations killed a test, no
vacuous tests, losslessness survived a 200k-string fuzz + codepoint sweep,
444 green). These are the real defects, two of which falsify claims made in
commit 857dc63c's own message:

1. **HIGH — v2 REPRODUCES the trainer/runtime split it claims to prevent.**
   The trainer carves out a hardcoded 4-tag tuple; the runtime carves out
   every multi-char entry of the live `special_tokens` (14, plus whatever
   `attach_chat_tokens` adds). Same text, different IDs, silently:
   `'a </s> b'` -> trainer `[111,5373,61,129,76,293]`, runtime
   `[111,46,2,293]`. Training on one and serving with the other is the exact
   silent-corruption class the commit message says is impossible.
   `test_v2_pre_tokenize_routes_through_shared_module` only checks text with
   no special token, so it cannot see this. **Fix: one shared carve-out set.**
2. **HIGH — a third coupled constant the uint16 note missed.**
   `chat_format.py` hardcodes `BASE_VOCAB = 4718` with the chat tags at
   4718..4723. Any v2 vocab bigger than that (the target is 16,384) makes
   those ids ALIAS real learned tokens, and `attach_chat_tokens` overwrites
   them with no exception (its guard checks by token NAME, which never
   collides). Verified: at vocab 5996, id 4718 was the real token `' crashes'`
   before being silently overwritten. **Fix: derive the chat-token base from
   the tokenizer's vocab_size — for v1 that is still 4718, so v1 is unchanged.**
3. **MED — the sha256 pin does not exist.** The commit message and this doc
   claimed tests pin the live vocab's hash; nothing does. Only key-presence is
   checked, so a byte edit preserving the 4 top-level keys goes undetected.
   Live hash, for the record: `83510aefe587eab78a9d653fb8c532cd3bbf239974dcbe51776e4c559838bf03`.
4. **MED — the v1-immutability net is 2 assertions wide.** Forcing v1 to use
   the v2 splitter turns only 2 of 444 tests red, both in the new file; one of
   the three pinned strings is insensitive to the swap and the round-trip
   cases do not discriminate at all. No serve/chat/inference test notices v1
   tokenization changing.
5. **MED — the tokenizer cache cannot see a vocab swap.** `_tokenizer_cache`
   is keyed on `(type, path)` with no mtime/size/version, so replacing the
   vocab in place (which the retokenize plan does) leaves warm processes
   serving the OLD instance — v1 rules on a v2 vocab, no error. Nothing calls
   `clear_tokenizer_cache()` on that path.
6. **MED — "round-trip lossless" is false at the serving default** when the
   input contains a special-token LITERAL: with `skip_special_tokens=True`,
   `'a </s> b'` decodes to `'a  b'` — text silently deleted, and a string that
   merely looks like a tag is indistinguishable from a real one.
7. **LOW — contractions are case-sensitive**, reintroducing the very
   inconsistency class v2 exists to kill: `"don't"` -> `['don', "'t"]` but
   `"DON'T"` -> `['DON', "'", 'T']`. cl100k uses `(?i:...)`; V2_PATTERN does not.
8. **LOW — the 0.0% bare-space headline excludes the common case.** The
   runtime splits on a tag BEFORE `pretokenize_v2` runs, orphaning the space
   before every special token into a bare `' '`. Enigma's serving format emits
   tags constantly; the 0.0% was measured on tag-free held-out text.
9. **LOW — `train_tokenizer()` was never plumbed for v2** (no `pretokenizer`
   arg, constructs `BPETokenizer()` bare) and its default output path is the
   LIVE vocab — so the supported entry point can only produce v1 and defaults
   to overwriting the file the whole safety story rests on.

## PRODUCTION VOCAB TRAINED 2026-07-20 (Arc 2) — supersedes the 07-19 pilot

`enigma_engine/vocab_model/bpe_vocab_v2_16k.json` (1,028,888 bytes). The live
v1 vocab is untouched (sha256 still `83510aef…bf03`).

- **16,366 real rows**, so the padded embedding is exactly **16,384** with the
  18 chat/reserved rows INSIDE the padding — the same 4718/4736 shape v1 uses,
  and still the 1024-multiple the Triton fused-CE bug class requires. (The
  earlier note said "vocab 16,384"; training to 16,384 REAL rows would have
  pushed the padded size to 16,402 and off the multiple.)
- **Reserve layout (allocated 2026-07-27, the last vocab window):** rows
  base+0..5 are the six chat tokens (`attach_chat_tokens` derives them, so
  v2 gets 16,366..16,371), rows base+6/base+7 are the **image-span
  delimiters `<|image|>`/`<|/image|>`** (`attach_image_tokens`; v2 =
  16,372/16,373, v1 layout 4724/4725), rows base+8..17 stay free. Both
  families are INSTANCE-attached constants, exactly the v1 chat-token
  pattern — the vocab FILE maps neither literal, so no corpus text can
  ever carve into a reserve row (the `<image>`-in-HTML hazard that the
  collectors sanitize for TABLE specials structurally cannot exist here),
  and this file's sha is unchanged by the allocation.
- Trained on a **266.6 MB** stratified sample (512 chunks x 512 KB, evenly
  spaced across all 88.59 GB), 706,036 unique words, 16,096 merges, **3.2 min**
  single-threaded.
- Held-out is **8.3 MB the training never saw** — a disjoint 16-chunk grid at a
  half-stride phase offset, so it cannot overlap the training grid.

| metric | v1 (live, 4718) | v2 (16,366) |
|---|---|---|
| chars/token, SAME held-out 8.3 MB | **1.6689** | **4.0206** (2.41x) |
| round-trip lossless | yes | **0/16 chunks failed** |
| bare-space rate, tag-free text | 25.5% (carried from the 07-19 pilot, NOT re-measured here) | **0.011%** |
| bare-space rate, chat-format render | — | **0.000%** (67 ids) |
| multi-digit tokens in vocab | many | **0** |
| leading-space tokens | 0% | **11,154 (68.2%)** |

Longest learned units: `' responsibilities'`, `' characteristics'`,
`' recommendations'`, `' representatives'`, `' Representatives'`.

**2.41x, not the pilot's 2.47x.** The pilot trained on a 56 MB slice and
measured on 2 MB; this is 266 MB trained / 8.3 MB held out. The bigger, more
diverse sample is the more honest number — quote **2.41x**.

**Sampling validated against a known-true number**: projecting v1's measured
1.6689 over the corpus gives 56,997,296,536 tokens against the real receipt of
56,708,655,637 — **0.51% off**. That is what makes the v2 projection a
measurement rather than a guess:

- **v2 corpus: ~23.66B tokens** (was estimated ~23B).
- **uint16 on disk: ~44.1 GiB**, down from today's 211.3 GiB — a **4.79x**
  shrink. (uint32 would be 88.1 GiB, which is why the 2-byte write path is a
  required step, not an optimisation.)
- **Context, for free**: block 1024 goes ~300 words -> ~722 words.

**HIGH-2 proved itself on a real 16k vocab**: chat tokens derive to
16366..16371 and `attach_chat_tokens` refused nothing — under the old
hardcoded `BASE_VOCAB = 4718` those six ids would have silently overwritten
real learned tokens.

## MEASURED 2026-07-19 (the 56 MB pilot — superseded above)

Method: 64 MB stratified sample of `data/pretrain/combined.txt` (256 chunks
evenly spaced across all 88.59 GB, so no single source dominates); v2 vocab
trained on 56 MB of it; both tokenizers measured on the SAME held-out 2 MB
the training never saw.

CAVEAT on the sample source (audit 2026-07-19): `combined.txt` is NOT what
`pretokenize_data.py` reads — that script walks `SOURCE_DIRS` and its own
header says it "doesn't touch combined.txt". The two are the same underlying
text: the present source dirs total **89.50 GB** against combined.txt's 88.59
GB (~1%), and projecting v1's measured ratio over it lands within 3% of the
real 56.7B-token receipt. So the chars/token RATIO measured here is valid for
the corpus. But the retokenize itself must run over `SOURCE_DIRS` through the
v1 filter+dedup path, NOT over combined.txt, or it silently changes the
corpus definition.

| metric | v1 (live, vocab 4718) | v2 (vocab 16384) |
|---|---|---|
| chars/token, SAME held-out 2 MB | **1.6618** | **4.1113** (2.47x) |
| bare-space tokens | 25.5% | **0.0%** |
| leading-space tokens | 0% | 67.9% |
| digit consistency | `1000`->`1,00,0` but `2500`->`2,5,00` | per-digit always |
| round-trip lossless | yes | yes |

(Audit correction: a first pass reported 2.52x by comparing v2 on the held-out
tail against v1 on a different 2 MB slice. Re-measured on identical text it is
**2.47x**. v1 reads 1.63-1.66 depending on slice; the held-out figure is the
one to quote. Using 1.6618 also tightens the receipt check below to 0.9%.)

Apples-to-apples at matched vocab size (v2 @ 4000 vs v1 @ 4718): **3.06 vs
1.63 chars/token**, so ~1.9x comes from the pre-tokenizer alone and the rest
from the larger vocab.

- **Sample is representative**: projecting v1's held-out 1.6618 over the full
  corpus gives 57.2B tokens vs the actual receipt of 56.7B — **within 0.9%**
  (the earlier 1.6322 slice projected 58.3B, within 3%). Either way the
  sampling method is validated against a known-true number, which is what
  makes the v2 projection trustworthy rather than a guess.
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
- **Context, for free**: block 1024 goes from ~299 words (v1) to ~739 words
  (v2) — a 2.47x effective context increase with no architecture change.
- Retokenize cost, MEASURED with the v2 tokenizer (2026-07-19). v2 encodes
  slower per MB than v1 (more merge work per unit) but produces 2.5x fewer
  tokens; single-threaded it is 4.4 MB/s = **5.7 hours** for 88.59 GB.
  `pretokenize_data.py` has NO parallelism today. It is NOT embarrassingly
  parallel (an earlier draft of this section said so and was wrong): it keeps
  a SHARED paragraph-level dedup set (`seen_hashes`, cap 50M) plus a
  `MIN_PARAGRAPH_LENGTH` filter, and that state is inherently sequential.
  The correct split is parent-does-IO/filter/dedup, workers-do-encode: the
  hashing is microseconds per paragraph against BPE encoding at ~4.4 MB/s, so
  the parent is nowhere near the bottleneck, exact v1 dedup semantics are
  preserved, and an ORDERED `imap` keeps output deterministic. Measured
  encode-side scaling on this box (16 logical cores):

  | workers | MB/s | speedup | full corpus |
  |---|---|---|---|
  | 1 | 4.42 | 1.0x | 5.7 h |
  | 2 | 8.49 | 1.9x | 3.0 h |
  | 4 | 15.69 | 3.5x | 1.6 h |
  | 8 | 28.45 | 6.4x | 0.9 h |
  | 12 | 37.34 | 8.4x | **0.7 h** |

  One-time pool init is ~3 s (negligible at corpus scale). ~70% parallel
  efficiency at 12 workers. Note these were taken while the machine was in
  use, so treat them as a floor. They also measured **CPU scaling only** —
  the benchmark read a 64 MB OS-cached file, not 89.5 GB off disk. That is
  fine here: 12 workers consume ~37 MB/s of text and write ~17 MB/s, both
  trivial for NVMe, so disk should not become the bottleneck — but the first
  real run is the only proof.

  **Remote-access budget (checked 2026-07-19).** This box is driven over
  **Chrome Remote Desktop** (`remoting_host.exe`), which encodes the video
  stream **in software on the CPU** — unlike Parsec/Sunshine, which use NVENC
  on the GPU. So CPU saturation degrades the operator's session directly
  (frame drops, input lag), and "the machine is idle" is never true while
  someone is connected. Compounding it: `Win32_Processor` reports **16 cores /
  16 logical — SMT is OFF**, so each worker eats a whole physical core with no
  hyperthread slack.
  Policy for any long CPU job here:
  - default `--workers 10`, leaving ~6 cores for the CRD encoder, the OS, and
    whatever the operator is doing;
  - run workers at **BelowNormal priority** so the scheduler always preempts
    them in favour of the interactive session — this matters MORE than the
    core count, because it keeps the session smooth even if the estimate is
    off;
  - 12+ workers only when nobody is connected.
  At 10 workers this lands around 1 hour for the full corpus instead of ~42
  minutes at 12 — a cheap trade for a usable desktop. **4 workers is the "run it while gaming"
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
     32k is right only if the model also grows to 350-700M (Gate B has since
     ruled 238m, so 16k stands). Embedding cost at
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
     (Gate B ruled 238m: optimal ~4.8B, the 28.26B corpus is ~5.9x it.)
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
     RMS) so it's a convention fix, cheap to do in a fresh lineage. ~~Also drop
     RoPE theta from 500000 -> 10k-100k (500k is a long-context setting wasted
     at 1024 ctx).~~ SUPERSEDED 2026-07-23: this line was written when the
     target was a 1024-context lineage. The `v2_deep_*` presets are native
     8192 and train at block 2048, where 500000 is the defensible setting —
     all three presets keep it. Optional cheap adds: gated attention (Qwen3-Next, NeurIPS
     2025 best paper — one sigmoid gate/layer, kills attention-sink spikes) and
     a deeper-thinner reshape (MobileLLM: at sub-1B more layers beat more width;
     16x1024 is on the wide side). All three UNPROVEN at sub-1B specifically —
     you'd be the replication; A/B on the locked eval.

## Cost estimate (SUPERSEDED -- kept as the record of the pre-measurement
## projection; the measured grid is 186m 4.1 / 238m 4.9 / 542m 20.8 d/epoch
## on the 28.26B v2b corpus, and Gate B ruled 238m on 2026-07-30)

- Retokenization: hours-to-a-day, no GPU. (Measured since: 42-66 min.)
- ~~Pretrain 182M over ~23B tokens ~= **~5 GPU-days** ... OR spend the budget
  on size: **350-700M over ~23B tokens ~= ~12-20 GPU-days**. Wall-clock
  **~1-3 weeks** depending on the size call.~~
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

**SUPERSEDED -- everything below landed 2026-07-20; kept for the requirement
list; receipts in "CORPUS LANDED" just after this section.**

NEXT, in order:
1. Train the PRODUCTION v2 vocab on a larger slice (the measured vocab used
   56 MB; training is cheap — 0.9 min — so a multi-GB slice is affordable and
   will sharpen the merge statistics).
2. Full corpus retokenize -> new `tokens.bin` (~43 GB as uint16; 1.4 TB free
   so headroom is fine). Needs a PARALLEL v2 write path, which does not exist
   yet — `pretokenize_data.py` is single-threaded and hardcodes uint32, and it
   is the script that produced the immutable v1 corpus, so v2 should get its
   own script rather than growing a mode flag. Requirements for it:
   - Input is `SOURCE_DIRS`, NOT combined.txt, through the same
     `MIN_PARAGRAPH_LENGTH` filter and paragraph dedup as v1 — otherwise the
     corpus definition silently changes.
   - **Missing source dirs are SILENTLY SKIPPED today** (`if not
     source_dir.exists(): continue`, no log). Three are already absent
     (dclm / finemath / the_stack — they contributed nothing, since the
     present 89.50 GB already accounts for the 56.7B-token receipt). The v2
     script must LOG or REFUSE on an absent source instead: a silently
     smaller corpus is a quality regression nothing would catch, and it
     breaks the repo's fail-honestly rule.
   - `--workers` (DEFAULT 10, not cpu_count -- see the remote-access note
     below); parent does read/filter/dedup, workers do encode only (see the
     scaling note above -- the dedup set is shared state and cannot be
     sharded); ORDERED `imap` so output is deterministic; one tokenizer per
     worker via a Pool initializer (Windows spawn-safe).
   - **Run workers at BelowNormal priority** (`psutil.BELOW_NORMAL_PRIORITY_CLASS`,
     or `os.nice` equivalent, set in the Pool initializer).
   - uint16 output AND a matching dtype in pretrain's memmap reader — change
     them together or the corpus reads as garbage.
   - New lineage, NEW directory: never overwrite the v1 `tokens.bin`.
   - Keep v1's guards: eos-id bounds check, refuse-on-vocab-mismatch, and the
     paragraph dedup.
   Wall clock once it exists: ~1.6 h at 4 workers (safe while gaming) or
   ~0.7 h at 12 (idle box).
3. Then, and only then, the GPU decision per the framing above.

### CORPUS v2b LANDED 2026-07-28: tokens_v2b.bin — THE LIVE TRAINING CORPUS

The T1-ruled diet, tokenized in 65.8 min (10 workers, python path):
**28,261,718,460 tokens**, 5,689,882 docs, 1,436,089 dupe paragraphs
skipped, 52.64 GiB (56.52 GB decimal) uint16. Sidecar-verified: **curated shard walks FIRST at
x5** (extent [0, 8,483,455], `repeated_sources {'Curated': 5}`), zero
absent sources, ETOK header == sidecar == file-size arithmetic. New
sources: DCLM 4.094B tokens / FineMath 3.483B / The Stack 4.875B = 12.45B
(44%) of the corpus is the new model-filtered web + math + code diet;
C4 and OpenWebText are out by ruling (their dirs ruled dead 2026-07-29 --
BACKLOG section 9).
Collector-side screening receipts (2026-07-27 run): 52,618 records dropped
against the sealed probes across the three fetches, 11,463 special-token
literals space-broken at fetch + **660 more at the pretokenize choke point**
(the pre-scrubber sources' residue -- recorded in the sidecar as
`special_literals_sanitized`). Bin + sidecar attrib +R; `tokens_v2.bin`
below remains on disk as the receipted rollback. Grid at the measured
rates: 186m 4.1 / 238m 4.9 / 542m 20.8 days/epoch (re-measured 2026-07-28 at
the real launch shapes over 150 steps, compile ON where it works; the earlier
10.0/10.5/22.9 came from ~40-step eager probes -- BACKLOG item 7 owns the
receipts).

### (superseded as the training corpus) CORPUS LANDED 2026-07-20: tokens_v2.bin

23,694,200,666 tokens (projection 23.66B -> 0.14% off), 5,688,823 docs,
1,382,216 dupe paragraphs skipped, 44.13 GiB (47.39 GB decimal) uint16,
41.9 min wall.
Validated: ETOK header == sidecar == file-size arithmetic; ids bounded
by vocab 16,366; random windows at start/middle/end decode clean.
Run history is a lesson in where walls really are: run 1 (pure python,
cold walk) projected 25 h -- NOT from encode speed but from ~4 ms
per-file open latency serializing the parent (workers at 2% CPU); run 2
(rust v2 encode + 16-thread read prefetch) did the whole corpus in 41.9
min. Dedup table hit its 50M cap mid-run -- v1's exact semantics, cap
point deterministic in walk order, so the corpus definition is unchanged.

### Item 2 DELIVERED 2026-07-20 (Arc 3) — with two spec deviations, on purpose

`pretokenize_data.py` grew the parallel path instead of a separate script
(deviation 1): the walk/dedup/filter code IS the corpus definition, and two
copies of it would drift. The v1 lineage is protected by a hard guard instead
of file separation — aiming a custom `--vocab`/`--dtype` at the default
`tokens.bin` path is a refusal, mutation-verified. Output goes to
`data/pretrain/tokens_v2.bin` beside v1 (deviation 2: same dir, new name —
the reader takes an explicit path, and the guard, not geography, is what
protects v1).

Delivered against the requirement list: parent-side walk + shared dedup,
ordered `imap` (byte-determinism vs sequential is THE pinned contract),
one tokenizer per spawn worker, `--workers` default 1 / 10 for the real run,
workers self-set BelowNormal via the Pool initializer (ctypes, no psutil
dep), absent sources are LOGGED loudly + recorded in the sidecar as
`sources_absent` (LOG chosen over REFUSE: dclm/finemath/the_stack are absent
by design), uint16 + dtype-aware memmap reader landed together, eos/vocab
bounds guards kept and extended (worker-side id bounds, vocab<=65536 for
uint16, header bpt derived from the array itemsize).

Verification: 8 tests in `tests/test_pretokenize_data.py`, 8/8 mutations
killed (incl. `imap`->`imap_unordered`, which required an asymmetric
heavy/light fixture to fail deterministically), suite 483. Adversarial audit
2026-07-20: no HIGH; its MED — the `--val-general-end` v1 default (56.6B)
silently clamping into the ~23.7B v2 corpus and evaluating train data as
"val-gen" — is fixed in `pretrain_enigma.py`: an offset beyond `train_end`
now DISABLES the window with a loud note instead of clamping. The v2
pretrain run needs no extra flag for this.

NEAR-MISS, recorded for the discipline ledger: the first version of the
lineage-guard test aimed `pd.main()` at the REAL `tokens.bin` — safe only
while the guard it tests exists. Mutation testing removes exactly that
guard; the mutated test run started a real retokenize toward the production
corpus and was stopped by the driver's 900s subprocess timeout (run 1) and
a manual kill (run 2). v1 verified untouched (226,834,622,804 bytes, mtime
2026-06-07). Rule: a test must never point a live write path at production
data it expects a guard to protect — mutation testing WILL remove the guard.

Open A/B questions unchanged: SuperBPE-style superword pass, 16k vs 32k
(16k stands at a flat 182M), and the v2-arch nits (QK-norm before RoPE, RoPE
theta 500k -> 10-100k).
