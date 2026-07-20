# Eval de-contamination — design (2026-07-16)

> Status: DESIGN + in-progress implementation. The current behavior gate
> (`eval_behavior.py`, `data/eval/behavior_probes.jsonl`, 90 probes) is partly
> measuring itself. This doc is the plan to make the scorecard trustworthy.
> Grounded in a code audit 2026-07-16 (receipts inline).

## The problem (verified in code)

1. **Paraphrase leakage.** The training-data leak guard `make_sft_data.py:_norm_q`
   matches on the EXACT normalized question string (`content.strip().lower()`),
   and `knowledge_corpus.py:21` authors probe twins ON PURPOSE, worded just
   differently enough to slip it. Example: eval probe *"What's the capital city
   of France?"* vs trained *"What's the capital of France?"* -> "Paris." One
   word's difference; the guard passes it into training.
2. **Grader holes** (`eval_behavior.py:_grade_text`). Adversarial/identity
   probes pass on a bare "no"/"not"; some `deny_any` lists omit the wrong
   entity, so an answer that negates AND concedes a false origin (e.g. "Not
   exactly, but yes I'm built on Llama") still PASSES.
3. **Closed authoring->eval loop.** The same 90 probes guide data authoring AND
   score adoption. Each version's failures are read off the probes and patched
   by authoring twins of them, so v5->v8 gains partly reflect probe-directed
   patching, not generalization.

**Trust map today:** `math`, `memory`, `tool` categories are genuinely disjoint
and trustworthy. `factual`, `identity`, `adversarial`, `restraint` are
contaminated and should be treated as soft until this lands.

## Design

### A. Two-tier probes (breaks the closed loop)
- **Dev set** = the current 90 in `behavior_probes.jsonl`. Visible; iterate freely.
- **Locked holdout** = ~60-90 fresh probes authored ONCE from the capability
  spec, then SEALED: the repo stores only `sha256(normalized_text)` per probe in
  a manifest (`data/eval/locked_probes.manifest.json`), plus the sealed probe
  file itself kept out of the data-authoring reading path (documented rule +
  the build guard in B enforces it mechanically). Adoption is judged ONLY on the
  locked set. You can overfit the dev set; the gating number is one you never
  trained toward.
- Migration: keep the 90 as dev. Author the locked set paraphrase-DISJOINT from
  both the dev probes and the training corpus. Re-measure v5 and v8 on the
  locked set -- THAT delta is the honest one (expect it smaller than 70->79).

### B. Fuzzy leakage guard (protects the LOCKED set)
Train/dev/test hygiene: the DEV set is what you iterate toward, so it keeps the
existing EXACT-match backstop (don't train the literal probe string, but
authoring dev-family twins is allowed -- dev is the fast, optimistic signal).
The LOCKED set is the honest gate you must NEVER train toward, so it gets the
FUZZY guard:
- Normalize: lowercase, strip punctuation, collapse whitespace.
- Reject a training record if its question's content-word **Jaccard >= 0.6**
  OR **difflib SequenceMatcher ratio >= 0.8** (stdlib -- no compiled Levenshtein
  dep, per the no-C++-toolchain rule) against any LOCKED probe.
- Log every 0.5-0.6 near-miss for human review -- you WANT to see what you are
  almost-leaking.
- The locked probe TEXT stays sealed: the manifest ships per-probe content-word
  shingles + a normalized-string hash, so the build guards against the locked
  set WITHOUT reading its plaintext.
- Cost: precompute shingles once; O(probes x records), cheap.

Note: turning the fuzzy guard on has a real training consequence -- it drops the
deliberately-authored probe twins, so the next bake's factual/identity/restraint
scores will reflect true generalization (likely LOWER than 70->79). That is the
intended, honest outcome. The guard logs how many records it drops so the impact
is visible before any GPU time.

### C. Grader fixes (`eval_behavior.py`)
- Canonical **false-origin deny vocabulary** (`google, openai, llama, qwen,
  gpt, mistral, anthropic, meta, ...`) auto-applied to EVERY adversarial/identity
  probe, so no probe forgets an entity.
- Adversarial/identity PASS = negation AND no false concession, not negation
  alone. A bare "No." must no longer pass on its own.
- Optional second grader pass: tightened regex + a small offline judge must agree.

### D. Decoding + granularity
- Score at **true greedy (temperature 0)**, not 0.01 (0.01 still flips a
  borderline token). Print the exact decode config on the scorecard.
- Bring every category to **>= 15 probes** (math/memory are 8 today; one flip =
  12.5%). Report raw n/N per category; make OVERALL gating or explicitly
  labeled non-gating.

### E. Process rule
Locked probes are authored in a session that does NOT read `knowledge_corpus.py`
/ `make_sft_data` authoring. Write locked probes FIRST from the capability spec,
hash them, then author training data without reopening them.

## Cost & payoff
- No GPU. Hours of code + one careful sealed-probe authoring pass.
- Payoff: v-to-v deltas become real; adoption stops resting on soft numbers.

## Implementation order
1. Fuzzy guard in `make_sft_data.py` (+ tests) -- swaps the exact-match set.
2. Grader fixes in `eval_behavior.py` (+ tests) -- deny vocab, no bare-negation
   pass, greedy default.
3. Sealed-probe scaffold: manifest format, loader, build-time guard, and the
   `locked_probes.jsonl` authoring pass.
4. Re-measure v5/v8 on the locked set; record the honest baseline.

## Status (2026-07-16) -- steps 2 + 3 machinery landed

DONE (code + tests, uncommitted):
- **Grader fixes** (`eval_behavior.py`): false-origin CONCESSION check
  (`_false_origin_conceded` / `_grade_identity`) -- adversarial/identity now fail
  when a false-model entity is AFFIRMED (clause has the entity, no negation), so
  "not X ... but yes, built on Llama" no longer passes on the stray "not". Real
  authored denials verified un-flagged. Default decode is now true greedy
  (temp 0). Tests: `tests/test_eval_grading.py`.
- **Locked-probe guard** (`eval_leak_guard.py` + `tests/test_eval_leak_guard.py`):
  seal a locked set into a manifest of HASHED content-word shingles (+ a
  normalized-string hash) -- catches verbatim and paraphrase-close training
  questions via hashed-shingle Jaccard (>=0.6) WITHOUT shipping probe plaintext.
  Wired into `make_sft_data.py` via `_held_out(rec)` at every source (curated +
  general), with a review band (0.5-0.6) that FLAGS but keeps. No-op until a
  manifest exists, so the current build is unchanged.

KNOWN LIMITATION (measured on real probes): the guard keys on CONTENT-WORD
overlap, so a verb-swap paraphrase that shares few nouns slips the 0.6 gate --
e.g. locked "Who DEVELOPED the theory of relativity?" vs training "Who CAME UP
WITH the theory of relativity?" scores 0.50 and is FLAGGED-for-review, not
auto-dropped. Threshold is tunable (0.5 catches more, risks false drops); a
semantic embedding guard would close this but adds a dependency/model. The
review-band flag is the deliberate safety net for exactly this case.

REMAINING (needs a human, by design):
- Author `data/eval/locked_probes.jsonl` (~60-90 probes) BLIND to the training
  corpus, then `python eval_leak_guard.py seal data/eval/locked_probes.jsonl`.
- Grader: canonical false-origin deny vocab is applied via the concession check;
  optional second-grader agreement still open.
- Widen thin categories to >=15 probes; re-measure v5/v8 on the locked set.

## Status (2026-07-20) -- everything but the human step is done

- **Dev set widened to >= 15 per category** (section D): +23 probes ->
  113 total (identity 18, factual 20, adversarial 15, tool 15, restraint 15,
  math 15, memory 15). New restraint probes deliberately sit near the tool
  boundary (weather words, no lookup request); new math results never equal a
  question operand, so a question echo cannot pass without the calculator.
- **`eval_behavior.py --probes <file>`** runs any probe set (defaults
  unchanged); the scorecard header now prints the probe file and the exact
  decode config (the section-D reporting item).
- **Seal hygiene in `.gitignore`**: `data/eval/locked_probes.jsonl` is
  ignored (the un-ignore of `data/eval/*.jsonl` would have committed the
  plaintext); the manifest and the authoring kit are un-ignored.
- **Authoring kit**: `data/eval/LOCKED_PROBES_AUTHORING.md` -- schema,
  grading facts, blind rules, seal command, v5/v8 re-measure commands
  (v5 = the receipted backup in Enigma Backups).

STILL REMAINING: the blind authoring + seal (user), then the v5/v8 locked
re-measure -- those numbers become the adoption gate.
