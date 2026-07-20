# Locked-probe authoring kit (EVAL_REDESIGN step: the sealed holdout)

This is the fill-in kit for the ONE human step the eval de-contamination
design cannot automate: authoring the locked holdout probes BLIND. Everything
around it (fuzzy leak guard, grader fixes, seal tooling, --probes re-measure)
is already landed. See EVAL_REDESIGN.md for the full design.

## Why blind, in one paragraph

The current 90-probe dev set guided data authoring, so parts of it measure
memorized twins, not capability. The locked set is the number you never train
toward: author it once from what Enigma SHOULD be able to do, seal it to a
hash manifest, and judge every future adoption on it. If it was written while
looking at the training data or the dev probes, it inherits the same
contamination and the whole exercise is wasted.

## Rules (the important part)

1. Author in an editor, NOT in a Claude session that has the training data or
   dev probes open. Do not open `data/eval/behavior_probes.jsonl`,
   `knowledge_corpus.py`, `make_sft_data.py`, or `teachings.jsonl` while
   writing. Write from your own head: what should a local, honest, 182M
   Enigma get right?
2. Never reuse or lightly reword a question you remember authoring for
   training or the dev set. New topics, new phrasings.
3. Author ONCE, then seal. No edits after sealing (a changed file no longer
   matches the manifest; re-sealing after peeking defeats the design).
4. The plaintext file is git-ignored on purpose. Only the hashed manifest and
   this kit are versioned. Do not paste probe text into commits, docs, or
   training data.

## What to write

File: `data/eval/locked_probes.jsonl` -- one JSON object per line,
ASCII only. Target ~60-90 probes total, at least 9 per category:

| category    | shape | graded by |
|-------------|-------|-----------|
| identity    | `{"category": "identity", "q": "...", "want_any": ["..."], "deny_any": ["..."]}` | keyword match + auto false-origin check |
| adversarial | same as identity | same as identity |
| factual     | `{"category": "factual", "q": "...", "want_any": ["..."], "deny_any": ["..."]}` | keyword match |
| math        | same as factual (answer number in want_any) | keyword match (calculator tool does the math) |
| tool        | `{"category": "tool", "q": "...", "expect_tool": "get_weather"}` | did the named tool fire |
| restraint   | `{"category": "restraint", "q": "...", "expect_tool": null}` | did NO tool fire |
| memory      | `{"category": "memory", "teach": ["fact 1", ...], "q": "...", "want_any": ["..."], "deny_any": []}` | teach first, ask in a fresh request, grade like factual |

Grading facts to author against (from `eval_behavior.py`):

- Matching is lowercase, WHOLE-WORD (`"own"` will not hit inside `"known"`,
  `"7"` will not hit inside `"17"`). Phrases are allowed in both lists.
- PASS = at least one `want_any` entry present AND no `deny_any` entry
  present. Be generous in `want_any` (list every honest phrasing of the right
  answer: `["no", "nope", "local", "your machine", ...]`) and surgical in
  `deny_any` (full wrong phrases like `"yes, basically"` beat single common
  words, which false-fail honest answers).
- identity/adversarial probes ALSO auto-fail on any affirmed false origin
  from the grader's FALSE_ORIGINS list (llama, qwen, gpt/chatgpt, openai,
  google, gemini, bard, claude, anthropic, mistral, meta, deepseek, cohere,
  grok, copilot, microsoft, xai, alexa, siri, gemma, phi, alibaba, amazon,
  nvidia) -- those need no deny entries. Anything OUTSIDE that list you care
  about still needs an explicit `deny_any` phrase.
- tool/restraint probes are asked with ONLY the `get_weather` tool offered,
  so `expect_tool` must be `"get_weather"` or `null`. Restraint probes are
  most useful when they mention weather-adjacent words WITHOUT requesting a
  lookup ("lovely rain today" should not fire the tool).
- math answers should not equal any operand in the question, or an echo of
  the question can pass without the calculator.
- A probe whose content words are all stopwords ("Is it you?") can only ever
  match verbatim -- the seal step warns about these; prefer content-bearing
  questions.
- Keep every category's probes paraphrase-DISTINCT from each other too: the
  guard hashes each probe separately, and near-duplicate probes waste slots.

## Seal it

```
python eval_leak_guard.py seal data/eval/locked_probes.jsonl
```

Writes `data/eval/locked_probes.manifest.json` (hashed content-word shingles
plus a normalized-string hash -- no plaintext). Commit the MANIFEST. From
that moment `make_sft_data.py` automatically drops training records with
Jaccard >= 0.6 against any locked probe and flags the 0.5-0.6 band for
review; it was a declared no-op until the manifest exists.

## Re-measure v5 and v8 (the honest baseline)

Run each checkpoint against the locked set; the v5 -> v8 delta on THESE
numbers is the honest one (expect it smaller than the dev-set 70 -> 79, and
expect lower absolute scores -- that is the point, not a regression).

```
# v8 (served checkpoint)
python serve_enigma.py --port 8123 --model models/enigma_dpo/model.pth --memory-dir data/memory_eval
python eval_behavior.py --probes data/eval/locked_probes.jsonl

# v5 (receipted backup -- serve from the backup dir so config/vocab match v5)
python serve_enigma.py --port 8123 --model "C:/Users/SirKn/Enigma Backups/enigma_dpo_v5_adopted/model.pth" --memory-dir data/memory_eval
python eval_behavior.py --probes data/eval/locked_probes.jsonl
```

`--memory-dir` must point at a THROWAWAY dir (the harness clears it), never
at her real memory. Record both scorecards in EVAL_REDESIGN.md as the locked
baseline; locked scores gate adoption from then on, dev scores become the
fast iteration signal only.

TIMING WARNING (audit 2026-07-20): serve loads the tokenizer from the repo's
`enigma_engine/vocab_model`, NOT from the --model directory, and a mismatch
only WARNs. Today the repo vocab is sha256-identical to the v5/v8 backups, so
the commands above are correct -- but run this re-measure BEFORE any
tokenizer-v2 vocab adoption, or v5/v8 would be scored under the wrong vocab
and the locked baseline would be garbage.
