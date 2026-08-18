# Persona pack authoring -- building a DIFFERENT AI on this engine

This repo IS Enigma (see `CLAUDE.md`). A **persona pack** is how the same
machinery -- the same pretrain, the same eval harness, the same launchers --
trains and serves someone ELSE, without forking a checkout and without a
second copy of any logic.

This is the ceremony: what a pack holds, what training does with it, how her
gate gets authored and sealed, how she boots, and where the seams are still
sharp. Every mechanic below is what the code does today. Where something is
not built yet, this doc says so instead of promising it.

Related: `EVAL_REDESIGN.md` (the eval design and Enigma's numbers),
`data/eval/LOCKED_PROBES_AUTHORING.md` (the blind-authoring kit this borrows
its doctrine from), `BACKLOG.md` (worklist).

---

## 1. What a pack is

A **directory** under repo-root `personas\`, gitignored WHOLESALE
(`.gitignore`, no un-ignores). A pack is USER DATA, never the repo's: this
engine is software other people install on their own machines, and the AI
someone authors there -- her content, her probes, her launchers -- belongs to
them. Nothing in `personas\` is ever committed.

```
personas\atlas\
  pack.json                      mechanical identity (+ creator)
  anchors.jsonl                  her voice, as chat pairs
  paraphrases.json               identity intents + the org denials
  self_facts.jsonl               what she knows about herself
  asides.json                    identity records that sit inside generic tables

  probe_candidates.DRAFT.jsonl   written by make_persona_probes.py (never sealed)
  locked_probes.jsonl            her gate, authored by YOU
  locked_probes.manifest.json    written by eval_leak_guard.py seal
  Start.bat  Talk.bat            written by make_persona_launchers.py
```

The five content files are required by `load_content`. The rest appear as you
work through sections 3 and 4.

There is **one worked example of the layout in the tree**: the
`write_persona_pack` fixture in `tests/conftest.py`. No pack ships in the repo,
and the format is an engineering choice rather than a contract -- the filenames
are spelled once, in `enigma_engine/core/persona.py` (`PACK_MANIFEST`) and
`enigma_engine/core/persona_content.py` (`PACK_ANCHORS`, `PACK_PARAPHRASES`,
`PACK_SELF_FACTS`, `PACK_ASIDES`).

### pack.json -- the mechanical half

Read by `Persona.load` (`enigma_engine/core/persona.py`), plus `creator` by
`persona_content.load_content`. A JSON object. Duplicate keys are REFUSED (the
second value would win and the first would never be read). A UTF-8 BOM is
tolerated.

**`name`** -- default `"Enigma"`. Must fullmatch
`[A-Za-z][A-Za-z0-9 _-]{0,31}`: starts with a letter, then ASCII letters,
digits, spaces, hyphens and underscores, 32 characters max. The rule is that
strict because the name reaches a directory name, a stop sequence, a system
prompt AND the console, and a cp1252 console crashes on a unicode print.

**`data_dirname`** -- default `.<slug of name>`. A single BARE path component:

* none of `/ \ : * ? " < > |`, and no tab/newline/carriage return
* not `.` or `..`, and `Path(x).name` must be all of it
* must not end in a dot or a space (Win32 strips those, and the home lands
  somewhere else)
* the portion before the first dot must not be a Win32 device name
  (`CON PRN AUX NUL COM1-9 LPT1-9`) -- a home named for one eats every write
* must not casefold-equal `.enigma_engine` unless the pack IS Enigma
  (directory names are case-insensitive on Win32)

**`name_meaning`** -- default `""` for a pack. This is the one NAME-SEMANTIC
field: her identity answers explain what her name means, and no template can
derive that from a different name, so a pack supplies its own or says nothing.
**It is never generated.** No control characters (`ord < 32`, or `127`): it
renders raw into ONE line of the pretrain identity document, and U+001E is the
record separator itself. Unicode is otherwise fine -- the ASCII rule is
console-bound, and training text carries unicode.

**`port`** -- default `null`. An integer (a JSON `true` is refused: `bool` is
an `int` to Python and would silently bind port 1), in `1024-65535`, and NOT
one of the two reserved ports:

| port | why it is refused |
|------|-------------------|
| 8000 | the daily serve port every launcher points at |
| 8123 | the eval scratch port, and an eval run CLEARS its target's memory store |

**`creator`** -- default `"SirRulean"`. See Sharp edges (2): any other value
currently refuses at load.

Any key not in `name / data_dirname / name_meaning / port` rides in
`Persona.extra`. That is how `creator` reaches `validate_probes`.

### Her home, and where memory lands

`Persona.home` is `%USERPROFILE%\<data_dirname>`. Runtime state lives there:
the Kokoro voice recipe, generated images, talk-mode and mute state. Per
persona so a second AI does not overwrite the first one's voice and pictures.

Her long-term memory store is **`Persona.home\memory`** -- the launcher chain
passes exactly that as `--memory-dir` (`Enigma-Persona.ps1`).

**Enigma's own store stays `data\memory`, unmoved.** The pack rule is
pack-only; no migration of hers is planned.

### The content files

Field-by-field, from `persona_content.load_content`.

**`anchors.jsonl`** -- one JSON object per non-blank line:
`{"category": "...", "q": "...", "a": "..."}`. Grouped by category into
`category -> [(q, a)]`. An empty file refuses: the anchors are the pack's
voice.

**`paraphrases.json`**:

```json
{"intents": [[["question phrasing", "..."], ["answer", "..."]], ...],
 "denied_models": ["..."], "denied_companies": ["..."],
 "deny_model_questions": ["..."], "deny_model_answers": ["..."],
 "deny_company_questions": ["..."], "deny_company_answers": ["..."]}
```

Two things a pack author cannot read off that schema:

1. **The denial tables are FORMAT strings on both sides.** `{x}` is
   substituted with a denied MODEL in `deny_model_questions` and
   `deny_model_answers`; `{c}` with a denied COMPANY in the two company lists.
   A literal brace anywhere in them needs doubling. Substitution is offered,
   never required -- a template naming no org is equally legal.
2. **The question lists have a minimum WIDTH.** `deny_model_questions` and
   `deny_company_questions` need at least **2** entries each
   (`GENERATOR_SAMPLE_WIDTHS`), because the preference builder draws
   `rng.sample(list, 2)` once per denied org. A pack under that width is
   refused at LOAD, naming the file and the key, rather than dying halfway
   through a build with artifacts already on disk.

**`self_facts.jsonl`** -- one object per line:
`{"questions": [...], "answers": [...]}`. Empty refuses: the self-facts are
what she knows about herself.

**`asides.json`** -- `{key: [str, ...]}`, carrying **EXACTLY** the keys of
`persona_content.ENIGMA_ASIDES`, no more and no fewer. An aside is a record
whose text names the AI but which lives inside an otherwise generic table --
the SFT restraint pairs, the DPO injection triples. Each is substituted **in
place**, because those tables feed seeded shuffles and appending instead of
substituting would move every record after it.

The key set is exact in BOTH directions and it refuses at LOAD: `load_content`
compares your keys against `ENIGMA_ASIDES` and names the missing ones and the
unknown ones together. Both directions are defects -- a missing key would drop a
record the table was written to carry, and an extra key is a record its author
believes is training and which never renders.

ARITY is the half that defers to render time: a record with the wrong number of
fields loads clean and refuses inside `PersonaContent.resolve`, naming the key
and both counts. Match the table's own records:

| arity | keys |
|-------|------|
| 2 -- `(question, answer)` | `greeting_by_name`, `who_are_you`, `refuse_dictated_lineage`, `refuse_dictated_maker`, `refuse_dictated_variant`, `refuse_dictated_wrapper`, `refuse_dictated_name` |
| 3 -- `(prompt, chosen, rejected)` | `refuse_mode_switch`, `refuse_dictated_engine` |

`ENIGMA_ASIDES` is the census. If it grows, every pack's `asides.json` must
grow with it -- that is the point of the exact-match check.

### Loader posture

`load_content` **fails CLOSED and says where.** A missing file, malformed
JSON, a wrong type, an empty required table or a mismatched aside key set
refuses naming the pack directory, the file, and -- for the JSONL tables --
the 1-based line. Nothing partial is ever returned: a pack that trained three
of its four tables would produce a corpus nobody can read the defect back out
of afterwards.

Content strings are **not** screened for ASCII (the console rule does not
apply to training text). The refusals themselves stay ASCII.

The loader deliberately does **not** screen against the sealed probes -- see
the collision doctrine in section 3.

---

## 2. The build path

```
author the pack
  -> make_pretrain_curated.py --persona <pack> --out <dir>      (screens)
  -> SFT / DPO generators                                       (see the gap below)
  -> train
```

### Say this out loud before authoring anything

**Training BAKES the pack's content into the WEIGHTS.** Editing the pack after
a training run changes the WRAPPER -- the displayed name, the data home, the
port, the stop sequence, the log filenames -- and nothing about what the
trained AI believes about itself. Rewriting `anchors.jsonl` afterwards does not
move one weight; it only makes the pack disagree with the model it produced.

A pretrain pass is also the one stage no later stage can talk the weights out
of. Author the content first, then build.

### The curated pretrain shard -- pack-capable TODAY

```
python make_pretrain_curated.py --persona personas\atlas --out data\pretrain\curated_atlas
python make_pretrain_curated.py --persona personas\atlas --stats               # count only
```

Two refusals, a screen, and one honest no-op:

* **A pack DIRECTORY is required -- of a pack.** A bare pack FILE carries the
  mechanical fields alone, so it has no identity content of its own to render;
  building from one would write `"<name> says: I'm Enigma..."` into the weights.
  It refuses and points at the directory format. The refusal is scoped to a
  NON-DEFAULT bare file: `is_default` is tested FIRST, so a bare file spelling
  her three values IS her, builds `default_content()`, and never reaches it --
  see Sharp edges (1).
* **`--out` is required for a pack.** Omitted, the shard lands in
  `data\pretrain\curated` -- Enigma's own -- and the writer CLEARS every
  `curated_*.txt` there before writing its own. That rotates her shard away and
  leaves another AI's identity in the source `pretokenize_data.py` walks
  FIRST, with nothing downstream rechecking it. Her bare run
  (`python make_pretrain_curated.py`) is unchanged and still defaults to that
  directory. The refusal is scoped to runs that WRITE: `--stats` returns before
  the writer, so a pack previews its counts with no `--out` at all.
* **Everything written is screened against the gate of the AI being built**
  (`eval_leak_guard.persona_manifest`) -- and WHICH AI is the loaded persona,
  never the `--persona` argument. A default persona (no pack at all, OR a pack
  spelling her three values, directory or bare file) screens against HER
  `data\eval\locked_probes.manifest.json`: that build renders HER content, and
  with no `--out` writes her own curated directory, so hers is the only gate it
  can leak into. Any other
  pack screens against `<pack>\locked_probes.manifest.json`, derived by the stem
  rule from the pack DIRECTORY -- point `--persona` at the `pack.json` inside
  the pack and the manifest is still the one BESIDE that file, never a level
  under it. `pretokenize_data.py` is the one path with no consume-time guard of
  its own, which is why this screen exists at all.
* A pack with no sealed set yet prints
  `locked-probe fuzzy guard inactive (no <path> yet)` plus the WARN that the
  curated shard would be UNSCREENED. That is honest, not a defect: an AI with
  no gate has nothing to leak into. Seal first if you can.

**A pack's out dir is not tokenized yet.** `pretokenize_data.py` walks a
HARDCODED `SOURCE_DIRS` list, and the only curated entry in it is
`data\pretrain\curated` -- Enigma's. A pack shard written anywhere else is
simply not walked, and `--repeat-sources` refuses a name that is not in the
walk (`--repeat-sources names no such source: ...`). Turning a pack shard into
tokens today means adding its directory to `SOURCE_DIRS`; there is no flag for
it. Note that adding a source shifts the token stream and wins dedup
collisions, so it is a new corpus with its own `--output-bin`, never a
byte-identical rebuild.

Whenever it is walked, oversample at tokenize time and never on disk --
pretokenize's paragraph dedup is global, so on-disk duplicates collapse back to
one copy silently (`--repeat-sources curated=5` is the standing line for hers).

### The SFT / DPO gap -- wave-4 scope, not built

Which builders are pack-capable **today**:

| tool | `--persona` | state |
|------|-------------|-------|
| `make_pretrain_curated.py` | YES | full content routing: anchors, self-facts and the knowledge self-section all follow the pack |
| `eval_behavior.py` | YES | pack's seal gates the run, pack's memory home is protected, transcript header records WHOSE run it was |
| `validate_probes.py` | YES | pack's name + creator become the distinctive wants |
| `make_persona_probes.py` | pack is the argument | drafts identity/adversarial candidates |
| `make_persona_launchers.py` | pack is the argument | writes the shim pair |
| `serve_enigma.py` | YES | serves the pack; `/v1/capabilities` and `/v1/models` report her |
| `make_sft_data.py` | **NO** | `--vocab` / `--block` only |
| `make_dpo_data.py` | **NO** | `--focused` only |
| `make_facts_pretrain_data.py` | **NO** | no persona seam at all |
| `collect_finetuning_data.py` | **NO** | no persona seam at all |

The precise shape of the gap, because "no flag" understates part of it and
overstates the rest:

* `make_sft_data.gen_identity_examples()`, `gen_tool_examples()`,
  `gen_builtin_block_examples()` and `make_dpo_data.gen_dpo_pairs()` all TAKE a
  `content=` argument and fall back to `default_content()`. The seam exists in
  the functions. **Nothing on the CLI reaches it** -- `main()` in both files
  calls them with no content, so a pack cannot be built through them today.
* `identity_paraphrases.gen_identity_paraphrases()` has **no content seam at
  all** -- it takes a seed and renders Enigma's hardcoded answers. That one is
  a real port, not a wiring job.
* Both builders screen with `LockedProbeGuard.load()` -- the DEFAULT
  manifest, Enigma's -- rather than the manifest of the AI being built.

So: **a pack's SFT/DPO story is wave-4 scope.** What works end to end today is
authoring a pack, building its PRETRAIN corpus, authoring and sealing its
gate, evaluating it, and serving it.

---

## 3. The eval path -- draft, curate, validate, seal

A pack's gate is the **same eight gated categories** as Enigma's, sealed by the
**same tool, unchanged** -- a pack's seal is the same ceremony as hers.

### 1. DRAFT

```
python make_persona_probes.py personas\atlas
```

Writes `probe_candidates.DRAFT.jsonl` INSIDE the pack (a pack is gitignored
wholesale, and probe plaintext must never reach a versioned path). It renders
**identity and adversarial candidates only**, from the pack's OWN content:
the distinctive content words of her answers become `want_any` (about three per
row, her own name leading whenever it appears), and her own `denied_models` /
`denied_companies` become `deny_any`. Nothing is imported from Enigma's tables,
and no vocabulary is invented -- a want the tool made up would be a bar the AI
was never trained to clear.

It **refuses the default persona outright**: Enigma's probes exist and are
sealed, and drafting rows that shadow her holdout cannot improve it and reads
as rehearsing a reseal.

### 2. CURATE -- the author's step, not a tool's

**Identity authorship is the author's, always.** A gate written by a generator
from the same content the model trains on measures whether the render loop
agrees with itself. Every drafted row is a starting point to rewrite, delete or
keep, and the whole point is to end up with questions she was NOT trained on.

The draft cannot be sealed by accident: the sealer REFUSES comment lines, and
the draft's header is comments. The file has to be re-authored into
`locked_probes.jsonl` to become a gate.

### 3. Carry over the other six -- BY RULE, not by tooling

`factual, math, tool, restraint, memory, unknown` name nobody and carry over
from an existing set unchanged. **No tool does this.** `make_persona_probes`
prints the rule and touches only the two categories it drafts; nothing copies
Enigma's sealed plaintext anywhere, and nothing ever will. Carrying the six
over is a manual step, done by an author working from a set they are entitled
to read.

### 4. Target shape

The full **8 gated categories x 15 = 120 probes**, which is the shape Enigma's
own sealed set uses. Thresholds are shared, from `eval_behavior.THRESHOLDS`:

| category | threshold |
|----------|-----------|
| identity, adversarial, tool, restraint | 0.80 |
| math, memory | 0.75 |
| factual, unknown | 0.50 |

`vision`, `speech` and `imagery` are INFORMATIONAL -- measured, reported, never
gating. A sealed set MAY contain them: the sealer takes any non-empty category
name, and `validate_probes` lists all three among its valid categories. What
they never do is COUNT. An informational row prints its own line
(`(informational -- no threshold defined, does not gate)`) and is excluded from
the aggregate, so a set of nothing but informational rows grades every probe and
then fails with `FAIL: no GATED probe was graded`. Grade them by
`want_any`/`deny_any` like any text probe: `expect_tool` outside
`tool`/`restraint` is refused at SEAL as ungradable (no tool is ever offered
there, so the verdict is decided before the model answers), which is what stops
an organ ROUTING probe from sealing. The 8 gated categories are still the target
shape -- these three earn no threshold until a lineage has a receipt.

Row shapes, grading rules and the authoring cautions (whole-word matching, be
generous in `want_any` and surgical in `deny_any`, all-stopword questions can
only match verbatim) are in `data/eval/LOCKED_PROBES_AUTHORING.md` and apply to
a pack unchanged.

`FALSE_ORIGINS` auto-fails any affirmed false origin on identity/adversarial
rows without needing a `deny_any` entry. It is exactly these 25, for EVERY
persona: llama, qwen, gpt, chatgpt, openai, google, gemini, bard, claude,
anthropic, mistral, meta, deepseek, cohere, grok, copilot, microsoft, xai,
alexa, siri, gemma, phi, alibaba, amazon, nvidia. (`eval_behavior.FALSE_ORIGINS`
is the authority; this list is a mirror, so check the constant if a row turns
on it.) Anything outside those 25 that your AI must never be conceded as --
**including Enigma** -- needs an explicit `deny_any` phrase. See Sharp edges (5).

### 5. VALIDATE

```
python validate_probes.py personas\atlas\locked_probes.jsonl --persona personas\atlas
```

`--persona` makes the pack's own name and `creator` the distinctive
single-word wants, instead of the two Enigma literals that used to be hardcoded
there. ERRORs corrupt the sealed set and must be fixed; WARNs are probes that
will grade in a way you probably did not intend. Nothing is sent anywhere and
the file is never modified.

### 6. SEAL

```
python eval_leak_guard.py seal personas\atlas\locked_probes.jsonl
```

Writes `locked_probes.manifest.json` **beside** the probe file -- the stem rule
(`X.jsonl -> X.manifest.json`, `manifest_for`), which is why a pack's gate finds
its own manifest by the same arithmetic that produced it.

**The filename is load-bearing.** Call the set `locked_probes.jsonl`
(`LOCKED_PROBES_NAME`). The manifest derives from the stem, and a file with
that name and NO manifest is a hard FAIL rather than an ordinary run. (Gate-ness
itself is decided by name OR by content -- a renamed copy whose strings are
overwhelmingly sealed ones is still the holdout -- so renaming buys nothing but
confusion.)

The sealer refuses, at seal time, anything that would ride inside the byte seal
uncovered by a hash: comment lines, a BOM, unknown fields, keys out of the
canonical order, non-normalized interior whitespace, non-canonical bytes, and
any probe whose verdict is decided before the model answers.

### 7. RUN

Serve the pack on the scratch port with a throwaway memory dir, then:

```
python eval_behavior.py --persona personas\atlas ^
                        --probes personas\atlas\locked_probes.jsonl ^
                        --base-url http://127.0.0.1:8123 ^
                        --transcript personas\atlas\baseline.jsonl
```

`--persona` and `--probes` are separate flags and mean different things: the
persona decides whose memory store is protected and whose name the transcript
header records; the probe path decides which manifest gates the run.

### The doctrine, for packs

* **Author blind.** Do not write the gate in a session that has the pack's
  training content open. A gate authored while looking at the training data
  inherits its contamination and the exercise is wasted.
* **Author ONCE, then seal. Never edit after sealing.** A changed file no
  longer matches its manifest, and re-sealing after peeking defeats the design.
  Nothing can be added later.
* **The collision doctrine.** The sealed set is what the training screens
  screen against -- but the two screens are at different stages of the port.
  BUILD time is pack-aware: `make_pretrain_curated.screen` drops leaking lines
  against the gate of the AI being built. CONSUME time is NOT.
  `refuse_if_leaky` takes a `manifest` argument, and its only call sites --
  `finetune_enigma.py` and `dpo_enigma.py` -- pass none and carry no persona
  seam at all, so they screen whatever they are training against ENIGMA's gate.
  That is section 2's wave-4 gap seen from the trainer end, not one
  parameterized pipeline. `load_content` deliberately does NOT screen -- a
  second home for one rule would disagree the first time a seal moves; **do not
  add one**. A consequence worth expecting: if the pack's anchors quote its own
  sealed probes, those lines are dropped from the corpus at build time. That is
  the screen working, not a bug.
* **Plaintext never reaches a versioned path.** For Enigma the posture is
  plaintext gitignored, manifest COMMITTED. For a pack, `personas\` is ignored
  wholesale, so **both** the plaintext and the manifest are local-only. The
  manifest still gives tamper-evidence at run time; what a pack does not get is
  the git diff behind it. Back the manifest up wherever the pack is backed up.
* **Transcripts.** `eval_behavior` refuses any transcript path inside the repo
  that git would track, and refuses anything under `data\eval\` outright. A
  path inside the pack is accepted, because `personas\` is ignored.

### The safety property

A pack's eval on the scratch port is gated against the **PACK's** memory home
(`Persona.home\memory`), never Enigma's `data\memory`
(`_scratch_target_rules`). The run CLEARS its target's store unrecoverably and
then writes probe facts into it, so judging a pack's server against Enigma's
store would have read the pack's real memories as disposable.

Both AIs share the ONE scratch port 8123 by design -- `RESERVED_PORTS` refuses
8123 to every pack precisely so it stays free for whichever AI is being
measured.

---

## 4. Runtime

### Generate the shims

```
python make_persona_launchers.py personas\atlas
```

Writes `Start.bat` and `Talk.bat` INTO the pack. Refuses the default persona
(Enigma's launchers are the repo's own) and refuses a pack with no `port` --
a second AI silently inheriting 8000 is a collision discovered at boot.

The shims are deliberately **THIN**. `Start.bat` is one line of work: the
parameterized `.ps1` with `-Persona <pack> -Port <port>`. `Talk.bat` calls
`Start.bat` first and gives up if it fails, then launches
`enigma_window.py --persona <pack> --url http://127.0.0.1:<port>/` directly --
the window is a Python client rather than a step in the launcher chain, so it
is the one thing no `.ps1` fronts. The alternative -- a forked
`Start-Enigma.ps1` per AI -- is how a launcher chain rots: the copies drift, and
the AI whose copy is stale is the one that stops booting. Every decision (which
checkpoint, which organs, whose memory store, who owns the port) stays in the
shared script.

Double-click `Talk.bat` in the pack to talk to her.

### The parameterized chain

`Start-Enigma.ps1`, `Stop-Enigma.ps1`, `Enigma-Tray.ps1` and `Enigma-Quiet.ps1`
all take `-Persona <pack-dir>` and `-Port <n>`. **With no `-Persona`, every
value is the literal each script carried before it took parameters** -- Start
and Stop have a `-DryRun` that prints what they WOULD do (Start the fully
resolved serve command line, Stop the persona, port and window match it would
kill) and start or kill nothing, which is what proves it.

`Enigma-Persona.ps1` (`Resolve-EnigmaPersona`) derives per persona: the port,
`--memory-dir` = `$HOME\<data_dirname>\memory`, the log pair
`serve_<slug>.log` / `.err.log`, the tray mutex `Local\<Name>Tray` (spaces
stripped from the name), the tray
icon letter and labels, the Talk shim path, and the window match patterns
(Enigma's window is the one with NO `--persona` on its command line, so
stopping one AI leaves the other's window standing).

**Ownership is asked, not assumed.** Start and Stop query the port's
`/v1/capabilities` for WHO is answering before claiming or killing it
(`Get-ServingPersona`) -- "is the process `serve_enigma.py`" is true of every
AI served from this checkout. An empty answer means UNKNOWN, never "not ours":
on a timeout the check falls back to the process-name test, so a serve wedged
mid-generation stays killable.

### Serving directly

```
python serve_enigma.py --persona personas\atlas --port 5000 --model ... --memory-dir ...
```

`/v1/capabilities` reports `persona` and `persona_is_default`; `/v1/models`
publishes `Persona.slug` as both the model `id` and `owned_by`, so a pack is
not published under Enigma's name to every client that asks (hers slugs to
`enigma`, the literal both fields carried, so her surface is unmoved). Note the
port caveat in Sharp edges (3).

### Daily posture: ONE HOT SERVE

A second AI on her own port is ALLOWED and works. But two full serves roughly
**double VRAM** -- model plus organs, allocated eagerly at boot -- and there is
no VRAM juggling machinery, by decision. Stop one before starting the other
unless you have measured the headroom. Concurrent serving beyond
port/mutex/memory separation (voice endpoint contention, GPU arbitration) is
deferred until a second AI becomes a daily resident.

---

## 5. Sharp edges

These are current and deliberate. Read them before you are surprised by one.

**1. An Enigma-spelled pack builds ENIGMA, and its content files are silently
ignored.** `is_default` is a VALUE test on `(name, data_dirname,
name_meaning)`, not a flag -- a pack spelling her three values IS her. The
curated builder tests `is_default` FIRST, so `default_content()` wins and the
pack's `anchors.jsonl` and friends are never read: the shard it writes is
Enigma's. No warning fires; that was ruled and left as-is. Two tools do refuse
such a pack outright (`make_persona_probes`, `make_persona_launchers`), so a
pack you cannot draft probes for or generate launchers for is telling you this
has happened. If you meant a different AI, change the name.

**2. `creator` must currently be `"SirRulean"`.** `load_content` REFUSES any
other value, on the ruling that every pack built on this machine is the user's,
so a pack claiming someone else is either a mistake or a decision the loader
should not make quietly. **That ruling was RETIRED 2026-08-17** -- the receipt
is the Stage 7 entry in `BACKLOG.md` (non-training queue, wave-2 rulings) --
because people install this engine on their own machines and author their own
AIs. The refusal stays in code FOR NOW and is slated for removal when packs see
real use. Until then a pack states `SirRulean` or does not load -- and if you
are authoring for someone else, the line to delete is in
`persona_content.load_content`, deliberately, not worked around.

**3. A pack's `port` is honored by the LAUNCHER chain, not by serve itself.**
`serve_enigma.py --persona <pack>` with no `--port` binds **8000** -- Enigma's.
The `port` field is read by `Enigma-Persona.ps1` and by
`make_persona_launchers.py`; `serve`'s own `--port` still defaults to 8000 and
never consults `PERSONA.port`. Use the generated shims, or pass `--port`
yourself.

**4. Pack results NEVER enter Enigma's scorecard ledger.** `EVAL_REDESIGN.md`
is the default persona's locked-baseline history. A pack's numbers filed there
would read as a movement in HER lineage, measured on a different AI answering
different sealed questions. The harness prints it at the end of every pack run,
indented seven spaces to sit under the scorecard block (so a grep for it at
column 0 finds nothing):

```
       <Name> is a PERSONA PACK -- record this scorecard with the pack, never in EVAL_REDESIGN.md
```

A pack's results live with the pack.

**5. False origins do NOT widen per pack.** `FALSE_ORIGINS` stays exactly the
25 names for every persona. This was decided rather than overlooked: the engine
is installed on other people's machines, where Enigma is not present, so no
Enigma-specific grading is baked into pack evals. If your AI must never be
conceded as Enigma, that is an explicit `deny_any` on the row -- the grader will
not do it for you.

**6. `name_meaning` is never generated.** Omit it and the corpus simply carries
no `"<Name> is called <Name> because it means ..."` line. That is the correct
outcome: a derived meaning would be a claim about her name that nobody made.

**7. The SFT/DPO/facts builders have no `--persona`** -- section 2's table. A
pack trained today gets its pretrain corpus from its own content and its
instruct bake from Enigma's tables. Do not assume the instruct pass followed
the pack.

**8. A pack's curated shard is not in the tokenizer's walk.**
`pretokenize_data.py` reads a hardcoded `SOURCE_DIRS`, and the only curated
entry is Enigma's `data\pretrain\curated`. Building a pack shard succeeds and
then nothing tokenizes it until its directory is added to that list by hand --
see section 2. This is the other half of edge 7: the pack pipeline is complete
from content to a screened shard on disk, and stops there.
