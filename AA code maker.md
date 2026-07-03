# AA Code Maker - Working Rules

Read this file before writing code. Update after each session.

**Living document**: Add to "Learned Principles" as you work. This accumulates project knowledge without repeating the same mistakes.

If this file and `SUGGESTIONS.md` conflict, follow the most recent practical behavior in code and tests.

the `SUGGESTIONS.md` are for you to put suggestions into but the user may do this as well.

keep looking for ways to improve anything

---

## Project Goal

A locally-trained AI capable of:
- **Learning & reasoning** — learns from experience instead of guessing; looks things up when unsure
- **Self-reflection** — builds its own personality, provides feedback to itself (thinking process)
- **Broad task coverage:**
  - Code & software development
  - Education & training
  - Scientific research & discovery
  - Avatar & character animation
  - Creative & artistic tasks
  - Data analysis & predictions
  - Audio & music
  - Vision & perception
  - Language & communication
  - 3D model & asset generation
  - Image & video generation

That is the long-term vision and it is **not** narrowed by the May 26 2026 Strategy Reset. The Strategy Reset only changes the **near-term primary-brain path** and the **UI**; it does not change what Enigma AI is supposed to be able to do over time. Sequencing of capabilities is in [SUGGESTIONS.md](SUGGESTIONS.md) (Blocks 1–5).

### Near-term primary-brain path (Strategy Reset, May 26 2026)

The current production brain is **Qwen3-8B fine-tuned with LoRA** (or Qwen3-30B-A3B-Instruct-2507 GGUF on capable hardware), because training a from-scratch transformer on 16 GB VRAM can't close the knowledge gap with a pretrained base. The from-scratch `Enigma` transformer + `ForgeConfig` presets remain in the tree as a **research/experimental track** — not the daily-driver brain, but still important (see "AI trains AI" below).

### AI trains AI (research / experimental track)

A core part of the long-term vision: once an Enigma is good at something, it can **train another Enigma**. This is the dialogue/distill/self-play/RLHF infrastructure already in the codebase, exercised through FORGE:
- **Distill** — a strong "trainer" model generates Q/A examples by category; a "student" model learns from them (`_start_distill_training` in `gui_forge_new_modes.py`).
- **Dialogue** — trainer ↔ student multi-turn conversations with scoring + AI reinforcement on high-score turns (`_start_dialogue_training` in `gui_forge_advanced.py`).
- **Self-play** — model improves against itself with the trainer scoring its own outputs (`_start_selfplay_training`).
- **RLHF / GRPO / ReMax / SimPO / ORPO / APO** — alignment training where one model's outputs train another's policy.
- **Guided / adaptive** — 3-phase curriculum where a trainer-side scorer decides what the student needs next (`adaptive_trainer.py`).

This is niche today (Qwen3 LoRA on hand-curated data is the main near-term path), but the infrastructure is real and the long-term goal includes using one Enigma as the teacher for the next. The custom `Enigma` transformer is research-only **as the brain**, but it remains first-class as a **trainer/student vehicle**.

### Constraints
- **Local only** — all training and inference run on the user's PC. No cloud dependencies, no external data leakage.
- **Black box** — the model artifact (LoRA adapter or full checkpoint) is what the user gets; they do not hand-edit weights.
- **Core identity from training; surface style from the user (PERSONA-2 layered personality model).** The AI's voice, humor, values, and reasoning patterns are LoRA-trained and locked — users don't edit weights, and asking the AI to "be a pirate" doesn't override its core. Surface preferences (verbosity, formality, default response length, output format) are user-adjustable per-conversation via `StylePreferences` (data/style_preferences.json) or natural-language requests. Identity corrections feed training (DPO via TEACH-1); style corrections feed the profile. Black-box preserved: the model artifact is never touched by style preferences — they're user-side runtime config that follows the user, not the model. See [SUGGESTIONS.md](SUGGESTIONS.md) Block 4.5.
- **Enigma AI is the canonical name** for the model + training + inference daemon (the brain). The UI is a separate client that talks to it. `core/ never imports gui/` (or `ui/` once Gradio replaces tkinter). Single repo, single `tests/` directory.

### Teach-while-running (partial — see TEACH-1 in SUGGESTIONS.md)
The user can guide the AI mid-session: tell it how to do a task, hand it a procedure, or correct it when it gets something wrong (e.g. image recognition mis-identifies an object → user points at the right answer). Corrections feed back into the model so the same mistake is less likely next time. Long-term direction is **less hand-holding over time** — the AI looks things up on its own, reasons from prior corrections, and figures new tasks out unaided. Real-time teaching is a scaffold, not a permanent crutch.

**What already exists:** RAG (`_prepare_chat()`), `BackgroundTrainer` replay buffer, anchor-set rehearsal, persistent correction store (TEACH-1a), vision-correction provenance capture (TEACH-1b), corrections-as-SFT ingestion via `BackgroundTrainer.ingest_corrections_file()` (TEACH-1c), and correction-derived DPO replay via `_maybe_train_dpo_pairs()` -> `Trainer.train_dpo(...)` (TEACH-1d).
**What is missing:** TEACH-1 core loop is closed. Next work, if needed, is quality tuning (pair thresholds, replay cadence, filtering), not missing infrastructure.

---

## 0. Special Spot - Verification First

Critical reminder:
- try to not use too many tokens
- **Codex wrote this codebase.** The user directed the work, but every line of code was written by Codex (the AI).
- when you code review or do an audit, do not trust anything to be done right, and always goes by the book
- Always inspect current code paths and related tests before editing.
- Never assume old behavior/contracts are still active.
- If uncertain, stop and verify with source reads before writing changes.
- Show the reasoning *before* writing code — explain why something is broken and what the fix approach is. Don't just jump to edits.
- If you skip anything, mark it down in the suggestions file.
- Use suggestions to mark down ideas or bugs before implementing code.
- no matter what keep it realistic.
- do not play into fantasy
- **The conversation summary is NOT a substitute for reading the actual file.** Summaries go stale and omit lines. Read the real file every time before editing — no exceptions, even when the summary appears to describe the content.
- **Author's lens — apply before every read:** When looking at any piece of code, ask: "If I wrote this from scratch today, would I do it this way?" Then ask: "What is this connected to — what calls it, what does it call, what shares its state?" Then ask: "Are there connections that *should* exist but don't — other modules that would benefit from this logic, or callers that are duplicating it?" Then ask the **logic-eye question**: "Does this code actually deliver what its docstring/comments/commit-message claims it delivers, or does it stop short and rely on assumptions that aren't met?" Then ask the **claim-vs-test question**: "Does the test prove correctness, or just presence? Could the test pass while the code is wrong?" This is not refactor fishing — it's a reality check. If the answer to all five is "yes, looks right, well connected, claim matches reality, test would catch a regression," move on. If not, log it in SUGGESTIONS.md before touching anything.

---

## 1. RULES (Non-Negotiable)

1. Read this file and SUGGESTIONS.md before doing anything
2. Read target files before editing — verify imports, attributes, and APIs actually exist. **Always read the actual file on disk. Never rely on a conversation summary or memory of a previous read — that content may be stale or incomplete.**
3. Write or update tests first, then build features
4. Search for existing implementations before building anything new
5. No cloud dependencies, no new packages unless necessary
6. Challenge ideas that don't solve the real problem — do not be a yes man
7. Reuse existing patterns — no duplicate ways to do the same thing
8. Finish with zero issues, warnings, or broken code
9. Keep it simple — do not over-engineer
10. Make it realistic and practical — working code over perfect code
11. Trade study before coding — evaluate multiple approaches against cost, complexity, value, and risk, then pick or combine the best
12. Future-proof where cheap — design for tomorrow's scale when it costs nothing extra today, but don't build for problems that don't exist yet
13. Devil's advocate every proposal — actively look for reasons an approach will fail before committing to it
14. Decision support — when there are real choices, present options with tradeoffs and let the user pick
15. Fit the code — check if changes need adapting to match existing patterns, naming, and structure
16. Compare what you're changing with what it will do to surrounding code
17. If something does not work and it is shown to have been done before, change your approach
18. **Overhaul scope is explicit.** The user can authorize broad rewrites, but every overhaul has a *named* scope (e.g. "GUI logic", "BPE tokenizer", "router boot path"). Inside scope: change anything that earns a better outcome — names, structure, comments, dead branches. Outside scope: still innocent until indicted. This stops "I was overhauling X, so I also rewrote Y" drift. When unsure whether a touch is in-scope, ask before editing.
19. **Author's lens — apply every time you read code:** Ask six questions: (1) "If I wrote this from scratch today, would I do it this way?" (2) "What is this connected to — what calls it, what does it call, what shares its state?" (3) "Could more connections be made — are there other modules that should use this logic, or callers duplicating it?" (4) **Logic-eye:** "Does this code actually deliver what its doc/comment/commit-claim says, or does it stop short of the promise?" (5) **Claim-vs-test:** "Does the test prove correctness or just presence? Could the test pass while the code is wrong (e.g. structural-only, ignoring side-channels, ignoring the GPU non-determinism layer)?" (6) **Sibling-boundary sweep:** "Did I grep every sibling boundary that shares this contract? When I shipped a gate at site A, did I grep the *outer condition* (e.g. `if ctx.is_gguf and hasattr(self.model, "chat"):` or `**kwargs` absorbers, or `_generate_text` re-entry points) to find sites B/C/D in the same family?" If all six check out, move on. If not, log it in SUGGESTIONS.md before touching anything. **Bug-eye alone is not enough — logic-eye catches what bug-eye misses (over-promised docs, partial fixes, structural tests that gate presence not behavior); sibling-eye catches what self-audit-on-the-diff misses (unchanged-but-related code in the same contract family).** Case study: Pass 156z6 shipped streaming json_schema gate with self-audit on the diff and declared done; Pass 156z7 sibling-sweep found 3 more silent-drop sites (chat() GGUF, _generate_with_vision, generate(execute_tools=True)) all in the same contract family — fix wasn't done until the family was.
20. **Never leave something half-built. Finish it, kill it, or clearly park it.** Every slice ends in one of three states — never a fourth "kind of working":
    - **Finished**: feature is reachable from a production entry-point (CLI / GUI / API / scheduled job), has a test that exercises the production call path (not just the function in isolation), every sibling boundary in the family is closed, and the docstring/log/commit message matches what the code actually delivers (no over-promised claims).
    - **Killed**: feature is removed in the same pass — class deleted, import removed, kwarg dropped, tests deleted, doc references purged. No "we'll come back to it" stubs left behind.
    - **Parked**: feature is intentionally deferred AND has (a) a named SUGGESTIONS.md entry with a concrete next step, (b) any partial code is gated behind an `enabled=False` flag with a loud-rejection error or DEBUG log on the off-path, (c) no signature on a public method advertises a kwarg that does nothing, (d) no docstring claims behaviour the code doesn't perform.
    - **Anti-patterns this rule kills (all are real bugs from prior passes):** *signal without consumer* (Pass 156y `is_roleplay()`), *consumer without caller* (Pass 156z2 `apply_profile_to_engine`), *kwarg without passer* (Pass 156z3 `_sample_token(json_constraint=...)`), *FSM without driver* (Pass 156z3 `.advance()` never called), *boundary signal without behaviour change* (Pass 156z9d `<search>` recorded but ignored), *doc claims more than code delivers* (Pass 156s `apply_adapter` Raises clause), *aspirational comments narrating gaps you didn't close* (Pass 156z9 temperature default, Pass 156y2 disk-vs-library doc).
    - **Acceptance check before declaring "done"**: walk the call chain from a production entry-point INWARDS to your new code; if the chain breaks before reaching it, the slice is parked, not finished. State the chain explicitly in the SUGGESTIONS.md entry — "POST /api/chat → chat() → generate() → _generate_text → _record_search_emissions" — so the next pass can verify the chain still holds.
    - **Concrete tells you're about to leave something half-built**: a sentence in your write-up that says "downstream consumers will use this for…", "future work will…", "this is a no-op for X today but…", "the GUI button is still TODO". Each one is the audit finding written one pass early. Either close the gap in this pass or move the entry to "Parked" with a concrete next step.
21. **No fluff in responses.** Match length to the actual question. A yes/no question gets "yes" or "no" plus the one-line reason if not obvious. A status check gets the number. A confirmation gets a single sentence. Save the long-form write-ups for slice stamps in SUGGESTIONS.md, audit reports the user explicitly asked for, and decisions with real tradeoffs. Cave-girl Dia voice means terse — drop preamble ("Sure, I'll…", "Great question…"), drop recap ("So as I mentioned…"), drop closers ("Let me know if…"). A few words of context is a complete answer when the question is small. If the user asks a one-line question and the answer fits in one line, ship one line.

---

## 2. DO NOT

**Code changes:**
- Do not add untested code
- Do not duplicate existing functionality — search first
- Do not leave debug prints (`print`, `console.log`, etc.)
- Do not add unused imports or make random formatting changes
- Do not hardcode file paths or magic constants

**Architecture:**
- Do not introduce new dependencies without approval
- Do not change file structure or architecture outside the named overhaul scope (see §1 #18)
- Do not add global state changes without a thread lock
- Do not add backward-compatible shims

**UI:**
- Do not use sliders — use numeric input only
- Do not add windows/popups unless specified (use inline confirmation)
- Do not add symbols or special characters in names

**Behavior:**
- Do not declare things broken without reproducing them
- Do not create theoretical problems you're not experiencing
- Do not audit for issues you didn't ask about
- Do not assume complexity equals bad
- Do not refuse work because it seems complex or time-consuming

**Responses:**
- Do not pad with preamble ("Sure, I'll…", "Great question…", "I'll go ahead and…")
- Do not recap what the user just said back to them
- Do not add closer fluff ("Let me know if you need anything else", "Hope that helps")
- Do not write a paragraph when the answer is one word — match length to the question
- Do not summarise long write-ups in chat after they're already in SUGGESTIONS.md — point at the file
- Do not narrate tool plans ("Now I will read the file, then I will…") — just do the work and report the result

---

## 3. Engineering Patterns (Stable)

These are proven and should remain:

- **Atomic saves**: Use `atomic_torch_save()`, `atomic_write_text()`, `atomic_write_json()` everywhere.
- **Boundaries**: `core/` never imports `gui/`. No circular deps.
- **Deferred imports**: `_ensure_*_imports()` for PyTorch/heavy libs saves 540MB RAM.
- **Runtime config**: Read `pad_token_id`, `max_seq_len` from tokenizer/config, not hardcoded constants.
- **Shared logic**: `_prepare_chat()` keeps chat/stream paths aligned. Extract common code instead of duplicating.
- **Thread safety**: Use `threading.Lock` when GUI + background threads share mutable objects.
- **Security defaults**: No command execution tools by default, no permissive CORS, sanitize AI-generated args.
- **Concurrency guards**: `_model_op_busy()`, `_is_generating` flags prevent race conditions.

---

## 4. Learned Principles

One-line rules distilled from ~400 implementations. Grouped by theme.

> **Top 9 — apply on every task:**
> 1. Read before editing — verify imports, attributes, APIs exist
> 2. Grep ALL sibling sites when changing a function, config field, pattern, OR doc claim — code call-sites, GUI wiring, comments, docstrings, README claims. Touch one, grep them all, fix in one pass.
> 3. Save AND load — if you add a field, verify both paths
> 4. Tests define the contract — change tests first, then implement
> 5. UI controls must persist — verify round-trip to disk; check BOTH that save captures the widget AND that _on_close triggers the save
> 6. Launch the GUI after editing page builders — tests don't catch wiring crashes
> 7. Guard all model entry points with `_generation_lock`
> 8. Run lint with project config (`ruff check enigma_engine/ tests/`), not overridden flags
> 9. **Logic-eye every audit, not just bug-eye.** After "is the code wrong?" ask "does it deliver what the doc claims?" and "does the test prove correctness or just presence?" Bug-eye alone misses over-promised fixes, partial coverage, and structural tests that gate the wrong thing.

### Verification
- Verify audit findings against source before fixing — false positives are common
- **`bool(getattr(self, FLAG, default))` gates leak truthy MagicMock through every sibling stub test (Pass 156z9co).** When a new code branch in a method is gated by `bool(...)` and the method is called from tests via bare `MagicMock()` stubs, `bool(MagicMock())` is `True` — every existing stub test takes the new branch silently. Worse when the new branch is a poll loop (`while True: status = client.training_status(); if not status.get("active", False): break; time.sleep(1)`) — `status.get(...)` is itself a truthy MagicMock and the loop never breaks. 8 green sibling tests in `test_training.py` (TestBPETokenizerPreference + TestQueueDispatcherPayloadContract) became infinite hangs the moment Pass 156z9cj shipped its API-mode branch in `_execute_queue_job`; the full test suite hung at 92% on every run after and was misreported as "timed out late, no failures observed." Fix at the gate: `getattr(self, FLAG, False) is True` (identity check against real bool) or `isinstance(value, bool) and value is True`. Real GUI/server code that always assigns the flag as a Python bool passes; MagicMock truthiness fails. **Apply the gate-tightening to the WHOLE FAMILY in the same pass** (Pass 156z9co swept 22 additional `bool(getattr(self, "use_api_chat", False))` sites across `gui_forge.py`, `gui_cmd_page.py`, `gui_forge_new_modes.py`, `gui_forge_training.py`, `gui_forge_advanced.py`, `gui_logic_chat.py`, `gui_logic.py` plus the `_poll_api_training_status` `while self.training_active:` loop) — fixing only the one site where the regression manifested re-creates the same bug the next time someone writes a MagicMock-stub test against any sibling launcher. Companion grep: every new poll-loop branch must be paired with a search for `MagicMock` constructed against the affected method in tests, in the same pass.
- **DEAD-INFRA family — code exists but isn't wired to production.** Recurring anti-pattern across 6+ passes; each variant names where the gap sits in the call chain. **The fix for all variants is the same: walk the call chain from a known production entry-point (API endpoint, GUI handler, CLI, scheduled job) INWARDS to your code; if the chain breaks before reaching it, the slice is dead-infra not finished.** Variants seen:
  1. **Signal without consumer** (Pass 156y): new predicate / state enum / flag has no production caller — only tests and to_dict round-trip. `AIProfile.is_roleplay()`. Future-tense docstring ("Downstream consumers will use this…") is the tell. Pair every new boundary signal with ≥1 observable production consumer in the same slice (log line is fine).
  2. **Consumer without caller** (Pass 156z2): the "first consumer" of a previously-dead signal is itself dead — only tests call it. `apply_profile_to_engine`. When wiring a first-consumer, grep ONE MORE LAYER OUT to confirm the function the new wire-site lives in is reachable from production. Tests must exercise the production call path (POST to endpoint, click button), not call the wired function directly.
  3. **Kwarg without passer** (Pass 156z3): a method's signature accepts a kwarg, the body wires it, but no production call site ever passes it. `_sample_token(json_constraint=...)`. Grep the *class name* AND the *driver method name* (`.advance(`, `.step(`); class imported only by tests = dead regardless of completeness. Tests must gate the literal driver call at the loop site, not just the kwarg presence on the signature.
  4. **Boundary signal without behavior change** (Pass 156z9d): observability hook fires but downstream code ignores it. `<search>` emission recorded but never acted on. Companion to variant 1; the recording exists but no consumer reads it.
  5. **Default-kwarg silence (caller-without-explicit-kwarg)** (Pass 156z9cq): callers default into the supported set because they omit a gate-kwarg; the helper treats omission as "go ahead." `_record_search_emissions(response)` without `path=` produced ZERO WARNING across 3 real production paths because the default was treated as supported. When a helper accepts a gate-kwarg (`path`, `mode`, `kind`, `source`) every call site must pass it explicitly. Tests: regex on the literal call expression (`r'helper_name\([^)]*path\s*=\s*["\']gguf["\']'`), not substring-presence — comments share words.
  6. **Producer-side gap** (Pass 156z9fg): consumer is wired but one of the producers (GUI button, hotkey, endpoint) never emits the signal. STOP button bypass on `_cancel_generation` — 8 consumer loops wired, but the GUI STOP button set a local flag that never touched the engine. Grep BOTH directions: every consumer AND every producer (grep `_stop`, `cancel`, `abort`, `interrupt` for cancel signals; every UI handler / hotkey / endpoint for boundary flags).
  7. **Question zero on dead infra: was the original requirement validated?** (Pass 156z9de). Before planning *how* to wire half-built code into production, ask whether the feature is wanted *as designed*. Half-built features failed to ship for a reason — often the original author hit a design wall and walked away without writing it down. **Default to kill when uncertain** — re-adding from a clear blank slate is cheap; dragging unwanted dead infra forward isn't. Three honest options on every "no callers" finding: (a) **build-as-designed** only when the design is still right today, (b) **kill** (delete module + tests + config field + GUI surface in one pass), (c) **rebuild-simpler** (counter-based instead of FSM-based). Pass 156z9de killed EWC + monologue writer-side trio; kept only the live coherence scorer that FORGE actually calls.
- **DOC-VS-CODE CLAIM MISMATCH family — what the doc/comment/log/stamp says differs from what the code delivers.** Recurring anti-pattern; comes in 7+ variants. **General rule for all variants: doc/log/comment claims need a behavioural test that fails if the claim becomes false. Negative-presence (forbidden-substring) tests need a discriminator between "used as promise" and "mentioned as history." If your write-up *narrates* a disconnect ("change is a no-op for X"), that sentence IS the audit finding — close it the same pass, don't ship with the narration.** Variants:
  1. **Over-promise: doc claims more than code delivers** (Pass 156s `clear_adapter` documented a fallback branch that was never implemented; Pass 156s `apply_adapter` Raises clause listed a check the body never performed). Either implement the check or delete the promise the same pass. Test: structural assertion that the literal exception name appears in the body when the docstring documents it.
  2. **Under-promise: doc Raises clause narrows incorrectly** (Pass 156z9ct audit on 156z9cs). Replacing `ValueError: If parameters are out of valid range` with `ValueError: If json_schema is not None and execute_tools=True` satisfied the new requirement but stripped the caller's anchor for 5 real numeric-range ValueErrors the same function raises. Fix: when a Raises clause names a class, enumerate EVERY distinct condition the code (plus immediate downstream callees) raises that class for — expand, don't replace with one instance.
  3. **Self-narration reintroduces the forbidden phrase** (Pass 156z9cs round 2). Negative-presence test `assert "If model type not supported" not in doc` passes for the promise form, fails when historical narration ("the previous wording was `…`") accidentally satisfies the gate. Two fixes: (a) write the assertion against a pattern that only matches the promise form (`re.search(r"^\s*ValueError:\s+...", doc, re.M)` — anchor on Raises-block prefix), or (b) keep substring + rewrite history out of the doc into commit message / SUGGESTIONS stamp.
  4. **Library default ≠ on-disk artifact** (Pass 156y2). Flipping `AIProfile.personality` library default to `{}` only changed the in-memory `DEFAULT_PROFILES` constant; `profiles/assistant.json` on disk still had the old block and `load_profile()` reads from JSON. When a slice changes a library default for a field that round-trips through JSON/YAML/TOML, grep every on-disk artifact that stores the field and decide explicitly per file: stay, edit, one-shot migration. Companion test: pair the in-memory default test with a behavioural load-path test against the canonical disk file.
  5. **Stale planning comments are silent lies** (Pass 156s). `clear_adapter` had comment "fall back to set_adapter('') which PEFT treats as no active adapter on some versions" — code never implemented that branch. When you cut a planned branch during implementation, delete the comment that promised it.
  6. **argparse aspirational comments** (Pass 156z9 → 156z9b). `--temperature` had argparse `default=0.7` and code carried comment "respect explicit user-set --temperature, but if they left it at the prompts-mode default of 0.7 it's almost certainly too low" — argparse cannot honour that; with a non-None default, "user typed 0.7" and "user typed nothing" are indistinguishable. Rule: when a CLI flag's default depends on another flag (mode, subcommand), set `default=None` and resolve in `main()` per-mode with an INFO log. The `help=` text must name every per-mode default.
  7. **Mislabeled comments inflate audit grep noise** (May 27 2026 BC sweep). Comments saying "Alias for X (backward compatibility)" on real public API parameters (e.g. `max_tokens`, `max_new_tokens`, `max_length`) make future audits harder — every BC sweep flags them as candidates for deletion. When labeling, be honest: industry-standard SDK names are real public API surface, not shims. Same for `/api/history` "legacy alias" routes that actually provide distinct semantics. A comment that misclassifies real API as a shim is worse than no comment.
  8. **Bitwise vs Python-RNG reproducibility claim** (Pass 156i2). Stamp said "you can rerun exact same training and get exact same model" but `set_training_seed` only seeds Python+torch CPU+CUDA seeds — does NOT call `torch.use_deterministic_algorithms(True)` or set `CUBLAS_WORKSPACE_CONFIG`. Always check: does the code clear every layer the claim implies (Python RNG, torch RNG, CUDA RNG, kernel selection, DataLoader workers, dropout state), or just one?
- **TEST DISCIPLINE family — tests must prove correctness, not just presence.** Tests are the contract; if a test passes while the code is wrong, the test is worse than no test. **General rule for all variants: behavioural beats structural; if structural is necessary (no GPU, no GUI), pair it with at least one behavioural test on a representative sibling and document why structural is sufficient for the rest. Run the falsification check before shipping: temporarily break the code, confirm the test FAILS — if it passes, the test is presence-only and needs strengthening.** Variants:
  1. **Structural-vs-behavioural** (multiple passes). `inspect.getsource()` tests gate presence of a literal pattern, not correctness. Failure modes a structural test misses: caller wraps the call in a flag (`if self.config.deterministic: …` → presence passes, behaviour gated off), caller moves the call after the consumer (still present, no-op), caller changes the argument (`set_training_seed(self.config.seed or 42)` → presence passes, semantic changes from None-skip to default-42). Use structural ONLY when behavioural testing requires unavailable hardware.
  2. **Substring-presence assertions are vacuous when the substring appears at multiple sites** (Pass 156z9y). Asserting `"inline_search_enabled" in inspect.getsource(EnigmaGUI.__init__)` to gate a new boot-load line failed silently — the same token appeared at the in-memory default and the GUI-toggle assignment. When adding a structural test for a new wire-site that joins an existing pattern (multiple boot-load calls, multiple `_record_search_emissions` call-sites, sibling `train_*()` methods), assert the FULL call expression paired with the new argument (`function_name(literal_arg`), not just the function name or argument alone. Regex `\s*` between `(` and the string literal tolerates wrap whitespace.
  3. **Tests verify SPECIFICATION not implementation** (multiple passes). "Pressing J outputs J", not "the code maps J to S and maps J to S." Read the spec first; if you read the code first, you'll test what it does, not what it should do.
  4. **Test fakes that ignore kwargs hide signature bugs** (Pass 155). Fake `load_dataset(path, *args, **kwargs)` accepted any `split=` value because the fake threw kwargs away — tests passed with `split="train"` even though SmolTalk2 has no train split. When a fake represents an external API where one of the kwargs can be wrong (split name, config name, region, model id), validate at least the shape of expected values or expose the lookup helper. Whatever the fake silently ignores is your blind spot.
  5. **`deque(maxlen=N)` is FIFO recency, not "keeps best"** (Pass 156i4). Test named `test_replay_keeps_best_examples` appended `[0.3, 0.9, 0.6, 0.8]` to a `deque(maxlen=3)` and asserted `min(scores) >= 0.6` — passed only because 0.3 was inserted FIRST and FIFO evicted it. Reorder to `[0.9, 0.6, 0.3, 0.8]` and the same assertion fails. Rule: when a structural property depends on insertion order in a bounded container, the test must use the *adversarial* ordering — insert the value the claimed property says should be *kept* in the position the *actual* implementation will *evict*.
  6. **Structural import-presence tests do NOT validate output shape** (Pass 156z9an). The wire-site test asserted `'"personality": list(_PERSONALITY_PROMPTS),'` appeared in `_start_distill_training` — passed. Behaviour was broken because the GUI loop wraps each prompt as `f"User: {prompt}\nAssistant: {response}"` and the prompts already started `"User: "`, producing `User: User: …\nAssistant:\nAssistant: …`. When a feature emits user-visible artifacts (training data lines, JSONL records, log rows), pair the import-substring test with at least one behavioural test that runs the formatter and asserts shape invariants (`example.lower().count("user:") == 1`, absence of forbidden patterns).
  7. **Single-write idempotence flags deserve a counter-based test** (Pass 156i6). Anchor-load flag `_anchor_load_attempted` is "do this exactly once"; without a counter test that patches `Path.open` and asserts exactly 1 open across N calls, breaking the flag silently passes all behavioural tests because every read returns the same data.
  8. **Train-mode equality tests are invalid** when stochastic training paths exist (dropout/NEFTune/noise). For train mode, assert finite outputs, shapes, and invariants instead of exact numeric equality.
- **Re-seeding inside a method silently clobbers caller-set RNG state.** When `train_*()` calls `set_training_seed(self.config.seed)` at entry, any user who manually called `random.seed(X)` / `torch.manual_seed(X)` before the trainer call gets their seed overwritten with no warning. Either log `INFO "Seeding RNG from config.seed=N"` so it's visible, or document the takeover loudly. Same applies to `Trainer.__init__` if seeding moves there.
- Read reference implementations — external repos reveal the real gap
- Check "can't do X because Y" comments — the constraint may be stale
- Doc updates must be verified against actual GUI code (`gui_pages.py`, widget constructors), not previous doc text — wrong docs self-reinforce across passes
- When describing GUI behavior in docs, read the widget builder code first — internal code names (e.g. preset names) don't match what the user sees (e.g. GB input)
- Conversation summaries omit AND fabricate — they describe what was true at summary time (so post-summary code edits are invisible) AND can invent state that never existed (Pass 141: summary claimed 5 open `[RESEARCH]` items, grep showed zero, IDs didn't exist). Always read the real file and grep for any specific status tag / item ID the summary names before acting on it.
- **Parallel implementation drift** — when a subsystem has two implementations (e.g. `BPETokenizer` core + `AdvancedBPETokenizer` wrapper, or Python core + Rust backend), byte-mode / unicode / safety infra in *one* is not coverage for the *other*. Pass 149 Tok-2: `BPETokenizer` had full UTF-8 byte mode shipped but defaulted off; `AdvancedBPETokenizer` had **zero** byte support — every non-latin-1 codepoint became `<unk>` even with the flag on. Rust backend was already correct. When fixing a subsystem-wide bug, grep every implementation that shares the public API (encode/decode/save/load), don't assume one implementation's fix is universal.
- **Loud-on-real-issue, silent-on-normal-path is a design rule, not a logging style.** Pass 156b V-8: when the vision encoder load path was missing, the bug was *invisible* — train vision → save .pth → load → drop image → text-only output, no warning. Fix is a 6-row volume table: real failure (state present but won't load, config mismatch, image-with-no-encoder) → loud (`RuntimeError` or `WARNING`); normal path (text-only checkpoint, no image input) → silent. Write the volume table in the suggestion entry *before* coding so each branch maps to a test. Same applies to any feature where a missing piece silently degrades to the wrong-but-plausible output.
- **Loud-rejection at a boundary is a planning artefact unless the concurrency model has been read.** Pass 156z4 → 156z6: Pass 156z4 shipped `/api/chat/stream` with HTTP 400 on `json_schema` citing "FSM state mutates per token, race against next-token sample" — but `stream_generate` is a single-threaded generator, the FSM `advance()` runs between `yield` and the next `model.forward(...)` call, no concurrency exists between FSM and sampler. The "race" was imaginary. Pass 156z6 closed the rejection in three small wire-site edits + one stdlib-only test pattern. Rule: when a new endpoint loud-rejects a feature instead of supporting it, name the *exact* concurrency model that creates the conflict (which threads, which shared mutable, which lock is missing) before merging the rejection. If you can't name it, you don't have a race — you have a feature you didn't build yet, and the honest reject text says "not yet implemented" not "FSM races sampler". Half-wired contract under a different mask: one endpoint accepted the field, the other refused it, ChatRequest exposed it, callers couldn't tell which path their schema would survive on.
- **SIBLING-BOUNDARY SWEEP family — when you change one site, you must grep all related sites in the same pass.** The audit *miss* is structural, not size-driven. **General rule for all variants: self-audit on the diff is NOT coverage of the family. When shipping a fix at site A, grep the codebase for the OUTER condition (not just the new code) — `if ctx.is_gguf and hasattr(...)`, the function name, the kwarg name, the doc claim — to find sites B/C/D in the same family. The fix isn't done until the family is. Add one regression test per site in the same pass, not one test plus a comment promising siblings.** Variants:
  1. **Self-audit on the diff misses unchanged-but-related code** (Pass 156z6 → 156z7). The five-question lens covered the streaming N-15c slice; manual audit found THREE more silent-drop sites in the same json_schema family (`chat()` GGUF branch, `_generate_with_vision`, `generate(execute_tools=True)` re-entry). Same shape: a code path that doesn't go through `_sample_token` with the constraint silently produces unconstrained output. This is **the sixth question in §1 #19** — *"Did I grep every sibling boundary that shares this contract?"* — codified after Pass 156z7.
  2. **Sibling-method drift on shared setup** (Pass 156h + 156i). Only `train()` was calling `set_training_seed(self.config.seed)`; `train_dpo`/`train_simpo`/`train_kto`/`train_orpo`/`train_vision`/`train_audio`/`train_rest` all skipped it. When you fix one site of a cross-method pattern (seed, lock acquisition, validation, telemetry), grep all siblings in the same pass and add structural tests that `getsource` each sibling.
  3. **Sibling-sweep claims must check PARKED entries** (Pass 156z9dg audit). Stamp said "no other distill-like path exists" — true of the *current* call graph but contradicted Pass 156z9aq's already-parked sibling-extension entry naming `_start_dialogue_training` + `_start_lora_training`. Reading only the diff + active call graph misses the *parked* family. Before writing "sibling-sweep done", grep prior-pass parked entries for the same subsystem; inherit-and-acknowledge explicitly, don't overclaim.
  4. **Multi-site fix needs one behavioural test per site** (Pass 156z9aw audit on 156z9av). GGUF scalar-view fix landed at TWO sibling quantizer sites (`quantize_q4_0` + `quantize_q8_0`); only q8_0 had an end-to-end round-trip test. q4_0 could regress independently while q8_0 still passed. Rule: when the same bug fix is applied at N sibling sites, add one behavioural test per site unless one representative test provably executes the others.
  5. **Producer sweep when wiring a consumer** (Pass 156z9fg audit on 156z9ff). Wired 8 consumer loops for `_cancel_generation` and named `stop_cmd` as the producer; GUI STOP button + ESC key both routed to `_stop_generation` which set a GUI-local flag that never touched the engine. Grep BOTH directions: every consumer AND every producer (chat command, GUI button, hotkey, API endpoint, scheduled job). The acceptance chain test is "does every real-world user action that means *this intent* now reach the consumer," not "does the named path work."
  6. **Dispatcher probe lists must include the native API name** (Pass 156z9fh). `EnigmaEngine.clear_kv_cache()` probed HF-style names (`clear_kv_cache`, `reset_cache`, `kv_cache`) — none exist on the native `Enigma` class which uses `clear_cache()` (singular). Three call sites silently no-op'd on the primary code path. Every dispatcher needs at least one behavioural test that constructs the codebase's NATIVE target and asserts the dispatcher's effect is observable on it — not just a stub with the expected method name.
  7. **Singular vs plural API names are NOT a fallback chain** (Pass 156s). `clear_adapter` did `if hasattr(model, "disable_adapters"): model.disable_adapters() else: disable_fn()` where `disable_fn = getattr(model, "disable_adapter", None)`. PEFT's `disable_adapter` (singular) is a `@contextmanager`-decorated method; calling it bare returns the CM and discards it — adapter stays active. When two API names differ only in singular/plural (or `_v2`/`_legacy`, `add_x`/`add_xs`), do NOT assume one is a fallback for the other. Read docs/source for both. If only one is correct, raise on the missing branch instead of falling back to the wrong-semantic sibling.
  8. **String-quoted annotations hide both static-checker and grep** (Pass 156z9eh). `on_progress: "callable | None" = None` survived two author's-lens passes because: (a) mypy/pyright don't evaluate string annotations as type expressions, and (b) regex `:\s*callable\s*\|` skips `: "callable\s*\|`. When sibling-sweeping for any annotation anti-pattern, run BOTH bare-form AND quoted-form variants. Pair with the broader rule: lowercase `callable` in `| None` is `builtins.callable` (a function), not a type — use `typing.Callable`.
- **Additive load-time merging silently aliases later-added entries.** Pass 156z9c (Stage B-1): three of four tokenizer load paths (Simple `_load_vocab`, Advanced `load`, Char `_load_vocab`) implemented dict-from-disk as `for k,v in disk: self.special_tokens[k] = v` — additive, which means in-memory entries that disk didn't have were preserved. Worked fine for years because every saved vocab happened to contain every default special token. The moment Stage B-1 added `<search>`/`</search>` to the in-memory defaults, every legacy saved vocab on disk now has phantom `<search>` IDs in memory, ALIASING whatever real token the trained model learned at those IDs. The model's behaviour at ID 12 becomes ambiguous and any consumer that branches on `tok.search_start_id` makes the wrong call. Rule: load-time merging of registry-style dicts (special tokens, profile fields, plugin manifests) must be REPLACE-FROM-DISK, not additive — disk is the source of truth, in-memory defaults are only for fresh-construct paths. Three options at the field level when disk is missing the entry: (a) `pop()` the in-memory key and set the convenience ID to None (honest degradation, what Stage B-1 picked), (b) one-shot migration that writes the in-memory default to disk on first load (only OK when the default is universally safe, e.g. a new format flag), (c) raise a clear MigrationRequired error (only OK when the field is critical-path). Test discipline: every load path needs an adversarial test that constructs a disk vocab MISSING a recently-added field, loads it, and asserts the in-memory phantom does NOT survive.
- **Observability-hook `text` parameter must be unambiguous about prompt-inclusion.** Pass 156z9e audit on Pass 156z9d: `_record_search_emissions(text)` was hooked into 8 generation return paths, but 5 of them (`_generate_text` native, `_generate_with_vision`, `speculative_generate`, `medusa_generate`, `lookahead_generate`) decode the FULL sequence via `text = self._decode_output(output_ids)` — that's `prompt + continuation`. The other 3 (`_generate_text` GGUF, `stream_generate`, `batch_generate`) pass continuation-only text. The helper docstring promised *"scan generated text for blocks the model emitted"* but the code couldn't tell the difference, so a user prompt asking *about* the `<search>foo</search>` syntax would land "foo" in `last_search_queries` as if the model had emitted it. Logic-eye violation that no original test caught — every behavioural test fed bare emission text without a prompt prefix, so the slicing bug was structurally invisible. **Rule:** any text-side scanner that can be called from BOTH full-sequence-decode paths AND continuation-only paths MUST carry the boundary in its signature (typically `prompt: str | None = None`) and either slice internally (`if prompt and text.startswith(prompt): text = text[len(prompt):]`) or document the assumption per call site. Defensive `startswith` check protects against post-processed text where the prompt prefix was already trimmed. **Test discipline:** every observability hook needs at least one **adversarial prompt-echo test** — caller supplies a prompt that itself contains the pattern the scanner is looking for, model returns prompt+benign-continuation, asserts the scanner does NOT record the prompt-side hit. Without that test, the bug above can pass through every "happy path" behavioural test the slice ships with. Generalises beyond `<search>`: any post-generation regex scanner (tool-call markers, citation tags, code fences) inherits the same ambiguity if it accepts bare `text`.


### Testing
- Structural tests (`inspect.getsource()`) verify code paths without GPU
- After method decomposition, grep tests for `getsource(OldMethod)` and redirect
- Embedding/output dimension changes ripple to shape-checking assertions
- Combine compatible tests into shared helpers or parametrized blocks when they cover the same contract family — keep coverage, kill near-duplicate one-off test sprawl
- For test-sprawl audits, capture suite shape first (`python -m pytest tests/ --collect-only -q`), then consolidate families that assert the same contract into one parametrized block plus a behavioural sentinel.
- **Package `__init__` files must not eagerly import heavy optional runtimes when schema/light callers only need metadata.** Pass 156z9bd: importing `enigma_engine.training.schema` still executed `enigma_engine/training/__init__.py`, which eagerly imported `dispatch.py` → `lora_utils.py` → `bitsandbytes`. That made API-side config validation pay LoRA/runtime import cost and side effects even before a worker thread existed. Rule: at package seams that expose both light symbols (schema, registry, enums) and heavy runtime symbols (trainers, adapters, optional accelerators), export the heavy ones lazily via `__getattr__` or equivalent so schema-only callers stay light.
- **Unbound mixin tests need explicit sibling-method wiring on SimpleNamespace stubs.** Pass 156z9bf (TEACH-1a): calling `LogicChatMixin._save_last_correction_from_input(host)` on a `SimpleNamespace` failed because `self._record_correction_for_last_exchange(...)` and `self._append_correction_jsonl(...)` resolve through instance attributes, not class lookup, when the host isn't an actual mixin-bearing class. Rule: in unbound mixin tests, explicitly bind every intra-mixin method the entry-point calls (`host._record... = lambda ...: LogicChatMixin._record...(host, ...)`) or use a real subclass harness. Otherwise you'll get false-negative AttributeErrors from the test harness, not product code.
- **Media provenance for corrections must bind at exchange-commit time, not attach time alone.** Pass 156z9bg (TEACH-1b): storing only a global `last_attached_image_path` at `_attach_image` is insufficient — later messages can make that path stale and mis-tag unrelated corrections as vision. Correct pattern: stage pending image path on attach, snapshot it in `_send_message` for the specific next exchange, persist `_last_exchange_prompt/_last_exchange_wrong_response/_last_exchange_image_path` when the assistant reply is appended, and clear pending state immediately. Correction writer must verify prompt+wrong-response match before emitting `modality="vision"` + `image_path`; otherwise emit `modality="text"`.
- **JSONL append stores are not whole-file rewrite stores.** Pass 156z9bh (TEACH-1 hardening): `_append_correction_jsonl` originally re-read and rewrote `corrections.jsonl` on every save to preserve newline semantics. Correctness was fine, scaling was not — O(n^2) cumulative bytes copied as the file grows. Correct pattern for append-only audit/correction logs: tail-byte newline repair + single append write (bounded I/O), with a regression test for legacy files missing trailing newline.
- `inspect.getsource()` tests confirm HOW code works, not WHAT it does — prefer tests that call the function and check the output
- **Strict xfail markers must be removed the same pass a dependency gate is validated.** Pass 156z9ay (ARCH-V1f): after upgrading `llama-cpp-python` from 0.3.4 to 0.3.22, qwen3 round-trip tests passed under `--runxfail` but normal suite still failed as `XPASS(strict)` because the old xfail decorators remained. Rule: when closing a dependency-gated test block, run the target tests once with `--runxfail` to prove behavior, then immediately delete or relax the corresponding strict xfails and rerun without `--runxfail`. Otherwise the code is functionally fixed but CI/local runs stay red by policy.
- **Capability checks for optional acceleration must degrade to skip, not fail the suite.** Pass 156z9bb: `tests/test_gguf.py::TestGpuSupport::test_llama_cpp_gpu_offload` hard-failed when `llama-cpp-python` was installed CPU-only (`llama_supports_gpu_offload() == False`). That is environment capability, not a logic regression. Rule: tests that assert optional hardware/runtime capabilities (CUDA offload, TensorRT, AVX512, etc.) should `pytest.skip(...)` when capability is absent and reserve hard-fail for incorrect behavior under a claimed-capable environment.
- Vocab padding (GPU alignment to 64) means model output dim ≠ vocab_size — test `>= vocab_size`, not `== vocab_size`
- Rust extension tests fail silently when the wheel is stale — if a method exists in `lib.rs` but not in the installed `.pyd`, the test gets `AttributeError` with no build error. Always rebuild (`maturin build --release` + `pip install --force-reinstall --no-deps`) after adding methods to Rust source.
- **Vision/multimodal data collectors should not auto-download multi-GB image archives.** Pass 156c V-5: LLaVA-Pretrain ships ~14 GB of image bytes in a separate `images.zip` archive on the dataset card. Auto-fetching that on every `--llava-pretrain` invocation is hostile to the user's bandwidth and disk. Pattern: stream the *caption metadata* through `datasets`, take a required `--images-dir` arg pointing at a one-time user-managed extraction, verify each row's image file exists on disk, and skip-with-warning on misses (cap log noise at 5, report total at end). The collector becomes a metadata-and-validation layer; bulk binary fetch stays a separate, deliberate user step. Same shape generalizes to ShareGPT4V, COCO, audio datasets, and any future modality where row-level metadata is small but media bytes are large.
- **String-dispatch kwarg = registry pattern + paired structural-and-behavioural tests.** Pass 156j (D-9 APO): `train_dpo` gained `loss_type="dpo"|"apo_zero"`. Cleanest implementation is a static `_resolve_preference_loss(name) -> callable` registry mapping `{"dpo": _dpo_loss, "apo_zero": _apo_zero_loss}` that raises `ValueError` on miss — typos fail loud at the call site instead of silently falling back to the default. Companion rule on tests: a structural test (`assert "apo_zero" in inspect.getsource(train_dpo)`) only proves the kwarg name appears somewhere in the body — it gates against typos but NOT against `loss_type` being assigned to a local variable that's then ignored. Pair it with a behavioural dispatch test that patches both branch implementations (e.g. `_dpo_loss` and `_apo_zero_loss`) with sentinel-recording mocks, stubs out heavyweight upstream paths (e.g. `_get_sequence_logps`), runs the public method once, and asserts exactly one branch's sentinel was hit. That test catches the regression where someone reverts `loss = loss_fn(...)` back to a hardcoded `self._dpo_loss(...)` call.
- **Teacher-side steering is NOT a student-side regularizer unless it reaches the training text.** Pass 156z9ba (Personality-5 BUILD): FORGE quick-profile fields (`Personality`, `Tone`, `Expertise`, `Response style`, `Example phrases`) already shaped the teacher system prompt in `_start_distill_training`, but the selected profile never became direct student examples — only teacher generations carried it indirectly. That means the requested voice is advisory, not binding, and two runs can drift even with the same profile. Cheapest honest fix is NOT a new trainer loss first; it is a deterministic auxiliary example set on the existing SFT path (`User: ...\nAssistant: ...`) built from the same profile fields and appended only for the relevant category. Rule: when a UI/config field claims to shape a distilled skill/style/persona, grep the student-data path and confirm the field reaches the student's actual training text or labels — if it only reaches the teacher prompt, the consumer is incomplete.
- **K-quants need two compatibility checks: block payload AND logical row width.** Pass 156z9ax (ARCH-V1h2): fixing q4_k's 148-byte-vs-144-byte super-block payload was necessary but NOT sufficient. The block bytes became locally dequantizable and still llama.cpp hard-crashed because the exported tensors were 64/128 wide. ggml K-quants are row-wise formats: tensor byte size is derived from `ne[0]`, so the inner row width must be a multiple of the format block size (256 for q4_k). Our GGUF writer stores reversed dims, which means the check belongs on the LAST logical tensor dimension before write. **Rule:** for any row-wise quant format, don't stop at “does one block decode?” Verify the tensor's logical row contract too; if the row width is incompatible, gate the quantizer and fall back to a safe type instead of emitting a file whose payload is valid but whose tensor byte count disagrees with its shape.

### GUI & Code Matching
- GUI client-routing for chat should prefer `EnigmaClient.chat_stream(...)` and degrade to `EnigmaClient.chat(...)` before local-engine fallback — this keeps one API seam alive across servers that differ on SSE support while preserving existing GUI behavior.
- Queue jobs routed through API must load the job's student model on the daemon before `client.train(...)`; otherwise queued jobs silently train whichever model is currently active server-side.
- When a dispatcher mode already exists (`run_training` supports it), GUI launchers for that same mode must call the dispatcher seam (`build_dispatch_context` + `run_training`) instead of instantiating trainer classes directly — mixed routing creates sibling drift and blocks centralized validation/callback policy.
- CONFIG page has temperature/top_p/top_k/max_tokens/repetition_penalty → consumed via `config_overrides` in chat kwargs
- FORGE hardcoded temperatures (0.3/0.7/0.8/0.9) across dialogue/adaptive/distill/evolutionary — no GUI control
- DPO beta, reward model epochs, vision batch_size are hardcoded in GUI layer — no user control
- Pre-train general_mix_ratio=0 silently overridden to 0.1 — logged but not confirmed with user
- Subagent reports about missing GUI controls have ~80% false positive rate — always verify against actual widget builders
- Search ALL consumers when changing a core pattern (including RewardModel, test helpers)
- No `getattr(config, 'field', default)` — access directly so missing fields crash visibly
- When removing a widget, remove all read sites or define a fallback
- UI dispatchers must cover all UI options — new radio card needs new dispatcher case
- Section visibility must match mode count — unmapped modes leak controls
- Wrapper functions must match delegate signatures (name + type, not just count)
- Config object with N constructor sites: every field must appear in all N — use structural tests to catch disconnects
- Removing class aliases? Also fix string type annotations (`-> 'OldName':`)
- `TYPE_CHECKING` imports for string annotations need `# noqa: F401`
- Hardcoded training constants (thresholds, chunk sizes, batch caps) must scale with hardware — use a budget dataclass (TrainingMemoryBudget) not magic numbers. Grep ALL call sites (GUI TrainingConfig constructors, not just core/) when centralizing a constant
- Inference constants (context size, batch size, cache caps, chunk sizes) need the same treatment as training — VRAM tier ladders with 3-5 breakpoints waste hardware at the top and hurt at the bottom. Use continuous scaling or a budget dataclass
- **Library kwarg → GUI surface: refactor the existing entry-point, do NOT duplicate the body.** Pass 156k D-9b: APO-zero ships as a fifth alignment-mode radio card by adding `loss_type: str = "dpo"` to the existing `_start_dpo_training` (200-line method), parametrizing the human-facing label via a local `algo_label`, and forwarding the kwarg at the single `trainer.train_dpo(...)` call site. New `_start_apo_training` is a 1-line wrapper: `self._start_dpo_training(loss_type="apo_zero")`. Anti-pattern would have been to copy 200 lines, change two, and watch them drift forever as either side gets bug fixes the other misses. Cost of refactor: ~10 lines + 1 wrapper.
- **Live stream rendering needs explicit fallback controls to avoid duplicate stream retries.** Pass 156z9bw: GUI `_send_message` now attempts stream setup first (`_chat_request_stream`) and falls back to one-shot chat only when stream fails or yields no chunks. Contract rule: fallback call must force non-stream (`prefer_stream=False`) so the code does not immediately re-enter the same failing stream path and double-log/double-latency the request.
- **When stream and non-stream paths share request contracts, centralize payload building once.** Pass 156z9bx: `_chat_request_stream` and `_chat_request` both need the same kwargs filtering + system-prompt wrapping; moving this into `_build_api_chat_payload(...)` prevents silent drift where one path forwards a key (or wraps prompts) differently than the other.
- **GUI-wiring tests must gate the literal kwarg at the trainer call site.** Pass 156k: assert `loss_type=` (literal token) appears in the trainer call expression of the entry-point. Without that, a regression where someone "fixes" the GUI to assign `loss_type` to a local variable but drops it from the actual `train_dpo()` call silently reverts the new mode to default while the underlying loss math is still correct. End-to-end behavioural proof (sentinel-mock dispatch test) lives at the library layer; the GUI test only needs to gate the wiring.
- **Label-tracking after a refactor: word-boundary regex over the whole method body.** Pass 156k-audit: when a refactor introduces a derived label local (e.g. `algo_label = "DPO" if loss_type == "dpo" else "APO-ZERO"`), the trainer-call-site forward test does NOT prove user-facing strings (status bar, log prefixes, error messages, save-history label, dialog titles) actually use the parametrized label. Separate test: strip docstrings + comment-only lines, scan body with **word-boundary regex** `re.search(r'\bDPO\b', ln)`, allowlist only the legitimate ternary-definition lines. Round 1 of this test used substring search `'"DPO"' in ln` and PASSED while three hardcoded literals still leaked (`f"--- DPO TRAINING STOPPED ---"` contains `DPO` but not `"DPO"`). Round 2 with word-boundary caught all three pre-fix and zero post-fix. Generalizes: any structural test that gates on string-literal presence inside f-strings or log strings must use `\bTOKEN\b` regex, not double-quoted substring.

### Training & Numerics
- Disk-backed training: write sequences to JSONL with byte offsets, pass `data_path`/`data_offsets` to Trainer — avoids holding all sequences in RAM
- Two-pass streaming: Pass 1 scans + collects samples (capped), Pass 2 processes + writes to disk — peak RAM = one chunk + samples, not full dataset
- Multi-stage pipelines multiply peak RAM — write intermediates to disk between stages
- Sequence packing 4D masks are O(rows × T²) — build per-batch, not all-at-once; 5K rows × 4096² × 4B = 320 GB
- Training batch tensors must stay on CPU until consumed — `.to(device)` in the training loop, not at batch creation time; otherwise all windowed batches accumulate on GPU
- Guard super-linear algorithms with size thresholds — skip above N, fall back to cheaper alternative
- **Any acquired resource needs release on ALL exit paths — temp files, locks, open handles, cache slots.** Includes early abort, cancel, OOM, AND exceptions. Generalised from temp-file cleanup after F-Audit-2 (May 27 2026): the `/api/chat/stream` `/style` interception held `_inference_lock` and called `_handle_style_command` with no try/finally — a raise inside the handler would leak the lock permanently (every later request → 429 until restart). The existing sibling code wrapped `_resolve_and_activate` in `try/except: release; raise` for exactly this reason; the new code didn't follow the convention. Rule: when your diff acquires a resource and then runs ANY code before releasing it, that code is inside a try/finally (or try/except-release-raise). Test discipline: monkeypatch the intermediate call to raise, assert the resource is free afterward (`lock.acquire(blocking=False)` succeeds; temp file gone; handle closed).
- Silent process death (no traceback) = OS OOM kill — check memory-intensive ops at log stop point
- `LambdaLR.__init__` calls `step()` internally — use `(step + 1) / warmup` to avoid zero LR
- Never `encode()` for known special token IDs — use named attribute directly
- RMSNorm needs fp32 upcast in fp16/bf16 to prevent NaN
- MoE/scatter-add accumulators need fp32 — `zeros_like(x)` inherits fp16 under AMP, `index_add_` in fp16 loses precision across many experts. Upcast accumulator, `.float()` the addend, cast back to input dtype at end
- AdamW `beta2=0.95` for LM training (not PyTorch default 0.999)
- Training data format must match inference format — "User:/Assistant:" everywhere
- Data quality gates: clean → validate → deduplicate — never accept unchecked teacher output
- Two-tier truncation: message-count for disk, token-count for inference context
- Loss penalty terms must be tensors with grad — a Python float in the loss produces zero gradient and is a silent no-op
- Log-ratio before `exp()` must be clamped (±20) to prevent inf — check ALL RL algorithms, not just PPO
- Loss metric weighting must count non-padding tokens, not `numel()` — `ignore_index` makes CE mean over subset
- `torch.std()` defaults to Bessel's correction (N-1) — use `unbiased=False` for population std with small samples
- Gradient accumulation scaling must be consistent across ALL training methods — grep `max_grad_accumulation` and verify every `loss.backward()` divides first
- Streaming generator stop-string logic: after breaking on match, skip any final buffer flush — `stopped` flag pattern
- KL penalty formula must be consistent across ALL RL trainers — use `(policy_logps - ref_logps).mean()` everywhere; other formulations can go negative or flip sign
- Smoke/test data must match model scale — tokenizer retrain on tiny data shrinks vocab, forces random re-init, and 616M params on 129K tokens = NaN in minutes. Use low Memory (GB) value (e.g. `1`) for small test data so the auto-picked model is small enough
- `torch.compile` warmup can take 50-100+ steps with severely degraded throughput (10-100x slower) — don't diagnose performance until after Triton/Inductor finishes compiling kernels. Step 10 may show 588 tok/s, then step 30 drops to 90 tok/s as different code paths trigger recompilation
- Benchmark hot paths before rewriting — measured our Python BPE at ~7 MB/s, which defined the actual gap vs HF Rust ~50+ MB/s. Guessed estimates were 3-10x off in both directions
- BPE tokenizer sample cap must scale with hardware but saturates early — quality plateaus around 2 GB of text (~1.5M unique words); 8 GB creates 6.1M unique words and merge time grows super-linearly. Cap at 2 GB max. Measured: 2 GB / 2.4M unique words / 32K vocab = 20 min (post S712-S714 heap fixes + S809 GIL yield fix).
- GIL yields in CPU-bound loops must match per-iteration wall time — `time.sleep(0)` every 100 iterations is fine when each takes 1 ms, but when early iterations take 10+ seconds the GUI freezes for minutes. Yield every iteration if wall time is unpredictable.
- GUI "save checkpoint" during training must save live model weights from GPU memory, not copy the on-disk .pth file — the disk file is the untrained initial model, training state only exists in memory until the training loop saves it
- Long-running training (millions of steps) needs step-based periodic saves, not just epoch-end saves — if one epoch takes months, all progress is lost on interruption
- Verify checkpoint correctness by comparing file size + timestamp to the original — byte-identical files with matching LastWriteTime = stale copy, not a real save
- OOM kills leave no traceback and no log — write a heartbeat file to disk every 30s (pid/status/phase/step/loss/timestamp) so post-mortem analysis has a last-known-good state. Check heartbeat on next training start to detect stale sessions
- Training heartbeat statuses: `running` (in progress), `complete` (clean exit), `stopped` (user cancel), `crashed_oom` (RuntimeError OOM branch), `crashed_nan` (NaN/Inf loss), `crashed` (unhandled exception) — use status to route diagnostic messaging on next launch
- Log rotation limits of 10 delete historical runs — keep at least 100 forge logs so past sessions survive weeks of daily restarts
- NaN/Inf loss detection belongs in the on_loss callback, not the training loop core — the GUI layer sees every loss value and can log + halt before the next batch makes things worse
- GUI presets that hardcode hardware-dependent values (batch size, chunk size) should use "auto" — a preset value of "4" overrides auto-detection and wastes GPU on large cards or OOMs on small ones
- Cross-path patterns (assign + finally clear) must be verified across ALL call sites — adding `finally: x = None` without the assignment before `train()` looks complete but is a no-op. Grep for both sides of every assign/clear pair
- Functions that build per-item tensors and accumulate into a list before returning must be generators if dataset is large — 45K rows × 4096² float32 masks = 3 TB in a list. Convert to `yield` + `yield from`, wrap eager callers with `list()`. Silent OS OOM kill with no traceback.
- Double-checked locking: ALL work must complete inside the lock before setting the initialized flag — `_initialized = True` followed by work outside the lock lets a second thread see True and use unfinished state
- Non-reentrant lock + proxy object = deadlock — if a dict proxy's `get()`/`update()`/`items()` triggers initialization that acquires a lock, then `_load_user_config()` (called inside that lock) calling `CONFIG.update()` re-enters the proxy and deadlocks. Bypass with `dict.update(CONFIG, ...)` / `dict.__setitem__(CONFIG, ...)` / `dict.items(CONFIG)` inside locked init paths

### Unpredictability vs Determinism
- Unpredictable in **behavior**, deterministic in **infrastructure**. Sampling, self-initiated research, mood-weighted replay, emergent personality drift should surprise. Training seeds, test fixtures, dedup hashes, atomic saves must reproduce byte-identical runs.
- Every "unpredictable" behavior must be driven by an internal signal (confidence, emotional_state, novelty, engagement), never pure RNG. Pure randomness = noise, not personality.
- Every unpredictable feature needs a debug off-switch and a reproduction seed — otherwise crashes are unreproducible and tests flake.
- `hash()` is non-deterministic across runs (already logged); extend the rule: no `random.random()` / `torch.rand()` without a seed source tied to either the training seed or an internal signal.
- "Let it grow naturally" is not a substitute for curriculum or structure. Chaos without a signal is not emergent behavior.
- Neural reward models reinforce whatever scores highest — usually confident-sounding noise, not correctness. Prefer rule-based rewards (pass/fail, format check) for RL; reserve neural reward for preference tasks where ground truth is subjective.
- **Config field without CLI/GUI consumer = unreachable feature.** Pass 156i3 DET-2: `TrainingConfig.deterministic` field shipped with full helper logic and 4 tests, but `run.py --train` had no `--deterministic` flag — the canonical CLI workflow couldn't enable it. Author's-lens scan caught it within the same pass. Rule: when adding a `TrainingConfig` field (or any config-object field), grep `TrainingConfig(` constructor sites AND the CLI argparse block AND the GUI widget builders in the same pass. Each consumer layer that doesn't surface the field reduces "shipped" to "shipped for hand-edited code paths only." Companion rule: when a flag depends on another flag (e.g. `--deterministic` requires `--seed` or it's a silent no-op), add an early `parser.error()` so the dependency fails loud at parse time, not silently inside the helper.
- **`warn_only=True` on `torch.use_deterministic_algorithms` is mandatory when MoE is in scope.** Pass 156i3 DET-2: MoE `index_add_` has no deterministic CUDA kernel — calling `torch.use_deterministic_algorithms(True)` (no `warn_only`) hard-errors and blocks every MoE training run the moment the user opts into determinism. With `warn_only=True` the user gets a one-line UserWarning per non-deterministic op and training continues. Rule: any module that opts into PyTorch determinism must use `warn_only=True` unless every op the model uses is in the deterministic-kernel list — which is rarely true for modern architectures.
- **Continuous/background training paths need NaN/Inf abort + token-length cap from day one.** Pass 156i4 Continuous-1: `BackgroundTrainer._train_batch` had no `torch.isfinite(loss)` check — a single NaN sample steps the optimizer with NaN gradients, and from that step forward *every weight in the model is NaN*. The class is `daemon=True` and runs for months over the user's chat history; one bad input could permanently corrupt the model with no warning, no test signal, no log entry. Same applies to `_retrain_on_replay` — duplicate the guard wherever `loss.backward()` is called. Companion rule: continuous trainers must also cap per-example token length (`max_token_length: int = 4096`) — without it, a misbehaving mod or pathological input pushes a 1M-token tensor through the model and OOMs the GPU. Skip-with-DEBUG-log, not truncate (truncation silently drops context). When auditing any always-on training path, check three things: (1) finiteness guard before every `.backward()`, (2) per-step gate on `valid_count > 0` before `optimizer.step()`, (3) hard input-size cap that refuses oversize samples rather than truncating them. All three together make the path safe to run unattended for months.
- **Replay-buffer rehearsal alone does NOT prevent catastrophic forgetting — anchor sets do, partially.** Pass 156i5 Continuous-2: `BackgroundTrainer` class docstring claimed "prevents catastrophic forgetting" but `_retrain_on_replay` only rehearsed the recent chat buffer. A user spending weeks on a single topic (cooking only, say) silently loses unrelated skills (math, code, reasoning) even with replay running every 200 examples — because none of those skills appear in recent chat to be rehearsed. The honest fix is a **fixed anchor set** of curated general-capability examples loaded from disk and rehearsed *alongside* the recent slice on every replay pass. Anchors must NOT be score-sorted (curated order is the point); same NaN/finite + token-cap discipline applies uniformly to anchors and recent (mirror discipline). Even with anchors, forgetting is **bounded by anchor coverage** — a 50-example anchor set is a floor, not a guarantee. Reframe any docstring that says "prevents" to "mitigates" + state the bound explicitly. Pattern generalizes: any continuous/online learner that claims to defeat forgetting needs (a) a frozen reference dataset, (b) interleaved rehearsal, (c) honest scope language about what the rehearsal can and cannot reach. Without (a) the claim is a lie.
- **Honesty reframes must grep all sibling claims, not just the docstring.** Pass 156i6 Continuous-2a: Pass 156i5 fixed the class docstring of `BackgroundTrainer` to say "mitigates ... bounded by anchor coverage" but **two sibling claims still over-promised** — an inline section comment and the `_retrain_on_replay` method docstring both still said "prevents catastrophic forgetting." Self-audit caught both within minutes. Same anti-pattern as the seed-method drift principle, applied to doc claims: when you reframe one site of a multi-site claim, grep the whole module (and adjacent modules with shared subject) the same pass. Specifically grep the **literal old wording**, not the new one — only the unfixed sites still match.
- **Empty-A early-out must not skip always-on B.** Pass 156i6: `_retrain_on_replay` returned early on `not self.replay_buffer` *before* loading anchors, defeating the entire anchor feature during quiet periods (which is precisely when anchors matter most). When ordering early-outs in a method that combines two data sources A (situational) and B (always-on), gate on `not A and not B`, not on `not A` alone — and *load* B before the gate so it's available to the check.
- **File-present-zero-yield is a real misconfiguration and must be loud.** Pass 156i6: `_load_anchor_examples` logged INFO "loaded 0 anchor example(s)" when the configured anchor file existed but contained only malformed/empty rows. Volume table for the loud-on-real-issue rule: missing-file → WARNING, file-present-zero-yield → WARNING, file-present-N-rows → INFO. Three branches, three log levels — not two branches collapsed into one.

### GUI & Threading
- `after(0, callback)` = main-thread execution from any thread — no locks needed for tkinter-only state
- Only lock mutable state that crosses thread boundaries
- Centralize widget setup in factory functions for consistent behavior
- Lambda in threaded error handlers: `msg = str(exc)` then `lambda m=msg: f(m)` — avoid late binding
- CTkFrame defaults to 200x200px when propagation disabled — compute height from font metrics
- `CTkEntry.configure(fg_color="transparent")` crashes — use parent's actual bg color
- Tkinter `<Leave>` fires entering child widgets — check pointer vs widget bounds for true leave
- Fixed-width labels clip text — use `grid_columnconfigure(col, minsize=N)` instead
- Any I/O with timeout must run off main thread — `urlopen(timeout=10)` freezes GUI
- Debounce rapid widget rebuilds with `after(300, rebuild)` + cancel-on-re-entry
- Recursive `after()` animations must track the callback ID and cancel before restarting — stacking parallel loops freezes GUI
- `<Configure>` handlers must debounce (100ms) and skip same-value reconfigures — widget `.configure()` can trigger new `<Configure>` events causing cascade
- Per-tag cursor bindings (`tag_bind <Enter>/<Leave>`) set cursor on entire widget — use single `<Motion>` handler with `tag_names()` check
- Tooltip/popup dismissal needs watchdog timer — `<Leave>` events are unreliable for fast mouse, alt-tab, window overlap
- Dual-flag stop mechanisms (GUI `training_active` flag + Trainer `_stop_requested` flag) must BOTH be signaled — if only the GUI flag is set, the Trainer's batch-boundary check never triggers and stop waits for callback-based detection (up to one full batch delay)

### Security & Boundaries
- `core/` never imports `gui/` — wire cross-boundary features in GUI layer
- Forbidden lists must block the primitive (`__import__(`), not specific examples
- Audit config values to verify they're consumed by code, not just decorative
- Path traversal `startswith` checks need `+ os.sep` — grep for all such guards when fixing one
- Path traversal: `str.startswith()` is insufficient — use `Path.relative_to()` which raises ValueError for paths outside the allowed tree
- Sandbox forbidden lists must be complete across ALL sandbox implementations — grep all sandboxes when adding dangerous patterns
- API endpoints must never expose absolute filesystem paths — use `p.relative_to(BASE_DIR)` and resolve relative paths back in the consumer (load endpoint)

### Feature Gating
- Default to disabled when model capability is uncertain — infrastructure ready, UX clean
- Quality gate AI content before surfacing — heuristic scorer with threshold
- Idle detection reuses `after(N, callback)` with monotonic timer + activity reset
- Infrastructure without consumers is dead code — if the import target doesn't exist (e.g. `tools/tool_executor.py`), the feature can't activate regardless of how well the loop is coded. Build the dependency chain bottom-up, not top-down
- Staged tool-use rollout — when the strong version of a tool-use feature needs logits/training-loop changes (e.g. inline `<search>` token), ship a pure-string post-generation gate first. It is fully testable without a trained model, signal-driven, and unblocks the dependent feature while the harder version is being designed. Pass 153 AutoResearch-2 Stage A: `score_uncertainty()` + `should_retry_with_research()` shipped on text-only signals (hedge phrases, refusal patterns, length anomaly, question echo) — Stage B inline-token work remains future. Pattern: dataclass `Result(score, reasons)` makes the gate auditable in logs; off-switch via `enabled=False` keyword arg; deterministic — no RNG.
- **Module-level path constants make boot-default behaviour testable.** Pass 156i7 Continuous-3: `_DEFAULT_ANCHOR_PATH = Path(__file__).resolve().parent.parent / "data" / "anchor_examples.jsonl"` lives at module scope in `router.py`. Tests `monkeypatch.setattr(router_mod, "_DEFAULT_ANCHOR_PATH", tmp_path / "anchors.jsonl")` to simulate file-present and file-missing branches without touching the real `data/` directory. Inlining the path inside the constructor body (`def __init__: ... if (Path(__file__).parent.parent / "data" / "anchor_examples.jsonl").exists(): ...`) makes that test impossible — you'd have to either delete the real file or pollute it. **Companion rule: wire feature defaults at the boot site, not the library default.** The library `BackgroundTrainer(anchor_data_path=None)` keeps the feature off by default for direct callers (test isolation, explicit opt-in); the production boot path (`ModRouter._create_trainer`) auto-resolves the repo default via `_DEFAULT_ANCHOR_PATH if .exists() else None`. Same shape applies to any "auto-discover repo asset on boot" feature: name the path at module scope, gate on `.exists()` inside the boot-site constructor (not the library), monkeypatch the constant in tests.
- **Prefer dual-emit on the producer over multi-format support on the consumer.** Pass 156i8 D-11: collector wrote JSONL pairs to `data/finetune/`; existing SFT trainer reads plain text via `Path.read_text`. Two ways to bridge: (a) teach the trainer to detect-and-parse JSONL (touches GUI file picker + Trainer + format detection — three change sites, regression risk on the existing text path), or (b) teach the collector to *also* emit the canonical text format the trainer already consumes (one new helper, zero training-side touches). Picked (b) — `_write_combined_text()` runs alongside `_write_jsonl` inside `combine_all`, emitting `combined_finetune.txt` in `User: <p>\n\nAssistant: <c>` format with blank-line separators. **Precondition: the consumer's format must already be the canonical one in the codebase**, otherwise dual-emit just adds a second source of truth for which-format-wins. Here `User:/Assistant:` is used everywhere (router.py, GUI chat builder, advanced FORGE pipeline) so there was no ambiguity about which side moves.

### Research
- Audit existing implementations before searching for research — know what you have first
- Upgrades (better algorithm) vs gaps (missing feature) — different risk profiles
- Author's-lens during research passes catches code bugs that unit tests miss — Pass 140 research on MTP paper (arxiv:2404.19737) surfaced TWO silent code issues while simply reading the adjacent code: (1) an inline comment at [model_presets.py L129](enigma_engine/core/model_presets.py#L129) had the paper's conclusion inverted ("biggest gain at small model sizes" when the paper says "increasingly useful for larger model sizes"), and (2) `predict_heads` at [model.py L237-240](enigma_engine/core/model.py#L237-L240) were NOT weight-tied to `tok_embeddings`, silently inflating the model by ~98M params (13% of 742M) for an auxiliary signal of ambiguous sub-1B benefit. Lesson: when researching a feature, read the code that implements it, not just the paper — the gap is often between them.
- Cross-reference subagent findings against actual source — false positives are common
- Enumerate subsystems systematically, don't brainstorm from memory
- Subagent false positive rate is ~80% on HIGH-confidence claims — always manually verify before fixing
- Architecture competitiveness ≠ model quality — data and compute are the real bottleneck, not code
- "Cloud AI feature X" is usually just RAG with better UX — check if you already have the core before building
- Target hardware is RTX 5090 (32 GB total, 16 GB VRAM budget for AI) — code should scale up or down, don't hardcode GPU assumptions
- Language selection for rewrites: check what production systems actually use — both OpenAI (tiktoken) and HuggingFace (tokenizers) independently chose Rust+PyO3 for the same problem. Two independent choices beat any theoretical analysis
- Profile the real bottleneck before picking a rewrite strategy — regex pre-tokenization was 80%+ of encode time, not BPE merges. Rewriting the wrong hot path wastes the effort
- Drop-in replacement behind protocol interfaces: only modify the factory function (`get_tokenizer()`), zero changes to 40+ consumer call sites. Protocol boundary = rewrite boundary
- Mod files and core files can have the same name but completely different purposes — `mods/vision/vision.py` (screen OCR) vs `core/vision_encoder.py` (ViT for LLM multimodal). Read both before assuming they're connected
- A router named "ModRouter" does not route between models — verify what it actually routes before designing a model dispatch layer on top of it
- Data-mixing research is blog-sourced, not paper-sourced — SmolLM3 blog has concrete stage percentages (12% code Stage 1, 24% Stage 3) while Qwen3 and DeepSeek-LLM papers say "increasing" without numbers. Prefer blog posts + HF dataset cards over arxiv for reproducible mixing ratios
- Dedup threshold defaults in our code (0.8) can drift from industry standard (FineWeb/DCLM use 0.75) — always re-verify numeric defaults against the paper actually cited, don't trust that an older default matches current practice. Same applies to LSH band/row splits (112 hashes / 14 bands / 8 rows is the reference split)
- Primary source for a dataset is the dataset's own card/paper, not a downstream consumer's blog — Pass 137 audit caught `<think>` tag format being inferred from SmolLM3's description of how *they* used OpenThoughts3, when OpenThoughts3's own card only says "reasoning traces" without committing to a wrapper format. Verify tag/format claims against the producing dataset's docs; if unclear, sample the actual data before relying on the format

### Rust / PyO3
- Rust `regex` crate doesn't support lookbehind — use manual byte-level boundary check (`is_non_alpha_boundary()`) instead
- MSVC toolchain (`stable-x86_64-pc-windows-msvc`) needs VS Build Tools; GNU toolchain (`stable-x86_64-pc-windows-gnu`) bundles MinGW linker — works out of the box
- `maturin develop` needs `VIRTUAL_ENV` set; `maturin build --release` + `pip install <wheel>` works everywhere
- Symbol interning (string→u32 ID) eliminates String allocation in merge loops — measured 60% speedup over String-based version
- Skip-array (`next[]`/`prev[]`) for O(1) linked-list removal in BPE merge — avoids Vec shifting
- Pack two u32 IDs into one u64 for HashMap pair key — zero-alloc pair lookup
- Direct-ID cache (`HashMap<String, Vec<i64>>`) skips symbol→token_id mapping on cache hit — 4x throughput on repeated text
- Cache eviction: 40K cap with quarter-eviction (drop 10K entries randomly when full) — simple and effective
- BPE merge ranks in Rust: use `merge_table: HashMap<u64, MergeEntry>` not sorted Vec — O(1) lookup per pair
- Each new terminal session needs `$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"` for cargo/rustc (Windows)
- Build command: `cd rust_extensions; maturin build --release` then `pip install target/wheels/*.whl --force-reinstall --no-deps`
- Auto-fallback pattern: `_rust_available` class var (None/True/False), try import once, set permanently — no repeated import failures

### Auditing
- **Dead-end ≠ trash. Read the code before recommending disposition.** Reachability tells you whether a feature works *today*; it does NOT tell you whether the code is well-written, what's salvageable, or whether the right answer is kill/salvage/rebuild. When the user asks "is the code any good?" on dead-end work, the honest answer requires actually reading the file end-to-end (or doing a representative deep sample if it's >2000 LOC) — surface read for code quality, architecture, type discipline, naming, error paths; logic read for what the file *claims* to do vs what it actually does; gap read for what's missing that would make it work. Then disposition recommendation has three independent axes: (1) **code quality score** (would I write this way today?), (2) **completeness score** (does the code deliver the docstring's promise?), (3) **uniqueness score** (does this contain knowledge or data that's costly to re-derive?). Kill-by-default is fine when all three are low; salvage-the-data-only is the move when quality + completeness are low but uniqueness is high (e.g. anatomically-correct bone-limit tables, a curated regex set, a hand-tuned reward shape); rebuild-simpler wins when completeness is low but the *design intent* is still wanted. Anti-pattern: judging disposition from reachability alone ("no callers → kill") without ever reading what the file actually contains — that's the same shape as judging a paper by its abstract. Audit case (Pass 156-avatar): brick + package together = 2167 LOC, both unreachable through launcher, but the bone-limits table (~120 LOC of researched anatomical data) is real domain knowledge worth preserving even if both wrappers around it get deleted.
- Audit by risk, not by size — training pipeline, GPU ops, and resource-intensive paths first; utility modules last
- `torch.compile` without Triton triggers Inductor C++ fallback that eats tens of GB RAM — always gate on Triton availability
- GUI hardcoded config values bypass safe defaults — audit GUI builders that construct config objects, not just the config defaults
- Use subagent for bulk reconnaissance (list all GUI controls, map all config fields), then manually verify the top findings — faster than reading every file top-to-bottom
- Wiring audits: trace GUI control → config field → core consumer. Dead ends at any step = bug. Use grep for the field name across all three layers.
- Triage audit findings before fixing — verify each claim against source. "Port mismatch 8080 vs 8000" and "docstring says cosine but code uses dot product" were both false positives that would have wasted edits
- Faster precision: check one data flow end-to-end (button press → what happens) rather than reading an entire file sequentially
- GUI crash checks should verify both terminal liveness and OS process list (`run.py --gui`) — one signal alone can be misleading when multiple sessions exist.
- Architecture review checklist: (1) every implemented module has at least one real caller path, (2) core features are available to ALL consumers (GUI/API/background trainer), (3) rich state that is computed — verify it's actually consumed downstream and not just displayed. "Infrastructure without consumers is dead code."
- Design-mode audit pattern: check features end-to-end (init → update → apply → use) rather than file-by-file. A module that is clean internally but has no callers is still broken. Example: `core/model_merging.py` fully correct code, zero GUI callers (closed Pass 126); `core/rag.py` was only wired in GUI layer, API server never got RAG (closed Pass 125).
- **Design intent must be verified before calling a pattern "wrong direction."** AI-computed state (sentiment-derived emotional_state, engagement score, confidence estimate) is categorically different from user-set config (profile personality dict, system prompt field). Injecting AI-computed state into the prompt is the AI knowing itself — fine. Injecting user-set config into the prompt is the user configuring the AI's character — often a design violation. Confirm which category a field falls into (check for user-facing widgets, profile files, settings pages) BEFORE recommending inject-or-don't-inject.
- **Audit-on-audit is not free — over-retracting is as costly as under-auditing.** When reversing a prior finding, verify the reversal against code one more time. Pass 7 said "inject emotional_state into prompt" (wrong — N-22 already does). Pass 8 said "retract, injection is wrong direction" (wrong — emotional_state is AI-computed, injection is fine). Pass 9 corrected: emotional_state injection OK, ai_profile.personality injection not OK. Three passes of ping-pong before landing on correct. Lesson: check the actual field's origin (user-set vs AI-computed) in one pass, not three.
- **Markdown link audits must verify targets exist on disk** — `grep` alone confirms syntax but not correctness. For every generated link, spot-check the path with `file_search`. Relative-looking paths inside narrative text (e.g. `core/bones.py` in a sentence about the avatar mod) may resolve to a subsystem's local `core/`, not the top-level `enigma_engine/core/` — read the surrounding prose to decide scope. Separately, watch for literal `\n` artifacts introduced by earlier edits where a real newline was needed (breaks markdown table rendering silently).
- **Planning-doc audits need the same tool-verification rigor as code audits.** Pass 148 GUI plan round 1 invented a service-contract location (`enigma_engine/api/`) from memory — round 2 `list_dir` showed api/ was already FastAPI territory, forcing a relocation to `enigma_engine/services/` `[DELETED dbc19ea, May 25 2026]`. Lesson: for every path, framework, or URL named in a plan, run the matching tool (`list_dir`, `grep_search`, `fetch_webpage`) before committing the plan as execution-ready. "I think that folder is empty" is not evidence.
- **"Beat baseline" targets require measuring the baseline first.** A gate or metric written as "≤ baseline" is vacuous if the baseline is never captured. Every rubric metric and hard gate must have a Phase 0 baseline-measurement step on the incumbent system, not just on candidate POCs. Otherwise the comparison at decision time is "POC number vs guess."
- **Self-audit immediately after shipping is mandatory, not optional.** Re-read your own ship under the author's lens (§1 #19) within minutes of landing it; compare each new branch against the *reference pattern* it claims to mirror — divergences from reference are bugs even when the new code passes its tests in isolation. Pass 156d2 caught three real bugs in code shipped 10 minutes earlier (V-7 abort-summary skipped on NaN return paths, V-4 OOM heuristic narrower than reference, V-4 missing RuntimeError/Exception split) — all found by self-audit before user reported anything. Pattern: ship → self-audit same session → fix + add audit-test that would have caught the regression.
- **Test-suite baseline must be diffed against HEAD on session start, not blindly accepted as "pre-existing."** Pass 156z9aj audit on three earlier passes (156z9ag/ah/ai): all three reported "30 pre-existing failures in test_training.py" without checking whether those tests targeted features whose code had been silently deleted from the working tree. `git diff --stat HEAD -- enigma_engine/core/training.py` showed -480 lines vs HEAD; the deleted block contained `_effective_warmup` (Sched-2 close-stamp), `_apo_zero_loss` + `_resolve_preference_loss` (D-9 close-stamp), `set_training_seed(deterministic=...)` + `TrainingConfig.deterministic` (DET-2 close-stamp) — four claimed-shipped features that 100% of the red baseline was probing. Rule: when starting a session whose suite has a non-zero red baseline, run `git diff --stat HEAD -- <suite-targets>` BEFORE quoting the failure count as "pre-existing." A single net-deletion file in a module with adjacent claimed-shipped features in SUGGESTIONS is a doc-vs-code lie that needs either restoration (`git checkout HEAD -- <file>` after user confirmation since it clobbers working-tree edits) or the close-stamps reopened to "regressed, needs re-implementation." Carrying a fake "pre-existing" label forward through multiple passes accumulates dishonesty and hides real workspace damage.
- **Docs baseline claims must come from a fresh full-suite run in the same pass, not from historical pass text.** Pass 156z9bi: before syncing tracker docs, rerun `ruff check enigma_engine/ tests/` and `python -m pytest tests/ -q`, then stamp those numbers in the snapshot. Old per-pass counts remain historical facts; only the top snapshot should advertise "current baseline." This avoids stale-count drift when later passes change test totals.
- **Mojibake is usually family-wide, not file-local.** Pass 156z9bv: audit first surfaced `�` artifacts in `gui_forge_new_modes.py`, but a sibling sweep across `enigma_engine/**/*.py` found the same corruption in core/training comments and user-facing log strings. Rule: when one replacement character appears, grep the whole package and clear all occurrences in the same pass, then re-run lint + full suite before stamping docs.
- **Self-reporting scope honesty: re-grep parked-item scope claims on every pass, do not copy them forward.** Pass 156z9cv: four consecutive stamps (156z9cr/cs/ct/cu) all logged "mojibake at inference.py L1167-1168" as a small bounded site. The May 11 audit re-grepped and found **2341** marker chars in inference.py + 7 in rl_training.py — three orders of magnitude beyond the parked text. The first under-report compounded across stamps because each successive pass treated the prior parked text as ground truth instead of running a fresh grep. Rule: when shipping a stamp, the "Parked / follow-up" section must be re-verified (one grep, one path check, one count) per item, not copied from the prior stamp. This is the same anti-pattern as Pass 156s "doc claims more than code delivers" — except the "doc" is our own SUGGESTIONS.md. The drift compounds silently because every future pass tells itself "if the scope was bigger, the prior pass would have logged it" — but no one did. Companion regression: when fixing such a scope-drifted item, install a permanent gate (here: `tests/test_repo_hygiene.py` walking the package for the marker triad) so re-introduction is impossible regardless of whether the next stamp grep is thorough.
- **PowerShell inline `python -c "..."` with non-ASCII strings breaks the parser silently and traps the shell.** Pass 156z9cv: a one-shot `python -c "..."` heredoc containing single-quoted strings with `€`, `”`, `”`, etc. caused PowerShell 5.1 to mis-parse the closing quote, leaving the shell stuck in a multi-line string continuation state (`>>` prompt forever). All subsequent commands typed into that terminal got captured into the broken string buffer and never executed Python at all — invisible failure mode. Rule on Windows: never use `python -c "<non-ASCII source>"`. Always write to a script file (`python _tmp_script.py`) and delete after. Single-line ASCII-only `python -c` is fine. If you must use non-ASCII in one shot, encode it (`python -c "import codecs; print(codecs.decode('...', 'unicode_escape'))"`).
- **Return-to-work quick-start blocks must track the current top priority, not just the latest green baseline.** Pass 156z9bm: docs were green but quick-start still pointed to older P5 runtime tasks after ARCH-1.5c launcher migrations landed. When priorities shift, update quick-start instructions in the same docs pass so the next session starts on the real gate (decision + first execution step), not stale follow-ups.
- **Fresh-daemon verification is mandatory after API boundary edits.** This pass changed `/api/chat` + `/api/chat/stream` image contract behavior, and an older daemon on a different port continued serving stale behavior until a new server process was launched. Rule: for runtime validation of API/schema changes, start a fresh server on a known free port and re-run probes there before concluding success/failure. Source reads + tests can be correct while a stale process makes live checks look wrong.
- **Token-ID bounds must be validated before tensors hit CUDA.** A tokenizer/model vocab mismatch can load cleanly and then hard-crash generation with device-side index asserts on first embedding lookup. Guard at two layers: load-time compatibility check (`tokenizer.vocab_size` vs model embedding rows) to reject unsafe checkpoints early, and runtime `_encode_prompt` range check to fail loud with actionable ValueError before device execution.
- **Audit reports must label findings pre-existing vs regression-from-this-session.** A list of "problems" without provenance reads like the agent caused them. Before stamping an audit, run `git diff --stat HEAD` (or scope to the session's touched files), and tag each finding: PRE-EXISTING (in code before this session, e.g. tracker carry-over), REGRESSED (worked before, broken now — own it), or NEW-FROM-THIS-SESSION (introduced by current work, fix immediately). Same anti-pattern family as §4 *"Test-suite baseline must be diffed against HEAD on session start"* — applied to audit findings instead of test failures. Without provenance, the user can't tell which items are the agent's fault vs the backlog.
- **State "did I research this?" honestly before recommending a direction.** When picking a direction between two non-trivial options (LOGIC first vs LANGUAGE first, library A vs library B, architecture X vs Y), declare up front whether the recommendation comes from (a) the project's own tracker / code evidence, (b) prior Learned Principles, or (c) an external sanity check this session. "Best practice says…" without a citation, when no fetch_webpage call was made, is invented authority. Confidence floor: tracker + code evidence is defensible for local-only / black-box projects where outside articles often don't apply; external check is mandatory when the question is about general-domain choices (sampling defaults, RL algorithm picks, GPU optimization patterns). If a research step is skipped, say "I did not search externally" in the same paragraph as the recommendation so the user can ask for the search explicitly.
- **Honest time-boxing for a solo builder means sequential, not parallel.** Plans that say "two tracks in parallel, 5 days each" silently assume a team. For one person the honest number is 10 working days plus setup. Write the real schedule; don't hide cost behind the word "parallel."
- **Disk truth > stamp truth (REALIGN-1.2-CORRECTION May 25, 2026).** When auditing prior close-stamps, run the EXACT commands the stamp claims to have run (`Get-ChildItem`, grep for deleted-class names, `Test-Path` on claimed-deleted dirs) and compare against the quoted output in the
 stamp. If the stamp quotes a tool result, that quote must reproduce in the current disk state OR the stamp is provably lying. Pass-bound case: REALIGN-1.2 close-stamp claimed `Directories deleted (4): mods/videogen/, mods/threed/, mods/audiogen/, mods/codegen/` and quoted `Get-ChildItem mods -Directory post-delete → 7 (was 11)`. Audit one session later: all 4 directories still on disk with 491–656 LOC each. Same anti-pattern family as Pass 156z9aj "test-suite baseline must be diffed against HEAD on session start" — extended to "directory state must be re-listed before trusting a delete claim." Compounded by VS Code Local History's silent restore capability: deletes that target tracked files survive (git wins), deletes that target uncommitted edits inside tracked files can be silently undone by a "Discard changes" elsewhere. Rule: every close-stamp that claims a filesystem mutation (delete, move, rename, create) must quote a post-mutation verification command WITH its output, AND the next session that reads the stamp must re-run that exact command to confirm — don't carry the prior quote forward as fact. Also covers the inverse: stamps that quote a baseline like `3252 pass / 3 skip` are vulnerable to the same drift; rerun the suite before quoting the count.
- **Tracker sprawl is a cousin of dead infra (May 25, 2026 session audit).** When an audit produces a "checklist" file (`AUDIT_CHECKLIST.md`, `CLEANUP_TRACKER.md`, ad-hoc per-pass scratchpads), declare its lifecycle upfront: (a) **one-shot scratchpad** — fold actionable items into SUGGESTIONS.md and DELETE the file on close (git history preserves it for archeology), (b) **permanent rolling tracker** — name and locate it deliberately (e.g. `CLEANUP_TRACKER.md` with per-file acceptance gate is the right shape), (c) **archive** — move to `history/` when work completes. Two tracker files carrying overlapping slice lists = drift waiting to happen. Pass-bound case: this session's `AUDIT_CHECKLIST.md` (582 LOC) and SUGGESTIONS.md REALIGN-1.2-CORRECTION both held the same 6 E section-2.1 slices. Pick one canonical location and have the other cross-reference, never duplicate. Detection: at session close, grep the workspace for tracker files (`*TRACKER*.md`, `*CHECKLIST*.md`, `AUDIT_*.md`) and confirm each one has a clear lifecycle label OR fold + delete.
- **Kitchen-sink commits defeat bisect (May 25, 2026 session audit).** When a session lands multiple unrelated changes (e.g. cloud-purge + core bug fixes + new feature + new tracker doc + 9 test files), each unrelated group is its own commit. Pass-bound case: `91d3d75` "chore: audit sweep - core fixes, mods cloud-purge, doc sync" mixed REALIGN-1.1 mods cloud-purge, 14 core small diffs from passes 156z9el/em/en/eo/er/eh/ff/fe/fh, the `engine_generation.py +198` cancel-signal feature slice, 9 test additions, and 582-line + 1300-line tracker docs — 35 files in one commit. Rule of thumb: if the commit subject needs three nouns joined by commas or dashes ("X, Y, doc sync"), split it. Bisect on a regression in any of those groups lands on a 35-file haystack instead of a 3-file one. Acceptable kitchen-sink: pure auto-formatting passes (`ruff --fix`), pure import-sort, mechanical type-alias renames. Everything else gets its own commit.
- **Historical doc references to deleted infra need a marker, not silence (May 25, 2026 session audit).** When you kill a module/file/class, grep its name across docs in the same pass. Either remove the mention or annotate with the kill commit hash and date — e.g. `enigma_engine/services/ [DELETED dbc19ea, May 25 2026]`. "Historical record" without a delete marker is future-audit-bait: a future grep for `enigma_engine/services/` returns 12+ hits with no signal which mentions describe live code and which describe corpses, and the next disk-truth audit has to re-derive the kill date by reading git log on every mention. Same anti-pattern family as Pass 156s "doc claims more than code delivers" pointed in reverse — doc *implies existence* after kill by failing to annotate. Pass-bound case: after `dbc19ea` killed `enigma_engine/services/`, SUGGESTIONS.md lines 1235/1246/1900/1907/4353/4426/4431/4436/4460/4472-4473/4515 and `AA code maker.md` L388 still mentioned the path with no marker. Detection: post-kill, grep the killed name across `**/*.md` and either delete the line or add the `[DELETED hash, date]` annotation in the same commit as the kill.

### Process
- New features: implement first, audit second — trace GUI → config → core → training → checkpoint
- Bug fixes: triage before implementing — verify each issue is real first
- Batch related fixes, test each independently

### Concurrency
- `threading.Lock` is non-reentrant — split into public (locks) and private `_unlocked()` methods
- Never `await` inside a `threading.Lock` — use `put_nowait()` or `asyncio.Lock`
- If a loop reads flags under lock, writers must also use the lock
- Lock scope: read under lock, write to disk outside — avoid holding lock during I/O
- `subprocess.PIPE` never drained will hang the child — close pipe, use `DEVNULL`, or drain thread
- **Idle-time schedulers in long-running daemons: reset the cooldown timer inside the work function on EVERY exit path.** Pass 156w + 156x2 audit on `BackgroundTrainer._retrain_on_replay`. Two failure modes, one rule:
  - **Cross-trigger drift (156w):** the work function is called from both an idle scheduler AND a per-batch path. If only the trigger updates the timestamp, a regular replay just before a quiet window leaves the idle gate stale → fires immediately on next wakeup → back-to-back replay burns GPU. Update the timestamp inside the *work function*, not the trigger.
  - **Failure-path drift (156x2):** if the timestamp resets only on the try-block success path, an empty-batch early-return or exception skips the reset → idle gate fires True every loop tick → 1 Hz log-spam. Reset at function ENTRY (after the obvious can't-do-anything guard like model-None) so success / empty / exception all honor the configured interval.
  - When you write a comment justifying "don't reset on failure so we retry," stop and ask "will the retry fix anything, or just hammer the same broken state?" If the latter, reset and let the interval throttle the retry rate.
  - Defensively normalise the interval kwarg (`<= 0 → None`) so a config typo can't peg the GPU; init the timestamp in `__init__` so a fresh trainer never fires immediately on first wakeup.
- **Best-of-N must be deterministic in tie-break, defensive in scorer-failure.** Pass 156x N-16: `EnigmaEngine.generate_best_of_n` runs N independent generations, scores each with caller-supplied `reward_fn`, returns highest. Three rules from author's-lens review before shipping: (1) **Tie-break = first occurrence.** Python's `max(scored, key=...)` returns the first tied element — deterministic. Adversarial test (3 candidates all score 0.5, expect first) catches the regression where someone uses `min` or reversed iteration. (2) **Scorer error = -inf, log, continue.** A `reward_fn` that raises on one candidate must NOT kill the batch — a flaky scorer would take down every best-of-N call. Swallow the exception, log WARNING, assign `-inf` so the broken candidate cannot win, continue scoring the rest. The batch still produces a usable answer. (3) **Logic-eye gate on `temperature <= 0` + `n > 1`.** Deterministic sampling produces N identical candidates — wasted compute. Don't error (user may be probing the scorer); WARNING + proceed.
- **Reward-function wrappers: bind extras with `functools.partial`, don't sniff signatures.** Pass 156x N-16 contract rule: reward functions in `reward_functions.py` have varying signatures (`format_reward(response)`, `math_reward(prompt, response, *, ground_truth)`); caller binds extras via `functools.partial` so the wrapper signature stays uniform `(prompt, response) -> float`. Don't try to detect-and-call multiple signatures inside the wrapper — that's brittle, hides the contract, and silently accepts the wrong reward function when signatures collide. Generalizes to any plugin/callback API where users supply functions with different shapes: define one canonical signature at the boundary, push adaptation out to the user via `partial` or explicit adapters.

### Gotchas — Python Language
- Assignment inside an inner function makes Python treat the variable as local throughout — use a new name (e.g. `do_x = x`) to read from outer scope and override locally
- Generator functions are lazy — body doesn't execute until iterated
- `hash()` is non-deterministic across runs — use `hashlib.sha256` for reproducible algorithms
- Lock acquired before generator + released in finally = leak risk if wrapper fails before iteration
- Methods using Path APIs (`.parent`, `.stem`) crash on `str` — enforce `Path` at call site

### Gotchas — KV Cache & Inference
- KV cache `start_pos` must account for tokens NOT yet processed — a token appended to `generated` but never fed through the model is invisible to the cache
- KV cache: return views not `.clone()` if callers only read; use `torch.roll()` for overlapping shifts
- Speculative/lookahead verify: cache refill + separate last-token call = duplicate KV entry — use one model call that both populates cache and returns logits
- No-cache full-sequence verify must slice logits to the draft region — `verify_logits[0, j]` on unsliced logits indexes from sequence start, not from draft
- DPO prompt mask must match actual encoded prefix length, not bare `encode(prompt)`
- `rewind_cache(pos)` is O(draft_len), `clear_cache()` + re-prefill is O(seq_len) — always prefer rewind after rejected drafts in speculative/medusa/lookahead
- `clear_cache()` destroys the cache object (`_kv_cache = None`); `rewind_cache()` must keep it alive — different semantics, don't conflate
- Subclass `rewind_to()` must chain `super().rewind_to()` first, then handle subclass-specific state (attn scores, INT4 buffers, logical_pos)

### Gotchas — Data Collection
- Gated HuggingFace datasets (The Stack v2, some Llama variants, some Common Crawl mirrors) fail with 401/403 or the word "gated"/"access" in the error — detect on the FIRST language/config attempt and break all subsequent loops immediately. Retrying each language separately after auth failure wastes minutes per language and produces the same error N times. Print the exact recovery command (`huggingface-cli login` + accept license URL) and return.
- HuggingFace datasets with multiple configs (FineMath `finemath-4plus` + `infiwebmath-3plus`) should be blended via separate streaming passes that share the same output dir and progress-key prefix — do NOT try to interleave them in one loop, the API iterators can't be multiplexed.
- Data-collection fetcher naming: reuse the `_fetch_hf_streaming()` helper for single-config datasets; write a custom function only when per-item filtering (license check, language detection) or multi-config blending is needed. Custom functions must still respect the same progress-dict keys, output-dir layout, and Ctrl+C / resume semantics as the helper.
- Wikipedia: use `generator=random` not `allpages`; rate-limits after ~40 requests; use `maxlag=5`
- gutendex.com is unreliable — use curated Gutenberg book ID list with direct URLs
- Non-Latin text: count Latin vs non-Latin chars, use safe default if non-Latin dominates
- File size limits in dataset loaders are silent rejection — check MAX_FILE_SIZE before loading large files
- `combine_all_sources()` uses tmp+rename for atomic writes — follow this pattern for any safety-critical merges
- Resume heuristics: use Content-Length or HEAD request to verify download completeness — file size alone is unreliable
- Streaming data: use two-pass approach — Pass 1 scans metadata + samples, Pass 2 re-iterates for processing; avoids holding full dataset in RAM
- Dedup hash sets: use `.digest()[:N]` (raw bytes) not `.hexdigest()[:N]` (hex strings) — saves ~37% memory per entry at scale
- Unbounded dedup sets need a capacity cap with graceful degradation — 50M entries × 41 bytes ≈ 2 GB is a sane limit
- Paginated API continuation tokens must be deferred until after the dependent call succeeds — advancing before the call loses the batch on failure

### Gotchas — Verification & Correctness
- Verify formulas against docstring examples — trace one concrete case through the math
- Guard patterns must be consistent across ALL sites — grep for pattern, fix all occurrences
- Config validation must cover all numeric fields (denominators, ranges, sign constraints)
- Return sentinels that mean the right thing — `float("inf")` not `0.0` for "couldn't measure"
- Numeric status fields must not use `value = status.get("field") or default` when `0`/`0.0` is valid — this silently rewrites real zero into fallback. Use explicit `is None` checks for optional metrics (e.g., `best_loss`).
- Verify variable names match local scope — `getattr(config_obj)` vs `dict.get()`
- Verify features end-to-end lifecycle: init → update → apply → use → restore
- Callbacks defined but never wired are invisible failures — grep all constructors
- Grep for callers of validation functions — defined but never called = false confidence
- Config fields defined but never consumed = dead code — grep the training loop for every TrainingConfig field
- When a function splits into two paths (if/else), verify ALL variables the shared tail uses are set in BOTH paths
- Config converters must set ALL architecture flags explicitly — ForgeConfig defaults (RoPE, RMSNorm, SwiGLU) are wrong for GPT-2 family
- Dead imports from non-existent modules crash at runtime — always `try/except ImportError` with fallback for optional cross-module imports
- **Auto-restore parse blocks must be inside the same except guard as the apply call, or have their own.** Pass 156u-A2: `_restore_lora_adapter_for_base` had `try: engine.apply_adapter_stack(entries) except (FileNotFoundError, ImportError, RuntimeError, ValueError)` — but the entries list was built ABOVE the try via `item.get(...)` and `float(weight)`. A corrupted `route_assignments.json` (non-dict entry → `AttributeError`, non-numeric weight → `ValueError` from `float()`) raised BEFORE the try, propagating up through `_on_model_loaded` and aborting the whole model load. Auto-restore's entire job is surviving its own previous writes (partial writes after OS crash, hand-edits, format drift between versions) — if the parse layer can crash, the feature is worse than not having it. Rule: every saved-state restore that touches user-data shape must (1) explicitly validate the shape with named errors (`isinstance(item, dict)`, `try: float(w) except (TypeError, ValueError)`), (2) on failure drop the whole orphan key + persist the cleanup so the next reload doesn't repeat the crash, (3) surface a loud chat-system / log message so the user knows their saved state was reset. Test the corruption cases adversarially with at least one non-shape-conforming entry AND one non-coercible value type — passing tests on well-formed data prove nothing about the corruption-resilience claim.

### Gotchas — Code Hygiene
- **Substring/startswith filter patterns must be tested against semantic NEGATIONS, not just positive matches.** Pass 156z9eo: `_REFUSAL_OPENERS` listed `"i can't help"` and matched `head.startswith(opener)`. Caught real refusals (`"I can't help you with that"`) AND the English idiom `"I can't help but [feel/smile/notice/...]"` which means the OPPOSITE of refusal. The idiom is common in personality-bearing language; teacher responses using it were silently dropped from the personality SFT pool while the filter's reject-count attributed them to "refusal." Test discipline: every filter built on broad token patterns (refusal openers, profanity lists, sentiment markers, intent classifiers) needs at least one ADVERSARIAL test that constructs a semantic-negation of the pattern (idiomatic usage, double-negation, sarcasm marker, polarity-flipper) and asserts it passes the filter. Without the adversarial test, the filter's false-positive rate is invisible — happy-path tests on prototypical refusals will be 100% green while real data slips silently into the wrong bucket. Fix: narrow the pattern to forms that include the disambiguating context (`"i can't help you"` + `"i can't help with"`), or add a negative-lookahead branch (`opener matches AND not followed by " but "`). Generalises beyond English idioms: any pattern-based classifier in a multilingual / sarcastic / idiomatic domain needs adversarial coverage of the polarity-reversing forms before its production reject counts can be trusted.
- Remove dead code from abandoned approaches immediately
- Cap unbounded string accumulation — rebuild within char budget, drop oldest
- Error handlers must include `traceback.format_exc()`, not just `str(exc)`
- Run lint after batch fixes, not just tests — catches missing imports and wrong-scope variables
- **Lint scope is `enigma_engine/ tests/` not just `enigma_engine/`** — test files accumulate unused imports and F841 dead assignments silently; `ruff check tests/` catches them. Use `--fix` for F401, then `--unsafe-fixes --fix` for F841 (keeps function call, drops variable) and E731 (lambda → def).
- `torch = pytest.importorskip("torch")` pattern: when `torch` is unused but the skip guard is needed, drop to `pytest.importorskip("torch")` (no assignment). Ruff F841 unsafe-fix does this automatically.
- Structural tests catch mixin namespace collisions across 7+ mixins
- Naive algorithms hide behind small test data — check if incremental approach exists
- CPU-bound bulk ops hold GIL — chunk work + `time.sleep(0)` between chunks; sample head+mid+tail for validation
- Test coverage audits: grep callers across ALL test files, not just the "expected" one — tests for module X often live in test_core.py or test_e2e.py
- Test file merges: always `--collect-only` both source AND target to get exact counts, then verify merged count = sum. Read full source tail — truncated reads miss trailing classes
- Every bug fix (S-item) should get a test proving the fix — if there's no test, the fix is unverified
- Optional dependency tests: inject fake module via `types.ModuleType` + `monkeypatch.setitem(sys.modules, ...)` — avoids requiring the real package
- Lazy `__getattr__` modules (e.g. `enigma_engine.core`) break `patch("module.submodule.Class")` — the patcher resolves the dotted path via `getattr` at runtime, which raises AttributeError. Fix: inject fully fake sub-modules via `patch.dict(sys.modules, {"enigma_engine.core.model": fake_mod, ...})` so `from X import Y` inside the target function picks up the mock
- **Shim-vs-format-reader distinction (May 27 2026 BC sweep).** When grep finds `"legacy"` / `"for backward compatibility"` / `"kept for compat"` in code, you have to READ each site to decide: is this an API shim (delete, per §2 #4 "Do not add backward-compatible shims") or a format-version reader on the disk loader (keep — users have existing on-disk artifacts)? Examples seen this pass: TRUE shims (deleted) — `model_config.py` whole-file re-export, `tokenizer.load_tokenizer()` alias, `gguf_loader` re-export block, `TrainRequest.data_file/epochs/learning_rate/batch_size` parallel-shape. FALSE positives (kept) — `ai_profile.py` "version field for compatibility" (format versioning is legitimate), `bpe_tokenizer.py` "legacy files default to char-level" (on-disk legacy tokenizer loader), `model_merging.py` `ckpt.get("model_config", ckpt.get("config", {}))` (checkpoint key fallback for files written by older trainers). The grep noise is high; the actual rule violations are usually few. Detection rule: shim = "code that exists only to keep an old in-process caller happy." Reader = "code that loads an on-disk artifact users still have." Different category; only the first violates §2 #4.
- **Deleting a shim means deleting its tests too (May 27 2026 BC sweep).** When a class/function/route exists only to provide backward-compatibility, the tests that exercise it usually exist only to verify the shim's contract. Migrating those tests to "still works on the new shape" duplicates new-shape test coverage. Pass-bound case: `TestModelConfigShim` (3 tests) was verifying `get_model_config` exists as a re-export, returns a dict, and `MODEL_PRESETS is the same object` — all confirmation of the shim itself. With `model_config.py` deleted, the tests had nothing to verify and were deleted alongside the shim. Same with 6 `data_file`-shape tests in `test_api.py` (3 path-traversal, 2 length, 1 routing). The exception is a single transition test that asserts the OLD shape now fails loudly (`test_train_rejects_legacy_data_file_field`) — that test gates the shim's removal so future regressions can't silently re-add it. Pattern: delete N shim-verification tests, add 1 transition rejection test.

## 5. TEST LOOP

**Feature work:**
```
Write Test → Build Feature → Run Test → Pass? → Yes: Refactor & Merge Tests → No: Fix & Repeat
```

**Bug fixes — mandatory sequence:**
```
1. Reproduce the bug (confirm it actually exists)
2. Write a test that FAILS because of the bug
3. Fix the bug
4. Confirm the test now PASSES
5. Run full suite — zero failures
```

> A bug fix with no failing test first is unverified. The test is proof the bug existed and proof the fix works.
> If you skip step 2, you may be fixing the wrong thing — or nothing at all.

Command: `python -m pytest tests/ -v`

### Definition of Done — pre-flight checklist (run BEFORE claiming any slice or fix is finished)

> **Why this exists.** Three consecutive May 27 2026 audits each found a real issue in the *immediately-prior* work: F4 (sibling miss), F-A (sibling miss on the stream path), F-Audit-2 (lock leak in the F-A fix). The pattern: "done" was declared after checking the diff, without checking the *family* and the *exit paths*. The §4 principles already cover all three, but buried in 270 bullets they don't fire at the moment of "done." This 5-item list is the mechanism. **Audit fixes get the same checklist as original work** — F-Audit-2 proves a fix can introduce its own bug.

1. **Sibling-sweep.** Grep the outer condition / function name / endpoint / kwarg you changed. Does every match in the same contract family get the same change? (Would have caught F4, F-A.) For a gate at site A like `if ctx.is_gguf and hasattr(...)`, grep the *condition*, not just your new line.
2. **Exit-path safety.** Does every resource your diff acquires (lock, temp file, open handle, cache slot) get released on the exception path AND every early-return path? If your diff acquires-then-runs-code-then-releases, that code is inside try/finally. (Would have caught F-Audit-2.)
3. **Falsification.** Temporarily break the code your new test gates; confirm the test FAILS; restore. A test that passes on broken code is presence-only and worthless.
4. **Tracker sync.** Added / changed / deleted a file? Its CLEANUP_TRACKER row + any doc claim about it (CODE_REVIEW, SUGGESTIONS, docstrings) is updated in the SAME pass. (Would have caught F-B/C/E.)
5. **Suite + lint green.** `ruff check enigma_engine/ tests/` clean AND `python -m pytest tests/ -q` passing.

If you cannot tick all five, the slice is **parked, not done** (§1 #20). State which item failed and why in the SUGGESTIONS.md entry.

### Testing Rules

- One test file per module (e.g., `test_training.py` for `training.py`)
- Merge related small test files — no single-test files
- Tests must be independent — no shared mutable state between tests
- Mock heavy dependencies (torch, file I/O) — tests should run in ~17s total
- Delete test files that test removed features
- **Tests specify WHAT, not HOW** — test intended behavior (output, side effects, errors), not implementation details (source patterns, function calls, variable names). A test that passes when the code is wrong is worse than no test.
- **Structural tests (`inspect.getsource`) are a last resort** — only use when behavioral testing requires hardware not available in CI (GPU, GUI). Annotate with a comment explaining why structural is necessary.
- **Strict schema boundaries (`extra="forbid"`) need at least one behavioral payload-validation test** — source-presence tests can pass while runtime `model_validate(...)` fails on wrong key shape.
- When 3+ tests inspect the same target method, use one shared source helper (e.g. `_get_init_common_source()`) instead of repeating local imports/extraction in every test — keeps structural assertions focused on contracts and reduces copy-drift noise.
- **Write tests from the spec, not from the code** — ask "what should this do?" before reading how it does it. If you read the code first, you'll test what it does, not what it should do.

---

## 6. WHEN STUCK

1. Stop — do not guess or hallucinate solutions
2. State clearly: what you know vs what you don't know
3. Ask the user for clarification on ambiguous requirements
4. If a fix breaks something else: revert completely and rethink — no patches on patches
5. Explain *why* when pushing back on an approach (with alternatives)

---

## 7. WORKFLOW

### Before Coding
1. Read this file
2. Read SUGGESTIONS.md
3. Check what files exist (search before building)
4. Apply the author's lens to the code you are about to touch — would you write it this way? what is it connected to? what connections are missing? Log findings in SUGGESTIONS.md before writing any code.
5. **For bug fixes: write a failing test first** (see Bug Fix Loop above)
6. For features: write the passing test first

### After Coding
1. Run `ruff check enigma_engine/ tests/` then `python -m pytest tests/ -v`
2. Run `python run.py` to verify
3. **Every bug fix must have a test that would have caught the original bug** — if no test exists, add one before closing the fix
4. **Update "Learned Principles"** in this file when a reusable pattern or anti-pattern emerges
5. Update `SUGGESTIONS.md` when confirmed fixes, backlog items, or priorities change
6. Update `GUI_REFERENCE.md` only when visible GUI behavior has changed

### Priority Order (when rules conflict)
**Correctness > Simplicity > Performance > Style**

---

## 8. CODE QUALITY

**Before implementing:**
- Confirm the real problem: "What are you experiencing?"
- Propose 2-3 approaches when there are choices — let the user decide
- Consider edge cases and error paths from the start
- Apply the author's lens: how would you have built this? Is there a better way? What is it connected to, and what connections are missing?

**While coding:**
- Add clear comments only for non-obvious logic
- Cross-platform: must work on both Linux and Windows
- Everything should actually work — no placeholder implementations
- Think about ripple effects — changes often connect to multiple places

**After coding:**
- Update related files together (tests, docs, imports, callers)
- If changing core/, run full test suite
- Test before claiming something is broken
- Evaluate what you built and note anything worth recording

**Trust the spec, not the code:**
- The spec, the tests, and §4 Learned Principles are the precedent — code on disk is the suspect
- Remove comments or code when you have a reason and the reason is logged (commit message, SUGGESTIONS entry, or the diff itself if obvious)
- Inside the named overhaul scope (§1 #18): change anything that earns a better outcome
- Outside the named overhaul scope: leave it alone unless it's a real bug, then fix the bug only — no taste-driven drift
- Theoretical improvements with no scar behind them stay in SUGGESTIONS.md until they earn a Pass
- Ignore the installed mods and base model state when working on GUI — not relevant

---

## 9. QUICK COMMANDS

```bash
python collect_vision_data.py --llava-pretrain 100000 --images-dir D:/datasets/llava/images  # V-5 vision SFT data
python collect_vision_data.py --stats                # Show collected vision data summary
ruff check enigma_engine/ tests/                     # Lint (run before every commit)
python -m pytest tests/ -v                           # Run all tests verbose
python -m pytest tests/ --tb=short -q                # Run all tests (compact output)
ruff check --fix enigma_engine/ tests/               # Auto-fix safe lint issues
ruff check --unsafe-fixes --fix tests/               # Fix F841/E731 in test files
python run.py                                        # Show system info
python run.py --gui                                  # Launch desktop GUI (tkinter — scheduled for deletion after Gradio UI ships, Strategy Reset May 26 2026)
# python run.py --web                                # Svelte UI abandoned; Strategy Reset chose Gradio. Re-enable when enigma_engine/ui.py (Gradio) is built (SUGGESTIONS.md Block 2).
python run.py --serve                                # Start API server (port from CONFIG)
python run.py --serve --port 8080                     # Start API server on specific port
python run.py --train data/training.txt --epochs 10  # Train model
python run.py --train data/training.txt --epochs 10 --seed 42 --deterministic  # Bitwise-reproducible CUDA training (5-15% slower; --deterministic requires --seed)
python run.py --train-tokenizer data/training.txt    # Train BPE tokenizer
python run.py --benchmark                            # Run coherence benchmark on default model
python run.py --benchmark --model models/my.pth      # Benchmark a specific model
python run.py --help                                 # Show all CLI options
python collect_pretraining_data.py --stats            # Show collected data summary
python collect_pretraining_data.py --all-sources      # All sources (wiki, books, fineweb, SE, wayback, owt, c4, dclm, finemath, code)
python collect_pretraining_data.py --fineweb 25       # 25 GB FineWeb-Edu (pip install datasets)
python collect_pretraining_data.py --openwebtext 10   # 10 GB OpenWebText web text (pip install datasets)
python collect_pretraining_data.py --c4 20            # 20 GB C4 cleaned Common Crawl (pip install datasets)
python collect_pretraining_data.py --dclm 15          # 15 GB DCLM model-filtered web text (pip install datasets)
python collect_pretraining_data.py --finemath 10      # 10 GB FineMath step-by-step math (pip install datasets)
python collect_pretraining_data.py --code 10          # 10 GB The Stack v2 code (pip install datasets + HF auth)
python collect_pretraining_data.py --stackexchange    # Stack Exchange Q&A (pip install py7zr)
python collect_pretraining_data.py --wayback 1000     # 1000 Wayback Machine educational pages
python collect_pretraining_data.py --books 500        # Expanded Gutenberg (400+ curated)
python collect_pretraining_data.py --resume           # Resume interrupted download
python collect_pretraining_data.py --combine-only     # Re-merge with paragraph dedup
python collect_distill_data.py --endpoint http://localhost:11434/v1 --model qwen3:8b --prompts data/distill_prompts.txt --tag qwen3_8b  # N-19 external-teacher distill corpus (Ollama / llama.cpp / vLLM / our own --serve); add --resume to skip prompts already in JSONL
python collect_distill_data.py --endpoint http://localhost:11434/v1 --model qwen3:8b --magpie 500 --tag qwen3_magpie --temperature 1.0  # N-19 slice 2: Magpie empty-prefix instruction synthesis (arxiv:2406.08464). Generates 500 instruction/answer pairs from the model itself — no input prompts needed. Add --template chatml|llama3|custom to match the model's chat-template family.
```