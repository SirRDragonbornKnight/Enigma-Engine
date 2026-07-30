# Enigma Engine — Backlog

> Consolidated open work, newest snapshot 2026-07-15. Sources: the 4-reviewer
> methods audit (verdicts in `ROADMAP.md` "Update 2026-07-15"), the
> ultrareview ledger (`_archive/ULTRAREVIEW_2026-07-12.md`), and the
> dormant-code audit (`_archive/AUDIT_2026-07-13.md`; both moved to
> `_archive/` in the 2026-07-18 compression pass).
> Priority = leverage x confidence, not size.
> Status: [ ] open  [~] in progress (uncommitted)  [x] done this arc.

---

## Open user decisions — the standing ledger

Every gate that waits on the user, in ONE place (2026-07-27 audit finding:
two gates existed only in session memory). Detail lives at each pointer; a
gate recorded only in conversation is a bug in this table.

TWO RULES THIS TABLE NEEDS, both from the 2026-07-28 audit, which found that
all four rulings closed here had left a contradiction somewhere else --
four for four:
* **Closing an item is not done until the docs that restated its premise are
  fixed.** Strike the row, then list the files touched, so the next audit is
  a grep instead of a read.
* **Closing a T-step opens the next one's gate.** T1 closing should have put
  the T2 go-hot on this table the same day; it did not, and the next decision
  was invisible in the file built to make decisions visible.

1. ~~Collector GB targets + DCLM's role~~ **RULED 2026-07-27: SWAP.** DCLM
   replaces C4+OpenWebText (both removed from `SOURCE_DIRS`; their 32.4 GB
   was ruled dead by section 9 on 2026-07-29); collector targets = the defaults
   15/10/10 GB (DCLM/FineMath/Stack). Quality up at roughly constant corpus
   size. The 7.9 grid was re-derived from the new sidecar and MOVED --
   10.0/10.5/22.9 d/epoch, not the pre-swap 8.4/8.8/19.2 (and re-measured
   again 2026-07-28 at the launch shapes: 4.1/4.9/20.8). -> 7.95 T1.
   Docs touched: BACKLOG 7.9 + 7.95, ROADMAP, TOKENIZER_V2_SPEC,
   information/training_guide.md.
2. ~~Image begin/end row mechanics~~ **EXECUTED 2026-07-27: `<|image|>`/
   `<|/image|>` at reserve rows base+6/base+7 (v2 = 16,372/16,373) via
   `attach_image_tokens` in chat_format.py -- INSTANCE-attached constants,
   the v1 chat-token pattern followed exactly.** Mechanism correction from
   the ruling's parenthetical, forced by measurement: v1's vocab FILE
   carries no beyond-table specials at all (its chat tokens are code
   constants), so the image rows follow suit -- the vocab file's sha is
   unchanged, and because the file maps neither literal, corpus text
   structurally CANNOT carve into a reserve row; no sanitizer entry is
   needed (the `<image>`-in-HTML hazard only exists for table specials).
   User text cannot forge the markers either (`_enc_content` neutralizes
   them like the chat tokens). Layout recorded in TOKENIZER_V2_SPEC;
   test-pinned on both the v1 and v2 vocabs. CLOSED. -> 7.95 T1.
   Docs touched: TOKENIZER_V2_SPEC, ROADMAP (which had said the tokens were
   never allocated), BACKLOG section 5.
3. ~~tokens_v2 output naming~~ **EXECUTED 2026-07-28: `tokens_v2b.bin`**
   (28,261,718,460 tokens, 52.64 GiB = 56.52 GB uint16, 65.8 min; sidecar
   verified -- curated x5 walked first, extents recorded, 660 literals
   scrubbed at consume time; bin + sidecar attrib +R). The old
   `tokens_v2.bin` stays on disk as the receipted rollback; every T2/T3
   command now names v2b. CLOSED. -> 7.9.
   Docs touched: TOKENIZER_V2_SPEC, ROADMAP, BACKLOG 7.9/7.95/section 8,
   information/training_guide.md, the SACRED list.
4. ~~Teach-line content reseal~~ **EXECUTED 2026-07-27 ("seal it"):
   reseal #7 = 120 probes / 15 per gated category / 135 sealed strings
   (manifest 4e4d4433, grading digest 4fb386d7, plaintext 6b0643db); array
   sort closed the element-order channel; pool pruned 233 -> 205; all four
   artifacts + the curated shard rebuilt; attributed baselines re-based:
   v8 56/120, v5 55/120.** EVAL_REDESIGN owns the receipts. CLOSED.
   Docs touched: EVAL_REDESIGN (owner of the scorecard numbers), plus the
   47/96 -> 56/120 correction in KNOWN_ISSUES, PHASE7_GATE, ROADMAP,
   TOKENIZER_V2_SPEC and VISION.
5. **Section 9 disk reclaim -- RULED 2026-07-29, EXECUTION PENDING (user):**
   "delete anything old and unused and will not be used in the future."
   Every row dies except `enigma_pi_zero.pth` (VISION heritage citation --
   HELD; its row is split below so the hold is file-granular). Dying under
   the same ruling from outside the table: C4+OpenWebText (30.2 GiB, out of
   SOURCE_DIRS by ruling 1), the forgotten June copy at `C:\Enigma-Backups`
   (18.3 GiB / 19.7 GB), the venv's llama-cpp-python package (0.69 GiB, GGUF-pivot
   residue), and the 07-29 audit's added orphans (enigma_dpo_v8,
   enigma_forge_tiny, enigma_lora_v1, two vision-align sibling copies,
   wiki_dump_index.txt.bz2, models\registry.json, data\prompts\, data\notes\,
   progress.json.bak, the serve logs, the orphan .venv\) -- ~245 GiB total,
   every target verified on disk first. NOTE: `data\enigma_voice.md` and
   `data\personality_corpus.jsonl` are GIT-TRACKED -- their deletion shows
   as ` D` in git status and rides the next ordered commit. tokens.bin /
   tokens_v2.bin were NOT named -- SACRED, pending their own word. ADDED
   post-sweep 07-29: the six sweep point models (models\sweeps\t2_238m\*,
   ~11.3 GB) are throwaway once Gate B is ruled -- user's word; the receipt
   `sweep_results.json` itself NEVER dies.
   FOLLOW-UP opened by the audit: `--all-sources` still re-downloads c4+owt
   (collect_pretraining_data.py:3011/:3033) -- drop them from that flag
   before the next collector run. The classifier blocks mass deletion from a
   Claude tool call, so the exact commands were handed to the user; this row
   stays OPEN until that run completes. -> section 9.
   Docs touched: this row, section 9 header + rows, CLEANUP_TRACKER.md:41,
   TOKENIZER_V2_SPEC.md:457, pretokenize_data.py SOURCE_DIRS comment,
   extend_length.ps1 header, KNOWN_ISSUES.md item 5, ROADMAP.md rulings
   block (the privacy scope + one-hot-job rules land THERE while CLAUDE.md
   stays classifier-blocked).
6. ~~T2 go-hot~~ **CLOSED 2026-07-29: the sweep RAN** -- detached,
   2026-07-28 21:16 -> 2026-07-29 (~9 h); receipt
   `models\sweeps\t2_238m\sweep_results.json`. FINALIZED 05:46 -- 6/6
   points rc=0; the last point (6e-3 s1) wrote 3.1641, WORSE than 3e-3, so
   the interior win stands on complete data. GPU RELEASED. Verdict at 7.95 T2: the lineage LEARNS and the
   LR is MEASURED -- **3e-3, an interior win**. Per the
   closing-opens-the-next rule: **the live gates are now item 7 (Gate B,
   size) and item 11 (rebuild y/n) -- both sit directly between here and
   T3, and T3 has a pre-flight list at 7.95 T3.**
   Docs touched: 7.95 T2 (status + thermal receipt), item 7 (LR-transfer
   note), 7.95 T3 (measured-LR annotation + pre-flight); stale-premise trio
   caught by the 07-29 audit and fixed same day: ROADMAP.md:335,
   PHASE7_GATE.md:9, VISION.md:65 (all still had T2 gating the launch).
7. ~~**Gate B -- pretrain size call**~~ **RULED 2026-07-30: `v2_deep_238m`.**
   Decided on the corrected ballot below (4.29x wall clock, measured LR at
   the exact shape, compile ON, 1.50x serve latency, ~197 GB archives).
   T3 launches the 238m command in section 7.9. The table and receipts stay
   for the record:
   238m vs 542m vs 186m on the v2b grid.
   **Decode latency MEASURED 2026-07-28** (RTX 5090, torch 2.10, serve path =
   generate_stream + serve's sampler under bf16 autocast, batch 1, prompt 32,
   48 decoded tokens, median of 5; random init -- shape drives batch-1 decode
   cost and no v2 checkpoint exists; each preset at the vocab it would serve).
   Receipt: `Enigma Backups\decode_latency_2026-07-28.json`.

   | preset | shape | params | ms/tok | tok/s | vs v8 | d/epoch |
   |---|---|---|---|---|---|---|
   | `large` (v8 today) | 16L dim1024 | 182.1M | 12.79 | 78.2 | 1.00x | -- |
   | `v2_deep_186m` | 28L dim768 | 186.1M | 26.59 | 37.6 | 2.08x | 4.1 |
   | `v2_deep_238m` | 20L dim1024 | 238.4M | 19.14 | 52.2 | 1.50x | 4.9 |
   | `v2_deep_542m` | 30L dim1280 | 542.1M | 28.21 | 35.4 | 2.21x | 20.8 |

   Latency is driven mainly by LAYER COUNT but not by it alone: 186m is 2.08x
   v8's for 1.75x the layers, 542m 2.21x for 1.88x -- depth dominates, and dim
   plus the 16,366-row vocab add the rest. **186m is not dominated, but it is
   a poor buy**: it saves 16% training wall-clock (4.1 vs 4.9 d/epoch) and
   costs 39% more serving latency than 238m for 22% fewer parameters. 542m
   costs 2.21x today's latency and **4.29x the wall clock** -- 20.8 vs 4.9
   d/epoch, the ratio this table's own rates give -- plus an archive bill of
   ~450 GB against 238m's ~197 GB at the cadence each shape needs (derivation
   under the launch commands). -> 7.95 T3.
   **NEW INPUT 2026-07-29 -- the LR sweep ran at 238m.** A 238m pick
   inherits a MEASURED 3e-3. A 542m pick inherits 3e-3 only via the Muon
   scale-invariance argument (unmeasured) -- or spends its own sweep at
   ~35 h (2B tok at 542m's 15.7k tok/s eager rate, compile unavailable at
   that shape). One more measured point in 238m's favor.
8. **Gate D -- adoption ratification**: T6's rule (beat the P2 aggregate
   with no category floor regression = adopt; anything else = user's call)
   is now code -- `eval_behavior --baseline <transcript>` prints the
   verdict. Swap mechanics at T6. -> 7.95 T6.
9. **Voice live-listen**: the Cortana blend was chosen by documented
   character on 2026-07-23 and has never been HEARD (the user had no audio);
   candidates A-D at `Desktop\Enigma Voice Audition`. Listen, tune or bless.
   Also open from that arc: the Quiet toggle gives no feedback popup.
10. **Vision training domain**: her eyes train on WHAT imagery ("different
   imagery, not everyday photos" -- domain never picked). ->
   VISION_QUALITY_SPEC section 4.
11. ~~**Is v2b good enough to train on, or is it worth ~1 day to rebuild?**~~
   **RULED 2026-07-30: REBUILD BEFORE T3** (the recommendation as written,
   incl. the representative val holdout riding the rebuild). v2b stays on
   disk as the receipted rollback until the rebuilt bin validates. The
   four defects, kept for the record:
   One decision, because the same re-collect + retokenize fixes all four
   (2026-07-28 audit, every figure measured against the shipped artifact):
   * **The Stack is 100% Python.** 2,032 files, all `stack_python` = 4.87 B
     tokens = **17.25% of the corpus, monolingual**, against a docstring
     promising 16 languages. The collector is FIXED (each language now takes
     an equal share of the remaining target), but the fix only helps a fresh
     pull.
   * **Cross-source dedup did not run to completion.** `MAX_DEDUP_ENTRIES` is
     50 M against a sampled estimate of 190-275 M paragraphs in the walk, so
     the table filled partway through and every source after that point was
     compared only against what came BEFORE it, never against the sources that
     follow. FineWeb-Edu and DCLM are both CommonCrawl-derived and sit on the
     far side of the fill. Recorded `dupes_skipped` 1,436,089 = under 1% of
     paragraphs, implausibly low for that mix.
     WHERE it filled is NOT established: two independent samplings put the
     crossing in different sources (Wiki Dump at walk position 2, and Gutenberg
     at position 5), because the estimate rests on a few files per directory
     across 100 GB. That the cap was hit at all is what the conclusion needs,
     and the totals support it under either sampling. The sidecar now records
     `dedup_capped` outright; v2b's predates the field, so for THIS corpus the
     cap-hit rests on the paragraph-count argument, not a receipt.
   * **~80% of the corpus has no document boundaries.** HF streaming packs
     ~5 MB of many records into one .txt and the walk treats one FILE as one
     document, so `<s>`/`</s>` are learned almost entirely from the 16.6%
     wiki/fandom slice. Measured bos in the first 5 M tokens of each extent:
     FineWeb-Edu 4, The Stack 3, DCLM 4 -- against Wiki Dump 4,870. A model
     that must stop cleanly in chat is learning where to stop from a sixth of
     what it reads.
   * **11 `<search>`/`</search>` pairs in the curated shard are broken.** The
     consume-time literal scrub has no source allowlist, so the curated
     dialogue's deliberate search turns became `< search>` / `< /search>` --
     ids 12/13 appear ZERO times in the curated extent, and at x5 that is 55
     broken pairs teaching a surface form the tokenizer can never carve back.
     (T1 ruling 4 deletes the 31 orphan tag records at T4 anyway, so the
     cheapest fix may be to drop them from curated rather than exempt them.)
   MY READ: items 1-3 are real quality costs and item 4 is noise-level. None
   of them corrupts the corpus -- v2b is trainable as it stands. But T3 is
   4.9 days at 238m and 20.8 at 542m, and a ~1-day rebuild before it is cheap
   insurance; a rebuild after it is not. RECOMMEND rebuild before T3. (T2 has since answered the
   "is this lineage learning" question on the current bin: yes.)
   **SCOPE ADDED 2026-07-29 (audit + sweep evidence):** the same rebuild
   must lay down a REPRESENTATIVE VAL HOLDOUT (strided/multi-source -- no
   flag over a contiguous corpus can produce one, and the sweep measured
   the FineWeb-Edu window's homogeneity directly: val-gen seed spread
   tighter than tail-val on every rung -- 23x / below-print-resolution /
   3.9x, ~9x on rung means). This is the one cheap moment for it; after T3
   starts it is unfixable for the lineage. The T2-swept LR carries to the
   rebuilt corpus (same vocab, similar mix) -- accepted 2026-07-29.
   **REBUILD RUNBOOK (code landed 2026-07-30; collect is the user-gated
   step -- network + hours of CPU):**
   1. The code fixes are IN: Bloom dedup (400M capacity, 0.1% design FPR,
      ~0.7 GiB -- replaces the 50M exact set that capped mid-walk),
      U+001E record separators written by every packed-file fetcher and
      split at the pretokenize walk (each record = own document = own
      `<s>`/`</s>`), the FineWeb-Edu fetcher now runs the same
      literal-scrub + sealed-probe screen as the other three, and the
      Stack pull takes an equal per-language share (16 languages).
   2. ~~**Old files must be CLEARED first, or the re-pull no-ops**~~
      **DONE 2026-07-30 -- and cleared by MOVE, not delete.** Every
      fetcher reads existing bytes against its target and skips when met,
      so re-pulling over the 2,032 all-Python Stack files would have kept
      the monolingual corpus, and old packs carry no separators. The
      15,314 files / 81.37 GB in `data\pretrain\{fineweb_edu, dclm,
      finemath, the_stack}` were MOVED to
      `C:\Users\SirKn\_retired_packed_2026-07-30\` -- a same-volume
      rename, so it cost no space and is reversible until the new corpus
      validates. The four source dirs remain, empty. Resume keys
      `dclm` / `finemath_4plus` / `infiwebmath_3plus` / `stack_python`
      stripped from `data\pretrain\progress.json` (a stale
      `records_consumed` skips the fresh stream past records it never
      saved); `fineweb_edu` had no key. `gutenberg_ids`, `stats` and
      `fandom_done_wikis` are untouched, and the pre-edit file is at
      `Enigma Backups\progress_2026-07-30_pre-rebuild.json`.
      All four sources were PROBED LIVE before the move (one streamed
      record each, incl. the license-gated Stack): all re-pullable.
      The one-article-per-file sources (wiki/SE/gutenberg/fandom/wayback/
      curated) keep: they have real boundaries already.
      **Delete the staging dir only after the rebuilt bin validates.**
   3. **RE-COLLECT LAUNCHED 2026-07-30, DETACHED** via scheduled task
      `EnigmaCollectRebuild` -> `run_collect_rebuild.ps1`. Parent chain
      verified `python <- python <- powershell <- svchost <- services <-
      wininit`, no `claude.exe`, so restarting the Claude app cannot kill
      it. Command:
      `--resume --no-combine --fineweb 40 --dclm 15 --finemath 10 --code 10`
      (targets are GiB and reproduce the v2b diet as measured on disk).
      **`--resume` is MANDATORY**: without it the collector resets
      progress to `{gutenberg_ids: [], stats: {}}` and drops
      `fandom_done_wikis`, forcing a full Fandom re-pull -- the four
      rebuilt sources start fresh anyway because their keys were
      stripped. **`--no-combine` is MANDATORY**: the combine step writes
      a ~95 GB `combined.txt` that pretokenize never reads and section 9
      ruled dead. Log `data\collect_rebuild_2026-07-30.log`, completion
      marker `.done` carries the exit code. Two launcher bugs found and
      fixed on the way in, both worth keeping: `-ArgumentList` as an
      ARRAY does not quote its elements, so the space in "Enigma Engine"
      split the script path; and python block-buffers redirected stdout,
      so the log stayed 0 B and a stall would have been invisible for
      ~80 minutes (`PYTHONUNBUFFERED=1`). Leave the task registered while
      it runs -- the launcher is what writes the marker.
   4. Then retokenize with the standing line
      (`--repeat-sources curated=5`, uint16, workers 10, BelowNormal) to
      a NEW versioned bin; v2b stays on disk as rollback until the new
      sidecar validates (extents present, `dedup_capped` false, bos
      counts in the packed extents no longer single-digit per 5M tokens).
   5. T3 then adds `--val-per-source 2000000` to the launch line: one
      fenced window per source, [val-src] = diet-weighted mean (the
      representative signal; landed with `--eval-only` 2026-07-30).
12. **Memory recall ceiling k=3.** `render_context` defaults to 3 facts and
   both serve call sites take the default, so she can never surface more than
   three memories however many she holds -- and ties break newest-first, so
   the OLDEST fact is the one silently dropped. Measured: five allergies
   stored, three recalled, token budget nowhere near binding. The sealed gate
   cannot see it (all 15 memory probes teach exactly one fact each). Raising
   the default changes her live behaviour, so it is a ruling, not a fix.
   -> `enigma_engine/core/memory_store.py` render_context.
13. **The `unknown` category does not penalise fabrication.** Measured on the
   live sealed set: `"I can't know that."` scores 13/15 and `"I can't know.
   It is blue."` also scores 13/15 -- appending an invented answer to a
   decline is free. This is the category most likely to move at v2 (both
   baselines are 0/15, with ZERO refusals in 30 unanswerable questions), so
   it could jump to 15/15 without epistemics improving at all. Fixing the
   deny keys means a content reseal, which is the user's gate. The designed
   alternative is the offline second grader. `tool_graded` unblocks it for
   FUTURE runs -- those transcripts re-grade from themselves. The two BASELINE
   transcripts predate the field (both written 2026-07-27), so re-grading them
   still needs the sealed plaintext beside them.
   -> EVAL_REDESIGN second-grader section.
14. **Structured output in v2 scope?** Zero support exists anywhere today
   (no response_format / grammar / logit_bias; in no doc or backlog line
   before this one). If v2 should have it, its DATA SHAPES must ride the T4
   regen -- missing that window costs a full regen cycle. Runtime
   enforcement can follow later; the trained shapes cannot. -> 7.95 T4.
15. **Memory Tier-2 + episodic at T4?** The structured fact store
   {subject, attribute, value, date, kind} is nearly free at the regen (SFT
   already trains the sentence shape), and reserving kind:"episode" makes
   session memory ("what were we working on yesterday?") an addition, not a
   rewrite. Same T4-window argument as item 14; interacts with item 12's
   k=3 ruling. -> 7.95 T4.

---

## 0. Instrument arc (2026-07-15) — landed in full

- [x] **SFT val-split dedup** (`finetune_enigma.py`) — val fills only from
  records whose ids appear exactly once; duplicates all train (`47f557ae`).
- [x] **Merge `teach_pairs.jsonl` into DPO** (`make_dpo_data.py`) — user `/fix`
  corrections fold into preference data x3, behind the probe holdout (`47f557ae`).
- [x] **DPO val grouping by prompt** (`dpo_enigma.py`) — `group_split` deals
  whole prompt-buckets to val; no (prompt, chosen) twin or x3-taught duplicate
  straddles the sides (`fd2776d1`).
- [x] **Eval grader word-boundary matching** (`eval_behavior.py` + probes) —
  substring grading passed wrong answers ("own" on "known", "no" on "nothing")
  AND a perfect trained hosting answer failed probe 4's key list; fixed both
  directions, locked by `tests/test_eval_grading.py` (`bacc7473`). v5
  re-measured 27/29 on the 29-probe suite — superseded the same day by the
  90-probe suite below.
- [x] **`/undo` really undoes** (`teach_enigma.py`) — retracts the persisted
  records by truncating to pre-append offsets; a second `/fix` replaces the
  first instead of rejecting the user's own correction (`deb7c182`).
- [x] **Eval probes 29 -> 90** (`090e6644`) — every new probe machine-vetted:
  no trained-string collisions, facts only from the knowledge corpus, keys
  aligned with trained answer families. v5's HONEST baseline: 70/90 (78%),
  RESULT FAIL (adversarial 8/12, restraint 9/12, factual 13/20).

---

## 0.5 Eval trust -- de-contamination (2026-07-16 realism audit; see EVAL_REDESIGN.md)

> The behavior gate is partly self-measuring: `knowledge_corpus.py:21` authors
> probe twins on purpose and the leak guard is exact-match only, so
> factual/identity/adversarial/restraint scores conflate recall with
> generalization (math/memory/tool are clean). Full design + status in
> `EVAL_REDESIGN.md`. Tokenizer ceiling spec in `TOKENIZER_V2_SPEC.md`.

- [x] **Grader concession fix** (`eval_behavior.py`) -- adversarial/identity fail
  on an AFFIRMED false origin (`_false_origin_conceded`/`_grade_identity`); true
  greedy default (temp 0). Tests in `tests/test_eval_grading.py`.
- [x] **Locked-probe guard machinery** (`eval_leak_guard.py`) -- sealed
  hashed-shingle manifest + fuzzy Jaccard guard, wired into `make_sft_data`
  (`_held_out`), no-op until a manifest exists. Tests in
  `tests/test_eval_leak_guard.py`. Known limit: verb-swap paraphrases land in
  the 0.5-0.6 review band (flagged, not dropped).
- [x] **Author the locked probe set** — DONE 2026-07-24: the user's 96-probe
  file was validated and sealed (108 strings incl. 12 teach lines); manifest
  committed, plaintext gitignored, durable copy in `Enigma Backups`.
- [x] Widen thin eval categories to >=15 probes: **SEALED set DONE
  2026-07-27 (reseal #7: 15 per gated category, meeting EVAL_REDESIGN's
  own >=15 rule; category stats are no longer directional-only).** DEV set
  remains 152 probes / 11 categories (EVAL_REDESIGN owns the count) with
  `unknown` and `speech` at 9 -- the dev side can widen any time, no seal
  involved. Re-measure v5/v8 on the locked set (P2) for the honest
  baseline.
- [ ] (Optional) second-grader agreement pass; semantic-embedding leak guard to
  close the verb-swap gap.

## 1. Correctness / measurement instruments (high leverage, small)

- [x] **Memory learns, corrects, and forgets on the spot** (user ask
  2026-07-24: "learn new things easily, preferences about me, correct on the
  spot"). Landed on the LIVE lineage (gate improvements, no v8 retrain):
  the write gate now catches professions/possessions/origin ("I'm a nurse",
  "I have two cats", "I was born in 1990") and NATURAL factual corrections
  ("Actually, my dog is Bruno now", "No, it's Samantha") so the supersede
  path fires from chat; a new `forget` built-in (MemoryStore.forget removes
  what recall would surface, never an unrelated fact) with a `_FORGETTABLE`
  gate that SUPPRESSES remember ("forget that I like tea" no longer re-saves
  it); recall keys on the last N user turns, not just the last message, so a
  follow-up no longer blanks the memory block. forget joins the standard
  built-in block (new models inherit it). STILL OPEN (Tier-2, deferred): the
  structured {subject,attribute,value} fact store and episodic/session memory.

- [x] Encoder **persistence bug** — FIXED `f9ec5184`: `_save_checkpoint` takes
  encoder/optimizer overrides, vision/audio save their encoder + the LOCAL
  optimizer that actually stepped, `_load_encoder_checkpoint` resumes and
  REFUSES text-only checkpoints; 6 tests. Residual open: serve-side
  native-encoder load path (Phase 4.5 step 5).
- [x] **Checkpoint-safety arc 2026-07-19** (`vision_align.py` +
  `align_vision.py`) — two audited rounds, converged; 12 regression tests
  (file now holds 17), each mutation-verified; suite 349 -> 361. Round 1:
  missing `resume_from` REFUSES (was warn-and-restart, which then overwrote
  the prior best), `_load_encoder_checkpoint` writes the `.keep` cleanup
  marker, train_vision body in try/finally (aborts left params frozen +
  encoder in train mode), `save_every_steps` implemented (rolling
  `{stem}_vision_step.pt`; `align_vision.py --save-steps`, default 500).
  Round 2 (adversarial audit of round 1): best tracking split — `best_loss`
  = pure metric (drives early stopping), `best_written` = what reached disk
  (drives retry); a run whose best never persisted ends with `abort_reason`
  set (align_vision SystemExits on it); mid-epoch rolling-checkpoint resume
  winds scheduler+step back to the epoch boundary (`epoch_start_step` is now
  load-bearing); str-path `.keep` fix; remainder-flush steps also fire the
  rolling save; `encoder_key` ValueError raised before the swallow-all try;
  fresh writes delete stale `.keep` markers; guarded finally.
- [x] **`--tokens-bin` resume-locked** (`pretrain_enigma.py`; final audit
  2026-07-16 M1) — FIXED 2026-07-16: `tokens_bin` is now recorded in the
  checkpoint schedule, and corpus resolution moved to AFTER the resume/schedule
  restore, so a bare `--resume` recovers the run's own corpus instead of
  silently finishing a facts run on the default 56.6B corpus. An explicit
  `--tokens-bin` still wins; checkpoints written before this fix predate the
  key and must re-pass the flag. (`test_pretrain_warmstart.py` still green.)
- [x] **`group_split` can't empty train / overshoot val** (`dpo_enigma.py`;
  final audit M2) — FIXED 2026-07-16: fewer than two prompt groups (small
  teach_pairs.jsonl) train the whole set with val empty; otherwise deal
  SMALLEST groups to val and never assign the largest group to val, so a giant
  group can neither empty train (`_batchify([])`) nor overshoot val_cap.
  Regression tests in `tests/test_dpo_split.py`.
- [x] **Facts-corpus val contract** (`make_facts_pretrain_data.py`; final
  audit M3) — FIXED 2026-07-16: `val_reserve = max(arg, target // 100)` so the
  pure-replay tail always covers pretrain's n//100 [val] window; a fence stops
  any fact doc from crossing `mixed_end` into that tail; documented command now
  passes `--val-general-end 0`. Regression tests in
  `tests/test_facts_pretrain_data.py`.
- [x] Repetition-penalty scope — was penalizing the prompt, suppressing her own
  primed vocabulary (ultrareview #9). Fixed + regression-tested (`b75ed617`).
- [x] Eval memory-store clear (#30) + golden-eval EOS strip (#12) — fixed (`fe5359a7`).
- [ ] Packing without doc-boundary attention masking — conversations attend
  across packed neighbors. Small effect at 182M/1024; INVESTIGATE only if
  context-bleed shows in chat.
- [ ] Instruct serve omits the trained "You are Enigma..." preamble when the
  client supplies its own system message (tools block joins without it).
  **VERIFIED AGAINST THE CORPUS 2026-07-19 — the shape mismatch is real, but
  the current behavior is DELIBERATE and test-pinned, so this is a decision,
  not a bug fix.** Evidence: `make_sft_data._system()` ALWAYS emits
  "You are Enigma. You can use tools...\nAvailable tools:...", and the
  memory_tools generator (`make_sft_data.py` ~764/784) shows that even when
  a block PRECEDES the tool spec, the preamble is retained
  (`block + "\n\n" + _system(subset)`). So the model has never seen a tool
  spec whose immediate left context isn't that preamble — which is exactly
  what serve renders when a client system message exists
  (`serve_enigma.py` ~942: the preamble is prepended only when there is NO
  client system message). Counter-argument: honoring the client's system
  message is correct OpenAI-compatible semantics, and
  `tests/test_serve_enigma.py::test_with_context_client_system_message_is_appended_not_preambled`
  pins "You are Enigma" NOT being injected. Middle path if we act: keep the
  client's message as the OPENER (their intent preserved) but restore the
  preamble to the tools block itself — client + "\n\n" + memories + "\n\n" +
  preamble + tools — which matches the trained join shape exactly; that test
  would need its assertion updated (its stated intent still holds).
  Reachability (checked 2026-07-19): LOW — no HTML/JS in the repo builds a
  system-role message, so her own chat page and the launcher chain never hit
  this branch; it only affects external OpenAI-compatible clients that send
  their own system prompt. That argues for leaving it alone until such a
  client actually matters. USER CALL.

## 2. Ultrareview backlog — verified-open correctness majors

> 11 confirmed open 2026-07-14. The LIVE-PATH subset was triaged and CLOSED
> 2026-07-17 (all six verified still present, then fixed + regression-tested;
> serve smoke + full 90-probe eval green): #6 (combined shape, data-side),
> #15 (name-less tool call kept via raw), #31 (stream/non-stream parity),
> #36 (serve bf16 autocast + TF32 — eval re-measured 79/90, same as fp32),
> #45 (memory-store fsync via atomic_write_text), #51 (/v1/memory 400).
> What remained below was the DORMANT training arsenal (LoRA / RL / queue).
> RESOLVED BY DELETION 2026-07-18: the compression pass removed the entire
> dormant Forge stack (training.py, dispatch/schema/registry,
> training_evaluation, rl_training, lora_utils, progressive_growing,
> reasoning, training_queue, training_monitor, run_training_diagnostic),
> so #5/#7/#8/#10/#11/#13/#33 and the ~63 unverified arsenal items no
> longer have code to be wrong in. #33's decision landed as "deprecate":
> `dpo_enigma.py` is the preference path. See CLEANUP_TRACKER.md.

- [x] #5 training_queue double-run — module deleted 2026-07-18.
- [x] #6 memory-block + tool-spec combined shape — FIXED 2026-07-17 (data
  side): `gen_memory_tools_examples` bakes serve's exact join (memories,
  blank line, preamble + tools; both answer-from-memory and still-call-the-
  tool behaviors), x8 in the mix (53 records), locked by
  `tests/test_memory_tools_data.py`. NOTE: the SERVED model only learns the
  shape at the next SFT->DPO cycle; until then serve still renders it.
- [x] #7/#8/#10/#11/#13/#33 — resolved by deletion 2026-07-18 (LoRA stack,
  online-DPO, Trainer preference paths all removed with the Forge bloc).
- [x] #14 non-SDPA attention used a square mask for rectangular cached decode
  (CPU/MPS path; CUDA SDPA path was already correct) — **FIXED 2026-07-18**.
  Characterized by execution first: it was a loud broadcast CRASH
  (`RuntimeError: size of tensor a (9) must match tensor b (3)`), not the
  silent corruption the old wording implied. Unreachable from the live serve
  loop (prefill once, then one token at a time), so it never bit us; any
  chunked prefill or multi-token continuation on CPU died. Fix mirrors the
  SDPA branch: bottom-right aligned `tril(diagonal=T_k - T)`; square prefill
  reduces to the old mask (served logits verified byte-identical).
  Regression tests: `tests/test_cpu_rectangular_decode.py` (5 tests, pins
  crash-freedom AND value-correctness vs no-cache recompute, plus a causality
  guard). MUTATION-VERIFIED against both the original square mask and a
  correctly-shaped-but-top-left-aligned mask.

## 3. Data strategy (the real quality ceiling)

- [x] **Drop OpenThoughts3** — out of `--all`, 58 MB source file deleted,
  regenerable via the explicit flag (`8104e09c`).
- [x] **Rebuild the diet** — collectors for No-Robots / Everyday-Conversations /
  TriviaQA / NQ-Open + SmolTalk2 diet mode (600-char completion cap, 800-char
  prompt cap, think-split skip). combined_finetune.jsonl: 105,203 SHORT pairs
  (`8104e09c`).
- [x] **Facts many-format CONTINUED PRETRAINING** — `make_facts_pretrain_data.py`
  (60M tokens, 2.4% facts, replay-anchored) + `pretrain_enigma.py --tokens-bin`;
  checkpoint `models/enigma_pretrain_facts` (`3b553038`, `701434be`). Measured:
  factual 13/20 -> 19/20 on v6. Val-contract nit open in section 1.
- [x] **knowledge_corpus format mixing** — `gen_knowledge_pretrain_text`: 914
  lines as declarative / QA / key-term-final cloze / in-context (`701434be`).
- [x] **v8 ADOPTED 2026-07-16** — measured **79/90 (88%), ALL SEVEN CATEGORIES
  PASS** — the first checkpoint to clear the full 90-probe gate (identity 15/18,
  adversarial 11/12, tool 12/12, restraint 10/12, math 7/8, memory 7/8, factual
  17/20). Lineage: v5 70/90 FAIL -> v6 76/90 FAIL (diet dilution) -> v7 72/90
  FAIL (repetition != coverage) -> v8 PASS (coverage-widened memory/identity +
  moderate fractions, on the facts continued-pretrain base). `models/enigma_dpo/
  model.pth` now holds v8 (SHA256 `A11DB8F0...`); receipted backup at `Enigma
  Backups\enigma_dpo_v8_adopted\` (model+config+vocab .sha256). v5's backup at
  `enigma_dpo_v5_adopted\` is untouched (revert target). Restart serve to load
  v8. Note: v8's memory score includes the corrected October probe; v6/v7 were
  measured under the old March key.
- [x] **Teach-loop auto-augment** — DONE 2026-07-16 (`teach_enigma.py`): each
  `/fix` now `augment_teaching`s the correction into >=3 deduped question
  phrasings + a declarative statement twin (only for simple `what/who/where is
  X`; behavioral corrections get none), then `review_augmentation` shows them
  for accept / edit / skip / cancel before ANY write (confirm-before-bake).
  Non-interactive stdin auto-accepts so scripted teaching still works. Bake
  weight `TEACHINGS_REPEAT` 8 -> 4 (`make_sft_data.py`) now corrections carry
  their own variety. Regression tests in `tests/test_teach_tool.py`.
- [x] Low-quality gate (URLs/HTML/encoding/loops; profanity NOT filtered per
  ruling) — done (`d0dd527e`).
- [x] Knowledge weight x2 -> x5 — done (`43870254`).

## 4. Phase 4.5 — Owned organs (the huge project; ~1-2 weeks GPU)

> Full ordered plan in `ROADMAP.md` Phase 4.5. Retire borrowed backbones one at
> a time; teachers used OFFLINE during distillation only.

- [x] 1. Encoder persistence — DONE `f9ec5184` (see section 1); the blocker
  is dead. Serve-side encoder loading folds into step 5.
- [x] 2. Collect LLaVA-Pretrain 558k — DONE (data staged; see CLAUDE.md
  multimodal state 2026-07-17).
- [x] 3. Distill DINOv2-S -> her own ViT-medium — DONE 2026-07-17
  (`models/enigma_vision_distill/`, val cosine 0.3469; [-1,1] contract
  test-pinned in `tests/test_vision_normalization.py`).
- [x] 4. `train_vision` align on 558k — DONE 2026-07-20 (val 1.4884;
  `models/enigma_vision_align/`). `serve --eyes` boots "eyes: on" and
  captions live. Captions are primitive at 182M — grounding errors and
  greedy loops (the loop is fixed by the caption repetition penalty,
  `d15bc6c`); quality work belongs to the next align cycle
  (VISION_QUALITY_SPEC: bigger student, pixel-shuffle connector, stage-2
  unfreeze), which is gated on the user's image-domain decision.
- [~] 5. Image begin/end tokens (ids 4724+ free), serve wiring, delete BLIP.
  Serve wiring DONE and BLIP deleted 2026-07-17; her own distilled ViT serves
  live under `serve --eyes`. STILL OPEN, both needing the next training cycle:
  (a) the image begin/end tokens are ALLOCATED (`IMAGE_START = 4724`,
  `<|image|>`/`<|/image|>` at 4724/4725 on v1 and 16372/16373 on v2, attached
  per instance by `attach_image_tokens`; 4726-4735 stay reserved) but not yet
  TRAINED -- no corpus or SFT record carries them, so captions still reach the
  model as the "[image: ...]" text marker; (b) captions are question-blind (serve passes
  `EYES.describe` as a bare 1-arg callable, so the pixels are gone before the
  question is asked) even though `model.forward_multimodal` already
  concatenates [vision][text] -- closing it is a stage-2 VQA align plus an
  `Eyes.answer(img, question)` path.
- [~] 6. Her ears: `collect_audio_data.py` DONE, `distill_audio_encoder.py`
  DONE-not-launched (own loop, survived the compression pass). Align
  trainer REBUILT 2026-07-19: `vision_align.py` generalized into
  `enigma_engine/training/encoder_align.py` (one `_train_encoder` core, a
  `_Modality` adapter, `train_vision`/`train_audio` wrappers) so audio
  inherits every hardening + regression pin from day one; `align_audio.py`
  entry point added; 6 audio contract tests re-lock the encoder
  persistence twin. Batched audio LANDED 2026-07-20: the mask-aware
  AudioEncoder makes padded-batch == unbatched at 3.6e-7, so
  `align_audio.py --batch-size 8` is the supported path. The remaining
  steps (teacher download, distill, align, serve wiring, retire whisper)
  are ROADMAP.md Phase 4.5 step 6's -- see there.
- [ ] 7. Her voice: train a small TTS on a chosen voice (later).
- [ ] 8. Her imagination: own image generator (much later; SD stays the tool she wields).

## 5. Interim organ upgrades (still borrowed, better scaffolding; pip-only)

> USER RULING 2026-07-16: voice/sound stays OFF for now ("we will work on it
> later when it matters") -- launchers no longer pass -Voice.
> RULING LIFTED 2026-07-23: voice work resumed by user order; the launchers
> pass `-Voice` again and boot with talk-mode OFF (she starts silent).

- [x] TTS SAPI -> **Kokoro-82M** (~330 MB; near-natural, pure-Python G2P).
  DONE 2026-07-23: `core/tts.py` runs on Kokoro. Synthesis measured at RTF
  ~0.25x on the 5090 (1.31 s of compute for 5.28 s of audio, first run
  including warmup) -- a session measurement with no committed benchmark, so
  re-measure before relying on it. Voices are style tensors that blend by
  weighted sum (`set_recipe` multiplies and sums whatever `load_voice`
  returns; no shape is asserted anywhere); the
  chosen recipe approximates the Cortana character and persists to
  `~/.enigma_engine/voice.json`. The `[voice]` extra installs kokoro +
  soundfile + sounddevice, and the launcher runs the server under the repo
  `venv/` where kokoro lives.
- [ ] ASR whisper-base -> **large-v3-turbo** (~1.6 GB; ~half the errors).
  VERIFY FIRST: CTranslate2 CUDA works on the 5090 (sm_120) — else it silently
  falls back to slow CPU int8. One-line check: `Ears(device="cuda").device`.
- [~] Eyes: **question-conditioned VQA** — captioning throws the user's
  question away; VQA answers what was actually asked. The BLIP half of this
  item is OBSOLETE (BLIP deleted 2026-07-17; her own distilled ViT serves
  today), and the borrowed-model swap is off the table under owned-organs.
  What SURVIVES is the real defect: serve passes `EYES.describe` as a bare
  one-argument callable, so the pixels are gone before the question is asked,
  even though `model.forward_multimodal` already concatenates [vision][text].
  Fix = stage-2 VQA align + `Eyes.answer(img, question)` — scheduled at T7.
- [ ] Image gen sd-turbo -> **sdxl-turbo** — a one-string change in `Painter`
  (already turbo-aware); higher fidelity, fits VRAM easily.
- [x] Offline-by-default privacy (organs load from cache; `--allow-downloads`
  gates the one first fetch) — done (`6d3cf598`).

## 6. Cost & efficiency (keep/raise quality, lower resource use)

- [ ] **Quantization** — `core/gguf.py` export exists. int8 is near-lossless and
  roughly halves her VRAM/footprint; int4 (~4x smaller) with slight quality
  cost lets her run on far weaker hardware and start faster. She's already free
  to run (local, no API), so this is pure headroom, not a cost cut.
  CAVEAT since the 2026-07-24 GGUF-serving rejection: this is the ONLY live
  reason gguf.py still exists, and its qwen3 auto-flip is math-wrong for the
  v1 architecture (norms before rope, missing NEOX permute) — fix that first
  or quantize inside the from-scratch serving path instead.
- [ ] **Load organs on-demand** vs eager-at-boot — keeps idle VRAM low when an
  organ isn't in use; matters once eyes/imagination models get bigger.
- [ ] **min_p-only sampling A/B** — drop top_p+top_k, keep min_p 0.05-0.1; the
  min_p literature says it tolerates higher temperature without rambling. One
  eval run decides. (KEEP verdict stands until measured.)
- [ ] The genuine cost tradeoff to be aware of: a bigger brain (Phase 7,
  350-700M) buys quality but costs more VRAM/time. Data quality is the cheaper
  quality lever at this scale — spend there first.

## 7. Housekeeping / dormant code (low priority, low risk)

- [x] `enigma_engine/core/adaptive_trainer.py` — DELETED 2026-07-17 along with
  `adaptive_prompts.json` and the "adaptive" mode registration
  (schema/registry/dispatch); regression assert in `test_training_dispatch.py`.
- [x] Unused deps `SpeechRecognition` + `sounddevice` — dropped from
  `pyproject.toml` (full + voice extras) 2026-07-17.
- [x] `data/sft/math.jsonl` — deleted 2026-07-17.
- [x] `enigma_engine/core/rl_training.py` guarded caller of the deleted
  `sentiment` module — removed 2026-07-17; `test_import_integrity.py`
  ALLOWED_MISSING is now empty (gate fully strict).
- [x] Config naming — `_load_user_config` now also searches
  `~/.enigma_engine/forge_config.json` (legacy `config.json` kept for
  back-compat) 2026-07-17.
- [x] **Scratch checkpoints PRUNED 2026-07-16** (user-approved in chat): ~500 GB
  freed (1.0T -> 1.5T free) across scratch sft/dpo checkpoints and five
  forgotten April `_pretrain_sequences.jsonl` caches. v8 is the adopted DPO
  (`models/enigma_dpo`, receipted backup `enigma_dpo_v8_adopted\`); kept:
  `enigma_sft`/`enigma_dpo`, `sft_v8`/`dpo_v8`, all pretrain runs, the Qwen
  zoo, smoke/trainv4 fixtures.
- [x] Docs: facts continued-pretrain recipe (training_guide.md Stage 1.5 +
  quick_commands.md rows) + diet collector flags documented; CLAUDE.md
  pipeline line mentions the optional facts hop (2026-07-17).
- [x] Teach tool nits (final audit m3/m4) — DONE 2026-07-16 alongside the
  auto-augment work: `/good` now refuses when the exchange already has a saved
  teaching (no double-write; `/undo` first to change it); `retract` only ever
  SHRINKS a file (guards against NUL-padding a hand-edited jsonl when the
  recorded offset is past the current end). Regression test for the shrink
  guard in `tests/test_teach_tool.py`.
- [x] Memory-read data nit (final audit m9) — FIXED 2026-07-19. Confirmed
  real: `gen_memory_read_examples` drew distractors from every other fact,
  so a block could assert BOTH "favorite color is green" and "likes the
  color orange" while the trained answer named one — the question has two
  valid answers in context, teaching an arbitrary pick instead of
  retrieval. Fix is a general mechanism, not a one-pair patch:
  `_CONFLICTING_FACTS` groups facts that answer the same question and the
  sampler excludes the target's group (add a group when widening with an
  attribute that already has a value). `tests/test_memory_read_data.py`
  pins the contract, that memory_tools inherits it, and — because the
  groups repeat fact strings — that a renamed fact can't silently drop out
  of its group. Both mutation-verified. Takes effect at the next SFT bake;
  the served v8 was trained on the old data.
- [ ] `teachings.jsonl` still the untouched example template — YOUR channel to
  author (values / personal facts); bakes in at x8.

## 7.5 2026-07-19 review — open cleanup/efficiency findings

> From the xhigh compression-pass review (25 verified findings; the
> correctness/latent-bug subset lives in KNOWN_ISSUES #12, the
> checkpoint-safety subset was fixed same day — section 1). All verified
> against the working tree. Efficiency items matter most before the 558k
> align run.

- [x] **Serial PIL decode inside the training step** — FIXED 2026-07-19
  (pre-align batch, round 3): path decodes run on an 8-thread pool with
  prefetch depth 2 while the GPU trains; augmentation stays on the main
  thread in batch order (seeded determinism test-pinned); the `verify()`
  probe pre-pass runs pooled in bounded 512-item chunks; text batches
  build on CPU with one `.to(device)` (train + val). In-memory PIL refs
  decode inline (no pool win; a shared lazy PIL object must not `load()`
  concurrently). Same round: token-weighted val loss; epochs whose metric
  a stop truncated are never ranked while a val pass that COMPLETED
  before the stop still ranks (closes the stop-mid-epoch ranking finding
  both ways); the four lying knob defaults refused;
  `ForgeConfig.from_dict` known-set derived from `dataclasses.fields()`.
  10 new tests; 19-mutation sweep all killed; suite 361 -> 371; audited
  to convergence (4 rounds, severity high -> med -> low -> none).
- [x] **Cleanup batch 2026-07-19 (round 4)** — closed in one audited pass
  (6 new mutation-verified tests, 25-mutation sweep, suite 371 -> 377):
  dead fallback optimizer -> lazy property, old-checkpoint optimizer state
  no longer materialized; `_estimate_batch_size` RETIRED (batch_size >= 1
  required; refusal points at `hardware_detection.
  recommend_training_batch_size`) — also removes the caller-config
  mutation; train/val share `_forward_ce` (drop-policy drift dead);
  `TrainingConfig` slimmed ~25 inert fields with field-derived
  to_dict/from_dict; one-allocation CPU mask; `training/__init__` shim
  and `--no-diff-attn` (+ both .ps1 consumers) removed; MoE/LoRA/
  speculative/MTP/test-prose doc drift grounded; `total_tokens`/
  `dataset_fingerprint` dead checkpoint keys and `_emit_loss`'s dead
  val_loss param removed.
- [ ] **Deliberately DEFERRED (rationale logged 2026-07-19):**
  trainable-subset intermediate best-saves — the best checkpoint is the
  primary resume artifact and must stay full-format; the flagship 1-epoch
  run writes ~one best save total, so the I/O win is small next to the
  resume-compat risk. Revisit only if multi-epoch align runs become the
  norm. Also still open: `_save_checkpoint` stores `config.__dict__`
  instead of canonical `ForgeConfig.to_dict()` and writes dual
  `model_config`+`config` keys (`test_encoder_persistence.py` pins both
  keys — change together; `config` is the live key with 7 readers).

## 7.9 v2 pretrain: measured launch constraints (2026-07-21, on the 5090)

Micro-batch fit search over the deep-thin presets, `--sanity` (one fwd/bwd),
`--no-grad-ckpt`, sdpa cudnn, corpus `tokens_v2.bin`:

| preset | block 2048 | block 8192 |
|---|---|---|
| `v2_deep_186m` | fits, micro-batch 8 | **does not fit at micro-batch 1** |
| `v2_deep_238m` | fits, micro-batch 16 | **does not fit at micro-batch 1** |
| `v2_deep_542m` | fits, micro-batch 8 | **does not fit at micro-batch 1** |

**That table is measured with `--sanity` and is NOT a training-shape receipt.**
`--sanity` runs one fwd/bwd and allocates no optimizer state, so it overstates
what fits: at 186m/2048 it declared micro-batch 8 usable, but the full step
(fwd + bwd + Muon) peaks at 31.68 GB on a 31.84 GB card and thrashes into
shared memory, running **2.6x slower** than micro-batch 6. Measured on the full
step, muon, 186m @ 2048:

| micro-batch | tok/s | peak | days/epoch |
|---|---|---|---|
| 8 | 12,499 | 31.68 GB | 21.9 |
| **6** | **31,894** | 24.21 GB | **8.6** |
| 4 | 27,038 | 16.75 GB | 10.1 |
| 1 | 10,109 | 5.66 GB | 27.1 |

**Full grid, corrected method (full step incl. Muon, non-power-of-2
micro-batches, 23.69B-token corpus). Best non-thrash config per size.**

> **SUPERSEDED 2026-07-28 as the throughput receipt.** Every tok/s below was
> measured while `torch.compile` was falling back to EAGER (triton was not
> importable then -- KNOWN_ISSUES 1(a), now resolved). Re-measured at the real
> launch shapes over 150 steps each, 186m and 238m are ~2.2x faster and 542m is
> unchanged (compile fails on that shape). The live grid is in 7.95; the MFU and
> peak-VRAM columns here still stand as an eager-path reference.

| preset | config | tok/s | MFU | peak | days/epoch |
|---|---|---|---|---|---|
| `v2_deep_186m` | block 2048, mb 6, no ckpt | 32,639 | 17.4% | 24.2 GB | **8.4** |
| `v2_deep_238m` | block 2048, mb 6, no ckpt | 31,311 | 21.4% | 23.8 GB | **8.8** |
| `v2_deep_542m` | block 2048, mb 16, **ckpt** | 14,294 | 22.2% | 15.3 GB | **19.2** |

- **238m costs only 5% more wall-clock than 186m for 28% more parameters**, and reaches
  higher MFU (21.4% vs 17.4%) because 20L@1024 has better arithmetic intensity than the
  launch-bound 28L@768.
- 542m must use checkpointing: its no-ckpt rows fall off the cliff catastrophically
  (mb 5/6/7 -> 845 / 578 / 503 tok/s at 37-50 GB "allocated", i.e. 325-545 days/epoch).
  With ckpt at mb 16 it is 14,294 tok/s and well-behaved.
- Block 8192 costs 35-40% throughput at every size and buys context the tokenizer already
  provides; 2048 stands.
- mb 7 measures faster than mb 6 at 186m/238m but peaks at 28.1 GB, leaving under 4 GB for
  the val batches, the corpus memmap and allocator fragmentation the synthetic probe does
  not carry. mb 6 is the recommended launch value.

Method rules this produced, for any future capacity search:
- Measure the FULL step including the optimizer, never `--sanity` alone.
- Sweep non-powers-of-two: a halving search cannot land on 6.
- Treat any config peaking above ~85% of VRAM as unusable however fast it
  looks in a fwd/bwd-only probe -- the allocator spills silently and the run
  merely gets slow, with no error to notice.
- Layer isolation on the same shape: forward alone reaches 94.6% MFU, so the
  model is not the problem; throughput collapses only as memory is added
  (fwd+bwd 11.6%, +adamw 5.3%, +muon 1.9% at the thrashing micro-batch).

- **Activation checkpointing is MANDATORY at block 8192** and the no-ckpt
  advice (SUGGESTIONS / the v2 research verdicts) holds only at 2048. Weights,
  grads and optimizer state are just 1.39 / 1.78 / 4.04 GB for the three
  presets -- activations dominate and scale with seq_len x layers.
- **Block 2048 is the recommended launch shape**: under the v2 tokenizer it
  carries ~1444 words (~4,945 v1-token-equivalents), inside the researched
  4k-8k target band, at full speed. Block 8192 overshoots the band (~19.8k
  equivalents) and pays 30-40% for the checkpointing it would then require.
- `--block` defaults to **1024**: a launch that omits it trains at 1024 no
  matter what the preset's `max_seq_len` says.

### Launch commands, BRANCHED on the size call

The flags are not shared across sizes. `--no-grad-ckpt` is right for 186m/238m
and catastrophic for 542m (the cliff above: 503-845 tok/s, 325-545 days/epoch),
and 542m's micro-batch is 16, not 6. Copying one size's line to the other
produces a run roughly 40x slower with nothing in the output to say so. Flag
surface verified against `pretrain_enigma.py --help` 2026-07-23.

THE LIVE CORPUS IS `tokens_v2b.bin` since 2026-07-28 (T1 executed:
28,261,718,460 tokens, 52.64 GiB (56.52 GB) uint16, curated x5 walked FIRST, DCLM/
FineMath/Stack in, C4+OWT out; sidecar `tokens_v2b.json` carries the
per-source extents and the scrub count; both files attrib +R, and the old
`tokens_v2.bin` stays on disk as the receipted rollback). The tok/s rates
below are per-step measurements and carry over; days/epoch are re-derived
at 28.26B tokens. **RE-MEASURED 2026-07-28 at the real launch shapes over 150
steps each, and the grid MOVED: 186m 4.1 / 238m 4.9 / 542m 20.8 d/epoch**
(the old 10.0/10.5/22.9 came from ~40-step probes running EAGER).

| preset | shape | tok/s | peak VRAM | compile | d/epoch |
|---|---|---|---|---|---|
| `v2_deep_186m` | mb8 ga16, no-ckpt | ~80,500 | 26.2 GB | ON | **4.1** |
| `v2_deep_238m` | mb6 ga16, no-ckpt | ~67,400 | 24.5 GB | ON | **4.9** |
| `v2_deep_542m` | mb16 ga16, ckpt ON | ~15,700 | 20.7 GB | **FAILS** | **20.8** |

`torch.compile` is worth ~2.2x on the two smaller shapes and is UNAVAILABLE at
542m -- it raises "No valid triton configs / out of resource" and degrades to
eager, which is why 542m alone matches its old number. That asymmetry is the
Gate B headline: the 238m-vs-542m cost ratio is **4.3x**, not the 2.18x the old
grid implied. Steady-state over 150 steps, not a multi-day thermal receipt.

238m -- wall-clock optimal (4.9 days/epoch on v2b):

    python pretrain_enigma.py --size v2_deep_238m --optimizer muon \
      --schedule wsd_sqrt --sdpa-backend cudnn --no-grad-ckpt \
      --block 2048 --micro-batch 6 --tokens 28.3e9 --lr 3e-3 \
      --tokens-bin data/pretrain/tokens_v2b.bin \
      --val-general-end 15055680259 --val-per-source 2000000 \
      --out models/enigma_v2_238m --seed <N> --archive-every 1440

542m -- largest the 5090 sanely trains (20.8 days/epoch on v2b). Note BOTH
changes: drop `--no-grad-ckpt` (checkpointing is mandatory here) and raise
the micro-batch to 16:

    python pretrain_enigma.py --size v2_deep_542m --optimizer muon \
      --schedule wsd_sqrt --sdpa-backend cudnn \
      --block 2048 --micro-batch 16 --tokens 28.3e9 --lr 3e-3 \
      --tokens-bin data/pretrain/tokens_v2b.bin \
      --val-general-end 15055680259 --val-per-source 2000000 \
      --out models/enigma_v2_542m --seed <N> --archive-every 540

`--lr 3e-3` is the T2 measurement (interior win at 238m, 2026-07-29), and it
is passed EXPLICITLY because the flag still defaults to the v1 lineage's 6e-4
until the Gate D sunset. **542m inherits 3e-3 unmeasured on two axes** -- width
(the Muon scale-invariance argument) and batch: at mb16 ga16 it trains
524,288 tok/step against the swept 238m's 196,608, a 2.67x larger batch. Pinning
`--grad-accum 6` at mb16 matches the swept tok/step and makes a 2-point
spot-check cost ~11.8 h instead of a ~35 h full re-sweep.

`--archive-every` derivation (post-hoc EMA needs ~10 archives inside the decay
tail; `--wsd-decay-frac` is 0.10, one archive is a full save with optimizer
state -- 1.9747 GB measured at 238m, `models\sweeps\t2_238m\*\model.pth`):

| preset | tok/step | steps @28.26B | decay tail | archive-every | per-archive | uniform bill |
|---|---|---|---|---|---|---|
| `v2_deep_238m` | 196,608 | 143,747 | 14,375 | **1440** | 1.97 GB | ~197 GB |
| `v2_deep_542m` | 524,288 | 53,905 | 5,390 | **540** | ~4.49 GB (scaled) | ~450 GB |

The bill is the cost of a UNIFORM cadence keeping ~100 archives to get the ~10
that matter. A tail-only archive gate -- archive only once
`step >= anneal_first_step(total_steps, wsd_decay_frac)` -- cuts it ~90%.

RESUMING EITHER OF THESE: `python pretrain_enigma.py --resume <ckpt>` and
nothing else. The checkpoint carries the schedule (`SCHEDULE_KEYS`), including
`--no-grad-ckpt` since 2026-07-28, so re-passing the launch line can only
contradict it. Do NOT re-pass `--seed` on a resume: it re-seeds the sampler and
replays the windows the run already trained on (the boot warns).
`resume_training.ps1` takes `-Run <model-dir>` and passes `--resume` plus only
what the checkpoint does NOT record: `-TokensBin <path>` (required when the
schedule has no corpus -- otherwise the run would train `--tokens-bin`'s
default) and `-NoGradCkpt` (opt in to disabling activation checkpointing;
omitted, checkpointing stays ON, which is slower but cannot OOM a lineage sized
for it). The archive cadence is NOT settable there: it is schedule-restored, so
a value passed on a resume would be ignored -- change it with a hand-run using
`--override-schedule` and the full launch line. EVERY checkpoint written before
2026-07-28 lacks both keys, so the script refuses them until `-TokensBin` names
the corpus.
CHECKPOINTS WRITTEN BEFORE 2026-07-28 (every lineage on disk today) have no
`no_grad_ckpt` in their schedule. Boot names them -- "this checkpoint predates
... no_grad_ckpt" -- and falls back to the CLI/default, which re-enables
checkpointing. Re-pass `--no-grad-ckpt` when resuming one of those.

- `--tokens` defaults to **2e9** — one fourteenth of the v2b corpus.
  Omitting it trains 7.1% of an epoch (~8.2 h at the re-measured 238m rate)
  and, because `total_steps` is derived from it, places the WSD decay tail
  there too: the run ends, looks finished, and is nowhere near the 10.5
  days/epoch this section quotes. Pass the token budget EXPLICITLY.
- `--tokens-bin` defaults to the v1 `tokens.bin`: pass the v2 corpus
  EXPLICITLY or the run trains the new architecture on the old tokenization.
- Size `--archive-every` so the decay tail leaves ~10 archives; post-hoc EMA
  has nothing to average otherwise, and an EMA checkpoint is `--init-from`
  only (it carries no optimizer state, so `--resume` refuses it).
- There is NO rope-theta flag: theta comes from the preset (all three
  `v2_deep_*` carry 500000). Changing it means editing `model_presets.py`,
  which is a lineage decision, not a launch knob.

## 7.95 THE TRAINING BLOCK — everything that trains, deferred to the end, in order

> RULED 2026-07-24: anything that trains the AI waits as long as possible and
> runs as one consolidated block. Local 5090 only (rental rejected same day).
> Non-training prerequisites run first, whenever convenient.

Prerequisites (not training):
- P1. Seal the locked probes (validate, seal manifest, record drop counts +
  shas + scorecards in EVAL_REDESIGN). **DONE 2026-07-24** — 108 strings
  (96 q + 12 teach), manifest `data/eval/locked_probes.manifest.json`,
  receipts in EVAL_REDESIGN, durable copy in `Enigma Backups`. eval_behavior
  now re-seals the file at run start and refuses a holdout that was edited.
- P2. v5/v8 locked re-measure (`--port 8123`, throwaway `--memory-dir`,
  `--transcript` OUTSIDE the repo) = the baseline v2 must beat. Ran before
  any vocab adoption, as required.
    - **RE-BASED 2026-07-27 under reseal #7 (120 probes / 15 per gated
      category / manifest 4e4d4433; EVAL_REDESIGN owns the full table +
      seal receipts):** v8 = **56/120 (47%)**, v5 = **55/120 (46%)**,
      attributed transcripts in Enigma Backups. THE BASELINE v2 MUST BEAT
      IS 56/120 with no gated-floor regression (`eval_behavior --baseline`
      prints the verdict; the v5 run carried the first live comparison:
      USER'S CALL, regressions adversarial/factual/restraint). The widened
      columns see what n=12 could not: restraint's weather-ADJACENT rows
      expose the false-fire defect (v8 83%->67% -- it calls get_weather on
      "Winter in my hometown was brutal"), and v5 answered the
      neighbour's-name unknown probe with "Marisol Quenby" -- the invented
      teach name from minutes earlier, the memory-bleed mechanism proven
      with a token that did not exist before this reseal. unknown is 0/30
      across both: epistemics stay the v2 recipe's biggest win condition.
      (The 96-probe 47/96 columns are superseded; diff trail in
      EVAL_REDESIGN.)
    - The harness CLEARS the target server's memory store before probing and
      then writes probe facts into it, so it refuses any target off the
      scratch port unless `--allow-live-server` is passed. Never point it at
      the daily server on 8000.
    - Both checkpoints live in `Enigma Backups\enigma_dpo_v5_adopted\` and
      `…v8_adopted\` with sha receipts; serve them from there, not from
      `models/`, so the live checkpoint is never in play.

The block, in execution order:
> **T1 RULINGS (user, 2026-07-27, all four made in session):**
> 1. **Corpus = FULL SPEC.** All three absent sources come in: FineMath
>    (worked math -- the remake charter's named gap), The Stack (code), and
>    the DCLM quality swap. The collectors must RUN (multi-day downloads);
>    training waits for the corpus, not the reverse. The user chose the
>    thorough road over the fast one, explicitly.
> 2. **Doc-boundary attention masking: SKIPPED for v2.** Meta's own measure
>    says negligible at short blocks; not worth new hot-path mask plumbing
>    in front of the measured throughput right before a days-long run.
>    REQUIRED at the future length-extension anneal (charter amended).
> 3. **Curated repeat: x5** (--repeat-sources curated=5). The proven facts
>    multiplier; ~8.6M tokens; boot guards verify placement + pass size.
> 4. **Vocab window CONFIRMED, all three:** keep <search>/</search> rows;
>    delete the 31 tag-teaching SFT records at T4; ALLOCATE the image
>    begin/end delimiter rows in this retokenize so the vision path never
>    needs vocab surgery. This is the last vocab window -- after this,
>    rows are set for the lineage.
- T1. Corpus prep (~1 day of tokenize work AFTER the collector downloads;
  rulings above): quality-score the raw third
  (edu-classifier or DCLM swap), add code+math (FineMath collector was never
  run), add short conversational register (chat was left entirely to SFT
  last lineage), 5-10 paraphrase variants of every must-know fact IN the
  corpus, decay-tail annealing set (~2-3B best tokens); then the rust
  retokenize (~42 min). Doc-boundary masking: RULED 2026-07-27, skipped for
  v2 (required at the length-extension anneal -- ruling 2 above).
    - **Curated shard BUILT 2026-07-25 -- and it must be REBUILT before the
      retokenize**: the shard was screened under the floor-3 manifest
      (built 14:03, floor-2 reseal f9814b8f landed 19:46 the same day), so
      its screening predates the seal it must satisfy. `make_pretrain_curated.py`
      reads the LIVE manifest at load, so a plain re-run is the fix.
      `tokens_v2.bin` PREDATES the shard entirely (tokenized 2026-07-20), so
      T1 must re-tokenize with `--repeat-sources curated=N` before T2 -- the
      existing bin contains none of the shard.
    - ~~**The anneal needs a MECHANISM, not just a token set.**~~ **BUILT
      2026-07-25.** `get_batch` drew uniformly over `[0, train_end)`, so
      position in the corpus meant nothing and a "best tokens at the end" file
      would have changed nothing. The sampler now knows a phase:
      `--anneal-tokens N` names the last N tokens of the train stream as the
      CURATED region and `--anneal-frac F` (default 0.5) sets how much of each
      micro-batch is drawn from it once the WSD decay phase begins. Both are
      recorded in the checkpoint `schedule`, because they change WHAT the tail
      sees — a resume that dropped them would finish on a different diet than
      it started. OFF by default (`--anneal-tokens 0`), so the sampler is the
      same single uniform draw the live lineage used.
      **Tail-position is DEAD (round-7 audit + fix + fix-arc audit, all
      2026-07-25):** val is carved off the very END of the bin, so "write the
      curated shard LAST" would have handed the shard to val -- 100% held
      out, never trained on. The curated oversample is now
      `pretokenize_data.py --repeat-sources curated=N` (repetition AFTER the
      global paragraph dedup, which silently collapses on-disk copies;
      passes are byte-identical). The fix-arc audit then broke the first fix
      four more ways, all closed: the curated source now walks FIRST (dedup
      is first-wins, so mid-list it silently LOST every paragraph a web
      source shared, and "not last" was one absent stackexchange dir from
      last on a fresh checkout); a declared oversample that emits zero
      tokens refuses at tokenize time (empty dir / fully-deduped shard had
      produced a stream byte-identical to no-shard-at-all under a meta
      claiming x5); pretrain boot-refuses ANY source lying entirely in val,
      a repeated source overlapping val or the fenced val-gen window, and a
      per-pass span <= block (adjacent copies in one window); the v1
      lineage tokens.bin is now write-protected outright -- and so is its
      SIDECAR (round-B: `--output-bin tokens.bin2` mapped its .json onto the
      gitignored lineage receipt); the repeat cache refuses web-scale
      sources (2 GB cap), and the refusal is a plain Exception because a
      SystemExit inside the doc stream was swallowed by pool.imap's feeder
      thread and HUNG the run under --workers > 1 (round-B, measured). A
      too-small `--anneal-tokens` (<= block+1) refuses at boot instead of
      dying at the decay boundary days in. The anneal mechanism itself stays built+OFF, waiting for a
      deliberately PLACED region (e.g. a length-extension anneal), not the
      T1 shard.
    - **The pretrain-path screen lives at COLLECTION time (landed
      2026-07-27)**: `pretokenize_data.py` itself has no leak screen by
      design (it reads whatever is on disk), so the collectors screen every
      saved record against the sealed manifest (`_locked_probe_guard` in
      `collect_pretraining_data.py`, drop counts printed per source) and
      space-break exact special-token literals so code text cannot carve
      real control ids (`_sanitize_special_literals`; "List<A>" carved id 7,
      a literal "</s>" wrote EOS mid-document -- measured). Text collected
      BEFORE that date (wiki/fineweb/c4/owt/books) was never screened.
    - Adding sources means editing the hardcoded `SOURCE_DIRS` in
      `pretokenize_data.py`; the v2 retokenize invocation is

          python pretokenize_data.py --vocab enigma_engine/vocab_model/bpe_vocab_v2_16k.json \
            --output-bin data/pretrain/tokens_v2b.bin --dtype uint16 --workers 10 \
            --repeat-sources curated=5

      PATHS IN FULL, always: a bare `--vocab bpe_vocab_v2_16k.json` from the
      repo root named a nonexistent file, and get_tokenizer fell back to an
      untrained char-level tokenizer whose ids all pass the bounds guard --
      the run would have COMPLETED and written a garbage corpus (measured
      2026-07-27; pretokenize now refuses a missing vocab path and a
      merges-free tokenizer outright). Note `tokens_v2.bin` + sidecar are
      attrib +R until the output-naming ruling: overwrite-in-place would
      destroy the 7.9 grid's substrate and its receipt.
    - **T1 changes the corpus size, so the 7.9 grid moves with it**: the
      8.4/8.8/19.2 days-per-epoch figures and the `--tokens` budget are
      derived from 23.69B tokens. Re-derive both before the size call, or
      Gate B is decided on stale arithmetic.
    - `<search>`/`</search>` (v2 vocab rows 12/13): RULED 2026-07-27
      (ruling 4 above) -- KEEP the rows, delete the 31 tag-teaching SFT
      records at T4, and ALLOCATE the image begin/end delimiters in this
      same retokenize window. HOW to allocate the image rows is still open
      (see OPEN USER DECISIONS): a naive append to 16,368 breaks the
      16,384 = table+reserve geometry (TOKENIZER_V2_SPEC) and the
      test_vocab_selection pins; the candidates are the 18-row reserve
      (zero surgery, v1's own chat-token pattern) or trimming the 2 lowest
      merges (table stays 16,366; clean only while no v2 checkpoint exists).
- T2. **DONE 2026-07-29 -- ran as the SWEEP FORM** (detached 2026-07-28
  21:16; receipt `models\sweeps\t2_238m\sweep_results.json`). Both T2
  questions answered: the lineage LEARNS (six independent runs, rc=0,
  finite finals at ~3.15 -- the trainer refuses the final save otherwise;
  the descent from the ln(16366)=9.703 start was watched live but is
  UNRECEIPTED: sweep_lr.py keeps child stdout in memory only, no per-step
  log survives -- tee it next sweep) and the LR is MEASURED: **3e-3 wins
  INSIDE the bracket**, 6/6 points rc=0, 8.5 h wall:

      lr       s0      s1      seed spread
      1.5e-3   3.1743  3.1732  0.0011
      3e-3     3.1544  3.1544  0.0000 (at 4dp)
      6e-3     3.1554  3.1641  0.0087

  -- an interior minimum, not an endpoint. 3e-3 beats 1.5e-3 by 0.019
  (~17x the low-LR seed noise); 6e-3's mean is 0.0054 worse AND its seed
  spread is ~8x the other rungs' -- variance growth at the top of the
  bracket is itself an instability signal, so 3e-3 is the pick both on
  loss and on stability margin for a run 85x longer than a sweep point.
  ~~Caveat: rank resolution is 1e-4 (the `[final]` print is 4dp --
  pre-flight raises it).~~ **RE-SCORED AND CONFIRMED 2026-07-30** (the
  score-first hold on the sweep dirs, discharged). All six checkpoints
  re-measured at 6dp with `--eval-only`, paired batches (seed 1234, so
  every point scores the SAME windows in the same order), 200 iters per
  window, on the new 30-window per-source holdout. Receipts:
  `Enigma Backups\t2_sweep_rescore_2026-07-30.json` + the six
  `t2_rescore_*.log`.

  | rung | val (tail) | val-gen | val-src (30 srcs) |
  |---|---|---|---|
  | 1.5e-3 | 3.500191 | 3.174746 | 2.675088 |
  | **3e-3** | **3.485606** | **3.156487** | **2.660042** |
  | 6e-3 | 3.488551 | 3.161336 | 2.664861 |

  **3e-3 wins all three signals, and wins 30/30 individual source
  windows** -- including the code/math/StackExchange windows no earlier
  val could see (The Stack 1.097, FineMath 2.280 at 3e-3_s0). It also has
  the tightest seed spread on all three (val-src sd 0.0011 vs 6e-3's
  0.0066), so the instability read at the bracket top holds. The one
  genuine disagreement the 4dp print had hidden is now resolved: at seed
  0 alone, tail-val does prefer 6e-3 (3.486331 vs 3.486709, a 3.8e-4
  margin) -- but that flips on the seed mean and is contradicted by every
  other signal, so it was seed noise, not a window-dependent winner.
  **LR = 3e-3 is settled for T3; no further sweep.**
  **THERMAL RECEIPT:** tok/s 67.5-69.9k across all
  six points over 8.5 h -- no sustained-load fade.
  The archive-cadence shakeout did NOT happen (sweep points never
  archive) -- verify the first decay-tail archives live, early in T3.
  Original entry, kept as the record of the options weighed:
  10k-step probe pretrain (hours): first val-loss receipt for the v2
  lineage + live shakeout of archive cadence. The only v2 runs on disk are
  throughput probes. Same flags as T3 at the chosen size, with the budget cut
  down — the defaults are the trap here too:

        python pretrain_enigma.py --size v2_deep_238m --optimizer muon \
          --schedule wsd_sqrt --sdpa-backend cudnn --no-grad-ckpt \
          --block 2048 --micro-batch 6 --tokens 2e9 --lr 3e-3 \
          --tokens-bin data/pretrain/tokens_v2b.bin \
          --val-general-end 15055680259 \
          --out models/enigma_v2_probe --seed 1 --archive-every 100

  ~2B tokens at the re-measured 67,400 tok/s is **~8.2 h** (the eager-era
  31,311 gave ~17 h; compile is worth ~2.2x at this size).
  THREE flags that are NOT optional here, each a default that silently does
  the wrong thing:
  * `--lr` defaults to **6e-4**, the v1 AdamW lineage peak. No v2 learning
    rate has ever been MEASURED. The 3e-3 below is not a result: it is the
    value typed into a 40-step throughput probe (`probe_tput_v2_deep_186m`,
    5.24M tokens, warmup 5, a different size on the v2 corpus) that never
    recorded a loss. It is here only because it sits inside `sweep_lr.py`'s
    own 1e-3..3e-3 bracket while the default sits below it. Treat it as a
    placeholder and prefer the sweep. Better still, spend this same 8.2 h on the sweep
    instead of one blind point (see the sweep note below).
  * `--val-general-end` — without it val is the corpus TAIL, and on v2b the
    tail is `SE/worldbuilding` alone (30.7 M tokens). Every `[val]` number
    would be speculative-fiction loss. 15,055,680,259 is the end of
    FineWeb-Edu. The boot now names the val split and warns when one source
    owns >=90% of it.
  * `--archive-every 100`, not 1000. T2 is 10,172 steps, so the wsd decay tail
    starts at step 9,154 and is 1,018 steps long: at 1000 exactly ONE archive
    (step 10,000) lands in the tail and post-hoc EMA has nothing to average.
  DECISION RULE: the probe answers "is this lineage learning at all", and it
  answers it INTRA-RUN — loss starts near ln(16366)=9.703, falls smoothly, no
  NaN. **READ `[val-gen]`, NOT `[val]`.** `--val-general-end` does not fix
  `[val]`; it ADDS a second window. `[val]` remains 100% `SE/worldbuilding`
  (the corpus tail) with the flag on, and boot says so. `[val-gen]` is the
  fenced general-domain window, and `sweep_lr` ranks on it when it exists. It is NOT comparable to the v1 lineage's curve in either direction:
  a different vocab makes bits/token a different unit (`pretrain_enigma.py`
  says so at the bits/token print), and v1 was a different corpus besides. If
  the sweep form is used, rank the points against EACH OTHER post-decay. A
  flat or rising curve stops the launch and sends the corpus back to T1.

  SWEEP FORM (same 8.2 h, six receipts instead of one — 6 points x 333M tok
  = 2.00B; each point gets its own decay tail and is judged post-decay):

        python sweep_lr.py --size v2_deep_238m --block 2048 --micro-batch 6 \
          --grad-accum 16 --tokens 333000000 --lrs 1.5e-3,3e-3,6e-3 \
          --seeds 0,1 --warmup 150 --optimizer muon --schedule wsd_sqrt \
          --sdpa-backend cudnn --no-grad-ckpt \
          --tokens-bin data/pretrain/tokens_v2b.bin \
          --out-root models/sweeps/t2_238m \
          --extra "--val-general-end 15055680259"

  (`sweep_lr.py` has no `--val-general-end` of its own; `--extra` is inserted
  before each point's own flags. Every flag above verified against the live
  argparse 2026-07-28.)

  **BRACKET RULED 2026-07-28: 1.5e-3 / 3e-3 / 6e-3**, raised from
  6e-4/1.5e-3/3e-3. `optim.py` scales each Muon step by
  `0.2*sqrt(max(p.shape))` -- about 6.4x on a 1024-wide attention matrix and
  ~10x on the SwiGLU projections. The old low end (6e-4, effective ~4e-3) sat
  under the band Muon is normally trained in and the old high end (3e-3,
  effective ~2e-2) sat at its top, so a win at 3e-3 would only have shown "at
  least 3e-3". The new bracket straddles the band instead of bounding it from
  below. 1,693 steps/point still clears the horizon guard (10 x warmup 150).

  RUN CONDITION: this holds the whole 5090 for ~8.2 h at ~96% and ~24.6 GB, so
  it is a wait-until-the-machine-is-free job -- and it must be the ONLY sweep
  running. Two copies write the same `--out-root` and clobber each other's
  `sweep_results.json`; that happened 2026-07-28 and neither run survived.
- T3. Full v2 pretrain (5090; size = user's call at launch -- on the v2b
  corpus 238m 4.9 d/epoch or 542m 20.8 d/epoch; commands above, flags
  BRANCH on size). `--lr 3e-3` is now MEASURED (the T2 sweep winner), no
  longer a placeholder -- measured at 238m; a 542m launch inherits it
  unmeasured (item 7).
  **T3 PRE-FLIGHT (before any launch; ~an hour of code+drill total):**
  1. ~~add `save_every` to SCHEDULE_KEYS + pin it~~ DONE 2026-07-30, and
     PROVEN by the drill below: the resumed boot printed
     `schedule[save_every] = 20 from checkpoint (CLI 250 ignored)`;
  2. ~~the RESUME DRILL~~ **RUN 2026-07-30, and it found a real launch
     defect.** Method: `--size tiny --block 256` for 920 steps carrying
     six DELIBERATELY non-default schedule values (`save_every 20`,
     `val_per_source 300000`, `archive_every 40`, `--no-grad-ckpt`,
     `lr 3e-3`, `grad_accum 2`), killed on a checkpoint, resumed through
     `resume_training.ps1 -Run drill_resume -TokensBin ...` under Task
     Scheduler. Both segments detached (no `claude.exe` in either chain).
     **RESULT: all 22 schedule keys restored**, each printing the CLI
     default it overrode -- including the two new keys and the
     `no_grad_ckpt` trap. Resume continued at step 920 with the restored
     20/40 save and archive cadences.
     **THE DEFECT (fixed, `8dd9f230`):** `cmd.exe` strips the outermost
     quote pair from the line after `/c` when it holds several quoted
     tokens, and both the interpreter path and this repo's path contain
     spaces -- so the script launched NOTHING, wrote no log, and said so
     only as a yellow note while exiting 0. Under Task Scheduler that
     reports a dead resume as a successful one. The command now carries
     an extra enclosing pair and a failed launch exits 1;
     `test_detached_launchers_survive_cmd_quote_stripping` pins it
     (mutation-verified against the old form). **A T3 resume would have
     hit this on the first try.**
  3. ~~raise the `[final]` val prints to 6 decimals~~ DONE 2026-07-30:
     `[val]`/`[val-gen]`/`[val-src]`/`[final]` all print 6dp;
  4. ~~drop c4/owt from `--all-sources`~~ DONE 2026-07-30 (explicit
     `--c4`/`--openwebtext` still work);
  5. item 11 went REBUILD (ruled 2026-07-30) -- it runs FIRST, per the
     runbook at item 11; the T2-swept LR carries to the rebuilt corpus
     (same vocab, similar mix; accepted 2026-07-29, not to be
     re-litigated mid-run).
- T4. SFT regen riding the bake (data work, minutes-hours of GPU):
  multi-turn, <think> reasoning, math re-enabled (per-digit vocab kills the
  old disable reason), widened knowledge, DPO pairs beyond identity, the
  contradiction/correction shape, teachings.jsonl dual-routed into the facts
  stream, identity anchors rewritten to actual capabilities, the
  ALWAYS-OFFERED built-in block (router gates retire here, ruled
  2026-07-24; the block is part of the STANDARD recipe -- any new AI the
  trainer molds inherits it, ruled same day; the block is now FIVE built-ins
  -- calculate, remember, forget, speak, imagine -- forget added
  2026-07-24). The MEASURED defects behind the retirement, since this entry
  is their one owner (2026-07-26 audit restored them after a doc trim
  orphaned the receipts): the per-built-in gates failed BOTH directions --
  missed asks ("Draw me a dragon" never offered `imagine`; word-number math
  like "seven times eight" misses `calculate`'s digit shape), and a missed
  ask means no offer, so no gradient at training time and no eval signal at
  gate time; false fires on negated asks ("Do not draw anything" armed the
  painter); and the eyes flatten images to `[image: ...]` TEXT before the
  gates read the turn, so her own caption can fire the painter -- a hazard
  that is LIVE today on the gate-mediated v8, not just v2 history. Also in
  the regen: the client-system "Available tools:" shape, image turns
  carrying system/tools/memory blocks, URL-bearing records now kept,
  trained-tool-name list pruned to what has a runtime, `--vocab`/`--block`
  passed explicitly, finetune `--block` raised. Also here per VISION.md
  destination 4: the deep-research organ's data shapes ride THIS regen if
  the organ lands by then -- the 31 orphan `<search>` tag records deleted at
  T4 (T1 ruling 4) get replaced by records whose tags have a runtime owner,
  or research waits for the next regen; no tag trains unowned either way.
  (Privacy scope ruled 2026-07-29 -- canonical text lives in ROADMAP's
  binding-rulings block, ONE home. Consequence here: privacy-preserving
  lookup, local first, is permitted, so the research organ is unblocked in
  principle. T4 REGEN FLAG from the same audit: identity_anchors.py:261
  trains the ABSOLUTE claim "nothing leaves this machine" -- rescope that
  anchor at the regen or she asserts a falsehood the day lookup ships.)
- T5. DPO/polish pass (safe recipe: lr 5e-7 x 1 epoch). Regenerate
  `dpo_pairs.jsonl` as part of this: both trainers now refuse an artifact
  carrying a sealed probe, so a stale file stops the run rather than rigging
  the gate.
- T6. Gate: locked eval vs the P2 baseline -> beat aggregate with no
  category floor regression = adopt; ambiguous = user's call (Gate D in the
  decisions ledger). The rule is CODE now, not an eyeball pass:
  `python eval_behavior.py --probes data/eval/locked_probes.jsonl
  --transcript <new> --baseline <P2 transcript>` prints the verdict, names
  every category regression, refuses a baseline from a different probe set,
  and warns when both transcripts report the SAME checkpoint sha (the
  forgot-to-restart shape). Transcripts record which WEIGHTS answered
  (serve exposes path/sha256/step via /v1/models since 2026-07-27) -- an
  unattributed transcript says so out loud. Adoption is more than a verdict,
  and the rest of it is below:
    - There is **no "vocab default" to flip** — serve and finetune both pick
      the vocab from the checkpoint's `vocab_size`
      (`serve_enigma.py` / `finetune_enigma.py`), which is exactly why v1 and
      v2 models coexist in one checkout. Nothing to switch; delete this from
      the mental model.
    - `Start-Enigma.ps1` hardcodes `models\enigma_dpo\model.pth`. Adoption
      means backing up the v8 checkpoint (receipted, alongside v5/v8 in
      `Enigma Backups`) and putting the v2 export at that path, or editing
      the launcher. Decide which, in writing, before the swap.
    - serve's `--max-context` still defaults to **1024**. A model trained at
      block 2048 and served at 1024 silently throws away the context win the
      whole lineage was for — raise it at adoption and re-check VRAM.
    - **The pretrain defaults re-point to the ADOPTED lineage at this gate**
      (ruled 2026-07-29 -- the SUNSET of the never-change-defaults
      guardrail, following the vocab-contract adoption-flip pattern): flip
      the `--optimizer/--schedule/--block/--tokens-bin/--val-general-end/
      --lr` defaults to the adopted v2 values in the same commit that swaps
      the checkpoint. AUDIT CORRECTION 2026-07-29 (re-corrected same day:
      the bit-identical CONTRACT is test-encoded --
      test_pretrain_arsenal.py:36 pins the cosine/adamw math to the live
      run -- what no test pins is the argparse default SELECTION, so
      flipping the defaults today would break ZERO tests), so FIRST add a
      source-scan test in the
      test_pretrain_seed.py:54 pattern pinning all six defaults, THEN flip
      it at adoption. Retire ONLY the warnings for flipped flags (`--lr`,
      `--val-general-end`, `--tokens-bin`, `--block`); the `--tokens` and
      `--archive-every` warnings cover flags NOT in the flip list and STAY.
      (The CLAUDE.md guardrail edit recording this sunset was
      classifier-blocked 2026-07-29; this bullet is the operative record
      until it can land. The paste package handed to the user had only the
      privacy-scope and one-hot-job lines -- audit caught the gap. THIRD
      LINE, for CLAUDE.md:109, append to the do-not-change-defaults rule:
      "-- SUNSETS at Gate D adoption (BACKLOG 2026-07-29): the defaults
      flip to the adopted v2 values in the checkpoint-swap commit,
      test-pinned first.")
    - **Run the OFFLINE SECOND GRADER on the T6 transcript** (`tool_graded`
      unblocks it for post-07-28 transcripts): it re-grades from the
      transcript alone and neutralizes the unknown-category gaming risk
      (item 13) without needing a content reseal.
    - Gate statistics: the sealed set is **15 probes per category** since
      reseal #7, meeting EVAL_REDESIGN's ">= 15" rule (one flip = 6.67%).
      Two things follow that the aggregate alone hides:
      * **A one-probe aggregate lead is not a result.** v8 56/120 and v5
        55/120 are a paired exact p = 1.00 (v8 won 10, v5 won 9, 19
        disagreements). `eval_behavior --baseline` now prints the paired
        split and its exact p, and stamps INDISTINGUISHABLE on the verdict
        when p >= 0.05. A real win is a net gain of roughly 11-15 probes (11 at 19
        disagreements, 12 at 30, 14 at 40).
      * **The comparator and the gate disagree by 31 probes.** `--baseline`
        adopts at 57/120; the absolute category floors that set the exit code
        need **88/120** (12/15 on the six 0.80-and-0.75 categories, 8/15 on
        factual and unknown). Both baselines already exit FAIL. Expect a
        plausible v2 to print ADOPT and exit FAIL in the same run, and read
        the two lines as answering different questions.
      * At n=15 the 0.75 and 0.80 thresholds are the SAME bar (both round to
        12/15), so `math` and `memory` are declared at 75% and gated at 80%.
        At the old n=12 they genuinely differed (9/12 vs 10/12).
- T7. Post-adoption organ training, in order: vision stage-2 VQA
  (needs the image-domain pick), ears distill + align (needs the
  whisper-base teacher download), then the far-future own TTS / own
  image-gen transplants.

### Non-training queue (GO'd 2026-07-24, runs alongside — nothing here trains)

Recorded because it existed only in conversation:
- Organ evals: **vision, speech and imagery DONE 2026-07-25.** Vision = 12
  synthetic-marker probes, measured v8 at 9/12. The execution-trace blocker is
  CLOSED: a looped built-in is consumed by its hop, so the surfaced reply
  carried no `tool_calls` and a server-side action was unobservable — serve now
  reports `enigma.tools_run` and the eval grades on it, which also closes the
  restraint inflation where a probe expecting NO call passed while she called
  one. `speech` and `imagery` are 9 routing probes each (6 positive, 3
  restraint), ungated like vision. **Consequence to settle: the v5/v8 locked
  baseline was measured with a grader that could not see server-side calls, so
  its restraint column is an upper bound — re-measure before comparing v2 to
  it.** `ears` still needs an audio fixture and the organ loaded.
- Chat page senses **DONE 2026-07-25**: hold-to-talk mic posting to
  `/v1/audio/transcriptions`, and generated images render inline via
  `GET /v1/images/file/{name}` (bare name, strict pattern, resolved-parent
  re-check). Both appear only when `GET /v1/capabilities` says the organ
  booted. Image UPLOAD from the page shipped in `0a2e0fc`, so eyes are
  reachable from the page as well as from an API client -- the item is CLOSED.
- Stage 7 persona pack + `--name` spawn scaffold. **Foundation landed
  2026-07-25**: `enigma_engine/core/persona.py` holds identity as DATA, and
  `Persona.load()` with no pack IS Enigma -- every value verified byte-identical
  to the literal it replaced, so nothing about her moved. Serve reads it for the
  data home (voice recipe + generated images), the transcript stop marker and
  the tools-block system line, so a second AI on this box no longer overwrites
  the first one's voice and pictures. Unsafe packs (a path separator or newline
  in the name, which reaches a directory name AND a stop sequence) refuse.
  **`serve --persona <pack.json>` serves that AI instead** -- her own data home,
  voice recipe, images, turn marker and system line; omitted is Enigma.
  **Design finding worth keeping: the identity data is NOT mechanically
  parameterizable.** Her answers explain what the WORD "Enigma" means ("a
  closed box, in the good sense"), so a pack carries `name_meaning` rather than
  a template deriving it. STILL TO DO: `identity_anchors.py` (26 sites),
  `identity_paraphrases.py` (15), `make_sft_data.py` (14) and `teach_enigma.py`
  (9) still hold literals; the tray mutex `Local\EnigmaTray` and the fixed
  serve port are the remaining one-AI-per-machine guards; the `--name` spawn
  scaffold itself is unwritten. `personas_dir` in `config/defaults.py:145` is
  still the vestigial hook it always was.

Serving stays FROM-SCRATCH (ruled 2026-07-24): the llama.cpp/GGUF pivot is
REJECTED -- her serving path is our own code. Consequence executed: the
vendored `enigma_engine/bin/llama-server/` binary (+~1 GB DLLs) was DELETED
2026-07-25 (it was gitignored and never committed, so no history carries it;
~1 GB freed). The deferred eager-path optimizations (enable_gqa, fused
RMSNorm, CUDA graphs) are back on the table as future serving work.

## 8. Long-term (Phase 7 / embodiment; weeks of GPU)

- [x] New tokenizer — DONE 2026-07-20: v2 vocab 16,366 rows kills the
  standalone-space waste (25.5% -> 0.0%) and splits digits per-character;
  2.41x chars/token. Corpus retokenized to `data/pretrain/tokens_v2.bin`
  -- superseded 2026-07-28 by `tokens_v2b.bin` (28,261,718,460 tokens
  uint16); `tokens_v2.bin` (23,694,200,666) is the receipted rollback and
  v1 `tokens.bin` is untouched.
- [ ] Length extension block 1024 -> 2048+ (Phase 4) — **LARGELY MET BY v2**:
  v2 trains at block 2048 with 2.41x text per token, which holds roughly what
  v1 held at block 4900. NOT met by an 8192 context: the `v2_deep_*` presets declare `max_seq_len=8192` but are LAUNCHED at
block 2048 (the ruled shape). Block 8192 fits only WITH activation
checkpointing, at 35-40% throughput -- the no-checkpointing fit table in
BACKLOG 7.9 is what shows it missing at micro-batch 1
at any of the three sizes -- see the VRAM table in BACKLOG 7.9.
- [ ] Deeper-thinner 350-700M architecture — the 5090 can carry it. Presets
  `v2_deep_186m`/`238m`/`542m` are BUILT and opt-in; the open steps are the
  size call and the pretrain launch.
- [ ] Embodiment: tool-executor bridge + avatar bus (`ws://127.0.0.1:8765`) —
  work lives in the Enigma Avatar repo.
- [ ] Training sim -> trajectory logs -> real-time game play (FNAF target).
- [ ] Video: organ-tier (frame-sample -> describe -> summarize) buildable ANY
  time; native video needs Phase 4 first.

## Design doctrine: GROUNDING BEATS PRIORS (ruled 2026-07-24, the "purple banana" test)

The user's test: once the image system is live, "what color is THIS banana?"
over a picture of a purple one must answer PURPLE, not the memorized "yellow".
Consequences, binding on the training block and organ work:
- **Do NOT bake observable-world facts as strong priors.** teachings.jsonl
  oversamples x8 -- it HAMMERS whatever it holds -- so it is for identity,
  values, the USER'S personal facts, and corrections, NOT an encyclopedia of
  "bananas are yellow". A hammered perceptual prior fights the senses it is
  supposed to yield to.
- **Perceptual facts come from the senses at inference, not from weights.**
  This is why vision must be question-conditioned (stage-2 VQA, T7): the
  caption/observation has to reach the model AS the answer source, and a
  default ("bananas are usually yellow") must yield to an observation ("this
  one is purple"). The doctrine already in the repo -- engines never guess,
  the model routes what it cannot hold -- extends to perception: observation
  outranks prior.
- **Common sense as REASONING (not facts) is fine and comes from the pretrain
  corpus at scale** (diffuse, weak defaults), never from hand-authored
  oversampled teachings. "A dropped glass breaks" is reasoning; "the glass on
  my desk is broken" is an observation she must look for.


---

## 9. Disk reclaim — RULED 2026-07-29; the table below is the manifest

> **THE RULING (user, 2026-07-29): "delete anything old and unused and will
> not be used in the future."** Made with disk-death on the table: the SSD
> is trusted to outlive the build, and old weights are never retrained --
> new ones get trained instead. Every row below dies EXCEPT
> `enigma_pi_zero.pth` (VISION.md heritage citation -- HELD for an explicit
> word). Also dying, from outside the table: `data\pretrain\c4` +
> `openwebtext` (out of SOURCE_DIRS by ledger ruling 1), the June copy at
> `C:\Enigma-Backups`, and the venv's llama-cpp-python package (GGUF-pivot
> residue). Execution is the user's hands -- the classifier blocks mass
> deletion from a Claude tool call. SACRED and untouchable regardless:
> **tokens_v2b.bin (THE LIVE TRAINING CORPUS, 65.8 min to rebuild)**,
> tokens.bin (+R since 07-26), tokens_v2.bin (receipted rollback),
> Enigma Backups\, the SOURCE_DIRS corpora, llava/, LibriSpeech/, and the
> adopted/backed-up checkpoints -- tokens.bin and tokens_v2.bin were NOT
> named by the ruling and stay pending their own word.

| Candidate | Size | Why it is an orphan |
|---|---|---|
| `data\pretrain\combined.txt` | 95.1 GB | Forge-era combined stream; pretokenize reads SOURCE_DIRS instead ("doesn't touch combined.txt" is in its docstring); the Forge arm is deleted. Rebuildable by `collect_pretraining_data.py --combine`. |
| `models\qwen3-30b-a3b\` + `models\qwen3-8b\` | 34.95 GB | Qwen-era local weights, zero references; the distill teacher runs over Ollama, not these dirs. |
| `data\pretrain\enwiki-latest-pages-articles.xml.bz2` | 25.1 GB | Source archive already fully extracted into `wikipedia_dump\` (a live source dir); re-downloadable. |
| `models\enigma_pretrain_large\step_*.pth` (9 files) | 19.7 GB | Intermediate v1 checkpoints. RE-CORRECTED 2026-07-29 (the first correction was itself wrong): only `model.pth` has a receipted backup (`Enigma Backups\enigma_pretrain_large_final\` = model.pth/config.json/bpe_vocab.json + .sha256s, nothing else). The repo's `latest`/`prev` (07-03 finals) are KEPT. `C:\Enigma-Backups` (also dying) holds 7 of the 9 step files PLUS its own stale mid-run `latest`/`prev` from 06-29 -- older than the repo finals, so still not independent coverage of the final state. Deleting trades mid-run archaeology for the space. |
| `models\enigma_sft_v8\` | 6.56 GB | Superseded by the `enigma_sft_phase2_pass` backup. |
| `data\audio\train-clean-100.tar.gz` | 6.4 GB | Extracted to `LibriSpeech\` already; re-downloadable. |
| `models\enigma_pretrain_probe\` + `probe_tput_v2_deep_186m\` | 5.6 GB | Probe runs, July; their numbers are recorded in BACKLOG/TOKENIZER_V2_SPEC. |
| `models\enigma_pretrain_base\` + `enigma_pretrain_base_v2\` | 4.2 GB | Abandoned 121M side-runs (base_v2 died at step 2010/5086); historical prose mentions only. |
| `models\checkpoints\` | 3.4 GB | Forge-era best/final/vision .pt + a May run; nothing loads them. |
| `models\enigma_sft\model_v2_diluted.pth` | 2.19 GB | A 4th weight copy beside model/latest/prev in a dir whose receipts are already in Backups. |
| Loose: `enigma_small.pth`, `smoke.pth` | 0.33 GB | Zero code references. DIE under the 2026-07-29 ruling. |
| `models\enigma_pi_zero.pth` | 0.01 GB | **HELD** -- VISION.md destination 5 cites it as the Pi-class heritage artifact; deleting it orphans that citation. Needs its own word. |
| `data\curriculum\`, `data\conversations\`, `data\model_contexts\` + 14 dead loose `data\` files: `.gui_lock`, `anchor_examples.jsonl`, `curated_dataset.jsonl`, `distilled_smoke.txt`, `enigma_gui_training.txt`, `enigma_voice.md`, `gui_settings.json`, `instructions.txt`, `personality_corpus.jsonl`, `prompts.json`, `route_assignments.json`, `training.txt`, `training_brief.json`, `training_history.json` | ~4 MB | Forge/GUI-era outputs, zero code references (census 2026-07-27). NOT in this row: `mute_state.json` (live serve runtime state) and `smoke_test_basic.txt`/`smoke_test_dpo.jsonl` (regenerable outputs of `create_smoke_test_data.py`). |

> **ONE DOT SEPARATES A CLEANUP FROM A BREAKAGE.** `venv\` and `.venv\` are
> different directories and only one of them dies. **`venv\` (7.73 GB) is
> LOAD-BEARING** -- kokoro lives there and `Start-Enigma.ps1` launches the
> server from `venv\Scripts\python.exe`, so deleting it disables voice on
> every launch. `.venv\` (0.01 GB) is the orphan. Deleting the orphan reclaims
> 10 MB; the typo costs 7.73 GB and the voice organ. The llama-cpp-python row
> above is a PACKAGE INSIDE `venv\`, not the directory: uninstall the package,
> never remove its parent. (Both sizes measured 2026-07-30.)

**Manifest additions -- candidates the ruling's language covers that the table
above never listed** (sizes measured 2026-07-30; `code_refs` = files matching
the name under `*.py`/`*.ps1`/`*.bat`):

| Candidate | Size | code_refs | Status |
|---|---|---|---|
| `models\enigma_dpo_v8\` | 2.19 GB | 0 | Orphan -- superseded DPO lineage. |
| `models\enigma_vision_align\` 2 of 3 `.pt` siblings | 1.96 GB | 2 | `vision1` / `vision_best` / `vision_step` are three 0.98 GB copies. The DIR is referenced -- keep `vision_best.pt`, the other two are the reclaim. |
| `data\pretrain\wiki_dump_index.txt.bz2` | 0.28 GB | 0 | Index for an archive already extracted to `wikipedia_dump\`; re-downloadable. |
| `models\enigma_forge_tiny\` | 0.15 GB | 0 | Forge-era, zero references. |
| `models\enigma_lora_v1\` | 0.13 GB | 0 | LoRA-era, zero references. |
| `.venv\` | 0.01 GB | 0 | The ORPHAN env -- read the one-dot warning above before typing this path. |
| `models\registry.json`, `data\prompts\` | <0.01 GB | 0 | Forge-era residue. |

Additions total ~4.72 GB. **NOT proposed and NOT orphans:** `venv\` (see
above) and `models\enigma_pretrain_facts\` (6.56 GB, 1 code ref -- it is the
facts-pretrain lineage, and the "facts need continued-pretrain not SFT" finding
rests on it; needs its own word before it dies).

**`models\sweeps\t2_238m\` (11.85 GB / 11.03 GiB, measured 2026-07-30) is on no
row and should NOT be deleted yet.** The six point dirs are throwaway final
models, but they are the only cheap way to re-rank the T2 winner: tail-val
already prefers 6e-3 at seed 0 while [val-gen] picked 3e-3, so the winner is
window-dependent, and re-scoring six existing checkpoints costs ~0.5-1 h of GPU
against ~8.5 h to re-run the sweep. **HOLD DISCHARGED 2026-07-30: the scoring
RAN** (~8.2 min/checkpoint, ~50 min total; receipts in `Enigma Backups\`, verdict
at item 7 -- 3e-3 confirmed on all three signals and 30/30 source windows). The
six point dirs have given up everything they hold and are now free to delete on
the user's word.

Ruled 2026-07-29 as above; when the user's deletion run completes, move this
section's record to CLEANUP_TRACKER with the per-item receipts.
