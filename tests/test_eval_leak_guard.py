"""Locked-probe leak guard (eval_leak_guard): a SEALED manifest of hashed
content-word shingles must reject verbatim AND paraphrase-close training
questions while (a) never shipping probe plaintext and (b) staying a no-op
before any locked set exists (EVAL_REDESIGN.md, 2026-07-16)."""

from __future__ import annotations

import inspect
import json

import pytest

from eval_leak_guard import LockedProbeGuard, _content_words, refuse_if_leaky, seal


def _manifest_file(tmp_path, texts):
    p = tmp_path / "m.json"
    p.write_text(json.dumps(seal(texts)), encoding="utf-8")
    return p


def test_the_manifest_seals_the_grading_keys_not_just_the_questions(tmp_path):
    """want_any/deny_any/expect_tool/category decide every verdict and are not
    sealed TEXT. Sealing a digest of them INTO the manifest is what makes them
    checkable without the plaintext: comparing against an on-disk copy could
    never work, because the canonical run points --probes AT that copy."""
    from eval_leak_guard import grading_digest, seal

    cases = [
        {"category": "factual", "q": "Largest planet?", "want_any": ["jupiter"], "deny_any": ["pluto"]},
        {"category": "tool", "q": "Weather in Denver?", "expect_tool": "get_weather"},
    ]
    manifest = seal([c["q"] for c in cases], cases=cases)
    assert manifest["grading_digest"] == grading_digest(cases)

    # each field that steers a score must move the digest
    for mutate in (
        lambda c: c[0].update(want_any=[]),
        lambda c: c[0].update(deny_any=[]),
        lambda c: c[1].update(expect_tool=None),
        lambda c: c[0].update(category="identity"),
    ):
        edited = [dict(c) for c in cases]
        mutate(edited)
        assert grading_digest(edited) != manifest["grading_digest"]

    # ...and a manifest sealed WITHOUT cases carries no digest at all, which is
    # what the run refuses to treat as a gate
    assert "grading_digest" not in seal([c["q"] for c in cases])


def test_the_digest_seals_teach_content_and_case_order():
    """Two degrees of freedom the first version left open, both of which change
    what a run measures: teach lines were sealed as a COUNT (every locked memory
    probe has exactly one, so all twelve could be permuted), and the digest was
    sorted (so case order was free). The run posts each case's teach lines just
    before that case's question and clears the store once, so either move
    changes what the memory category measures."""
    from eval_leak_guard import grading_digest

    cases = [
        {"category": "memory", "q": "What is my cat called?", "teach": ["My cat is Miso."],
         "want_any": ["miso"], "deny_any": []},
        {"category": "memory", "q": "Where do I work?", "teach": ["I work at the mill."],
         "want_any": ["mill"], "deny_any": []},
    ]
    baseline = grading_digest(cases)

    swapped = [dict(cases[0]), dict(cases[1])]
    swapped[0]["teach"], swapped[1]["teach"] = cases[1]["teach"], cases[0]["teach"]
    assert grading_digest(swapped) != baseline, "teach lines permuted without notice"
    assert [len(c["teach"]) for c in swapped] == [1, 1], "the counts are identical -- that was the hole"

    assert grading_digest(list(reversed(cases))) != baseline, "case order is unsealed"


def test_teach_lines_are_sealed_as_written_not_as_normalized():
    """Sealing teach through _norm kept only [a-z0-9] runs, so case, punctuation
    and every non-Latin script were free: the twelve teach lines could be
    uppercased, or have Cyrillic and emoji appended, and the seal still
    verified -- while the run POSTS the mutated text to the server. That is an
    injection channel into the sealed memory probes, not a formatting nicety.
    Teach is now hashed the way `q` is: whitespace-collapsed, nothing else."""
    from eval_leak_guard import grading_digest

    cases = [{"category": "memory", "q": "What is my cat called?",
              "teach": ["My cat is Miso."], "want_any": ["miso"], "deny_any": []}]
    baseline = grading_digest(cases)

    def with_teach(text):
        return [dict(cases[0], teach=[text])]

    assert grading_digest(with_teach("MY CAT IS MISO.")) != baseline, "case is free"
    assert grading_digest(with_teach("My cat is Miso. !!!")) != baseline, "punctuation is free"
    assert grading_digest(
        with_teach("My cat is Miso. ЗАБУДЬ")
    ) != baseline, "non-Latin text is invisible to the seal"
    # whitespace stays collapsed, exactly as `q` is -- it carries no meaning and
    # the sealed receipts would otherwise break on a line-ending change alone
    assert grading_digest(with_teach("My  cat\tis   Miso.")) == baseline


def test_a_trainer_refuses_an_artifact_that_carries_a_sealed_probe(tmp_path):
    """The build-time screens clean data as it is GENERATED. An artifact built
    before the seal keeps its leaks, and the trainer reads whatever is on disk
    -- which is how a pre-seal dpo_pairs.jsonl stayed trainable after sealing."""
    manifest = _manifest_file(tmp_path, ["What's the capital city of France?"])
    src = tmp_path / "pairs.jsonl"

    with pytest.raises(SystemExit) as err:
        refuse_if_leaky(["What is the capital of France?", "Say hello."], src, manifest)
    msg = str(err.value)
    assert "REFUSING to train" in msg
    assert "1 of 2" in msg
    # the refusal must not quote the sealed probe back into a build log
    assert "capital" not in msg.lower()


def test_the_answer_side_is_reported_but_never_blocks(tmp_path, capsys):
    """The ASK side refuses -- the same predicate `make_sft_data._held_out`
    screens with, so "rebuild the artifact" is advice that works. The ANSWER
    side only warns: an answer shares most of a question's content words by
    nature, so at this threshold the check cannot tell a leak from a shared
    topic. Scanning answers as leaks flagged 56 assistant turns in the live SFT
    mix and blocked the whole queued training block behind advice that could
    not clear it, because the builder screens the question side only."""
    manifest = _manifest_file(tmp_path, ["What's the capital city of France?"])

    # a leak in the ANSWER position is counted, not fatal
    refuse_if_leaky(["Tell me something."], tmp_path / "mix.jsonl", manifest,
                    advisory=["What is the capital of France?"])
    out = capsys.readouterr().out
    assert "1 of 1 answer-side" in out
    assert "reviewed not blocked" in out

    # ...but the same string in the ASK position still refuses
    with pytest.raises(SystemExit) as err:
        refuse_if_leaky(["What is the capital of France?"], tmp_path / "mix.jsonl", manifest)
    assert "ASKS that match" in str(err.value)


def test_quoting_a_probe_leaks_however_it_is_padded_or_edited(tmp_path):
    """Jaccard is a ratio, so unrelated words shrink it while the probe sits
    intact inside the text: a 10-content-word probe drops to 0.529 at eight
    filler words and trained freely. A ratio and a set both had to trade
    dilution against false-firing on long documents; an ORDERED run trades
    neither, because padding cannot remove a run and an unrelated document does
    not reproduce a probe's word order by accident."""
    probe = "Which planet in our solar system has the greatest total mass overall"
    guard = LockedProbeGuard(seal([probe]))
    filler = ("bicycle tyre lever puncture repair kit spanner workshop garage "
              "afternoon weather forecast pencil notebook lantern kettle").split()
    for pad in (0, 6, 8, 12, 16):
        text = probe + " " + " ".join(filler[:pad])
        assert guard.leaks(text), f"diluted with {pad} filler words and slipped through"
    assert guard.score(probe + " " + " ".join(filler)) < guard.threshold, \
        "fixture is not exercising dilution -- the ratio must fall below the bar"

    # one substituted word used to re-open dilution: the run survives it
    swapped = "Which zebra in our solar system has the greatest total mass overall"
    assert guard.leaks(swapped + " " + " ".join(filler))

    # ...and a probe split across turns is still quoted by any PART that keeps a
    # whole run. The set-based predicate saw a 3-way split as three innocent
    # fragments (0.222/0.444/0.444) and trained the whole probe.
    assert guard.leaks("planet in our solar system has the greatest")
    # The honest boundary: a fragment shorter than one run is not a quotation
    # and is not screened -- splitting finely enough always defeats any
    # quotation test, and at that point the jaccard net is what remains.
    assert not guard.contains_probe("our solar system")


def test_a_two_content_word_probe_is_sealed_and_survives_dilution(tmp_path):
    """18 of the 108 sealed strings -- five of them locked memory TEACH lines
    -- carry exactly two content words, and at floor 3 they were sealed as
    NOTHING: a training turn held a sealed teach fact whole while the guard
    printed a clean bill (round-7, 2026-07-25; measured 0/10 such turns
    caught on the live mix, 10/10 at floor 2). A 2-word probe now seals its
    single run, which padding cannot remove and jaccard dilution cannot
    hide."""
    guard = LockedProbeGuard(seal(["I work as a nurse."]))
    assert [w for w in _content_words("I work as a nurse.")] == ["work", "nurse"], \
        "fixture assumption gone: the probe no longer has exactly two content words"
    diluted = ("These days, believe it or not, I actually work as a nurse "
               "over at the big hospital across town.")
    assert guard.contains_probe(diluted)
    assert guard.leaks(diluted)
    assert guard.score(diluted) < guard.threshold, \
        "fixture is not exercising dilution -- jaccard alone must miss this"
    # order still separates quoting from topic overlap
    assert not guard.contains_probe("Any nurse might work anywhere at all.")
    # one content word stays run-free: that is a membership test, not a
    # quotation, and exact/jaccard are what remain for it
    g1 = LockedProbeGuard(seal(["What's your name?"]))
    assert not g1.contains_probe("her name was lost in the ledger pages")


def test_a_manifest_sealed_at_another_floor_is_refused(tmp_path):
    """The trainers never re-seal: under floor-2 code a stale floor-3 manifest
    keeps every 2-content-word string sealed as NOTHING while the guard still
    prints ACTIVE. Seal and test must speak the same unit, so a run-parameter
    mismatch is a Weakened refusal, not a silent downgrade."""
    for key, value in (("ngram_min", 3), ("ngram_n", 5)):
        man = seal(["I work as a nurse."])
        man[key] = value
        with pytest.raises(LockedProbeGuard.Weakened, match="re-seal"):
            LockedProbeGuard(man)
    # a legacy manifest that carries runs but predates the keys was sealed at
    # (3, 4) -- refused for the same reason
    legacy = seal(["Which planet in our solar system has the greatest total mass overall"])
    del legacy["ngram_min"], legacy["ngram_n"]
    with pytest.raises(LockedProbeGuard.Weakened, match="re-seal"):
        LockedProbeGuard(legacy)
    # no sealed set at all stays a safe no-op, never a refusal
    assert len(LockedProbeGuard(None)) == 0
    assert not LockedProbeGuard(None).leaks("anything")


def test_stripping_the_sealed_arrays_fails_closed(tmp_path):
    """The round-5 jaccard_threshold lesson re-entering through the arrays
    (fix-arc audit, 2026-07-25): emptying probes[].n or .s passed every
    digest -- the seal comparison is over exact-hash lists and the file
    digest covers the plaintext, not the sidecar -- so the quotation or
    paraphrase tier vanished while the banner still printed ACTIVE."""
    probe = "Which planet in our solar system has the greatest total mass overall"
    man = seal([probe])
    for strip in ("n", "s"):
        crippled = json.loads(json.dumps(man))
        for p in crippled["probes"]:
            p[strip] = []
        with pytest.raises(LockedProbeGuard.Weakened, match="re-seal"):
            LockedProbeGuard(crippled)
    # removing the key outright is the same edit
    crippled = json.loads(json.dumps(man))
    for p in crippled["probes"]:
        del p["n"]
    with pytest.raises(LockedProbeGuard.Weakened, match="re-seal"):
        LockedProbeGuard(crippled)
    # stripping BOTH arrays slips between the two one-sided checks -- it is
    # also the legitimate shape of a stopword-only probe -- but a whole
    # manifest of them can only be a strip or a pre-4-gram seal, and refuses
    # in aggregate (round-B audit, 2026-07-25)
    crippled = json.loads(json.dumps(man))
    for p in crippled["probes"]:
        p["s"], p["n"] = [], []
    with pytest.raises(LockedProbeGuard.Weakened, match="re-seal"):
        LockedProbeGuard(crippled)
    # DELETING the fields (not emptying them) on one probe of a larger
    # manifest slipped past every payload check and died as a bare KeyError
    # at construction -- fail-closed, but a traceback where the class
    # promises a refusal (round-C audit, 2026-07-25)
    two = seal(["I work as a nurse.", probe])
    for missing in ("s", "n", "h"):
        crippled = json.loads(json.dumps(two))
        del crippled["probes"][1][missing]
        with pytest.raises(LockedProbeGuard.Weakened, match="re-seal"):
            LockedProbeGuard(crippled)
    # the genuine manifest shape stays constructible: a 1-content-word probe
    # is legitimately run-free and must not read as "stripped"
    ok = seal(["What's your name?", probe])
    assert len(LockedProbeGuard(ok)) == 2


def test_a_long_document_that_merely_shares_words_does_not_leak(tmp_path):
    """The set-based containment test fired on a 1407-word record that happened
    to use all six content words of a sealed probe. Order is what separates
    quoting a probe from writing about the same subject."""
    probe = "Which planet in our solar system has the greatest total mass overall"
    guard = LockedProbeGuard(seal([probe]))
    scattered = (
        "The total mass of a star dwarfs every planet around it. Our solar "
        "neighbourhood formed from one cloud, and the greatest share of that "
        "material never became a planet at all. Which of the bodies kept the "
        "most is a question of accretion, not of overall size."
    )
    assert not guard.contains_probe(scattered), "word soup read as a quotation"
    assert not guard.leaks(scattered)

    # The load-bearing case: the SAME words, ADJACENT, in a different order.
    # Distance alone would be caught by an unordered containment test too, so
    # only a permutation proves the run is what is being matched.
    permuted = "A greatest system solar planet argument settles nothing here."
    assert set(_content_words(permuted)) >= {"planet", "solar", "system", "greatest"}, \
        "fixture assumption gone: the permutation no longer carries the probe's words"
    assert not guard.contains_probe(permuted), "an out-of-order run read as a quotation"


def test_the_guard_records_a_durable_verdict(tmp_path):
    """A console count vanishes under redirection and left a finished checkpoint
    with no evidence the screen ran. The verdict lands beside the artifact and
    names the manifest it was screened against."""
    manifest = _manifest_file(tmp_path, ["What is the capital city of France?"])
    src = tmp_path / "mix.jsonl"
    src.write_text("{}\n", encoding="utf-8")
    refuse_if_leaky(["Say hello."], src, manifest,
                    advisory=["Paris is the capital city of France."])

    from eval_leak_guard import last_verdict

    v = last_verdict(src)
    assert v is not None
    assert v["asks_screened"] == 1
    assert v["answer_side_flagged"] == 1
    assert v["manifest_sha256"] and v["sealed_probes"] == 1
    assert last_verdict(tmp_path / "never_screened.jsonl") is None


def test_an_inactive_run_replaces_any_earlier_verdict(tmp_path):
    """Returning without touching the verdict let `last_verdict` hand back a
    previous run's "108 sealed probes enforced", which finetune then stamped
    into a checkpoint screened by nothing at all. Absence of a write was being
    read as a passing result."""
    from eval_leak_guard import last_verdict

    manifest = _manifest_file(tmp_path, ["What is the capital city of France?"])
    src = tmp_path / "mix.jsonl"
    src.write_text("{}\n", encoding="utf-8")

    refuse_if_leaky(["Say hello."], src, manifest)
    active = last_verdict(src)
    assert active["active"] is True and active["sealed_probes"] == 1
    assert active["source_sha256"], "a verdict must name WHICH bytes were screened"

    refuse_if_leaky(["Say hello."], src, tmp_path / "absent.json")
    after = last_verdict(src)
    assert after["active"] is False and after["sealed_probes"] == 0


def test_a_clean_artifact_trains(tmp_path):
    manifest = _manifest_file(tmp_path, ["What's the capital city of France?"])
    refuse_if_leaky(["What is the tallest mountain?", "Say hello."], tmp_path / "ok.jsonl", manifest)


def test_the_consume_time_guard_is_a_noop_before_a_seal_exists(tmp_path, capsys):
    refuse_if_leaky(["anything at all"], tmp_path / "x.jsonl", tmp_path / "absent.json")
    # ...and it SAYS so. Returning in silence made a training log unable to tell
    # a clean run from one where the guard never ran: a missing or emptied
    # manifest read exactly like success.
    assert "INACTIVE" in capsys.readouterr().out


def test_the_trainer_sees_tool_call_arguments_and_system_turns(tmp_path, monkeypatch):
    """`content` is "" on a tool-calling assistant turn, so the guard read an
    empty string and printed "asks clean" while the payload -- inside the
    trainable mask, scoring 0.875 against a sealed probe -- trained normally.
    tool_calls.jsonl is built ENTIRELY of that shape. System turns are prompt
    side and were filed as advisory, so a probe pasted into a system block
    warned instead of refusing.

    This drives the REAL load_examples and captures what it hands the guard. An
    earlier version of this test rebuilt the projection locally and asserted on
    its own copy, which passed happily with the bug reintroduced."""
    import finetune_enigma
    from enigma_engine.core.chat_format import attach_chat_tokens
    from enigma_engine.core.tokenizer import get_tokenizer

    probe = "What is the capital city of France?"
    seen = {}

    def capture(texts, source, manifest=None, advisory=None):
        seen["asks"] = list(texts)
        seen["answers"] = list(advisory or [])

    monkeypatch.setattr(finetune_enigma, "refuse_if_leaky", capture)
    tok = get_tokenizer("bpe")
    attach_chat_tokens(tok)

    def load(msgs):
        seen.clear()
        p = tmp_path / "rec.jsonl"
        p.write_text(json.dumps({"messages": msgs}) + "\n", encoding="utf-8")
        finetune_enigma.load_examples(p, tok, 1024)
        return seen

    got = load([
        {"role": "user", "content": "How do I change a tyre?"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "search_notes",
                                      "arguments": json.dumps({"query": probe})}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "(results)"},
        {"role": "assistant", "content": "Paris."},
    ])
    assert any(probe in s for s in got["asks"] + got["answers"]), \
        "tool-call arguments are trained on but reach no screen"

    got = load([{"role": "system", "content": probe},
                {"role": "user", "content": "Hello."},
                {"role": "assistant", "content": "Hi."}])
    assert any(probe in s for s in got["asks"]), "a system turn is prompt side, not advisory"


def test_empty_guard_is_a_noop():
    g = LockedProbeGuard(None)
    assert len(g) == 0
    assert not g.leaks("literally anything at all")
    assert g.score("whatever") == 0.0


def test_verbatim_and_paraphrase_leak_but_distinct_facts_do_not():
    g = LockedProbeGuard(seal([
        "What's the capital city of France?",
        "Who developed the theory of relativity?",
    ]))
    assert g.leaks("What's the capital city of France?")          # verbatim
    assert g.leaks("What is the capital of France?")               # paraphrase twin
    assert g.leaks("Which city is the capital of France, again?")  # heavier reword
    # a DIFFERENT fact must not trip the guard
    assert not g.leaks("What's the capital of Italy?")
    assert not g.leaks("What is the tallest mountain?")


def test_near_miss_band_is_flagged_not_dropped():
    """is_near_miss must return True for a genuinely in-band paraphrase.

    The old assertion `is_near_miss(x) or score(x) < 0.5` reduced to
    `score(x) < threshold` -- already guaranteed by the `not leaks` line
    above it -- AND its example scored ~0.17, nowhere near the band. The
    band feeds make_sft_data's human-review flagging, and an is_near_miss
    that always returned False stayed green (test-suite audit 2026-07-17).
    """
    g = LockedProbeGuard(seal(["Who developed the theory of relativity?"]))
    # Verb-swap paraphrase (measured 2026-07-16): content words
    # {came, theory, relativity} vs {developed, theory, relativity}
    # -> Jaccard 2/4 = 0.5, inside [0.5, 0.6).
    q = "Who came up with the theory of relativity?"
    assert not g.leaks(q)  # below the drop threshold...
    assert g.is_near_miss(q)  # ...but the review band MUST flag it
    assert 0.5 <= g.score(q) < 0.6


def test_clearly_different_question_is_not_near_missed():
    g = LockedProbeGuard(seal(["What's the capital city of France?"]))
    # shares one content word ("france"): Jaccard ~0.17 -- neither a leak
    # nor review-band noise.
    other = "What language do they speak in France?"
    assert not g.leaks(other)
    assert not g.is_near_miss(other)
    assert g.score(other) < 0.5


def test_manifest_seals_the_plaintext():
    manifest = seal(["What's the capital city of France?"])
    blob = json.dumps(manifest).lower()
    # no probe word survives into the manifest -- only hashes
    for word in ("capital", "city", "france", "what"):
        assert word not in blob
    assert manifest["probes"][0]["h"] and manifest["probes"][0]["s"]


def test_a_stricter_threshold_is_carried_through_the_manifest():
    g = LockedProbeGuard(seal(["Who built you and why?"], threshold=0.4))
    assert g.threshold == 0.4


def test_a_threshold_at_or_below_zero_is_refused():
    """The bound was one-sided, so 0.0 and -1.0 read as "stricter than asked"
    and made the guard refuse EVERY artifact -- a broken manifest blaming the
    data it was screening."""
    for bad in (0.0, -1.0):
        with pytest.raises(LockedProbeGuard.Weakened) as err:
            LockedProbeGuard(seal(["Who built you and why?"], threshold=bad))
        assert "every string" in str(err.value)


def test_the_eval_applies_the_threshold_rule_the_trainers_enforce():
    """`eval_behavior` read the manifest with a bare `json.loads`, so the
    Weakened check that makes every TRAINER refuse a loosened manifest was not
    applied by the file whose result decides adoption."""
    import eval_behavior

    src = inspect.getsource(eval_behavior._sealed_manifest)
    assert "LockedProbeGuard" in src
    for fn in (eval_behavior._sealed_hashes, eval_behavior._probe_hashes,
               eval_behavior._seal_mismatch):
        assert "json.loads(LOCKED_MANIFEST" not in inspect.getsource(fn), fn.__name__


def test_a_weakened_threshold_in_the_manifest_is_refused():
    """The threshold is the one enforcement parameter that lived in an editable
    sidecar with nothing verifying it: probe hashes and the grading digest are
    identical under any threshold, so eval printed "seal verified" over a
    manifest edited to 0.99 while every paraphrase trained freely. Raising it is
    purely a weakening, so it is refused rather than obeyed."""
    with pytest.raises(LockedProbeGuard.Weakened) as err:
        LockedProbeGuard(seal(["Who built you and why?"], threshold=0.99))
    assert "weaker than the code default" in str(err.value)
    with pytest.raises(LockedProbeGuard.Weakened):
        LockedProbeGuard({"jaccard_threshold": "0.6", "probes": []})


def test_seal_warns_on_stopword_only_probe(tmp_path, monkeypatch, capsys):
    """An all-stopword probe ("Is it you?") has an empty shingle set and can
    only ever match verbatim -- seal must say so instead of arming a guard
    that silently never fires on paraphrases."""
    import eval_leak_guard as elg

    src = tmp_path / "locked.jsonl"
    src.write_text(
        '{"q": "Is it you?"}\n{"q": "What is the capital of France?"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(elg, "LOCKED_MANIFEST", tmp_path / "manifest.json")
    assert elg._cli_seal(str(src)) == 0
    out = capsys.readouterr().out
    assert "no content words" in out
    assert "Is it you?" in out
    assert out.count("WARN") == 1  # the France probe is fine


def test_the_sealer_refuses_a_commented_file(tmp_path, monkeypatch, capsys):
    """A skipped comment enters the file's BYTE identity (probe_file_sha256)
    without entering any sealed hash -- so a commented file could seal and
    then gate-run while carrying editable text no hash covers (2026-07-26
    audit; the docs' "a commented holdout can never match its seal" was only
    true if this refusal exists)."""
    import eval_leak_guard as elg

    src = tmp_path / "locked.jsonl"
    src.write_text(
        '# a stray authoring note\n{"q": "What is the capital of France?"}\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(elg, "LOCKED_MANIFEST", tmp_path / "manifest.json")
    assert elg._cli_seal(str(src)) == 1
    assert "comment line" in capsys.readouterr().out
    assert not (tmp_path / "manifest.json").exists(), "refusal must not write a seal"
