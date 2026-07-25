"""Locked-probe leak guard (eval_leak_guard): a SEALED manifest of hashed
content-word shingles must reject verbatim AND paraphrase-close training
questions while (a) never shipping probe plaintext and (b) staying a no-op
before any locked set exists (EVAL_REDESIGN.md, 2026-07-16)."""

from __future__ import annotations

import inspect
import json

import pytest

from eval_leak_guard import LockedProbeGuard, refuse_if_leaky, seal


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


def test_padding_a_verbatim_probe_does_not_hide_it(tmp_path):
    """Jaccard is a ratio, so unrelated words shrink it while the probe sits
    intact inside the text: a 10-content-word probe drops to 0.529 at eight
    filler words and trained freely. Containment cannot be diluted, because
    padding only ever adds words."""
    probe = "Which planet in our solar system has the greatest total mass overall"
    guard = LockedProbeGuard(seal([probe]))
    filler = ("bicycle tyre lever puncture repair kit spanner workshop garage "
              "afternoon weather forecast pencil notebook lantern kettle").split()
    for pad in (0, 6, 8, 12, 16):
        text = probe + " " + " ".join(filler[:pad])
        assert guard.leaks(text), f"diluted with {pad} filler words and slipped through"
    assert guard.score(probe + " " + " ".join(filler)) < guard.threshold, \
        "fixture is not exercising dilution -- the ratio must fall below the bar"

    # ...and a SHORT probe keeps the ratio test only: too few content words for
    # containment to distinguish a leak from an ordinary mention.
    short = LockedProbeGuard(seal(["Largest planet?"]))
    assert not short.contains_probe("The largest planet question came up at dinner tonight")


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
