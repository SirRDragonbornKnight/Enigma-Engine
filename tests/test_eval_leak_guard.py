"""Locked-probe leak guard (eval_leak_guard): a SEALED manifest of hashed
content-word shingles must reject verbatim AND paraphrase-close training
questions while (a) never shipping probe plaintext and (b) staying a no-op
before any locked set exists (EVAL_REDESIGN.md, 2026-07-16)."""

from __future__ import annotations

import json

from eval_leak_guard import LockedProbeGuard, seal


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


def test_threshold_is_carried_through_the_manifest():
    g = LockedProbeGuard(seal(["Who built you and why?"], threshold=0.9))
    assert g.threshold == 0.9


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
