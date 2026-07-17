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
    g = LockedProbeGuard(seal(["What's the capital city of France?"]))
    # share one content word ("france") but clearly a different question:
    # in the review band, not a hard leak.
    assert not g.leaks("What language do they speak in France?")
    assert g.is_near_miss("What language do they speak in France?") or \
        g.score("What language do they speak in France?") < 0.5


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
