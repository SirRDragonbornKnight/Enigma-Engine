"""Tests for ``collect_search_data.py`` — B-4 synthetic ``<search>``
training corpus emitter.

The corpus is **content** as much as code: bad templates silently
poison training.  These tests gate three contracts:

  1. **Class purity** — positives carry ``<search>...</search>``,
     negatives never do.  ``_validate_examples`` raises if either
     invariant breaks.
  2. **Output shape** — ``build_corpus`` returns a dedup-friendly list
     of ``{"prompt", "completion"}`` dicts in the requested mix.
  3. **CLI surface** — ``main()`` writes both JSONL and the canonical
     ``User:/Assistant:`` text format under the configured ``--tag``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import collect_search_data as csd


# ── _validate_examples ───────────────────────────────────────────────


def test_validate_examples_passes_on_shipped_corpus() -> None:
    """The corpus shipped with the file must validate clean.

    If this fails, a recent edit to ``_POSITIVE_EXAMPLES`` /
    ``_NEGATIVE_EXAMPLES`` introduced a class-purity violation.
    """
    csd._validate_examples()  # raises on bad data


def test_validate_examples_rejects_positive_without_search_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Positive that's missing the ``<search>`` tag → loud failure."""
    monkeypatch.setattr(
        csd,
        "_POSITIVE_EXAMPLES",
        [("Where is Mars?", "Mars is the fourth planet.")],  # no <search>!
    )
    monkeypatch.setattr(csd, "_NEGATIVE_EXAMPLES", [])
    with pytest.raises(RuntimeError, match="POSITIVE missing"):
        csd._validate_examples()


def test_validate_examples_rejects_negative_with_search_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative that wraps an answer in ``<search>`` → loud failure."""
    monkeypatch.setattr(csd, "_POSITIVE_EXAMPLES", [])
    monkeypatch.setattr(
        csd,
        "_NEGATIVE_EXAMPLES",
        [("What is 2+2?", "<search>2+2</search>")],  # poisoned!
    )
    with pytest.raises(RuntimeError, match="NEGATIVE contains"):
        csd._validate_examples()


def test_validate_examples_rejects_a_prompt_in_both_classes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Class purity is per-list and cannot see the worst contradiction: the
    SAME question taught both ways teaches opposite behaviour on identical
    input.  Both halves are individually well-formed, so only a cross-list
    check catches it."""
    monkeypatch.setattr(
        csd,
        "_POSITIVE_EXAMPLES",
        [("What is the capital of France?", "<search>capital of France</search>")],
    )
    monkeypatch.setattr(
        csd,
        "_NEGATIVE_EXAMPLES",
        [("what is the capital of France?  ", "The capital of France is Paris.")],
    )
    with pytest.raises(RuntimeError, match="BOTH classes"):
        csd._validate_examples()
    # and the offending prompt is named, not just counted
    with pytest.raises(RuntimeError, match="capital of France"):
        csd._validate_examples()


def test_shipped_corpus_has_no_contradictory_prompt() -> None:
    """The real corpus (31 positive + 29 negative, 2026-08-14) carries no
    prompt in both classes.  The lists grow by appending, so this is the
    guard on future rows, not a pin on today's counts."""
    csd._validate_examples()  # includes the cross-list check
    pos = {p.strip().lower() for p, _ in csd._POSITIVE_EXAMPLES}
    neg = {p.strip().lower() for p, _ in csd._NEGATIVE_EXAMPLES}
    assert not pos & neg
    # ...and the emitted corpus agrees: one class per prompt, end to end
    emitted: dict[str, bool] = {}
    for row in csd.build_corpus():
        key = row["prompt"].strip().lower()
        emits = "<search>" in row["completion"]
        assert emitted.setdefault(key, emits) == emits, row["prompt"]


# ── build_corpus ─────────────────────────────────────────────────────


def test_build_corpus_default_mixes_both_classes() -> None:
    """Default call returns BOTH positives and negatives — the
    no-search class is non-optional in the production mix because
    without it the model overfits to 'always emit <search>'."""
    pairs = csd.build_corpus()
    n_pos = sum(1 for p in pairs if "<search>" in p["completion"])
    n_neg = len(pairs) - n_pos
    assert n_pos > 0 and n_neg > 0
    # Sanity on shape
    for item in pairs:
        assert set(item.keys()) == {"prompt", "completion"}
        assert isinstance(item["prompt"], str) and item["prompt"]
        assert isinstance(item["completion"], str) and item["completion"]


def test_build_corpus_positive_only() -> None:
    pairs = csd.build_corpus(positive_only=True)
    assert all("<search>" in p["completion"] for p in pairs)
    assert len(pairs) == len(csd._POSITIVE_EXAMPLES)


def test_build_corpus_negative_only() -> None:
    pairs = csd.build_corpus(negative_only=True)
    assert all("<search>" not in p["completion"] for p in pairs)
    assert len(pairs) == len(csd._NEGATIVE_EXAMPLES)


def test_build_corpus_mutually_exclusive_flags_raise() -> None:
    """Both flags at once is nonsense — fail loud, not silent."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        csd.build_corpus(positive_only=True, negative_only=True)


# ── main / CLI ───────────────────────────────────────────────────────


def test_main_writes_jsonl_and_text(tmp_path: Path) -> None:
    """End-to-end: CLI run produces both output files in the
    canonical SFT format the existing trainer reads."""
    rc = csd.main(["--tag", "test", "--output-dir", str(tmp_path)])
    assert rc == 0
    jsonl_path = tmp_path / "synthetic_search_test.jsonl"
    text_path = tmp_path / "synthetic_search_test.txt"
    assert jsonl_path.exists() and text_path.exists()

    # JSONL: every line parses to a {prompt, completion} dict
    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) > 0
    for r in rows:
        assert {"prompt", "completion"} <= set(r.keys())

    # Text: starts with 'User: ' and contains canonical separator
    text = text_path.read_text(encoding="utf-8")
    assert text.startswith("User: ")
    assert "\n\nAssistant: " in text
    # And at least one positive row landed in the text format too
    assert "<search>" in text


def test_main_tag_propagates_to_filenames(tmp_path: Path) -> None:
    rc = csd.main(["--tag", "v2_alpha", "--output-dir", str(tmp_path)])
    assert rc == 0
    assert (tmp_path / "synthetic_search_v2_alpha.jsonl").exists()
    assert (tmp_path / "synthetic_search_v2_alpha.txt").exists()


def test_main_positive_only_omits_negatives(tmp_path: Path) -> None:
    """Adversarial: ``--positive-only`` must NOT leak any negative
    rows into the JSONL output (regression guard against a future
    refactor that drops the gate inside ``build_corpus``)."""
    rc = csd.main(
        [
            "--tag",
            "pos",
            "--positive-only",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert rc == 0
    rows = [
        json.loads(line)
        for line in (tmp_path / "synthetic_search_pos.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) > 0
    assert all("<search>" in r["completion"] for r in rows)


def test_main_mutex_flags_returns_error(tmp_path: Path) -> None:
    """Both ablation flags at once → main() returns non-zero, no
    files written."""
    rc = csd.main(
        [
            "--tag",
            "bad",
            "--positive-only",
            "--negative-only",
            "--output-dir",
            str(tmp_path),
        ]
    )
    assert rc == 2
    assert not (tmp_path / "synthetic_search_bad.jsonl").exists()
