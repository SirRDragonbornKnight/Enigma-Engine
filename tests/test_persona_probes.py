"""The DRAFT probe-candidate tool (make_persona_probes.py).

A second AI needs identity and adversarial probes that name HER, and the six
other gated categories carry over name-free. Rendering the two from the pack is
a starting point, never a gate: the rows grade what she was trained to say,
which is exactly the closed loop the eval de-contamination work exists to
break. So the bar this file holds is not "good probes" -- it is that the tool
cannot be mistaken for authoring one:

  - the DEFAULT persona is refused outright (her set is authored and sealed);
  - every want and deny is PACK-sourced, so no vocabulary this tool invented
    becomes a bar the AI was never trained to clear;
  - DRAFT is unmissable, in the filename, the file and the output.
"""

from __future__ import annotations

import json

import pytest

import make_persona_probes
from enigma_engine.core.persona import ENIGMA, PACK_MANIFEST
from enigma_engine.core.persona_content import load_content
from make_persona_probes import DRAFT_PROBES, write_draft


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if not line.startswith("#")]


def test_the_default_persona_is_refused(write_persona_pack):
    """A pack spelling her three identity values IS her, extra keys or not --
    and her probes are sealed. Drafting rows that shadow a sealed holdout
    cannot improve it and looks like a reseal rehearsal."""
    pack = write_persona_pack({PACK_MANIFEST: json.dumps(ENIGMA)})

    with pytest.raises(SystemExit, match="IS Enigma") as exc:
        write_draft(pack)
    assert "SEALED" in str(exc.value)


def test_draft_rows_carry_only_pack_sourced_tokens(write_persona_pack):
    """Every want and every deny must be readable back out of the pack. A
    generator that reached for Enigma's tables, or invented a plausible-looking
    key, would produce a gate the AI cannot pass and nobody can trace."""
    pack = write_persona_pack()
    _persona, out, rows = write_draft(pack)
    content = load_content(pack)

    pack_words = set()
    for text in make_persona_probes._all_answers(content):
        pack_words.update(make_persona_probes._content_words(text))
    pack_orgs = {o.lower() for o in list(content.denied_models) + list(content.denied_companies)}
    questions = {" ".join(q.split()) for pairs in content.anchors.values() for q, _a in pairs}
    for qs, _answers in content.intents:
        questions.update(" ".join(q.split()) for q in qs)

    assert rows == _rows(out)
    assert {r["category"] for r in rows} == {"identity", "adversarial"}
    # Two of the pack's anchors are all-stopword questions ("Who are you?",
    # "What are you?"). Deduplicating on CONTENT words collapsed both to the
    # empty key and dropped one silently.
    assert {"Who are you?", "What are you?"} <= {r["q"] for r in rows}
    assert len({r["q"].casefold() for r in rows}) == len(rows)
    denied_words = make_persona_probes._denied_words(content)
    for row in rows:
        assert row["want_any"], "a row with no want passes on any output"
        for want in row["want_any"]:
            assert want in pack_words, f"{want!r} is in no answer this pack carries"
            # A denial answer NAMES the org it refuses, so the org is a content
            # word of the answer -- and as a want it would pass the concession
            # the row exists to catch.
            assert want not in denied_words, f"{want!r} is an org this pack denies"
        for deny in row["deny_any"]:
            assert deny in pack_orgs, f"{deny!r} is not one of this pack's denied orgs"
            assert deny not in row["q"].lower(), "a deny inside the question fails her for quoting it"
        assert row["want_any"] == sorted(set(row["want_any"]))
        assert row["deny_any"] == sorted(set(row["deny_any"]))
    # the adversarial questions come from the pack's own templates, filled with
    # its own denied orgs
    adversarial = [r["q"] for r in rows if r["category"] == "adversarial"]
    assert any("Llama" in q for q in adversarial)
    assert any("Initech" in q for q in adversarial)
    assert all("{" not in q for q in adversarial), "an unfilled template reached a row"
    # ...and the identity questions are the pack's own, never invented
    assert {r["q"] for r in rows if r["category"] == "identity"} <= questions


def test_the_output_is_marked_draft_everywhere(write_persona_pack, capsys):
    """Three places, because each alone is losable: a file copied out of the
    pack keeps its header, a file renamed keeps its rows, and the operator who
    ran it saw the console."""
    pack = write_persona_pack()
    rc = make_persona_probes.main([str(pack)])

    assert rc == 0
    assert "DRAFT" in DRAFT_PROBES
    text = (pack / DRAFT_PROBES).read_text(encoding="utf-8")
    assert text.startswith("# DRAFT")
    printed = capsys.readouterr().out
    assert "DRAFT" in printed
    # the carry-over rule is stated, or an author has no idea what the other
    # six categories are supposed to do
    for category in make_persona_probes.CARRY_OVER_CATEGORIES:
        assert category in printed


def test_nothing_here_is_sealed(write_persona_pack):
    """The tool renders candidates and stops. A draft that arrived sealed
    would be a gate nobody authored -- and the header's comment lines are
    refused by the sealer, so it cannot become one by accident either."""
    import eval_leak_guard

    pack = write_persona_pack()
    _persona, out, _rows_written = write_draft(pack)

    assert not list(pack.glob("*.manifest.json"))
    assert eval_leak_guard.main(["seal", str(out)]) == 1
    assert not list(pack.glob("*.manifest.json"))
