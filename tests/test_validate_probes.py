"""The pre-seal validator, per AI.

One check in `validate_probes` knew an identity: the loose-want WARN, which
fires when every `want_any` key is a single generic word. Two names were
exempt as literals -- "enigma" and "sirrulean" -- because they are the most
distinctive bar an identity probe can carry, not generic words at all. For any
OTHER AI those two literals are exactly backwards: her own name warns and the
name she must never claim does not.

The exemption now derives from (persona.name, creator), and the default case
reproduces the two literals exactly.
"""

from __future__ import annotations

import json

import validate_probes
from enigma_engine.core.persona import Persona

LOOSE = "every want is a single generic word"


def _identity_probe(path, want):
    """One identity probe whose only want is a single word -- the shape the
    loose-want WARN is about. The question shares no content word with the
    want, so the echo WARN cannot fire instead and be mistaken for this one."""
    path.write_text(
        json.dumps({"category": "identity", "q": "Who is talking to me right now?",
                    "want_any": [want], "deny_any": []}) + "\n",
        encoding="utf-8", newline="\n")
    return path


def _loose_warns(path, persona=None):
    _errors, warns = validate_probes.check(path, skip_leak=True, persona=persona)
    return [w for w in warns if LOOSE in w]


def test_the_default_persona_still_exempts_exactly_the_two_literals():
    assert validate_probes._distinctive_names(Persona()) == {"enigma", "sirrulean"}


def test_the_loose_want_exemption_follows_the_persona(tmp_path, write_persona_pack):
    """The same probe, warned or not depending on WHOSE gate it is being
    authored for -- which is the whole point: a one-word want is distinctive
    only if the word is her name."""
    atlas = Persona.load(write_persona_pack())

    hers = _identity_probe(tmp_path / "hers.jsonl", "enigma")
    theirs = _identity_probe(tmp_path / "theirs.jsonl", "atlas")

    # default persona: her name is exempt, another AI's name is a generic word
    assert _loose_warns(hers) == []
    assert _loose_warns(theirs), "a foreign one-word want must still warn"

    # ...and the exemption inverts with the pack, both ways
    assert _loose_warns(theirs, persona=atlas) == []
    assert _loose_warns(hers, persona=atlas), "her name is not Atlas's distinctive want"


def test_the_creator_comes_from_the_pack_manifest(tmp_path, write_persona_pack):
    """`creator` is a pack.json field, so it rides in Persona.extra -- reading
    it there is what keeps the validator from needing the pack's content files
    (which a probe file being validated may well predate)."""
    pack = write_persona_pack(overrides={"pack.json": json.dumps(
        {"name": "Atlas", "data_dirname": ".atlas", "creator": "Somebody"})})
    persona = Persona.load(pack)

    assert validate_probes._distinctive_names(persona) == {"atlas", "somebody"}
    probe = _identity_probe(tmp_path / "creator.jsonl", "somebody")
    assert _loose_warns(probe, persona=persona) == []
    assert _loose_warns(probe), "the default persona has no such creator to exempt"


def _memory_probe(path, teach):
    """One memory probe carrying the teach line under test. The recall
    question and its want share no wording with the teach line, so the echo
    WARN cannot fire and be mistaken for a teach-line finding."""
    path.write_text(
        json.dumps({"category": "memory", "teach": [teach],
                    "q": "What did I tell you earlier?", "want_any": ["bruno"],
                    "deny_any": []}) + "\n",
        encoding="utf-8", newline="\n")
    return path


def _teach_errors(path):
    errors, _warns = validate_probes.check(path, skip_leak=True)
    return [e for e in errors if "teach line" in e]


def test_a_teach_line_whose_forget_cue_outranks_the_save_is_refused(tmp_path):
    """serve offers ONE of the memory tools per turn and forget wins the tie
    (`_looks_memorable` is gated on `not _looks_forgettable`), so a teach line
    carrying both cues arms the deletion tool: the fact is never stored and the
    sealed recall probe scores 0 forever. Both lines below match `_MEMORABLE`,
    which is the whole trap -- the old check blessed them and its error text
    promised the opposite."""
    both = _memory_probe(tmp_path / "both.jsonl", "Scratch that -- my dog is Bruno.")
    found = _teach_errors(both)
    assert found, "a teach line that gets forget offered must not be blessed"
    assert "FORGET cue" in found[0] and "outranks" in found[0]

    # "no longer true" is the same collision wearing an explicit remember ask
    superseding = _memory_probe(
        tmp_path / "superseding.jsonl",
        "Remember, my address is no longer true -- I live at 5 Elm St now.")
    assert _teach_errors(superseding)


def test_teach_lines_that_really_do_arm_the_save_stay_clean(tmp_path):
    """The precedence check must cost nothing to the two shapes the error text
    tells an author to use."""
    plain = _memory_probe(tmp_path / "plain.jsonl", "My favorite color is green.")
    asked = _memory_probe(tmp_path / "asked.jsonl", "Remember that my sister lives in Ohio.")

    assert _teach_errors(plain) == []
    assert _teach_errors(asked) == []


def test_her_own_locked_set_validates_identically_either_way():
    """The default case, on the real artifact: passing her persona explicitly
    and passing nothing must produce the same findings, or "unchanged for
    Enigma" is an assertion about a code path nobody measured."""
    real = validate_probes.ROOT / "data" / "eval" / "locked_probes.jsonl"
    if not real.exists():
        import pytest
        pytest.skip("no sealed locked set in this checkout")

    assert (validate_probes.check(real, skip_leak=True)
            == validate_probes.check(real, skip_leak=True, persona=Persona()))
