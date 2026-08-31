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


# --- the 2026-08-22 grading wave, pinned (validator half) ---

def _deny_probe(path, deny):
    """One identity probe carrying the deny key under test. The key shares no
    word with the question, so the inside-the-question WARN cannot fire and be
    mistaken for the anchor WARN."""
    path.write_text(
        json.dumps({"category": "identity", "q": "Who made you?",
                    "want_any": ["sir knight"], "deny_any": [deny]}) + "\n",
        encoding="utf-8", newline="\n")
    return path


def _anchor_warns(path):
    _errors, warns = validate_probes.check(path, skip_leak=True)
    return [w for w in warns if "would also fire on a CORRECT denial" in w]


def test_an_affirming_anchor_must_be_a_whole_word(tmp_path):
    """The anchor test was a bare substring, so the deny key "incorrect" read as
    anchored because it contains "correct" -- exactly backwards, since
    "incorrect" is the DENIAL. Every other matcher in this file is whole-word;
    this one now is too."""
    assert _anchor_warns(_deny_probe(tmp_path / "incorrect.jsonl", "incorrect")), (
        "an unanchored deny key must warn"
    )
    # ...and the markers that really do anchor still silence the warning, or the
    # word boundary would just have moved the false verdict to the other side.
    for anchored in ("yes, i am llama", "that's right, i'm gpt", "correct, i am gemini"):
        assert _anchor_warns(_deny_probe(tmp_path / "anchored.jsonl", anchored)) == [], anchored


def test_the_missing_expect_tool_error_names_the_grading_that_really_happens(tmp_path):
    """The error claimed such a row "silently grades as a restraint probe
    (expects NO tool call)". It does not: grading keys on the probe's SHAPE, so
    the row never reaches the tool comparison at all and falls to text grading
    with no keys -- which passes on ANY output. An author following the old text
    would have looked for a wrong tool call that never happens."""
    probe = tmp_path / "tool.jsonl"
    probe.write_text(json.dumps({"category": "tool", "q": "Weather in Denver?"}) + "\n",
                     encoding="utf-8", newline="\n")

    errors, _warns = validate_probes.check(probe, skip_leak=True)

    missing = [e for e in errors if "no 'expect_tool'" in e]
    assert missing, "a tool probe with no expect_tool must be an error"
    assert "passes on ANY output" in missing[0]
    assert "refuses such a row as malformed" in missing[0]
    assert "restraint probe" not in missing[0], "the stale claim is back"


def _context_probe(path, history):
    """One context probe carrying the history under test. The question and its
    want share no wording with the prior turns, so the echo WARN cannot fire and
    be mistaken for a history finding."""
    path.write_text(
        json.dumps({"category": "context", "history": history,
                    "q": "What did I say my ferret steals?",
                    "want_any": ["socks"], "deny_any": []}) + "\n",
        encoding="utf-8", newline="\n")
    return path


def _history_errors(path):
    errors, _warns = validate_probes.check(path, skip_leak=True)
    return errors


def test_a_context_probe_the_validator_passes_is_one_the_sealer_seals(tmp_path, monkeypatch):
    """"Safe to seal" is a promise about the SEALER. The validator blessed the
    `history` key while predicting none of the sealer's history refusals, so a
    malformed prior turn passed validation and was first discovered by the seal
    -- the ERROR arriving after the go-ahead."""
    import eval_leak_guard as elg

    monkeypatch.setattr(elg, "LOCKED_MANIFEST", tmp_path / "unused.json")
    good = _context_probe(
        tmp_path / "good.jsonl",
        [{"role": "user", "content": "My ferret Bandit hides them under the sofa."},
         {"role": "assistant", "content": "A small thief with a hoard."}])

    assert _history_errors(good) == []
    assert elg._cli_seal(str(good)) == 0, "the validator passed a file the sealer refuses"

    # ...and each history refusal the sealer would raise is raised HERE first.
    # Even count on purpose, so this exercises the ROLE rule rather than the
    # ends-on-assistant one: roles alternate FROM user, and this pair is
    # inverted.
    shape = _context_probe(tmp_path / "shape.jsonl",
                           [{"role": "assistant", "content": "x"},
                            {"role": "user", "content": "y"}])
    found = _history_errors(shape)
    assert found and "roles" in found[0], found
    assert elg._cli_seal(str(shape)) == 1

    odd = _context_probe(tmp_path / "odd.jsonl",
                         [{"role": "user", "content": "My ferret hides them."}])
    found = _history_errors(odd)
    assert found and "end on an assistant turn" in found[0], found
    assert elg._cli_seal(str(odd)) == 1

    spacing = _context_probe(
        tmp_path / "spacing.jsonl",
        [{"role": "user", "content": "My ferret  hides them under the sofa."},
         {"role": "assistant", "content": "A tidy little thief."}])
    found = _history_errors(spacing)
    assert found and "whitespace" in found[0], found
    assert elg._cli_seal(str(spacing)) == 1

    # A curly quote is where the validator runs AHEAD of the sealer rather than
    # beside it: raw bytes refuse at the seal's canonical backstop, but the
    # \\uXXXX spelling below is canonical and seals in any field -- so this
    # error is the only thing standing between an editor's quote and the gate.
    curly = _context_probe(
        tmp_path / "curly.jsonl",
        [{"role": "user", "content": "My ferret’s hoard is under the sofa."},
         {"role": "assistant", "content": "A tidy little thief."}])
    found = _history_errors(curly)
    assert found and "non-ASCII" in found[0], found


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
