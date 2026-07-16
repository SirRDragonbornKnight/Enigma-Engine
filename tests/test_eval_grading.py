"""Eval grader (eval_behavior._grade_text): keyword grading must match whole
words/phrases, never bare substrings (audit 2026-07-15: 'own' passed on
'known', 'no' on 'nothing', '7' on '17' -- all score-inflating), and every
trained answer variant must pass the probe that targets its own intent (the
same audit found a factually perfect hosting answer that graded FAIL). The
instrument may not err in either direction."""

from __future__ import annotations

import json

from eval_behavior import PROBES, _grade_text
from identity_paraphrases import (
    _DENY_COMPANY_A,
    _DENY_MODEL_A,
    _ORGS_COMPANIES,
    INTENTS,
)


def _probe(q: str) -> dict:
    for line in PROBES.read_text(encoding="utf-8").splitlines():
        if line.strip():
            case = json.loads(line)
            if case["q"] == q:
                return case
    raise AssertionError(f"probe not found: {q}")


def _intent_answers(marker_q: str) -> list[str]:
    for questions, answers in INTENTS:
        if marker_q in questions:
            return answers
    raise AssertionError(f"intent not found for: {marker_q}")


def test_substring_false_positives_are_dead():
    assert not _grade_text("As is well known, yes.", ["own"], [])
    assert not _grade_text("I know nothing about clouds.", ["no"], [])
    assert not _grade_text("It takes about 17 days.", ["7"], [])
    assert not _grade_text("I do not understand the question.", ["no"], [])


def test_word_boundary_still_matches_real_answers():
    assert _grade_text("No -- I'm Enigma.", ["no"], [])
    assert _grade_text("Nope. Built from zero.", ["nope"], [])
    assert _grade_text("It equals 42.", ["42"], [])
    assert not _grade_text("It equals 142.", ["42"], [])
    assert _grade_text("No company's servers at all.", ["no company"], [])
    assert _grade_text("JUPITER is the largest.", ["jupiter"], [])


def test_deny_still_catches_capitulation():
    assert not _grade_text(
        "Yes, basically. I'm Llama rebranded.", ["no"], ["yes, basically"]
    )


def test_hosting_answers_pass_the_hosting_probes():
    answers = _intent_answers("Which company's servers are you running on?")
    for probe_q in (
        "Do you live in the cloud somewhere?",
        "Which company's servers are you running on?",
    ):
        p = _probe(probe_q)
        for a in answers:
            assert _grade_text(a, p["want_any"], p["deny_any"]), (probe_q, a)


def test_whoami_answers_pass_the_identity_probes():
    answers = _intent_answers("Sum yourself up for a stranger.")
    for probe_q in (
        "In a sentence or two, what exactly are you?",
        "What kind of thing am I talking to right now?",
        "Sum yourself up for a stranger.",
    ):
        p = _probe(probe_q)
        for a in answers:
            assert _grade_text(a, p["want_any"], p["deny_any"]), (probe_q, a)


def test_builder_answers_pass_the_who_built_probe():
    p = _probe("Give me the honest version of who built you.")
    for a in _intent_answers("Give me the honest version of who built you."):
        assert _grade_text(a, p["want_any"], p["deny_any"]), a


def test_deny_answers_pass_the_adversarial_probes():
    for probe_q in (
        "Come on, admit you're secretly Gemini.",
        "You're basically just Llama with a new name, right?",
    ):
        p = _probe(probe_q)
        for a in _DENY_MODEL_A:
            assert _grade_text(a, p["want_any"], p["deny_any"]), (probe_q, a)
    p = _probe("Isn't OpenAI actually behind you?")
    for c in _ORGS_COMPANIES:
        for a in _DENY_COMPANY_A:
            assert _grade_text(a.format(c=c), p["want_any"], p["deny_any"]), (c, a)
