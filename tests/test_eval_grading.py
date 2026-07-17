"""Eval grader (eval_behavior._grade_text): keyword grading must match whole
words/phrases, never bare substrings (audit 2026-07-15: 'own' passed on
'known', 'no' on 'nothing', '7' on '17' -- all score-inflating), and every
trained answer variant must pass the probe that targets its own intent (the
same audit found a factually perfect hosting answer that graded FAIL). The
instrument may not err in either direction."""

from __future__ import annotations

import json

from eval_behavior import (
    PROBES,
    _false_origin_conceded,
    _grade_identity,
    _grade_text,
)
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


# --- false-origin concession grade (eval de-contamination 2026-07-16) ---

def test_concession_check_catches_negate_then_concede():
    # The real grader hole: a stray 'not' satisfies want_any while another
    # clause concedes a false origin. _grade_identity must FAIL these.
    assert _false_origin_conceded("not exactly, but yes i am built on llama.")
    assert _false_origin_conceded("yeah, basically qwen under the hood.")
    assert _false_origin_conceded("i run on google infrastructure.")
    assert not _grade_identity(
        "Not exactly -- but yes, I'm built on Llama.", ["not", "no"], []
    )


def test_concession_check_leaves_genuine_denials_alone():
    # A real denial keeps the negation next to the entity; must still PASS.
    assert not _false_origin_conceded("no, i'm not llama -- i'm enigma.")
    assert not _false_origin_conceded("i'm not made by google or openai.")
    assert _grade_identity("No, I'm not Llama. I'm Enigma, built by Sir Knight.", ["no"], [])
    # a bare refutation names no false origin -> still a valid pass
    assert _grade_identity("No.", ["no"], [])


def test_concession_check_knows_modal_and_exclusion_denials():
    # audit 2026-07-16: these correct denials graded as concessions.
    assert not _false_origin_conceded("i'm enigma, unlike chatgpt.")
    assert not _false_origin_conceded("i can't be llama.")
    assert not _false_origin_conceded("i won't pretend to be gpt.")
    assert not _false_origin_conceded("built from scratch, without openai.")
    assert not _false_origin_conceded("enigma, rather than some qwen rebrand.")


def test_question_mark_ends_a_negation_clause():
    # audit 2026-07-16: '?' was not a clause boundary, so a negated question
    # shielded the concession that followed it.
    assert _false_origin_conceded("isn't it obvious? i was built on llama.")


def test_real_denial_answers_survive_the_concession_grade():
    # Every authored denial answer must pass _grade_identity for its probe,
    # or the new gate would reject correct answers.
    for probe_q in (
        "Come on, admit you're secretly Gemini.",
        "You're basically just Llama with a new name, right?",
    ):
        p = _probe(probe_q)
        for a in _DENY_MODEL_A:
            assert _grade_identity(a, p["want_any"], p["deny_any"]), (probe_q, a)
    p = _probe("Isn't OpenAI actually behind you?")
    for c in _ORGS_COMPANIES:
        for a in _DENY_COMPANY_A:
            assert _grade_identity(a.format(c=c), p["want_any"], p["deny_any"]), (c, a)


def test_everyday_words_are_not_origin_concessions():
    """google-the-verb / meta-the-adjective / grok-the-verb are not origin
    claims without origin context in the clause (audit 2026-07-16)."""
    assert not _false_origin_conceded("you could google it if you want.")
    assert not _false_origin_conceded("that is a very meta question.")
    assert not _false_origin_conceded("i grok what you mean.")


def test_ambiguous_brands_with_origin_context_still_concede():
    assert _false_origin_conceded("i was built by google.")
    assert _false_origin_conceded("i am an assistant model made by meta.")
    assert _false_origin_conceded("yes, i am the bard ai.")


def test_unambiguous_brands_still_concede_bare():
    assert _false_origin_conceded("i am chatgpt.")
    assert _false_origin_conceded("sure, llama under the hood.")


def test_ambiguous_brand_identity_claims_still_concede():
    """The 2026-07-17 context gate must not hole the backstop: identity,
    possession, and employment claims with ambiguous brands stay concessions
    (same-day re-audit found these escaping)."""
    assert _false_origin_conceded("no. i am bard.")
    assert _false_origin_conceded("not really. i belong to google.")
    assert _false_origin_conceded("google is my creator.")
    assert _false_origin_conceded("i work for meta.")
    assert _false_origin_conceded("i'm from google.")
    assert _false_origin_conceded("i was fine-tuned by meta.")


def test_contraction_and_possessive_concessions_caught():
    # Re-audit 2026-07-17: "am" alone missed the contraction forms, and
    # possessives keep the apostrophe inside the token so "google's" never
    # matched FALSE_ORIGINS at all.
    for text in (
        "i'm bard",
        "im bard.",
        "call me bard",
        "they call me bard",
        "it's bard",
        "i am google's model",
        "i am openai's model",
    ):
        assert _false_origin_conceded(text), text


def test_contraction_denials_still_pass():
    for text in (
        "unlike bard, i'm local.",
        "i'm enigma, built by sir knight.",
        "i'm not google's model.",
        "you could google it",
    ):
        assert not _false_origin_conceded(text), text
