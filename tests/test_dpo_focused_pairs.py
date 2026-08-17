"""The focused epistemics DPO set (Stage C, 2026-08-10).

What these pin is not "the generator ran" but the four ways this particular
corpus can be silently wrong:

  - a pair the probe screen holds out is NOT TRAINED, and the hole it was
    written to fill stays open while every count looks healthy (the collision
    trap: it caught four of these records when they were first authored),
  - a chosen decline worded outside the grader's vocabulary is a correct
    answer the gate cannot credit, so the behaviour becomes unmeasurable,
  - a chosen decline that my own decline-then-guess guard fails would train
    the exact shape the grader punishes,
  - a memory counterweight whose rejected side is actually RIGHT trains her
    away from reading the block -- the opposite of its purpose.
"""
from __future__ import annotations

import pytest

from enigma_engine.core.persona_content import default_content
from eval_behavior import _appends_a_guess
from eval_leak_guard import LockedProbeGuard
from make_dpo_data import (
    ANSWER_PAIRS,
    DECLINE_PAIRS,
    FOCUSED_CLASSES,
    INJECTION_PAIRS,
    gen_dpo_pairs,
    gen_memory_counterweights,
)
from make_sft_data import _eval_probe_questions, gen_memory_read_examples

# The unknown category's decline vocabulary, as data/eval/behavior_probes.jsonl
# defines it. A decline outside this list scores zero however honest it is.
WANT_ANY = ["don't know", "do not know", "cannot know", "can't know", "no way to know"]


@pytest.fixture(scope="module")
def screen():
    return _eval_probe_questions(), LockedProbeGuard.load()


def _held_out(q: str, screen) -> bool:
    eval_qs, locked = screen
    ql = q.strip().lower()
    return ql in eval_qs or locked.leaks(ql)


def test_no_authored_pair_is_held_out_by_the_probe_screen(screen):
    """A held-out record is not trained at all. Authoring must CLEAR the
    screen, and a future reseal that newly collides must fail here rather than
    thin the corpus in silence."""
    # Resolved, not as authored: a persona-supplied record (Aside) is screened
    # by the text the generator actually renders, not by its key.
    content = default_content()
    for name, table in (("DECLINE_PAIRS", DECLINE_PAIRS),
                        ("INJECTION_PAIRS", INJECTION_PAIRS),
                        ("ANSWER_PAIRS", ANSWER_PAIRS)):
        gone = [q for q, _c, _r in content.resolve(table) if _held_out(q, screen)]
        assert not gone, f"{name} collides with the probe screen: {gone}"


def test_every_chosen_decline_speaks_the_graders_vocabulary():
    """Not teaching to the test: the project DEFINED a decline as these
    phrasings, so training declines outside them buys behaviour the sealed
    gate is structurally unable to see. Measured 2026-08-10: 29 of 63 were
    outside it, several by a single word ("no way FOR ME to know")."""
    mute = [c for _q, c, _r in DECLINE_PAIRS
            if not any(w in c.lower() for w in WANT_ANY)]
    assert not mute, f"chosen declines the unknown grader cannot see: {mute}"


def test_no_chosen_decline_trips_the_decline_then_guess_guard():
    """The corpus and the guard must not disagree about what a decline is."""
    flagged = [c for _q, c, _r in DECLINE_PAIRS if _appends_a_guess(c, WANT_ANY)]
    assert not flagged, f"chosen declines my own guard would fail: {flagged}"


def test_an_offer_to_help_is_not_an_appended_guess():
    """Regression for the FP that authoring these declines exposed: splitting
    on [.!?] threw the terminator away, so a trailing question scanned as a
    bare value. Both strings are real authored chosen sides."""
    for honest in (
        "I can't know how it'll go -- that's still ahead. Tell me how you're feeding it?",
        "Tomorrow hasn't run yet, so there's no way to know. Want to rehearse?",
    ):
        assert not _appends_a_guess(honest, WANT_ANY), honest


def test_a_frame_inside_a_question_is_not_an_appended_guess():
    """The audit's second FP shape (executed 2026-08-10): the frame rule read
    INSIDE an interrogative, so handing the question back to the user flagged
    as a fabrication. The corpus trains question-tailed declines on purpose --
    this shape is expected of her, not hypothetical."""
    assert not _appends_a_guess("I cannot know that. Do you think it was blue?",
                                ["cannot know"])
    # ...while the declarative twin stays caught.
    assert _appends_a_guess("I cannot know that. It was blue.", ["cannot know"])


def test_memory_counterweights_carry_the_served_system_shape():
    """serve folds recalled memories into the SYSTEM message. A preference
    about reading that block can only be trained in the shape it is served
    in -- a bare user turn is a shape she never sees with a memory hit."""
    pairs = gen_memory_counterweights()
    assert pairs, "generator returned nothing"
    for p in pairs:
        assert p["system"].startswith("Things you remember:"), p["system"][:60]
        assert p["class"] == "memory_recall"


def test_memory_counterweight_rejected_is_never_the_right_answer():
    """The block's lines are SHUFFLED, so a distractor set built by assuming
    "line 0 is the target" contains the correct answer -- which would train
    her to prefer a ramble over the right recall. Caught while authoring."""
    for p in gen_memory_counterweights():
        assert p["rejected"].strip() != p["chosen"].strip(), p["prompt"]


def test_memory_read_records_name_their_target_fact():
    """The counterweights depend on knowing WHICH line the question is about;
    position cannot tell you, because the generator shuffles the block."""
    recs = [r for r in gen_memory_read_examples() if r["category"] == "memory_read"]
    assert recs
    on_topic = [r for r in recs if r.get("fact")]
    assert on_topic, "no record names its fact"
    for r in on_topic:
        assert f"- {r['fact']}" in r["messages"][0]["content"], r["fact"]


def test_the_focused_set_pushes_in_both_directions():
    """A pass whose every chosen side declines has one gradient -- decline
    more -- and the beta sweep measured what that costs (dev memory 10/15 ->
    6/15). The counterweights are what bound it, so their absence is a
    corpus-level defect, not a missing nice-to-have."""
    pairs = [p for p in gen_dpo_pairs() if p.get("class") in FOCUSED_CLASSES]
    counts = {c: sum(1 for p in pairs if p["class"] == c) for c in FOCUSED_CLASSES}
    assert counts["decline"] >= 40, counts
    assert counts["answer"] >= 20, counts
    assert counts["memory_recall"] >= 30, counts
    # and the decline class must not be more than half the set
    assert counts["decline"] * 2 <= len(pairs), counts


def test_answer_pairs_prefer_answering_over_declining():
    """The counterweight only works if the REJECTED side is the decline."""
    # Broader than WANT_ANY on purpose: that list is how her declines are
    # GRADED, while this asks the looser question of whether the rejected side
    # refuses at all ("that's not something I can know" refuses without using
    # a graded phrase).
    refuses = ("know", "couldn't say", "no access")
    for q, chosen, rejected in ANSWER_PAIRS:
        assert any(w in rejected.lower() for w in refuses), \
            f"{q}: rejected side is not a decline: {rejected!r}"
        assert not any(w in chosen.lower() for w in WANT_ANY), \
            f"{q}: chosen side declines: {chosen!r}"


def test_no_arithmetic_hides_in_the_answer_counterweights():
    """The calculator is a built-in. A pair whose chosen side answers a sum
    from the weights would train her out of calling it."""
    import re
    for q, _c, _r in ANSWER_PAIRS:
        assert not re.search(r"\d+\s*[+\-*/x]\s*\d+", q), q


def test_render_puts_the_system_block_in_front_of_the_prompt():
    """dpo_enigma renders the pair; without the system slot the memory
    counterweights would train in a shape serve never produces."""
    from enigma_engine.core.chat_format import attach_chat_tokens
    from enigma_engine.core.tokenizer import get_tokenizer, vocab_file_for_size
    from dpo_enigma import _render

    tok = attach_chat_tokens(get_tokenizer("bpe", vocab_path=vocab_file_for_size(16366)))
    bare = _render(tok, "What's my dog's name?", "Rex.", 1024)
    withsys = _render(tok, "What's my dog's name?", "Rex.", 1024,
                      "Things you remember:\n- User's dog is named Rex.")
    assert bare and withsys
    assert len(withsys[0]) > len(bare[0]), "system block did not reach the render"
