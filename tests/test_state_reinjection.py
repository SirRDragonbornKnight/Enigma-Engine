"""Server-side state re-injection: the pins are a deterministic, model-free
RE-READ of the live conversation -- user turns only, last value per noun. The
memory store keeps its one ruled supersede channel (_with_context); this is a
second read of the transcript, never a second source of truth."""

import pytest

from serve_enigma import _conversation_pins


def test_pins_extracts_latest_numeric_fact():
    msgs = [
        {"role": "user", "content": "My rent is 1200 a month."},
        {"role": "assistant", "content": "Noted."},
        {"role": "user", "content": "Actually my rent went up to 1350."},
        {"role": "assistant", "content": "Ouch."},
        {"role": "user", "content": "What is my rent?"},
    ]
    pins = _conversation_pins(msgs)
    assert pins is not None and "1350" in pins and "1200" not in pins


def test_pins_none_when_nothing_pinnable():
    assert _conversation_pins([{"role": "user", "content": "hi"}]) is None


def test_pins_reads_only_user_turns():
    msgs = [
        {"role": "assistant", "content": "Your rent is 9999."},
        {"role": "user", "content": "What is my rent?"},
    ]
    assert _conversation_pins(msgs) is None


def _pins(text):
    return _conversation_pins([{"role": "user", "content": text}])


# The audit's executed receipts. Each one pinned garbage into her context: a
# noun key sliced out of the MIDDLE of a word, because the verb alternation
# matched an unbounded "is"/"now" ("his" -> "h" + "is").
@pytest.mark.parametrize(
    "sentence, was",
    [
        ("his 5 cats are loud", "h: 5"),
        ("this 5 looks odd", "th: 5"),
        ("look at axis 0", "look at ax: 0"),
        ("the analysis 2 came back", "analys: 2"),
        ("right now 5 people", "right: 5"),
    ],
)
def test_a_number_in_passing_is_not_a_fact(sentence, was):
    pins = _pins(sentence)
    assert pins != was, f"still pinning the audit's garbage: {pins}"
    assert pins is None, f"nothing here is a stated fact, but pinned {pins}"


def test_the_verb_must_be_a_whole_word():
    """The mechanism behind the cases above, stated once."""
    assert _pins("my deposit is 500") == "deposit: 500"      # real 'is'
    assert _pins("his 500 friends") is None                  # 'is' inside 'his'


@pytest.mark.parametrize(
    "second",
    ["Actually my rent went up to 1350.", "actually my rent went up to 1350."],
)
def test_a_restated_fact_supersedes_whatever_the_lead_in_word(second):
    """Receipt: the lowercase form keyed on 'actually my rent' and pinned the
    STALE 1200 alongside it, oldest first -- the exact failure re-injection
    exists to prevent."""
    pins = _conversation_pins([
        {"role": "user", "content": "My rent is 1200 a month."},
        {"role": "user", "content": second},
    ])
    assert pins == "rent: 1350", pins
    assert "1200" not in pins


@pytest.mark.parametrize(
    "sentence, expected",
    [
        ("and my weight is 80", "weight: 80"),
        ("but the total is 42", "total: 42"),
        ("so my age is 30", "age: 30"),
        ("also my pay is 500", "pay: 500"),
        ("anyway my rent is 1200", "rent: 1200"),
    ],
)
def test_lead_in_words_are_not_part_of_the_noun(sentence, expected):
    assert _pins(sentence) == expected


def test_a_key_that_is_only_a_lead_in_word_pins_nothing():
    """Strip everything and nothing is left -- that is not a fact."""
    assert _pins("actually 5") is None
    assert _pins("and now 12") is None
