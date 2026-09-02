"""Server-side state re-injection: the pins are a deterministic, model-free
RE-READ of the live conversation -- user turns only, last value per noun. The
memory store keeps its one ruled supersede channel (_with_context); this is a
second read of the transcript, never a second source of truth."""

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
