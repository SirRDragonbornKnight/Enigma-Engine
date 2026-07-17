"""The chat/tool token format (instruct-pass foundation, chat_format.py).

Contracts locked here:
- attaching chat tokens NEVER changes how plain text encodes (the BPE tables
  and tokens.bin compatibility stay byte-identical);
- all chat token IDs live in the padded free rows [4718, 4736);
- render -> parse round-trips tool calls and thinking spans;
- the training mask marks assistant content (+ its <|im_end|> + final EOS)
  and nothing else.
"""

import pytest

from enigma_engine.core import chat_format as cf
from enigma_engine.core.tokenizer import get_tokenizer


@pytest.fixture(scope="module")
def tok():
    t = get_tokenizer("bpe")
    cf.attach_chat_tokens(t)
    return t


def test_chat_token_ids_live_in_the_padded_rows(tok):
    for s, i in cf.CHAT_TOKENS.items():
        assert cf.BASE_VOCAB <= i < cf.PADDED_VOCAB, (s, i)
    # and they collide with nothing the tokenizer already had
    assert len(set(cf.CHAT_TOKENS.values())) == len(cf.CHAT_TOKENS)


def test_attach_is_idempotent_and_plain_text_is_untouched(tok):
    sample = "Hello world! The ocean is <think>deep</think> blue."
    before = tok.encode(sample)
    cf.attach_chat_tokens(tok)  # second attach: no-op
    assert tok.encode(sample) == before
    # specials now encode to single IDs
    assert tok.encode("<|im_start|>", add_special_tokens=False) == [cf.IM_START]
    assert tok.encode("<|/tool_call|>", add_special_tokens=False) == [cf.TOOL_CALL_END]


def test_render_chat_shape_and_generation_prompt(tok):
    msgs = [{"role": "system", "content": "You are Enigma."}, {"role": "user", "content": "Hi!"}]
    ids = cf.render_chat(tok, msgs, add_generation_prompt=True)
    assert ids[0] == tok.bos_token_id
    assert ids[1] == cf.IM_START
    assert ids.count(cf.IM_START) == 3  # system, user, generation prompt
    assert ids.count(cf.IM_END) == 2  # system, user (assistant is hers)
    assert ids[-1] != cf.IM_END  # ends mid-assistant-header
    tail = tok.decode(ids[-6:], skip_special_tokens=True)
    assert "assistant" in tail


def test_tool_call_render_parse_roundtrip(tok):
    msgs = [
        {"role": "user", "content": "Make her wave."},
        {
            "role": "assistant",
            "content": "Done.",
            "tool_calls": [{"name": "avatar_express", "arguments": {"emotion": "happy", "wave": True}}],
        },
    ]
    ids, mask = cf.render_training(tok, msgs)
    trainable = [t for t, m in zip(ids, mask) if m]
    out = cf.parse_assistant_ids(tok, trainable)
    assert out["content"] == "Done."
    assert out["tool_calls"] == [{"name": "avatar_express", "arguments": {"emotion": "happy", "wave": True}}]


def test_nameless_json_tool_call_keeps_raw(tok):
    # Valid JSON but no "name" key ({"tool": ...}): the action must surface
    # as a raw call, not vanish -- serve's filters can only see a name-less
    # call through its "raw" text (ultrareview #15).
    body = '{"tool": "calculate", "arguments": {"expression": "7*8"}}'
    ids = (
        tok.encode("Sure.", add_special_tokens=False)
        + [cf.TOOL_CALL]
        + tok.encode(body, add_special_tokens=False)
        + [cf.TOOL_CALL_END]
    )
    out = cf.parse_assistant_ids(tok, ids)
    assert out["content"] == "Sure."
    assert len(out["tool_calls"]) == 1
    call = out["tool_calls"][0]
    assert call["name"] is None
    assert call.get("raw"), "name-less call lost its raw text"
    assert "calculate" in call["raw"]


def test_tool_result_role_is_wrapped(tok):
    ids, _ = cf.render_training(tok, [{"role": "tool", "content": "ok"}])
    assert cf.TOOL_RESULT in ids and cf.TOOL_RESULT_END in ids


def test_think_span_extracted_via_native_tokens(tok):
    msgs = [{"role": "user", "content": "hm?"}, {"role": "assistant", "content": "<think>plan it</think>Answer."}]
    ids, mask = cf.render_training(tok, msgs)
    assert cf.THINK in ids and cf.THINK_END in ids
    out = cf.parse_assistant_ids(tok, [t for t, m in zip(ids, mask) if m])
    assert out["thinking"] == "plan it"
    assert out["content"] == "Answer."


def test_training_mask_covers_only_assistant_plus_stops(tok):
    msgs = [
        {"role": "system", "content": "Be kind."},
        {"role": "user", "content": "Hello there, who are you?"},
        {"role": "assistant", "content": "I am Enigma."},
    ]
    ids, mask = cf.render_training(tok, msgs)
    assert len(ids) == len(mask)
    trues = [t for t, m in zip(ids, mask) if m]
    assert cf.IM_END in trues  # she learns to close her turn
    assert tok.eos_token_id in trues  # and to end the document
    assert cf.IM_START not in trues  # headers are given, not learned
    # nothing before the assistant's start position is trainable
    a_start = len(ids) - 1 - ids[::-1].index(cf.IM_START)
    assert not any(mask[:a_start])
    # a conversation with no assistant turn trains on nothing
    ids2, mask2 = cf.render_training(tok, msgs[:2])
    assert not any(mask2)


def test_trim_keeps_system_and_newest_turn(tok):
    long = "word " * 60
    msgs = (
        [{"role": "system", "content": "SYS"}]
        + [{"role": "user", "content": long}, {"role": "assistant", "content": long}] * 4
        + [{"role": "user", "content": "newest question"}]
    )
    ids = cf.render_chat(tok, msgs, add_generation_prompt=True, max_ids=160)
    assert len(ids) <= 160
    text = tok.decode(ids, skip_special_tokens=True)
    assert "SYS" in text
    assert "newest question" in text


def test_unknown_role_raises(tok):
    with pytest.raises(ValueError):
        cf.render_chat(tok, [{"role": "narrator", "content": "x"}])


def test_render_tools_system_accepts_openai_nesting():
    flat = {"name": "get_weather", "description": "weather", "parameters": {"city": "string"}}
    nested = {"type": "function", "function": flat}
    for spec in (flat, nested):
        text = cf.render_tools_system([spec])
        assert "get_weather" in text
        assert cf.TOOL_SYNTAX in text
    assert cf.render_tools_system([]) == ""
    assert cf.render_tools_system(None) == ""
