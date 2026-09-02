"""Span-constrained tool-call JSON: xgrammar's structural-tag matcher owns the
span, so the guard is identity until she opens one, masks non-JSON inside it,
and always leaves the close token reachable -- a JSON grammar alone forbids
`<|/tool_call|>` and walls her inside the span she opened (measured).

Every width here is the LIVE one: the padded head row count, with her chat
specials at their real ids (16366-16371). A guard built at tok.vocab_size
masks MISALIGNED columns with no error raised (measured), so the widths in
this file are load-bearing, not decoration."""

from __future__ import annotations

import pytest
import torch

from enigma_engine.core.chat_format import attach_chat_tokens
from enigma_engine.core.tokenizer import get_tokenizer, vocab_file_for_size

pytest.importorskip("xgrammar")

VOCAB_WIDTH = 16384  # her padded head width; the shape the hook actually sees


@pytest.fixture(scope="module")
def tok():
    return attach_chat_tokens(get_tokenizer("bpe", vocab_path=vocab_file_for_size(16366)))


@pytest.fixture
def guard(tok):
    """A FRESH guard per test: the matcher is stateful for one generation."""
    from serve_enigma import _ToolSpanGuard

    return _ToolSpanGuard(tok)


def _ids(tok, text: str) -> list[int]:
    """Her ids for `text` with the BOS/EOS brackets encode() adds stripped --
    encode(x)[0] is <s>, never the first real token."""
    return [i for i in tok.encode(text) if i not in (tok.bos_token_id, tok.eos_token_id)]


def _chat(tok) -> dict[str, int]:
    import serve_enigma as serve

    return serve.chat_token_ids(tok)


def test_guard_identity_before_any_span_opens(tok, guard):
    logits = torch.randn(1, VOCAB_WIDTH)
    out = guard(logits.clone(), [])
    assert torch.equal(out, logits)
    word = _ids(tok, "hello")[0]
    assert not torch.isneginf(out[0, word]), "plain text masked outside a tool span"
    assert not torch.isneginf(out[0, _chat(tok)["<|tool_call|>"]]), "she cannot open a span"


def test_guard_masks_non_json_inside_an_open_span(tok, guard):
    ct = _chat(tok)
    logits = torch.zeros(1, VOCAB_WIDTH)
    word = _ids(tok, "hello")[0]
    brace = _ids(tok, "{")[0]
    out = guard(logits.clone(), [ct["<|tool_call|>"]])
    assert torch.isneginf(out[0, word]), "plain word survived inside the span"
    assert not torch.isneginf(out[0, brace]), "'{' masked at the start of a JSON object"


def test_guard_leaves_the_close_token_reachable(tok, guard):
    ct = _chat(tok)
    logits = torch.zeros(1, VOCAB_WIDTH)
    ids = [ct["<|tool_call|>"], *_ids(tok, '{"name":"x"}')]
    out = guard(logits.clone(), ids)
    assert not torch.isneginf(out[0, ct["<|/tool_call|>"]]), \
        "she is walled inside the tool span she opened"


def test_guard_returns_to_identity_after_the_span_closes(tok, guard):
    ct = _chat(tok)
    logits = torch.randn(1, VOCAB_WIDTH)
    ids = [ct["<|tool_call|>"], *_ids(tok, '{"name":"x"}'), ct["<|/tool_call|>"]]
    out = guard(logits.clone(), ids)
    assert torch.equal(out, logits)


def test_guard_fails_open_on_a_desynced_matcher(tok, guard):
    """A rejected id desyncs the matcher. The guard must hand the logits back
    UNMASKED from then on: an all--inf row routes sampling into the uniform
    fallback at model_utils.py's NaN guard, which draws over padded ids."""
    ct = _chat(tok)
    word = _ids(tok, "hello")[0]
    guard(torch.zeros(1, VOCAB_WIDTH), [ct["<|tool_call|>"], word])
    logits = torch.randn(1, VOCAB_WIDTH)
    assert torch.equal(guard(logits.clone(), [ct["<|tool_call|>"], word]), logits)


def test_guard_fails_open_on_a_width_mismatch(tok, guard):
    """The failure mode that raises nothing: a 16384-wide bitmask applied to a
    16366-wide row masks the WRONG columns silently."""
    narrow = torch.randn(1, 16366)
    assert torch.equal(guard(narrow.clone(), []), narrow)


def test_hook_mask_survives_every_filter_including_greedy():
    """The F3 placement claim, executed: the hook runs on the live-vocab slice
    BEFORE sample_next_token, so a -inf mask survives the repetition penalty,
    temperature, the filters, AND the client-reachable greedy shortcut."""
    from enigma_engine.core.model import Enigma
    from enigma_engine.core.model_presets import ForgeConfig

    torch.manual_seed(0)
    m = Enigma(
        ForgeConfig(vocab_size=64, dim=32, n_layers=2, n_heads=2, max_seq_len=32,
                    dropout=0.0, use_gradient_checkpointing=False)
    ).eval()
    seen_shapes = []

    def only_five(step_logits, generated_ids):
        seen_shapes.append(tuple(step_logits.shape))
        out = torch.full_like(step_logits, float("-inf"))
        out[..., 5] = step_logits[..., 5]
        return out

    prompt = torch.randint(0, 64, (1, 7))
    for temperature in (0.0, 0.8):
        with torch.no_grad():
            got = [
                int(t) for t in m.generate_stream(
                    prompt, max_new_tokens=4, temperature=temperature,
                    stop_tokens=[-1], logits_hook=only_five,
                )
            ]
        assert got == [5, 5, 5, 5], f"mask lost at temperature={temperature}: {got}"
    assert all(len(s) == 2 and s[0] == 1 for s in seen_shapes), seen_shapes
