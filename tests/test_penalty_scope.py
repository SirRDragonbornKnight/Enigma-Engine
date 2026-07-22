"""Repetition-penalty scope (ultrareview #9 fix): generate/generate_stream
must penalize only tokens SHE produced, never the prompt -- penalizing prompt
tokens suppresses exactly the vocabulary the system prompt primes her to
answer with. Verified by capturing what sample_next_token actually receives."""

from __future__ import annotations

import torch

import enigma_engine.core.model as model_mod
from enigma_engine.core.model import Enigma
from enigma_engine.core.model_presets import ForgeConfig
from enigma_engine.core.model_utils import apply_repetition_penalty


def _reference_penalty(logits, generated_tokens, penalty, window=128):
    """The per-id loop the vectorized implementation replaced. Kept here as the
    oracle: the fast path has to agree with it on every shape, including the
    out-of-range ids and the union-across-batch scope.

    One deliberate divergence from the shipped old code: this oracle uses
    reshape(-1) where the original used view(-1), because the original CRASHED
    on batched non-contiguous history (a [3, 200] window slice) -- no caller
    reaches that shape, and the vectorized version handling it is a strict
    improvement, not a parity break."""
    out = logits.clone()
    vocab_size = out.shape[-1]
    tokens = generated_tokens[:, -window:] if generated_tokens.dim() >= 2 else generated_tokens[-window:]
    for token_id in set(tokens.reshape(-1).tolist()):
        if 0 <= token_id < vocab_size:
            if out.dim() == 1:
                score = out[token_id]
                out[token_id] = score / penalty if score > 0 else score * penalty
            else:
                scores = out[..., token_id]
                out[..., token_id] = torch.where(scores > 0, scores / penalty, scores * penalty)
    return out


def test_vectorized_penalty_matches_the_reference_loop():
    torch.manual_seed(7)
    for shape, tok_shape in (((17,), (5,)), ((1, 17), (1, 5)), ((3, 17), (3, 9))):
        for penalty in (1.1, 1.3, 2.0):
            logits = torch.randn(*shape) * 4.0
            tokens = torch.randint(0, 17, tok_shape)
            got = apply_repetition_penalty(logits, tokens, penalty)
            want = _reference_penalty(logits, tokens, penalty)
            assert torch.equal(got, want), f"{shape} penalty={penalty}"


def test_vectorized_penalty_ignores_out_of_range_ids():
    # Alignment pad rows and sentinel ids land outside the vocab; they must not
    # penalize a real row, and must not raise.
    logits = torch.randn(8)
    tokens = torch.tensor([-1, 3, 99, 8])
    got = apply_repetition_penalty(logits, tokens, 1.5)
    want = _reference_penalty(logits, tokens, 1.5)
    assert torch.equal(got, want)


def test_vectorized_penalty_honors_the_window_and_leaves_caller_intact():
    logits = torch.randn(12)
    before = logits.clone()
    tokens = torch.arange(12).repeat(30)  # far longer than the window
    got = apply_repetition_penalty(logits, tokens, 1.4, window=4)
    want = _reference_penalty(logits, tokens, 1.4, window=4)
    assert torch.equal(got, want)
    assert torch.equal(logits, before)  # never mutates the caller's tensor


def _tiny() -> Enigma:
    torch.manual_seed(0)
    return Enigma(
        ForgeConfig(
            vocab_size=64,
            dim=32,
            n_layers=2,
            n_heads=2,
            max_seq_len=32,
            dropout=0.0,
            use_gradient_checkpointing=False,
        )
    )


def _capture_penalty_windows(monkeypatch):
    seen: list[int] = []
    real = model_mod.sample_next_token

    def spy(logits, generated_tokens, *args, **kwargs):
        seen.append(generated_tokens.shape[1])
        return real(logits, generated_tokens, *args, **kwargs)

    monkeypatch.setattr(model_mod, "sample_next_token", spy)
    return seen


def test_stream_penalty_window_excludes_prompt(monkeypatch):
    seen = _capture_penalty_windows(monkeypatch)
    m = _tiny().eval()
    prompt = torch.randint(0, 64, (1, 7))
    with torch.no_grad():
        for _ in m.generate_stream(prompt, max_new_tokens=3, stop_tokens=[-1]):
            pass
    # Step one sees zero prior tokens (not the 7-token prompt); each later
    # step sees exactly what she generated so far. The exact sequence matters:
    # `all(n < 7)` alone accepted [0,0,0] -- a penalty silently applied to an
    # always-empty window, i.e. disabled -- (test-suite audit 2026-07-17).
    assert seen == [0, 1, 2]


def test_generate_penalty_window_excludes_prompt(monkeypatch):
    seen = _capture_penalty_windows(monkeypatch)
    m = _tiny().eval()
    prompt = torch.randint(0, 64, (1, 7))
    with torch.no_grad():
        m.generate(prompt, max_new_tokens=3, stop_tokens=[-1])
    assert seen == [0, 1, 2]


def test_penalty_noop_on_empty_window():
    logits = torch.randn(1, 64)
    empty = torch.empty(1, 0, dtype=torch.long)
    out = apply_repetition_penalty(logits, empty, penalty=1.1)
    assert torch.equal(out, logits)


def test_penalty_actually_reduces_repeated_token():
    """The positive case the file lacked: a token in the window must come out
    LESS likely, and untouched tokens must be byte-identical. Without this, a
    penalty that runs on the right window but modifies nothing stays green."""
    logits = torch.zeros(1, 64)
    logits[0, 5] = 2.0   # positive logit: penalty divides it down
    logits[0, 9] = -1.0  # negative logit: penalty multiplies it further down
    window = torch.tensor([[5, 9]], dtype=torch.long)
    out = apply_repetition_penalty(logits.clone(), window, penalty=1.5)
    assert out[0, 5] < logits[0, 5]
    assert out[0, 9] < logits[0, 9]
    untouched = [i for i in range(64) if i not in (5, 9)]
    assert torch.equal(out[0, untouched], logits[0, untouched])
