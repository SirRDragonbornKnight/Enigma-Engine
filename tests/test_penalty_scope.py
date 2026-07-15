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
    # step sees only what she generated so far.
    assert seen[0] == 0
    assert all(n < 7 for n in seen)


def test_generate_penalty_window_excludes_prompt(monkeypatch):
    seen = _capture_penalty_windows(monkeypatch)
    m = _tiny().eval()
    prompt = torch.randint(0, 64, (1, 7))
    with torch.no_grad():
        m.generate(prompt, max_new_tokens=3, stop_tokens=[-1])
    assert seen[0] == 0
    assert all(n < 7 for n in seen)


def test_penalty_noop_on_empty_window():
    logits = torch.randn(1, 64)
    empty = torch.empty(1, 0, dtype=torch.long)
    out = apply_repetition_penalty(logits, empty, penalty=1.1)
    assert torch.equal(out, logits)
