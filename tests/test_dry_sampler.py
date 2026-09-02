"""DRY (Don't-Repeat-Yourself) sampling semantics: one penalty per candidate,
taken from the LONGEST match, never accumulated across earlier occurrences.
Covers both logits shapes -- the live call site hands a 2-D [1, vocab] slice."""

import pytest
import torch
from enigma_engine.core.model_utils import _dry_penalty


def test_dry_zero_multiplier_is_identity():
    logits = torch.randn(64)
    out = _dry_penalty(logits, [1, 2, 3, 1, 2], 0.0, 1.75, 2)
    assert torch.equal(out, logits)


def test_dry_penalizes_token_extending_a_repeat():
    # history ...5 6 7 5 6 -> token 7 extends the repeat; match len 2,
    # allowed 1 -> penalty 1.0 * 2.0**(2-1) = 2.0 subtracted from logit 7
    logits = torch.zeros(8)
    out = _dry_penalty(logits, [5, 6, 7, 5, 6], 1.0, 2.0, 1)
    assert out[7] == -2.0
    assert torch.equal(out[:7], torch.zeros(7))


def test_dry_penalty_at_exact_allowed_length():
    # canonical semantics: match_len == allowed_length DOES penalize (base**0)
    logits = torch.zeros(8)
    out = _dry_penalty(logits, [5, 6, 7, 5, 6], 1.0, 2.0, 2)
    assert out[7] == -1.0


def test_dry_no_penalty_below_allowed_length():
    logits = torch.zeros(8)
    out = _dry_penalty(logits, [5, 6, 7, 5, 6], 1.0, 2.0, 3)
    assert torch.equal(out, torch.zeros(8))


def test_dry_longest_match_only_never_accumulates():
    # 7 appears after "5 6" twice. The LONGEST match here is the OVERLAPPING
    # length-5 suffix (ids[0:5] == ids[3:8]) -- round-2 audit F1 executed this
    # and the naive expected value (-2.0, from the len-2 match) was WRONG.
    # One penalty, from the longest match, never a sum across occurrences:
    logits = torch.zeros(9)
    out = _dry_penalty(logits, [5, 6, 7, 5, 6, 7, 5, 6], 1.0, 2.0, 1)
    assert out[7] == -(2.0 ** (5 - 1))       # exactly one penalty, m=5


def test_dry_returns_new_tensor():
    logits = torch.zeros(8)
    out = _dry_penalty(logits, [5, 6, 7, 5, 6], 1.0, 2.0, 1)
    assert out is not logits and logits[7] == 0.0


def test_dry_handles_2d_logits_as_at_the_live_call_site():
    # model.py:857 hands sample_next_token a [1, vocab] slice -- the shape the
    # integration actually sees (round-2 audit F2)
    logits = torch.zeros(1, 8)
    out = _dry_penalty(logits, [5, 6, 7, 5, 6], 1.0, 2.0, 1)
    assert out[0, 7] == -2.0 and torch.equal(out[0, :7], torch.zeros(7))


def test_dry_huge_match_does_not_overflow():
    # round-2 audit F4: uncapped base**m raised OverflowError at n=2048
    # all-identical history; the exponent cap must keep this finite-or--inf,
    # never an exception
    logits = torch.zeros(4)
    out = _dry_penalty(logits, [3] * 2048, 1.0, 1.75, 2)
    assert torch.isfinite(out[3]) or out[3] == float("-inf")


# --- the serve-side default (--dry-multiplier): the resolver, unit-tested ---


def _req(**kw):
    import serve_enigma as serve

    return serve.ChatReq(messages=[{"role": "user", "content": "hi"}], **kw)


def _with_flag(monkeypatch, value):
    """Set the serve-side --dry-multiplier without inventing a module global."""
    import argparse

    import serve_enigma as serve

    monkeypatch.setattr(serve, "ARGS", argparse.Namespace(dry_multiplier=value))


def test_serve_flag_defaults_to_off_so_today_is_byte_identical(monkeypatch):
    import serve_enigma as serve

    _with_flag(monkeypatch, 0.0)
    assert serve._dry_multiplier_for(_req()) == 0.0
    assert serve._p.parse_known_args([])[0].dry_multiplier == 0.0


def test_serve_flag_fills_in_for_a_silent_request(monkeypatch):
    import serve_enigma as serve

    _with_flag(monkeypatch, 0.8)
    assert serve._dry_multiplier_for(_req()) == 0.8


def test_an_explicit_request_value_beats_the_serve_flag(monkeypatch):
    import serve_enigma as serve

    _with_flag(monkeypatch, 0.8)
    assert serve._dry_multiplier_for(_req(dry_multiplier=0.3)) == 0.3


def test_an_explicit_zero_turns_dry_off_against_a_defaulting_server(monkeypatch):
    """The case a value comparison cannot see: 0.0 asked for is not 0.0 unasked."""
    import serve_enigma as serve

    _with_flag(monkeypatch, 0.8)
    assert serve._dry_multiplier_for(_req(dry_multiplier=0.0)) == 0.0


def test_the_flag_refuses_an_out_of_range_value():
    import serve_enigma as serve

    for bad in ("-1", "11", "nan"):
        with pytest.raises(SystemExit):
            serve._p.parse_known_args(["--dry-multiplier", bad])
