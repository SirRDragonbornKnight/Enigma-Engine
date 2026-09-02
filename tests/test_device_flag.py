"""--device resolution: `auto` keeps today's pick byte-identically, `cpu` is
honored even on a CUDA box, and `cuda` on a machine without CUDA FAILS rather
than falling back silently -- a CPU baseline that quietly ran on the 5090 is
the measurement this flag exists to make impossible."""

from __future__ import annotations

import pytest

import serve_enigma as serve


def _cuda(monkeypatch, available: bool):
    monkeypatch.setattr(serve.torch.cuda, "is_available", lambda: available)


def test_auto_picks_cuda_when_available(monkeypatch):
    _cuda(monkeypatch, True)
    assert serve._resolve_device("auto") == "cuda"


def test_auto_falls_back_to_cpu_without_cuda(monkeypatch):
    _cuda(monkeypatch, False)
    assert serve._resolve_device("auto") == "cpu"


def test_cpu_is_honored_even_on_a_cuda_box(monkeypatch):
    _cuda(monkeypatch, True)
    assert serve._resolve_device("cpu") == "cpu"


def test_cuda_resolves_when_available(monkeypatch):
    _cuda(monkeypatch, True)
    assert serve._resolve_device("cuda") == "cuda"


def test_cuda_refuses_when_unavailable_instead_of_silent_cpu(monkeypatch):
    _cuda(monkeypatch, False)
    with pytest.raises(SystemExit) as exc:
        serve._resolve_device("cuda")
    assert "cuda" in str(exc.value).lower()


def test_auto_matches_the_expression_it_replaced(monkeypatch):
    """The byte-identical claim, executed both ways rather than asserted."""
    for available in (True, False):
        _cuda(monkeypatch, available)
        legacy = "cuda" if serve.torch.cuda.is_available() else "cpu"
        assert serve._resolve_device("auto") == legacy


def test_parser_defaults_to_auto_and_offers_the_three_choices():
    args = serve._p.parse_known_args([])[0]
    assert args.device == "auto"
    assert serve._p.parse_known_args(["--device", "cpu"])[0].device == "cpu"
    with pytest.raises(SystemExit):
        serve._p.parse_known_args(["--device", "tpu"])


def test_bench_shares_the_same_resolver():
    """bench_generate must not grow a second, drifting copy of the rule."""
    import bench_generate

    assert bench_generate._resolve_device is serve._resolve_device
