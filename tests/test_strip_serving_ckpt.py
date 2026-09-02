"""Serving-only checkpoint strip: the output carries exactly the four keys
boot reads, and an existing --out is refused rather than rebuilt in place
(model artifacts are versioned -- the house rule the trainers already follow)."""

import torch, pytest, subprocess, sys


def _fake_ckpt(path):
    torch.save({
        "model_state_dict": {"w": torch.zeros(2)},
        "config": {"dim": 2},
        "step": 7,
        "meta": {"chat_format": "x"},
        "optimizer_state_dict": {"ballast": torch.zeros(1000)},
        "scheduler": {"more": 1},
    }, path)


def test_strip_keeps_exactly_four_keys(tmp_path):
    src, out = tmp_path / "a.pth", tmp_path / "b.pth"
    _fake_ckpt(src)
    from strip_serving_ckpt import strip
    strip(src, out)
    ck = torch.load(out, weights_only=True)
    assert set(ck) == {"model_state_dict", "config", "step", "meta"}
    assert isinstance(ck, dict) and "model_state_dict" in ck and "config" in ck


def test_strip_refuses_existing_out(tmp_path):
    src, out = tmp_path / "a.pth", tmp_path / "b.pth"
    _fake_ckpt(src); out.write_bytes(b"x")
    from strip_serving_ckpt import strip
    with pytest.raises(SystemExit):
        strip(src, out)
