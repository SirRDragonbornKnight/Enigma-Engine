"""Model merging: config truth, key parity, SLERP geometry, TIES election."""

import pytest
import torch

from enigma_engine.core.model_merging import (
    _slerp_tensor,
    linear_merge,
    ties_merge,
)

CFG = dict(vocab_size=64, dim=32, n_layers=2, n_heads=2, max_seq_len=32, use_gradient_checkpointing=False)


def _save_ckpt(path, sd, cfg=None):
    ckpt = {"model_state_dict": sd}
    if cfg is not None:
        ckpt["config"] = cfg
    torch.save(ckpt, path)


def test_configless_checkpoint_refused(tmp_path):
    """A checkpoint without a config must not be stamped with the default
    architecture (dim=512/vocab=32000) in the merged output."""
    a = tmp_path / "a.pth"
    b = tmp_path / "b.pth"
    sd = {"w": torch.randn(4, 4)}
    _save_ckpt(a, sd)  # no config
    _save_ckpt(b, sd, CFG)
    with pytest.raises(ValueError, match="no model config"):
        linear_merge(a, b)


def test_key_mismatch_refused(tmp_path):
    """Tensors present in only one model fail loud instead of being
    silently dropped from the merged file."""
    a = tmp_path / "a.pth"
    b = tmp_path / "b.pth"
    _save_ckpt(a, {"w": torch.randn(4)}, CFG)
    _save_ckpt(b, {"w": torch.randn(4), "extra": torch.randn(2)}, CFG)
    with pytest.raises(ValueError, match="only some models"):
        linear_merge(a, b)


def test_slerp_antiparallel_stays_bounded():
    """Anti-parallel vectors have sin(omega) ~ 0; the fallback is linear
    interpolation, never amplified coefficients."""
    a = torch.randn(64)
    out = _slerp_tensor(a, -a, 0.25)
    assert torch.isfinite(out).all()
    assert out.abs().max() <= 2 * a.abs().max()
    # halfway between a and a is a itself (parallel end of the guard)
    same = _slerp_tensor(a, a.clone(), 0.5)
    assert torch.allclose(same, a)


def test_ties_election_is_magnitude_weighted(tmp_path):
    """Two +0.1 pushes vs one -10 push: summing trimmed VALUES elects the
    negative sign; a per-model sign count would have elected positive."""
    base = tmp_path / "base.pth"
    p1 = tmp_path / "m1.pth"
    p2 = tmp_path / "m2.pth"
    p3 = tmp_path / "m3.pth"
    _save_ckpt(base, {"w": torch.zeros(4)}, CFG)
    _save_ckpt(p1, {"w": torch.full((4,), 0.1)}, CFG)
    _save_ckpt(p2, {"w": torch.full((4,), 0.1)}, CFG)
    _save_ckpt(p3, {"w": torch.full((4,), -10.0)}, CFG)
    result = ties_merge([p1, p2, p3], base_path=base, density=1.0)
    assert (result["model_state_dict"]["w"] < 0).all()


def test_ties_weights_steer_the_election(tmp_path):
    """An outweighted large delta loses the election to a heavily
    weighted small one."""
    base = tmp_path / "base.pth"
    p1 = tmp_path / "m1.pth"
    p2 = tmp_path / "m2.pth"
    _save_ckpt(base, {"w": torch.zeros(4)}, CFG)
    _save_ckpt(p1, {"w": torch.full((4,), 1.0)}, CFG)
    _save_ckpt(p2, {"w": torch.full((4,), -3.0)}, CFG)
    result = ties_merge([p1, p2], base_path=base, density=1.0, weights=[10.0, 1.0])
    assert (result["model_state_dict"]["w"] > 0).all()
