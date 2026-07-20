"""Arc-C v2 trainer pieces: the wsd_sqrt schedule and post-hoc checkpoint EMA.

wsd_sqrt = the IMU-1 ablation's shape (hold at peak, 1-sqrt(t) decay to a
0.01x floor). ema_checkpoints.py = offline EMA over one lineage's archived
checkpoints (beta 0.8 over the final ~10 = the recipe's free win); it must
refuse cross-lineage input rather than average incompatible weights.
"""

from __future__ import annotations

import math

import pytest
import torch

import ema_checkpoints
from ema_checkpoints import ema_state_dicts
from enigma_engine.core.optim import get_lr

PEAK = 6e-4
TOTAL = 1000
WARMUP = 100


def _lr(step: int) -> float:
    return get_lr(step, WARMUP, TOTAL, PEAK, schedule="wsd_sqrt", decay_frac=0.1)


# ---------------------------------------------------------------------------
# wsd_sqrt schedule
# ---------------------------------------------------------------------------


def test_wsd_sqrt_warmup_and_hold():
    assert _lr(0) == PEAK * 1 / WARMUP
    assert _lr(WARMUP) == PEAK
    assert _lr(899) == PEAK  # last hold step before decay_start=900


def test_wsd_sqrt_decays_to_floor_not_zero():
    assert _lr(TOTAL) == pytest.approx(0.01 * PEAK)
    assert _lr(TOTAL + 50) == pytest.approx(0.01 * PEAK)  # continuation stays at floor


def test_wsd_sqrt_shape_is_one_minus_sqrt():
    # Midpoint of the decay window (step 950 of 900..1000): t = 0.5.
    expected = PEAK * (0.01 + 0.99 * (1 - math.sqrt(0.5)))
    assert _lr(950) == pytest.approx(expected)


def test_wsd_sqrt_monotone_nonincreasing_after_warmup():
    lrs = [_lr(s) for s in range(WARMUP, TOTAL + 1)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))


def test_existing_schedules_untouched():
    # cosine and wsd are checkpoint-recorded contracts; adding wsd_sqrt must
    # not perturb them.
    assert get_lr(500, WARMUP, TOTAL, PEAK) == pytest.approx(
        PEAK * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (500 - WARMUP) / (TOTAL - WARMUP))))
    )
    assert get_lr(TOTAL - 1, WARMUP, TOTAL, PEAK, schedule="wsd", decay_frac=0.1) == pytest.approx(
        PEAK * 1 / 100
    )


# ---------------------------------------------------------------------------
# EMA math
# ---------------------------------------------------------------------------


def _sd(value: float, dtype=torch.float32) -> dict:
    return {"w": torch.full((4, 4), value, dtype=dtype), "count": torch.tensor([1])}


def test_ema_math_beta_08():
    out = ema_state_dicts([_sd(0.0), _sd(1.0)], beta=0.8)
    assert torch.allclose(out["w"], torch.full((4, 4), 0.2))
    # three-way: ((0*0.8 + 1*0.2)*0.8 + 2*0.2) = 0.56
    out3 = ema_state_dicts([_sd(0.0), _sd(1.0), _sd(2.0)], beta=0.8)
    assert torch.allclose(out3["w"], torch.full((4, 4), 0.56))


def test_ema_int_tensors_take_last_and_dtype_preserved():
    a, b = _sd(0.0, torch.bfloat16), _sd(1.0, torch.bfloat16)
    a["count"] = torch.tensor([7])
    b["count"] = torch.tensor([9])
    out = ema_state_dicts([a, b], beta=0.8)
    assert out["w"].dtype == torch.bfloat16
    assert out["count"].item() == 9


def test_ema_refuses_bad_input():
    with pytest.raises(ValueError, match=">= 2"):
        ema_state_dicts([_sd(0.0)], beta=0.8)
    bad_keys = {"other": torch.zeros(4, 4)}
    with pytest.raises(ValueError, match="key set"):
        ema_state_dicts([_sd(0.0), bad_keys], beta=0.8)
    bad_shape = {"w": torch.zeros(2, 2), "count": torch.tensor([1])}
    with pytest.raises(ValueError, match="shape"):
        ema_state_dicts([_sd(0.0), bad_shape], beta=0.8)


# ---------------------------------------------------------------------------
# CLI: lineage guards + output format
# ---------------------------------------------------------------------------


VALID_CFG = {"vocab_size": 64, "dim": 32, "n_layers": 1, "n_heads": 2, "max_seq_len": 32}


def _ckpt(tmp_path, name: str, value: float, step: int, config=None):
    p = tmp_path / name
    torch.save(
        {
            "model_state_dict": {"w": torch.full((4, 4), value)},
            "config": config or dict(VALID_CFG),
            "step": step,
        },
        p,
    )
    return p


def test_cli_writes_loadable_ema(tmp_path):
    a = _ckpt(tmp_path, "a.pth", 0.0, 100)
    b = _ckpt(tmp_path, "b.pth", 1.0, 200)
    out = tmp_path / "ema.pth"
    assert ema_checkpoints.main([str(a), str(b), "--out", str(out), "--beta", "0.8"]) == 0
    ck = torch.load(out, weights_only=False)
    assert torch.allclose(ck["model_state_dict"]["w"], torch.full((4, 4), 0.2))
    assert ck["config"] == dict(VALID_CFG)
    assert ck["step"] == 200
    assert ck["ema"]["beta"] == 0.8 and ck["ema"]["n"] == 2


def test_cli_refuses_wrong_order(tmp_path):
    a = _ckpt(tmp_path, "a.pth", 0.0, 200)
    b = _ckpt(tmp_path, "b.pth", 1.0, 100)
    with pytest.raises(SystemExit, match="oldest first"):
        ema_checkpoints.main([str(a), str(b), "--out", str(tmp_path / "e.pth")])


def test_cli_refuses_cross_lineage_configs(tmp_path):
    a = _ckpt(tmp_path, "a.pth", 0.0, 100)
    b = _ckpt(tmp_path, "b.pth", 1.0, 200, config={**VALID_CFG, "dim": 64, "n_heads": 4})
    with pytest.raises(SystemExit, match="lineage"):
        ema_checkpoints.main([str(a), str(b), "--out", str(tmp_path / "e.pth")])


def test_cli_accepts_config_key_drift_within_a_lineage(tmp_path):
    # audit 2026-07-20: raw dict equality refused archive windows straddling a
    # code upgrade -- one save carries a retired key, the next carries a field
    # added later. Same architecture must average fine.
    old_era = {**VALID_CFG, "early_exit_layer": 0}  # retired key, filtered on load
    new_era = {**VALID_CFG, "norm_scheme": "pre"}  # explicit default, same arch
    a = _ckpt(tmp_path, "a.pth", 0.0, 100, config=old_era)
    b = _ckpt(tmp_path, "b.pth", 1.0, 200, config=new_era)
    out = tmp_path / "ema.pth"
    assert ema_checkpoints.main([str(a), str(b), "--out", str(out)]) == 0
    assert out.exists()


def test_cli_warns_when_no_steps_recorded(tmp_path, capsys):
    # audit 2026-07-20: the order guard was silently vacuous for stepless
    # checkpoints -- a wrong-order EMA ran to completion with no signal.
    a = _ckpt(tmp_path, "a.pth", 0.0, None)
    b = _ckpt(tmp_path, "b.pth", 1.0, None)
    assert ema_checkpoints.main([str(a), str(b), "--out", str(tmp_path / "e.pth")]) == 0
    assert "CANNOT be" in capsys.readouterr().out


def test_ema_refuses_non_tensor_entries():
    bad = {"w": torch.zeros(4, 4), "count": "seven"}
    with pytest.raises(ValueError, match="not a tensor"):
        ema_state_dicts([_sd(0.0), bad], beta=0.8)


def test_cli_refuses_overwriting_a_source(tmp_path):
    a = _ckpt(tmp_path, "a.pth", 0.0, 100)
    b = _ckpt(tmp_path, "b.pth", 1.0, 200)
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        ema_checkpoints.main([str(a), str(b), "--out", str(b)])
