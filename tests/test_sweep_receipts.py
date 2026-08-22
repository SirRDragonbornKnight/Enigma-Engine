"""The sweep's receipt file, which a second sweep used to erase.

`--out-root` has a fixed default, so `python sweep_lr.py ...` twice wrote
sweep_results.json over the same path both times: the first grid's numbers --
the whole reason the grid was run -- were gone with nothing left to diff. The
rotate-then-write pattern is make_sft_data's `_write_artifact`, applied once
per invocation rather than once per point, because the per-point rewrite is
this sweep's own file and rotating it every point would push its OWN first
point into .prev.
"""

from __future__ import annotations

import json
import sys

import sweep_lr


def _stub_points(monkeypatch, marker: str):
    """Replace the subprocess-launching point runner with a labelled result."""
    def fake_run_point(args, lr, seed, out_dir):
        return {"lr": lr, "seed": seed, "out": str(out_dir), "marker": marker,
                "seconds": 1.0, "returncode": 0, "val_loss": 2.0, "ppl": 7.4,
                "bits_per_token": 2.9, "tok_per_s": 1000, "error": None,
                "ranked_on": "val", "rank_loss": 2.0}

    monkeypatch.setattr(sweep_lr, "run_point", fake_run_point)


def _sweep(monkeypatch, out_root, marker: str):
    _stub_points(monkeypatch, marker)
    monkeypatch.setattr(sys, "argv", [
        "sweep_lr.py", "--size", "v2_deep_238m", "--lrs", "1e-3",
        "--tokens", "1000000", "--out-root", str(out_root),
    ])
    sweep_lr.main()


def test_a_second_sweep_rotates_the_first_ones_receipts(monkeypatch, tmp_path, capsys):
    out_root = tmp_path / "sweeps"
    _sweep(monkeypatch, out_root, "first")
    results = out_root / "sweep_results.json"
    previous = out_root / "sweep_results.prev.json"
    assert json.loads(results.read_text(encoding="utf-8"))[0]["marker"] == "first"
    assert not previous.exists(), "nothing to rotate on the first sweep"

    _sweep(monkeypatch, out_root, "second")
    assert json.loads(results.read_text(encoding="utf-8"))[0]["marker"] == "second"
    assert json.loads(previous.read_text(encoding="utf-8"))[0]["marker"] == "first", (
        "the second sweep erased the first sweep's receipts"
    )
    assert "rotated" in capsys.readouterr().out, "a rotation is said out loud"


def test_the_rotation_is_once_per_sweep_not_once_per_point(monkeypatch, tmp_path):
    """Rotating on every write would leave .prev holding this sweep's own
    second-to-last point -- the previous sweep gone all the same."""
    out_root = tmp_path / "sweeps"
    _sweep(monkeypatch, out_root, "first")
    _stub_points(monkeypatch, "second")
    monkeypatch.setattr(sys, "argv", [
        "sweep_lr.py", "--size", "v2_deep_238m", "--lrs", "1e-3,2e-3,3e-3",
        "--tokens", "1000000", "--out-root", str(out_root),
    ])
    sweep_lr.main()

    current = json.loads((out_root / "sweep_results.json").read_text(encoding="utf-8"))
    previous = json.loads((out_root / "sweep_results.prev.json").read_text(encoding="utf-8"))
    assert [p["lr"] for p in current] == [1e-3, 2e-3, 3e-3]
    assert [p["marker"] for p in previous] == ["first"]
