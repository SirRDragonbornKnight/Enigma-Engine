"""Direct tests for safe_save's rotation contract -- none existed (Round-2
review 2026-08-13), and the one crash window the docstring promises against
had an exception path that broke the promise: once the old file was rotated
away, a failing promotion rename hit the BaseException cleanup and DELETED
the complete .tmp -- destroying the new checkpoint and leaving nothing at
the target at all.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from enigma_engine.core.safe_save import atomic_torch_save


def test_rotate_promotes_and_keeps_one_generation(tmp_path):
    target = tmp_path / "latest.pth"
    prev = tmp_path / "prev.pth"
    atomic_torch_save({"v": 1}, target, rotate_to=prev)
    atomic_torch_save({"v": 2}, target, rotate_to=prev)
    assert torch.load(target, weights_only=True)["v"] == 2
    assert torch.load(prev, weights_only=True)["v"] == 1
    assert not (tmp_path / "latest.pth.tmp").exists()


def test_a_failed_promotion_after_rotation_keeps_the_new_data(tmp_path, monkeypatch):
    """If the promotion rename fails AFTER the old file rotated away, the
    .tmp is the ONLY complete copy of the new checkpoint -- the cleanup must
    not unlink it (review 2026-08-13: it did, leaving prev.pth as the sole
    survivor and nothing at the target)."""
    target = tmp_path / "latest.pth"
    prev = tmp_path / "prev.pth"
    atomic_torch_save({"v": 1}, target, rotate_to=prev)

    real_replace = os.replace
    calls = {"n": 0}

    def flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] == 2:  # 1st call = rotation, 2nd = promotion
            raise OSError("sharing violation on promote")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky_replace)
    with pytest.raises(OSError):
        atomic_torch_save({"v": 2}, target, rotate_to=prev)
    monkeypatch.undo()

    tmp = tmp_path / "latest.pth.tmp"
    assert tmp.exists(), "the only complete copy of the new checkpoint was deleted"
    assert torch.load(tmp, weights_only=True)["v"] == 2
    assert torch.load(prev, weights_only=True)["v"] == 1


def test_a_failed_rotation_leaves_the_original_in_place(tmp_path, monkeypatch):
    """The rotation rename ITSELF failing (prev.pth locked by a reader) must
    leave the pre-call state exactly: the original still at the target, no
    partial rotation, the .tmp removed (audit 2026-08-14: this path was
    correct but unpinned)."""
    target = tmp_path / "latest.pth"
    prev = tmp_path / "prev.pth"
    atomic_torch_save({"v": 1}, target, rotate_to=prev)

    real_replace = os.replace

    def locked_rotation(src, dst):
        if Path(dst) == prev:
            raise OSError("sharing violation on rotate")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", locked_rotation)
    with pytest.raises(OSError):
        atomic_torch_save({"v": 2}, target, rotate_to=prev)
    monkeypatch.undo()

    assert torch.load(target, weights_only=True)["v"] == 1
    assert not prev.exists()
    assert not (tmp_path / "latest.pth.tmp").exists()


def test_a_failed_write_before_rotation_still_cleans_up(tmp_path, monkeypatch):
    """The pre-rotation failure path keeps its old behavior: the original is
    untouched and the partial .tmp is removed."""
    target = tmp_path / "latest.pth"
    atomic_torch_save({"v": 1}, target)

    def boom(*a, **k):
        raise RuntimeError("torch.save died mid-write")

    monkeypatch.setattr(torch, "save", boom)
    with pytest.raises(RuntimeError):
        atomic_torch_save({"v": 2}, target, rotate_to=tmp_path / "prev.pth")
    monkeypatch.undo()

    assert torch.load(target, weights_only=True)["v"] == 1
    assert not (tmp_path / "latest.pth.tmp").exists()
