"""Distillation math for the owned vision encoder (distill_vision_encoder.py).

Locks the grid resampling and the cosine loss contract offline -- no teacher
download, no images: shape-compatible random tensors only.
"""

import pytest
import torch
from torch import nn

from distill_vision_encoder import distill_loss, teacher_to_student_grid


def test_grid_resample_shapes():
    t = torch.randn(2, 256, 384)  # 16x16 teacher grid
    out = teacher_to_student_grid(t, 196)  # 14x14 student grid
    assert out.shape == (2, 196, 384)


def test_grid_resample_identity_when_grids_match():
    t = torch.randn(2, 196, 384)
    out = teacher_to_student_grid(t, 196)
    assert torch.equal(out, t)


def test_grid_resample_refuses_non_square():
    with pytest.raises(ValueError, match="non-square"):
        teacher_to_student_grid(torch.randn(1, 250, 384), 196)


def test_distill_loss_zero_when_aligned():
    torch.manual_seed(0)
    student = torch.randn(2, 196, 512)
    proj = nn.Linear(512, 384)
    with torch.no_grad():
        teacher = proj(student)  # same grid, exact projection target
    loss = distill_loss(student, teacher, proj)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_distill_loss_random_is_near_expected():
    torch.manual_seed(0)
    student = torch.randn(4, 196, 512)
    teacher = torch.randn(4, 196, 384)
    loss = distill_loss(student, teacher, nn.Linear(512, 384))
    # random cosine ~ 0 -> patch term ~ 1.0, global term ~ 0.5
    assert 0.5 < loss.item() < 2.5


def test_distill_loss_backward_flows_to_student():
    student = torch.randn(1, 196, 512, requires_grad=True)
    teacher = torch.randn(1, 256, 384)
    loss = distill_loss(student, teacher, nn.Linear(512, 384))
    loss.backward()
    assert student.grad is not None and student.grad.abs().sum() > 0
