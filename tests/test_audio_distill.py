"""Distillation math for the owned audio encoder (distill_audio_encoder.py).

Locks the frame masking and the cosine loss contract offline -- no teacher
download, no audio files: shape-compatible random tensors only.
"""

import pytest
import torch
from torch import nn

from distill_audio_encoder import masked_distill_loss


def test_padding_frames_are_excluded_from_the_loss():
    torch.manual_seed(0)
    student = torch.randn(2, 100, 512)
    proj = nn.Linear(512, 512)
    with torch.no_grad():
        teacher = proj(student)
        # wreck the teacher BEYOND the real frames -- masked, must not matter
        teacher[:, 60:] = torch.randn_like(teacher[:, 60:]) * 100
    real = torch.tensor([60, 60])
    loss = masked_distill_loss(student, teacher, real, proj)
    assert loss.item() == pytest.approx(0.0, abs=1e-5)


def test_unmasked_mismatch_is_penalized():
    torch.manual_seed(0)
    student = torch.randn(2, 100, 512)
    teacher = torch.randn(2, 100, 512)
    real = torch.tensor([100, 100])
    loss = masked_distill_loss(student, teacher, real, nn.Linear(512, 512))
    assert 0.5 < loss.item() < 2.5  # random cosine ~ 0 -> ~1.0 + ~0.5


def test_length_mismatch_truncates_to_shorter_side():
    student = torch.randn(1, 1500, 512)
    teacher = torch.randn(1, 1498, 512)
    real = torch.tensor([1499])  # clamped to the truncated length
    loss = masked_distill_loss(student, teacher, real, nn.Linear(512, 512))
    assert torch.isfinite(loss)


def test_backward_flows_to_student():
    student = torch.randn(1, 50, 512, requires_grad=True)
    teacher = torch.randn(1, 50, 512)
    loss = masked_distill_loss(student, teacher, torch.tensor([30]), nn.Linear(512, 512))
    loss.backward()
    assert student.grad is not None
    # gradient touches only real frames (padding contributes nothing)
    assert student.grad[:, 30:].abs().sum().item() == 0.0
    assert student.grad[:, :30].abs().sum().item() > 0.0
