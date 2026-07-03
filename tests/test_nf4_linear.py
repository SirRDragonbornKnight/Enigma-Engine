"""TC-7: Tests for NF4 (NormalFloat 4-bit) quantization — nf4_linear.py."""

import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from enigma_engine.core.nf4_linear import (
    NF4_BOUNDARIES,
    NF4_TABLE,
    NF4Linear,
    quantize_linear_nf4,
)


# ================================================================
# NF4 Table & Boundaries
# ================================================================


class TestNF4Table:
    """Verify the NF4 lookup table properties."""

    def test_table_has_16_entries(self):
        assert NF4_TABLE.shape == (16,)

    def test_table_sorted(self):
        for i in range(15):
            assert NF4_TABLE[i] < NF4_TABLE[i + 1]

    def test_table_range(self):
        assert NF4_TABLE[0].item() == -1.0
        assert NF4_TABLE[15].item() == 1.0

    def test_boundaries_are_midpoints(self):
        assert NF4_BOUNDARIES.shape == (15,)
        for i in range(15):
            expected = (NF4_TABLE[i] + NF4_TABLE[i + 1]) / 2
            assert abs(NF4_BOUNDARIES[i].item() - expected.item()) < 1e-6


# ================================================================
# NF4Linear — Construction
# ================================================================


class TestNF4LinearInit:
    """Test direct construction of NF4Linear."""

    def test_basic_construction(self):
        layer = NF4Linear(32, 16)
        assert layer.in_features == 32
        assert layer.out_features == 16
        assert layer.block_size == 64
        assert layer.bias is None

    def test_construction_with_bias(self):
        layer = NF4Linear(32, 16, bias=True)
        assert layer.bias is not None
        assert layer.bias.shape == (16,)

    def test_packed_buffer_size(self):
        layer = NF4Linear(16, 8)
        n_elements = 16 * 8  # 128
        expected_bytes = (n_elements + 1) // 2  # 64
        assert layer._packed.shape == (expected_bytes,)
        assert layer._packed.dtype == torch.uint8

    def test_scales_buffer_size(self):
        layer = NF4Linear(128, 64, block_size=64)
        n_elements = 128 * 64
        n_blocks = (n_elements + 64 - 1) // 64
        assert layer._scales.shape == (n_blocks,)

    def test_custom_block_size(self):
        layer = NF4Linear(32, 16, block_size=32)
        assert layer.block_size == 32


# ================================================================
# NF4Linear.from_linear — Quantization
# ================================================================


class TestFromLinear:
    """Test creating NF4Linear from an existing nn.Linear."""

    def test_from_linear_preserves_shape(self):
        linear = nn.Linear(64, 32)
        nf4 = NF4Linear.from_linear(linear)
        assert nf4.in_features == 64
        assert nf4.out_features == 32

    def test_from_linear_no_bias(self):
        linear = nn.Linear(32, 16, bias=False)
        nf4 = NF4Linear.from_linear(linear)
        assert nf4.bias is None

    def test_from_linear_with_bias(self):
        linear = nn.Linear(32, 16, bias=True)
        nf4 = NF4Linear.from_linear(linear)
        assert nf4.bias is not None
        # Bias should be copied exactly (not quantized)
        assert torch.allclose(nf4.bias, linear.bias)

    def test_from_linear_custom_block_size(self):
        linear = nn.Linear(64, 32)
        nf4 = NF4Linear.from_linear(linear, block_size=32)
        assert nf4.block_size == 32


# ================================================================
# Quantize / Dequantize Roundtrip
# ================================================================


class TestQuantizeDequantize:
    """Test quantization roundtrip error is within reasonable bounds."""

    def test_roundtrip_error_bounded(self):
        """NF4 roundtrip should have <20% relative error on random weights."""
        linear = nn.Linear(64, 32, bias=False)
        nn.init.normal_(linear.weight, std=0.02)
        nf4 = NF4Linear.from_linear(linear)
        recovered = nf4._dequantize()
        assert recovered.shape == linear.weight.shape
        # NF4 is 4-bit — expect moderate reconstruction error
        rel_err = (recovered - linear.weight).abs().mean() / linear.weight.abs().mean()
        assert rel_err < 0.20, f"Relative error {rel_err:.3f} too high"

    def test_zero_weight_survives(self):
        """Zero weights should round-trip close to zero."""
        linear = nn.Linear(64, 32, bias=False)
        nn.init.zeros_(linear.weight)
        nf4 = NF4Linear.from_linear(linear)
        recovered = nf4._dequantize()
        assert recovered.abs().max() < 1e-6

    def test_dequantize_output_shape(self):
        linear = nn.Linear(128, 64, bias=False)
        nf4 = NF4Linear.from_linear(linear)
        w = nf4._dequantize()
        assert w.shape == (64, 128)


# ================================================================
# Forward Pass
# ================================================================


class TestNF4Forward:
    """Test NF4Linear.forward() produces correct output shapes and values."""

    def test_forward_shape(self):
        linear = nn.Linear(64, 32)
        nf4 = NF4Linear.from_linear(linear)
        x = torch.randn(4, 64)
        out = nf4(x)
        assert out.shape == (4, 32)

    def test_forward_batched(self):
        linear = nn.Linear(32, 16)
        nf4 = NF4Linear.from_linear(linear)
        x = torch.randn(2, 8, 32)
        out = nf4(x)
        assert out.shape == (2, 8, 16)

    def test_forward_dtype_match(self):
        """Output dtype should match input dtype."""
        linear = nn.Linear(64, 32)
        nf4 = NF4Linear.from_linear(linear)
        x = torch.randn(4, 64, dtype=torch.float32)
        out = nf4(x)
        assert out.dtype == torch.float32

    def test_forward_close_to_original(self):
        """NF4 forward should approximate original linear within reason."""
        linear = nn.Linear(64, 32)
        nn.init.normal_(linear.weight, std=0.02)
        if linear.bias is not None:
            nn.init.zeros_(linear.bias)
        nf4 = NF4Linear.from_linear(linear)
        x = torch.randn(8, 64)
        original = linear(x)
        quantized = nf4(x)
        # Relative error should be bounded
        diff = (quantized - original).abs().mean()
        scale = original.abs().mean().clamp(min=1e-6)
        assert diff / scale < 0.25, f"Forward pass error too high: {diff / scale:.3f}"


# ================================================================
# Memory Savings
# ================================================================


class TestMemorySavings:
    """Test memory_savings_ratio()."""

    def test_ratio_less_than_one(self):
        """NF4 should use less memory than fp16."""
        linear = nn.Linear(256, 128)
        nf4 = NF4Linear.from_linear(linear)
        ratio = nf4.memory_savings_ratio()
        assert 0 < ratio < 1.0, f"Expected <1.0, got {ratio:.3f}"

    def test_ratio_approximately_quarter(self):
        """NF4 (4-bit) should be roughly 1/4 of fp16 (16-bit)."""
        linear = nn.Linear(1024, 512)
        nf4 = NF4Linear.from_linear(linear)
        ratio = nf4.memory_savings_ratio()
        # 4/16 = 0.25, plus scale overhead → roughly 0.25-0.35
        assert 0.2 < ratio < 0.4, f"Expected ~0.25, got {ratio:.3f}"


# ================================================================
# quantize_linear_nf4 — Model-wide Quantization
# ================================================================


class TestQuantizeModel:
    """Test quantize_linear_nf4() on a small model."""

    def _make_model(self):
        model = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
        )
        return model

    def test_replaces_linear_layers(self):
        model = self._make_model()
        count = quantize_linear_nf4(model)
        assert count == 2
        assert isinstance(model[0], NF4Linear)
        assert isinstance(model[2], NF4Linear)

    def test_non_linear_unchanged(self):
        model = self._make_model()
        quantize_linear_nf4(model)
        assert isinstance(model[1], nn.ReLU)

    def test_skip_names(self):
        model = self._make_model()
        count = quantize_linear_nf4(model, skip_names={"0"})
        assert count == 1
        assert isinstance(model[0], nn.Linear)  # skipped
        assert isinstance(model[2], NF4Linear)  # quantized

    def test_quantized_layers_frozen(self):
        model = self._make_model()
        quantize_linear_nf4(model)
        for p in model[0].parameters():
            assert not p.requires_grad

    def test_forward_after_quantize(self):
        model = self._make_model()
        x = torch.randn(4, 32)
        quantize_linear_nf4(model)
        out = model(x)
        assert out.shape == (4, 16)

    def test_empty_model_returns_zero(self):
        model = nn.Sequential(nn.ReLU(), nn.Dropout())
        count = quantize_linear_nf4(model)
        assert count == 0
