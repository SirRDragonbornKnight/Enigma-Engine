"""AudioEncoder padding mask -- the batched-align enabler.

The contract that lifts train_audio's batch_size=1 refusal: a padded
ragged batch with `lengths` must produce, at every VALID position, the
same features as running each sample alone -- and NOTHING from the pad
region (whose content is arbitrary) may leak into valid rows. Attention
is global, so without the mask every real frame would see pad garbage;
that silent corruption is exactly what the old refusal existed to stop.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from enigma_engine.core.audio_encoder import AudioEncoder, AudioEncoderConfig  # noqa: E402

LENS = [37, 60, 12]


def _encoder(**overrides):
    cfg = AudioEncoderConfig(
        n_mels=80, dim=64, n_layers=2, n_heads=4, dropout=0.0, max_audio_len=64, **overrides
    )
    torch.manual_seed(1234)
    return AudioEncoder(cfg).eval()


def _ragged_batch():
    torch.manual_seed(99)
    mels = [torch.randn(80, n) for n in LENS]
    t_max = max(LENS)
    batch = torch.stack([F.pad(m, (0, t_max - m.shape[-1])) for m in mels])
    return mels, batch, torch.tensor(LENS)


def test_padded_batch_matches_each_unbatched_forward():
    """THE contract: valid rows identical to per-sample runs, pad rows
    exact zeros."""
    enc = _encoder()
    mels, batch, lengths = _ragged_batch()
    with torch.no_grad():
        out = enc(batch, lengths=lengths)
        singles = [enc(m[None]) for m in mels]
    for i, n in enumerate(LENS):
        fl = (n + 1) // 2
        assert torch.allclose(out[i, :fl], singles[i][0], atol=1e-5), f"sample {i} diverged"
        pad_rows = out[i, fl:]
        if pad_rows.numel():  # the longest sample has no pad rows
            assert pad_rows.abs().max().item() == 0.0, f"sample {i} pad rows not zeroed"


def test_pad_garbage_cannot_leak_into_valid_rows():
    """The pad region's CONTENT must be irrelevant -- fill it with huge
    garbage and every valid row must be unchanged."""
    enc = _encoder()
    _, batch, lengths = _ragged_batch()
    garbage = batch.clone()
    for i, n in enumerate(LENS):
        garbage[i, :, n:] = 1e4
    with torch.no_grad():
        clean = enc(batch, lengths=lengths)
        dirty = enc(garbage, lengths=lengths)
    assert torch.allclose(clean, dirty, atol=1e-6), "pad content leaked into valid features"


def test_no_lengths_full_length_and_masked_agree():
    """lengths=[T] (nothing padded) must equal the legacy no-mask path."""
    enc = _encoder()
    torch.manual_seed(7)
    m = torch.randn(1, 80, 44)
    with torch.no_grad():
        legacy = enc(m)
        masked = enc(m, lengths=torch.tensor([44]))
    assert torch.allclose(legacy, masked, atol=1e-6)


def test_conformer_with_mask_refuses_loudly():
    """BatchNorm1d in the conformer conv would mix pad frames into batch
    statistics; until masked BN exists this must refuse, not corrupt."""
    enc = _encoder(use_conformer=True)
    _, batch, lengths = _ragged_batch()
    with pytest.raises(NotImplementedError, match="masked BN"):
        enc(batch, lengths=lengths)


def test_feature_lengths_matches_conv_arithmetic():
    """ceil(T/2) must be what conv2 (k=3, s=2, p=1) actually produces."""
    enc = _encoder()
    for t in (1, 2, 3, 12, 37, 44, 63):
        with torch.no_grad():
            out = enc(torch.randn(1, 80, t))
        assert out.shape[1] == AudioEncoder.feature_lengths(torch.tensor([t])).item() == (t + 1) // 2
