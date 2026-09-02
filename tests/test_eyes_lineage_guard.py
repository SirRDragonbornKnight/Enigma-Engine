"""The --eyes graft must refuse a FOREIGN-LINEAGE align checkpoint.

The graft loads strict-clean whenever the encoder dim matches, so an align
checkpoint trained against a different text lineage grafts a projection that
targets an embedding space the served model does not have -- no warning, eyes
reported live, and every image captioning to one constant string (measured
2026-08-29 against the 2026-07-20 v1-aligned checkpoint under the v2 238M).
The align trainer stamps `model_config` into every checkpoint, so the lineage
is checkable at load time. Tiny fixtures, CPU only, no downloads.
"""

from __future__ import annotations

import pytest
import torch

import serve_enigma as serve
from enigma_engine.core.eyes import EyesError
from enigma_engine.core.model_presets import ForgeConfig
from enigma_engine.core.vision_encoder import VISION_PRESETS, VisionEncoder

# The real incident's fingerprint: the shipped align checkpoint's stored
# model_config (v1/182M lineage) against the served v2.
V1_LINEAGE = {"vocab_size": 4718, "dim": 1024, "n_layers": 16, "n_heads": 16}


def _served_config() -> ForgeConfig:
    """The tiny served shape (mirrors test_native_eyes.py's model)."""
    return ForgeConfig(
        vocab_size=64, dim=32, n_layers=2, n_heads=2,
        max_seq_len=64, dropout=0.0, use_gradient_checkpointing=False,
    )


def _align_ckpt(tmp_path, name, model_config=None):
    """An otherwise-valid align checkpoint, optionally carrying model_config."""
    enc = VisionEncoder(VISION_PRESETS["small"])
    ck = {
        "vision_encoder_state_dict": enc.state_dict(),
        "model_state_dict": {
            "vision_projection.0.weight": torch.zeros(4, 4),
            "vision_projection.0.bias": torch.zeros(4),
        },
    }
    if model_config is not None:
        ck["model_config"] = model_config
    path = tmp_path / name
    torch.save(ck, path)
    return path


def test_foreign_lineage_refused(tmp_path):
    ckpt = _align_ckpt(tmp_path, "foreign.pt", V1_LINEAGE)
    with pytest.raises(EyesError, match="foreign lineage") as exc:
        serve._load_eyes(ckpt, "small", _served_config())
    msg = str(exc.value)
    assert "vocab_size" in msg
    assert "4718" in msg and "64" in msg  # align's value and the served one


def test_matching_lineage_loads(tmp_path):
    cfg = _served_config()
    same = {k: getattr(cfg, k) for k in serve._EYES_LINEAGE_KEYS}
    ckpt = _align_ckpt(tmp_path, "same.pt", same)
    venc, proj_sd, dim = serve._load_eyes(ckpt, "small", cfg)
    assert dim == VISION_PRESETS["small"].dim
    assert set(proj_sd) == {"0.weight", "0.bias"}
    assert venc is not None


def test_missing_model_config_refused(tmp_path):
    """No stored fingerprint = the lineage cannot be verified; refuse rather
    than graft a projection that may target another model's embeddings."""
    ckpt = _align_ckpt(tmp_path, "no_mc.pt")
    with pytest.raises(EyesError, match="model_config"):
        serve._load_eyes(ckpt, "small", _served_config())


@pytest.mark.parametrize(
    "stored",
    [
        ["not", "a", "dict"],  # no key matches: compares as clean, grafts blind
        ["vocab_size"],  # a key DOES match: indexing a list raises TypeError
    ],
)
def test_non_dict_model_config_refused(tmp_path, stored):
    """A truthy non-dict fingerprint is UNUSABLE, not a lineage. Both shapes are
    wrong in their own way -- one compares as no-keys-present and grafts blind,
    the other raises TypeError, which boot's degrade catch does not cover, so
    text serving would die with the eye instead of dropping it."""
    ckpt = _align_ckpt(tmp_path, "bad_mc.pt", stored)
    with pytest.raises(EyesError, match="model_config"):
        serve._load_eyes(ckpt, "small", _served_config())


def test_two_arg_call_skips_the_check(tmp_path):
    """The served config is OPTIONAL: the 2-arg call (every standing
    test_serve_enigma fixture) must load exactly as it did before."""
    ckpt = _align_ckpt(tmp_path, "foreign_2arg.pt", V1_LINEAGE)
    _venc, proj_sd, dim = serve._load_eyes(ckpt, "small")
    assert dim == VISION_PRESETS["small"].dim
    assert "0.weight" in proj_sd


def test_partial_stored_config_compares_present_keys_only(tmp_path):
    """An older writer's partial fingerprint is compared on what it HAS --
    absent keys are not evidence of a mismatch, present ones still refuse."""
    cfg = _served_config()
    ok = _align_ckpt(tmp_path, "partial_ok.pt", {"vocab_size": 64, "dim": 32})
    _venc, _proj, dim = serve._load_eyes(ok, "small", cfg)
    assert dim == VISION_PRESETS["small"].dim

    bad = _align_ckpt(tmp_path, "partial_bad.pt", {"vocab_size": 4718})
    with pytest.raises(EyesError, match="foreign lineage"):
        serve._load_eyes(bad, "small", cfg)


def test_a_model_config_with_no_lineage_keys_is_refused(tmp_path):
    """The zero-key edge: "compare only the keys it HAS" degenerates to
    comparing NOTHING, so a dict of unrelated fields grafted blind -- exactly
    the silent wrong-lineage graft this guard exists to stop."""
    cfg = _served_config()
    blind = _align_ckpt(tmp_path, "no_keys.pt", {"trained_by": "someone", "epochs": 3})
    with pytest.raises(EyesError, match="no comparable lineage keys"):
        serve._load_eyes(blind, "small", cfg)
