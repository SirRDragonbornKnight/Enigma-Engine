"""Direct tests for model_registry's load path -- the two functions
``Enigma.from_pretrained`` calls back to back (model.py:913-914) and that
``encoder_align`` reuses with a prefix (encoder_align.py:709). Neither had a
test of its own before (2026-08-23 review: the negative-space list).

Scope is the load path on purpose: ``ModelRegistry`` and ``get_model_hash``
are exported but have no caller anywhere in the tree, so pinning their shapes
here would lock a contract nobody consumes. The ``.safetensors`` arm of
``safe_load_weights`` is likewise uncovered -- every checkpoint the engine
loads is a ``.pth``.
"""

from __future__ import annotations

import logging

import pytest
import torch

from enigma_engine.core import model_registry as mr


class _NotATensor:
    """A pickled object that weights_only loading must refuse to construct."""


def _sd() -> dict[str, torch.Tensor]:
    return {"tok_embeddings.weight": torch.ones(2, 3), "output.bias": torch.zeros(3)}


@pytest.mark.parametrize(
    "wrap",
    [
        lambda sd: sd,
        lambda sd: {"model_state_dict": sd},
        lambda sd: {"state_dict": sd, "epoch": 7},
        lambda sd: {"model": sd},
        lambda sd: {"outer": sd},
    ],
    ids=["bare", "model_state_dict", "state_dict_with_siblings", "model", "single_key_nested"],
)
def test_the_load_path_unwraps_every_checkpoint_shape(tmp_path, wrap):
    """load-then-unwrap is one breath at model.py:913-914, so every wrapping
    the docstring promises has to come back as the same tensors under the same
    names -- including the single-key nesting, which recurses."""
    sd = _sd()
    path = tmp_path / "model.pth"
    torch.save(wrap(sd), path)

    out = mr.get_state_dict(mr.safe_load_weights(path, map_location="cpu"))

    assert set(out) == set(sd)
    for key, value in sd.items():
        assert torch.equal(out[key], value)


def test_prefix_strips_the_compile_wrapper_and_leaves_the_rest():
    """encoder_align.py:709 unwraps with ``prefix="_orig_mod."``: a
    torch.compile checkpoint carries that prefix on the compiled keys only, so
    the strip is per key, never all-or-nothing."""
    out = mr.get_state_dict(
        {"_orig_mod.w": torch.ones(1), "bias": torch.zeros(1)},
        prefix="_orig_mod.",
    )

    assert set(out) == {"w", "bias"}


def test_an_empty_payload_warns_and_continues(caplog):
    """Pins warn-and-continue rather than refusal. History: KNOWN_ISSUES entry
    12 is CLOSED and records the empty-state-dict FALLTHROUGH as dead since the
    2026-07-19 round-4 cleanup routed load_checkpoint through this function.
    What survives is the warning, and the ruling (2026-08-25) is to pin the
    behavior here rather than reopen the guard -- a future refusal is a
    deliberate edit to this test, not a silent change."""
    with caplog.at_level(logging.WARNING, logger=mr.logger.name):
        out = mr.get_state_dict({})

    assert out == {}
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("empty" in m for m in msgs), msgs


def test_an_unrecognized_payload_is_named_in_the_warning_and_passed_through(caplog):
    """A checkpoint with none of the three known keys is used as-is; the
    warning names the offending keys so the caller can tell a metadata blob
    from weights."""
    payload = {"epoch": 3, "notes": "no weights in here"}

    with caplog.at_level(logging.WARNING, logger=mr.logger.name):
        out = mr.get_state_dict(payload)

    assert out == payload
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("epoch" in m and "notes" in m for m in msgs), msgs


def test_a_missing_weights_file_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError) as excinfo:
        mr.safe_load_weights(tmp_path / "nope.pth", map_location="cpu")

    assert "nope.pth" in str(excinfo.value)


def test_an_unloadable_file_becomes_a_runtimeerror(tmp_path):
    bad = tmp_path / "garbage.pth"
    bad.write_bytes(b"not a checkpoint at all")

    with pytest.raises(RuntimeError) as excinfo:
        mr.safe_load_weights(bad, map_location="cpu")

    assert "Failed to load model weights" in str(excinfo.value)


def test_a_pickled_object_is_refused_rather_than_constructed(tmp_path):
    """weights_only=True with no insecure fallback is the whole security claim
    of this loader: a checkpoint carrying a pickled object must fail loudly."""
    path = tmp_path / "pickled.pth"
    torch.save({"cfg": _NotATensor()}, path)

    with pytest.raises(RuntimeError) as excinfo:
        mr.safe_load_weights(path, map_location="cpu")

    assert "Failed to load model weights" in str(excinfo.value)
