"""Regression locks for the verify-by-execution audit fixes (2026-07-04,
2026-08-22).

Each test here reproduces a bug that existed (or a silent hazard that was
reachable) before the fix, so a revert fails loudly:

  * RoPE guard — ``model.to(bf16)`` REALifies the complex ``freqs_cis``
    buffer (PyTorch casts complex buffers with a warning, dropping the
    sin/imaginary half). Rotations then go silently wrong. The fix makes
    ``apply_rotary_embedding`` raise instead of computing garbage.
  * MemoryStore id race — serve_enigma's endpoints run in FastAPI's
    threadpool; unlocked ``add()`` minted duplicate ids under concurrency.
  * Deepcopy hygiene — a training forward must not pin graph-attached
    activations on the module (reference-model ``deepcopy`` path).
  * Chunked-CE degenerate contract — an all-ignored batch returned 0.0 with
    no grad graph where the unchunked path returns NaN, so which loss path a
    trainer took changed both the value and whether backward raised.

(The SFT-schema, curated-dataset, LoRA, and KV-share locks retired with
their modules/features in the 2026-07-18 compression pass.)
"""

import copy
import threading
import warnings

import pytest
import torch

from enigma_engine.core.memory_store import MemoryStore
from enigma_engine.core.model import Enigma, _chunked_cross_entropy
from enigma_engine.core.model_presets import ForgeConfig


def _tiny(**overrides) -> Enigma:
    torch.manual_seed(0)
    kwargs = dict(
        vocab_size=64,
        dim=32,
        n_layers=2,
        n_heads=2,
        max_seq_len=32,
        dropout=0.0,
        use_gradient_checkpointing=False,
    )
    kwargs.update(overrides)
    return Enigma(ForgeConfig(**kwargs))


def test_rope_guard_raises_on_realified_freqs():
    """model.to(bf16) corrupts the complex RoPE buffer; forward must fail loud."""
    m = _tiny()
    assert m.freqs_cis.is_complex()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # torch warns while discarding imag
        m = m.to(torch.bfloat16)
    assert not m.freqs_cis.is_complex()  # the corruption this guard exists for
    x = torch.randint(0, 64, (1, 4))
    with pytest.raises(TypeError, match="freqs_cis"):
        m(x, targets=None)


def test_rope_autocast_path_still_works():
    """The supported mixed-precision route (fp32 weights + autocast) stays open."""
    m = _tiny()
    x = torch.randint(0, 64, (1, 4))
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        out = m(x, targets=None)
    logits = out[0] if isinstance(out, tuple) else out
    assert torch.isfinite(logits.float()).all()


def test_memory_store_concurrent_add_unique_ids(tmp_path):
    store = MemoryStore(tmp_path)
    ids: list[int] = []
    lock = threading.Lock()

    def add(i: int) -> None:
        rec = store.add(f"fact number {i}")
        with lock:
            ids.append(rec["id"])

    threads = [threading.Thread(target=add, args=(i,)) for i in range(24)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(ids)) == 24, f"duplicate ids under concurrency: {sorted(ids)}"
    # and the JSONL on disk holds all 24 intact lines
    lines = [ln for ln in store.file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 24


def test_deepcopy_after_training_forward():
    """deepcopy(model) after a training forward still works.

    HONEST SCOPE (audit 2026-07-18): this replaced
    ``test_kv_share_leader_gating_and_cleanup``, which pinned the actual
    mechanism (``_shared_kv is None`` after the forward, plus leader/follower
    topology). That mechanism was DELETED with cross-layer KV sharing, so no
    current code path can pin a graph-attached activation on a module and this
    test cannot fail for that reason -- it is NOT an equivalent lock.

    What it does still guard is a live workflow: ``dpo_enigma.py`` builds its
    frozen reference via ``copy.deepcopy(policy)``, so "a forward pass must not
    make the model un-deepcopy-able" remains a real invariant, and any future
    feature that re-introduces module-pinned activations trips it.
    """
    m = _tiny(n_layers=4)
    x = torch.randint(0, 64, (2, 8))
    _, loss = m(x, targets=x)
    assert torch.isfinite(loss)
    copy.deepcopy(m)


def test_recommend_model_size_reflects_hardware():
    """The dict-facing wrapper sizes by the actual VRAM/RAM tiers rather
    than answering a constant."""
    from enigma_engine.core.model_utils import recommend_model_size

    big = {"gpu_available": True, "gpu_vram_gb": 24.0, "ram_gb": 64.0}
    tiny = {"gpu_available": False, "gpu_vram_gb": 0.0, "ram_gb": 1.0}
    assert recommend_model_size(big) == "large"
    assert recommend_model_size(tiny) == "pi_zero"


def test_config_for_param_target_keeps_preset_fields():
    """A preset match carries the WHOLE preset (rope_theta included), not
    a handful of fields with the rest reset to dataclass defaults."""
    from enigma_engine.core.model_presets import (
        MODEL_PRESETS,
        config_for_param_target,
        estimate_parameters,
    )

    target = estimate_parameters(MODEL_PRESETS["medium"])
    name, cfg = config_for_param_target(target)
    assert name == "medium"
    assert cfg.rope_theta == MODEL_PRESETS["medium"].rope_theta == 500000.0


# ---------------------------------------------------------------------------
# Chunked cross-entropy: the two loss paths must agree, degenerate case
# included (audit 2026-08-22).
# ---------------------------------------------------------------------------


def _both_ce_paths(targets: torch.Tensor):
    """(chunked, unchunked) loss for one batch, over the same weights and
    hidden states. chunk_size=2 forces several chunks over a 6-wide batch."""
    torch.manual_seed(0)
    proj = torch.nn.Linear(8, 16)
    hidden = torch.randn(1, targets.shape[1], 8, requires_grad=True)
    chunked = _chunked_cross_entropy(proj, hidden, targets, chunk_size=2, ignore_index=-100)
    full = torch.nn.functional.cross_entropy(
        proj(hidden).reshape(-1, 16), targets.reshape(-1), ignore_index=-100
    )
    return chunked, full


def test_chunked_ce_matches_the_full_path_on_an_all_ignored_batch():
    """Every target ignored: 0/0. The full path returns NaN attached to the
    graph; the chunked path used to short-circuit to a bare 0.0 whose
    backward() raised 'does not require grad'."""
    chunked, full = _both_ce_paths(torch.full((1, 6), -100, dtype=torch.long))

    assert torch.isnan(chunked) and torch.isnan(full)
    assert chunked.requires_grad == full.requires_grad
    chunked.backward()  # must not raise, exactly like the full path's


def test_chunked_ce_still_equals_the_full_path_when_some_targets_are_real():
    """The non-degenerate result must be untouched by the fix: an empty
    chunk's reduction='sum' CE is exactly 0.0, so dropping the skip changes
    nothing."""
    chunked, full = _both_ce_paths(torch.tensor([[3, -100, 7, -100, -100, 2]]))
    assert torch.allclose(chunked, full)
