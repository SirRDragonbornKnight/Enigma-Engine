"""Regression locks for the 2026-07-04 verify-by-execution audit fixes.

Each test here reproduces a bug that existed (or a silent hazard that was
reachable) before the fix, so a revert fails loudly:

  * RoPE guard — ``model.to(bf16)`` REALifies the complex ``freqs_cis``
    buffer (PyTorch casts complex buffers with a warning, dropping the
    sin/imaginary half). Rotations then go silently wrong. The fix makes
    ``apply_rotary_embedding`` raise instead of computing garbage.
  * MemoryStore id race — serve_enigma's endpoints run in FastAPI's
    threadpool; unlocked ``add()`` minted duplicate ids under concurrency.
  * KV-share hygiene — only group leaders may pin ``_shared_kv``, and the
    pin must not outlive the forward (VRAM + ``deepcopy`` for DPO/KTO
    reference models).
  * SFT schema — dispatch materializes a .jsonl 'data' path into a list
    BEFORE validation, so list-shaped data must validate (and empty must
    still fail loud).
  * Curated dataset — one corrupt JSONL line must not discard the whole
    dataset (a later save() would then clobber every curated entry).
"""

import copy
import json
import threading
import warnings

import pytest
import torch

from enigma_engine.core.curated_dataset import CuratedDataset, DatasetEntry
from enigma_engine.core.memory_store import MemoryStore
from enigma_engine.core.model import Enigma
from enigma_engine.core.model_presets import ForgeConfig
from enigma_engine.training.schema import TrainingJobConfig


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


def test_kv_share_leader_gating_and_cleanup():
    m = _tiny(n_layers=4, kv_share_groups=2)
    leaders = [i for i, lay in enumerate(m.layers) if lay.attention._kv_is_leader]
    followers = [i for i, lay in enumerate(m.layers) if lay.attention._kv_share_source is not None]
    assert leaders == [0, 2]
    assert followers == [1, 3]

    x = torch.randint(0, 64, (2, 8))
    _, loss = m(x, targets=x)
    assert torch.isfinite(loss)
    pinned = [i for i, lay in enumerate(m.layers) if lay.attention._shared_kv is not None]
    assert pinned == [], f"_shared_kv outlived the forward on layers {pinned}"
    # graph-free modules must deepcopy (DPO/KTO reference-model path)
    copy.deepcopy(m)


def test_sft_schema_accepts_materialized_list():
    cfg = TrainingJobConfig(
        mode="sft", data=[{"messages": [{"role": "user", "content": "hi"}]}]
    )
    assert isinstance(cfg.data, list)
    with pytest.raises(ValueError):
        TrainingJobConfig(mode="sft", data=[])
    with pytest.raises(ValueError):
        TrainingJobConfig(mode="sft", data="")


def test_curated_dataset_survives_corrupt_line(tmp_path):
    path = tmp_path / "ds.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(DatasetEntry(text="example one", source="test").to_dict()),
                "{corrupt json!!",
                json.dumps(DatasetEntry(text="example two", source="test").to_dict()),
            ]
        ),
        encoding="utf-8",
    )
    ds = CuratedDataset(path=path)
    assert len(ds._entries) == 2
    ds.save()
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2


def test_kv_share_config_normalizes_gradient_checkpointing():
    """kv_share_groups + checkpointing cannot work (followers reuse the
    leader's K/V, which checkpointed recomputation cannot rebuild): the
    config normalizes the flag off so stamped checkpoints still load, and
    the explicit enabler refuses outright."""
    cfg = ForgeConfig(
        vocab_size=64, dim=32, n_layers=2, n_heads=2, max_seq_len=32,
        kv_share_groups=2, use_gradient_checkpointing=True,
    )
    assert cfg.use_gradient_checkpointing is False
    # round-trip: a checkpoint config stamped with the old default loads
    cfg2 = ForgeConfig.from_dict(cfg.to_dict() | {"use_gradient_checkpointing": True})
    assert cfg2.use_gradient_checkpointing is False
    m = _tiny(n_layers=4, kv_share_groups=2)
    with pytest.raises(ValueError):
        m.gradient_checkpointing_enable()


def test_load_lora_mismatched_adapter_stays_unregistered(tmp_path):
    """A LoRA file whose keys match nothing raises AND leaves no phantom
    entry in _lora_adapters (a later merge_lora() must find nothing)."""
    m = _tiny()
    p = tmp_path / "bad_adapter.pth"
    torch.save({"nonexistent.lora_A.weight": torch.zeros(2, 2)}, p)
    with pytest.raises(ValueError):
        m.load_lora(p, adapter_name="bad")
    assert "bad" not in getattr(m, "_lora_adapters", {})
    m.merge_lora()  # nothing registered -> warning path, no raise
