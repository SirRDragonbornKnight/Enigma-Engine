"""The future-run training arsenal (2026-06-11 landscape additions): WSD
schedule, Muon optimizer, composite optimizer state, min-p plumbing.

The contract these tests enforce: the DEFAULTS (--optimizer adamw,
--schedule cosine) must keep reproducing the live 51k run's math exactly —
same LR values, same optimizer grouping/order (its state_dict must keep
fitting the lineage's checkpoints).
"""

import math
from dataclasses import replace

import pytest
import torch

from enigma_engine.core import optim as pe


def _old_cosine(step, warmup, total, peak, min_ratio=0.1):
    """The pre-arsenal get_lr, verbatim — the live run's recorded schedule."""
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    if step >= total:
        return peak * min_ratio
    prog = (step - warmup) / max(1, total - warmup)
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog)))


def _nano():
    from enigma_engine.core.model import Enigma
    from enigma_engine.core.model_presets import MODEL_PRESETS

    return Enigma(replace(MODEL_PRESETS["nano"], vocab_size=256))


def test_cosine_path_bit_identical_to_live_run():
    warmup, total, peak = 200, 287_882, 6e-4
    steps = [0, 1, 50, 199, 200, 201, 5_000, 51_000, 143_941, 287_881, 287_882, 400_000]
    for step in steps:
        assert pe.get_lr(step, warmup, total, peak) == _old_cosine(step, warmup, total, peak)


def test_wsd_shape_warmup_plateau_decay_to_zero():
    warmup, total, peak = 10, 1000, 1.0

    def lr(s):
        return pe.get_lr(s, warmup, total, peak, schedule="wsd", decay_frac=0.1)

    assert lr(0) == pytest.approx(peak / 10)  # warmup ramps
    assert lr(9) == pytest.approx(peak)  # warmup ends at peak
    assert lr(500) == peak  # stable plateau
    assert lr(899) == peak  # last stable step
    decay = [lr(s) for s in range(900, 1000)]
    assert decay[0] == peak  # decay starts from peak
    assert all(a > b for a, b in zip(decay, decay[1:]))  # strictly decreasing
    assert lr(999) > 0.0
    assert lr(1000) == 0.0  # decay-to-ZERO, not to a floor


def test_newton_schulz_orthogonalizes_both_orientations():
    for shape in [(16, 64), (64, 16)]:  # wide + tall (transpose branch)
        g = torch.randn(*shape)
        out = pe._newton_schulz5(g)
        assert out.shape == g.shape and out.dtype == g.dtype
        s = torch.linalg.svdvals(out.float())
        # bf16 iteration: singular values driven toward 1, loosely bounded
        assert s.max() < 1.7 and s.min() > 0.3


def test_muon_step_updates_finite_and_decays():
    p = torch.nn.Parameter(torch.randn(8, 16))
    before = p.detach().clone()
    opt = pe.Muon([p], lr=0.02, weight_decay=0.5)
    p.grad = torch.randn_like(p)
    opt.step()
    assert not torch.allclose(p, before)
    assert torch.isfinite(p).all()

    # Decoupled decay applies even when the orthogonalized update is zero.
    p2 = torch.nn.Parameter(torch.randn(4, 4))
    b2 = p2.detach().clone()
    opt2 = pe.Muon([p2], lr=0.1, weight_decay=0.5)
    p2.grad = torch.zeros_like(p2)
    opt2.step()
    assert torch.allclose(p2, b2 * (1 - 0.1 * 0.5))


def test_builder_adamw_matches_live_grouping_exactly():
    m = _nano()
    opt = pe.build_optimizer(m, "adamw", lr=1e-3, weight_decay=0.1)
    assert isinstance(opt, torch.optim.AdamW)
    decay = [p for p in m.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay = [p for p in m.parameters() if p.requires_grad and p.dim() < 2]
    # Same params in the same parameters()-iteration ORDER — the optimizer
    # state_dict maps params by group index, so order is lineage-compatibility.
    assert [id(p) for p in opt.param_groups[0]["params"]] == [id(p) for p in decay]
    assert [id(p) for p in opt.param_groups[1]["params"]] == [id(p) for p in no_decay]
    assert opt.param_groups[0]["weight_decay"] == 0.1
    assert opt.param_groups[1]["weight_decay"] == 0.0


def test_builder_muon_splits_body_from_embeddings():
    m = _nano()
    opt = pe.build_optimizer(m, "muon", lr=1e-3, weight_decay=0.1)
    muon_ids = {id(p) for g in opt.opts[0].param_groups for p in g["params"]}
    assert id(m.tok_embeddings.weight) not in muon_ids  # tied head stays on AdamW
    assert all(p.dim() >= 2 for g in opt.opts[0].param_groups for p in g["params"])
    assert all(p.dim() < 2 for p in opt.opts[1].param_groups[1]["params"])
    # Every trainable param is covered exactly once across both halves.
    covered = sorted(id(p) for o in opt.opts for g in o.param_groups for p in g["params"])
    want = sorted(id(p) for p in m.parameters() if p.requires_grad)
    assert covered == want


def test_composite_state_roundtrip_and_mismatch_guard():
    m = _nano()
    opt = pe.build_optimizer(m, "muon", lr=1e-3, weight_decay=0.1)
    for p in m.parameters():
        p.grad = torch.randn_like(p) * 0.01
    opt.step()
    sd = opt.state_dict()

    m2 = _nano()
    opt2 = pe.build_optimizer(m2, "muon", lr=1e-3, weight_decay=0.1)
    opt2.load_state_dict(sd)  # fits, and stays usable:
    for p in m2.parameters():
        p.grad = torch.randn_like(p) * 0.01
    opt2.step()

    # Cross-loading fails LOUDLY in both directions (the resume guard's premise).
    adamw = pe.build_optimizer(_nano(), "adamw", lr=1e-3, weight_decay=0.1)
    with pytest.raises(ValueError):
        opt2.load_state_dict(adamw.state_dict())
    with pytest.raises(Exception):
        adamw.load_state_dict(sd)


def test_composite_param_groups_propagate_lr():
    opt = pe.build_optimizer(_nano(), "muon", lr=1e-3, weight_decay=0.1)
    for g in opt.param_groups:  # exactly how the training loop sets lr
        g["lr"] = 0.123
    assert all(g["lr"] == 0.123 for o in opt.opts for g in o.param_groups)


def test_min_p_plumbs_through_generation(monkeypatch):
    """min_p must actually REACH the sampler. The old version asserted only
    output length, which cannot fail while generate runs at all -- silently
    dropping the kwarg stayed green (test-suite audit 2026-07-17)."""
    import enigma_engine.core.model as model_mod

    seen: list[float] = []
    real = model_mod.sample_next_token

    def spy(logits, generated_tokens, *args, **kwargs):
        seen.append(kwargs.get("min_p"))
        return real(logits, generated_tokens, *args, **kwargs)

    monkeypatch.setattr(model_mod, "sample_next_token", spy)
    m = _nano().eval()
    ids = torch.randint(0, 256, (1, 8))
    out = m.generate(ids, max_new_tokens=3, temperature=0.8, min_p=0.5)
    assert out.shape[1] > ids.shape[1]
    assert seen and all(v == 0.5 for v in seen)

    seen.clear()
    toks = list(m.generate_stream(ids, max_new_tokens=3, min_p=0.5))
    assert 1 <= len(toks) <= 3
    assert seen and all(v == 0.5 for v in seen)
