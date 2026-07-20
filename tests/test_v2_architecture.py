"""Arc-B v2 architecture options (2026-07-20 research verdicts).

Three opt-in config fields whose DEFAULTS must reproduce v1 exactly (every
shipped checkpoint predates them), plus the deep-thin size-call presets:

- norm_scheme "peri": also norm each sub-block's output inside the residual.
- qk_norm_before_rope: v1 rotates then norms; v2 norms then rotates.
- init_scheme "olmo2_flat": flat N(0, 0.02) everywhere (no wo/w2 depth scale).
- v2_deep_186m / v2_deep_238m / v2_deep_542m presets carrying the v2 recipe.
"""

from __future__ import annotations

import pytest
import torch

from enigma_engine.core.model import Enigma
from enigma_engine.core.model_presets import (
    ForgeConfig,
    MODEL_PRESETS,
    estimate_parameters,
    get_preset,
)


def _cfg(**kw) -> ForgeConfig:
    base = dict(
        vocab_size=64, dim=64, n_layers=8, n_heads=2, n_kv_heads=2, max_seq_len=32, dropout=0.0
    )
    base.update(kw)
    return ForgeConfig(**base)


# ---------------------------------------------------------------------------
# Back-compat: defaults are v1
# ---------------------------------------------------------------------------


def test_old_checkpoint_config_dict_gets_v1_defaults():
    # A shipped checkpoint's config blob has none of the new keys.
    old = {"vocab_size": 64, "dim": 64, "n_layers": 2, "n_heads": 2, "max_seq_len": 32}
    cfg = ForgeConfig.from_dict(old)
    assert cfg.norm_scheme == "pre"
    assert cfg.qk_norm_before_rope is False
    assert cfg.init_scheme == "gpt2"


def test_default_model_has_no_peri_params():
    model = Enigma(_cfg(n_layers=2))
    assert not any("post_norm" in k for k in model.state_dict()), (
        "a default (v1) config grew post-norm params -- every shipped checkpoint would fail to load"
    )


def test_new_fields_roundtrip_to_dict():
    cfg = _cfg(norm_scheme="peri", qk_norm_before_rope=True, init_scheme="olmo2_flat")
    back = ForgeConfig.from_dict(cfg.to_dict())
    assert back.norm_scheme == "peri"
    assert back.qk_norm_before_rope is True
    assert back.init_scheme == "olmo2_flat"


def test_invalid_schemes_refused():
    with pytest.raises(ValueError, match="norm_scheme"):
        _cfg(norm_scheme="post")
    with pytest.raises(ValueError, match="init_scheme"):
        _cfg(init_scheme="gamma1")


# ---------------------------------------------------------------------------
# Peri-LN
# ---------------------------------------------------------------------------


def test_peri_model_has_post_norms_and_trains():
    model = Enigma(_cfg(n_layers=2, norm_scheme="peri"))
    keys = model.state_dict().keys()
    for i in range(2):
        assert f"layers.{i}.attention_post_norm.weight" in keys
        assert f"layers.{i}.ffn_post_norm.weight" in keys
    logits, loss = model.forward(torch.tensor([[1, 2, 3]]), targets=torch.tensor([[2, 3, 4]]))
    assert torch.isfinite(loss)
    loss.backward()
    g = model.layers[0].attention_post_norm.weight.grad
    assert g is not None and torch.isfinite(g).all(), "post-norm got no gradient -- not in the graph"


def test_peri_changes_the_function():
    torch.manual_seed(7)
    pre = Enigma(_cfg(n_layers=2))
    torch.manual_seed(7)
    peri = Enigma(_cfg(n_layers=2, norm_scheme="peri"))
    x = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        assert not torch.allclose(pre.forward(x), peri.forward(x)), (
            "peri output identical to pre -- the post-norms are not applied"
        )


# ---------------------------------------------------------------------------
# QK-norm order
# ---------------------------------------------------------------------------


def test_qk_order_flag_changes_outputs():
    # Same seed -> identical parameters (the flag adds none). Subtlety: RoPE
    # rotations preserve norms, and at init the RMSNorm weight is all-ones, so
    # BOTH orders compute the same function until the norm weight trains away
    # from 1 (this test originally compared fresh models and proved nothing).
    # Give both models the same NON-uniform norm weights; then rotation no
    # longer commutes with the elementwise scale and the orders must differ.
    torch.manual_seed(11)
    after = Enigma(_cfg(n_layers=1))
    torch.manual_seed(11)
    before = Enigma(_cfg(n_layers=1, qk_norm_before_rope=True))
    head_dim = 64 // 2
    ramp = torch.linspace(0.5, 1.5, head_dim)
    with torch.no_grad():
        for m in (after, before):
            m.layers[0].attention.q_norm.weight.copy_(ramp)
            m.layers[0].attention.k_norm.weight.copy_(ramp)
    x = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        assert not torch.allclose(after.forward(x), before.forward(x)), (
            "qk_norm_before_rope=True produced v1's rotate-then-norm math"
        )


def test_qk_order_flag_inert_without_qk_norm():
    torch.manual_seed(11)
    a = Enigma(_cfg(n_layers=1, use_qk_norm=False))
    torch.manual_seed(11)
    b = Enigma(_cfg(n_layers=1, use_qk_norm=False, qk_norm_before_rope=True))
    x = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        assert torch.allclose(a.forward(x), b.forward(x))


# ---------------------------------------------------------------------------
# Init scheme
# ---------------------------------------------------------------------------


def test_gpt2_init_depth_scales_wo():
    torch.manual_seed(3)
    model = Enigma(_cfg())  # n_layers=8 -> scaled std = 0.02/4 = 0.005
    std = model.layers[0].attention.wo.weight.std().item()
    assert 0.003 < std < 0.007, f"wo std {std}: v1 depth-scaled init changed"


def test_olmo2_flat_init_keeps_wo_at_002():
    torch.manual_seed(3)
    model = Enigma(_cfg(init_scheme="olmo2_flat"))
    for tensor_name in ("attention.wo.weight", "feed_forward.w2.weight"):
        mod = model.layers[0]
        obj = mod
        for part in tensor_name.split("."):
            obj = getattr(obj, part)
        std = obj.std().item()
        assert 0.017 < std < 0.023, f"{tensor_name} std {std}: olmo2_flat did not stay flat"


# ---------------------------------------------------------------------------
# Deep-thin presets
# ---------------------------------------------------------------------------


V2_EXPECTED = {
    "v2_deep_186m": (768, 28, 186e6),
    "v2_deep_238m": (1024, 20, 238e6),
    "v2_deep_542m": (1280, 30, 542e6),
}


@pytest.mark.parametrize("name", sorted(V2_EXPECTED))
def test_v2_presets_carry_the_recipe(name):
    dim, layers, params = V2_EXPECTED[name]
    cfg = get_preset(name, vocab_size=16384)
    assert (cfg.dim, cfg.n_layers) == (dim, layers)
    assert cfg.dim // cfg.n_heads == 64, "head_dim must be 64"
    assert cfg.n_heads == 4 * cfg.n_kv_heads, "GQA must stay 4:1"
    assert cfg.dropout == 0.0, "single-epoch pretrain must not carry dropout"
    assert cfg.max_seq_len == 8192
    # The v2 recipe itself -- and via get_preset, which used to hand-copy 8
    # fields and would have silently dropped all three of these.
    assert cfg.norm_scheme == "peri"
    assert cfg.qk_norm_before_rope is True
    assert cfg.init_scheme == "olmo2_flat"
    est = estimate_parameters(cfg)
    assert abs(est - params) / params < 0.03, f"{name}: {est / 1e6:.1f}M vs expected {params / 1e6:.0f}M"


def test_v1_presets_untouched_by_v2_fields():
    for name in ("small", "medium", "base", "large", "xl"):
        cfg = MODEL_PRESETS[name]
        assert cfg.norm_scheme == "pre" and not cfg.qk_norm_before_rope and cfg.init_scheme == "gpt2", (
            f"preset {name} picked up v2 architecture flags -- v1 lineages would rebuild differently"
        )
