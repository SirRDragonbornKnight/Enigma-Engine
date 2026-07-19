"""Old checkpoint configs must keep loading after the 2026-07-18 compression pass.

That pass deleted a dozen experimental ForgeConfig fields (use_moe,
kv_share_groups, n_predict_heads, mla_latent_dim, use_differential_attn,
neftune_alpha, tome_ratio, use_weight_norm, early_exit_layer,
use_mixture_of_depths, mod_capacity_factor, use_shifted_attention,
shifted_group_size) plus their to_dict()/from_dict() entries. EVERY
checkpoint on disk -- including the served v8 and the immutable pretrain
lineage -- still carries those keys in its config.json / model_config dict.

`ForgeConfig.from_dict` tolerates them ONLY because it filters against a
`known` set before constructing. If anyone "simplifies" it to `cls(**d)`,
every existing checkpoint dies with an unexpected-keyword TypeError at load
time -- serve, resume, finetune, dpo, and align all at once. Nothing pinned
that behavior before; these tests do.

Also pins the reverse direction: a config round-trips through
to_dict/from_dict without losing an architecture field (a dropped key would
silently rebuild a differently-shaped model and load_state_dict(strict=False)
would leave the mismatch at random init -- the silent-corruption class
pretrain_enigma.py's resume path explicitly guards against).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from enigma_engine.core.model_presets import MODEL_PRESETS, ForgeConfig

ROOT = Path(__file__).resolve().parent.parent

# Keys removed by the compression pass (and pre-existing serialization noise
# that was never a ForgeConfig field). Every one of these appears in at least
# one on-disk config.json in models/.
RETIRED_KEYS = {
    "use_moe": False,
    "num_experts": 8,
    "num_experts_per_token": 2,
    "moe_load_balancing": 0.01,
    "kv_share_groups": 0,
    "n_predict_heads": 0,
    "mla_latent_dim": 0,
    "use_differential_attn": False,
    "neftune_alpha": 0.0,
    "tome_ratio": 0.0,
    "use_weight_norm": False,
    "early_exit_layer": 0,
    "use_mixture_of_depths": False,
    "mod_capacity_factor": 0.5,
    "use_shifted_attention": False,
    "shifted_group_size": 256,
    "use_paged_attn": False,
    "kv_cache_dtype": "float16",
    "sliding_window": None,
}


def test_from_dict_ignores_retired_keys():
    """A config dict carrying every retired key still builds a ForgeConfig."""
    d = MODEL_PRESETS["nano"].to_dict()
    d.update(RETIRED_KEYS)
    cfg = ForgeConfig.from_dict(d)
    # The surviving architecture fields came through untouched.
    assert cfg.dim == MODEL_PRESETS["nano"].dim
    assert cfg.n_layers == MODEL_PRESETS["nano"].n_layers
    assert cfg.n_heads == MODEL_PRESETS["nano"].n_heads
    assert cfg.rope_theta == MODEL_PRESETS["nano"].rope_theta
    # And the retired ones did NOT become attributes.
    for key in RETIRED_KEYS:
        assert not hasattr(cfg, key), f"retired field {key} came back onto ForgeConfig"


def test_from_dict_ignores_unknown_future_keys():
    """Forward-compat too: an unrecognized key must not raise.

    Mutation check for the whole class of failure -- if from_dict ever becomes
    cls(**d), this fails with TypeError: unexpected keyword argument.
    """
    d = MODEL_PRESETS["nano"].to_dict()
    d["some_field_invented_next_year"] = 42
    cfg = ForgeConfig.from_dict(d)
    assert cfg.dim == MODEL_PRESETS["nano"].dim


def test_to_dict_from_dict_round_trip_preserves_architecture():
    """Round-tripping must not drop an architecture field.

    A dropped key rebuilds a differently-shaped model; load_state_dict(
    strict=False) then leaves the mismatched tensors at random init -- silent
    corruption rather than a loud failure.
    """
    original = MODEL_PRESETS["medium"]
    rebuilt = ForgeConfig.from_dict(original.to_dict())
    shape_fields = (
        "vocab_size", "dim", "n_layers", "n_heads", "n_kv_heads", "hidden_dim",
        "max_seq_len", "use_rope", "use_rms_norm", "use_swiglu", "use_bias",
        "rope_theta", "rope_scaling_type", "rope_scaling_factor",
        "use_qk_norm", "use_layer_scale", "drop_path_rate",
        "vision_hidden_size", "audio_hidden_size",
    )
    for field in shape_fields:
        assert getattr(rebuilt, field) == getattr(original, field), f"round-trip lost {field}"


def test_to_dict_covers_every_public_field():
    """to_dict is a hand-written dict: a field added to the dataclass but
    forgotten there would silently vanish from every checkpoint's config.
    Pin the two sets equal so the omission is impossible to miss."""
    import dataclasses

    public = {f.name for f in dataclasses.fields(ForgeConfig) if not f.name.startswith("_")}
    assert set(ForgeConfig().to_dict().keys()) == public


def test_from_dict_preserves_every_public_field():
    """from_dict must pass EVERY public field through -- a field it strips
    rebuilds the model at that field's default (the silent-corruption class
    this file's docstring describes). Probes each field with a non-default,
    construction-valid value; special cases keep the divisibility rules
    (dim % n_heads, n_heads % n_kv_heads) intact."""
    import dataclasses

    base = ForgeConfig().to_dict()
    special = {
        "dim": 1024,
        "n_heads": 16,
        "n_kv_heads": 4,
        "hidden_dim": 2048,
        "rope_scaling_type": "linear",
        "vision_hidden_size": 32,
        "audio_hidden_size": 32,
    }
    for f in dataclasses.fields(ForgeConfig):
        if f.name.startswith("_"):
            continue
        default = base[f.name]
        if f.name in special:
            probe = special[f.name]
        elif isinstance(default, bool):
            probe = not default
        elif isinstance(default, int):
            probe = default + 1
        elif isinstance(default, float):
            probe = default + 0.125
        elif isinstance(default, str):
            probe = default + "_probe"
        else:
            raise AssertionError(f"no probe rule for field {f.name} (default {default!r})")
        cfg = ForgeConfig.from_dict({**base, f.name: probe})
        assert getattr(cfg, f.name) == probe, f"from_dict dropped {f.name}"


def _on_disk_configs() -> list[Path]:
    return sorted((ROOT / "models").glob("*/config.json"))


@pytest.mark.skipif(not (ROOT / "models").is_dir(), reason="no models/ dir on this box")
def test_every_on_disk_config_still_loads():
    """The real checkpoints on this machine load, not just synthetic dicts.

    Skips cleanly on a fresh clone with no models/ (the repo ships without
    weights), so this never becomes a machine-specific failure.
    """
    configs = _on_disk_configs()
    if not configs:
        pytest.skip("no checkpoint configs present")
    for path in configs:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if "dim" not in raw:  # foreign (e.g. HF) config, not a ForgeConfig dump
            continue
        cfg = ForgeConfig.from_dict(raw)
        assert cfg.dim == raw["dim"], f"{path} loaded with the wrong dim"
        assert cfg.n_layers == raw["n_layers"], f"{path} loaded with the wrong n_layers"
