"""Arc-A v2 trap guards (2026-07-20 gap audit).

Four latent traps found while mapping the v2 research verdicts to code:

1. KV cache hardcap: Attention clamped its cache to MAX_CACHE_SEQ_LEN=4096
   regardless of config.max_seq_len -- an 8k-context v2 config would silently
   slide the cache and forget the earliest tokens. Cache now follows config.
2. Sampleable pad rows: the vocab is padded to a multiple of 64 for GPU
   alignment, and sampling ran over the FULL padded head -- ids >=
   config.vocab_size (random-init rows on a fresh/early checkpoint) could win
   top-k/argmax and crash decode. generate/generate_stream now -inf them.
3. Explicit mask + warm cache: forward materializes pad/packing masks SQUARE
   over the new tokens, blind to cached keys -- broadcast-crash latent
   (KNOWN_ISSUES #12). Now a loud ValueError refusal.
4. start_pos-blind multimodal mask: forward_multimodal built the same square
   mask for any T>1 regardless of start_pos. Cached multi-token continuation
   now refuses; single-token continuation stays supported.
"""

from __future__ import annotations

import pytest
import torch

from enigma_engine.core.model import Enigma
from enigma_engine.core.model_presets import ForgeConfig


def _tiny(vocab_size: int = 100, max_seq_len: int = 64) -> Enigma:
    cfg = ForgeConfig(
        vocab_size=vocab_size,
        dim=32,
        n_layers=1,
        n_heads=2,
        n_kv_heads=2,
        max_seq_len=max_seq_len,
        dropout=0.0,
    )
    model = Enigma(cfg)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# 1. KV cache cap follows config
# ---------------------------------------------------------------------------


def test_kv_cache_cap_follows_config():
    model = _tiny(max_seq_len=8192)
    attn = model.layers[0].attention
    assert attn.max_cache_len == 8192, (
        f"max_cache_len={attn.max_cache_len}: the old min(max_seq_len, 4096) clamp is back -- "
        "an 8k v2 config would silently slide its KV cache"
    )


def test_kv_cache_cap_matches_v1_configs_exactly():
    # The live v1 lineage runs max_seq_len=4096; the clamp removal must be a
    # no-op there (4096 either way).
    model = _tiny(max_seq_len=4096)
    assert model.layers[0].attention.max_cache_len == 4096


# ---------------------------------------------------------------------------
# 2. Pad-row sampling guard
# ---------------------------------------------------------------------------


def test_live_vocab_logits_masks_alignment_padding():
    model = _tiny(vocab_size=100)  # padded head = 128
    step = torch.zeros(1, 128)
    step[0, 120] = 99.0  # a pad column dominating
    out = model._live_vocab_logits(step)
    assert torch.isinf(out[0, 100:]).all() and (out[0, 100:] < 0).all()
    assert torch.isfinite(out[0, :100]).all()
    # caller's tensor must not be mutated in place
    assert step[0, 120] == 99.0


def test_live_vocab_logits_noop_when_head_unpadded():
    model = _tiny(vocab_size=128)  # already a multiple of 64 -> no pad columns
    step = torch.randn(1, 128)
    out = model._live_vocab_logits(step)
    assert torch.equal(out, step)


@pytest.mark.parametrize("streaming", [False, True])
def test_pad_ids_never_sampled(streaming):
    # Fresh random weights: pad logits are the same magnitude as real ones, so
    # unguarded sampling picks a pad id (28/128 per token) with probability
    # ~1 - 0.78^60 -- this test is a deterministic kill for a deleted guard.
    torch.manual_seed(0)
    model = _tiny(vocab_size=100)
    prompt = torch.tensor([[1, 2, 3]])
    ids: list[int] = []
    for seed in (0, 1):
        torch.manual_seed(seed)
        if streaming:
            ids += [
                int(t.item())
                for t in model.generate_stream(
                    prompt, max_new_tokens=30, temperature=1.0, top_k=0, top_p=1.0, stop_tokens=[-1]
                )
            ]
        else:
            out = model.generate(
                prompt, max_new_tokens=30, temperature=1.0, top_k=0, top_p=1.0, stop_tokens=[-1]
            )
            ids += out[0, 3:].tolist()
    assert ids, "generation produced no tokens -- test proves nothing"
    bad = [i for i in ids if i >= 100]
    assert not bad, f"sampled alignment-padding token ids {bad} (>= vocab_size=100)"


# ---------------------------------------------------------------------------
# 3. Explicit mask + warm cache refuses (was a latent broadcast crash)
# ---------------------------------------------------------------------------


def test_explicit_mask_with_warm_cache_refuses():
    model = _tiny()
    model.clear_cache()
    model.forward(torch.tensor([[1, 2, 3, 4]]), use_cache=True)
    with pytest.raises(ValueError, match="cached\\s+continuation"):
        model.forward(
            torch.tensor([[5, 6]]),
            use_cache=True,
            start_pos=4,
            attention_mask=torch.ones(1, 2),
        )
    model.clear_cache()


def test_fresh_padded_prefill_still_allowed():
    # The refusal must not overfire: a padded prefill on a COLD cache is the
    # supported case (square mask matches the key length).
    model = _tiny()
    model.clear_cache()
    logits = model.forward(
        torch.tensor([[1, 2, 3, 0]]),
        use_cache=True,
        attention_mask=torch.tensor([[1, 1, 1, 0]]),
    )
    assert torch.isfinite(logits[:, :3]).all()
    model.clear_cache()


def test_padded_training_forward_still_allowed():
    # No cache at all (the training path) must be untouched by the guard.
    model = _tiny()
    logits = model.forward(
        torch.tensor([[1, 2, 3, 0]]),
        attention_mask=torch.tensor([[1, 1, 1, 0]]),
    )
    assert torch.isfinite(logits[:, :3]).all()


# ---------------------------------------------------------------------------
# 4. Multimodal cached continuation
# ---------------------------------------------------------------------------


def test_multimodal_cached_multitoken_refuses():
    model = _tiny()
    with pytest.raises(ValueError, match="start_pos"):
        model.forward_multimodal(input_ids=torch.tensor([[1, 2, 3]]), start_pos=2)


def test_multimodal_single_token_continuation_allowed():
    model = _tiny()
    model.clear_cache()
    model.forward_multimodal(input_ids=torch.tensor([[1, 2, 3]]), use_cache=True)
    logits = model.forward_multimodal(input_ids=torch.tensor([[4]]), use_cache=True, start_pos=3)
    assert logits.shape[1] == 1 and torch.isfinite(logits).all()
    model.clear_cache()
