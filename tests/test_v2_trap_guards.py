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


def test_declared_chat_specials_survive_the_pad_mask():
    """The chat/tool specials are registered in the FIRST padding rows and are
    trained there. Masking at config.vocab_size deleted `<|tool_call|>` from
    the distribution -- measured p=0.997 on a live weather ask -- so every tool
    call, every built-in loop, and the `<|im_end|>` turn ending died silently
    while plain chat still looked fine."""
    model = _tiny(vocab_size=100)  # padded head = 128; specials would sit at 100..105
    model.set_live_vocab_size(106)
    step = torch.zeros(1, 128)
    step[0, 100] = 99.0  # a declared special dominating, like <|tool_call|> does

    out = model._live_vocab_logits(step)

    assert torch.isfinite(out[0, :106]).all(), "a declared special was masked out"
    assert out[0, 100] == 99.0
    # the REST of the padding stays masked -- the original guard still holds
    assert torch.isinf(out[0, 106:]).all() and (out[0, 106:] < 0).all()


def test_live_vocab_size_refuses_a_boundary_outside_the_head():
    model = _tiny(vocab_size=100)  # padded head = 128
    with pytest.raises(ValueError):
        model.set_live_vocab_size(129)  # past the head
    with pytest.raises(ValueError):
        model.set_live_vocab_size(0)
    assert getattr(model, "live_vocab_size", None) is None  # nothing committed


def test_a_boundary_below_the_declared_vocab_is_legal_and_masks():
    """A checkpoint can declare more vocab than its tokenizer table holds; those
    rows cannot be decoded, so emitting one crashes decode. Refusing to declare
    the lower boundary pushed the caller into declaring NOTHING, and the default
    masks at config.vocab_size -- which on a vocab that is already a multiple of
    64 masks nothing at all and hands sampling every undecodable row."""
    model = _tiny(vocab_size=128)  # head == config.vocab_size: no padding to hide behind
    model.set_live_vocab_size(120)
    out = model._live_vocab_logits(torch.zeros(1, 128))
    assert torch.isfinite(out[0, :120]).all()
    assert torch.isinf(out[0, 120:]).all(), "undecodable rows stayed samplable"


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
    # A REAL warm cache this time -- the 2026-07-20 re-audit caught the first
    # version of this test passing start_pos with no cache at all, pinning an
    # overfired stateless refusal instead of the cached trap it names.
    model = _tiny()
    model.clear_cache()
    model.forward_multimodal(input_ids=torch.tensor([[1, 2, 3]]), use_cache=True)
    with pytest.raises(ValueError, match="start_pos"):
        model.forward_multimodal(input_ids=torch.tensor([[4, 5]]), use_cache=True, start_pos=3)
    model.clear_cache()


def test_multimodal_stateless_offset_scoring_allowed():
    # use_cache=False with a RoPE offset is well-defined (keys are only the
    # new tokens; the square mask is correct) and must match forward(), which
    # allows the same call.
    model = _tiny()
    mm = model.forward_multimodal(input_ids=torch.tensor([[1, 2, 3]]), start_pos=2)
    plain = model.forward(torch.tensor([[1, 2, 3]]), start_pos=2)
    assert torch.allclose(mm, plain)


def test_multimodal_cached_t1_with_mask_refuses():
    # T==1 never materializes the mask, so a cached masked call used to DROP
    # the mask silently -- refuse it like the T>1 case.
    model = _tiny()
    model.clear_cache()
    model.forward_multimodal(input_ids=torch.tensor([[1, 2, 3]]), use_cache=True)
    with pytest.raises(ValueError, match="attention_mask"):
        model.forward_multimodal(
            input_ids=torch.tensor([[4]]), use_cache=True, start_pos=3, attention_mask=torch.ones(1, 1)
        )
    model.clear_cache()


def test_multimodal_single_token_continuation_allowed():
    model = _tiny()
    model.clear_cache()
    model.forward_multimodal(input_ids=torch.tensor([[1, 2, 3]]), use_cache=True)
    logits = model.forward_multimodal(input_ids=torch.tensor([[4]]), use_cache=True, start_pos=3)
    assert logits.shape[1] == 1 and torch.isfinite(logits).all()
    # ...and it must EQUAL the uncached recompute, not merely be finite.
    model.clear_cache()
    full = model.forward_multimodal(input_ids=torch.tensor([[1, 2, 3, 4]]))
    assert torch.allclose(logits[:, -1], full[:, -1], atol=1e-5)
    model.clear_cache()


def test_forward_cached_t1_with_mask_refuses():
    # Same silent-drop hole in the main forward: T==1 + warm cache +
    # attention_mask used to be silently unmasked (verified by execution
    # 2026-07-20: output was byte-identical with and without the mask).
    model = _tiny()
    model.clear_cache()
    model.forward(torch.tensor([[1, 2, 3, 4]]), use_cache=True)
    with pytest.raises(ValueError, match="cached\\s+continuation"):
        model.forward(
            torch.tensor([[5]]), use_cache=True, start_pos=4, attention_mask=torch.ones(1, 1)
        )
    model.clear_cache()


def test_nan_logits_uniform_fallback_respects_pad_mask():
    # audit 2026-07-20: NaN real-vocab logits made every softmax NaN and the
    # uniform fallback sampled over the FULL padded width, emitting pad ids in
    # ~23% of draws -- routing around the guard in its own target regime.
    from enigma_engine.core.model_utils import sample_next_token

    step = torch.full((1, 128), float("nan"))
    step[:, 100:] = float("-inf")  # what _live_vocab_logits produces
    torch.manual_seed(0)
    for _ in range(60):
        tok = int(sample_next_token(step, torch.zeros(1, 0, dtype=torch.long), temperature=1.0, top_k=0, top_p=1.0).item())
        assert tok < 100, f"uniform fallback sampled pad id {tok}"
