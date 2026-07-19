"""KV-cache correctness for the from-scratch Enigma — the serving path.

The model's ``generate``/``generate_stream`` decode token-by-token with
``use_cache=True`` (prefill once, then one new token per step at an
advancing ``start_pos``). For that to be *correct*, the logits produced by
the incremental cached path must equal the logits from a full no-cache
recompute over the same realized sequence — otherwise served output silently
diverges from the model's true distribution.

These tests lock that equivalence — the deep model coverage for the
serving path.

⚠️ COVERAGE MATRIX — learned the hard way. A genuine KV-cache decode bug
(SDPA called with ``is_causal=True`` for the rectangular q_len=1 / k_len=L
decode step, which top-left-aligns the mask so the new token sees only key 0)
shipped *despite* an earlier version of this test passing — because that
version ran only on CPU, which routes around the buggy branch (CPU uses the
standard non-SDPA attention path). The broken branch is **CUDA + SDPA
incremental decode** — exactly the production serving config. So we cross
**device × position-scheme** and assert cache==no-cache on every combination;
the CUDA cells are the ones that actually exercise SDPA incremental decode.
(Differential attention was removed in the 2026-07-18 compression pass, so
the old attention-variant axis is gone.)
"""

from dataclasses import replace

import pytest
import torch

from enigma_engine.core.kv_cache import (
    H2OKVCache,
    KVCache,
    StreamingLLMCache,
    TurboQuantKVCache,
)
from enigma_engine.core.model import Enigma
from enigma_engine.core.model_presets import MODEL_PRESETS

# Run on CPU always; add CUDA when present (no-op skip on CPU-only boxes —
# honors the "works on any device" constraint).
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _tiny(
    use_rope: bool = True,
    device: str = "cpu",
    seed: int = 0,
) -> Enigma:
    """A small, deterministic Enigma in eval mode (dropout off) on ``device``."""
    torch.manual_seed(seed)
    cfg = replace(
        MODEL_PRESETS["nano"],
        vocab_size=64,
        max_seq_len=64,
        use_rope=use_rope,
    )
    # Sanity: the preset we lean on must keep n_kv_heads < n_heads so the
    # cache↔GQA-repeat interaction is actually under test.
    assert cfg.n_kv_heads < cfg.n_heads, "nano preset must use GQA for this test"
    return Enigma(cfg).eval().to(device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("use_rope", [True, False], ids=["rope", "learned_pos"])
@torch.no_grad()
def test_kv_cache_decode_matches_no_cache(use_rope: bool, device: str):
    """Incremental cached decode == full no-cache recompute, logit-for-logit.

    Walk a realized sequence one token at a time: at each step compare the
    cached path's next-token logits against recomputing the whole sequence
    from scratch with ``use_cache=False``. They must agree to within float32
    numerical noise at *every* step (prefill + each decode step), and the
    greedy argmax must agree (that's what actually drives generation).

    The CUDA parametrization is the production serving path and the one a
    CPU-only test silently skipped.
    """
    model = _tiny(use_rope=use_rope, device=device)
    vocab = model.config.vocab_size
    torch.manual_seed(1)
    prefix = torch.randint(0, vocab, (1, 5), device=device)

    def ref_last_logits(seq: torch.Tensor) -> torch.Tensor:
        return model(seq, use_cache=False)[0, -1]

    # Prefill: cache the prompt, compare against a plain forward over it.
    model.clear_cache()
    cached = model(prefix, use_cache=True)[0, -1]
    seq = prefix.clone()
    assert torch.allclose(cached, ref_last_logits(seq), atol=1e-4), "prefill diverged"

    # Decode: feed each new token (argmax → deterministic) with an advancing
    # start_pos, exactly as Enigma.generate does, and re-verify each step.
    for step in range(1, 9):
        nxt = cached.argmax().view(1, 1)
        seq = torch.cat([seq, nxt], dim=1)
        cached = model(nxt, use_cache=True, start_pos=seq.shape[1] - 1)[0, -1]
        ref = ref_last_logits(seq)
        assert torch.allclose(cached, ref, atol=1e-4), (
            f"decode step {step} (len {seq.shape[1]}) diverged on {device}: "
            f"max|Δ|={(cached - ref).abs().max().item():.2e}"
        )
        assert cached.argmax() == ref.argmax(), f"argmax disagrees at step {step} on {device}"


@pytest.mark.parametrize("device", DEVICES)
@torch.no_grad()
def test_generate_is_deterministic_and_clear_cache_resets(device: str):
    """``generate`` is greedy-deterministic and ``clear_cache`` fully resets state.

    Two back-to-back greedy generations (top_k=1) from the same prompt must
    produce byte-identical token sequences — proving the per-call
    ``clear_cache()`` wipes the previous run's KV entries (a stale-cache leak
    would corrupt the second call). Drives the SDPA decode path on CUDA."""
    model = _tiny(device=device)
    prompt = torch.tensor([[1, 7, 11, 3]], device=device)

    out1 = model.generate(prompt, max_new_tokens=12, temperature=1.0, top_k=1)
    out2 = model.generate(prompt, max_new_tokens=12, temperature=1.0, top_k=1)

    assert out1.shape[0] == 1 and out1.shape[1] > prompt.shape[1], "nothing generated"
    assert torch.equal(out1, out2), "greedy generate is not deterministic / cache leaked between runs"
    # The prompt is preserved verbatim as the prefix of the output.
    assert torch.equal(out1[:, : prompt.shape[1]], prompt)


@pytest.mark.parametrize("device", DEVICES)
@torch.no_grad()
def test_generate_respects_stop_token(device: str):
    """Generation halts as soon as a stop token is emitted."""
    model = _tiny(device=device)
    prompt = torch.tensor([[1, 7, 11, 3]], device=device)

    # Discover the first token greedy-generation would emit, then make THAT a
    # stop token: generation should append it and immediately halt.
    full = model.generate(prompt, max_new_tokens=12, temperature=1.0, top_k=1)
    first_gen = int(full[0, prompt.shape[1]].item())

    stopped = model.generate(prompt, max_new_tokens=12, temperature=1.0, top_k=1, stop_tokens=[first_gen])
    assert stopped.shape[1] == prompt.shape[1] + 1, "did not stop on the stop token"
    assert int(stopped[0, -1].item()) == first_gen


# ---------------------------------------------------------------------------
# KVCache-class unit coverage: capacity limits, eviction correctness, and
# mixed-precision storage invariants. CPU-only, deterministic.
# ---------------------------------------------------------------------------


def _cache_kwargs(**overrides) -> dict:
    kw = dict(
        batch_size=1,
        max_seq_len=8,
        n_kv_heads=2,
        head_dim=8,
        device=torch.device("cpu"),
    )
    kw.update(overrides)
    return kw


def test_kv_cache_oversize_update_raises():
    """An update longer than the cache must raise, not silently truncate."""
    cache = KVCache(**_cache_kwargs())
    k = torch.randn(1, 9, 2, 8)
    with pytest.raises(ValueError, match="capacity"):
        cache.update(k, k.clone())


@torch.no_grad()
def test_turboquant_rebalance_preserves_history():
    """Flipping a head's INT4/INT8 assignment re-encodes its cached history.

    Without the re-encode, get() reads a flipped head's past positions from
    a buffer that was never written (zeros / default scale), returning
    garbage for every token cached before the rebalance."""
    torch.manual_seed(0)
    cache = TurboQuantKVCache(
        int4_fraction=0.5,
        rebalance_interval=10_000,  # no automatic rebalance during the test
        **_cache_kwargs(max_seq_len=16, n_kv_heads=4, head_dim=8),
    )
    k = torch.empty(1, 6, 4, 8).uniform_(-1.0, 1.0)
    v = torch.empty(1, 6, 4, 8).uniform_(-1.0, 1.0)
    cache.update(k, v)
    before_k, before_v = cache.get()

    # Make the current INT4 heads the most important ones, forcing a full
    # assignment swap on rebalance.
    importance = torch.zeros_like(cache._head_importance)
    importance[cache._is_int4] = 10.0
    cache._head_importance = importance
    old_assign = cache._is_int4.clone()
    cache.rebalance()
    assert not torch.equal(old_assign, cache._is_int4), "rebalance did not flip any head"

    after_k, after_v = cache.get()
    # Re-encoding costs at most one extra quantization step of error
    # (INT4 step is ~0.13 * scale for values in [-1, 1]).
    assert torch.allclose(before_k, after_k, atol=0.2), "K history corrupted by rebalance"
    assert torch.allclose(before_v, after_v, atol=0.2), "V history corrupted by rebalance"


@torch.no_grad()
def test_turboquant_overflow_shift_keeps_all_heads_consistent():
    """Overflow must shift the INT4 buffers together with the INT8 ones,
    and new tokens must land in whichever buffer get() reads per head."""
    torch.manual_seed(0)
    cache = TurboQuantKVCache(
        int4_fraction=0.5,
        rebalance_interval=10_000,
        **_cache_kwargs(max_seq_len=4, n_kv_heads=4, head_dim=8),
    )
    k = torch.empty(1, 4, 4, 8).uniform_(-1.0, 1.0)
    v = torch.empty(1, 4, 4, 8).uniform_(-1.0, 1.0)
    cache.update(k, v)

    k_new = torch.empty(1, 1, 4, 8).uniform_(-1.0, 1.0)
    v_new = torch.empty(1, 1, 4, 8).uniform_(-1.0, 1.0)
    cache.update(k_new, v_new)  # end_pos 5 > max_seq_len 4 -> sliding shift

    got_k, got_v = cache.get()
    assert got_k.shape[1] == 4
    # Layout after the shift: original tokens 1..3, then the new token.
    assert torch.allclose(got_k[:, :3], k[:, 1:], atol=0.2)
    assert torch.allclose(got_k[:, -1:], k_new, atol=0.2)
    assert torch.allclose(got_v[:, :3], v[:, 1:], atol=0.2)
    assert torch.allclose(got_v[:, -1:], v_new, atol=0.2)


@torch.no_grad()
def test_h2o_eviction_selects_heavy_hitters_per_batch_row():
    """Each batch row keeps its own heavy hitters, not row 0's."""
    cache = H2OKVCache(
        heavy_hitter_count=2,
        recent_window=2,
        **_cache_kwargs(batch_size=2, max_seq_len=16),
    )
    # Token t in row b stores the constant value t + 100 * b (fp cache, exact).
    n_tok = 8
    k = torch.zeros(2, n_tok, 2, 8)
    for b in range(2):
        for t in range(n_tok):
            k[b, t] = t + 100 * b
    cache.update(k, k.clone())

    # Row 0's heavy hitters are tokens 0 and 1; row 1's are tokens 3 and 4.
    scores = torch.zeros(2, cache.max_seq_len)
    scores[0, 0] = scores[0, 1] = 5.0
    scores[1, 3] = scores[1, 4] = 5.0
    cache._attn_scores = scores

    cache.evict_if_needed()
    assert cache.current_pos == 4  # 2 heavy hitters + 2 recent

    got_k, _ = cache.get()

    def row_tokens(row: int) -> list[int]:
        return [round(float(got_k[row, t, 0, 0])) - 100 * row for t in range(4)]

    assert row_tokens(0) == [0, 1, 6, 7]
    assert row_tokens(1) == [3, 4, 6, 7]
    # Evicted tail slots are zeroed.
    assert torch.all(cache._cache_k[:, 4:8] == 0)
    assert torch.all(cache._cache_v[:, 4:8] == 0)


@torch.no_grad()
def test_h2o_quantized_eviction_zeroes_tail():
    """The quantized path zeroes evicted slots like the fp path does,
    and resets their scales/zero-points to the defaults."""
    torch.manual_seed(0)
    cache = H2OKVCache(
        heavy_hitter_count=2,
        recent_window=2,
        **_cache_kwargs(max_seq_len=16, quantize_to_int8=True),
    )
    k = torch.randn(1, 8, 2, 8)
    cache.update(k, k.clone())
    cache._attn_scores[0, 0] = cache._attn_scores[0, 1] = 5.0

    cache.evict_if_needed()
    assert cache.current_pos == 4
    assert torch.all(cache._cache_k[:, 4:8] == 0)
    assert torch.all(cache._cache_v[:, 4:8] == 0)
    assert torch.all(cache._scale_k[:, 4:8] == 1.0)
    assert torch.all(cache._scale_v[:, 4:8] == 1.0)
    assert torch.all(cache._zp_k[:, 4:8] == 0)
    assert torch.all(cache._zp_v[:, 4:8] == 0)


@torch.no_grad()
def test_streaming_cache_shifts_zero_points_with_data():
    """Window-shift eviction must move zero-points along with data/scales.

    Constant-vector tokens quantize to q=0 with zero_point = the value, so
    each retained token decodes exactly to its zero-point — a stale zp is
    directly visible as the wrong token value."""
    cache = StreamingLLMCache(
        n_sink=2,
        window_size=4,
        **_cache_kwargs(max_seq_len=8, quantize_to_int8=True),
    )

    def tok(t: int) -> torch.Tensor:
        return torch.full((1, 1, 2, 8), 100.0 * t)

    k = torch.cat([tok(t) for t in range(6)], dim=1)
    cache.update(k, k.clone())
    cache.update(tok(6), tok(6))  # exceeds sink+window budget -> shift

    got_k, got_v = cache.get()
    vals_k = [round(float(got_k[0, t, 0, 0]) / 100.0) for t in range(got_k.shape[1])]
    vals_v = [round(float(got_v[0, t, 0, 0]) / 100.0) for t in range(got_v.shape[1])]
    # Sinks 0-1 stay; token 2 is evicted; window keeps 3-5 plus the new 6.
    assert vals_k == [0, 1, 3, 4, 5, 6]
    assert vals_v == [0, 1, 3, 4, 5, 6]


@torch.no_grad()
def test_h2o_overflow_shift_moves_scores_with_positions():
    """A sliding-window overflow shift rolls the score accumulator with
    the K/V it scores, and the slots the new tokens land in start at zero."""
    cache = H2OKVCache(
        heavy_hitter_count=2,
        recent_window=2,
        **_cache_kwargs(max_seq_len=8),
    )
    k = torch.zeros(1, 8, 2, 8)
    for t in range(8):
        k[0, t] = t
    cache.update(k, k.clone())
    cache._attn_scores[0] = torch.arange(8, dtype=torch.float32)

    # 2 more tokens overflow: positions shift left by 2
    k2 = torch.zeros(1, 2, 2, 8)
    k2[0, 0], k2[0, 1] = 8, 9
    cache.update(k2, k2.clone())

    # Surviving positions carry their own scores (score t follows token t)
    assert cache._attn_scores[0, :6].tolist() == [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    # The new tokens' slots accumulated nothing yet
    assert cache._attn_scores[0, 6:].tolist() == [0.0, 0.0]
    # And the K/V really did shift (token value == original position)
    keys, _ = cache.get()
    assert keys[0, 0, 0, 0].item() == 2.0
    assert keys[0, 7, 0, 0].item() == 9.0
