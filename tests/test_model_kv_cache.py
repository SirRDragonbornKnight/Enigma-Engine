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

from enigma_engine.core.kv_cache import KVCache, KVCacheFull
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
# KVCache-class unit coverage: capacity limits. CPU-only, deterministic.
# ---------------------------------------------------------------------------


def test_kv_cache_oversize_update_raises():
    """An update longer than the cache must raise, not silently truncate."""
    cache = KVCache(
        batch_size=1,
        max_seq_len=8,
        n_kv_heads=2,
        head_dim=8,
        device=torch.device("cpu"),
    )
    k = torch.randn(1, 9, 2, 8)
    with pytest.raises(ValueError, match="capacity"):
        cache.update(k, k.clone())


# ---------------------------------------------------------------------------
# The window edge (audit 2026-08-22). The cache used to SLIDE on overflow
# while RoPE kept absolute positions: generation silently dropped its oldest
# context, warned once per layer per token, then died on the RoPE table at
# exactly 2x max_seq_len. Serve cannot reach it (its --max-context clamp), a
# direct generate() caller can. The window edge is now a clean stop.
# ---------------------------------------------------------------------------

WINDOW = 16
PROMPT_LEN = 12


def _window_model() -> Enigma:
    torch.manual_seed(0)
    cfg = replace(MODEL_PRESETS["nano"], vocab_size=64, max_seq_len=WINDOW, use_rope=True)
    return Enigma(cfg).eval()


def _window_prompt(device: str = "cpu") -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randint(0, 64, (1, PROMPT_LEN), device=device)


# A token id the tiny model will not greedily emit is not knowable up front;
# an out-of-vocab stop id is never emitted, which is what these need.
NEVER_STOPS = [999]


@torch.no_grad()
def test_generation_stops_at_the_window_edge_instead_of_sliding():
    model, prompt = _window_model(), _window_prompt()
    out = model.generate(prompt, max_new_tokens=40, temperature=0.0, stop_tokens=NEVER_STOPS)
    # The cache holds WINDOW positions; the last token is the prediction made
    # FROM the final in-window position, so it is real output, not a tail.
    assert out.shape[1] == WINDOW + 1


@torch.no_grad()
def test_the_tail_is_not_degraded_by_the_stop():
    """Every token returned was produced under the model's true context: the
    over-window run agrees with a wholly in-window run, token for token."""
    model, prompt = _window_model(), _window_prompt()
    short = model.generate(prompt, max_new_tokens=4, temperature=0.0, stop_tokens=NEVER_STOPS)
    over = model.generate(prompt, max_new_tokens=40, temperature=0.0, stop_tokens=NEVER_STOPS)
    assert torch.equal(over[:, : short.shape[1]], short)


@torch.no_grad()
def test_twice_the_window_no_longer_crashes():
    """The old failure was a ValueError out of apply_rotary_embedding at
    exactly start_pos == 2 * max_seq_len."""
    model, prompt = _window_model(), _window_prompt()
    out = model.generate(prompt, max_new_tokens=2 * WINDOW + 8, temperature=0.0, stop_tokens=NEVER_STOPS)
    assert out.shape[1] == WINDOW + 1


@torch.no_grad()
def test_exactly_at_the_boundary_is_untouched():
    """The in-window case must not pay for the guard: prompt + new == window
    fits exactly and generates every token asked for, with nothing logged."""
    model, prompt = _window_model(), _window_prompt()
    out = model.generate(
        prompt, max_new_tokens=WINDOW - PROMPT_LEN, temperature=0.0, stop_tokens=NEVER_STOPS
    )
    assert out.shape[1] == WINDOW


@torch.no_grad()
def test_the_stop_is_one_note_not_one_per_layer(caplog):
    """The warning used to come from the cache -- one line per layer per
    token (64 lines on a 4-layer model). It belongs to the caller that
    actually stops."""
    model, prompt = _window_model(), _window_prompt()
    assert model.config.n_layers > 1, "a one-layer model cannot show the difference"
    with caplog.at_level("WARNING", logger="enigma_engine.core.model"):
        model.generate(prompt, max_new_tokens=40, temperature=0.0, stop_tokens=NEVER_STOPS)
    notes = [r for r in caplog.records if "context limit" in r.message]
    assert len(notes) == 1, [r.message for r in caplog.records]


@torch.no_grad()
def test_generate_stream_stops_at_the_window_edge_too():
    model, prompt = _window_model(), _window_prompt()
    got = list(model.generate_stream(prompt, max_new_tokens=40, temperature=0.0, stop_tokens=NEVER_STOPS))
    assert len(got) == WINDOW + 1 - PROMPT_LEN


def test_the_cache_refuses_to_slide():
    """The unit fact underneath: a write past the end raises instead of
    rolling the buffers, so the stored positions never shift out from under
    RoPE's absolute ones."""
    cache = KVCache(batch_size=1, max_seq_len=8, n_kv_heads=2, head_dim=8, device=torch.device("cpu"))
    k = torch.randn(1, 8, 2, 8)
    cache.update(k, k.clone())
    before = cache.get()[0].clone()

    one_more = torch.randn(1, 1, 2, 8)
    with pytest.raises(KVCacheFull):
        cache.update(one_more, one_more.clone())
    assert cache.current_pos == 8, "a refused write must not move the position"
    assert torch.equal(cache.get()[0], before), "the cache slid anyway"
