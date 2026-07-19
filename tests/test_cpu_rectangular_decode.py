"""Non-SDPA (CPU/MPS) attention must handle RECTANGULAR cached decode.

BACKLOG #14, confirmed by execution 2026-07-18 and fixed the same pass. The
standard attention fallback built a SQUARE (T, T) causal mask and added it to
scores shaped (B, heads, T, T_k). Whenever a cached continuation feeds more
than one new token, T_k > T and the add broadcast-crashed:

    RuntimeError: The size of tensor a (9) must match the size of tensor b (3)

The live serve loop never hit it (prefill once, then one token at a time), so
it sat open; anything doing chunked prefill or multi-token continuation on CPU
died. The fix mirrors what the SDPA branch already did: a BOTTOM-RIGHT aligned
mask, tril(diagonal=T_k - T).

These tests pin two properties of the FIXED branch:
  * it no longer crashes, and
  * the values are right -- cached multi-token continuation equals a full
    no-cache recompute over the same sequence, which is the only thing that
    makes the mask alignment observable (a wrong-but-shaped mask would pass a
    smoke test and silently corrupt attention).

SCOPE LIMIT (be precise about what is NOT covered). The fix and these tests
cover the ``mask is None`` branch only -- the plain-causal case, which is what
every live path uses. The sibling branches still receive the SQUARE (B,1,T,T)
mask that ``model.py`` builds via ``_get_causal_mask(T)``:

    model_components.py  ``scores = scores + mask``      (standard path)
    model_components.py  ``attn_mask=mask``              (SDPA path)

With ``use_cache=True`` AND a warm cache AND an explicit padding
``attention_mask``, T_k > T and those two would broadcast-crash the same way.
That combination is currently UNREACHABLE: serve_enigma.py never passes
``attention_mask`` and neither does ``forward_multimodal`` -- padded batches
only appear in training, which never uses the cache. Fixing it properly means
teaching ``model.py`` the cached key length; deliberately deferred rather than
silently claimed. If a future path ever pairs a pad mask with a warm cache,
this is the first place to look.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from enigma_engine.core.model import Enigma
from enigma_engine.core.model_presets import MODEL_PRESETS


def _cpu_model(seed: int = 0) -> Enigma:
    """Tiny model on CPU -- forced onto the standard (non-SDPA) branch."""
    torch.manual_seed(seed)
    cfg = replace(MODEL_PRESETS["nano"], vocab_size=64, max_seq_len=64)
    model = Enigma(cfg).eval()
    assert not next(model.parameters()).is_cuda, "this test must run the CPU branch"
    return model


@torch.no_grad()
def test_multi_token_cached_continuation_does_not_crash():
    """The exact BACKLOG #14 repro: prefill 6, then feed 3 more with the cache."""
    model = _cpu_model()
    prefix = torch.randint(0, 64, (1, 6))
    model.clear_cache()
    model(prefix, use_cache=True)
    out = model(torch.randint(0, 64, (1, 3)), use_cache=True, start_pos=6)
    assert out.shape == (1, 3, model.config.vocab_size)
    assert torch.isfinite(out).all()


@torch.no_grad()
@pytest.mark.parametrize("chunk", [2, 3, 5], ids=["chunk2", "chunk3", "chunk5"])
def test_chunked_cached_prefill_matches_no_cache(chunk: int):
    """Values, not just shapes: chunked cached prefill == one full recompute.

    This is what catches a mask that is correctly SHAPED but wrongly ALIGNED
    (e.g. top-left tril, which would let each query see too few keys).
    """
    model = _cpu_model(seed=1)
    torch.manual_seed(2)
    seq = torch.randint(0, 64, (1, 11))

    reference = model(seq, use_cache=False)[0, -1]

    model.clear_cache()
    pos = 0
    cached = None
    while pos < seq.shape[1]:
        piece = seq[:, pos : pos + chunk]
        cached = model(piece, use_cache=True, start_pos=pos)[0, -1]
        pos += piece.shape[1]

    assert torch.allclose(cached, reference, atol=1e-4), (
        f"chunked cached prefill (chunk={chunk}) diverged from no-cache recompute: "
        f"max|delta|={(cached - reference).abs().max().item():.2e}"
    )
    assert cached.argmax() == reference.argmax()


@torch.no_grad()
def test_square_prefill_still_causal():
    """The T_k == T case must stay strictly causal (mask reduces to plain triu).

    Guard against "fixing" the rectangular case by loosening the square one:
    if causality broke, an early position's logits would change when LATER
    tokens in the same forward are altered.
    """
    model = _cpu_model(seed=3)
    torch.manual_seed(4)
    base = torch.randint(0, 64, (1, 8))
    altered = base.clone()
    altered[0, -1] = (int(base[0, -1]) + 1) % 64  # change only the LAST token

    out_base = model(base, use_cache=False)
    out_altered = model(altered, use_cache=False)

    # Every position before the change must be untouched by it.
    assert torch.allclose(out_base[0, :-1], out_altered[0, :-1], atol=1e-5), (
        "a later token changed earlier positions -- causality is broken"
    )
