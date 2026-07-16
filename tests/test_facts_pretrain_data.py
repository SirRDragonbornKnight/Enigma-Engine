"""Facts continued-pretrain corpus (make_facts_pretrain_data.interleave): the
pure-replay tail is the window pretrain uses as [val], so it must contain ZERO
fact tokens or [val] stops measuring general-domain retention (final audit
2026-07-16 M3). A fact doc used to be inserted just under mixed_end and spill
past the fence."""

from __future__ import annotations

import numpy as np

from make_facts_pretrain_data import interleave

REPLAY_TOK = 500  # distinct replay token; fact tokens are small ( < 10 )


def _replay(n: int = 200_000) -> np.ndarray:
    return np.full(n, REPLAY_TOK, dtype=np.uint32)


def test_val_tail_is_pure_replay():
    fact_docs = [[1, 2, 3, 4, 5]] * 200
    target, val_reserve = 100_000, 20_000
    out = interleave(fact_docs, _replay(), target, fact_frac=0.05,
                     val_reserve=val_reserve, chunk=2048, seed=1)
    tail = out[target - val_reserve:]
    assert (tail == REPLAY_TOK).all(), "fact tokens leaked into the [val] tail"
    # sanity: facts DID land in the mixed region (the fence isn't just dropping them all)
    assert (out[: target - val_reserve] != REPLAY_TOK).any()


def test_oversized_fact_doc_never_crosses_the_fence():
    # A doc longer than the cadence gap must not straddle mixed_end.
    fact_docs = [list(range(1, 9))] * 50 + [list(range(1, 300))]  # one long doc
    target, val_reserve = 60_000, 10_000
    out = interleave(fact_docs, _replay(), target, fact_frac=0.1,
                     val_reserve=val_reserve, chunk=1024, seed=3)
    tail = out[target - val_reserve:]
    assert (tail == REPLAY_TOK).all()
