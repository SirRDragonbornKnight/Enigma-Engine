"""DPO train/val split (dpo_enigma.group_split): all pairs sharing a prompt
must land on the same side. The generated pairs carry the same (prompt,
chosen) against several rejected variants, and user-taught pairs ride x3 as
exact duplicates -- a flat shuffle put twins on both sides and the val
"preference accuracy" partly measured training-set recall (audit
2026-07-15)."""

from __future__ import annotations

from dpo_enigma import group_split


def _pairs(n_prompts: int = 40, per_prompt: int = 3) -> list[tuple[str, str, str]]:
    # chosen row encodes its prompt so sides can be reconstructed after the
    # prompt is stripped from the returned rows.
    return [
        (f"q{i}", f"c{i}", f"r{i}.{j}")
        for i in range(n_prompts)
        for j in range(per_prompt)
    ]


def test_no_prompt_straddles_the_split():
    train, val = group_split(_pairs(), val_frac=0.25, seed=7)
    train_prompts = {c for c, _ in train}
    val_prompts = {c for c, _ in val}
    assert val, "val must be non-empty at this size"
    assert not (train_prompts & val_prompts)
    assert len(train) + len(val) == 120


def test_exact_duplicate_pairs_stay_on_one_side():
    # User-taught pairs ride x3 as byte-identical triples.
    pairs = _pairs(20, 1) + [("taught?", "good", "bad")] * 3
    train, val = group_split(pairs, val_frac=0.3, seed=3)
    dupes_in_val = sum(1 for row in val if row == ("good", "bad"))
    dupes_in_train = sum(1 for row in train if row == ("good", "bad"))
    assert (dupes_in_val, dupes_in_train) in ((3, 0), (0, 3))


def test_split_is_deterministic_and_respects_zero_val():
    assert group_split(_pairs(), 0.25, seed=7) == group_split(_pairs(), 0.25, seed=7)
    train, val = group_split(_pairs(), 0.0, seed=7)
    assert val == [] and len(train) == 120
