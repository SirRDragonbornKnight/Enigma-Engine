"""DPO train/val split (dpo_enigma.group_split): all pairs sharing a prompt
must land on the same side. The generated pairs carry the same (prompt,
chosen) against several rejected variants, and user-taught pairs ride x3 as
exact duplicates -- a flat shuffle put twins on both sides and the val
"preference accuracy" partly measured training-set recall (audit
2026-07-15). Since the Stage-C review (2026-08-10) each pair also carries a
class tag through the split, so end-of-run accuracy can be reported per
class -- rows are (class, chosen, rejected)."""

from __future__ import annotations

from dpo_enigma import group_split


def _pairs(n_prompts: int = 40, per_prompt: int = 3) -> list[tuple[str, str, str, str]]:
    # chosen row encodes its prompt so sides can be reconstructed after the
    # prompt is stripped from the returned rows.
    return [
        (f"q{i}", "identity", f"c{i}", f"r{i}.{j}")
        for i in range(n_prompts)
        for j in range(per_prompt)
    ]


def test_no_prompt_straddles_the_split():
    train, val = group_split(_pairs(), val_frac=0.25, seed=7)
    train_prompts = {c for _, c, _ in train}
    val_prompts = {c for _, c, _ in val}
    assert val, "val must be non-empty at this size"
    assert not (train_prompts & val_prompts)
    assert len(train) + len(val) == 120


def test_exact_duplicate_pairs_stay_on_one_side():
    # User-taught pairs ride x3 as byte-identical triples.
    pairs = _pairs(20, 1) + [("taught?", "untagged", "good", "bad")] * 3
    train, val = group_split(pairs, val_frac=0.3, seed=3)
    dupes_in_val = sum(1 for row in val if row == ("untagged", "good", "bad"))
    dupes_in_train = sum(1 for row in train if row == ("untagged", "good", "bad"))
    assert (dupes_in_val, dupes_in_train) in ((3, 0), (0, 3))


def test_split_is_deterministic_and_respects_zero_val():
    assert group_split(_pairs(), 0.25, seed=7) == group_split(_pairs(), 0.25, seed=7)
    train, val = group_split(_pairs(), 0.0, seed=7)
    assert val == [] and len(train) == 120


def test_single_prompt_never_empties_train():
    # A tiny teach_pairs.jsonl can be a single prompt with a few rejected rows.
    # It must ALL train (val empty) rather than land in val and crash
    # _batchify([]) at the first train step (final audit 2026-07-16 M2).
    pairs = [("only?", "untagged", "good", f"bad{j}") for j in range(4)]
    train, val = group_split(pairs, val_frac=0.5, seed=1)
    assert len(train) == 4 and val == []


def test_oversized_group_cannot_empty_train_or_overshoot_val():
    # One giant prompt group among a few small ones: the giant group must go to
    # train (never straddle into val), train must be non-empty, and val must
    # not exceed its target (the old greedy fill added a whole group while under
    # target, so a giant first group blew past it).
    pairs = [("big?", "identity", "c", f"r{j}") for j in range(50)]
    pairs += [(f"q{i}", "identity", f"c{i}", "r") for i in range(6)]
    n_val_target = min(int(len(pairs) * 0.25), 64)
    train, val = group_split(pairs, val_frac=0.25, seed=5, val_cap=64)
    assert train, "train must never be empty"
    assert len(val) <= n_val_target
    assert sum(1 for row in train if row[1] == "c") == 50  # the 50-row group trains
    assert not any(row[1] == "c" for row in val)
