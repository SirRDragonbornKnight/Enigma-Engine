"""Memory-read supervision must never be self-contradictory (audit m9).

`gen_memory_read_examples` teaches Enigma to PICK the right line out of an
injected "Things you remember:" block, so each record ships 1-2 distractor
facts alongside the target. Before 2026-07-19 the distractor pool was every
OTHER fact, which let a block assert both "User's favorite color is green."
and "User likes the color orange." while the trained answer named just one:
the question has two valid answers in context, so the record teaches an
arbitrary pick rather than retrieval. `_CONFLICTING_FACTS` groups the facts
that answer the same question and the sampler excludes the target's group.

These tests pin the contract AND the mechanism's integrity -- a renamed fact
string would otherwise silently drop out of its group and re-open the hole.
"""

from __future__ import annotations

import make_sft_data
from make_sft_data import _CONFLICTING_FACTS, gen_memory_read_examples, gen_memory_tools_examples


def _blocks(records: list[dict]) -> list[list[str]]:
    """The remembered-fact lines of every record's system message."""
    out = []
    for rec in records:
        content = rec["messages"][0]["content"]
        block = content.split("\n\n")[0]  # memory_tools appends the tool spec
        if not block.startswith("Things you remember:"):
            continue
        out.append([ln[2:] for ln in block.splitlines()[1:] if ln.startswith("- ")])
    return out


def test_conflict_groups_reference_real_facts():
    """Guards the mechanism: every grouped string must still be a live fact.
    If a fact is reworded and its group entry isn't, the exclusion silently
    stops matching and contradictory blocks come back."""
    recs = gen_memory_read_examples()
    live = {line for block in _blocks(recs) for line in block}
    for group in _CONFLICTING_FACTS:
        for fact in group:
            assert fact in live, f"_CONFLICTING_FACTS names a fact that no longer exists: {fact!r}"


def test_conflict_groups_are_not_vacuous():
    """A group of one excludes nothing -- it would pass the test above while
    protecting nothing."""
    for group in _CONFLICTING_FACTS:
        assert len(group) >= 2, f"conflict group {group!r} needs 2+ members to exclude anything"


def test_no_block_contains_contradictory_facts():
    """The contract: no remembered-block may carry two facts that answer the
    same question."""
    for block in _blocks(gen_memory_read_examples()):
        for group in _CONFLICTING_FACTS:
            present = [f for f in block if f in group]
            assert len(present) <= 1, f"contradictory facts in one block: {present}"


def test_memory_tools_records_inherit_the_deconfliction():
    """gen_memory_tools_examples reuses the memory-read records wholesale, so
    the joined tool-block shape must carry the same guarantee."""
    for block in _blocks(gen_memory_tools_examples()):
        for group in _CONFLICTING_FACTS:
            present = [f for f in block if f in group]
            assert len(present) <= 1, f"contradictory facts in one memory_tools block: {present}"


def test_target_fact_still_present_with_distractors():
    """De-confliction must not have emptied the distractor pool. Blocks are
    either the off-topic single line (memory present but irrelevant) or the
    PICK-don't-parrot shape: target + 1-2 distractors."""
    blocks = _blocks(gen_memory_read_examples())
    assert blocks, "no remembered-blocks generated"
    assert all(1 <= len(b) <= 3 for b in blocks), "block outside the 1-3 remembered-line shape"
    with_distractors = [b for b in blocks if len(b) > 1]
    # The pick-lesson records are the bulk; a collapse to singletons would
    # mean the exclusion emptied the pool.
    assert len(with_distractors) > len(blocks) - len(with_distractors)


def test_generation_is_deterministic():
    assert gen_memory_read_examples() == gen_memory_read_examples()
    assert make_sft_data.gen_memory_read_examples(seed=5) == make_sft_data.gen_memory_read_examples(seed=5)
