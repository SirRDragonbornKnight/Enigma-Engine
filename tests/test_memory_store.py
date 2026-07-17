"""The local memory layer (memory_store.py) — her runtime learning lives here,
not in the frozen weights. Stdlib BM25 over inspectable JSONL."""

import pytest

from enigma_engine.core.memory_store import MemoryStore
from enigma_engine.core.tokenizer import get_tokenizer


@pytest.fixture(scope="module")
def tok():
    return get_tokenizer("bpe")


def test_add_search_relevance(tmp_path):
    m = MemoryStore(tmp_path)
    m.add("The user's cat is named Miso and sleeps on the GPU box.")
    m.add("The avatar's favorite emote is the little wave.")
    m.add("Pasta water needs more salt than feels reasonable.")
    hits = m.search("what is the cat called", k=2)
    assert hits and "Miso" in hits[0]["text"]
    assert m.search("zorbulating frizzlewumps") == []  # no shared terms, no match


def test_persistence_across_reopen(tmp_path):
    MemoryStore(tmp_path).add("Enigma paused at step 51000.")
    m2 = MemoryStore(tmp_path)
    assert len(m2) == 1
    assert m2.search("paused step")[0]["text"].startswith("Enigma paused")


def test_render_context_respects_budget_and_silence(tmp_path, tok):
    m = MemoryStore(tmp_path)
    assert m.render_context("anything", tok) == ""  # empty store -> silence
    for i in range(8):
        m.add(f"Fact number {i}: the workshop window {i} faces the rain.")
    # space-heavy tokenizer: ~2 tokens per word — budget for a couple of lines
    ctx = m.render_context("rain window workshop", tok, max_ids=80, k=8)
    assert ctx.startswith("Things you remember:")
    assert ctx.count("\n") < 8  # the budget pruned the hit list
    n_ids = len(tok.encode(ctx, add_special_tokens=False))
    assert n_ids <= 88  # budget held (small slack for the header line)
    # irrelevant query -> no context injected, never noise
    assert m.render_context("quantum frogs", tok) == ""


def test_empty_memory_rejected(tmp_path):
    with pytest.raises(ValueError):
        MemoryStore(tmp_path).add("   ")


# --- remember(): the tool entry point (dedupe + supersede + provenance) ---


def test_remember_exact_duplicate_is_idempotent(tmp_path):
    m = MemoryStore(tmp_path)
    a = m.remember("User's dog is named Rex.")
    b = m.remember("user's dog is named rex.")  # case-insensitive duplicate
    assert len(m) == 1
    assert b["id"] == a["id"]


def test_remember_supersedes_contradicting_fact(tmp_path):
    m = MemoryStore(tmp_path)
    m.remember("User's dog is named Rex.")
    rec = m.remember("User's dog is named Bruno.")
    assert len(m) == 1  # replaced, not stacked
    assert rec["superseded"] == "User's dog is named Rex."
    assert m.search("dog name")[0]["text"].endswith("Bruno.")


def test_remember_keeps_unrelated_facts_apart(tmp_path):
    m = MemoryStore(tmp_path)
    m.remember("User's dog is named Rex.")
    m.remember("User's cat is named Whiskers.")
    m.remember("User lives in Denver.")
    assert len(m) == 3  # low content overlap -> separate facts


def test_remember_stamps_provenance(tmp_path):
    rec = MemoryStore(tmp_path).remember("User prefers short answers.", source="chat")
    assert rec["date"] and rec["source"] == "chat" and rec["kind"] == "user_fact"


def test_supersede_persists_across_reopen(tmp_path):
    m = MemoryStore(tmp_path)
    m.remember("User's car is a red hatchback.")
    m.remember("User's car is a silver hatchback.")
    m2 = MemoryStore(tmp_path)
    assert len(m2) == 1
    assert "silver hatchback" in m2.all()[0]["text"]


def test_low_overlap_correction_coexists_documented_limit(tmp_path):
    # The documented lexical limit: rewording MOST of the fact does not
    # supersede (needs semantics). Locks the behavior so a future fix is
    # a deliberate change, not an accident.
    m = MemoryStore(tmp_path)
    m.remember("User's car is a red hatchback.")
    m.remember("User's car is a silver van.")
    assert len(m) == 2


def test_add_never_reuses_an_id_after_delete(tmp_path):
    # add() must mint max+1, not len+1: after a delete (or a supersede) a
    # len-based id collides with a surviving record, and delete-by-id then
    # hits the wrong one.
    m = MemoryStore(tmp_path)
    m.add("First fact.")
    b = m.add("Second fact.")
    c = m.add("Third fact.")
    m.delete(b["id"])
    d = m.add("Fourth fact.")
    ids = [r["id"] for r in m.all()]
    assert len(ids) == len(set(ids))  # every id unique
    assert d["id"] > c["id"]  # ids only move forward
    assert m.delete(d["id"]) is True  # targeted delete hits the new record


def test_delete_and_clear(tmp_path):
    m = MemoryStore(tmp_path)
    a = m.remember("User's dog is named Rex.")
    m.remember("User lives in Denver.")
    assert m.delete(a["id"]) is True
    assert m.delete(999) is False
    assert len(m) == 1
    assert m.clear() == 1
    assert len(m) == 0
    assert len(MemoryStore(tmp_path)) == 0  # clear persisted


def test_load_renumbers_duplicate_ids_even_when_file_is_readonly(tmp_path):
    """Renumbered ids go live in memory even when the file cannot be
    rewritten (read-only attribute); loading must not raise."""
    import os
    import stat

    f = tmp_path / "memories.jsonl"
    f.write_text('{"id": 1, "text": "alpha"}\n{"id": 1, "text": "beta"}\n', encoding="utf-8")
    os.chmod(f, stat.S_IREAD)
    try:
        m = MemoryStore(tmp_path)
        ids = [r["id"] for r in m.all()]
        assert len(set(ids)) == 2, f"duplicate ids survived load: {ids}"
    finally:
        os.chmod(f, stat.S_IREAD | stat.S_IWRITE)


def test_delete_and_clear_leave_no_backup_copy(tmp_path):
    """Privacy contract (re-audit 2026-07-17): the fsync'd rewrite must NOT
    rotate a .bak -- a pre-delete copy on disk would silently defeat
    delete/clear. Any .bak an earlier build left behind is scrubbed too."""
    m = MemoryStore(tmp_path)
    m.add("secret one")
    m.add("secret two")
    bak = (tmp_path / "memories.jsonl").with_suffix(".jsonl.bak")
    bak.write_text("stale pre-delete copy from an older build\n", encoding="utf-8")
    m.delete(1)
    assert not bak.exists(), "delete left a backup copy on disk"
    m.clear()
    assert not bak.exists()
    assert (tmp_path / "memories.jsonl").read_text(encoding="utf-8") == ""
