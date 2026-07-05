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
