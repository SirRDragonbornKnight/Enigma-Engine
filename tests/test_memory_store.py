"""The local memory layer (memory_store.py) — her runtime learning lives here,
not in the frozen weights. Stdlib BM25 over inspectable JSONL."""

import pytest

from enigma_engine.core.memory_store import MemoryStore
from enigma_engine.core.tokenizer import get_tokenizer


@pytest.fixture(scope="module")
def tok():
    return get_tokenizer("bpe")


def _two_store(tmp_path, first, second):
    """A fresh store with two facts remembered in order (unique dir per pair)."""
    m = MemoryStore(tmp_path / str(abs(hash((first, second))) % 999983))
    m.remember(first)
    m.remember(second)
    return m


def _two(tmp_path, first, second):
    return _two_store(tmp_path, first, second).all()


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


def test_reworded_correction_supersedes_on_attribute(tmp_path):
    # Same attribute ("car"), fully reworded value: one fact, the new one.
    m = MemoryStore(tmp_path)
    m.remember("User's car is a red hatchback.")
    m.remember("User's car is a silver van.")
    assert len(m) == 1
    assert "silver van" in m.all()[0]["text"]


def test_shared_value_across_attributes_never_deletes(tmp_path):
    # Two people can share a name; the store must keep both. Lexical overlap
    # here is 0.60 -- above the old 0.5 bar, which destroyed the first record.
    m = MemoryStore(tmp_path)
    m.remember("User's brother is named Leo.")
    m.remember("User's sister is named Leo.")
    texts = {r["text"] for r in m.all()}
    assert texts == {"User's brother is named Leo.", "User's sister is named Leo."}


def test_recall_matches_possessive_and_inflected_queries(tmp_path):
    m = MemoryStore(tmp_path)
    m.remember("User's dog is named Rex.")
    m.remember("User works as a teacher.")
    assert "Rex" in m.search("What's my dog's name?")[0]["text"]
    assert "teacher" in m.search("What do I do for work?")[0]["text"]


def test_stopword_only_query_retrieves_nothing(tmp_path):
    # "is" appears in every stored fact; a query made only of stopwords must
    # not inject arbitrary personal facts into the prompt.
    m = MemoryStore(tmp_path)
    m.remember("User's sister is Nora.")
    m.remember("User's favorite color is teal.")
    assert m.search("is that right?") == []
    assert m.search("is it?") == []


def test_complementary_facts_about_one_subject_coexist(tmp_path):
    # The subject alone is not the fact's identity: a name and an age about
    # the same subject are two facts. Keying supersede on the subject made
    # "he's 3" destroy the dog's name (audit 2026-07-22) -- the exact shape
    # the model is trained to store.
    cases = [
        ("User's dog is named Rex.", "User's dog is 3 years old."),
        ("My brother is named Leo.", "My brother is 25 years old."),
        ("My sister's cat is named Biscuit.", "My sister's cat is orange."),
        ("User's kids are named Ana and Ben.", "User's kids are 5 and 7."),
    ]
    for first, second in cases:
        m = MemoryStore(tmp_path / str(abs(hash(first)) % 99999))
        m.remember(first)
        m.remember(second)
        assert len(m) == 2, (first, second)


def test_same_kind_corrections_still_supersede(tmp_path):
    # Rename replaces the name, age update replaces the age.
    m = MemoryStore(tmp_path / "rename")
    m.remember("User's dog is named Rex.")
    m.remember("User's dog is named Bruno.")
    assert [r["text"] for r in m.all()] == ["User's dog is named Bruno."]

    m2 = MemoryStore(tmp_path / "age")
    m2.remember("User's dog is 3 years old.")
    m2.remember("User's dog is 4 years old.")
    assert len(m2) == 1
    assert "4 years old" in m2.all()[0]["text"]


def test_single_valued_relation_corrections_supersede(tmp_path):
    # The trained "User VERBs X." shapes with a single-valued relation keep
    # their correction path: these superseded before the 0.75 lexical bar
    # orphaned them (audit 2026-07-22).
    cases = [
        ("User lives in Denver.", "User lives in Austin.", "Austin"),
        ("User works as a teacher.", "User works as a nurse.", "nurse"),
        ("User drives a blue pickup.", "User drives a red sedan.", "red sedan"),
        ("User's name is Sam.", "User's name is Samantha.", "Samantha"),
    ]
    for first, second, kept in cases:
        m = MemoryStore(tmp_path / str(abs(hash(first)) % 99999))
        m.remember(first)
        rec = m.remember(second)
        assert len(m) == 1, (first, second)
        assert kept in m.all()[0]["text"]
        assert rec.get("superseded")  # serve returns "updated:" off this


def test_go_by_travel_is_not_a_nickname(tmp_path):
    # "go by bus" is a commute, not an alias; a later name must not delete it
    # (audit 2026-07-22 r4). The ambiguous "goes by" phrasings coexist -- the
    # safe direction -- while "User's name is X" carries the correction.
    m = _two_store(tmp_path, "I go by bus to work.", "User's name is Sam.")
    assert len(m) == 2
    assert len(_two(tmp_path, "User goes by Sam.", "User goes by Sammy.")) == 2


def test_open_ended_go_activities_coexist(tmp_path):
    # "go by" is a nickname (single-valued); "go running"/"go to church" are
    # open-ended activities that must accumulate. A verb-only single-valued
    # set collapsed them (audit 2026-07-22 r3).
    assert len(_two(tmp_path, "User goes running every morning.", "User goes swimming on weekends.")) == 2
    assert len(_two(tmp_path, "User goes to church on Sundays.", "User goes to the gym on Mondays.")) == 2


def test_many_valued_facts_never_delete_each_other(tmp_path):
    # Many-valued relations coexist unconditionally -- no lexical "is this a
    # correction?" guess, which cannot tell "reading books" from "writing
    # books" and deletes when it guesses wrong (audit 2026-07-22 r3). Losing a
    # recorded allergy or fear is the unrecoverable error.
    pairs = [
        ("User is allergic to peanuts.", "User is allergic to shellfish."),
        ("I am allergic to peanuts.", "I am allergic to shellfish."),  # first-person
        ("I am afraid of spiders.", "I am afraid of heights."),
        ("User loves reading books.", "User loves writing books."),
        ("User loves spicy food.", "User loves mild food."),
        ("I have three cats.", "I have two dogs."),
    ]
    for a, b in pairs:
        assert len(_two(tmp_path, a, b)) == 2, (a, b)


def test_value_capitalization_does_not_block_a_correction(tmp_path):
    # A capitalized value must not read as a different KIND than a lowercase
    # one, or a colour correction coexists as a contradiction (audit r3).
    m = _two_store(tmp_path, "User's favorite color is Red.", "User's favorite color is teal.")
    assert len(m) == 1
    assert "teal" in m.all()[0]["text"]


def test_distinct_self_measures_coexist(tmp_path):
    # Age and height are different facts; a single coarse "measure" kind
    # collapsed them (audit 2026-07-22 r3).
    assert len(_two(tmp_path, "User is 30 years old.", "User is 6 feet tall.")) == 2


def test_exact_duplicate_never_deletes_a_neighbor(tmp_path):
    # The dup check must scan the WHOLE store before any supersede: folded
    # into one loop it broke on a key match first, deleted that record, and
    # appended a twin of an existing one (audit 2026-07-22).
    m = MemoryStore(tmp_path)
    m.add("User's dog is young.")
    m.add("User's dog is named Rex.")
    m.remember("User's dog is named Rex.")
    texts = sorted(r["text"] for r in m.all())
    assert texts == ["User's dog is named Rex.", "User's dog is young."]


def test_stemmer_does_not_merge_unrelated_words(tmp_path):
    # care/cared must not answer car queries, notes must not answer "not"
    # (audit 2026-07-22: the old e-stripper injected both into her context).
    m = MemoryStore(tmp_path)
    m.remember("User cared for a sick bird last winter.")
    m.remember("User does not like mushrooms.")
    assert m.search("Where did I park my car?") == []
    assert m.search("Where are my notes?") == []


def test_boss_and_will_are_findable(tmp_path):
    m = MemoryStore(tmp_path)
    m.remember("User's bosses are Jim and Pam.")
    m.remember("User's brother is named Will.")
    assert m.search("Who is my boss?")
    assert m.search("Who is Will?")


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
