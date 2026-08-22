"""The local memory layer (memory_store.py) — her runtime learning lives here,
not in the frozen weights. Stdlib BM25 over inspectable JSONL."""

import pytest

from enigma_engine.core.memory_store import (
    MAX_MEMORY_CHARS,
    MemoryStore,
    _strip_lead_in,
)
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


def test_distinct_measures_about_one_subject_coexist(tmp_path):
    """Round-2 review 2026-08-13, verified live before the fix: the dog's
    WEIGHT silently deleted the dog's AGE -- any two digit-bearing values
    about one subject shared the coarse {subject, kind:measure} key and
    REPLACED. The ruling licenses 'a measure replaces THAT measure', never
    'any number replaces any other number about the subject'."""
    kept = [r["text"] for r in _two(tmp_path, "User's dog is 3 years old.",
                                    "User's dog is 40 pounds.")]
    assert "User's dog is 3 years old." in kept
    assert "User's dog is 40 pounds." in kept

    # ...while an UPDATE of the same measure still replaces.
    kept = [r["text"] for r in _two(tmp_path, "User's dog is 3 years old.",
                                    "User's dog is 4 years old.")]
    assert kept == ["User's dog is 4 years old."]


def test_a_memory_over_the_length_cap_is_refused(tmp_path):
    """No door capped the text, so a pasted megabyte was filed as one "fact" --
    re-tokenized on every retrieval (search scores the whole store) and echoed
    into a context that cannot hold it. The cap lives on the store because the
    store is the one owner both doors go through."""
    m = MemoryStore(tmp_path)
    at_the_cap = "x" * MAX_MEMORY_CHARS
    assert m.remember(at_the_cap)["text"] == at_the_cap

    with pytest.raises(ValueError, match="memory too long"):
        m.remember("y" * (MAX_MEMORY_CHARS + 1))
    assert [r["text"] for r in m.all()] == [at_the_cap]
    assert [r["text"] for r in MemoryStore(tmp_path).all()] == [at_the_cap]


def test_a_second_store_on_one_directory_is_refused_not_merged(tmp_path):
    """Measured before the guard: two MemoryStores on one dir both minted id 1,
    and the first _rewrite either of them ran erased the other's records. There
    is no merge to do -- the other store's records live in another object -- so
    the mutation that would clobber them refuses instead, and says why."""
    a = MemoryStore(tmp_path)
    b = MemoryStore(tmp_path)

    b.add("User's dog is named Rex.")
    with pytest.raises(MemoryStore.ConcurrentWriter, match="another MemoryStore"):
        a.add("User likes tea.")
    with pytest.raises(MemoryStore.ConcurrentWriter):
        a.clear()  # the rewrite path is the destructive one, and refuses too

    assert a.all() == []  # a refused write leaves nothing live in memory either
    b.add("User likes coffee.")  # ...and the store that OWNS the file still writes
    assert [r["text"] for r in MemoryStore(tmp_path).all()] == [
        "User's dog is named Rex.",
        "User likes coffee.",
    ]


def test_forget_none_deletes_nothing(tmp_path):
    """forget(None) coerced to the literal query "None", whose single term
    subset-matched ANY record containing the word "none" -- verified live:
    it deleted "User's allergies are none." (review 2026-08-13)."""
    m = _two_store(tmp_path, "User likes tea.", "User's allergies are none.")
    assert m.forget(None) == []
    assert len(m.all()) == 2


def test_a_failed_supersede_write_leaves_the_store_exactly_as_it_was(tmp_path, monkeypatch):
    """The supersede path mutated in memory BEFORE persisting -- the rule
    delete()/forget()/clear() already follow and _rewrite's own docstring
    states. Verified live: a failed write reported an error while the live
    session behaved as if the update landed, then reverted at restart."""
    import enigma_engine.core.memory_store as ms

    m = MemoryStore(tmp_path)
    m.remember("User's dog is named Rex.")

    def _boom(*a, **k):
        raise OSError("disk says no")

    monkeypatch.setattr(ms, "atomic_write_text", _boom)
    with pytest.raises(OSError):
        m.remember("User's dog is named Bruno.")
    monkeypatch.undo()

    assert [r["text"] for r in m.all()] == ["User's dog is named Rex."], \
        "memory diverged from disk"
    m2 = MemoryStore(tmp_path)
    assert [r["text"] for r in m2.all()] == ["User's dog is named Rex."]


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


def test_non_positive_k_recalls_nothing(tmp_path, tok):
    # The top-k slice counts from the END on a negative k, so a request for
    # nothing came back holding every record but the lowest-ranked one.
    m = MemoryStore(tmp_path)
    for pet in ("Rex", "Bubbles", "Milo"):
        m.add(f"User's pet is named {pet}.")
    assert len(m.search("pet", k=2)) == 2  # precondition: the query really matches all three
    assert m.search("pet", k=0) == []
    assert m.search("pet", k=-1) == []
    assert m.render_context("pet", tok, k=0) == ""
    assert m.render_context("pet", tok, k=-1) == ""
    # the reserved focus slot must not smuggle a hit past a zero budget either
    assert m.render_context("pet", tok, k=0, focus_query="pet") == ""


def test_the_current_ask_keeps_a_recall_slot(tmp_path, tok):
    # Widening recall to the recent thread let prior-turn chatter fill every
    # slot and evict the answer to the question actually being asked. k=2 is
    # the smallest budget where the eviction is unambiguous; the reserved-slot
    # mechanism is the same at the production k.
    m = MemoryStore(tmp_path)
    for text in ("User's dog Rex is 4 years old.", "User's dog Rex eats twice daily.",
                 "User walks Rex every morning.", "User's fish is called Bubbles."):
        m.add(text)
    ask = "what is my fish called"
    thread = f"is Rex eating twice daily how old is Rex does Rex walk every morning {ask}"
    # precondition: the thread query really does rank the answer out of the slots
    assert "Bubbles" not in [h["text"] for h in m.search(thread, k=2)]
    assert m.search(ask, k=1)[0]["text"] == "User's fish is called Bubbles."

    assert "Bubbles" not in m.render_context(thread, tok, max_ids=512, k=2)
    focused = m.render_context(thread, tok, max_ids=512, k=2, focus_query=ask)
    assert "Bubbles" in focused
    assert focused.splitlines()[1] == "- User's fish is called Bubbles."  # first slot
    assert len(focused.splitlines()) == 3  # header + k lines, never more


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
    m.remember("User's dog is 3 years old.")
    m.remember("User's dog is 4 years old.")
    m2 = MemoryStore(tmp_path)
    assert len(m2) == 1
    assert "4 years old" in m2.all()[0]["text"]


def test_forget_removes_the_matching_fact_and_leaves_the_rest(tmp_path):
    m = MemoryStore(tmp_path)
    m.remember("User likes tea.")
    m.remember("User's dog is named Rex.")
    m.remember("User lives in Denver.")
    removed = m.forget("the user likes tea")
    assert [r["text"] for r in removed] == ["User likes tea."]
    kept = {r["text"] for r in m.all()}
    assert kept == {"User's dog is named Rex.", "User lives in Denver."}
    assert len(MemoryStore(tmp_path)) == 2  # deletion persisted


def test_forget_never_touches_an_unrelated_memory(tmp_path):
    # forget removes what RECALL would surface -- an unrelated fact shares no
    # content term (score 0) and must survive, or "forget my tea" nukes the dog.
    m = MemoryStore(tmp_path)
    m.remember("User's dog is named Rex.")
    assert m.forget("forget that I like tea") == []
    assert len(m) == 1


def test_forget_spares_facts_that_merely_share_a_word(tmp_path):
    # The adversarial case: three facts share the verb "like". Deleting every
    # scoring record took the dogs and the jazz out with the tea.
    m = MemoryStore(tmp_path)
    for text in ("User likes tea.", "User likes dogs.", "User likes jazz."):
        m.remember(text)
    assert [r["text"] for r in m.forget("that I like tea")] == ["User likes tea."]
    assert {r["text"] for r in m.all()} == {"User likes dogs.", "User likes jazz."}


def test_forget_spares_a_sibling_sharing_the_attribute(tmp_path):
    m = MemoryStore(tmp_path)
    m.remember("User's sister is named Amy.")
    m.remember("User's brother is named Leo.")
    assert [r["text"] for r in m.forget("my sister's name")] == ["User's sister is named Amy."]
    assert [r["text"] for r in m.all()] == ["User's brother is named Leo."]


def test_forget_refuses_a_sweep_instead_of_half_doing_it(tmp_path):
    # Two failure modes, one fix. Deleting the top 3 of 5 matches reported
    # "forgot" while the fact class survived; deleting all 5 on a one-word ask
    # is an unrecoverable sweep (no .bak). An ambiguous ask refuses WHOLE and
    # says how many it found, so the user can name one.
    m = MemoryStore(tmp_path)
    for i in range(5):
        m.remember(f"User likes tea blend {i}.")
    with pytest.raises(MemoryStore.TooBroad) as err:
        m.forget("tea")
    assert "5 memories" in str(err.value)
    assert len(m) == 5, "a refused forget must not delete anything"


@pytest.mark.parametrize("ask", [
    "forget that I am tall", "forget I'm tall", "forget that I'm tall now", "tall",
])
def test_a_one_term_fact_stays_reachable_however_it_is_phrased(tmp_path, ask):
    """A revision that demanded an EXACT term-set match made one-term facts
    nearly undeletable: "forget that I am tall" worked while "forget I'm tall"
    did not, because the tokenizer leaves an "m" fragment behind. Identical
    sentences, opposite outcomes, decided by something the user cannot see."""
    m = MemoryStore(tmp_path / str(abs(hash(ask)) % 999983))
    m.remember("User is tall.")
    m.remember("User's dog is named Rex.")
    assert [r["text"] for r in m.forget(ask)] == ["User is tall."]
    assert [r["text"] for r in m.all()] == ["User's dog is named Rex."]


def test_one_term_facts_cannot_be_swept_in_one_call(tmp_path):
    """Five unrelated facts went in a single call when the matcher tried to
    decide which of several matches was meant. It does not decide any more."""
    m = MemoryStore(tmp_path / "sweep")
    for word in ("happy", "tall", "married", "rich", "calm"):
        m.remember(f"User is {word}.")
    with pytest.raises(MemoryStore.TooBroad) as err:
        m.forget("the happy tall married rich calm neighbor story")
    assert "5 memories" in str(err.value)
    assert "User is happy." in str(err.value), "the refusal must NAME the candidates"
    assert len(m) == 5


def test_an_ask_about_someone_else_never_takes_the_users_facts(tmp_path):
    """The sharpest repro of the old rule: an ask about the SISTER and the
    BROTHER deleted two facts about the user, silently, because each stored
    record's terms happened to appear somewhere in the sentence."""
    m = MemoryStore(tmp_path / "others")
    for text in ("User likes tea.", "User hates jazz.", "User plays guitar."):
        m.remember(text)
    with pytest.raises(MemoryStore.TooBroad):
        m.forget("forget that my sister likes tea and my brother hates jazz")
    assert len(m) == 3


def test_two_records_sharing_a_term_set_are_never_both_deleted(tmp_path):
    """Two different facts can normalize to the same content terms. An exact
    naming used to delete BOTH -- an exact match is not a licence."""
    m = MemoryStore(tmp_path / "twins")
    m.add("User's sister is taller than user's brother.")
    m.add("User's brother is taller than user's sister.")
    with pytest.raises(MemoryStore.TooBroad):
        m.forget("forget that my sister is taller than my brother")
    assert len(m) == 2


def test_the_forget_cap_refuses_a_large_restatement_batch(tmp_path):
    """The cap branch had no coverage: the sweep test raised via the AMBIGUOUS
    pointer rule, so `_FORGET_MAX = 500` passed the whole suite. This pins the
    cap itself -- many multi-term facts all restated by one long ask."""
    m = MemoryStore(tmp_path / "cap")
    colours = ("red", "blue", "green", "black", "white", "amber")
    for colour in colours:
        m.remember(f"User owns a {colour} mug.")
    ask = "forget that I own a " + " ".join(f"{c} mug" for c in colours)
    with pytest.raises(MemoryStore.TooBroad) as err:
        m.forget(ask)
    assert f"{len(colours)} memories" in str(err.value)
    assert len(m) == len(colours), "a refused forget must not delete anything"


def test_forget_refuses_an_ambiguous_pointer(tmp_path):
    # "my name" reaches the user's name, the dog's name and the sister's. The
    # ask points at ONE fact and matched three, so it names none of them.
    m = MemoryStore(tmp_path)
    for text in ("User's name is Sam.", "User's dog is named Rex.",
                 "User's sister is named Amy."):
        m.remember(text)
    with pytest.raises(MemoryStore.TooBroad):
        m.forget("my name")
    assert len(m) == 3


def test_forget_on_a_contentless_query_deletes_nothing(tmp_path):
    # "forget everything" is an unbounded wipe; that belongs to clear(), behind
    # its own explicit ask, not to a fuzzy match.
    m = MemoryStore(tmp_path)
    m.remember("User likes tea.")
    assert m.forget("forget everything") == []
    assert len(m) == 1


def test_a_wipe_ask_does_not_become_a_needle_via_filler(tmp_path):
    # "forget everything now" is the same unbounded wipe. "now" is not a
    # subject, but it survived the noise filter as the ask's only term -- and
    # then deleted the one record containing that word, which is a record
    # supersede itself writes ("...is named Bruno now.").
    m = MemoryStore(tmp_path)
    m.remember("User's dog is named Rex.")
    m.remember("Actually, my dog is named Bruno now.")
    assert m.forget("forget everything now") == []
    assert m.forget("forget everything, thanks") == []
    assert len(m.all()) == 1  # the supersede collapsed the pair, the wipe took nothing


def test_naming_a_nested_fact_word_for_word_resolves_the_tie(tmp_path):
    # Nested facts made every ask that covers one cover the other, so BOTH
    # spellings refused and repeating either named candidate refused again --
    # advice that could not be followed. An ask whose terms EQUAL one record's
    # is that record being named, so it wins.
    m = MemoryStore(tmp_path)
    m.remember("User likes tea.")
    m.remember("User likes green tea.")
    assert [r["text"] for r in m.forget("forget that I like green tea")] == [
        "User likes green tea."
    ]
    assert [r["text"] for r in m.all()] == ["User likes tea."]
    assert [r["text"] for r in m.forget("forget that I like tea")] == ["User likes tea."]


def test_forget_coerces_a_non_string_query_instead_of_crashing(tmp_path):
    # The emptiness guard called str(query) but handed the RAW object to the
    # term split, so None and ints raised AttributeError out of a delete path.
    m = MemoryStore(tmp_path)
    m.remember("User likes tea.")
    assert m.forget(None) == []
    assert m.forget(123) == []
    assert len(m) == 1


def test_a_failed_write_leaves_the_store_exactly_as_it_was(tmp_path, monkeypatch):
    """The store mutated `_records` and then rewrote the file, so a failed write
    reported an error to the user while the live store had ALREADY dropped the
    record -- and the next successful write made that loss permanent."""
    from enigma_engine.core import memory_store as ms

    m = MemoryStore(tmp_path)
    m.remember("User's dog is named Rex.")
    m.remember("User's car is red.")
    monkeypatch.setattr(ms, "atomic_write_text",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))

    with pytest.raises(OSError):
        m.forget("forget that my dog is named Rex")
    assert any("Rex" in r["text"] for r in m.all()), "the record vanished despite the error"

    monkeypatch.undo()
    m.forget("forget that my car is red")
    assert any("Rex" in r["text"] for r in MemoryStore(tmp_path).all()), \
        "the next successful write made the phantom delete permanent"


def test_the_same_filter_normalizes_both_the_ask_and_the_record(tmp_path):
    """Filtering only the ask meant a record holding a noise word could never
    equal the ask's terms, so quoting that record word for word deleted its
    shorter sibling instead of itself -- and reported near-identical text, which
    reads as success."""
    m = MemoryStore(tmp_path)
    m.add("User's phone is still broken.")
    m.add("User's phone is broken.")
    with pytest.raises(MemoryStore.TooBroad):
        m.forget("User's phone is still broken.")
    assert len(m.all()) == 2


def test_a_word_that_tells_two_records_apart_is_never_filtered(tmp_path):
    """Stripping "right" sent "forget that my right knee hurts" to the LEFT
    knee; stripping "memory" sent "forget that my memory is bad" to the mood.
    Leaving a filler word in costs a match that deletes nothing; taking a
    content word out costs the wrong delete."""
    m = MemoryStore(tmp_path)
    m.remember("User's left knee hurts.")
    assert m.forget("forget that my right knee hurts") == []
    m2 = MemoryStore(tmp_path / "b")
    m2.remember("User's mood is bad.")
    assert m2.forget("forget that my memory is bad") == []


def test_the_refusal_is_bounded(tmp_path):
    """The refusal is fed back into a 1024-token context as a tool result; 21
    matches measured 728 tokens of it."""
    m = MemoryStore(tmp_path)
    for i in range(21):
        m.add(f"User's fact {i} is about tea.")
    with pytest.raises(MemoryStore.TooBroad) as err:
        m.forget("forget about tea")
    assert len(str(err.value)) < 400
    assert "more)" in str(err.value)
    assert len(m.all()) == 21


def test_an_ambiguous_refusal_names_ids(tmp_path):
    # Records with identical term sets cannot be separated by restating them,
    # so a refusal that named only text was advice with no way to follow it.
    m = MemoryStore(tmp_path)
    m.add("User likes tea.")
    m.add("User likes tea.")
    with pytest.raises(MemoryStore.TooBroad) as err:
        m.forget("forget that I like tea")
    assert "#1" in str(err.value) and "#2" in str(err.value)
    assert m.delete(2) is True
    assert [r["id"] for r in m.all()] == [1]


def test_an_ask_covering_both_nested_facts_still_refuses(tmp_path):
    # The exact-name rule must not become a licence to delete on a superset ask
    # that names neither record.
    m = MemoryStore(tmp_path)
    m.remember("User likes tea.")
    m.remember("User likes green tea.")
    with pytest.raises(MemoryStore.TooBroad) as err:
        m.forget("forget that I like green tea and coffee")
    assert "word for word" in str(err.value)
    assert len(m.all()) == 2


def test_a_correction_with_a_lead_in_supersedes_the_stale_value(tmp_path):
    # "Actually, ..." defeated both ^-anchored fact parses, so the correction
    # coexisted and BM25 ranked the stale value first.
    m = MemoryStore(tmp_path)
    m.remember("User's dog is named Rex.")
    rec = m.remember("Actually, my dog is named Bruno now.")
    assert rec["superseded"] == "User's dog is named Rex."
    assert [r["text"] for r in m.all()] == ["Actually, my dog is named Bruno now."]
    assert m.search("what is my dog's name", k=3)[0]["text"].endswith("Bruno now.")


def test_forget_reaches_a_fact_the_ask_only_points_at(tmp_path):
    # "forget about my dog" carries fewer terms than the record it names, so
    # the match runs the other way -- but only once the asking words are out of
    # the query, or "forget" itself blocks every subset.
    m = MemoryStore(tmp_path)
    m.remember("User's dog is named Rex.")
    m.remember("User's car is a silver hatchback.")
    assert [r["text"] for r in m.forget("please forget everything about my dog")] == [
        "User's dog is named Rex."
    ]
    assert [r["text"] for r in m.all()] == ["User's car is a silver hatchback."]


def test_stripping_a_lead_in_never_empties_the_text():
    # A text that is nothing BUT a marker keeps its words rather than becoming
    # an empty, keyless string.
    assert _strip_lead_in("No,") == "No,"
    assert _strip_lead_in("Actually, my dog is Bruno.") == "my dog is Bruno."


@pytest.mark.parametrize("stored, hedged", [
    ("User drives a Toyota.", "Sorry, I drive a rental this week."),
    ("User lives in Denver.", "Oh, I live in a hotel right now."),
    ("User works as a nurse.", "Well, I work as a volunteer at weekends."),
])
def test_a_hedge_is_not_a_correction(tmp_path, stored, hedged):
    """Stripping general hedges as if they were correction markers fed ordinary
    additions to the DESTRUCTIVE path: these three relations are single-valued,
    so "Sorry, I drive a rental this week" DELETED the Toyota. Before the strip
    they keyed None and coexisted; only markers that announce a replacement
    belong in the list."""
    m = MemoryStore(tmp_path / str(abs(hash(stored)) % 999983))
    m.remember(stored)
    m.remember(hedged)
    assert len(m) == 2, "a hedged addition deleted the fact it hedged"


def test_forget_empty_query_is_a_noop(tmp_path):
    m = MemoryStore(tmp_path)
    m.remember("User lives in Denver.")
    assert m.forget("   ") == []
    assert len(m) == 1


def test_tied_retrieval_ranks_the_newest_value_first(tmp_path):
    # Coexisting plain values share a term vector, so BM25 ties. The newer
    # fact must surface first or the coexist default hands the model the
    # STALE value as the top hit -- "outranked, not erased" has to be true.
    m = MemoryStore(tmp_path)
    m.remember("My mood is happy.")
    m.remember("My mood is sad.")
    hits = m.search("what is my mood", k=2)
    assert len(hits) == 2
    assert "sad" in hits[0]["text"], "the stale value outranked the correction"


def test_word_number_corrections_stack_by_design(tmp_path):
    # The documented asymmetry of the coexist ruling: digits classify as a
    # measure and REPLACE ("age is 30" -> "31"), spelled-out numbers classify
    # as plain values and COEXIST ("age is thirty" -> "thirty-one"). Pinned so
    # the surface-form dependence is a stated limit, not a surprise.
    m = _two_store(tmp_path, "My age is thirty.", "My age is thirty-one.")
    assert len(m) == 2


def test_plain_values_about_one_subject_coexist(tmp_path):
    # Same attribute ("car"), two plain values: TWO facts. The shared coarse
    # kind is not proof of a correction -- "a red hatchback" and "a silver
    # van" could as easily be a repaint as a second car, and a wrong
    # supersede destroys a fact while a kept duplicate is merely outranked.
    m = MemoryStore(tmp_path)
    m.remember("User's car is a red hatchback.")
    m.remember("User's car is a silver van.")
    texts = {r["text"] for r in m.all()}
    assert texts == {"User's car is a red hatchback.", "User's car is a silver van."}


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


def test_value_capitalization_does_not_change_the_kind(tmp_path):
    # A capitalized value must classify to the same KIND as a lowercase one --
    # "Red" keying as a name while "teal" keys as other would put the pair on
    # different keys for a capitalization accident. Both are plain values, so
    # under the coexist default the pair lands on ONE key and both are kept.
    from enigma_engine.core.memory_store import _fact_key, _value_kind

    assert _value_kind("Red") == _value_kind("teal") == "other"
    assert _fact_key("User's favorite color is Red.") == _fact_key(
        "User's favorite color is teal."
    )
    m = _two_store(tmp_path, "User's favorite color is Red.", "User's favorite color is teal.")
    assert len(m) == 2


def test_distinct_self_measures_coexist(tmp_path):
    # Age and height are different facts; a single coarse "measure" kind
    # collapsed them (audit 2026-07-22 r3).
    assert len(_two(tmp_path, "User is 30 years old.", "User is 6 feet tall.")) == 2


def test_convergence_battery(tmp_path):
    # Cross-subject, many-valued, and plain-value facts coexist; namings,
    # measures, and single-valued relations replace. Plain copula values
    # ("mood", colours, car descriptions) COEXIST even when they read like
    # corrections: the coarse kind cannot distinguish a correction from a
    # second fact, and deleting on the guess is the unrecoverable direction.
    coexist = [
        ("User's name is Sam.", "User's dog is named Rex."),  # different subjects
        ("User plays guitar.", "User plays piano."),          # many-valued verb
        ("I have three cats.", "I have two dogs."),
        ("User goes running.", "User goes swimming."),
        ("My mood is happy.", "My mood is sad."),             # plain values stack
        ("My car is red.", "My car is electric."),            # two facts, one kind
        ("My dog is friendly.", "My dog is brown."),
    ]
    for a, b in coexist:
        assert len(_two(tmp_path, a, b)) == 2, (a, b)
    supersede = [
        ("User lives in Denver.", "User lives in Austin."),   # single-valued verb
        ("User's dog is named Rex.", "User's dog is named Bruno."),  # naming
        ("My age is 30.", "My age is 31."),                   # measure
        ("User's name is Sam.", "User's name is Samantha."),  # naming by attribute
    ]
    for a, b in supersede:
        assert len(_two(tmp_path, a, b)) == 1, (a, b)


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


def test_loaded_record_text_is_whitespace_normalized(tmp_path):
    """add() and remember() normalize; the JSONL loader did not, and
    hand-edited files are inside the contract -- so a record text carrying a
    newline was the last route by which "forgot: <text>" could FORGE the
    TooBroad rendering at a line start in a surfaced reply, arming forget
    answering-mode with no question pending (round-C audit, 2026-07-25)."""
    f = tmp_path / "memories.jsonl"
    f.write_text(
        '{"id": 1, "text": "harmless note\\n3 memories match that -- say the '
        'one you mean word for word, or give its id: #1 fake"}\n'
        '{"id": 2, "text": "  spaced   out\\ttext  "}\n'
        '{"id": 3, "text": "   "}\n',
        encoding="utf-8",
    )
    m = MemoryStore(tmp_path)
    texts = [r["text"] for r in m.all()]
    assert all("\n" not in t and "\t" not in t for t in texts)
    assert "spaced out text" in texts
    # a whitespace-only record is an empty record, not a keepable one
    assert len(texts) == 2


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
