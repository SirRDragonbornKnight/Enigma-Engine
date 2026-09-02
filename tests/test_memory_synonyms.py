"""Query-time synonym expansion for memory retrieval.

The measured gap (2026-08-10): "What's my job?" retrieved NOTHING against a
stored "I work as a nurse" -- BM25 shares no term between `job` and `work`, so
a fact she holds is invisible to the question that asks for it.

Everything here is RETRIEVAL. What gets stored, superseded or deleted is ruled
territory and untouched: the last test in this file is the guard that says so.
"""

from __future__ import annotations

import pytest

from enigma_engine.core.memory_store import MemoryStore


@pytest.fixture
def store(tmp_path):
    return MemoryStore(tmp_path)


def _texts(records):
    return [r["text"] for r in records]


def test_the_recorded_job_nurse_case(store):
    """The headline: the 2026-08-10 measurement, verbatim."""
    store.remember("I work as a nurse.")
    assert _texts(store.search("What's my job?", k=3)) == ["I work as a nurse."]


@pytest.mark.parametrize(
    "stored, question",
    [
        ("I work as a nurse.", "What is my occupation?"),
        ("My name is Sam.", "What am I called?"),
        ("I live in Bristol.", "What is my home address?"),
        ("My salary is 41000.", "How much do I get paid?"),
        ("I weigh 71 kilos.", "What is my weight?"),
    ],
)
def test_paraphrased_questions_reach_the_fact(store, stored, question):
    store.remember(stored)
    assert _texts(store.search(question, k=3)) == [stored]


def test_a_literal_match_still_outranks_a_synonym_match(store):
    """Expansion must WIDEN the net, never reorder it.

    The short synonym record is the case that makes the dampening load-bearing:
    BM25 rewards a short document, so at full weight "Work." OUTRANKS a long
    record that uses the asked-for word literally (measured -- and the reason
    this test is not the obvious symmetrical pair, which passes either way).
    """
    store.remember("Work.")                                     # synonym only, short
    store.remember("My job situation has been complicated and quite stressful lately.")
    hits = _texts(store.search("job", k=2))
    assert hits[0].startswith("My job situation"), hits
    assert "Work." in hits, "the synonym match should still be found"


def test_expansion_does_not_invent_matches(store):
    """A question sharing nothing with the store still retrieves nothing --
    widening the query must not turn BM25 into a random-fact generator."""
    store.remember("I work as a nurse.")
    assert store.search("What is the capital of France?", k=3) == []


def test_expansion_is_retrieval_only_and_leaves_the_store_alone(store):
    """`job` and `work` are one cluster for QUESTIONS. They must not merge as
    FACTS: the supersede key is subject + kind, and folding synonyms into it
    would let one fact silently delete another."""
    store.remember("I work as a nurse.")
    store.remember("My job is stressful.")
    kept = _texts(store.search("work job", k=5))
    assert len(kept) == 2, f"a synonym supersede ate a fact: {kept}"
