"""Knowledge corpus (knowledge_corpus.py) + the low-quality QA gate in
make_sft_data: the clean study material must stay clean, and the gate must
drop exactly the junk classes it names."""

from __future__ import annotations

from knowledge_corpus import KNOWLEDGE, gen_knowledge_examples
from make_sft_data import _eval_probe_questions, _is_low_quality, _norm_q


def test_records_are_wellformed_and_clean():
    recs = gen_knowledge_examples()
    assert len(recs) > 400  # enough volume to matter in the mix
    for r in recs:
        user, assistant = r["messages"][0], r["messages"][1]
        assert user["role"] == "user" and user["content"].strip()
        assert assistant["role"] == "assistant" and assistant["content"].strip()
        assert r["category"] == "knowledge"
        # The study material must pass its own quality bar.
        assert not _is_low_quality(r), assistant["content"]


def test_no_eval_probe_leaks():
    eval_qs = _eval_probe_questions()
    assert all(_norm_q(r) not in eval_qs for r in gen_knowledge_examples())


def test_every_intent_has_multiple_surfaces():
    for questions, answers in KNOWLEDGE:
        assert len(questions) >= 2, questions
        assert len(answers) >= 2, questions


def test_low_quality_gate_drops_junk():
    def rec(text):
        return {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": text}]}

    assert _is_low_quality(rec("See https://example.com for details."))
    assert _is_low_quality(rec("<div>hello</div>"))
    assert _is_low_quality(rec("bad bad bad bad bad bad loop"))
    assert not _is_low_quality(rec("The capital of France is Paris."))
    # Profanity is NOT junk -- she may use any type of language (user
    # ruling 2026-07-15); only structural garbage gets dropped.
    assert not _is_low_quality(rec("Well fuck that noise, the build is green."))
