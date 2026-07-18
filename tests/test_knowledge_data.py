"""Knowledge corpus (knowledge_corpus.py) + the low-quality QA gate in
make_sft_data: the clean study material must stay clean, and the gate must
drop exactly the junk classes it names."""

from __future__ import annotations

from knowledge_corpus import KNOWLEDGE, gen_knowledge_examples, gen_knowledge_pretrain_text
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
    # An absent probes file returns an empty set, and `x not in {}` is
    # vacuously true for everything -- guard like the pretrain twin below
    # (test-suite audit 2026-07-17).
    assert eval_qs, "behavior_probes.jsonl missing -- this test proves nothing"
    assert all(_norm_q(r) not in eval_qs for r in gen_knowledge_examples())


def test_every_intent_has_multiple_surfaces():
    for questions, answers in KNOWLEDGE:
        assert len(questions) >= 2, questions
        assert len(answers) >= 2, questions


def test_pretrain_text_is_deterministic():
    # Byte-identical across calls -- the corpus is versioned training data.
    assert gen_knowledge_pretrain_text() == gen_knowledge_pretrain_text()
    assert gen_knowledge_pretrain_text(seed=123) == gen_knowledge_pretrain_text(seed=123)


def test_pretrain_text_dodges_eval_probes():
    probes = _eval_probe_questions()
    assert probes  # the probes file must exist, or this test proves nothing
    for line in gen_knowledge_pretrain_text():
        low = line.lower()
        assert low not in probes, line
        # The generator dodges harder than equality: no probe string may
        # ride INSIDE a line either (QA lines embed question text).
        assert all(p not in low for p in probes), line


def test_pretrain_text_lines_are_clean():
    lines = gen_knowledge_pretrain_text()
    assert lines
    for line in lines:
        assert line and line == line.strip()
        assert "\n" not in line and "\r" not in line
        assert line.isascii(), line


def test_pretrain_text_count_bounds():
    n = len(gen_knowledge_pretrain_text())
    assert 600 <= n <= 1100, n
    assert n >= 4 * len(KNOWLEDGE), n  # several textual forms per fact


def test_pretrain_text_covers_jupiter_in_multiple_forms():
    # The audit's brittleness case ("largest planet" -> Jupiter but "biggest
    # planet" -> Saturn): the fact must surface as a QA line, a key-term-final
    # cloze, and plain prose -- at least three distinct textual forms.
    lines = [line for line in gen_knowledge_pretrain_text() if "jupiter" in line.lower()]
    qa = [line for line in lines if line.startswith("Q: ")]
    cloze = [line for line in lines if line.endswith("is Jupiter.")]
    prose = [line for line in lines if line not in qa and line not in cloze]
    assert qa and cloze and prose, lines


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
