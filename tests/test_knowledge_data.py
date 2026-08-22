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
    """The codepoint pin catches ENCODING DAMAGE, not typography: ASCII plus
    the one non-ASCII character the training-content convention spells the
    prose dash with, U+2014 (the em-dash unification 2026-08-19 -- Kokoro's
    G2P deletes " -- " from the phoneme stream, so the ASCII form went
    unspoken). EVERY other non-ASCII codepoint still fails, so mojibake stays
    caught: an em-dash mangled by a cp1252/UTF-8 round trip arrives as the
    three codepoints U+00E2 U+20AC U+201D, and all three are still refused.
    C0 controls and DEL fail on the same rule: the line iteration already
    stripped the newline, and U+001E is DOC_SEP_CHAR, the packed-file record
    separator `pretokenize_data.py` splits on -- one riding inside a fact line
    forges a document boundary and cuts the record in half, silently."""
    lines = gen_knowledge_pretrain_text()
    assert lines
    for line in lines:
        assert line and line == line.strip()
        assert "\n" not in line and "\r" not in line
        bad = sorted({f"U+{ord(c):04X}" for c in line
                      if ord(c) < 32 or ord(c) == 127 or (ord(c) > 127 and c != "—")})
        assert not bad, f"{','.join(bad)}: {line}"


def test_pretrain_text_count_bounds():
    # Corridor re-based 2026-08-08 for the ruled floor widening (138 -> 240
    # intents incl. the Enigma self/capability section). Measured 1570 lines
    # on 2026-08-14 after the audit's cloze guards (1574 on 2026-08-13; the
    # earlier "1573" receipt never reproduced against any tree -- a
    # phantom). The corridor is a sanity band -- several forms per fact, no
    # runaway generator -- not a pin on the exact count.
    n = len(gen_knowledge_pretrain_text())
    assert 1200 <= n <= 2400, n
    assert n >= 4 * len(KNOWLEDGE), n  # several textual forms per fact


def test_answer_variants_are_distinct():
    """Five intents shipped twin answers ("Nairobi." / "Nairobi."), which the
    picks-then-dedup path silently collapsed to HALF the intended records
    (review 2026-08-13). The module rule is the pin: answers vary in
    wording, never in fact."""
    for questions, answers in KNOWLEDGE:
        assert len(set(answers)) == len(answers), questions[0]


def test_cloze_lines_never_quote_auxiliary_questions():
    """'The answer to "Was Enigma built from another model?" is None.' -- an
    ambiguous-polarity SELF fact (review 2026-08-13). The full auxiliary
    family skips the cloze form."""
    import re

    for line in gen_knowledge_pretrain_text():
        if line.startswith("The answer to"):
            assert not re.search(
                r'"(Is|Are|Do|Does|Can|Why|Was|Were|Did|Will|Has|Have|Should|Could|Would|How do|How does|Who can)\b',
                line), line


def test_context_lines_carry_a_real_payload():
    """52 authoritative leads carried a sub-4-word payload ("As any
    reference book will confirm: 1945.") -- the confident-non-answer shape
    the restraint work suppresses (review 2026-08-13)."""
    from knowledge_corpus import _CONTEXT_LEADS

    for line in gen_knowledge_pretrain_text():
        for lead in _CONTEXT_LEADS:
            if line.startswith(lead):
                assert len(line[len(lead):].split()) >= 4, line


def test_number_words_lowercase_mid_sentence_by_rule():
    """The hand allowlist drifted behind every widening ("is Three." vs
    "is seven." in one corpus, 33 lines); number-words now match by RULE
    (review 2026-08-13)."""
    import re

    from knowledge_corpus import _NUMBER_WORD, _mid_case

    assert _mid_case("Three") == "three"
    assert _mid_case("Fifty-two") == "fifty-two"
    assert _mid_case("Twenty-four hours") == "twenty-four hours"
    assert _mid_case("Jupiter") == "Jupiter"  # proper nouns keep their capital
    # a number-word leading a TITLE is a name, not a quantity (audit
    # 2026-08-14: the bare regex pre-committed 'one Direction'); a
    # mixed-case tail is a quantity with a unit's proper noun inside
    assert _mid_case("Ten Commandments") == "Ten Commandments"
    assert _mid_case("One Direction") == "One Direction"
    assert _mid_case("Zero degrees Celsius") == "zero degrees Celsius"
    for line in gen_knowledge_pretrain_text():
        m = re.search(r" is ([A-Z][a-z]+(?:-[a-z]+)?)[ .]", line)
        if m:
            assert not _NUMBER_WORD.match(m.group(1).lower()), line


def test_the_probe_screen_fails_closed(monkeypatch, tmp_path):
    """A missing probes file used to silently DISABLE the screen (changing
    the facts stream's bytes with no error); an empty q/teach string would
    have matched every line and zeroed the stream (review 2026-08-13)."""
    import json

    import pytest

    import knowledge_corpus as kc

    monkeypatch.setattr(kc, "_EVAL_PROBES", tmp_path / "absent.jsonl")
    with pytest.raises(SystemExit, match="probe screen"):
        kc.gen_knowledge_pretrain_text()

    poisoned = tmp_path / "probes.jsonl"
    poisoned.write_text(json.dumps({"category": "unknown", "q": "  ",
                                    "teach": ["", "a real teach line"]}) + "\n",
                        encoding="utf-8")
    monkeypatch.setattr(kc, "_EVAL_PROBES", poisoned)
    lines = kc.gen_knowledge_pretrain_text()
    assert len(lines) > 1000, "an empty probe string zeroed the facts stream"


def test_clozes_are_never_tautologies_or_broken_copulas():
    """The skip-list widening SHIFTED clozes onto later questions and
    created 'The answer to "What is photosynthesis?" is photosynthesis.'
    plus '... is Through its memory store.' -- a self-answer and a broken
    copula (audit 2026-08-14). Guarded by RULE; which-questions keep their
    option-picks ("Which is longer, a mile or a kilometer?" is a mile.)."""
    lines = gen_knowledge_pretrain_text()
    clozes = [l for l in lines if l.startswith('The answer to "')]
    assert clozes
    for line in clozes:
        q = line.split('"')[1]
        ans = line.rsplit('" is ', 1)[1].rstrip(".")
        aw = {w.strip('.,?!\'"').lower() for w in ans.split()}
        qw = {w.strip('.,?!\'"').lower() for w in q.split()}
        if not q.lower().startswith("which"):
            assert not aw <= qw, line
        assert ans.split()[0].lower() not in (
            "through", "with", "by", "via", "using", "because", "from"), line
    # ...and the sealed boil row's DIGIT surface is the cloze surface (the
    # first fix re-created the word-number-only line one step downstream)
    assert any("is 100 degrees Celsius." in l for l in clozes), \
        "the boil cloze lost its digit surface again"
    # ...and the self row's parameter count keeps its word form: the extractor
    # reads the term off the leading answer fragment, so a re-worded first
    # answer moves this line without moving any count.
    assert any(l.endswith("is about 240 million parameters.") for l in clozes), \
        "the parameter-count cloze lost its word form"


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

    assert _is_low_quality(rec("<div>hello</div>"))
    assert _is_low_quality(rec("bad bad bad bad bad bad loop"))
    assert not _is_low_quality(rec("The capital of France is Paris."))
    # A cited link is supervision, not junk: dropping URL-bearing answers
    # removed all URL supervision, so she could never cite a source.
    assert not _is_low_quality(rec("See https://example.com for details."))
    # Profanity is NOT junk -- she may use any type of language (user
    # ruling 2026-07-15); only structural garbage gets dropped.
    assert not _is_low_quality(rec("Well fuck that noise, the build is green."))


def test_a_commented_dev_probe_file_reads_the_same_through_the_facts_screen(
        tmp_path, monkeypatch):
    """The THIRD reader of behavior_probes.jsonl. eval_behavior.run and
    make_sft_data._eval_probe_questions both skip `#` comment lines -- the dev
    file invites annotation -- while knowledge_corpus._probe_strings json.loads'd
    every non-blank line, so the same comment killed the screen that keeps the
    facts stream off the held-out probes, with a raw JSONDecodeError.

    Comments must contribute NOTHING and cost NOTHING: the strings this reader
    keeps are the strings make_sft_data's reader keeps."""
    import json

    import knowledge_corpus
    import make_sft_data

    rows = [
        {"category": "identity", "q": "Which observatory catalogued Vantill-9?",
         "want_any": ["never"]},
        {"category": "memory", "q": "What did I name my telescope?",
         "teach": ["I named my telescope Vantill."], "want_any": ["vantill"]},
    ]
    probes = tmp_path / "behavior_probes.jsonl"
    probes.write_text(
        "# retired probe: superseded by the locked set\n"
        + "\n".join(json.dumps(r) for r in rows) + "\n"
        + "   # an indented comment is still a comment\n"
        + "\n",
        encoding="utf-8")
    monkeypatch.setattr(knowledge_corpus, "_EVAL_PROBES", probes)
    monkeypatch.setattr(make_sft_data, "EVAL_PROBES", probes)

    expected = {r["q"].strip().lower() for r in rows}
    expected |= {f.strip().lower() for r in rows for f in r.get("teach", [])}
    assert set(knowledge_corpus._probe_strings()) == expected
    assert set(knowledge_corpus._probe_strings()) == _eval_probe_questions()
