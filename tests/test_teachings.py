"""gen_teaching_examples must render user-authored facts into training
records, tolerate comments/blank lines, skip malformed lines loudly, and
stay deterministic (mix.jsonl must be reproducible run to run)."""

import json

from make_sft_data import gen_teaching_examples


def _write(tmp_path, lines):
    p = tmp_path / "teachings.jsonl"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def test_missing_file_is_empty(tmp_path):
    assert gen_teaching_examples(tmp_path / "absent.jsonl") == []


def test_comments_and_blanks_skipped(tmp_path):
    p = _write(tmp_path, ["# a comment", "", "   ", "# {\"q\": \"commented out\", \"a\": \"x\"}"])
    assert gen_teaching_examples(p) == []


def test_simple_pair_shape(tmp_path):
    p = _write(tmp_path, [json.dumps({"q": "What's my cat's name?", "a": "Mochi."})])
    out = gen_teaching_examples(p)
    assert len(out) == 1
    assert out[0]["category"] == "teaching"
    assert out[0]["messages"][0] == {"role": "user", "content": "What's my cat's name?"}
    assert out[0]["messages"][1] == {"role": "assistant", "content": "Mochi."}


def test_questions_cross_rotating_answers(tmp_path):
    rec = {
        "questions": ["Q1?", "Q2?", "Q3?"],
        "answers": ["A1.", "A2."],
    }
    p = _write(tmp_path, [json.dumps(rec)])
    out = gen_teaching_examples(p)
    # Every question appears, each paired with 2 distinct answers.
    qs = [r["messages"][0]["content"] for r in out]
    assert set(qs) == {"Q1?", "Q2?", "Q3?"}
    for q in ("Q1?", "Q2?", "Q3?"):
        answers = {r["messages"][1]["content"] for r in out if r["messages"][0]["content"] == q}
        assert answers == {"A1.", "A2."}


def test_single_answer_not_duplicated(tmp_path):
    p = _write(tmp_path, [json.dumps({"questions": ["Q1?", "Q2?"], "answers": ["Only."]})])
    out = gen_teaching_examples(p)
    assert len(out) == 2  # one record per question, no duplicate answer pairs


def test_malformed_lines_skipped(tmp_path, capsys):
    p = _write(
        tmp_path,
        [
            "not json at all",
            json.dumps({"questions": [], "answers": ["orphan"]}),
            json.dumps({"q": "Good?", "a": "Yes."}),
        ],
    )
    out = gen_teaching_examples(p)
    assert len(out) == 1  # only the good record survives
    printed = capsys.readouterr().out
    assert "SKIPPED" in printed  # loud, not silent


def test_deterministic(tmp_path):
    lines = [
        json.dumps({"questions": ["Qa?", "Qb?", "Qc?"], "answers": ["A1.", "A2.", "A3."]}),
        json.dumps({"q": "Solo?", "a": "Yes."}),
    ]
    p = _write(tmp_path, lines)
    assert gen_teaching_examples(p) == gen_teaching_examples(p)
