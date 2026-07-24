"""Eval transcript (eval_behavior.run --transcript): a scorecard with no saved
answers cannot be re-graded, handed to a second grader, or defended after the
server is gone. The locked baseline is the run this exists for, so the file has
to carry the FULL answer plus the conditions that produced it -- probe-file
digest, decode config, and the tree the code came from.

It also has to refuse to write itself somewhere versioned: a transcript holds
every probe question and answer verbatim, so a locked-set transcript inside
data/eval is one `git add` from unsealing the gate permanently."""

from __future__ import annotations

import json

import pytest

import eval_behavior

# Longer than the 60-char console line: a transcript that stored `detail`
# instead of the answer would silently truncate exactly the evidence a
# re-grade needs.
LONG_ANSWER = (
    "Jupiter is the largest planet in the solar system, and it is large enough "
    "that every other planet would fit inside it with room to spare."
)
# Deliberately NOT the argparse defaults (0.0 / 60 / 127.0.0.1:8123): a
# hardcoded field would match the defaults and the assertion would prove
# nothing.
TEMP, MAXTOK, URL = 0.7, 13, "http://probe-host:9999"


def _probe_file(tmp_path, name="probes.jsonl", q="Largest planet?"):
    p = tmp_path / name
    p.write_text(
        json.dumps({"category": "factual", "q": q, "want_any": ["jupiter"], "deny_any": []})
        + "\n"
        + json.dumps({"category": "tool", "q": "Weather in Denver?", "expect_tool": "get_weather"})
        + "\n",
        encoding="utf-8",
    )
    return p


def _fake_server(monkeypatch, answer: str, tool_on_text: str | None = None):
    monkeypatch.setattr(eval_behavior, "_wait_for_server", lambda *a, **k: True)
    monkeypatch.setattr(eval_behavior, "_clear_memory", lambda *a, **k: None)

    def fake_post(base_url, payload):
        if payload.get("tools"):
            return {"choices": [{"message": {
                "content": "",
                "tool_calls": [{"function": {"name": "get_weather"}}],
            }}]}
        msg = {"content": answer}
        if tool_on_text:  # a non-tool probe that fires a tool anyway
            msg["tool_calls"] = [{"function": {"name": tool_on_text}}]
        return {"choices": [{"message": msg}]}

    monkeypatch.setattr(eval_behavior, "_post", fake_post)


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_transcript_carries_full_answers_and_run_conditions(tmp_path, monkeypatch):
    probes = _probe_file(tmp_path)
    out = tmp_path / "transcript.jsonl"
    _fake_server(monkeypatch, LONG_ANSWER)

    eval_behavior.run(URL, TEMP, MAXTOK, probes, out)

    rows = _rows(out)
    conditions, scorecard = rows[0], rows[-1]
    probe_rows = [r for r in rows if r["record"] == "probe"]

    assert conditions["record"] == "run_conditions"
    assert conditions["probe_count"] == 2
    # Each of these would be wrong if the field were hardcoded to its default.
    assert conditions["temperature"] == TEMP
    assert conditions["max_tokens"] == MAXTOK
    assert conditions["base_url"] == URL
    assert "git_dirty" in conditions

    assert len(probe_rows) == 2
    factual = next(r for r in probe_rows if r["category"] == "factual")
    assert factual["content"] == LONG_ANSWER, "the answer was truncated or dropped"
    assert factual["graded_ok"] is True

    tool = next(r for r in probe_rows if r["category"] == "tool")
    assert tool["tool_called"] == "get_weather"

    assert scorecard["record"] == "scorecard"
    assert scorecard["overall_n"] == 2


def test_digest_identifies_the_probe_set_and_ignores_line_endings(tmp_path):
    """The digest's whole job is proving two runs scored the SAME set.

    Asserting its LENGTH proved nothing -- every sha256 is 64 chars. Different
    content must differ; the same content with CRLF must NOT, or a re-measure
    on a fresh clone looks like a different probe set.

    The two variants are written as BYTES: `write_text` applies the platform's
    newline translation, so on Windows it already emits CRLF and a naive
    "add CRLF" fixture builds \\r\\r\\n and tests nothing real.
    """
    rec_a = json.dumps({"category": "factual", "q": "Largest planet?", "want_any": ["jupiter"]})
    rec_b = json.dumps({"category": "factual", "q": "Deepest ocean?", "want_any": ["pacific"]})
    lf_a, lf_b, crlf_a = (tmp_path / n for n in ("a.jsonl", "b.jsonl", "a_crlf.jsonl"))
    lf_a.write_bytes((rec_a + "\n").encode("utf-8"))
    lf_b.write_bytes((rec_b + "\n").encode("utf-8"))
    crlf_a.write_bytes((rec_a + "\r\n").encode("utf-8"))

    da, db, dcrlf = (eval_behavior._probe_digest(p) for p in (lf_a, lf_b, crlf_a))
    assert da != db, "different probe sets must not share a digest"
    assert da == dcrlf, "line-ending normalization must survive a checkout"


def test_a_failing_answer_is_recorded_as_failing(tmp_path, monkeypatch):
    """The adversarial half: a transcript that marked everything ok would look
    identical on a passing run and hide every regression on a failing one."""
    probes = _probe_file(tmp_path)
    out = tmp_path / "transcript.jsonl"
    _fake_server(monkeypatch, "I have no idea which planet that would be.")

    eval_behavior.run(URL, TEMP, MAXTOK, probes, out)

    factual = next(r for r in _rows(out) if r.get("category") == "factual")
    assert factual["graded_ok"] is False
    assert "no idea" in factual["content"]


def test_a_tool_fired_on_a_text_probe_is_visible(tmp_path, monkeypatch):
    """False-fires are the router defect under review; a transcript that read
    tool_calls only for tool/restraint probes could never show one."""
    probes = _probe_file(tmp_path)
    out = tmp_path / "transcript.jsonl"
    _fake_server(monkeypatch, LONG_ANSWER, tool_on_text="imagine")

    eval_behavior.run(URL, TEMP, MAXTOK, probes, out)

    factual = next(r for r in _rows(out) if r.get("category") == "factual")
    assert factual["tool_called"] == "imagine"


def test_an_aborted_run_still_saves_what_it_collected(tmp_path, monkeypatch):
    probes = _probe_file(tmp_path)
    out = tmp_path / "transcript.jsonl"
    _fake_server(monkeypatch, LONG_ANSWER)
    calls = {"n": 0}
    real_post = eval_behavior._post

    def dying_post(base_url, payload):
        calls["n"] += 1
        if calls["n"] > 1:
            raise ConnectionError("server died mid-suite")
        return real_post(base_url, payload)

    monkeypatch.setattr(eval_behavior, "_post", dying_post)

    with pytest.raises(ConnectionError):
        eval_behavior.run(URL, TEMP, MAXTOK, probes, out)

    rows = _rows(out)
    assert any(r["record"] == "probe" for r in rows), "the completed probe was thrown away"
    assert rows[-1]["record"] == "aborted"


@pytest.mark.parametrize("rel", [
    # direct child of data/eval: matched by the !data/eval/*.jsonl un-ignore
    ("data", "eval", "_pytest_refusal_probe.jsonl"),
    # SUBFOLDER of data/eval: matches no ignore rule at all -- the hole the
    # round-2 audit found in a parent-equality check
    ("data", "eval", "_pytest_refusal_dir", "_pytest_refusal_probe.jsonl"),
    # repo root: also trackable, guarded by the git-check branch
    ("_pytest_refusal_probe.jsonl",),
])
def test_transcript_refuses_repo_paths_git_would_track(tmp_path, monkeypatch, rel):
    """A transcript anywhere git would track it unseals every probe on the next
    commit -- the guard has to follow the actual risk (git), not one directory.

    Targets are unique names on REAL repo paths, removed in finally either way,
    so a leftover from any run cannot turn a passing guard into a red test.
    """
    probes = _probe_file(tmp_path)
    _fake_server(monkeypatch, LONG_ANSWER)
    target = eval_behavior.ROOT.joinpath(*rel)
    target.unlink(missing_ok=True)

    try:
        with pytest.raises(SystemExit) as exc:
            eval_behavior.run(URL, TEMP, MAXTOK, probes, target)

        assert "refusing" in str(exc.value)
        assert not target.exists(), "the guard raised but the file was written anyway"
    finally:
        target.unlink(missing_ok=True)
        if "_pytest_refusal_dir" in rel and target.parent.exists():
            target.parent.rmdir()  # a failed guard mkdir'd it


def test_gitignored_repo_path_is_still_allowed(tmp_path, monkeypatch):
    """The guard must not over-reach: data/pretrain is gitignored, so a
    transcript there is safe and refusing it would just push operators toward
    worse locations."""
    probes = _probe_file(tmp_path)
    _fake_server(monkeypatch, LONG_ANSWER)
    target = eval_behavior.ROOT / "data" / "pretrain" / "_pytest_allowed_transcript.jsonl"
    target.unlink(missing_ok=True)

    try:
        eval_behavior.run(URL, TEMP, MAXTOK, probes, target)
        assert target.exists(), "a gitignored repo path was wrongly refused or not written"
    finally:
        target.unlink(missing_ok=True)


def test_no_transcript_flag_writes_nothing(tmp_path, monkeypatch):
    probes = _probe_file(tmp_path)
    _fake_server(monkeypatch, LONG_ANSWER)

    eval_behavior.run(URL, TEMP, MAXTOK, probes, None)

    # Recursive: a default path written into a subdirectory must fail this too.
    assert list(tmp_path.rglob("*.jsonl")) == [probes]


def test_ungated_category_reports_informational_and_never_gates(tmp_path, monkeypatch, capsys):
    """A category missing from THRESHOLDS used to gate at >= 0% and print
    PASS -- a typo'd category name was invisible green. It must be labeled
    informational and must not decide the run's result either way."""
    p = tmp_path / "probes.jsonl"
    # The ungated category FAILS its probe; the run must still PASS overall.
    p.write_text(
        json.dumps({"category": "factual", "q": "Largest planet?", "want_any": ["jupiter"], "deny_any": []})
        + "\n"
        + json.dumps({"category": "reasoning", "q": "Can a bloop be a nork?", "want_any": ["zzz-never"], "deny_any": []})
        + "\n",
        encoding="utf-8",
    )
    _fake_server(monkeypatch, LONG_ANSWER)

    rc = eval_behavior.run(URL, TEMP, MAXTOK, p, None)

    out = capsys.readouterr().out
    assert "no threshold defined" in out
    assert rc == 0, "an ungated category's failure leaked into the gate"


def test_main_wires_the_transcript_flag(monkeypatch):
    """run() being correct is worthless if the CLI never passes the flag."""
    import inspect

    # The fake below hides run's real signature, so pin it separately: a drift
    # in the parameter name would otherwise slip past this test.
    assert "transcript" in inspect.signature(eval_behavior.run).parameters

    seen = {}

    def fake_run(base_url, temperature, max_tokens, probes, transcript):
        seen["transcript"] = transcript
        return 0

    monkeypatch.setattr(eval_behavior, "run", fake_run)
    monkeypatch.setattr("sys.argv", ["eval_behavior.py", "--transcript", "out/t.jsonl"])

    with pytest.raises(SystemExit):
        eval_behavior.main()

    assert seen["transcript"] is not None
    assert seen["transcript"].name == "t.jsonl"
