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
import eval_leak_guard

# Longer than the 60-char console line: a transcript that stored `detail`
# instead of the answer would silently truncate exactly the evidence a
# re-grade needs.
LONG_ANSWER = (
    "Jupiter is the largest planet in the solar system, and it is large enough "
    "that every other planet would fit inside it with room to spare."
)
# Deliberately NOT the argparse defaults (temperature 0.0 / max_tokens 60 /
# PORT 8123): a hardcoded field would match the defaults and the assertion
# would prove nothing. The HOST must stay local -- the scratch gate verifies
# host + reported memory dir, not just the port (review 2026-08-13) -- so
# the port alone carries the not-the-default duty here.
TEMP, MAXTOK, URL = 0.7, 13, "http://127.0.0.1:9999"


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
    # The run refuses a target outside the scratch ports because it wipes that
    # server's memory store first; this fake host is disposable by definition.
    monkeypatch.setattr(eval_behavior, "SCRATCH_PORTS", frozenset({9999}))
    # ...and it reports no store at all, so the dir half of the gate passes
    # (the gate's own behavior is pinned in test_eval_scratch_gate.py).
    monkeypatch.setattr(eval_behavior, "_capabilities",
                        lambda *a, **k: {"memory": False, "memory_dir": None})

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


def test_the_transcript_may_not_overwrite_its_own_baseline(tmp_path, monkeypatch, capsys):
    """--transcript and --baseline naming one file is not a no-op: the baseline
    is parsed into memory up front, the run proceeds normally, and the write at
    the end lands on top of the sealed receipt this run was measured against.
    Nothing downstream can notice -- the comparison already happened -- and the
    only artifact that could contest the verdict is gone."""
    probes = _probe_file(tmp_path)
    base = _baseline_file(tmp_path, probes,
                          {"factual": {"hits": 0, "n": 1}, "tool": {"hits": 1, "n": 1}})
    before = base.read_bytes()

    def touched(*a, **k):
        raise AssertionError("server contacted despite the transcript/baseline collision")

    monkeypatch.setattr(eval_behavior, "_wait_for_server", touched)
    monkeypatch.setattr(eval_behavior, "_clear_memory", touched)

    assert eval_behavior.run(URL, TEMP, MAXTOK, probes, base, baseline=base) == 2
    assert "FAIL" in capsys.readouterr().out
    assert base.read_bytes() == before

    # The same file reached by another spelling is the same file.
    (tmp_path / "sub").mkdir()
    spelled = tmp_path / "sub" / ".." / base.name
    assert eval_behavior.run(URL, TEMP, MAXTOK, probes, spelled, baseline=base) == 2
    assert base.read_bytes() == before


def test_a_torn_transcript_write_leaves_the_previous_one_intact(tmp_path, monkeypatch):
    """The transcript is written at the end of a run and again on the ABORT
    path, whose whole purpose is not losing collected answers. A plain
    write_text truncates in place, so a write torn in half by a full disk or a
    killed process leaves a short receipt where a complete one was -- and JSONL
    offers no way to tell the two apart afterwards."""
    from enigma_engine.core import safe_save

    out = tmp_path / "transcript.jsonl"
    eval_behavior._write_transcript(out, [{"record": "run_conditions"},
                                          {"record": "probe", "q": "Largest planet?"}])
    eval_behavior._write_transcript(out, [{"record": "run_conditions"}])
    # A rotated .bak would be a second plaintext copy of every sealed probe and
    # answer, and only the transcript path itself cleared the unsealing check.
    assert not list(tmp_path.glob("*.bak"))
    good = out.read_bytes()

    def die(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(safe_save.os, "replace", die)
    with pytest.raises(OSError):
        eval_behavior._write_transcript(out, [{"record": "run_conditions"},
                                              {"record": "aborted"}])
    assert out.read_bytes() == good, "a failed write replaced a complete transcript"
    assert not list(tmp_path.glob("*.tmp")), "a torn write left its temp file behind"


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


def test_a_non_scratch_target_is_refused_before_anything_is_cleared(tmp_path, monkeypatch):
    """The run DELETES the target server's memories before probing. Pointed at
    the daily server it wiped her real store, then wrote probe facts into it --
    so the refusal has to land before the clear, not after."""
    probes = _probe_file(tmp_path)
    cleared = []
    monkeypatch.setattr(eval_behavior, "_wait_for_server", lambda *a, **k: True)
    monkeypatch.setattr(eval_behavior, "_clear_memory", lambda *a, **k: cleared.append(1))
    # No live network I/O from a unit test: the gate's caps argument is
    # evaluated before the port check runs, and an unstubbed _capabilities
    # is a real urlopen at a 10s timeout (audit 2026-08-14).
    monkeypatch.setattr(eval_behavior, "_capabilities", lambda *a, **k: {})
    monkeypatch.setattr(eval_behavior, "_post", lambda *a, **k: pytest.fail("probed a live server"))

    rc = eval_behavior.run("http://127.0.0.1:8000", TEMP, MAXTOK, probes, None)

    assert rc == 2
    assert cleared == [], "the memory store was cleared before the refusal"
    # ...and the escape hatch still works for a server the caller vouches for.
    monkeypatch.setattr(eval_behavior, "_post",
                        lambda *a, **k: {"choices": [{"message": {"content": LONG_ANSWER}}]})
    assert eval_behavior.run("http://127.0.0.1:8000", TEMP, MAXTOK, probes, None,
                             allow_live_server=True) in (0, 1)
    assert cleared == [1]


def test_an_edited_locked_file_cannot_be_scored(tmp_path, monkeypatch, capsys):
    """The transcript records a probe sha AFTER the fact. Nothing checked that
    the holdout still WAS the sealed set, so an edited gate produced a
    perfectly normal-looking scorecard."""
    real = eval_behavior.ROOT / "data" / "eval" / "locked_probes.jsonl"
    if not real.exists() or not eval_behavior.LOCKED_MANIFEST.exists():
        pytest.skip("no sealed locked set in this checkout")
    cases = [json.loads(x) for x in real.read_text(encoding="utf-8").splitlines() if x.strip()]
    cases[0]["q"] += " (edited)"
    tampered = tmp_path / "locked_probes.jsonl"
    tampered.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n",
                        encoding="utf-8")
    _fake_server(monkeypatch, LONG_ANSWER)

    assert eval_behavior.run(URL, TEMP, MAXTOK, tampered, None) == 2
    assert "not the sealed holdout" in capsys.readouterr().out
    # the untouched file still verifies, or the check would be refusing everything
    intact = [json.loads(x) for x in real.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert eval_behavior._seal_mismatch(intact, real) is None


def test_grading_keys_are_verified_without_the_plaintext_on_disk(tmp_path, monkeypatch):
    """The reference-file version of this check failed OPEN three ways: the
    plaintext is gitignored (absent on a clone), the canonical run compares the
    file with ITSELF, and anyone able to drop a rig could overwrite the
    reference. All three printed 'seal verified' over unverified keys. The
    digest now lives in the manifest, so none of them depend on disk state."""
    real = eval_behavior.ROOT / "data" / "eval" / "locked_probes.jsonl"
    if not real.exists() or not eval_behavior.LOCKED_MANIFEST.exists():
        pytest.skip("no sealed locked set in this checkout")
    cases = [json.loads(x) for x in real.read_text(encoding="utf-8").splitlines() if x.strip()]

    # the canonical run: probes IS the reference, and tampering is still caught
    assert eval_behavior._seal_mismatch(cases, real) is None
    gutted = [dict(c) for c in cases]
    for c in gutted:
        c["want_any"], c["deny_any"] = [], []
        if "expect_tool" in c:
            c["expect_tool"] = None
    assert "grading keys" in (eval_behavior._seal_mismatch(gutted, real) or "")

    # a manifest predating the grading seal must FAIL CLOSED, not pass quietly
    legacy = json.loads(eval_behavior.LOCKED_MANIFEST.read_text(encoding="utf-8"))
    legacy.pop("grading_digest", None)
    stand_in = tmp_path / "legacy.manifest.json"
    stand_in.write_text(json.dumps(legacy), encoding="utf-8")
    monkeypatch.setattr(eval_behavior, "LOCKED_MANIFEST", stand_in)
    reason = eval_behavior._seal_mismatch(cases, real)
    assert reason and "re-seal" in reason


def test_a_manifest_with_stripped_arrays_cannot_verify(tmp_path, monkeypatch):
    """The s/n arrays are the trainers' enforcement payload, and the trainers
    cannot verify them (no plaintext). h-only comparison here printed "seal
    verified" over a manifest whose arrays were emptied -- every downstream
    training screen then ran exact-hash-only (round-B audit, 2026-07-25). The
    guard refuses the FULL strip outright; the partial strip (indistinguishable
    from a stopword-only probe without plaintext) is caught here, where the
    gate run holds the plaintext and recomputes the whole payload."""
    from eval_leak_guard import LockedProbeGuard

    real = eval_behavior.ROOT / "data" / "eval" / "locked_probes.jsonl"
    if not real.exists() or not eval_behavior.LOCKED_MANIFEST.exists():
        pytest.skip("no sealed locked set in this checkout")
    cases = [json.loads(x) for x in real.read_text(encoding="utf-8").splitlines() if x.strip()]
    original = json.loads(eval_behavior.LOCKED_MANIFEST.read_text(encoding="utf-8"))

    # full strip: refused by the guard before any comparison runs
    stripped = json.loads(json.dumps(original))
    for p in stripped["probes"]:
        p["s"], p["n"] = [], []
    stand_in = tmp_path / "stripped.manifest.json"
    stand_in.write_text(json.dumps(stripped), encoding="utf-8")
    monkeypatch.setattr(eval_behavior, "LOCKED_MANIFEST", stand_in)
    with pytest.raises(LockedProbeGuard.Weakened):
        eval_behavior._sealed_manifest()

    # partial strip: empty the shingles of one run-free (1-content-word)
    # probe -- loads clean, and only the full-payload comparison sees it
    partial = json.loads(json.dumps(original))
    victim = next((p for p in partial["probes"] if len(p["s"]) == 1 and not p["n"]), None)
    if victim is None:
        pytest.skip("sealed set carries no 1-content-word probe to strip")
    victim["s"] = []
    stand_in2 = tmp_path / "partial.manifest.json"
    stand_in2.write_text(json.dumps(partial), encoding="utf-8")
    monkeypatch.setattr(eval_behavior, "LOCKED_MANIFEST", stand_in2)
    reason = eval_behavior._seal_mismatch(cases, real)
    assert reason and "sealed arrays" in reason


def test_a_diluted_copy_of_the_locked_set_is_still_the_locked_set(tmp_path):
    """Share alone was evadable in one direction: padding a full copy with 13
    junk strings dropped it under the bar while still carrying every sealed
    question, so it ran ungated and printed PASS. Junk cannot REMOVE sealed
    content."""
    real = eval_behavior.ROOT / "data" / "eval" / "locked_probes.jsonl"
    if not real.exists() or not eval_behavior.LOCKED_MANIFEST.exists():
        pytest.skip("no sealed locked set in this checkout")
    cases = [json.loads(x) for x in real.read_text(encoding="utf-8").splitlines() if x.strip()]
    for pad in (13, 60):
        diluted = cases + [
            {"category": "factual", "q": f"unrelated filler question {i}", "want_any": ["x"]}
            for i in range(pad)
        ]
        assert eval_behavior._touches_sealed_probes(diluted), f"evaded with {pad} junk strings"
        assert eval_behavior._seal_mismatch(diluted, real) is not None


def test_the_gate_is_identified_by_its_bytes(tmp_path):
    """Every hash-set test answers "does this file MEAN the same thing", which
    is the wrong question for identity: it was evaded once per audit round
    through whichever normalization dimension was left over -- case,
    punctuation, non-Latin script, then whitespace, where doubling every space
    kept the seal intact while changing what all 96 questions posted to the
    model. Bytes have no dimensions left to evade."""
    real = eval_behavior.ROOT / "data" / "eval" / "locked_probes.jsonl"
    if not real.exists() or not eval_behavior.LOCKED_MANIFEST.exists():
        pytest.skip("no sealed locked set in this checkout")
    cases = [json.loads(x) for x in real.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert eval_behavior._seal_mismatch(cases, real) is None

    def written(name, recs):
        p = tmp_path / name
        p.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in recs),
                     encoding="utf-8", newline="\n")
        return p, recs

    # the whitespace channel: same grading digest, same probe hashes, different bytes
    spaced = [dict(c, q=(c.get("q") or "").replace(" ", "  "),
                   teach=[t.replace(" ", "  ") for t in (c.get("teach") or [])])
              for c in cases]
    p, recs = written("spaced.jsonl", spaced)
    assert eval_leak_guard.grading_digest(recs) == eval_leak_guard.grading_digest(cases), (
        "fixture assumption gone: whitespace now moves the grading digest, so this "
        "test no longer exercises the channel bytes were introduced to close"
    )
    assert eval_behavior._seal_mismatch(recs, p) == "this file is not byte-identical to the sealed holdout"

    # and every other single-dimension edit falls to the same check
    for name, recs in (
        ("gutted.jsonl", [dict(c, want_any=[], deny_any=[]) for c in cases]),
        ("reordered.jsonl", list(reversed(cases))),
        ("dropped.jsonl", cases[:-1]),
    ):
        p, recs = written(name, recs)
        assert eval_behavior._seal_mismatch(recs, p) is not None, name


def test_a_manifest_without_the_file_seal_is_refused(tmp_path, monkeypatch):
    """Fail CLOSED, the same way the grading seal does: a manifest that cannot
    prove byte identity cannot prove this file is the holdout."""
    real = eval_behavior.ROOT / "data" / "eval" / "locked_probes.jsonl"
    if not real.exists() or not eval_behavior.LOCKED_MANIFEST.exists():
        pytest.skip("no sealed locked set in this checkout")
    cases = [json.loads(x) for x in real.read_text(encoding="utf-8").splitlines() if x.strip()]
    legacy = json.loads(eval_behavior.LOCKED_MANIFEST.read_text(encoding="utf-8"))
    legacy.pop("probe_file_sha256", None)
    stand_in = tmp_path / "legacy.manifest.json"
    stand_in.write_text(json.dumps(legacy), encoding="utf-8")
    monkeypatch.setattr(eval_behavior, "LOCKED_MANIFEST", stand_in)
    reason = eval_behavior._seal_mismatch(cases, real)
    assert reason and "re-seal" in reason


def test_trimming_one_string_and_padding_does_not_defeat_both_detectors(tmp_path):
    """Containment and share are both PROPORTIONS of the file, so one lever bends
    both: drop a single sealed string (containment fails) and pad with a dozen
    junk ones (share falls under the bar). A copy still carrying 95 of 96 sealed
    questions then ran ungated and printed RESULT: PASS, differing from a real
    gate run by one missing line. The cheapest padding was junk `teach` lines on
    a non-memory probe -- counted by _probe_hashes, never posted, never graded,
    invisible in the scorecard. An absolute floor cannot be diluted, because
    padding only ever adds strings."""
    real = eval_behavior.ROOT / "data" / "eval" / "locked_probes.jsonl"
    if not real.exists() or not eval_behavior.LOCKED_MANIFEST.exists():
        pytest.skip("no sealed locked set in this checkout")
    cases = [json.loads(x) for x in real.read_text(encoding="utf-8").splitlines() if x.strip()]

    for junk in (12, 13, 40, 200):
        rigged = [dict(c) for c in cases]
        for c in rigged:  # kill containment: one sealed teach line removed
            if c.get("category") == "memory" and c.get("teach"):
                c["teach"] = []
                break
        for c in rigged:  # kill share: ballast that never reaches the server
            if c.get("category") != "memory":
                c["teach"] = [f"ballast string {i}" for i in range(junk)]
                break
        for c in rigged:  # and rig the grading so any answer passes
            c["want_any"], c["deny_any"], c["expect_tool"] = [], [], None
        assert eval_behavior._touches_sealed_probes(rigged), f"evaded with {junk} junk strings"
        # written out, because the seal check reads the file to digest it
        rigged_path = tmp_path / f"rigged_{junk}.jsonl"
        rigged_path.write_text(
            "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in rigged),
            encoding="utf-8", newline="\n",
        )
        assert eval_behavior._seal_mismatch(rigged, rigged_path) is not None


def test_gutted_grading_keys_are_caught_even_though_the_questions_match(tmp_path, monkeypatch, capsys):
    """The questions are sealed; `want_any`, `deny_any`, `expect_tool` and
    `category` are NOT, and they decide every score. Emptying them re-seals
    perfectly and then passes five of eight gated categories unconditionally
    (`_grade_text` with no wants and no denies returns True for any answer)."""
    real = eval_behavior.ROOT / "data" / "eval" / "locked_probes.jsonl"
    if not real.exists() or not eval_behavior.LOCKED_MANIFEST.exists():
        pytest.skip("no sealed locked set in this checkout")
    cases = [json.loads(x) for x in real.read_text(encoding="utf-8").splitlines() if x.strip()]
    for c in cases:  # every question untouched, every grading key neutered
        c["want_any"], c["deny_any"] = [], []
        if "expect_tool" in c:
            c["expect_tool"] = None
    gutted = tmp_path / "locked_probes.jsonl"
    gutted.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n",
                      encoding="utf-8")
    _fake_server(monkeypatch, LONG_ANSWER)

    assert eval_behavior.run(URL, TEMP, MAXTOK, gutted, None) == 2
    # The byte check now refuses this file before the grading digest is
    # consulted -- an earlier and stronger refusal for the same edit.
    out = capsys.readouterr().out
    assert "not the sealed holdout" in out
    assert "byte-identical" in out or "grading keys" in out


def test_the_dev_set_is_not_mistaken_for_the_locked_set():
    """Content-based gate detection has to tell a COPY of the sealed set from a
    file that merely shares a string with it. Since reseal #7 (2026-07-27) the
    dev and sealed sets share NOTHING (the one shared teach line was
    re-contented on purpose -- no sealed string is public), so the
    shares-a-string case is constructed: dev plus one sealed record must still
    read as an ordinary file (1 hit is far under the content floor), while any
    trimmed copy of the holdout reads as locked."""
    real = eval_behavior.ROOT / "data" / "eval" / "locked_probes.jsonl"
    if not real.exists() or not eval_behavior.LOCKED_MANIFEST.exists():
        pytest.skip("no sealed locked set in this checkout")

    def cases(path):
        return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

    dev = cases(eval_behavior.PROBES)
    locked = cases(real)
    assert not eval_behavior._touches_sealed_probes(dev), "the dev set reads as the locked set"
    # the real overlap is ZERO now -- pin that, it is a seal property
    sealed = set(eval_behavior._sealed_hashes())
    assert sum(h in sealed for h in eval_behavior._probe_hashes(dev)) == 0, (
        "a sealed string is in the open dev set -- it is public; re-content one side"
    )
    # a file that SHARES one sealed record is still not the holdout...
    assert not eval_behavior._touches_sealed_probes(dev + locked[:1])
    assert eval_behavior._touches_sealed_probes(locked)
    # ...but a TRIMMED copy is still locked content -- the PASS-on-12-probes hole
    assert eval_behavior._touches_sealed_probes(locked[:12])


def test_renaming_the_locked_set_does_not_dodge_the_seal_check(tmp_path, monkeypatch, capsys):
    """Gate-ness keyed on the FILENAME: a copy called anything else skipped
    verification entirely, and a trimmed copy then scored one category and
    printed PASS with seven gates unmeasured."""
    real = eval_behavior.ROOT / "data" / "eval" / "locked_probes.jsonl"
    if not real.exists() or not eval_behavior.LOCKED_MANIFEST.exists():
        pytest.skip("no sealed locked set in this checkout")
    cases = [json.loads(x) for x in real.read_text(encoding="utf-8").splitlines() if x.strip()]
    trimmed = [c for c in cases if c.get("category") == "factual"]
    assert trimmed, "fixture needs at least one factual probe"
    sneaky = tmp_path / "my_probes.jsonl"  # nothing in the name says locked
    sneaky.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in trimmed) + "\n",
                      encoding="utf-8")
    _fake_server(monkeypatch, LONG_ANSWER)

    rc = eval_behavior.run(URL, TEMP, MAXTOK, sneaky, None)

    out = capsys.readouterr().out
    assert rc == 2, "a renamed subset of the sealed set was scored as an ordinary file"
    assert "not the sealed holdout" in out
    assert "RESULT: PASS" not in out


def test_a_file_with_no_gated_category_never_reports_pass(tmp_path, monkeypatch, capsys):
    """PASS on a file where nothing has a threshold means only that nothing was
    checked -- benchmark_future_capabilities.jsonl is entirely such categories."""
    p = tmp_path / "probes.jsonl"
    p.write_text(json.dumps({"category": "creative", "q": "Write a haiku.", "want_any": []}) + "\n",
                 encoding="utf-8")
    _fake_server(monkeypatch, LONG_ANSWER)

    rc = eval_behavior.run(URL, TEMP, MAXTOK, p, None)

    out = capsys.readouterr().out
    assert "NOT GATED" in out
    assert "RESULT: PASS" not in out
    assert rc == 1


def test_an_empty_probe_file_is_an_error_before_the_store_is_touched(tmp_path, monkeypatch):
    """It used to divide by zero at the scorecard -- after clearing the target's
    memories. Nothing worth wiping a store for happens with no probes."""
    p = tmp_path / "probes.jsonl"
    p.write_text("", encoding="utf-8")
    cleared = []
    monkeypatch.setattr(eval_behavior, "_wait_for_server", lambda *a, **k: True)
    monkeypatch.setattr(eval_behavior, "_clear_memory", lambda *a, **k: cleared.append(1))
    monkeypatch.setattr(eval_behavior, "SCRATCH_PORTS", frozenset({9999}))

    assert eval_behavior.run(URL, TEMP, MAXTOK, p, None) == 2
    assert cleared == []


def test_malformed_probe_records_refuse_before_the_server_is_touched(tmp_path, monkeypatch, capsys):
    """The scorer reads c["q"] and c["category"] unconditionally and runs only
    after the target's memory store has been cleared -- a record missing either
    used to crash mid-suite with the store already gone. The refusal must fire
    before ANY server contact."""
    p = tmp_path / "probes.jsonl"
    p.write_text(
        json.dumps({"category": "factual", "q": "Largest planet?", "want_any": ["jupiter"]})
        + "\n"
        + json.dumps({"question": "Alias spelling, no q at all", "category": "factual"})
        + "\n"
        + json.dumps({"q": "No category on this one", "want_any": ["x"]})
        + "\n",
        encoding="utf-8",
    )

    def touched(*a, **k):
        raise AssertionError("server contacted despite malformed probe records")

    monkeypatch.setattr(eval_behavior, "_wait_for_server", touched)
    monkeypatch.setattr(eval_behavior, "_clear_memory", touched)
    monkeypatch.setattr(eval_behavior, "SCRATCH_PORTS", frozenset({9999}))

    assert eval_behavior.run(URL, TEMP, MAXTOK, p, None) == 2
    out = capsys.readouterr().out
    assert "missing a non-empty 'q' or 'category'" in out
    assert "[2, 3]" in out  # both offending records named, 1-indexed


def test_tool_arguments_reach_the_transcript(tmp_path, monkeypatch):
    """Grading reads the first call's NAME only, so a right tool with wrong
    arguments scores identically. The evidence has to survive the run."""
    probes = _probe_file(tmp_path)
    out = tmp_path / "transcript.jsonl"
    monkeypatch.setattr(eval_behavior, "_wait_for_server", lambda *a, **k: True)
    monkeypatch.setattr(eval_behavior, "_clear_memory", lambda *a, **k: None)
    monkeypatch.setattr(eval_behavior, "SCRATCH_PORTS", frozenset({9999}))
    # The gate FAILS CLOSED on an unanswered capabilities probe (audit
    # 2026-08-14) -- an unstubbed _capabilities returns {} here and the run
    # would honestly refuse before writing any transcript.
    monkeypatch.setattr(eval_behavior, "_capabilities",
                        lambda *a, **k: {"memory": False, "memory_dir": None})
    monkeypatch.setattr(eval_behavior, "_post", lambda base_url, payload: {"choices": [{"message": {
        "content": "",
        "tool_calls": [{"function": {"name": "get_weather", "arguments": '{"city": "Sydney"}'}}],
    }}]})

    eval_behavior.run(URL, TEMP, MAXTOK, probes, out)

    tool = next(r for r in _rows(out) if r.get("category") == "tool")
    assert tool["tool_calls_full"] == [{"name": "get_weather", "arguments": '{"city": "Sydney"}'}]


def test_transcript_records_which_weights_answered(tmp_path, monkeypatch, capsys):
    """The seal proves which questions were asked; model_checkpoint is the only
    record of what answered them. A server that reports identity gets it into
    the conditions row and the banner; one that cannot gets an empty dict and
    a loud WARN -- never a silent omission."""
    probes = _probe_file(tmp_path)
    out = tmp_path / "transcript.jsonl"
    _fake_server(monkeypatch, LONG_ANSWER)
    identity = {"path": "C:/models/enigma_dpo/model.pth", "sha256": "ab" * 32, "step": 51000}
    monkeypatch.setattr(eval_behavior, "_model_identity", lambda base_url: identity)

    eval_behavior.run(URL, TEMP, MAXTOK, probes, out)

    conditions = _rows(out)[0]
    assert conditions["model_checkpoint"] == identity
    assert "model under test: sha256 abababababababab" in capsys.readouterr().out

    # Pre-identity server: {} recorded, WARN printed.
    monkeypatch.setattr(eval_behavior, "_model_identity", lambda base_url: {})
    eval_behavior.run(URL, TEMP, MAXTOK, probes, out)
    assert _rows(out)[0]["model_checkpoint"] == {}
    assert "cannot prove which weights answered" in capsys.readouterr().out


def _baseline_file(tmp_path, probes, by_category, model_sha=None, name="baseline.jsonl"):
    hits = sum(c["hits"] for c in by_category.values())
    n = sum(c["n"] for c in by_category.values())
    conditions = {
        "record": "run_conditions",
        "probe_sha256": eval_behavior._probe_digest(probes),
        "temperature": TEMP,
        "max_tokens": MAXTOK,
        "model_checkpoint": {"sha256": model_sha} if model_sha else {},
    }
    card = {
        "record": "scorecard",
        "by_category": by_category,
        "overall_hits": hits,
        "overall_n": n,
        "sealed_gate_run": False,
    }
    p = tmp_path / name
    p.write_text(json.dumps(conditions) + "\n" + json.dumps(card) + "\n", encoding="utf-8")
    return p


def test_baseline_comparator_speaks_the_adoption_rule(tmp_path, monkeypatch, capsys):
    """BACKLOG T6 as code: beat the aggregate with no category floor regression
    = ADOPT; anything else = the user's call. Comparing by eye at n=12 per
    category is where a one-cell slip decides adoption."""
    probes = _probe_file(tmp_path)  # factual 1 + tool 1; fake server scores 2/2
    out = tmp_path / "transcript.jsonl"
    _fake_server(monkeypatch, "Jupiter is the largest planet.")

    # Current 2/2 vs baseline 1/2, no regression -> ADOPT.
    base = _baseline_file(tmp_path, probes,
                          {"factual": {"hits": 0, "n": 1}, "tool": {"hits": 1, "n": 1}})
    eval_behavior.run(URL, TEMP, MAXTOK, probes, out, baseline=base)
    out_text = capsys.readouterr().out
    assert "VERDICT: ADOPT" in out_text
    assert "category regressions: none" in out_text
    comparison = next(r for r in _rows(out) if r["record"] == "baseline_comparison")
    assert comparison["verdict"].startswith("ADOPT")
    assert comparison["regressions"] == []

    # A tie does not adopt.
    base = _baseline_file(tmp_path, probes,
                          {"factual": {"hits": 1, "n": 1}, "tool": {"hits": 1, "n": 1}})
    eval_behavior.run(URL, TEMP, MAXTOK, probes, out, baseline=base)
    assert "VERDICT: USER'S CALL (aggregate not beaten)" in capsys.readouterr().out


def test_baseline_comparator_names_regressions_and_refusals(tmp_path, monkeypatch, capsys):
    probes = _probe_file(tmp_path)
    _fake_server(monkeypatch, "It is Saturn, I think.")  # factual MISSES, tool still hits

    # Baseline 2/2 vs current 1/2: the factual regression must be NAMED.
    base = _baseline_file(tmp_path, probes,
                          {"factual": {"hits": 1, "n": 1}, "tool": {"hits": 1, "n": 1}})
    eval_behavior.run(URL, TEMP, MAXTOK, probes, None, baseline=base)
    out_text = capsys.readouterr().out
    assert "USER'S CALL" in out_text
    assert "factual 1/1 -> 0/1" in out_text

    # A baseline for a DIFFERENT probe set refuses to compare at all.
    other = tmp_path / "other.jsonl"
    other.write_text(json.dumps({"category": "factual", "q": "Deepest ocean?", "want_any": ["pacific"]}) + "\n",
                     encoding="utf-8")
    base = _baseline_file(tmp_path, other, {"factual": {"hits": 1, "n": 1}}, name="other_base.jsonl")
    eval_behavior.run(URL, TEMP, MAXTOK, probes, None, baseline=base)
    assert "comparison refused" in capsys.readouterr().out

    # Same checkpoint sha on both sides = someone forgot a restart. Say so.
    sha = "cd" * 32
    monkeypatch.setattr(eval_behavior, "_model_identity",
                        lambda base_url: {"sha256": sha, "step": 1, "path": "x"})
    base = _baseline_file(tmp_path, probes,
                          {"factual": {"hits": 0, "n": 1}, "tool": {"hits": 1, "n": 1}},
                          model_sha=sha)
    eval_behavior.run(URL, TEMP, MAXTOK, probes, None, baseline=base)
    assert "compares a model against itself" in capsys.readouterr().out

    # A missing baseline file fails before any server is touched.
    def touched(*a, **k):
        raise AssertionError("server contacted despite missing baseline")

    monkeypatch.setattr(eval_behavior, "_wait_for_server", touched)
    assert eval_behavior.run(URL, TEMP, MAXTOK, probes, None,
                             baseline=tmp_path / "nope.jsonl") == 2


def test_malformed_baseline_refuses_before_the_run_not_after(tmp_path, monkeypatch, capsys):
    """The baseline used to be parsed at the very END -- after 96 sealed
    answers were collected and before the transcript was written -- so a
    hand-made or truncated file destroyed a completed gate run's receipt.
    Every malformed shape must now refuse up front, with no server contact."""
    probes = _probe_file(tmp_path)

    def touched(*a, **k):
        raise AssertionError("server contacted despite malformed baseline")

    monkeypatch.setattr(eval_behavior, "_wait_for_server", touched)
    monkeypatch.setattr(eval_behavior, "_clear_memory", touched)
    monkeypatch.setattr(eval_behavior, "SCRATCH_PORTS", frozenset({9999}))

    shapes = {
        "cell not a dict": {"record": "scorecard", "by_category": {"factual": 1},
                            "overall_hits": 1, "overall_n": 2},
        "cell missing hits": {"record": "scorecard", "by_category": {"factual": {"n": 1}},
                              "overall_hits": 1, "overall_n": 2},
        "by_category a list": {"record": "scorecard", "by_category": [1, 2],
                               "overall_hits": 1, "overall_n": 2},
        "overall null": {"record": "scorecard", "by_category": {},
                         "overall_hits": None, "overall_n": 2},
    }
    for name, card in shapes.items():
        base = tmp_path / "bad_base.jsonl"
        base.write_text(json.dumps({"record": "run_conditions"}) + "\n" + json.dumps(card) + "\n",
                        encoding="utf-8")
        assert eval_behavior.run(URL, TEMP, MAXTOK, probes, None, baseline=base) == 2, name
        assert "malformed" in capsys.readouterr().out, name

    # rows that are not objects are skipped, not crashed on
    base = tmp_path / "junk_rows.jsonl"
    base.write_text("[1, 2]\nnot json\n" + json.dumps({"record": "scorecard"}) + "\n",
                    encoding="utf-8")
    assert eval_behavior.run(URL, TEMP, MAXTOK, probes, None, baseline=base) == 2

    # a non-UTF-8 baseline refuses cleanly
    base = tmp_path / "binary.jsonl"
    base.write_bytes(b"\xff\xfe\x00garbage")
    assert eval_behavior.run(URL, TEMP, MAXTOK, probes, None, baseline=base) == 2
    assert "unreadable" in capsys.readouterr().out

    # a null teach in a probe record also refuses before the server
    p = tmp_path / "null_teach.jsonl"
    p.write_text(json.dumps({"category": "memory", "teach": None,
                             "q": "What is my cat called?", "want_any": ["mochi"]}) + "\n",
                 encoding="utf-8")
    assert eval_behavior.run(URL, TEMP, MAXTOK, p, None) == 2
    assert "malformed 'teach'" in capsys.readouterr().out


def test_comparator_floors_are_gated_categories_only(capsys):
    """'Category floor regression' is defined over the gated categories: an
    informational organ column (vision/speech/imagery) has no floor, and an
    eyes-on vs eyes-off server pair would otherwise veto adoption with a
    regression no model caused. The capabilities drift IS said out loud."""
    from pathlib import Path as _P

    by_cat = {"factual": [True], "vision": [False]}
    base_cond = {"probe_sha256": "X", "temperature": TEMP, "max_tokens": MAXTOK,
                 "capabilities": {"eyes": True}}
    base_card = {"by_category": {"factual": {"hits": 0, "n": 1},
                                 "vision": {"hits": 1, "n": 1}},
                 "overall_hits": 1, "overall_n": 2, "sealed_gate_run": False}
    conditions = {"probe_sha256": "X", "temperature": TEMP, "max_tokens": MAXTOK,
                  "capabilities": {"eyes": False}, "model_checkpoint": {}}
    rec = eval_behavior._compare_to_baseline(_P("base.jsonl"), base_cond, base_card,
                                             conditions, by_cat, 2, 2, False)
    out = capsys.readouterr().out
    assert rec["regressions"] == []  # the vision drop names no regression
    assert "capabilities differ" in out


def test_model_identity_tolerates_foreign_bodies(monkeypatch):
    """Pointing --base-url at a non-Enigma OpenAI-ish endpoint must yield {}
    ('target could not say'), never a crash after the store was cleared."""
    import io

    for body in (["gpt-4"], "ok", None, {"data": "x"}, {"data": []},
                 {"data": [{"id": "m"}]}, {"data": [{"checkpoint": "str"}]}):
        payload = json.dumps(body).encode()

        class _Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        monkeypatch.setattr(eval_behavior.urllib.request, "urlopen",
                            lambda req, timeout=10, _p=payload: _Resp(_p))
        assert eval_behavior._model_identity("http://h") == {}, body

    # a non-UTF-8 body is a byte shape, not a JSON shape -- it must also
    # yield {} (UnicodeDecodeError is a ValueError, not a JSONDecodeError)
    class _Bin(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(eval_behavior.urllib.request, "urlopen",
                        lambda req, timeout=10: _Bin(b"\xff\xfe\x00garbage"))
    assert eval_behavior._model_identity("http://h") == {}
    assert eval_behavior._capabilities("http://h") == {}


def test_main_wires_the_transcript_flag(monkeypatch):
    """run() being correct is worthless if the CLI never passes the flag."""
    import inspect

    # The fake below hides run's real signature, so pin it separately: a drift
    # in the parameter name would otherwise slip past this test.
    assert "transcript" in inspect.signature(eval_behavior.run).parameters
    assert "allow_live_server" in inspect.signature(eval_behavior.run).parameters
    assert "baseline" in inspect.signature(eval_behavior.run).parameters

    seen = {}

    def fake_run(base_url, temperature, max_tokens, probes, transcript,
                 allow_live_server=False, baseline=None):
        seen["transcript"] = transcript
        seen["allow_live_server"] = allow_live_server
        seen["baseline"] = baseline
        return 0

    monkeypatch.setattr(eval_behavior, "run", fake_run)
    monkeypatch.setattr("sys.argv", ["eval_behavior.py", "--transcript", "out/t.jsonl"])

    with pytest.raises(SystemExit):
        eval_behavior.main()

    assert seen["transcript"] is not None
    assert seen["transcript"].name == "t.jsonl"
    assert seen["allow_live_server"] is False  # the guard is on unless asked for
    assert seen["baseline"] is None

    monkeypatch.setattr("sys.argv", ["eval_behavior.py", "--allow-live-server",
                                     "--baseline", "out/base.jsonl"])
    with pytest.raises(SystemExit):
        eval_behavior.main()
    assert seen["allow_live_server"] is True
    assert seen["baseline"] is not None and seen["baseline"].name == "base.jsonl"


def test_transcript_records_which_branch_graded_each_probe(tmp_path, monkeypatch):
    """`expect_tool` is present-and-null on a restraint probe ("call nothing")
    and absent on a text probe, and json serializes BOTH to null. Without a
    separate flag the transcript cannot be re-graded from itself: restraint
    reads as text, and text grading with empty want/deny passes anything, so an
    offline re-grade scores the whole restraint column as passing.

    Driven through run() against the fake server, because the field only means
    something if the writer actually emits it -- asserting on hand-built dicts
    tests the `in` operator, not the transcript."""
    probes = tmp_path / "probes.jsonl"
    probes.write_text(
        json.dumps({"category": "restraint", "q": "Winter here was brutal.",
                    "expect_tool": None}) + "\n"
        + json.dumps({"category": "factual", "q": "Largest planet?",
                      "want_any": ["jupiter"]}) + "\n",
        encoding="utf-8")
    out = tmp_path / "t.jsonl"
    _fake_server(monkeypatch, LONG_ANSWER)
    eval_behavior.run(URL, TEMP, MAXTOK, probes, out)

    by_cat = {r["category"]: r for r in _rows(out) if r.get("record") == "probe"}
    # both serialize expect_tool to null -- that is the ambiguity
    assert by_cat["restraint"]["expect_tool"] is None
    assert by_cat["factual"]["expect_tool"] is None
    # ...and only tool_graded tells them apart
    assert by_cat["restraint"]["tool_graded"] is True
    assert by_cat["factual"]["tool_graded"] is False

    # the point of the field: a re-grade from the transcript ALONE reaches the
    # same verdict the run did. Without it the restraint row would be graded as
    # text against empty keys and pass unconditionally.
    for row in by_cat.values():
        if row["tool_graded"]:
            regraded = (row["tool_called"] == row["expect_tool"])
        else:
            regraded = eval_behavior._grade_text(
                row["content"].lower(), row.get("want_any") or [],
                row.get("deny_any") or [])
        assert regraded == row["graded_ok"], f"re-grade diverged on {row['category']}"


def test_malformed_grading_keys_are_refused_before_the_store_is_wiped():
    """run() clears the target's memory store, then grades. A record whose
    want_any/deny_any is null, a bare string, a dict or a list of non-strings
    reaches the scorer and dies there -- with the store already gone. Each of
    these must be caught by the pre-flight instead."""
    bad_shapes = [
        {"q": "x", "category": "c", "deny_any": None},
        {"q": "x", "category": "c", "deny_any": [1, 2]},
        {"q": "x", "category": "c", "want_any": 5},
        {"q": "x", "category": "c", "want_any": [None]},
        {"q": "x", "category": "c", "want_any": "jupiter"},
        {"q": "x", "category": "c", "want_any": {"a": 1}},
        {"q": "x", "category": "c", "expect_tool": 7},
    ]
    for rec in bad_shapes:
        assert eval_behavior._malformed_probe(rec) is True,             f"accepted a malformed record: {rec}"
    # the legal shapes still pass the same gate
    for rec in ({"q": "x", "category": "c"},
                {"q": "x", "category": "c", "want_any": ["a"], "deny_any": []},
                {"q": "x", "category": "restraint", "expect_tool": None},
                {"q": "x", "category": "tool", "expect_tool": "get_weather"}):
        assert eval_behavior._malformed_probe(rec) is False, rec


def test_a_one_probe_lead_is_reported_as_indistinguishable():
    """The two recorded baselines differ by a single probe and the adoption rule
    calls that "beats the aggregate". Over the same sealed questions the honest
    test is the paired one, and a 10-9 split of 19 disagreements is a coin flip.
    A lopsided split must still come back significant, or the guard is inert."""
    assert eval_behavior._mcnemar_exact(0, 0) == 1.0
    assert eval_behavior._mcnemar_exact(10, 9) == 1.0        # the live v8/v5 split
    assert eval_behavior._mcnemar_exact(8, 0) < 0.05         # lopsided: a result
    assert eval_behavior._mcnemar_exact(15, 5) < 0.05
    assert eval_behavior._mcnemar_exact(6, 1) > 0.05         # small: not yet


def test_baseline_scorecard_counts_must_be_real_counts():
    """The comparison divides by n. A negative hits or hits>n yields a finite
    nonsense rate and the verdict prints ADOPT off it."""
    ok = {"by_category": {"a": {"hits": 1, "n": 2}}, "overall_hits": 1, "overall_n": 2}
    assert eval_behavior._valid_scorecard(ok) is True
    for bad in ({"by_category": {}, "overall_hits": -5, "overall_n": 1},
                {"by_category": {}, "overall_hits": 5, "overall_n": 1},
                {"by_category": {"a": {"hits": -1, "n": 2}}, "overall_hits": 0, "overall_n": 2},
                {"by_category": {"a": {"hits": 3, "n": 2}}, "overall_hits": 0, "overall_n": 2}):
        assert eval_behavior._valid_scorecard(bad) is False, bad


def test_only_gated_categories_reach_the_headline_aggregate(tmp_path, monkeypatch):
    """The aggregate is what --baseline compares and what "56/120" names, so a
    column that gates nothing must not carry the adoption verdict. Driven
    through run() with a mixed file, because the previous version of this test
    asserted two constants were disjoint and never touched the aggregate -- the
    change was completely unpinned."""
    probes = tmp_path / "probes.jsonl"
    probes.write_text(
        json.dumps({"category": "factual", "q": "Largest planet?",
                    "want_any": ["jupiter"]}) + "\n"
        + json.dumps({"category": "vision", "q": "Describe the picture.",
                      "want_any": ["nothing-matches-this"]}) + "\n",
        encoding="utf-8")
    out = tmp_path / "t.jsonl"
    _fake_server(monkeypatch, LONG_ANSWER)
    eval_behavior.run(URL, TEMP, MAXTOK, probes, out)

    card = [r for r in _rows(out) if r.get("record") == "scorecard"][0]
    # both probes were graded and both appear per-category...
    assert set(card["by_category"]) == {"factual", "vision"}
    assert sum(c["n"] for c in card["by_category"].values()) == 2
    # ...but only the gated one is aggregated
    assert card["overall_n"] == 1, "an informational column entered the headline"
    # and the record SAYS so, or a reader summing by_category cannot tell which
    # number is wrong
    assert card["overall_scope"] == "gated"
    assert card["informational_categories"] == ["vision"]


def test_a_baseline_counted_under_the_old_rule_is_refused(tmp_path, monkeypatch):
    """Before the gated-only rule the aggregate counted every category. The two
    populations differ whenever an organ column ran, so comparing across the
    change divides different denominators and reports the difference as the
    model -- and the probe-set check cannot see it, because the probe file is
    byte-identical. Refused outright rather than flagged: a warning beside a
    printed delta reads as a caveat on a real number."""
    rule = eval_behavior.baseline_aggregate_rule
    by_cat = {"factual": {"hits": 8, "n": 15}, "vision": {"hits": 12, "n": 12}}
    assert rule({"by_category": by_cat, "overall_hits": 8, "overall_n": 15}) == "gated"
    assert rule({"by_category": by_cat, "overall_hits": 20, "overall_n": 27}) == "all"
    assert rule({"by_category": by_cat, "overall_hits": 3, "overall_n": 4}) == "unknown"
    # with no informational column the two rules agree, so an older transcript
    # of a purely gated set stays usable -- which is every archived baseline
    gated_only = {"factual": {"hits": 8, "n": 15}, "math": {"hits": 9, "n": 15}}
    assert rule({"by_category": gated_only, "overall_hits": 17, "overall_n": 30}) == "gated"

    # end to end: run() must refuse before any server is touched
    old_rule = tmp_path / "old.jsonl"
    old_rule.write_text(
        json.dumps({"record": "run_conditions", "probe_sha256": "x", "probe_count": 2}) + "\n"
        + json.dumps({"record": "scorecard", "by_category": by_cat,
                      "overall_hits": 20, "overall_n": 27}) + "\n",
        encoding="utf-8")
    probes = _probe_file(tmp_path)
    touched = []
    monkeypatch.setattr(eval_behavior, "_clear_memory", lambda *a, **k: touched.append(1))
    rc = eval_behavior.run(URL, TEMP, MAXTOK, probes, tmp_path / "t.jsonl",
                           baseline=old_rule)
    assert rc == 2
    assert not touched, "the target's memory store was wiped before the refusal"


def test_a_baseline_that_measured_nothing_cannot_produce_a_verdict(tmp_path, monkeypatch, capsys):
    """A truncated or half-written receipt scores 0 of 0, and every count in it
    is a legal count. The comparison then INVENTED the denominator
    (`overall_n or 1`) and printed "aggregate 0/1 -> 1/1 (+100.0%)", "category
    regressions: none", "VERDICT: ADOPT" -- the strongest verdict this tool
    emits, off a baseline that measured nothing. A comparison against nothing
    is refused at load, by name."""
    probes = _probe_file(tmp_path)

    def touched(*a, **k):
        raise AssertionError("server contacted despite an empty baseline")

    monkeypatch.setattr(eval_behavior, "_wait_for_server", touched)
    monkeypatch.setattr(eval_behavior, "_clear_memory", touched)

    empty = {
        "nothing at all": {"by_category": {}, "overall_hits": 0, "overall_n": 0},
        "categories that graded nobody": {"by_category": {"factual": {"hits": 0, "n": 0}},
                                          "overall_hits": 0, "overall_n": 0},
        "an aggregate with no categories behind it": {"by_category": {},
                                                      "overall_hits": 1, "overall_n": 2},
    }
    for name, card in empty.items():
        base = tmp_path / "empty_base.jsonl"
        base.write_text(
            json.dumps({"record": "run_conditions",
                        "probe_sha256": eval_behavior._probe_digest(probes),
                        "temperature": TEMP, "max_tokens": MAXTOK}) + "\n"
            + json.dumps(dict(card, record="scorecard")) + "\n", encoding="utf-8")
        out = tmp_path / "t.jsonl"
        assert eval_behavior.run(URL, TEMP, MAXTOK, probes, out, baseline=base) == 2, name
        printed = capsys.readouterr().out
        assert "FAIL" in printed and str(base) in printed, name
        assert "VERDICT" not in printed, name
        assert not out.exists(), name

    # A baseline that actually measured something still compares.
    _fake_server(monkeypatch, "Jupiter is the largest planet.")
    base = _baseline_file(tmp_path, probes,
                          {"factual": {"hits": 0, "n": 1}, "tool": {"hits": 1, "n": 1}})
    eval_behavior.run(URL, TEMP, MAXTOK, probes, None, baseline=base)
    assert "VERDICT: ADOPT" in capsys.readouterr().out
