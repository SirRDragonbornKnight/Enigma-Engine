"""Tests for ``collect_distill_data.py`` — N-19 external-teacher data CLI."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

import collect_distill_data as cdd


# ── load_prompts ──────────────────────────────────────────────────────


def test_load_prompts_txt_skips_blank_and_comments(tmp_path: Path) -> None:
    p = tmp_path / "prompts.txt"
    p.write_text(
        "first prompt\n\n# this is a comment\nsecond prompt\n  third with whitespace  \n",
        encoding="utf-8",
    )
    prompts = cdd.load_prompts(p)
    assert prompts == ["first prompt", "second prompt", "third with whitespace"]


def test_load_prompts_jsonl_extracts_prompt_field(tmp_path: Path) -> None:
    p = tmp_path / "prompts.jsonl"
    p.write_text(
        json.dumps({"prompt": "alpha", "extra": "ignored"}) + "\n"
        "\n"  # blank line tolerated
        + "not-json-line\n"  # malformed → warned + skipped
        + json.dumps({"prompt": "beta"})
        + "\n"
        + json.dumps({"prompt": "   "})
        + "\n",  # whitespace-only → skipped
        encoding="utf-8",
    )
    prompts = cdd.load_prompts(p)
    assert prompts == ["alpha", "beta"]


def test_load_prompts_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        cdd.load_prompts(tmp_path / "nope.txt")


# ── _warn_if_remote ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "endpoint",
    ["http://localhost:11434/v1", "http://127.0.0.1:8080/v1", "http://[::1]:8000/v1"],
)
def test_warn_if_remote_silent_for_localhost(endpoint: str, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger=cdd.logger.name):
        cdd._warn_if_remote(endpoint)
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_warn_if_remote_logs_for_non_localhost(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger=cdd.logger.name):
        cdd._warn_if_remote("https://api.openai.com/v1")
    msgs = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("api.openai.com" in m and "NOT localhost" in m for m in msgs)


# ── resume ────────────────────────────────────────────────────────────


def test_load_done_keys_reads_existing_rows(tmp_path: Path) -> None:
    jsonl = tmp_path / "out.jsonl"
    jsonl.write_text(
        json.dumps({"prompt": "alpha", "completion": "A"})
        + "\n"
        + json.dumps({"prompt": "beta", "completion": "B"})
        + "\n"
        + "garbage line\n",
        encoding="utf-8",
    )
    done = cdd._load_done_keys(jsonl)
    assert cdd._prompt_key("alpha") in done
    assert cdd._prompt_key("beta") in done
    assert len(done) == 2


def test_load_done_keys_missing_file_returns_empty(tmp_path: Path) -> None:
    assert cdd._load_done_keys(tmp_path / "missing.jsonl") == set()


def test_prompt_key_stable() -> None:
    a = cdd._prompt_key("hello world")
    b = cdd._prompt_key("hello world")
    c = cdd._prompt_key("hello world!")
    assert a == b
    assert a != c
    assert len(a) == 16


# ── _rewrite_combined_text ────────────────────────────────────────────


def test_rewrite_combined_text_canonical_format(tmp_path: Path) -> None:
    jsonl = tmp_path / "src.jsonl"
    jsonl.write_text(
        json.dumps({"prompt": "Q1", "completion": "A1"})
        + "\n"
        + json.dumps({"prompt": "Q2", "completion": "A2"})
        + "\n"
        # Empty completion must be filtered out
        + json.dumps({"prompt": "Q3", "completion": "  "})
        + "\n",
        encoding="utf-8",
    )
    txt = tmp_path / "out.txt"
    written = cdd._rewrite_combined_text(jsonl, txt)
    assert written == 2
    body = txt.read_text(encoding="utf-8")
    assert body == "User: Q1\n\nAssistant: A1\n\nUser: Q2\n\nAssistant: A2\n"


# ── collect end-to-end with a fake client ─────────────────────────────


class _FakeClient:
    def __init__(self, replies: dict[str, str] | None = None, raises_on: set[str] | None = None) -> None:
        self.replies = replies or {}
        self.raises_on = raises_on or set()
        self.calls: list[str] = []

    def ask(self, prompt: str, **_: object) -> str:
        self.calls.append(prompt)
        if prompt in self.raises_on:
            raise RuntimeError(f"simulated upstream failure for {prompt!r}")
        return self.replies.get(prompt, f"reply-to-{prompt}")


def test_collect_writes_jsonl_and_txt(tmp_path: Path) -> None:
    client = _FakeClient()
    summary = cdd.collect(
        endpoint="http://localhost:9999/v1",
        model="fake",
        prompts=["one", "two", "three"],
        tag="t1",
        output_dir=tmp_path,
        client=client,
    )
    assert summary["ok"] == 3
    assert summary["failed"] == 0
    assert summary["skipped_done"] == 0
    jsonl = tmp_path / "distill_t1.jsonl"
    txt = tmp_path / "distill_t1.txt"
    assert jsonl.exists() and txt.exists()
    rows = [json.loads(ln) for ln in jsonl.read_text("utf-8").splitlines() if ln.strip()]
    assert {r["prompt"] for r in rows} == {"one", "two", "three"}
    body = txt.read_text("utf-8")
    assert body.count("User: ") == 3
    assert body.count("Assistant: ") == 3


def test_collect_failed_prompt_does_not_abort_run(tmp_path: Path) -> None:
    client = _FakeClient(raises_on={"two"})
    summary = cdd.collect(
        endpoint="http://localhost:9999/v1",
        model="fake",
        prompts=["one", "two", "three"],
        tag="t2",
        output_dir=tmp_path,
        client=client,
    )
    assert summary["ok"] == 2
    assert summary["failed"] == 1
    assert client.calls == ["one", "two", "three"]
    rows = [json.loads(ln) for ln in (tmp_path / "distill_t2.jsonl").read_text("utf-8").splitlines() if ln.strip()]
    assert {r["prompt"] for r in rows} == {"one", "three"}


def test_collect_empty_completion_counted_as_failure(tmp_path: Path) -> None:
    client = _FakeClient(replies={"two": "   "})
    summary = cdd.collect(
        endpoint="http://localhost:9999/v1",
        model="fake",
        prompts=["one", "two"],
        tag="t3",
        output_dir=tmp_path,
        client=client,
    )
    assert summary["ok"] == 1
    assert summary["failed"] == 1


def test_collect_resume_skips_done_prompts_without_calling_client(
    tmp_path: Path,
) -> None:
    # Pre-seed two rows, then re-run with all three prompts + resume=True.
    jsonl = tmp_path / "distill_t4.jsonl"
    jsonl.write_text(
        json.dumps({"prompt": "one", "completion": "old-A"})
        + "\n"
        + json.dumps({"prompt": "two", "completion": "old-B"})
        + "\n",
        encoding="utf-8",
    )
    client = _FakeClient()
    summary = cdd.collect(
        endpoint="http://localhost:9999/v1",
        model="fake",
        prompts=["one", "two", "three"],
        tag="t4",
        output_dir=tmp_path,
        resume=True,
        client=client,
    )
    # Only the missing prompt was sent.
    assert client.calls == ["three"]
    assert summary["ok"] == 1
    assert summary["skipped_done"] == 2
    # Final JSONL has all three (two old + one new).
    rows = [json.loads(ln) for ln in jsonl.read_text("utf-8").splitlines() if ln.strip()]
    assert {r["prompt"] for r in rows} == {"one", "two", "three"}


def test_collect_overwrites_existing_when_resume_false(tmp_path: Path) -> None:
    jsonl = tmp_path / "distill_t5.jsonl"
    jsonl.write_text(
        json.dumps({"prompt": "stale", "completion": "X"}) + "\n",
        encoding="utf-8",
    )
    client = _FakeClient()
    cdd.collect(
        endpoint="http://localhost:9999/v1",
        model="fake",
        prompts=["fresh"],
        tag="t5",
        output_dir=tmp_path,
        resume=False,
        client=client,
    )
    rows = [json.loads(ln) for ln in jsonl.read_text("utf-8").splitlines() if ln.strip()]
    assert [r["prompt"] for r in rows] == ["fresh"]


# ── TeacherClient HTTP error paths (no real network) ──────────────────


def test_teacher_client_builds_url_correctly() -> None:
    c = cdd.TeacherClient(endpoint="http://localhost:11434/v1/", model="m")
    assert c._url == "http://localhost:11434/v1/chat/completions"


def test_teacher_client_unreachable_raises_runtimeerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import urllib.error

    def boom(*_a: object, **_kw: object) -> object:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    c = cdd.TeacherClient(endpoint="http://localhost:1/v1", model="m")
    with pytest.raises(RuntimeError, match="unreachable"):
        c.ask("hi")


def test_teacher_client_http_error_raises_runtimeerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import io
    import urllib.error

    def boom(*_a: object, **_kw: object) -> object:
        raise urllib.error.HTTPError(
            url="http://localhost/v1/chat/completions",
            code=404,
            msg="Not Found",
            hdrs=None,  # type: ignore[arg-type]
            fp=io.BytesIO(b'{"error":"model not found"}'),
        )

    monkeypatch.setattr("urllib.request.urlopen", boom)
    c = cdd.TeacherClient(endpoint="http://localhost/v1", model="m")
    with pytest.raises(RuntimeError, match="HTTP 404"):
        c.ask("hi")


# ── Magpie templates + parser ─────────────────────────────────────────


def test_resolve_magpie_template_known_chatml() -> None:
    t = cdd._resolve_magpie_template("chatml")
    assert t["prefix"].startswith("<|im_start|>user")
    assert "assistant" in t["user_end"]
    assert t["assistant_end"] == "<|im_end|>"


def test_resolve_magpie_template_known_llama3() -> None:
    t = cdd._resolve_magpie_template("llama3")
    assert "begin_of_text" in t["prefix"]
    assert "<|eot_id|>" in t["user_end"]
    assert t["assistant_end"] == "<|eot_id|>"


def test_resolve_magpie_template_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown magpie template"):
        cdd._resolve_magpie_template("not-a-real-template")


def test_resolve_magpie_template_custom_requires_all_three() -> None:
    # Missing any of the three custom strings must fail loud at parse
    # time, not silently produce broken templates after a long run.
    with pytest.raises(ValueError, match="--magpie-prefix"):
        cdd._resolve_magpie_template(
            "custom",
            custom_prefix="",
            custom_user_end="X",
            custom_assistant_end="Y",
        )
    with pytest.raises(ValueError, match="--magpie-user-end"):
        cdd._resolve_magpie_template(
            "custom",
            custom_prefix="A",
            custom_user_end=None,
            custom_assistant_end="Y",
        )
    with pytest.raises(ValueError, match="--magpie-assistant-end"):
        cdd._resolve_magpie_template(
            "custom",
            custom_prefix="A",
            custom_user_end="X",
            custom_assistant_end="",
        )


def test_resolve_magpie_template_custom_happy() -> None:
    t = cdd._resolve_magpie_template(
        "custom",
        custom_prefix="P",
        custom_user_end="U",
        custom_assistant_end="A",
    )
    assert t == {"prefix": "P", "user_end": "U", "assistant_end": "A"}


def test_parse_magpie_completion_chatml_happy() -> None:
    template = cdd._MAGPIE_TEMPLATES["chatml"]
    raw = (
        "Write a haiku about cats."
        "<|im_end|>\n<|im_start|>assistant\n"
        "Whiskers in moonlight\nSilent paws upon the mat\nDreams of distant prey"
        "<|im_end|>"
    )
    instr, ans = cdd._parse_magpie_completion(raw, template)
    assert instr == "Write a haiku about cats."
    assert ans.startswith("Whiskers in moonlight")
    assert "<|im_end|>" not in ans  # marker stripped


def test_parse_magpie_completion_missing_marker_raises() -> None:
    """Wrong template / base model with no chat training → no
    user→assistant marker → fail loud, count as failed, batch
    continues. Otherwise the user gets a JSONL full of raw model
    output mislabelled as instruction-tuning data."""
    template = cdd._MAGPIE_TEMPLATES["chatml"]
    # Llama3 markers in chatml-template parse → marker missing
    raw = "Write a poem.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    with pytest.raises(RuntimeError, match="marker not found"):
        cdd._parse_magpie_completion(raw, template)


def test_parse_magpie_completion_empty_instruction_raises() -> None:
    """Model emitted only the assistant marker without first proposing
    a user instruction — the resulting row would be JSONL with empty
    prompt, polluting the corpus."""
    template = cdd._MAGPIE_TEMPLATES["chatml"]
    raw = "   \n\n<|im_end|>\n<|im_start|>assistant\nSome reply<|im_end|>"
    with pytest.raises(RuntimeError, match="empty user instruction"):
        cdd._parse_magpie_completion(raw, template)


def test_parse_magpie_completion_empty_answer_raises() -> None:
    """Model proposed an instruction but ran out of tokens before
    answering — empty completion, useless training row."""
    template = cdd._MAGPIE_TEMPLATES["chatml"]
    raw = "Explain entropy.<|im_end|>\n<|im_start|>assistant\n   "
    with pytest.raises(RuntimeError, match="empty assistant answer"):
        cdd._parse_magpie_completion(raw, template)


# ── Magpie driver ─────────────────────────────────────────────────────


class _FakeMagpieClient:
    """Mimics MagpieClient.ask_pair without HTTP. Pops scripted pairs
    or raises scripted RuntimeErrors so we can test failure handling."""

    def __init__(self, scripted: list) -> None:
        # Each entry is either (instruction, answer) or an Exception
        # instance to raise.
        self._scripted = list(scripted)
        self.calls: list[dict] = []

    def ask_pair(self, **kwargs: object) -> tuple[str, str]:
        self.calls.append(dict(kwargs))
        if not self._scripted:
            raise RuntimeError("test ran out of scripted responses")
        item = self._scripted.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_magpie_collect_writes_jsonl_and_txt(tmp_path: Path) -> None:
    fake = _FakeMagpieClient([("What is entropy?", "A measure of disorder."), ("Name a prime.", "Seven.")])
    summary = cdd.magpie_collect(
        endpoint="http://localhost:11434/v1",
        model="m",
        n=2,
        tag="t",
        output_dir=tmp_path,
        client=fake,
    )
    assert summary["ok"] == 2
    assert summary["failed"] == 0
    assert summary["duplicate"] == 0
    jsonl = tmp_path / "distill_t.jsonl"
    assert jsonl.exists()
    rows = [json.loads(ln) for ln in jsonl.read_text("utf-8").splitlines() if ln]
    assert rows[0] == {"prompt": "What is entropy?", "completion": "A measure of disorder."}
    txt = (tmp_path / "distill_t.txt").read_text("utf-8")
    assert "User: What is entropy?" in txt
    assert "Assistant: A measure of disorder." in txt


def test_magpie_collect_rejects_non_positive_n(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="n >= 1"):
        cdd.magpie_collect(
            endpoint="http://localhost:11434/v1",
            model="m",
            n=0,
            tag="t",
            output_dir=tmp_path,
            client=_FakeMagpieClient([]),
        )


def test_magpie_collect_failed_pair_does_not_abort(tmp_path: Path) -> None:
    """A flaky teacher / parse failure on one call must not kill the
    batch — same robustness contract as prompts-driven collect()."""
    fake = _FakeMagpieClient(
        [RuntimeError("magpie parse failed: marker not found"), ("Define gravity.", "Mass attracts mass.")]
    )
    summary = cdd.magpie_collect(
        endpoint="http://localhost:11434/v1",
        model="m",
        n=2,
        tag="t",
        output_dir=tmp_path,
        client=fake,
    )
    assert summary["ok"] == 1
    assert summary["failed"] == 1
    rows = [json.loads(ln) for ln in (tmp_path / "distill_t.jsonl").read_text("utf-8").splitlines() if ln]
    assert rows[0]["prompt"] == "Define gravity."


def test_magpie_collect_deduplicates_repeated_instructions(tmp_path: Path) -> None:
    """When the model samples a duplicate instruction (same string),
    the second hit must be dropped not appended — otherwise the JSONL
    accumulates duplicate prompts that bias training."""
    fake = _FakeMagpieClient([("What is X?", "A unique thing."), ("What is X?", "Different answer, same instruction.")])
    summary = cdd.magpie_collect(
        endpoint="http://localhost:11434/v1",
        model="m",
        n=2,
        tag="t",
        output_dir=tmp_path,
        client=fake,
    )
    assert summary["ok"] == 1
    assert summary["duplicate"] == 1


def test_magpie_collect_warns_on_high_failure_rate(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Failure rate >50% almost always means the wrong --template
    was chosen for the model. Run summary must surface this so the
    user doesn't keep banging on a wrong setup."""
    fake = _FakeMagpieClient(
        [
            RuntimeError("magpie parse failed: marker not found"),
            RuntimeError("magpie parse failed: marker not found"),
            ("OK instruction.", "OK answer."),
        ]
    )
    with caplog.at_level(logging.WARNING):
        cdd.magpie_collect(
            endpoint="http://localhost:11434/v1",
            model="m",
            n=3,
            tag="t",
            output_dir=tmp_path,
            client=fake,
        )
    assert any("failure rate" in rec.message.lower() and "template" in rec.message.lower() for rec in caplog.records), (
        "high-failure-rate warning missing — user has no signal that their --template choice is wrong"
    )


def test_magpie_collect_resume_skips_already_present_instructions(
    tmp_path: Path,
) -> None:
    """Resume contract: instructions already in the JSONL are not
    re-appended even if the model samples them again. Important
    because Magpie can converge on a few popular instructions when
    temperature is low — without dedup the corpus duplicates."""
    jsonl = tmp_path / "distill_t.jsonl"
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    jsonl.write_text(
        json.dumps({"prompt": "Already seen.", "completion": "Old answer."}) + "\n",
        encoding="utf-8",
    )
    fake = _FakeMagpieClient([("Already seen.", "Different answer."), ("New question.", "New answer.")])
    summary = cdd.magpie_collect(
        endpoint="http://localhost:11434/v1",
        model="m",
        n=2,
        tag="t",
        output_dir=tmp_path,
        resume=True,
        client=fake,
    )
    assert summary["ok"] == 1
    assert summary["duplicate"] == 1
    rows = [json.loads(ln) for ln in jsonl.read_text("utf-8").splitlines() if ln]
    assert len(rows) == 2
    assert rows[1]["prompt"] == "New question."


# ── Magpie HTTP client ────────────────────────────────────────────────


def test_magpie_client_builds_completions_url() -> None:
    """Magpie hits /v1/completions, NOT /v1/chat/completions — raw
    template control is the whole point. URL regression here would
    silently fall through to chat completions and break the technique."""
    template = cdd._MAGPIE_TEMPLATES["chatml"]
    c = cdd.MagpieClient(endpoint="http://localhost:11434/v1/", model="m", template=template)
    assert c._url == "http://localhost:11434/v1/completions"
    assert "chat/completions" not in c._url


def test_magpie_client_sends_template_prefix_and_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wire-test: the body sent over HTTP must contain (1) the
    template's exact prefix string in the 'prompt' field and (2) the
    assistant_end marker in the 'stop' list. A regression that forgets
    the stop string would let the model bleed into a hallucinated
    second turn that the parser then chokes on."""
    template = cdd._MAGPIE_TEMPLATES["chatml"]
    captured: dict = {}

    class _FakeResp:
        def __enter__(self_) -> "_FakeResp":
            return self_

        def __exit__(self_, *args: object) -> None:
            pass

        def read(self_) -> bytes:
            return json.dumps({"choices": [{"text": "Hello.<|im_end|>\n<|im_start|>assistant\nHi.<|im_end|>"}]}).encode(
                "utf-8"
            )

    def fake_urlopen(req: object, timeout: float = 0.0) -> _FakeResp:
        _ = timeout
        captured["body"] = json.loads(req.data.decode("utf-8"))  # type: ignore[attr-defined]
        return _FakeResp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    c = cdd.MagpieClient(endpoint="http://localhost:11434/v1", model="m", template=template)
    instr, ans = c.ask_pair(max_tokens=128, temperature=1.0)
    assert instr == "Hello."
    assert ans == "Hi."
    assert captured["body"]["prompt"] == template["prefix"]
    assert template["assistant_end"] in captured["body"].get("stop", [])


def test_magpie_client_unreachable_raises_runtimeerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import urllib.error

    def boom(*_a: object, **_kw: object) -> object:
        raise urllib.error.URLError("refused")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    template = cdd._MAGPIE_TEMPLATES["chatml"]
    c = cdd.MagpieClient(endpoint="http://localhost:1/v1", model="m", template=template)
    with pytest.raises(RuntimeError, match="unreachable"):
        c.ask_pair()


# ── CLI mutual-exclusion ──────────────────────────────────────────────


def test_main_rejects_both_prompts_and_magpie(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """--prompts and --magpie are different paradigms (one reads
    instructions from a file, the other generates them). Asking for
    both is a misconfiguration the CLI must flag, not silently pick
    one and ignore the other."""
    prompts_file = tmp_path / "p.txt"
    prompts_file.write_text("hello\n", encoding="utf-8")
    with caplog.at_level(logging.ERROR):
        rc = cdd.main(
            [
                "--model",
                "m",
                "--prompts",
                str(prompts_file),
                "--magpie",
                "5",
            ]
        )
    assert rc == 2
    assert any("mutually exclusive" in rec.message.lower() for rec in caplog.records)


def test_main_rejects_neither_prompts_nor_magpie(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.ERROR):
        rc = cdd.main(["--model", "m"])
    assert rc == 2
    assert any("--prompts" in rec.message and "--magpie" in rec.message for rec in caplog.records)


# ── CLI temperature default per mode (Pass 156z9b) ────────────────────


def test_main_magpie_unset_temperature_defaults_to_one_point_zero(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Magpie's empty-prefix instruction synthesis needs higher diversity
    than the prompts-mode default of 0.7. Per arxiv:2406.08464 the
    paper recommends T=1.0; the CLI must default to that on the Magpie
    path when --temperature is not explicitly set, and log a visible
    INFO line so the user sees which value is in effect."""
    captured: dict[str, object] = {}

    def fake_magpie_collect(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"ok": 1, "failed": 0, "skipped": 0, "out_jsonl": "x", "out_text": "x"}

    monkeypatch.setattr(cdd, "magpie_collect", fake_magpie_collect)
    with caplog.at_level(logging.INFO):
        rc = cdd.main(["--model", "m", "--magpie", "1"])
    assert rc == 0
    assert captured["temperature"] == 1.0
    assert any("1.0" in rec.message and "magpie" in rec.message.lower() for rec in caplog.records)


def test_main_magpie_explicit_temperature_is_passed_through(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """User-set --temperature must override the per-mode default with
    no INFO hint (the hint only fires when defaulting)."""
    captured: dict[str, object] = {}

    def fake_magpie_collect(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"ok": 1, "failed": 0, "skipped": 0, "out_jsonl": "x", "out_text": "x"}

    monkeypatch.setattr(cdd, "magpie_collect", fake_magpie_collect)
    with caplog.at_level(logging.INFO):
        rc = cdd.main(["--model", "m", "--magpie", "1", "--temperature", "0.5"])
    assert rc == 0
    assert captured["temperature"] == 0.5
    assert not any("paper recommendation" in rec.message for rec in caplog.records)


def test_main_prompts_unset_temperature_defaults_to_zero_point_seven(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Prompts-mode default stays at 0.7 (preserving prior behaviour
    after the per-mode resolution refactor)."""
    captured: dict[str, object] = {}

    def fake_collect(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"ok": 1, "failed": 0, "skipped": 0, "out_jsonl": "x", "out_text": "x"}

    monkeypatch.setattr(cdd, "collect", fake_collect)
    prompts_file = tmp_path / "p.txt"
    prompts_file.write_text("hello\n", encoding="utf-8")
    rc = cdd.main(["--model", "m", "--prompts", str(prompts_file)])
    assert rc == 0
    assert captured["temperature"] == 0.7
