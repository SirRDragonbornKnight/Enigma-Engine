"""Serve's half of the wake loop: the five flags (all inert by default), the
announce path, and the /v1/wake/recent feed.

The live ON boot is the orchestrator's smoke; everything here is headless.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import serve_enigma as serve

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def client(monkeypatch):
    """The app answering as a booted server, without booting one."""
    monkeypatch.setattr(serve, "_BOOTED", True)
    return TestClient(serve.app)


# --- flags -----------------------------------------------------------------

def test_the_five_wake_flags_exist_and_default_to_inert():
    args = serve._p.parse_known_args([])[0]
    assert args.wake is False, "--wake must ship OFF; her first wake is the user's to switch on"
    assert args.wake_interval == 1800
    assert args.wake_watch is None
    assert args.wake_cooldown == 900
    assert args.wake_quiet == (23, 8)


def test_wake_quiet_parses_hour_dash_hour():
    assert serve._p.parse_known_args(["--wake-quiet", "22-7"])[0].wake_quiet == (22, 7)
    assert serve._p.parse_known_args(["--wake-quiet", "0-0"])[0].wake_quiet == (0, 0)


@pytest.mark.parametrize("bad", ["23", "23-", "23-8-9", "x-8", "24-8", "23-99", ""])
def test_wake_quiet_refuses_a_malformed_window(bad):
    with pytest.raises(SystemExit):
        serve._p.parse_known_args(["--wake-quiet", bad])


def test_wake_flags_are_accepted_together():
    args = serve._p.parse_known_args(
        ["--wake", "--wake-interval", "5", "--wake-watch", "C:/tmp", "--wake-cooldown", "0", "--wake-quiet", "1-2"]
    )[0]
    assert args.wake and args.wake_interval == 5 and args.wake_cooldown == 0


# --- the log's home --------------------------------------------------------

def test_the_wake_log_lives_in_her_data_home_never_the_repo():
    path = serve._wake_log_path()
    assert path.name == "wake_log.jsonl"
    assert path.parent == serve.PERSONA.home
    assert REPO_ROOT not in path.parents, "the wake log must never be written into the repo"


# --- announce --------------------------------------------------------------

class _FakeSpeaker:
    def __init__(self):
        self.said = []

    def speak(self, text):
        self.said.append(text)


def _log_to(monkeypatch, tmp_path):
    log = tmp_path / "wake_log.jsonl"
    monkeypatch.setattr(serve, "_wake_log_path", lambda: log)
    return log


def test_announce_appends_the_row_it_promises(monkeypatch, tmp_path):
    log = _log_to(monkeypatch, tmp_path)
    monkeypatch.setattr(serve, "SPEAKER", None)
    serve._wake_announce("the report file looks new", "file")
    serve._wake_announce("second thing", "tick")
    rows = [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert [r["kind"] for r in rows] == ["file", "tick"]
    assert rows[0]["text"] == "the report file looks new"
    assert isinstance(rows[0]["ts"], (int, float))


def test_announce_speaks_only_when_talk_mode_is_on_and_she_is_not_muted(monkeypatch, tmp_path):
    _log_to(monkeypatch, tmp_path)
    speaker = _FakeSpeaker()
    monkeypatch.setattr(serve, "SPEAKER", speaker)

    monkeypatch.setattr(serve, "TALK_MODE", False)
    monkeypatch.setattr(serve, "MUTED", False)
    serve._wake_announce("talk mode off", "tick")
    assert speaker.said == []

    monkeypatch.setattr(serve, "TALK_MODE", True)
    monkeypatch.setattr(serve, "MUTED", True)
    serve._wake_announce("muted", "tick")
    assert speaker.said == [], "she spoke out loud while muted"

    monkeypatch.setattr(serve, "MUTED", False)
    serve._wake_announce("out loud", "tick")
    assert speaker.said == ["out loud"]


def test_announce_still_logs_when_the_voice_is_off(monkeypatch, tmp_path):
    """The log is the record; the mouth is optional."""
    log = _log_to(monkeypatch, tmp_path)
    monkeypatch.setattr(serve, "SPEAKER", None)
    monkeypatch.setattr(serve, "TALK_MODE", True)
    serve._wake_announce("written even with no voice", "tick")
    assert "written even with no voice" in log.read_text(encoding="utf-8")


# --- the feed --------------------------------------------------------------

def test_wake_recent_is_empty_on_a_cold_boot(client, monkeypatch, tmp_path):
    """The route exists even with --wake off, and a missing log is [] not 500."""
    monkeypatch.setattr(serve, "_wake_log_path", lambda: tmp_path / "nothing_here.jsonl")
    r = client.get("/v1/wake/recent")
    assert r.status_code == 200
    assert r.json() == []


def test_wake_recent_returns_the_last_n_rows(client, monkeypatch, tmp_path):
    log = _log_to(monkeypatch, tmp_path)
    with log.open("w", encoding="utf-8") as fh:
        for i in range(30):
            fh.write(json.dumps({"ts": i, "kind": "tick", "text": f"row {i}"}) + "\n")
    body = client.get("/v1/wake/recent").json()
    assert len(body) == 20 and body[-1]["text"] == "row 29"
    assert client.get("/v1/wake/recent", params={"n": 3}).json()[0]["text"] == "row 27"


def test_wake_recent_clamps_a_negative_n(client, monkeypatch, tmp_path):
    """Receipt: n=-3 returned the WHOLE log. rows[-max(0, -3):] is rows[0:] --
    the same negative-slice trap memory search closed with `if k <= 0`."""
    log = _log_to(monkeypatch, tmp_path)
    with log.open("w", encoding="utf-8") as fh:
        for i in range(5):
            fh.write(json.dumps({"ts": i, "kind": "tick", "text": f"row {i}"}) + "\n")
    assert client.get("/v1/wake/recent", params={"n": -3}).json() == []
    assert client.get("/v1/wake/recent", params={"n": 0}).json() == []


def test_wake_recent_skips_a_corrupt_line_instead_of_failing(client, monkeypatch, tmp_path):
    log = _log_to(monkeypatch, tmp_path)
    log.write_text(
        json.dumps({"ts": 1, "kind": "tick", "text": "good"}) + "\n{not json\n"
        + json.dumps({"ts": 2, "kind": "file", "text": "also good"}) + "\n",
        encoding="utf-8",
    )
    body = client.get("/v1/wake/recent").json()
    assert [r["text"] for r in body] == ["good", "also good"]


# --- off by default --------------------------------------------------------

def test_wake_off_constructs_no_loop():
    """A server that was never asked to wake carries no loop at all."""
    assert getattr(serve.app.state, "wake_loop", None) is None


# --- boot must SAY whether file events can fire ----------------------------

def test_boot_warns_when_wake_has_nothing_to_watch():
    """The B3 root cause: the loop ran with no watch dir, so file events were
    impossible while ticks fired, and the boot output never said so."""
    args = serve._p.parse_known_args(["--wake"])[0]
    lines = serve._wake_status_lines(args)
    assert any("no --wake-watch" in ln and "WARN" in ln for ln in lines), lines


def test_boot_warns_when_the_watch_dir_is_not_there(tmp_path):
    args = serve._p.parse_known_args(["--wake", "--wake-watch", str(tmp_path / "nope")])[0]
    lines = serve._wake_status_lines(args)
    assert any("WARN" in ln and "not a directory" in ln for ln in lines), lines


def test_boot_names_the_folder_it_is_really_watching(tmp_path):
    args = serve._p.parse_known_args(["--wake", "--wake-watch", str(tmp_path)])[0]
    lines = serve._wake_status_lines(args)
    assert any(str(tmp_path) in ln and "WARN" not in ln for ln in lines), lines
    assert not any("WARN" in ln for ln in lines)


def test_boot_states_that_ticks_do_not_call_the_model(tmp_path):
    args = serve._p.parse_known_args(["--wake", "--wake-watch", str(tmp_path)])[0]
    assert any("ticks do NOT call the model" in ln for ln in serve._wake_status_lines(args))


def test_a_second_boot_stops_the_first_wake_loop():
    """boot() is re-entrant. Without this the old thread kept its queue, its
    timers and its mouth while a new one started beside it -- two loops
    announcing into one log."""
    from enigma_engine.core.wake import WakeLoop

    first = WakeLoop(submit=lambda p: "x", announce=lambda t, k: None,
                     busy=lambda: False, interval_s=86_400, poll_s=0.01)
    first.start()
    serve.app.state.wake_loop = first
    try:
        serve._stop_wake_loop(serve.app)
        assert not first.is_alive(), "the previous wake loop is still running"
        assert getattr(serve.app.state, "wake_loop", None) is None
        serve._stop_wake_loop(serve.app)          # idempotent on a clean state
    finally:
        first.stop()
        first.join(2)
        serve.app.state.wake_loop = None


# --- end to end, the way boot() actually builds it -------------------------

def _loop_as_boot_builds_it(argv, **fakes):
    """Construct WakeLoop from parsed ARGS exactly as boot() does.

    The unit tests exercise FileDropSource and WakeLoop separately, which is
    how a real dropped file went missing end to end while every test stayed
    green: the construction path itself was never covered.
    """
    from enigma_engine.core.wake import WakeLoop

    args = serve._p.parse_known_args(argv)[0]
    return WakeLoop(
        submit=fakes["submit"],
        announce=fakes["announce"],
        busy=fakes.get("busy", lambda: False),
        interval_s=args.wake_interval,
        cooldown_s=args.wake_cooldown,
        quiet=args.wake_quiet,
        watch_dir=args.wake_watch,
        heartbeat_ticks=False,
        poll_s=0.05,
    )


def test_a_dropped_file_reaches_announce_through_boots_own_construction(tmp_path):
    seen = []
    loop = _loop_as_boot_builds_it(
        ["--wake", "--wake-watch", str(tmp_path), "--wake-interval", "1",
         "--wake-cooldown", "0", "--wake-quiet", "0-0"],
        submit=lambda p: "the report file you dropped looks new",
        announce=lambda t, k: seen.append((k, t)),
    )
    loop.start()
    try:
        deadline = time.time() + 8
        (tmp_path / "report.txt").write_text("new", encoding="utf-8")
        while time.time() < deadline and not any(k == "file" for k, _ in seen):
            time.sleep(0.05)
    finally:
        loop.stop()
        loop.join(3)
    kinds = [k for k, _ in seen]
    assert "file" in kinds, f"the dropped file never became an event; got {kinds}"


def test_bare_ticks_stay_silent_end_to_end(tmp_path):
    """With heartbeat_ticks off, a watched folder nothing lands in produces
    NOTHING -- the smoke's 15 rows of tick chatter cannot recur."""
    calls = []
    loop = _loop_as_boot_builds_it(
        ["--wake", "--wake-watch", str(tmp_path), "--wake-interval", "1",
         "--wake-cooldown", "0", "--wake-quiet", "0-0"],
        submit=lambda p: calls.append(p) or "chatter",
        announce=lambda t, k: calls.append(("announce", t)),
    )
    loop.start()
    try:
        time.sleep(3)          # several tick windows go by
    finally:
        loop.stop()
        loop.join(3)
    assert calls == [], f"a bare tick spent the model anyway: {calls}"
