"""The autonomy lane, headless: salience rules, busy-deferral, the NO_REPLY
sentinel, file-drop detection and the thread's start/stop mechanics.

Every clock here is INJECTED. The wall clock decides quiet hours, so a test
that reads the real one is green by day and red between 23:00 and 08:00 --
which is exactly when a wake loop matters most.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime

from enigma_engine.core.wake import (
    FileDropSource,
    WakeLoop,
    heartbeat_prompt,
    is_no_reply,
    tick_allowed,
)

NOON = datetime(2026, 9, 1, 12, 0)
NIGHT = datetime(2026, 9, 1, 23, 30)


def _loop(**kw):
    """A loop with both clocks pinned; callers override what they are testing."""
    kw.setdefault("submit", lambda p: "NO_REPLY")
    kw.setdefault("announce", lambda t, k: None)
    kw.setdefault("busy", lambda: False)
    kw.setdefault("clock", lambda: 10_000.0)
    kw.setdefault("wall_clock", lambda: NOON)
    return WakeLoop(**kw)


def test_no_reply_suppressed_and_variants():
    assert is_no_reply("NO_REPLY") and is_no_reply(" no_reply \n") and is_no_reply("")
    assert not is_no_reply("The oven timer file just appeared.")


def test_quiet_hours_wrap():
    q = (23, 8)
    assert not tick_allowed(datetime(2026, 9, 1, 23, 30), 0.0, 10_000.0, 900, q)
    assert not tick_allowed(datetime(2026, 9, 2, 7, 59), 0.0, 10_000.0, 900, q)
    assert tick_allowed(datetime(2026, 9, 1, 12, 0), 0.0, 10_000.0, 900, q)


def test_cooldown_gates_even_at_noon():
    assert not tick_allowed(datetime(2026, 9, 1, 12, 0), 9_500.0, 10_000.0, 900, (23, 8))


def test_never_spoken_passes_the_cooldown():
    """Cold boot: no last-spoke mark must not read as "spoke a moment ago"."""
    assert tick_allowed(NOON, None, 10_000.0, 900, (23, 8))


def test_busy_defers_and_never_submits():
    calls = []
    loop = WakeLoop(submit=lambda p: calls.append(p) or "NO_REPLY",
                    announce=lambda t, k: None, busy=lambda: True)
    loop._handle_event((5, 0, "tick", "heartbeat"))   # direct, no thread
    assert calls == []


def test_reply_survives_to_announce_and_cooldown_arms():
    out = []
    loop = _loop(submit=lambda p: "The report file you dropped looks new.",
                 announce=lambda t, k: out.append((k, t)))
    loop._handle_event((1, 0, "file", "report.txt"))
    assert out and out[0][0] == "file"
    assert loop._last_spoke is not None


def test_no_reply_never_announces_and_never_arms_the_cooldown():
    out = []
    loop = _loop(submit=lambda p: "NO_REPLY", announce=lambda t, k: out.append(t))
    loop._handle_event((5, 0, "tick", "heartbeat"))
    assert out == []
    assert loop._last_spoke is None, "a silent heartbeat must not spend the cooldown"


def test_the_heartbeat_prompt_names_the_event_and_asks_for_silence():
    prompt = heartbeat_prompt("file", "report.txt")
    assert "report.txt" in prompt
    assert prompt.rstrip().endswith("NO_REPLY.")


def test_file_events_outrank_ticks_in_the_queue():
    loop = _loop()
    loop.post("tick", "heartbeat")
    loop.post("file", "report.txt")
    first = loop.queue.get_nowait()
    assert first[2] == "file", "a dropped file waited behind a routine tick"
    assert loop.queue.get_nowait()[2] == "tick"


def test_file_events_honor_quiet_hours_and_cooldown():
    """They bypass the INTERVAL, not the manners."""
    calls = []
    submit = lambda p: calls.append(p) or "something worth saying"  # noqa: E731
    night = _loop(submit=submit, wall_clock=lambda: NIGHT)
    night._handle_event((1, 0, "file", "report.txt"))
    assert calls == [], "she spoke during quiet hours"

    cooling = _loop(submit=submit)
    cooling._last_spoke = 9_500.0          # 500s ago, cooldown is 900
    cooling._handle_event((1, 0, "file", "report.txt"))
    assert calls == [], "she spoke inside her own cooldown"


def test_a_bare_tick_never_spends_the_model():
    """B3 measured: sft2 cannot emit the NO_REPLY sentinel -- it paraphrases the
    instruction instead, so every tick announced chatter (15 rows in ~90s). A
    heartbeat with nothing to review must not reach the model at all."""
    calls = []
    loop = _loop(submit=lambda p: calls.append(p) or "chatter about nothing")
    loop._handle_event((5, 0, "tick", "heartbeat"))
    assert calls == []


def test_heartbeat_ticks_true_keeps_the_older_posture():
    """The future posture, for when she can answer a heartbeat with silence."""
    calls = []
    loop = _loop(submit=lambda p: calls.append(p) or "NO_REPLY", heartbeat_ticks=True)
    loop._handle_event((5, 0, "tick", "heartbeat"))
    assert len(calls) == 1


def test_a_running_loop_with_ticks_disabled_never_calls_the_model():
    """The B3 defect's shape on a real thread: the timer keeps firing, the
    model is never spent.

    The monotonic clock here is REAL on purpose -- the pinned clock the other
    tests use never advances past next_tick, so no tick would ever fire and
    the test would pass without testing anything (caught by mutation).
    """
    calls = []
    loop = WakeLoop(
        submit=lambda p: calls.append(p) or "chatter",
        announce=lambda t, k: None,
        busy=lambda: False,
        interval_s=0.01,
        cooldown_s=0,
        poll_s=0.01,
        wall_clock=lambda: NOON,      # only the WALL clock is pinned (quiet hours)
    )
    loop.start()
    try:
        time.sleep(0.4)          # dozens of tick windows
    finally:
        loop.stop()
        loop.join(2)
    assert calls == [], f"a bare tick spent the model: {calls}"


def test_silencing_ticks_does_not_silence_real_events():
    calls = []
    loop = _loop(submit=lambda p: calls.append(p) or "the file looks new")
    loop._handle_event((1, 0, "file", "report.txt"))
    assert len(calls) == 1


def test_a_file_event_that_hits_a_busy_lane_comes_back_once():
    """A dropped TICK loses nothing; a dropped FILE event is a real thing she
    was told about and never mentioned. It gets ONE retry once the lane frees."""
    busy = [True]
    out = []
    loop = WakeLoop(
        submit=lambda p: "the report file looks new",
        announce=lambda t, k: out.append(k),
        busy=lambda: busy[0],
        interval_s=86_400, cooldown_s=0, poll_s=0.01,
        requeue_delay_s=0.4,      # the retry must land AFTER the lane frees
        wall_clock=lambda: NOON,
    )
    loop.start()
    try:
        loop.post("file", "report.txt")
        time.sleep(0.15)          # first attempt hits busy; retry is scheduled
        assert out == [], "she spoke over a busy generation"
        busy[0] = False           # lane frees before the retry arrives
        deadline = time.time() + 3
        while time.time() < deadline and not out:
            time.sleep(0.01)
    finally:
        loop.stop()
        loop.join(2)
    assert out == ["file"], f"the deferred file event never came back: {out}"


def test_a_file_event_gives_up_after_the_second_busy_hit():
    """One retry, not an unbounded loop against a permanently busy server."""
    out, calls = [], []
    loop = WakeLoop(
        submit=lambda p: calls.append(p) or "text",
        announce=lambda t, k: out.append(k),
        busy=lambda: True,                      # never frees
        interval_s=86_400, cooldown_s=0, poll_s=0.01,
        requeue_delay_s=0.05,
        wall_clock=lambda: NOON,
    )
    loop.start()
    try:
        time.sleep(0.05)
        loop.post("file", "report.txt")
        time.sleep(0.8)                         # room for the one retry and more
    finally:
        loop.stop()
        loop.join(2)
    assert calls == [] and out == []


def test_a_busy_tick_is_still_dropped_outright():
    """Ticks keep the old policy: a deferred heartbeat arriving after the reply
    it interrupted is worse than one that never happened."""
    posted = []
    loop = _loop(submit=lambda p: "x", busy=lambda: True, heartbeat_ticks=True)
    loop.post = lambda *a, **k: posted.append(a)  # type: ignore[method-assign]
    loop._handle_event((5, 0, "tick", "heartbeat"))
    assert posted == []


def test_file_events_bypass_the_interval():
    """A file event is handled without waiting for the next tick window."""
    calls = []
    loop = _loop(submit=lambda p: calls.append(p) or "NO_REPLY", interval_s=86_400)
    loop._handle_event((1, 0, "file", "report.txt"))
    assert len(calls) == 1


def test_file_source_is_silent_on_a_cold_boot_then_sees_one_new_file(tmp_path):
    now = [0.0]
    (tmp_path / "already_here.txt").write_text("old", encoding="utf-8")
    (tmp_path / "and_this_one.txt").write_text("old", encoding="utf-8")
    src = FileDropSource(tmp_path, poll_s=5, clock=lambda: now[0])

    assert src.poll() == [], "a folder that was already full announced itself"

    (tmp_path / "report.txt").write_text("new", encoding="utf-8")
    now[0] += 5
    assert src.poll() == ["report.txt"]

    now[0] += 5
    assert src.poll() == [], "the same file was announced twice"


def test_file_source_respects_its_poll_interval(tmp_path):
    now = [0.0]
    src = FileDropSource(tmp_path, poll_s=5, clock=lambda: now[0])
    src.poll()                                  # cold boot adopts the (empty) dir
    (tmp_path / "report.txt").write_text("new", encoding="utf-8")
    now[0] += 1                                 # too soon
    assert src.poll() == []
    now[0] += 4                                 # now due
    assert src.poll() == ["report.txt"]


def test_the_thread_starts_takes_an_injected_event_and_stops_clean():
    seen = threading.Event()
    loop = _loop(submit=lambda p: "the file you dropped is new",
                 announce=lambda t, k: seen.set(),
                 interval_s=86_400, poll_s=0.01)
    loop.start()
    try:
        loop.post("file", "report.txt")
        assert seen.wait(2.0), "the loop never handled an injected event"
    finally:
        loop.stop()
        loop.join(2.0)
    assert not loop.is_alive(), "the loop thread did not stop"
    assert loop.daemon, "the wake thread must never outlive the server"
