"""The autonomy lane: she notices things and may speak without being asked.

One thread beside the request lane, never inside it. Sources push events onto a
priority queue (a dropped file outranks a routine tick); cheap RULES decide
whether the moment is even eligible before the model is spent on it; the model
then gets one turn whose expected answer is the literal NO_REPLY sentinel, and
NO_REPLY is swallowed. Silence is the cheap path and the common one.

Three manners, all enforced here rather than trusted to the prompt:
  - BUSY-DEFERRAL: a tick arriving mid-generation is dropped, never queued
    against the user. The reactive lane always wins.
  - QUIET HOURS: a window that WRAPS midnight (23..8 is the default).
  - COOLDOWN: she does not speak twice in quick succession. Only a turn she
    actually SPOKE arms it -- a silent heartbeat costs nothing.

Nothing here imports torch, serve, or the model: the caller passes in its own
mouth (`announce`), its own generation path (`submit`) and its own notion of
busy, so the whole lane is testable headless with injected clocks.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from datetime import datetime
from itertools import count
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Lower number wins. A file the user just dropped is about NOW; a timer tick is
# about nothing in particular.
PRIORITY_FILE = 1
PRIORITY_TICK = 5
_PRIORITIES = {"file": PRIORITY_FILE, "tick": PRIORITY_TICK}

# The sentinel and the instruction that earns it. Kept as one literal so the
# prompt and the check can never drift apart.
NO_REPLY = "NO_REPLY"
HEARTBEAT_SUFFIX = "\nIf nothing needs attention, reply exactly NO_REPLY."


def is_no_reply(text: Optional[str]) -> bool:
    """True when she chose silence. An empty reply counts: a model that emitted
    nothing has said nothing, and shipping "" to the speaker is noise."""
    return (text or "").strip().casefold() == NO_REPLY.casefold() or not (text or "").strip()


def heartbeat_prompt(kind: str, detail: str) -> str:
    """The single user turn a wake event becomes."""
    if kind == "file":
        body = f"A new file just appeared in the folder you watch: {detail}. Say something only if it is worth her attention."
    else:
        body = "Nothing in particular happened. Say something only if there is something worth saying right now."
    return body + HEARTBEAT_SUFFIX


def in_quiet_hours(now_wall: datetime, quiet) -> bool:
    """Quiet hours as a half-open [start, end) window on the hour, WRAPPING
    midnight when start > end (23..8 is the whole night, not an empty set).
    start == end disables the window rather than silencing her forever."""
    if not quiet:
        return False
    start, end = int(quiet[0]), int(quiet[1])
    if start == end:
        return False
    hour = now_wall.hour
    if start > end:
        return hour >= start or hour < end
    return start <= hour < end


def tick_allowed(now_wall, last_spoke_mono, now_mono, cooldown_s, quiet) -> bool:
    """The cheap gate, run BEFORE the model is spent on anything.

    ``last_spoke_mono`` is None until she has actually spoken once -- which must
    read as "free to speak", not as "spoke at time zero".
    """
    if in_quiet_hours(now_wall, quiet):
        return False
    if last_spoke_mono is not None and (now_mono - last_spoke_mono) < cooldown_s:
        return False
    return True


class FileDropSource:
    """Emits one event per file that is NEW since the last look.

    Cold boot adopts whatever is already there and says nothing: a watch folder
    with two hundred old files in it is not two hundred things to announce.
    Name-based, no mtime heuristics and no new dependency -- a rename in is a
    new file, which is exactly what a drop looks like.
    """

    def __init__(self, watch_dir, poll_s: float = 5, clock: Callable[[], float] = time.monotonic):
        self.watch_dir = Path(watch_dir)
        self.poll_s = float(poll_s)
        self._clock = clock
        self._seen: Optional[set[str]] = None  # None = the cold boot has not happened yet
        self._last_poll = float("-inf")

    def _names(self) -> set[str]:
        try:
            return {p.name for p in self.watch_dir.iterdir() if p.is_file()}
        except OSError:
            # A watch dir that vanished is not a crash; it is nothing to report.
            return set()

    def poll(self) -> list[str]:
        now = self._clock()
        if self._seen is not None and (now - self._last_poll) < self.poll_s:
            return []
        self._last_poll = now
        current = self._names()
        if self._seen is None:
            self._seen = current  # cold boot: remember, announce nothing
            return []
        new = sorted(current - self._seen)
        self._seen = current
        return new


class WakeLoop(threading.Thread):
    """The autonomy lane. Daemon by construction: she never outlives her server."""

    def __init__(
        self,
        submit: Callable[[str], str],
        announce: Callable[[str, str], None],
        busy: Callable[[], bool],
        interval_s: float = 1800,
        cooldown_s: float = 900,
        quiet=(23, 8),
        watch_dir=None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], datetime] = datetime.now,
        poll_s: float = 1.0,
        heartbeat_ticks: bool = False,
    ):
        super().__init__(name="enigma-wake", daemon=True)
        self._submit = submit
        self._announce = announce
        self._busy = busy
        self._interval_s = float(interval_s)
        self._cooldown_s = float(cooldown_s)
        self._quiet = quiet
        self._clock = clock
        self._wall_clock = wall_clock
        self._poll_s = float(poll_s)
        # BARE TICKS DO NOT SPEND THE MODEL (measured 2026-09-01, B3 smoke).
        # The cheap-silence pattern assumes the model can emit the literal
        # NO_REPLY sentinel. sft2 CANNOT: asked to stay silent it paraphrases
        # the instruction instead ("Sure, I'll say it only if there's something
        # worth saying right now. No need to reply."), which is not NO_REPLY, so
        # every single tick announced chatter -- 15 rows in ~90 seconds at
        # --wake-interval 5. That is the banked instruction-following weakness,
        # not a prompt bug, so the loop stops asking a question she cannot
        # answer with silence. A timer tick carrying no real event has nothing
        # to report by construction; the model-heartbeat only becomes meaningful
        # once she has ambient inputs to REVIEW. File events always submit.
        self._heartbeat_ticks = bool(heartbeat_ticks)
        self._stop_event = threading.Event()
        self._seq = count()
        self._last_spoke: Optional[float] = None
        self.queue: queue.PriorityQueue = queue.PriorityQueue()
        self.source = (
            FileDropSource(watch_dir, poll_s=max(1.0, self._poll_s), clock=clock) if watch_dir else None
        )

    # -- public ------------------------------------------------------------
    def post(self, kind: str, detail: str, priority: Optional[int] = None) -> None:
        """Put an event on the queue. The seq counter is what keeps two events
        of equal priority from being compared on their strings."""
        if priority is None:
            priority = _PRIORITIES.get(kind, PRIORITY_TICK)
        self.queue.put((priority, next(self._seq), kind, detail))

    def stop(self) -> None:
        self._stop_event.set()

    # -- internals ---------------------------------------------------------
    def _handle_event(self, event) -> None:
        _priority, _seq, kind, detail = event
        # A bare heartbeat with nothing to review never reaches the model.
        if kind == "tick" and not self._heartbeat_ticks:
            return
        # The user's lane always wins: a tick that lands mid-generation is
        # DROPPED, not queued behind her -- a deferred heartbeat arriving after
        # the reply it interrupted is worse than one that never happened.
        if self._busy():
            return
        if not tick_allowed(self._wall_clock(), self._last_spoke, self._clock(), self._cooldown_s, self._quiet):
            return
        try:
            reply = self._submit(heartbeat_prompt(kind, detail))
        except Exception as exc:
            # A failed heartbeat must never take the server with it.
            logger.warning("wake: heartbeat generation failed: %r", exc)
            return
        if is_no_reply(reply):
            return  # silence costs nothing, including the cooldown
        try:
            self._announce(reply, kind)
        except Exception as exc:
            logger.warning("wake: announce failed: %r", exc)
            return
        self._last_spoke = self._clock()

    def run(self) -> None:
        next_tick = self._clock() + self._interval_s
        while not self._stop_event.is_set():
            if self.source is not None:
                for name in self.source.poll():
                    self.post("file", name)
            if self._clock() >= next_tick:
                # Posted unconditionally; _handle_event is the one place that
                # decides whether a tick is worth the model. Gating here too
                # would be a second copy of that rule with nothing observable
                # to keep it honest.
                self.post("tick", "heartbeat")
                next_tick = self._clock() + self._interval_s
            try:
                event = self.queue.get(timeout=self._poll_s)
            except queue.Empty:
                continue
            if self._stop_event.is_set():
                break
            self._handle_event(event)
