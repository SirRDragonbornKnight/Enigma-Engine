"""Barge-in detection logic and mic wiring. The DECISION core is deterministic
and fully tested here; the sounddevice stream is injected with a fake so no real
microphone is opened. (Acoustic threshold tuning is a live task, not a test.)"""

from __future__ import annotations

import numpy as np

from enigma_engine.core.barge_in import BargeInDetector, MicBargeIn


# ------------------------------------------------------------ detector

def test_detector_fires_after_sustained_speech():
    d = BargeInDetector(threshold=0.02, min_speech_s=0.25)
    assert not d.feed(0.001, 0.05)          # quiet
    for _ in range(4):
        assert not d.feed(0.05, 0.05)       # loud, accumulates to 0.20 (< 0.25)
    assert d.feed(0.05, 0.05)               # 0.25 reached -> fire
    assert not d.feed(0.05, 0.05)           # accumulator reset after firing


def test_detector_resets_on_a_gap_of_quiet():
    d = BargeInDetector(threshold=0.02, min_speech_s=0.20)
    assert not d.feed(0.05, 0.15)           # accum 0.15
    assert not d.feed(0.001, 0.05)          # silence wipes it
    assert not d.feed(0.05, 0.15)           # only 0.15 again
    assert d.feed(0.05, 0.06)               # now 0.21 -> fire


# ------------------------------------------------------------ mic wiring

class _FakeStream:
    def __init__(self, cb):
        self.cb = cb
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

    def close(self):
        pass


def test_mic_opens_only_when_active_and_stops_on_loud_input():
    fired = []
    made = {}

    def factory(cb):
        s = _FakeStream(cb)
        made["s"] = s
        return s

    mb = MicBargeIn(on_detect=lambda: fired.append(1), threshold=0.02,
                    min_speech_s=0.05, samplerate=16000, block=1600, stream_factory=factory)
    mb.set_active(True)
    assert made["s"].started  # mic opened

    # one 1600-sample block at 16 kHz = 0.1 s of loud audio (>= 0.05 s) -> fire
    made["s"].cb(np.full(1600, 0.2, dtype=np.float32))
    assert fired == [1]

    mb.set_active(False)
    assert not made["s"].started  # mic closed again


def test_mic_ignores_quiet_input():
    fired = []
    holder = {}

    def factory(cb):
        holder["cb"] = cb
        return _FakeStream(cb)

    mb = MicBargeIn(on_detect=lambda: fired.append(1), threshold=0.05,
                    min_speech_s=0.05, samplerate=16000, block=1600, stream_factory=factory)
    mb.set_active(True)
    holder["cb"](np.full(1600, 0.001, dtype=np.float32))  # near silence
    assert fired == []


def test_late_callback_after_stop_does_not_fire():
    """PortAudio can deliver one in-flight callback after stop(); a stale
    detection would cancel the NEXT queued utterance (audit 2026-07-23 M3)."""
    fired = []
    holder = {}

    def factory(cb):
        holder["cb"] = cb
        return _FakeStream(cb)

    mb = MicBargeIn(on_detect=lambda: fired.append(1), threshold=0.02,
                    min_speech_s=0.05, samplerate=16000, block=1600, stream_factory=factory)
    mb.set_active(True)
    mb.set_active(False)  # she stopped speaking; mic closed
    holder["cb"](np.full(1600, 0.2, dtype=np.float32))  # late in-flight block
    assert fired == []


def test_detection_fires_once_per_utterance():
    """Continuous talking must not re-fire every min_speech_s -- repeat stop()
    calls bump the Speaker epoch and can kill a reply queued after the first
    stop (audit 2026-07-23 M3)."""
    fired = []
    holder = {}

    def factory(cb):
        holder["cb"] = cb
        return _FakeStream(cb)

    mb = MicBargeIn(on_detect=lambda: fired.append(1), threshold=0.02,
                    min_speech_s=0.05, samplerate=16000, block=1600, stream_factory=factory)
    mb.set_active(True)
    for _ in range(6):  # 0.6 s of sustained loud speech
        holder["cb"](np.full(1600, 0.2, dtype=np.float32))
    assert fired == [1]  # exactly one detection for the whole utterance

    # A NEW utterance re-arms the detector.
    mb.set_active(False)
    mb.set_active(True)
    holder["cb"](np.full(1600, 0.2, dtype=np.float32))
    assert fired == [1, 1]


def test_failed_mic_start_degrades_inactive_with_one_warning(capsys):
    """A mic held by another app must leave barge-in INACTIVE (no armed-but-dead
    stream) and say so once -- not fail silently forever (audit 2026-07-23 F3)."""
    class _BrokenStream(_FakeStream):
        def start(self):
            raise RuntimeError("device busy")

    mb = MicBargeIn(on_detect=lambda: None, stream_factory=lambda cb: _BrokenStream(cb))
    mb.set_active(True)
    assert mb._stream is None and not mb._active  # degraded, not armed-but-dead
    mb.set_active(False)
    mb.set_active(True)  # second utterance: still degrades, but no second WARN
    out = capsys.readouterr().out
    assert out.count("barge-in mic unavailable") == 1
