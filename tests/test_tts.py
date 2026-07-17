"""Voice organ (core/tts.py): the Speaker must serialize utterances through
its worker thread, write WAVs synchronously, survive failing jobs, and fail
honestly when the backend is missing. A fake engine stands in for pyttsx3 --
the suite must never actually play audio or require SAPI."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from enigma_engine.core.tts import MAX_SPEAK_CHARS, Speaker, TTSError


class FakeEngine:
    """Records the call sequence; materializes save_to_file on runAndWait
    (mirrors pyttsx3, where runAndWait flushes the queued command)."""

    def __init__(self, fail_on: str | None = None):
        self.calls: list[tuple] = []
        self.fail_on = fail_on
        self._pending_wav: tuple[str, str] | None = None

    def setProperty(self, key, value):
        self.calls.append(("set", key, value))

    def getProperty(self, key):
        assert key == "voices"
        return [SimpleNamespace(id="v1", name="Test Voice")]

    def say(self, text):
        if self.fail_on is not None and self.fail_on in text:
            raise RuntimeError("synthetic engine failure")
        self.calls.append(("say", text))

    def save_to_file(self, text, path):
        self._pending_wav = (text, path)

    def runAndWait(self):
        if self._pending_wav is not None:
            text, path = self._pending_wav
            Path(path).write_bytes(b"RIFFfake")
            self.calls.append(("wav", text, path))
            self._pending_wav = None
        self.calls.append(("run",))


@pytest.fixture()
def rig():
    engine = FakeEngine()
    speaker = Speaker(engine_factory=lambda: engine)
    yield engine, speaker
    speaker.close()


def test_voice_name_substring_resolves_to_installed_id():
    engine = FakeEngine()
    speaker = Speaker(voice="test", engine_factory=lambda: engine)
    speaker.speak("hi", wait=True)
    assert ("set", "voice", "v1") in engine.calls
    speaker.close()


def test_unknown_voice_name_fails_honestly_at_init():
    with pytest.raises(TTSError, match="no installed voice matches"):
        Speaker(voice="does-not-exist", engine_factory=FakeEngine)


def test_speak_says_then_flushes(rig):
    engine, speaker = rig
    spoken = speaker.speak("hello there", wait=True)
    assert spoken == "hello there"
    assert engine.calls == [("say", "hello there"), ("run",)]


def test_utterances_play_in_order(rig):
    engine, speaker = rig
    speaker.speak("first")
    speaker.speak("second")
    speaker.speak("third", wait=True)  # worker drains FIFO, so this waits for all
    says = [c[1] for c in engine.calls if c[0] == "say"]
    assert says == ["first", "second", "third"]


def test_save_wav_writes_file(rig, tmp_path):
    engine, speaker = rig
    out = speaker.save_wav("to disk", tmp_path / "utterance.wav")
    assert out.read_bytes() == b"RIFFfake"
    assert ("wav", "to disk", str(out)) in engine.calls


def test_missing_backend_fails_at_construction():
    def broken_factory():
        raise TTSError("pyttsx3 not installed")

    with pytest.raises(TTSError, match="failed to initialize"):
        Speaker(engine_factory=broken_factory)


def test_empty_text_is_refused(rig):
    _, speaker = rig
    with pytest.raises(TTSError, match="nothing to say"):
        speaker.speak("   ")


def test_long_text_is_clipped(rig):
    engine, speaker = rig
    spoken = speaker.speak("x" * (MAX_SPEAK_CHARS * 3), wait=True)
    assert len(spoken) == MAX_SPEAK_CHARS
    assert engine.calls[0] == ("say", "x" * MAX_SPEAK_CHARS)


def test_worker_survives_a_failing_job():
    engine = FakeEngine(fail_on="boom")
    speaker = Speaker(engine_factory=lambda: engine)
    try:
        with pytest.raises(TTSError, match="synthetic engine failure"):
            speaker.speak("boom", wait=True)
        assert isinstance(speaker.last_error, RuntimeError)
        # The worker must still be serving jobs after the failure.
        speaker.speak("still alive", wait=True)
        assert ("say", "still alive") in engine.calls
    finally:
        speaker.close()


def test_voices_discovered_at_init(rig):
    _, speaker = rig
    assert speaker.voices == [{"id": "v1", "name": "Test Voice"}]


def test_close_stops_worker(rig):
    _, speaker = rig
    speaker.close()
    speaker._worker.join(timeout=5)
    assert not speaker._worker.is_alive()
