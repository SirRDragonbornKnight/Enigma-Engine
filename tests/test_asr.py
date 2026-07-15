"""Ears organ (core/asr.py): transcription joins segments, reports honest
errors, and never requires a real whisper download -- a fake model stands in
for faster-whisper."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from enigma_engine.core.asr import ASRError, Ears


class FakeWhisper:
    def __init__(self, segments=None, fail=False):
        self.segments = segments if segments is not None else [" Hello ", "world. "]
        self.fail = fail
        self.calls: list[str] = []

    def transcribe(self, path):
        self.calls.append(path)
        if self.fail:
            raise RuntimeError("synthetic decode failure")
        segs = (SimpleNamespace(text=t) for t in self.segments)
        return segs, SimpleNamespace(language="en", duration=1.5)


@pytest.fixture()
def wav(tmp_path):
    p = tmp_path / "clip.wav"
    p.write_bytes(b"RIFFfake")
    return p


def test_transcribe_joins_segments(wav):
    fake = FakeWhisper()
    ears = Ears(model_factory=lambda: fake)
    out = ears.transcribe(wav)
    assert out == {"text": "Hello world.", "language": "en", "duration": 1.5}
    assert fake.calls == [str(wav)]


def test_missing_file_is_refused(tmp_path):
    ears = Ears(model_factory=FakeWhisper)
    with pytest.raises(ASRError, match="not found"):
        ears.transcribe(tmp_path / "nope.wav")


def test_backend_failure_is_wrapped(wav):
    ears = Ears(model_factory=lambda: FakeWhisper(fail=True))
    with pytest.raises(ASRError, match="synthetic decode failure"):
        ears.transcribe(wav)


def test_empty_audio_yields_empty_text(wav):
    ears = Ears(model_factory=lambda: FakeWhisper(segments=[]))
    assert ears.transcribe(wav)["text"] == ""
