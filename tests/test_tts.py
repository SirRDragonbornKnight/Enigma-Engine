"""Voice organ (core/tts.py): the Speaker serializes utterances through one
worker thread, synthesizes WAVs synchronously, plays interruptibly, survives
failing jobs, and fails honestly when the backend is missing. A fake synth
backend and a fake player stand in -- the suite never downloads Kokoro, touches
the GPU, or opens an audio device."""

from __future__ import annotations

import json
import threading
import time
import wave

import numpy as np
import pytest

from enigma_engine.core.tts import (
    DEFAULT_RECIPE,
    MAX_SPEAK_CHARS,
    PRESET_VOICES,
    Speaker,
    TTSError,
    _normalize_recipe,
    split_sentences,
)


class FakeBackend:
    """Records synth/set_recipe calls; returns a tiny fixed signal. A marker
    substring makes synth raise, to exercise the worker's error path."""

    sample_rate = 24000

    def __init__(self, fail_on: str | None = None):
        self.synths: list[str] = []
        self.recipes: list[dict] = []
        self.fail_on = fail_on

    def synth(self, text: str):
        if self.fail_on is not None and self.fail_on in text:
            raise RuntimeError("synthetic synth failure")
        self.synths.append(text)
        return np.zeros(240, dtype=np.float32)

    def set_recipe(self, recipe: dict):
        # lang_code "zz" stands in for a recipe the backend accepts by shape but
        # rejects on apply (e.g. an unloadable language) -- exercises set_voice's
        # commit-after-apply contract.
        if recipe.get("lang_code") == "zz":
            raise TTSError("backend rejected recipe")
        self.recipes.append(recipe)


class FakePlayer:
    """Records playback. By default wait() returns at once (playback is
    instant in tests). block=True holds wait() until stop()/release, so a test
    can pin one utterance 'playing' while it queues and cancels the rest."""

    def __init__(self, block: bool = False):
        self.plays: list[int] = []
        self.stopped = False
        self.block = block
        self.playing = threading.Event()
        self._release = threading.Event()

    def play(self, audio, sample_rate: int):
        self.plays.append(len(audio))
        self.playing.set()

    def wait(self):
        if self.block:
            self._release.wait(5)

    def stop(self):
        self.stopped = True
        self._release.set()


def make_speaker(backend=None, player=None, **kw):
    backend = backend or FakeBackend()
    player = player or FakePlayer()
    sp = Speaker(synth_factory=lambda: backend, player_factory=lambda: player, **kw)
    return sp, backend, player


@pytest.fixture()
def rig():
    sp, backend, player = make_speaker()
    yield sp, backend, player
    sp.close()


# --------------------------------------------------------------- speaking

def test_speak_synthesizes_and_plays(rig):
    sp, backend, player = rig
    spoken = sp.speak("hello there.", wait=True)
    assert spoken == "hello there."
    assert backend.synths == ["hello there."]
    assert len(player.plays) == 1


def test_reply_is_spoken_sentence_by_sentence(rig):
    sp, backend, player = rig
    sp.speak("First sentence. Second one! And a third?", wait=True)
    assert backend.synths == ["First sentence.", "Second one!", "And a third?"]
    assert len(player.plays) == 3


def test_utterances_play_in_order(rig):
    sp, backend, _ = rig
    sp.speak("one.")
    sp.speak("two.")
    sp.speak("three.", wait=True)  # worker drains FIFO, so this waits for all
    assert backend.synths == ["one.", "two.", "three."]


def test_on_speaking_hook_brackets_playback():
    events = []
    sp, _, _ = make_speaker(on_speaking=lambda active: events.append(active))
    try:
        sp.speak("hello there.", wait=True)
    finally:
        sp.close()
    assert events == [True, False]  # opens then closes around the utterance


def test_empty_text_is_refused(rig):
    sp, _, _ = rig
    with pytest.raises(TTSError, match="nothing to say"):
        sp.speak("   ")


def test_long_text_is_clipped(rig):
    sp, _, _ = rig
    spoken = sp.speak("x" * (MAX_SPEAK_CHARS * 3), wait=True)
    assert len(spoken) == MAX_SPEAK_CHARS


def test_worker_survives_a_failing_job():
    backend = FakeBackend(fail_on="boom")
    sp, _, _ = make_speaker(backend=backend)
    try:
        with pytest.raises(TTSError, match="synthetic synth failure"):
            sp.speak("boom.", wait=True)
        assert isinstance(sp.last_error, RuntimeError)
        sp.speak("still alive.", wait=True)  # worker keeps serving
        assert "still alive." in backend.synths
    finally:
        sp.close()


def test_last_error_clears_when_a_later_job_succeeds():
    """A transient failure must not pin the /v1/audio/status health indicator to
    "error" for the rest of the session -- the next job that actually runs clears
    it (audit 2026-07-23 round 3)."""
    backend = FakeBackend(fail_on="boom")
    sp, _, _ = make_speaker(backend=backend)
    try:
        with pytest.raises(TTSError):
            sp.speak("boom.", wait=True)
        assert sp.last_error is not None
        sp.speak("recovered fine.", wait=True)
        assert sp.last_error is None  # health is honest again
    finally:
        sp.close()


# ------------------------------------------------------------------- stop

def test_stop_cancels_current_and_queued_utterances():
    backend = FakeBackend()
    player = FakePlayer(block=True)  # pin the first utterance 'playing'
    sp, _, _ = make_speaker(backend=backend, player=player)
    try:
        sp.speak("one.")  # will block in player.wait()
        assert player.playing.wait(5), "worker never reached playback"
        sp.speak("two.")  # queued behind the blocked utterance
        sp.speak("three.")
        sp.stop()
        # The blocked utterance was aborted; the queued two never synthesized.
        assert player.stopped
        assert backend.synths == ["one."]
        # A fresh utterance after stop plays normally (new epoch).
        sp.speak("after.", wait=True)
        assert "after." in backend.synths
    finally:
        sp.close()


# ---------------------------------------------------------------- save_wav

def test_save_wav_writes_a_real_wav(rig, tmp_path):
    sp, _, _ = rig
    out = sp.save_wav("to disk.", tmp_path / "utterance.wav")
    assert out.exists()
    with wave.open(str(out), "rb") as w:  # a valid, readable PCM WAV
        assert w.getnchannels() == 1
        assert w.getframerate() == 24000
        assert w.getnframes() == 240


def test_queued_save_survives_a_stop_that_drops_plays(tmp_path):
    """A save already SITTING IN THE QUEUE when stop() drains it must be kept
    (stop's keep-branch), while a queued play is dropped. The old test ran
    stop() on an empty queue, so the keep-branch never executed and a
    regression dropping queued saves passed (audit 2026-07-23 T1)."""
    backend = FakeBackend()
    player = FakePlayer(block=True)  # pin the worker mid-utterance
    sp, _, _ = make_speaker(backend=backend, player=player)
    out = tmp_path / "s.wav"
    saved: list = []
    try:
        sp.speak("one.")  # pulled, plays, blocks in player.wait()
        assert player.playing.wait(5), "worker never reached playback"
        sp.speak("two.")  # a play job queued behind the blocked one
        t = threading.Thread(target=lambda: saved.append(sp.save_wav("keep me.", out)))
        t.start()
        deadline = time.time() + 5  # both jobs queued behind the blocked utterance
        while sp._queue.qsize() < 2 and time.time() < deadline:
            time.sleep(0.01)
        assert sp._queue.qsize() >= 2, "save/play did not queue behind the block"
        sp.stop()  # releases the block; drains: drop "two.", KEEP the save
        t.join(5)
        assert out.exists()  # the queued save survived the drain
        assert "keep me." in backend.synths
        assert "two." not in backend.synths  # the queued play was dropped
    finally:
        sp.close()


def test_cancelled_wait_speak_raises_not_success():
    """speak(wait=True) whose job was dropped by stop() must raise, not return
    as if it spoke -- cancellation and success were indistinguishable before
    (audit 2026-07-23 M5)."""
    backend = FakeBackend()
    player = FakePlayer(block=True)
    sp, _, _ = make_speaker(backend=backend, player=player)
    result: list = []
    try:
        sp.speak("one.")  # blocks in playback
        assert player.playing.wait(5), "worker never reached playback"

        def waiter():
            try:
                sp.speak("two.", wait=True)
                result.append("returned")
            except TTSError:
                result.append("raised")

        t = threading.Thread(target=waiter)
        t.start()
        deadline = time.time() + 5
        while sp._queue.qsize() < 1 and time.time() < deadline:
            time.sleep(0.01)
        sp.stop()  # drops the queued "two." -- the waiter must see cancellation
        t.join(5)
        assert result == ["raised"]
        assert "two." not in backend.synths
    finally:
        sp.close()


def test_init_timeout_releases_the_orphaned_worker(monkeypatch):
    """A construction timeout must leave the worker a way OUT: if its build ever
    finishes, the queued close sentinel makes it exit and drop the backend
    (else an orphaned Kokoro pipeline holds VRAM forever; audit 2026-07-23 M6)."""
    import enigma_engine.core.tts as tts_mod

    monkeypatch.setattr(tts_mod, "_INIT_TIMEOUT_S", 0.05)
    release = threading.Event()
    slow_backend = FakeBackend()

    def slow_factory():
        release.wait(5)  # simulate a slow model build
        return slow_backend

    sp = None
    with pytest.raises(TTSError, match="did not initialize"):
        sp = Speaker(synth_factory=slow_factory, player_factory=lambda: None)
    release.set()  # the build now "finishes" -- worker must exit, not serve
    # The constructor raised, so grab the thread via gc-free handle: none exists;
    # assert instead that no worker thread is left alive with our name.
    deadline = time.time() + 5
    while time.time() < deadline:
        alive = [t for t in threading.enumerate() if t.name == "enigma-tts"]
        if not alive:
            break
        time.sleep(0.02)
    assert not [t for t in threading.enumerate() if t.name == "enigma-tts"], \
        "orphaned tts worker still alive after its build completed"


def test_stop_breaks_utterance_at_the_next_sentence(tmp_path):
    """stop() during a multi-sentence utterance must break the worker's loop at
    the next boundary -- pins the per-sentence epoch check in _play_utterance,
    which a prior mutation showed was 100% untested (deleting it passed the whole
    suite; audit 2026-07-23 A2)."""
    backend = FakeBackend()
    player = FakePlayer(block=True)  # block on sentence 1's wait()
    sp, _, _ = make_speaker(backend=backend, player=player)
    try:
        sp.speak("first sentence. second sentence.")
        assert player.playing.wait(5), "worker never reached playback"
        sp.stop()  # bumps epoch, releases wait()
        time.sleep(0.1)  # let the worker wake and (not) proceed to sentence 2
        assert backend.synths == ["first sentence."]  # sentence 2 never synthesized
    finally:
        sp.close()


# ------------------------------------------------------------ construction

def test_missing_backend_fails_at_construction():
    def broken():
        raise TTSError("kokoro not installed")

    with pytest.raises(TTSError, match="failed to initialize"):
        Speaker(synth_factory=broken)


def test_close_stops_worker(rig):
    sp, _, _ = rig
    sp.close()
    sp._worker.join(timeout=5)
    assert not sp._worker.is_alive()


# --------------------------------------------------------------- recipe

def test_voices_lists_presets(rig):
    sp, _, _ = rig
    ids = {v["id"] for v in sp.voices}
    assert "af_heart" in ids and "bf_emma" in ids
    assert len(sp.voices) == len(PRESET_VOICES)


def test_default_recipe_is_used_when_no_file(rig):
    sp, _, _ = rig
    recipe = sp.get_voice()
    assert recipe["engine"] == "kokoro"
    assert recipe["blend"][0][0] == DEFAULT_RECIPE["blend"][0][0]


def test_voice_name_overrides_blend_with_single_preset():
    sp, backend, _ = make_speaker(voice_name="af_bella")
    try:
        assert sp.get_voice()["blend"] == [["af_bella", 1.0]]
    finally:
        sp.close()


def test_set_voice_applies_persists_and_validates(tmp_path):
    path = tmp_path / "voice.json"
    sp, backend, _ = make_speaker(recipe_path=path)
    try:
        sp.set_voice(blend=[["af_heart", 0.7], ["af_bella", 0.3]], speed=1.1)
        recipe = sp.get_voice()
        assert recipe["speed"] == 1.1
        assert recipe["blend"] == [["af_heart", 0.7], ["af_bella", 0.3]]
        assert backend.recipes and backend.recipes[-1]["speed"] == 1.1  # applied
        saved = json.loads(path.read_text(encoding="utf-8"))  # persisted
        assert saved["speed"] == 1.1
        with pytest.raises(TTSError, match="unknown voice"):
            sp.set_voice(blend=[["not_a_voice", 1.0]])
    finally:
        sp.close()


def test_recipe_loads_from_file(tmp_path):
    path = tmp_path / "voice.json"
    path.write_text(json.dumps({"blend": [["bf_emma", 1.0]], "speed": 0.8}), encoding="utf-8")
    sp, _, _ = make_speaker(recipe_path=path)
    try:
        recipe = sp.get_voice()
        assert recipe["blend"] == [["bf_emma", 1.0]]
        assert recipe["speed"] == 0.8
    finally:
        sp.close()


def test_corrupt_recipe_file_falls_back_to_default(tmp_path):
    path = tmp_path / "voice.json"
    path.write_text("{ not valid json", encoding="utf-8")
    sp, _, _ = make_speaker(recipe_path=path)
    try:
        assert sp.get_voice()["blend"][0][0] == DEFAULT_RECIPE["blend"][0][0]
    finally:
        sp.close()


@pytest.mark.parametrize("body", [
    '{"blend": [["af_heart", NaN]]}',        # non-finite weight (the 1e999 -> NaN kill)
    '{"blend": [["af_heartt", 1.0]]}',       # typo'd voice name
    '{"blend": [["af_heart", 1.0]], "speed": 9.0}',  # out-of-range speed
])
def test_parseable_but_invalid_recipe_file_falls_back(tmp_path, body):
    """A file that PARSES as JSON but fails validation must degrade to the
    default voice, not raise out of __init__ and disable the organ at every
    boot (audit 2026-07-23 H2). The earlier fallback covered only parse errors."""
    path = tmp_path / "voice.json"
    path.write_text(body, encoding="utf-8")
    sp, _, _ = make_speaker(recipe_path=path)
    try:
        assert sp.get_voice()["blend"] == DEFAULT_RECIPE["blend"]
    finally:
        sp.close()


def test_set_voice_does_not_commit_when_backend_rejects(tmp_path):
    """set_voice commits in-memory state and voice.json ONLY after the worker
    applies the recipe. A backend rejection must leave get_voice reporting the
    old voice and write nothing (audit 2026-07-23 A3)."""
    path = tmp_path / "voice.json"
    sp, _, _ = make_speaker(recipe_path=path)
    try:
        before = sp.get_voice()
        with pytest.raises(TTSError):
            sp.set_voice(lang_code="zz")  # shape-valid, backend rejects on apply
        assert sp.get_voice() == before   # in-memory unchanged
        assert not path.exists()          # nothing persisted
    finally:
        sp.close()


# ------------------------------------------------------ pure-function units

def test_normalize_recipe_normalizes_weights():
    r = _normalize_recipe({"blend": [["af_heart", 3], ["af_kore", 1]], "speed": 1.0})
    assert r["blend"] == [["af_heart", 0.75], ["af_kore", 0.25]]


def test_normalize_recipe_rejects_bad_speed():
    with pytest.raises(TTSError, match="speed"):
        _normalize_recipe({"blend": [["af_heart", 1.0]], "speed": 9.0})


def test_normalize_recipe_rejects_unknown_voice():
    with pytest.raises(TTSError, match="unknown voice"):
        _normalize_recipe({"blend": [["nope", 1.0]]})


@pytest.mark.parametrize("weight", [float("inf"), float("nan"), 1e999])
def test_normalize_recipe_rejects_nonfinite_weight(weight):
    with pytest.raises(TTSError):  # inf/nan would poison the normalized ratio
        _normalize_recipe({"blend": [["af_heart", weight]]})


def test_normalize_recipe_rejects_overflowing_weight_sum():
    # Each weight is finite but the sum overflows to inf; w/inf = 0.0 for every
    # voice would make a "valid" recipe that synthesizes silence.
    with pytest.raises(TTSError, match="unusable total"):
        _normalize_recipe({"blend": [["af_heart", 1e308], ["af_kore", 1e308]]})


@pytest.mark.parametrize("blend", [
    [["af_heart"]],          # missing weight
    [{"name": "af_heart"}],  # wrong item shape
    [["af_heart", "loud"]],  # non-numeric weight
    [["af_heart", None]],    # null weight
])
def test_normalize_recipe_rejects_malformed_item_as_ttserror(blend):
    with pytest.raises(TTSError):  # a raw ValueError/TypeError here 500s the endpoint
        _normalize_recipe({"blend": blend})


def test_split_sentences():
    assert split_sentences("Hi there. How are you? Good!") == [
        "Hi there.",
        "How are you?",
        "Good!",
    ]
    assert split_sentences("no terminator here") == ["no terminator here"]
    assert split_sentences("   ") == []


def test_split_sentences_keeps_decimals_and_abbreviations():
    # A decimal's point has no following space, so it must not split (the bug:
    # "3.14" -> "3." + "14", read aloud as "three ... fourteen").
    assert split_sentences("Pi is 3.14. Next.") == ["Pi is 3.14.", "Next."]
    assert split_sentences("It costs 9.99 total.") == ["It costs 9.99 total."]
    # A period ending a known abbreviation is not a sentence break.
    assert split_sentences("Dr. Smith left. Bye.") == ["Dr. Smith left.", "Bye."]
    # A run of terminators stays with its sentence.
    assert split_sentences("Really?! Yes.") == ["Really?!", "Yes."]
