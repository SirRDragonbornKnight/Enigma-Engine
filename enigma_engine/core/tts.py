"""Text-to-speech -- the voice organ (Kokoro-82M backend).

Enigma's voice is a LOCAL neural model (Kokoro-82M), not the Windows SAPI
voices she used to borrow. A voice here is a "recipe": one or more Kokoro
preset voices blended by weight, plus a speaking speed and a language. The
recipe is data (JSON), so the voice can be retuned -- or swapped for a whole
new blend -- at runtime without touching code. Her shipped default is a
Cortana-flavored blend (warm + smooth + composed).

serve_enigma.py exposes this as the `speak` built-in tool and the
/v1/audio/speech endpoint (both behind --voice). `speak` plays on THIS
machine's speakers; /v1/audio/speech returns WAV bytes a client plays itself.

Three design points that matter:

1. One worker thread owns ALL backend access. Kokoro runs on torch/CUDA and is
   not safe to drive from several threads at once, so every synth, playback,
   and recipe change is serialized through one queue -- utterances play in
   order, never over each other.

2. Playback is interruptible. Unlike SAPI (which finished a sentence no matter
   what), a spoken reply is synthesized and played sentence-by-sentence through
   a stream we control, so stop() halts her mid-word in a fraction of a second
   and drops anything queued behind it. That is what makes her feel like a
   conversation partner instead of a screen reader.

3. Every heavy dependency is injectable and lazily imported, so the test suite
   never downloads a model, touches the GPU, or opens an audio device. A fake
   synth backend and a fake player stand in; `import ...core.tts` stays clean.
"""

from __future__ import annotations

import json
import math
import os
import queue
import re
import threading
import wave
from dataclasses import dataclass, field
from pathlib import Path

# A runaway generation could ask for a minutes-long monologue; cap and say so.
MAX_SPEAK_CHARS = 1000

# First build may download the ~330 MB model + voice files, so init is patient.
_INIT_TIMEOUT_S = 90.0
_JOB_TIMEOUT_S = 120.0  # synth of a long utterance is slow, not hung
_SAMPLE_RATE = 24000  # Kokoro output rate

# Shipped default: a Cortana-character blend (warm af_heart, breathy af_nicole,
# composed af_kore), spoken a touch slow for a measured delivery. Overridden by
# the runtime recipe file when one exists.
DEFAULT_RECIPE: dict = {
    "engine": "kokoro",
    "lang_code": "a",  # 'a' American English, 'b' British English
    "blend": [["af_heart", 0.50], ["af_nicole", 0.30], ["af_kore", 0.20]],
    "speed": 0.92,
}

# Kokoro's built-in presets, offered by /v1/audio/voices and accepted in blends.
# Names encode language+gender: a/b = American/British, f/m = female/male.
PRESET_VOICES: tuple[str, ...] = (
    "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore",
    "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
    "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
    "am_onyx", "am_puck", "am_santa",
    "bf_alice", "bf_emma", "bf_isabella", "bf_lily",
    "bm_daniel", "bm_fable", "bm_george", "bm_lewis",
)


class TTSError(Exception):
    """The voice backend is unavailable or a synthesis/playback job failed."""


# Words whose trailing period is an abbreviation dot, not a sentence end.
_ABBREVIATIONS = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc",
    "e.g", "i.e", "no", "vol", "fig", "inc", "ltd", "co", "u.s", "a.m", "p.m",
})
_TRAILING_WORD = re.compile(r"[A-Za-z.]+$")


def split_sentences(text: str) -> list[str]:
    """Break text into sentences for streaming, interruptible playback.

    A break happens only at a `.`/`!`/`?` (run) that is FOLLOWED by whitespace
    or end-of-text, so a decimal ("3.14") never splits -- its point has no
    following space. A trailing run ("?!") stays with its sentence, a period
    ending a known abbreviation ("Dr.") does not break, and newlines always
    break. (The old regex broke at every period, cutting "3.14" into "3." +
    "14" -- talk-mode then misread numbers.)
    """
    out: list[str] = []
    n = len(text)
    start = i = 0
    while i < n:
        ch = text[i]
        if ch == "\n":
            seg = text[start:i].strip()
            if seg:
                out.append(seg)
            start = i = i + 1
            continue
        if ch in ".!?":
            j = i
            while j + 1 < n and text[j + 1] in ".!?":  # absorb "?!" / "..."
                j += 1
            nxt = text[j + 1] if j + 1 < n else ""
            at_boundary = nxt == "" or nxt.isspace()
            if at_boundary and i == j and ch == ".":  # single dot: abbreviation?
                word = _TRAILING_WORD.search(text[start:i])
                if word and word.group().rstrip(".").lower() in _ABBREVIATIONS:
                    at_boundary = False
            if at_boundary:
                seg = text[start:j + 1].strip()
                if seg:
                    out.append(seg)
                start = j + 1
            i = j + 1
            continue
        i += 1
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def _write_wav(path: str | Path, audio, sample_rate: int) -> None:
    """Write a float32 [-1, 1] mono signal as a 16-bit PCM WAV (stdlib only)."""
    import numpy as np

    a = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    pcm = (a * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def _normalize_recipe(recipe: dict) -> dict:
    """Validate + copy a recipe into canonical form, or raise TTSError.

    Every failure mode is a TTSError (never a bare ValueError/TypeError/inf),
    so the endpoint returns 400, not 500, and a hand-edited or hostile recipe
    file cannot poison persistence: a non-finite weight (1e999 -> inf -> NaN
    ratio) is rejected here rather than written to voice.json and killing every
    later boot.
    """
    try:
        speed = float(recipe.get("speed", 1.0))
    except (TypeError, ValueError) as exc:
        raise TTSError(f"speed must be a number, got {recipe.get('speed')!r}") from exc
    if not math.isfinite(speed) or not 0.5 <= speed <= 2.0:
        raise TTSError(f"speed must be between 0.5 and 2.0, got {speed}")
    lang = str(recipe.get("lang_code", "a"))
    raw = recipe.get("blend") or [[str(recipe.get("voice", "af_heart")), 1.0]]
    if not isinstance(raw, (list, tuple)):
        raise TTSError("recipe 'blend' must be a list of [voice, weight] pairs")
    blend: list[list] = []
    for item in raw:
        try:
            name, weight = str(item[0]), float(item[1])
        except (TypeError, ValueError, KeyError, IndexError) as exc:
            raise TTSError(f"malformed blend item {item!r} -- want [voice, weight]") from exc
        if not math.isfinite(weight):
            raise TTSError(f"blend weight for '{name}' must be finite, got {weight}")
        if name not in PRESET_VOICES:
            allowed = ", ".join(PRESET_VOICES)
            raise TTSError(f"unknown voice '{name}' -- available: {allowed}")
        if weight > 0:
            blend.append([name, weight])
    if not blend:
        raise TTSError("recipe blend is empty (no voices with positive weight)")
    total = sum(w for _, w in blend)
    # Per-item weights are finite, but their SUM can still overflow to inf
    # (two 1e308s) -- w/inf would silently produce an all-zero, silent voice.
    if not math.isfinite(total) or total <= 0:
        raise TTSError(f"blend weights sum to an unusable total ({total})")
    blend = [[n, w / total] for n, w in blend]  # weights always sum to 1.0
    return {"engine": "kokoro", "lang_code": lang, "blend": blend, "speed": speed}


class KokoroBackend:
    """Wraps a Kokoro pipeline and a resolved (blended) voice tensor.

    Built eagerly so a missing model or bad voice surfaces at organ startup,
    not mid-conversation. synth(text) returns a float32 mono numpy array.
    """

    sample_rate = _SAMPLE_RATE

    def __init__(self, recipe: dict):
        try:
            from kokoro import KPipeline
        except ImportError as exc:
            raise TTSError(
                "kokoro not installed -- pip install 'enigma-engine[voice]'"
            ) from exc
        self._KPipeline = KPipeline
        self._pipes: dict = {}
        self._voice_cache: dict = {}
        self._recipe: dict = {}
        self.set_recipe(recipe)

    def _pipe(self, lang: str):
        if lang not in self._pipes:
            try:
                self._pipes[lang] = self._KPipeline(lang_code=lang)
            except Exception as exc:  # download/build failure -> honest error
                raise TTSError(f"could not load the voice model: {exc}") from exc
        return self._pipes[lang]

    def set_recipe(self, recipe: dict) -> None:
        """Adopt a new voice recipe and resolve its blended tensor now."""
        norm = _normalize_recipe(recipe)
        pipe = self._pipe(norm["lang_code"])
        tensor = None
        for name, weight in norm["blend"]:
            if name not in self._voice_cache:
                try:
                    self._voice_cache[name] = pipe.load_voice(name)
                except Exception as exc:
                    raise TTSError(f"could not load voice '{name}': {exc}") from exc
            part = self._voice_cache[name] * weight
            tensor = part if tensor is None else tensor + part
        self._recipe = norm
        self._voice = tensor

    def synth(self, text: str):
        import numpy as np

        pipe = self._pipe(self._recipe["lang_code"])
        speed = self._recipe["speed"]
        chunks = [audio for _, _, audio in pipe(text, voice=self._voice, speed=speed)]
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        parts = [
            c.detach().cpu().numpy() if hasattr(c, "detach") else np.asarray(c, dtype=np.float32)
            for c in chunks
        ]
        return np.concatenate(parts).astype(np.float32)


class SoundDevicePlayer:
    """Plays a float32 array on the machine's speakers, interruptibly.

    sd.play() starts a non-blocking stream; wait() blocks until it finishes OR
    until stop() aborts it from another thread. Constructed lazily (opening an
    audio device) only when the first utterance actually plays.
    """

    def __init__(self):
        import sounddevice as sd

        self._sd = sd

    def play(self, audio, sample_rate: int) -> None:
        self._sd.play(audio, sample_rate)

    def wait(self) -> None:
        self._sd.wait()

    def stop(self) -> None:
        self._sd.stop()


@dataclass
class _Job:
    kind: str  # "play" | "save" | "apply"
    text: str = ""
    wav_path: str | None = None
    recipe: dict | None = None
    epoch: int = 0  # play jobs run only if still the current epoch (see stop())
    done: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None
    cancelled: bool = False  # set when stop() dropped this job before it played


class Speaker:
    """Serialized, interruptible text-to-speech through one worker thread.

    speak() is fire-and-forget by default so tool execution never blocks
    generation for the seconds an utterance takes; pass wait=True (tests,
    scripts) to block until it finished and surface any error. stop() cuts off
    the current utterance mid-word and drops anything queued behind it.
    """

    def __init__(
        self,
        recipe_path: str | Path | None = None,
        voice_name: str | None = None,
        recipe: dict | None = None,
        synth_factory=None,
        player_factory=None,
        on_speaking=None,
    ):
        self._recipe_path = Path(recipe_path) if recipe_path else None
        self._recipe = self._load_recipe(recipe, voice_name)
        self._synth_factory = synth_factory or (lambda: KokoroBackend(self._recipe))
        self._player_factory = player_factory or SoundDevicePlayer
        self._player = None  # lazily built on first playback
        # Optional callback fired with True when an utterance starts playing and
        # False when it ends -- lets barge-in open the mic only while she talks.
        self._on_speaking = on_speaking

        self._queue: queue.Queue = queue.Queue()
        # stop() bumps _epoch; the worker compares the job's captured epoch
        # against it at every step, so cancellation is a monotonic counter
        # (race-free) rather than a clearable flag that a stop() could lose.
        self._epoch = 0
        self._voice_lock = threading.Lock()  # serializes set_voice commits
        self.last_error: BaseException | None = None

        init_done = threading.Event()
        init_error: list[BaseException] = []
        self._worker = threading.Thread(
            target=self._run, args=(init_done, init_error), daemon=True, name="enigma-tts"
        )
        self._worker.start()
        if not init_done.wait(_INIT_TIMEOUT_S):
            # The worker cannot be killed, but the close sentinel makes it exit
            # the moment its build ever finishes -- otherwise an orphaned Kokoro
            # pipeline sits on VRAM for the process lifetime with no owner.
            self._queue.put(None)
            raise TTSError("voice engine did not initialize within timeout")
        if init_error:
            raise TTSError(f"voice engine failed to initialize: {init_error[0]}") from init_error[0]

    # ----------------------------------------------------------- recipe I/O

    def _load_recipe(self, recipe: dict | None, voice_name: str | None) -> dict:
        from_file = False
        if recipe is not None:
            base = recipe
        elif self._recipe_path is not None and self._recipe_path.exists():
            try:
                base = json.loads(self._recipe_path.read_text(encoding="utf-8"))
                from_file = True
            except (OSError, ValueError):
                base = dict(DEFAULT_RECIPE)  # unparseable file must not break voice
        else:
            base = dict(DEFAULT_RECIPE)
        # --voice-name overrides the blend with a single named preset.
        if voice_name:
            base = {**base, "blend": [[voice_name, 1.0]]}
            base.pop("voice", None)
        try:
            return _normalize_recipe(base)
        except TTSError:
            # A PARSEABLE-but-invalid file (hand-edit typo, or a poisoned weight)
            # must degrade to the shipped voice, not disable the organ at every
            # boot. An explicit caller-supplied recipe is a real error -- reraise.
            if not from_file:
                raise
            print("  WARN: voice.json invalid; using the default voice", flush=True)
            fallback = dict(DEFAULT_RECIPE)
            if voice_name:
                fallback = {**fallback, "blend": [[voice_name, 1.0]]}
                fallback.pop("voice", None)
            return _normalize_recipe(fallback)

    def _persist(self) -> None:
        if self._recipe_path is None:
            return
        try:
            self._recipe_path.parent.mkdir(parents=True, exist_ok=True)
            # Write-temp + atomic replace: a crash mid-write can never leave a
            # half-written recipe that fails to parse on the next boot.
            tmp = self._recipe_path.with_name(self._recipe_path.name + ".tmp")
            tmp.write_text(json.dumps(self._recipe, indent=2), encoding="utf-8")
            os.replace(tmp, self._recipe_path)
        except OSError:
            pass  # a failed write must not stop her talking this session

    # ------------------------------------------------------------- worker

    def _run(self, init_done: threading.Event, init_error: list[BaseException]) -> None:
        try:
            self._backend = self._synth_factory()
        except BaseException as exc:  # surfaced to the constructor
            init_error.append(exc)
            init_done.set()
            return
        init_done.set()

        while True:
            job = self._queue.get()
            if job is None:  # close() sentinel
                return
            try:
                if job.kind == "apply":
                    self._backend.set_recipe(job.recipe)
                elif job.kind == "save":
                    audio = self._backend.synth(job.text)
                    _write_wav(job.wav_path, audio, self._backend.sample_rate)
                elif job.kind == "play":
                    if job.epoch == self._epoch:  # else stop() cancelled it
                        self._play_utterance(job.text, job.epoch)
                    else:
                        job.cancelled = True  # do not report a skipped job as spoken
            except BaseException as exc:  # job errors are data, not crashes
                job.error = exc
                self.last_error = exc
                print(f"  WARN: tts job failed: {exc}", flush=True)
            else:
                # A job that actually RAN clearing the flag keeps /v1/audio/status
                # honest both ways: a transient device hiccup must not pin the
                # health indicator to "error" forever, and a job merely skipped
                # by stop() proves nothing about the audio path, so it does not
                # clear.
                if not job.cancelled:
                    self.last_error = None
            finally:
                job.done.set()

    def _play_utterance(self, text: str, epoch: int) -> None:
        # `epoch` is the value current when this job was pulled. stop() only ever
        # INCREMENTS self._epoch, so `self._epoch != epoch` means "a stop fired
        # after we started" -- checked before each synth and each play so a stop
        # lands within one sentence. No clearable flag, so a stop cannot be lost.
        if self._player is None:
            self._player = self._player_factory()
        self._signal_speaking(True)
        try:
            for sentence in split_sentences(text):
                if self._epoch != epoch:
                    break
                audio = self._backend.synth(sentence)
                if self._epoch != epoch:  # a stop during synth cancels playback
                    break
                self._player.play(audio, self._backend.sample_rate)
                self._player.wait()  # returns early if stop() aborted the stream
        finally:
            self._signal_speaking(False)

    def _signal_speaking(self, active: bool) -> None:
        if self._on_speaking is None:
            return
        try:
            self._on_speaking(active)
        except Exception:  # a listener fault must never break playback
            pass

    def set_on_speaking(self, callback) -> None:
        """Register (or clear) the speaking-state callback after construction."""
        self._on_speaking = callback

    # ------------------------------------------------------------- public

    def speak(self, text: str, wait: bool = False) -> str:
        """Queue an utterance for the speakers. Returns the text (post-cap)."""
        text = self._clip(text)
        job = _Job(kind="play", text=text, epoch=self._epoch)
        self._queue.put(job)
        if wait:
            self._finish(job)
        return text

    def save_wav(self, text: str, path: str | Path) -> Path:
        """Synthesize to a WAV file and block until it is written."""
        path = Path(path)
        job = _Job(kind="save", text=self._clip(text), wav_path=str(path))
        self._queue.put(job)
        self._finish(job)
        if not path.exists():
            raise TTSError(f"engine reported success but wrote no file: {path}")
        return path

    def stop(self) -> None:
        """Silence her immediately: abort the current utterance mid-word and
        cancel everything queued behind it. Save jobs are never dropped.

        Bumping the epoch cancels every play job queued so far (the worker
        skips any whose epoch is stale, even one it pulled a heartbeat before
        this drain); stopping the player aborts the utterance already playing,
        and the worker's per-sentence epoch check breaks its loop.
        """
        self._epoch += 1
        if self._player is not None:
            try:
                self._player.stop()
            except Exception:  # a flaky device must not wedge stop
                pass
        kept: list = []
        try:
            while True:
                job = self._queue.get_nowait()
                if job is not None and job.kind == "play":
                    job.cancelled = True  # a waiter must not read this as spoken
                    job.done.set()
                else:
                    kept.append(job)  # keep saves + the close sentinel
        except queue.Empty:
            pass
        for job in kept:
            self._queue.put(job)

    def get_voice(self) -> dict:
        """Return the current voice recipe (a copy)."""
        return json.loads(json.dumps(self._recipe))

    def set_voice(self, blend=None, speed=None, lang_code=None) -> dict:
        """Change the voice and persist it. Any omitted field is unchanged.
        Returns the new recipe; raises TTSError on an invalid one.

        The in-memory recipe and voice.json are committed ONLY after the worker
        has applied the new recipe to the backend. If the backend rejects it
        (e.g. an unloadable lang_code that passed shape validation), _finish
        raises and the committed state is untouched -- get_voice never reports a
        voice the backend refused. The lock serializes concurrent callers so two
        POSTs cannot interleave read-modify-write and lose an update.
        """
        with self._voice_lock:
            new = dict(self._recipe)
            if blend is not None:
                new["blend"] = blend
            if speed is not None:
                new["speed"] = speed
            if lang_code is not None:
                new["lang_code"] = lang_code
            norm = _normalize_recipe(new)  # shape validation (raises -> 400)
            job = _Job(kind="apply", recipe=norm)
            self._queue.put(job)
            self._finish(job)  # raises if the backend rejected it -> nothing committed
            self._recipe = norm
            self._persist()
            return self.get_voice()

    @property
    def voices(self) -> list[dict]:
        """Presets available for a recipe (the /v1/audio/voices shape)."""
        return [{"id": name, "name": name} for name in PRESET_VOICES]

    def close(self) -> None:
        self._queue.put(None)

    # ------------------------------------------------------------ helpers

    @staticmethod
    def _clip(text: str) -> str:
        text = str(text).strip()
        if not text:
            raise TTSError("nothing to say")
        return text[:MAX_SPEAK_CHARS]

    def _finish(self, job: _Job) -> None:
        if not job.done.wait(_JOB_TIMEOUT_S):
            raise TTSError("tts job timed out")
        if job.error is not None:
            raise TTSError(f"tts job failed: {job.error}") from job.error
        if job.cancelled:  # stop() dropped it -- do not report it as spoken
            raise TTSError("utterance cancelled by stop()")
