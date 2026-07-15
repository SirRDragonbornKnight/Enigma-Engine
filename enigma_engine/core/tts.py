"""Text-to-speech primitive -- the voice organ.

First organ of the hub direction: capabilities live as services the model
reaches through tools, not as weights. serve_enigma.py exposes this as the
`speak` built-in tool and the /v1/audio/speech endpoint (both behind --voice).

Backend is pyttsx3 (SAPI5 on Windows) -- the `voice` optional-dependency
group in pyproject.toml. The import happens lazily inside Speaker so this
module always imports cleanly; a missing backend fails honestly as TTSError
when a Speaker is actually constructed.

Threading: SAPI COM objects are apartment-threaded, so EVERY engine touch
(including construction) happens on one dedicated worker thread. Public
methods enqueue jobs; utterances play strictly in order instead of over
each other.

One engine per JOB, not per Speaker: on this box's SAPI driver, say() then
save_to_file() on the same engine hangs the second runAndWait() forever
(measured 2026-07-14; repeat-say and wav-only sequences are fine, the MIX
deadlocks). A fresh engine per job costs tens of ms and sidesteps that bug
plus pyttsx3's other stale-loop states. pyttsx3.init() caches engines in a
weakref dict, so dropping our reference is what makes the next init fresh.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path

# A runaway generation could ask for a minutes-long monologue; cap and say so.
MAX_SPEAK_CHARS = 1000

_INIT_TIMEOUT_S = 15.0
_JOB_TIMEOUT_S = 120.0  # WAV synthesis of a long utterance is slow, not hung


class TTSError(Exception):
    """The voice backend is unavailable or a synthesis job failed."""


def _default_engine_factory():
    try:
        import pyttsx3
    except ImportError as exc:
        raise TTSError(
            "pyttsx3 not installed -- pip install 'enigma-engine[voice]'"
        ) from exc
    return pyttsx3.init()


@dataclass
class _Job:
    text: str
    wav_path: str | None = None  # None = play through speakers
    done: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None


class Speaker:
    """Serialized text-to-speech through one background engine thread.

    speak() is fire-and-forget by default so tool execution never blocks
    generation for the seconds an utterance takes; pass wait=True (tests,
    scripts) to block until the utterance finished and surface its error.
    """

    def __init__(self, rate: int | None = None, voice: str | None = None, engine_factory=None):
        self._factory = engine_factory or _default_engine_factory
        self._rate = rate
        self._voice = voice
        self._queue: queue.Queue[_Job | None] = queue.Queue()
        self.voices: list[dict] = []  # [{"id", "name"}] discovered at init
        self.last_error: BaseException | None = None

        init_done = threading.Event()
        init_error: list[BaseException] = []
        self._worker = threading.Thread(
            target=self._run, args=(init_done, init_error), daemon=True, name="enigma-tts"
        )
        self._worker.start()
        if not init_done.wait(_INIT_TIMEOUT_S):
            raise TTSError("voice engine did not initialize within timeout")
        if init_error:
            raise TTSError(f"voice engine failed to initialize: {init_error[0]}") from init_error[0]

    # ------------------------------------------------------------- worker

    def _new_engine(self):
        engine = self._factory()
        if self._rate is not None:
            engine.setProperty("rate", self._rate)
        if self._voice is not None:
            engine.setProperty("voice", self._voice)
        return engine

    def _run(self, init_done: threading.Event, init_error: list[BaseException]) -> None:
        try:
            probe = self._new_engine()  # validate the backend before accepting jobs
            try:
                self.voices = [
                    {"id": v.id, "name": getattr(v, "name", v.id)}
                    for v in probe.getProperty("voices")
                ]
            except Exception:
                self.voices = []  # discovery is best-effort; speaking still works
            del probe  # clear pyttsx3's cache so job 1 gets a fresh engine
        except BaseException as exc:  # surfaced to the constructor
            init_error.append(exc)
            init_done.set()
            return
        init_done.set()

        while True:
            job = self._queue.get()
            if job is None:  # close() sentinel
                return
            engine = None
            try:
                engine = self._new_engine()  # per job -- see module docstring
                if job.wav_path is not None:
                    engine.save_to_file(job.text, job.wav_path)
                else:
                    engine.say(job.text)
                engine.runAndWait()
            except BaseException as exc:  # job errors are data
                job.error = exc
                self.last_error = exc
                print(f"  WARN: tts job failed: {exc}", flush=True)
            finally:
                del engine  # a broken engine must not survive into the next job
                job.done.set()

    # ------------------------------------------------------------- public

    def speak(self, text: str, wait: bool = False) -> str:
        """Queue an utterance. Returns the text actually spoken (post-cap)."""
        text = self._clip(text)
        job = _Job(text=text)
        self._queue.put(job)
        if wait:
            self._finish(job)
        return text

    def save_wav(self, text: str, path: str | Path) -> Path:
        """Synthesize to a WAV file and block until it is written."""
        path = Path(path)
        job = _Job(text=self._clip(text), wav_path=str(path))
        self._queue.put(job)
        self._finish(job)
        if not path.exists():
            raise TTSError(f"engine reported success but wrote no file: {path}")
        return path

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
