"""Barge-in: stop her the moment you start talking over her.

While Enigma is speaking, a mic stream feeds short-window loudness (RMS) to a
BargeInDetector; sustained loudness above a threshold means you cut in, and the
stop callback fires. The mic is opened only while she talks (bind set_active to
Speaker.set_on_speaking), so it is not listening the rest of the time.

HONEST LIMITS -- why this is OFF by default and wants one live tuning pass:
- With SPEAKERS the mic also hears HER voice, so the threshold must sit above
  that bleed or she cuts herself off on her own words. Robust speaker use needs
  acoustic echo cancellation (AEC), which this does NOT do. With HEADPHONES
  there is no bleed and energy VAD works well.
- The right threshold depends on your mic, room, and volume -- tune it live. The
  DECISION logic here is unit-tested; the acoustic values are not (they cannot
  be, without your setup making real sound).
"""

from __future__ import annotations

# Defaults: 16 kHz mono, 50 ms blocks. threshold is RMS on float32 [-1, 1];
# 0.02 is a rough "someone is talking" level for a close mic on headphones.
DEFAULT_SAMPLERATE = 16000
DEFAULT_BLOCK = 800
DEFAULT_THRESHOLD = 0.02
DEFAULT_MIN_SPEECH_S = 0.25  # ignore taps/clicks; require a quarter-second of voice


class BargeInDetector:
    """Fires once loudness stays above threshold for min_speech_s of audio.

    Pure and deterministic: feed it (rms, dt) pairs. A gap of quiet resets the
    accumulator, so only *sustained* input trips it -- not a single click.
    """

    def __init__(self, threshold: float = DEFAULT_THRESHOLD, min_speech_s: float = DEFAULT_MIN_SPEECH_S):
        self.threshold = float(threshold)
        self.min_speech_s = float(min_speech_s)
        self._accum = 0.0

    def reset(self) -> None:
        self._accum = 0.0

    def feed(self, rms: float, dt: float) -> bool:
        if rms >= self.threshold:
            self._accum += dt
            if self._accum >= self.min_speech_s:
                self._accum = 0.0
                return True
        else:
            self._accum = 0.0
        return False


class MicBargeIn:
    """Ties a mic input stream to a BargeInDetector and a stop callback.

    stream_factory(callback) -> stream is injectable; callback takes one arg, a
    block of float32 samples. The default opens a sounddevice InputStream. Call
    set_active(True/False) to open/close the mic (wire it to Speaker's
    speaking-state callback so it listens only while she talks).
    """

    def __init__(
        self,
        on_detect,
        threshold: float = DEFAULT_THRESHOLD,
        min_speech_s: float = DEFAULT_MIN_SPEECH_S,
        samplerate: int = DEFAULT_SAMPLERATE,
        block: int = DEFAULT_BLOCK,
        stream_factory=None,
    ):
        self._on_detect = on_detect
        self._detector = BargeInDetector(threshold, min_speech_s)
        self._samplerate = samplerate
        self._block = block
        self._stream_factory = stream_factory
        self._stream = None
        self._active = False  # gates late in-flight callbacks after stop()
        self._fired = False   # one detection per utterance -- stop() is idempotent
        self._warned_start = False  # the mic-unavailable WARN prints once, not per utterance

    def _process_block(self, samples) -> None:
        # PortAudio can deliver a callback already in flight when stop() runs;
        # an inactive or already-fired window must not fire again -- a stale
        # detection here would bump the Speaker epoch and silently cancel the
        # NEXT queued utterance, not the one barged in on.
        if not self._active or self._fired:
            return
        import numpy as np

        a = np.asarray(samples, dtype=np.float32).reshape(-1)
        if a.size == 0:
            return
        rms = float(np.sqrt(np.mean(a * a)))
        if self._detector.feed(rms, a.size / self._samplerate):
            self._fired = True
            self._on_detect()

    def _make_stream(self):
        if self._stream_factory is not None:
            return self._stream_factory(self._process_block)
        import sounddevice as sd

        def _cb(indata, frames, time_info, status):
            self._process_block(indata)

        return sd.InputStream(
            samplerate=self._samplerate, blocksize=self._block,
            channels=1, dtype="float32", callback=_cb,
        )

    def set_active(self, active: bool) -> None:
        if active:
            self.start()
        else:
            self.stop()

    def start(self) -> None:
        if self._stream is not None:
            return
        self._detector.reset()
        self._fired = False
        try:
            self._stream = self._make_stream()
            self._stream.start()
        except Exception as exc:
            # A busy/missing mic must not break playback (the caller swallows
            # exceptions) -- but a permanently unavailable mic means barge-in
            # never works, so say so once instead of failing silently forever.
            self._stream = None
            self._active = False
            if not self._warned_start:
                self._warned_start = True
                print(f"  WARN: barge-in mic unavailable ({exc}) -- she cannot be interrupted by voice", flush=True)
            return
        self._active = True

    def stop(self) -> None:
        self._active = False  # gate first: an in-flight callback must not fire
        if self._stream is None:
            return
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:  # a flaky device must not wedge the next utterance
            pass
        self._stream = None
        self._detector.reset()
