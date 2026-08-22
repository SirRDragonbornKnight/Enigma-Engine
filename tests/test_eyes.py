"""Eyes organ (core/eyes.py): captioning through a fake pipeline and the pure
OpenAI-multimodal-content flattener. No transformers download, no GPU."""

from __future__ import annotations

import base64
import io
import threading
import time

import pytest
from PIL import Image

from enigma_engine.core.eyes import Eyes, EyesError, flatten_image_content


def _png_bytes() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (200, 30, 30)).save(buf, format="PNG")
    return buf.getvalue()


class FakeCaptioner:
    def __init__(self, caption="a red square", fail=False):
        self.caption = caption
        self.fail = fail

    def __call__(self, img):
        if self.fail:
            raise RuntimeError("synthetic caption failure")
        assert img.mode == "RGB"
        return self.caption


def test_describe_bytes():
    eyes = Eyes(captioner_factory=FakeCaptioner)
    assert eyes.describe(_png_bytes()) == "a red square"


def test_describe_path(tmp_path):
    p = tmp_path / "img.png"
    p.write_bytes(_png_bytes())
    eyes = Eyes(captioner_factory=FakeCaptioner)
    assert eyes.describe(p) == "a red square"


def test_garbage_bytes_are_refused():
    eyes = Eyes(captioner_factory=FakeCaptioner)
    with pytest.raises(EyesError, match="could not read image"):
        eyes.describe(b"not an image at all")


def test_captioner_failure_is_wrapped():
    eyes = Eyes(captioner_factory=lambda: FakeCaptioner(fail=True))
    with pytest.raises(EyesError, match="synthetic caption failure"):
        eyes.describe(_png_bytes())


def test_empty_caption_is_refused():
    eyes = Eyes(captioner_factory=lambda: FakeCaptioner(caption="  "))
    with pytest.raises(EyesError, match="returned nothing"):
        eyes.describe(_png_bytes())


# ------------------------------------------------------------ caption cache


class CountingCaptioner:
    """Counts the captions actually COMPUTED -- one per full forward pass."""

    def __init__(self):
        self.calls = 0

    def __call__(self, img):
        self.calls += 1
        return f"caption {self.calls}"


def _png(color) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), color).save(buf, format="PNG")
    return buf.getvalue()


def test_the_same_image_is_captioned_once_per_process():
    """Every chat turn resends the WHOLE history and serve flattens every
    message on every request, so an image three turns back was re-captioned on
    each new message -- up to 48 full forwards, holding the shared generation
    lock, before she could answer at all (audit 2026-08-22)."""
    counter = CountingCaptioner()
    eyes = Eyes(captioner_factory=lambda: counter)
    img = _png_bytes()

    assert eyes.describe(img) == "caption 1"
    assert eyes.describe(img) == "caption 1"  # the cached bytes, not a second pass
    assert counter.calls == 1

    assert eyes.describe(_png((10, 200, 10))) == "caption 2"  # a DIFFERENT image still captions
    assert counter.calls == 2


def test_one_history_captions_each_distinct_image_once():
    """The shape the chat path actually sends: the same picture repeated across
    turns, plus a new one. The cache is keyed on the payload, so the repeat is
    free and the new image is still seen."""
    counter = CountingCaptioner()
    eyes = Eyes(captioner_factory=lambda: counter)
    first, second = _data_url(), "data:image/png;base64," + base64.b64encode(_png((9, 9, 200))).decode("ascii")

    def turn(url):
        return flatten_image_content(
            [{"type": "text", "text": "and this?"}, {"type": "image_url", "image_url": {"url": url}}],
            describe=eyes.describe,
        )

    a = turn(first)
    assert turn(first) == a  # a later request re-flattening the same history
    assert counter.calls == 1
    turn(second)
    assert counter.calls == 2
    turn(first)  # ...and the first one is still remembered
    assert counter.calls == 2


def test_the_cache_is_bounded_and_evicts_the_least_recently_used():
    counter = CountingCaptioner()
    eyes = Eyes(captioner_factory=lambda: counter, cache_size=2)
    a, b, c = _png((1, 1, 1)), _png((2, 2, 2)), _png((3, 3, 3))

    eyes.describe(a)
    eyes.describe(b)
    eyes.describe(a)  # a is now the most recently USED, b the oldest
    assert counter.calls == 2
    eyes.describe(c)  # evicts b, not a
    assert counter.calls == 3
    eyes.describe(a)
    assert counter.calls == 3, "the most recently used entry was evicted"
    eyes.describe(b)
    assert counter.calls == 4, "an evicted image must simply caption again"


def test_a_failed_caption_is_not_remembered():
    """A failure is worth retrying, not caching -- the flatten path turns it
    into an honest '[image error: ...]' marker and the next turn tries again."""
    captioner = FakeCaptioner(fail=True)
    eyes = Eyes(captioner_factory=lambda: captioner)
    img = _png_bytes()
    for _ in range(2):
        with pytest.raises(EyesError, match="synthetic caption failure"):
            eyes.describe(img)
    captioner.fail = False
    assert eyes.describe(img) == "a red square"


# ------------------------------------------------------- concurrent callers


class BlockingCaptioner:
    """Counts computations and holds every one open until it is released, so a
    test can pin one caption 'in flight' while the other callers arrive."""

    def __init__(self, fail: bool = False):
        self.calls = 0
        self.fail = fail
        self.entered = threading.Event()
        self.release = threading.Event()

    def __call__(self, img):
        self.calls += 1
        self.entered.set()
        self.release.wait(5)
        if self.fail:
            raise RuntimeError("synthetic caption failure")
        return "a shared caption"


def _run_together(target, n: int):
    """Start n threads on target(); return (threads, started, results).

    A raised exception lands in `results` like any other outcome -- these
    tests are about what every caller ENDED UP with, failures included."""
    out: list = []
    started: list = []

    def call():
        started.append(1)
        try:
            out.append(target())
        except BaseException as exc:
            out.append(exc)

    threads = [threading.Thread(target=call) for _ in range(n)]
    for t in threads:
        t.start()
    return threads, started, out


def test_concurrent_callers_of_one_image_caption_it_once():
    """The cache only helped the SECOND turn. One picture arriving on several
    requests at once measured eight full captioning passes for eight threads
    (audit 2026-08-22) -- every one of them holding the shared generation lock
    ahead of the answer the user was waiting for. The first caller in computes;
    the rest wait on it and share the result."""
    captioner = BlockingCaptioner()
    eyes = Eyes(captioner_factory=lambda: captioner)
    img = _png_bytes()

    threads, started, out = _run_together(lambda: eyes.describe(img), 8)
    try:
        assert captioner.entered.wait(5), "no caller reached the captioner"
        deadline = time.time() + 5
        while len(started) < 8 and time.time() < deadline:
            time.sleep(0.01)
        time.sleep(0.1)  # let the seven followers reach the in-flight wait
    finally:
        captioner.release.set()
    for t in threads:
        t.join(5)

    assert not [t for t in threads if t.is_alive()], "a caller never returned"
    assert out == ["a shared caption"] * 8
    assert captioner.calls == 1


def test_a_failed_in_flight_caption_does_not_wedge_the_callers_waiting_on_it():
    """A shared compute must share its FAILURE too: waiters that never learn
    the leader died would hold the request thread until the client gave up.
    The slot is released either way, so the next call simply tries again."""
    captioner = BlockingCaptioner(fail=True)
    eyes = Eyes(captioner_factory=lambda: captioner)
    img = _png_bytes()

    threads, started, out = _run_together(lambda: eyes.describe(img), 8)
    try:
        assert captioner.entered.wait(5), "no caller reached the captioner"
        deadline = time.time() + 5
        while len(started) < 8 and time.time() < deadline:
            time.sleep(0.01)
        time.sleep(0.1)
    finally:
        captioner.release.set()
    for t in threads:
        t.join(5)

    assert not [t for t in threads if t.is_alive()], "a waiter is wedged on a dead caption"
    assert len(out) == 8 and all(isinstance(r, EyesError) for r in out)
    assert captioner.calls == 1

    captioner.fail = False  # the in-flight slot was released, so a retry works
    assert eyes.describe(img) == "a shared caption"
    assert captioner.calls == 2


def test_distinct_images_are_each_captioned_once_under_concurrency():
    """Sharing is keyed on the PAYLOAD: four different pictures owe four
    captions, and none of the four callers may collect another's answer."""
    counter = CountingCaptioner()
    eyes = Eyes(captioner_factory=lambda: counter)
    payloads = [_png((30 * i + 5, 7, 9)) for i in range(4)]
    seen: dict[int, str] = {}

    def call(i):
        seen[i] = eyes.describe(payloads[i])

    threads = [threading.Thread(target=call, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)

    assert not [t for t in threads if t.is_alive()]
    assert counter.calls == 4
    assert len(set(seen.values())) == 4, "two images shared one caption"


def test_the_cache_lock_is_free_while_a_caption_is_being_computed():
    """The cache lock must never nest around the caption, which takes the
    shared GENERATION lock: a cache that queued its hits behind generation
    would be slower than having no cache at all."""
    captioner = BlockingCaptioner()
    eyes = Eyes(captioner_factory=lambda: captioner)
    t = threading.Thread(target=lambda: eyes.describe(_png_bytes()))
    t.start()
    try:
        assert captioner.entered.wait(5), "the captioner was never reached"
        assert eyes._cache_lock.acquire(timeout=1), "the cache lock is held across the caption"
        eyes._cache_lock.release()
    finally:
        captioner.release.set()
        t.join(5)
    assert not t.is_alive()


# ---------------------------------------------------------------- flattener


def _data_url() -> str:
    return "data:image/png;base64," + base64.b64encode(_png_bytes()).decode("ascii")


def test_flatten_captions_data_url():
    content = [
        {"type": "text", "text": "What is this?"},
        {"type": "image_url", "image_url": {"url": _data_url()}},
    ]
    out = flatten_image_content(content, describe=lambda b: "a red square")
    assert out == "What is this?\n[image: a red square]"


def test_flatten_without_eyes_marks_honestly():
    content = [{"type": "image_url", "image_url": {"url": _data_url()}}]
    out = flatten_image_content(content, describe=None)
    assert out == "[image ignored -- eyes disabled (start serve with --eyes)]"


def test_flatten_refuses_remote_urls():
    content = [{"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}}]
    out = flatten_image_content(content, describe=lambda b: "never called")
    assert out == "[image ignored -- only data: URLs are supported]"


def test_flatten_surfaces_describe_errors():
    def broken(_b):
        raise EyesError("captioner down")

    out = flatten_image_content([{"type": "image_url", "image_url": {"url": _data_url()}}], describe=broken)
    assert out == "[image error: captioner down]"


def test_flatten_passes_plain_text_parts():
    assert flatten_image_content(["hi", {"type": "text", "text": "there"}]) == "hi\nthere"


def test_flatten_marks_unknown_part_types():
    out = flatten_image_content([{"type": "input_audio", "data": "..."}])
    assert out == "[unsupported content part: 'input_audio']"
