"""Eyes organ (core/eyes.py): captioning through a fake pipeline and the pure
OpenAI-multimodal-content flattener. No transformers download, no GPU."""

from __future__ import annotations

import base64
import io

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
