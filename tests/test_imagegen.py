"""Imagination organ (core/imagegen.py): generation through a fake diffusers
pipeline -- turbo vs standard step defaults, prompt validation, PNG on disk.
No model download, no GPU."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from PIL import Image

from enigma_engine.core.imagegen import ImageGenError, Painter


class FakePipe:
    def __init__(self, fail=False):
        self.fail = fail
        self.calls: list[dict] = []

    def __call__(self, prompt, num_inference_steps, guidance_scale, width, height):
        if self.fail:
            raise RuntimeError("synthetic diffusion failure")
        self.calls.append(
            {
                "prompt": prompt,
                "steps": num_inference_steps,
                "guidance": guidance_scale,
                "size": (width, height),
            }
        )
        return SimpleNamespace(images=[Image.new("RGB", (width, height), (10, 20, 30))])


def test_generate_writes_png(tmp_path):
    pipe = FakePipe()
    painter = Painter(pipe_factory=lambda: pipe)
    out = painter.generate("a red apple", tmp_path / "apple.png")
    assert out.exists()
    assert Image.open(out).size == (512, 512)


def test_turbo_defaults_one_step_no_guidance(tmp_path):
    pipe = FakePipe()
    painter = Painter(model_id="stabilityai/sd-turbo", pipe_factory=lambda: pipe)
    painter.generate("x", tmp_path / "a.png")
    assert pipe.calls[0]["steps"] == 1
    assert pipe.calls[0]["guidance"] == 0.0


def test_standard_model_defaults_full_schedule(tmp_path):
    pipe = FakePipe()
    painter = Painter(model_id="runwayml/stable-diffusion-v1-5", pipe_factory=lambda: pipe)
    painter.generate("x", tmp_path / "a.png")
    assert pipe.calls[0]["steps"] == 30
    assert pipe.calls[0]["guidance"] == 7.5


def test_empty_prompt_is_refused(tmp_path):
    painter = Painter(pipe_factory=FakePipe)
    with pytest.raises(ImageGenError, match="empty prompt"):
        painter.generate("   ", tmp_path / "a.png")


def test_pipe_failure_is_wrapped(tmp_path):
    painter = Painter(pipe_factory=lambda: FakePipe(fail=True))
    with pytest.raises(ImageGenError, match="synthetic diffusion failure"):
        painter.generate("x", tmp_path / "a.png")


def test_a_save_that_raises_valueerror_is_wrapped(tmp_path):
    """PIL answers a path whose extension names no writer with a ValueError,
    not an OSError -- and the save wrapper caught only OSError, so a bare
    ValueError escaped the organ's error type. serve's narrow `except
    ImageGenError` then turned an `imagine` into a 500 mid-conversation, the
    one thing the builtin contract promises never happens."""
    painter = Painter(pipe_factory=FakePipe)
    with pytest.raises(ImageGenError, match="could not write the image"):
        painter.generate("x", tmp_path / "a.xyz")


def test_creates_parent_dirs(tmp_path):
    painter = Painter(pipe_factory=FakePipe)
    out = painter.generate("x", tmp_path / "deep" / "nested" / "a.png")
    assert out.exists()
