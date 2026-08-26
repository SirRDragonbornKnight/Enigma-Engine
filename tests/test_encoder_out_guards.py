"""Output guards on the encoder writers: the two distills (model.pth /
latest.pth, the three-name set) and the two aligns, whose artifacts are
run-derived ({stem}_{modality}_best.pt and friends) and so need the pattern
form of the same refusal. Model artifacts are versioned, never rebuilt in
place; the sanctioned exceptions (--resume into its OWN dir, --sanity) are
decided by the callers before the guard runs.
"""

from __future__ import annotations

import pytest

from enigma_engine.core.safe_save import refuse_existing_patterns


def test_pattern_guard_refuses_first_hit(tmp_path):
    d = tmp_path / "align"
    d.mkdir()
    (d / "align_vision_best.pt").write_bytes(b"x")
    with pytest.raises(SystemExit) as e:
        refuse_existing_patterns(d, ("*.pt",))
    assert "align_vision_best.pt" in str(e.value)


def test_pattern_guard_passes_missing_and_empty(tmp_path):
    refuse_existing_patterns(tmp_path / "nope", ("*.pt",))
    d = tmp_path / "empty"
    d.mkdir()
    refuse_existing_patterns(d, ("*.pt",))


def test_pattern_guard_refuses_file_shaped_out(tmp_path):
    f = tmp_path / "out.pt"
    f.write_bytes(b"x")
    with pytest.raises(SystemExit):
        refuse_existing_patterns(f, ("*.pt",))


def test_all_four_writers_carry_a_guard():
    import inspect

    for mod, needle in (
        ("distill_vision_encoder", "refuse_existing_artifact("),
        ("distill_audio_encoder", "refuse_existing_artifact("),
        ("align_vision", "refuse_existing_patterns("),
        ("align_audio", "refuse_existing_patterns("),
    ):
        src = inspect.getsource(__import__(mod))
        assert needle in src, f"{mod} lost its output guard"
