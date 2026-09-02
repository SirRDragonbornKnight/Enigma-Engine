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


# The test above reads SOURCE, so it survives a guard whose body has been
# emptied (verified 2026-09-02: both guards neutered to a bare `return` and it
# still passed). What follows pins the BEHAVIOR, on the artifact names these
# writers really use, and the ordering that makes a guard worth having.


@pytest.mark.parametrize("name", ["model.pth", "latest.pth", "prev.pth"])
def test_the_distills_rotation_names_are_each_refused(tmp_path, name):
    """The distills write model.pth + latest.pth (prev.pth is the rotation).
    An interrupted run leaves one of the three and no model.pth, and those are
    receipts too -- any one of them must stop a bare rerun."""
    from enigma_engine.core.safe_save import refuse_existing_artifact

    d = tmp_path / name.replace(".", "_")
    d.mkdir()
    (d / name).write_bytes(b"x")
    with pytest.raises(SystemExit) as e:
        refuse_existing_artifact(d)
    assert name in str(e.value)


def test_the_artifact_guard_lets_a_fresh_run_through(tmp_path):
    from enigma_engine.core.safe_save import refuse_existing_artifact

    refuse_existing_artifact(tmp_path / "brand_new_run")       # missing dir
    empty = tmp_path / "empty_run"
    empty.mkdir()
    refuse_existing_artifact(empty)                            # nothing in it
    other = tmp_path / "unrelated"
    other.mkdir()
    (other / "notes.txt").write_bytes(b"x")
    refuse_existing_artifact(other)                            # non-artifacts are fine


def test_the_artifact_guard_refuses_a_file_shaped_out(tmp_path):
    """The model.pth-typo class: <file>/model.pth never exists, so a dir-only
    check passed and a DPO run trained to completion before dying on it."""
    from enigma_engine.core.safe_save import refuse_existing_artifact

    with pytest.raises(SystemExit):
        refuse_existing_artifact(tmp_path / "run.pth")


@pytest.mark.parametrize("name", ["sft2_vision_best.pt", "sft2_audio_best.pt"])
def test_the_aligns_real_best_names_are_refused(tmp_path, name):
    """The aligns write {stem}_{modality}_best.pt -- run-derived names no fixed
    list would catch, which is why they take the pattern form of the refusal."""
    d = tmp_path / "align_run"
    d.mkdir()
    (d / name).write_bytes(b"x")
    with pytest.raises(SystemExit) as e:
        refuse_existing_patterns(d, ("*.pt",))
    assert name in str(e.value)


def test_every_writer_guards_before_it_writes():
    """A guard that runs AFTER the first save is decoration. Compares the line
    of the guard call against the first save call in each writer's source."""
    import ast
    import inspect

    for mod, guard in (
        ("distill_vision_encoder", "refuse_existing_artifact"),
        ("distill_audio_encoder", "refuse_existing_artifact"),
        ("align_vision", "refuse_existing_patterns"),
        ("align_audio", "refuse_existing_patterns"),
    ):
        tree = ast.parse(inspect.getsource(__import__(mod)))
        guards, saves = [], []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            name = getattr(fn, "id", None) or getattr(fn, "attr", None)
            if name == guard:
                guards.append(node.lineno)
            elif name in ("atomic_torch_save", "atomic_safetensors_save"):
                saves.append(node.lineno)
        assert guards, f"{mod} has no {guard} call at all"
        if saves:
            assert min(guards) < min(saves), (
                f"{mod} calls {guard} at line {min(guards)} but writes first at line {min(saves)}"
            )
