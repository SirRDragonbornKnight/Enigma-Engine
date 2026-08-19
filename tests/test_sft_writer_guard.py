"""make_sft_data's artifact writes: rotate-then-write + build manifest.

Round-2 review (2026-08-13): the builder was the LAST artifact writer with
no receipt protection -- a bare `python make_sft_data.py` truncated
data/sft/mix.jsonl (the SERVED model's training receipt) in place, and the
build flags lived only in stdout, so the shipped mix could not even be
identified after the fact.
"""
from __future__ import annotations

from make_sft_data import _write_artifact


def test_rebuild_rotates_the_previous_artifact(tmp_path):
    p = tmp_path / "mix.jsonl"
    _write_artifact(p, "first generation\n")
    assert p.read_text(encoding="utf-8") == "first generation\n"

    _write_artifact(p, "second generation\n")
    assert p.read_text(encoding="utf-8") == "second generation\n"
    prev = tmp_path / "mix.prev.jsonl"
    assert prev.read_text(encoding="utf-8") == "first generation\n", \
        "the previous receipt did not survive the rebuild"

    _write_artifact(p, "third generation\n")
    assert prev.read_text(encoding="utf-8") == "second generation\n"


def test_writer_leaves_no_tmp_droppings(tmp_path):
    p = tmp_path / "mix.jsonl"
    _write_artifact(p, "content\n")
    _write_artifact(p, "content 2\n")
    assert [f.name for f in sorted(tmp_path.iterdir())] == ["mix.jsonl", "mix.prev.jsonl"]


def test_main_writes_through_the_rotating_writer():
    """Wiring pin: the three artifact writes must route through
    _write_artifact -- a direct write_text on any of them reopens the
    clobber. (Source pin; running main() is a full build.)

    `out_dir` is main()'s resolved destination -- `OUT_DIR` unless --out named
    one. Pinned on that name rather than the constant: a write spelled with
    the constant would put a pack's artifacts in HER directory, which is the
    same clobber this file exists for, one argument further out."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "make_sft_data.py").read_text(encoding="utf-8")
    for name in ("tool_calls.jsonl", "identity.jsonl", "mix.jsonl"):
        assert f'_write_artifact(out_dir / "{name}"' in src, \
            f"{name} is no longer written through the rotating writer"
    assert 'mix.manifest.json' in src, "the build manifest write disappeared"
