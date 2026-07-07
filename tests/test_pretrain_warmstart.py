"""Guards on pretrain_enigma.py --init-from: a warm-start is a NEW run and
must never overwrite the model it started from. These lock the guards that
fire BEFORE any checkpoint load, so they need no GPU, corpus, or real weights.
"""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "pretrain_enigma.py"


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.skipif(not SCRIPT.exists(), reason="pretrain script missing")
def test_resume_and_init_from_are_mutually_exclusive():
    r = _run(["--resume", "a.pth", "--init-from", "b.pth", "--out", "models/_x"])
    assert r.returncode != 0
    assert "mutually exclusive" in (r.stdout + r.stderr)


@pytest.mark.skipif(not SCRIPT.exists(), reason="pretrain script missing")
def test_init_from_requires_explicit_out():
    r = _run(["--init-from", "some/model.pth", "--block", "2048"])
    assert r.returncode != 0
    assert "requires an explicit --out" in (r.stdout + r.stderr)


@pytest.mark.skipif(not SCRIPT.exists(), reason="pretrain script missing")
def test_init_from_refuses_to_write_into_source_dir():
    # --out equals the source checkpoint's own directory: pure path check,
    # no file needs to exist.
    r = _run(["--init-from", "foo/model.pth", "--out", "foo"])
    assert r.returncode != 0
    assert "source model's directory" in (r.stdout + r.stderr)
