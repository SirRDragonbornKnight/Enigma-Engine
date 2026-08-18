"""The suite's own guard, tested.

`tests/conftest.py` fingerprints the real `~/.enigma_engine` before and after
the session and fails the run if anything moved -- the one check that catches a
boot test which forgot the hermetic-state patch and read, migrated or wrote the
developer's live data home. It runs exactly once, at session teardown, on a
tree that is supposed to be clean: a guard that has never fired is a guard
nobody has seen work, and its comparison is where it can be quietly wrong.

So the comparison is a pure function (`_diff`) and this module exercises it on
real directories, one damage shape at a time.
"""

from __future__ import annotations

import os

from tests.conftest import _diff, _fingerprint


def _home(tmp_path):
    """A stand-in data home with a file at the root and one nested."""
    root = tmp_path / ".enigma_engine"
    (root / "images").mkdir(parents=True)
    (root / "voice.json").write_text('{"blend": []}', encoding="utf-8")
    (root / "images" / "cat.png").write_bytes(b"not really a png")
    return root


def test_the_imported_conftest_is_the_one_pytest_loaded(request):
    """`from tests.conftest import ...` must reach the SAME module object
    pytest registered, not a second copy of it -- a duplicate would register a
    second session fixture and run the guard twice against a home the first
    copy had already fingerprinted."""
    import tests.conftest

    assert tests.conftest in request.config.pluginmanager.get_plugins()


def test_an_untouched_tree_reports_nothing(tmp_path):
    """Zero behavior change for a clean suite is the whole licence for an
    autouse session fixture -- it may only ever ADD a failure."""
    root = _home(tmp_path)
    before = _fingerprint(root)
    assert _diff(before, _fingerprint(root)) == ([], [], [])


def test_a_file_the_suite_left_behind_is_named(tmp_path):
    root = _home(tmp_path)
    before = _fingerprint(root)
    (root / "images" / "leftover.png").write_bytes(b"written by a test")
    assert _diff(before, _fingerprint(root)) == (["images/leftover.png"], [], [])


def test_a_file_the_suite_deleted_is_named(tmp_path):
    root = _home(tmp_path)
    before = _fingerprint(root)
    (root / "voice.json").unlink()
    assert _diff(before, _fingerprint(root)) == ([], ["voice.json"], [])


def test_a_rewritten_file_is_named_by_its_size(tmp_path):
    root = _home(tmp_path)
    before = _fingerprint(root)
    (root / "voice.json").write_text('{"blend": ["af_heart"]}', encoding="utf-8")
    assert _diff(before, _fingerprint(root)) == ([], [], ["voice.json"])


def test_a_rewrite_that_kept_its_size_is_named_by_its_mtime(tmp_path):
    """Size alone would miss an in-place edit of the same length, and every
    runtime state file in that home is a small JSON object whose length barely
    moves -- a mute flag flipping is a one-byte difference at most."""
    root = _home(tmp_path)
    before = _fingerprint(root)
    path = root / "voice.json"
    path.write_text('{"blend":[1]}', encoding="utf-8")
    assert path.stat().st_size == before["voice.json"][0], "same size, on purpose"
    os.utime(path, ns=(path.stat().st_atime_ns, before["voice.json"][1] + 1_000_000_000))
    assert _diff(before, _fingerprint(root)) == ([], [], ["voice.json"])


def test_a_data_home_the_suite_creates_out_of_nothing_is_seen(tmp_path):
    """The root is an entry of its own (None when absent), so a suite that
    CREATES a home the developer does not have is caught too -- otherwise the
    before-fingerprint would be empty, the after-fingerprint would be empty of
    files, and a directory nobody asked for would pass."""
    root = tmp_path / ".enigma_engine"
    before = _fingerprint(root)
    assert before == {".": None}
    root.mkdir()
    assert _diff(before, _fingerprint(root)) == ([], [], ["."])
