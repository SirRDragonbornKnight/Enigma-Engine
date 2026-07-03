"""Repository hygiene regression gates.

Pass 156z9cv (May 11, 2026): Mojibake regression gate.

History: SUGGESTIONS stamps 156z9cr/cs/ct/cu all logged "mojibake at
inference.py L1167-1168" as a small bounded site. The May 11 audit
re-grepped and found 2341 mojibake markers in inference.py and 7 in
rl_training.py — three orders of magnitude beyond what was logged.
Root cause was a cp1252-mis-decoded-as-UTF-8 round-trip on the
inference.py module docstring ASCII art (and a smaller version on
rl_training.py).  Both files were re-fixed via ``ftfy.fix_text`` plus
one surgical replace for a triple-encoded ``⚡`` sequence.

This test prevents a future re-introduction by flagging the
canonical mojibake triad characters (``â``, ``Â``, ``Ã``) appearing
anywhere in package source.  All three are valid in legitimate
text (Portuguese, French, etc.) — but in a primarily English code
base they are overwhelmingly a tell of double-encoded UTF-8.  The
gate is intentionally narrow:

  * Allow legitimate Unicode (box drawing, arrows, bullets, math).
  * Reject the three "mojibake leading bytes" that almost never
    appear in honest English source.

If a future change legitimately needs one of these characters in
a string literal or comment (e.g. a non-English test fixture or a
deliberate test of mojibake detection), allowlist that file path
inside ``ALLOWED_FILES`` with a clear comment justifying it.
"""

from __future__ import annotations

import pathlib

# Files explicitly allowed to contain mojibake triad characters.
# Add a comment when listing a file here so future readers understand
# why the exception exists.
ALLOWED_FILES: set[str] = set()

# The three canonical mojibake "leading bytes" when cp1252 text is
# mis-decoded as UTF-8 and then re-encoded as UTF-8.  In honest
# English source these almost never appear; when they do, it is
# nearly always double-encoded text.
MOJIBAKE_TRIAD: tuple[str, ...] = ("â", "Â", "Ã")

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent.parent / "enigma_engine"


def test_package_source_is_free_of_mojibake_markers() -> None:
    """Every ``.py`` file under ``enigma_engine/`` must be free of
    the three canonical mojibake leading bytes.

    Pass 156z9cv: regression gate for the inference.py + rl_training.py
    mojibake corruption that was under-reported in prior stamps.
    """
    assert PACKAGE_ROOT.is_dir(), f"package root missing: {PACKAGE_ROOT}"

    failures: list[tuple[str, dict[str, int]]] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        rel = path.relative_to(PACKAGE_ROOT.parent).as_posix()
        if rel in ALLOWED_FILES:
            continue
        text = path.read_text(encoding="utf-8")
        counts = {ch: text.count(ch) for ch in MOJIBAKE_TRIAD if ch in text}
        if counts:
            failures.append((rel, counts))

    if failures:
        lines = ["mojibake markers found in package source:"]
        for rel, counts in failures:
            summary = ", ".join(f"{ch!r}={n}" for ch, n in counts.items())
            lines.append(f"  {rel}: {summary}")
        lines.append(
            "If a character is legitimate, add the file path to "
            "ALLOWED_FILES in tests/test_repo_hygiene.py with a "
            "justification comment."
        )
        raise AssertionError("\n".join(lines))
