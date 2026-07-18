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

import ast
import pathlib

# Files explicitly allowed to contain mojibake triad characters.
# Add a comment when listing a file here so future readers understand
# why the exception exists.
ALLOWED_FILES: set[str] = {
    # this very file defines the triad literals it hunts
    "tests/test_repo_hygiene.py",
}

# The three canonical mojibake "leading bytes" when cp1252 text is
# mis-decoded as UTF-8 and then re-encoded as UTF-8.  In honest
# English source these almost never appear; when they do, it is
# nearly always double-encoded text.
MOJIBAKE_TRIAD: tuple[str, ...] = ("â", "Â", "Ã")

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "enigma_engine"
_SKIP_PARTS = {"__pycache__", "venv", ".venv", "_archive", ".git", "node_modules"}


def _first_party_py():
    """Root scripts + package + tests. The old gate scanned only
    enigma_engine/, leaving the most console-bound files (serve_enigma,
    eval_behavior, teach_enigma...) outside every hygiene rule -- which is
    why manual ASCII sweeps kept 'finishing' and then recurring (test-suite
    audit 2026-07-17)."""
    yield from sorted(REPO_ROOT.glob("*.py"))
    for base in (PACKAGE_ROOT, REPO_ROOT / "tests"):
        for p in sorted(base.rglob("*.py")):
            if not any(part in _SKIP_PARTS for part in p.parts):
                yield p


def test_first_party_source_is_free_of_mojibake_markers() -> None:
    """Every first-party ``.py`` file must be free of the three canonical
    mojibake leading bytes.

    Pass 156z9cv: regression gate for the inference.py + rl_training.py
    mojibake corruption that was under-reported in prior stamps.
    2026-07-17: scope widened from enigma_engine/ to the whole first-party
    tree (root scripts included).
    """
    assert PACKAGE_ROOT.is_dir(), f"package root missing: {PACKAGE_ROOT}"

    failures: list[tuple[str, dict[str, int]]] = []
    for path in _first_party_py():
        rel = path.relative_to(REPO_ROOT).as_posix()
        if rel in ALLOWED_FILES:
            continue
        text = path.read_text(encoding="utf-8")
        counts = {ch: text.count(ch) for ch in MOJIBAKE_TRIAD if ch in text}
        if counts:
            failures.append((rel, counts))

    if failures:
        lines = ["mojibake markers found in first-party source:"]
        for rel, counts in failures:
            summary = ", ".join(f"{ch!r}={n}" for ch, n in counts.items())
            lines.append(f"  {rel}: {summary}")
        lines.append(
            "If a character is legitimate, add the file path to "
            "ALLOWED_FILES in tests/test_repo_hygiene.py with a "
            "justification comment."
        )
        raise AssertionError("\n".join(lines))


# ---------------------------------------------------------------------------
# Console-ASCII gate (CLAUDE.md: "Console output must be ASCII" -- cp1252
# consoles crash printing unicode). Until 2026-07-17 NO test enforced this
# rule anywhere; every "sweep now zero" was a manual grep that later turned
# out to have scanned too narrowly (three separate times). This gate defines
# the scope once: string literals inside the direct console sinks.
#
# Known limitation (accepted): strings that reach the console indirectly
# (appended to a warnings list that a caller prints later) are out of static
# reach. The rule for those stays manual; the sinks below are where every
# past regression actually lived.
# ---------------------------------------------------------------------------

_LOGGER_METHODS = {"debug", "info", "warning", "warn", "error", "critical", "exception"}


def _nonascii_strings(node: ast.AST):
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            bad = sorted({f"U+{ord(c):04X}" for c in sub.value if ord(c) > 127})
            if bad:
                yield sub.lineno, ",".join(bad)


def test_console_bound_strings_are_ascii() -> None:
    violations: list[str] = []
    for path in _first_party_py():
        rel = path.relative_to(REPO_ROOT).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as exc:  # a file that cannot parse is its own bug
            violations.append(f"  {rel}: SyntaxError {exc}")
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            if isinstance(f, ast.Name) and f.id in {"print", "SystemExit"}:
                targets = [node]
            elif isinstance(f, ast.Attribute) and f.attr in _LOGGER_METHODS:
                targets = [node]
            elif isinstance(f, ast.Attribute) and f.attr == "add_argument":
                targets = [kw.value for kw in node.keywords if kw.arg == "help"]
            elif isinstance(f, ast.Name) and f.id == "ArgumentParser":
                targets = [kw.value for kw in node.keywords if kw.arg in ("description", "epilog")]
            else:
                continue
            for target in targets:
                for lineno, chars in _nonascii_strings(target):
                    violations.append(f"  {rel}:{lineno}: {chars}")
    assert not violations, (
        "non-ASCII in console-bound strings (crashes cp1252 consoles; "
        "CLAUDE.md ASCII rule):\n" + "\n".join(sorted(set(violations)))
    )
