"""Every enigma_engine.* import in the tree must resolve to a real module.

Guards against the Modkit-refactor failure mode found 2026-07-13: deleting a
module while its callers survive (vision_encoder, audio_encoder, gguf and
reasoning all crashed on import until restored -- see KNOWN_ISSUES.md #11).

Static text sweep -- nothing is executed or imported.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG = ROOT / "enigma_engine"

# Directories that hold first-party code. data/, models/, venv/ etc. are
# deliberately out of scope (vendored or generated).
SCAN_DIRS = ["enigma_engine", "tests"]
SKIP_PARTS = {"__pycache__", "venv", "_archive", ".git", "node_modules"}

# Known dangling imports that are ALLOWED because every call site degrades
# gracefully. Remove an entry here when the module is restored or the caller
# is deleted. Empty since 2026-07-17 (rl_training's sentiment caller removed).
ALLOWED_MISSING: set[str] = set()

_IMPORT = re.compile(r"(?:from|import)\s+(enigma_engine(?:\.[A-Za-z_][A-Za-z0-9_]*)+)")


def _module_exists(dotted: str) -> bool:
    parts = dotted.split(".")[1:]
    base = PKG
    for part in parts[:-1]:
        base = base / part
        if not base.is_dir():
            return False
    last = base / (parts[-1] + ".py")
    pkg = base / parts[-1] / "__init__.py"
    return last.exists() or pkg.exists()


def _iter_py_files():
    for name in SCAN_DIRS:
        d = ROOT / name
        if d.is_dir():
            yield from d.rglob("*.py")
    yield from ROOT.glob("*.py")


def test_all_enigma_imports_resolve():
    unresolved = set()
    for py in _iter_py_files():
        if any(p in SKIP_PARTS for p in py.parts):
            continue
        text = py.read_text(encoding="utf-8", errors="replace")
        for m in _IMPORT.finditer(text):
            dotted = m.group(1)
            if not _module_exists(dotted) and dotted not in ALLOWED_MISSING:
                unresolved.add(f"{py.relative_to(ROOT)} -> {dotted}")
    assert not unresolved, (
        "Dangling enigma_engine imports (module deleted but callers kept?):\n"
        + "\n".join(sorted(unresolved))
    )


def test_allowlist_is_not_stale():
    """If an ALLOWED_MISSING module comes back, drop it from the allowlist."""
    stale = {m for m in ALLOWED_MISSING if _module_exists(m)}
    assert not stale, f"Modules restored; remove from ALLOWED_MISSING: {stale}"
