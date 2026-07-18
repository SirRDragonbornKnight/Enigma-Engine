"""Every enigma_engine.* import in the tree must resolve to a real module.

Guards against the Modkit-refactor failure mode found 2026-07-13: deleting a
module while its callers survive (vision_encoder, audio_encoder, gguf and
reasoning all crashed on import until restored -- see KNOWN_ISSUES.md #11).

Two sweeps (both static -- nothing is executed or imported; PKG below stands
for the package name, spelled out would trip this very gate):
1. dotted paths (`import PKG.core.x`, `from PKG.core.x import y`) -- every
   dotted module path must exist on disk;
2. from-lists (`from PKG.core import x, y`) -- the OLD regex needed a dot
   after the package name, so this live style resolved only the package and
   a deleted submodule stayed invisible (test-suite audit 2026-07-17). Each
   imported name must be a real submodule OR mentioned in the package
   __init__ (re-export/attribute); names in module (non-package) bases are
   attributes and out of static reach.
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
_FROM_LINE = re.compile(r"\s*from\s+(enigma_engine(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\s+import\s+(.*)")
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


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


def _iter_from_imports(text: str):
    """Yield (base, [names]) for every `from enigma_engine... import ...`,
    including multi-line parenthesized name lists. Pure text -> tuples, so
    the parser itself is unit-testable."""
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = _FROM_LINE.match(lines[i])
        if m:
            base, rest = m.group(1), m.group(2)
            if "(" in rest:
                while ")" not in rest and i + 1 < len(lines):
                    i += 1
                    rest += " " + lines[i]
            names = []
            for raw in rest.replace("(", " ").replace(")", " ").split(","):
                name = raw.split("#")[0].strip()
                name = name.split(" as ")[0].strip()
                if name and _NAME.fullmatch(name):
                    names.append(name)
            yield base, names
        i += 1


def _submodule_missing(base: str, name: str) -> bool:
    """True when `from BASE import name` names a submodule that no longer
    exists AND the package __init__ never mentions the name (so it is not a
    re-export or package attribute either)."""
    pkg_dir = PKG
    for part in base.split(".")[1:]:
        pkg_dir = pkg_dir / part
    if not pkg_dir.is_dir():
        return False  # BASE is a module: `name` is an attribute, not checkable statically
    if (pkg_dir / f"{name}.py").exists() or (pkg_dir / name / "__init__.py").exists():
        return False
    init = pkg_dir / "__init__.py"
    return not (
        init.exists()
        and re.search(rf"\b{re.escape(name)}\b", init.read_text(encoding="utf-8", errors="replace"))
    )


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
        for base, names in _iter_from_imports(text):
            for name in names:
                dotted = f"{base}.{name}"
                if _submodule_missing(base, name) and dotted not in ALLOWED_MISSING:
                    unresolved.add(f"{py.relative_to(ROOT)} -> {dotted}")
    assert not unresolved, (
        "Dangling enigma_engine imports (module deleted but callers kept?):\n"
        + "\n".join(sorted(unresolved))
    )


def test_allowlist_is_not_stale():
    """If an ALLOWED_MISSING module comes back, drop it from the allowlist."""
    stale = {m for m in ALLOWED_MISSING if _module_exists(m)}
    assert not stale, f"Modules restored; remove from ALLOWED_MISSING: {stale}"


# ---- the guard's own machinery (a broken sweep is a silent one) ----


def test_from_import_parser_reads_all_shapes():
    text = (
        "from enigma_engine.core import chat_format, optim\n"
        "from enigma_engine import core\n"
        "from enigma_engine.core.chat_format import (\n"
        "    render_chat,\n"
        "    parse_assistant_ids,  # comment\n"
        ")\n"
        "import enigma_engine.core.model\n"
    )
    got = list(_iter_from_imports(text))
    assert ("enigma_engine.core", ["chat_format", "optim"]) in got
    assert ("enigma_engine", ["core"]) in got
    assert ("enigma_engine.core.chat_format", ["render_chat", "parse_assistant_ids"]) in got


def test_submodule_missing_detects_a_deleted_module():
    # a real submodule and a real init re-export both pass...
    assert not _submodule_missing("enigma_engine.core", "chat_format")
    assert not _submodule_missing("enigma_engine", "core")
    # ...a module-base import is out of scope...
    assert not _submodule_missing("enigma_engine.core.chat_format", "render_chat")
    # ...and a name that is neither submodule nor mentioned in __init__ FLAGS.
    assert _submodule_missing("enigma_engine.core", "definitely_deleted_module_xyz")
