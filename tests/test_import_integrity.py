"""Every enigma_engine.* import in the tree must resolve to a real module.

Guards against the Modkit-refactor failure mode found 2026-07-13: deleting a
module while its callers survive (vision_encoder, audio_encoder, gguf and
reasoning all crashed on import until restored -- see KNOWN_ISSUES.md #11).

AST-based (2026-07-18 re-audit): the earlier text/regex sweep false-positived
on import statements quoted inside docstrings and missed backslash
continuations; parsing real Import/ImportFrom nodes kills both classes.
Known blind spot: dynamic imports via importlib.import_module("...") strings
are invisible to the AST (none exist for enigma_engine.* today).

Two checks per file (static -- nothing from the tree is executed):
1. every dotted module path in `import a.b.c` / `from a.b.c import x` must
   exist on disk;
2. for `from <package> import x, y`, each name must be a real submodule OR a
   name the package __init__ actually DEFINES (import aliases, assignments,
   def/class, __all__ strings) -- prose mentions no longer count (the old
   substring check was satisfiable by a docstring word). A package __init__
   with module-level __getattr__ (lazy exports) makes names statically
   unverifiable, so such packages are exempt from check 2.
"""
from __future__ import annotations

import ast
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

_PKG_NAME = "enigma_engine"


def _module_exists(dotted: str) -> bool:
    parts = dotted.split(".")[1:]
    base = PKG
    for part in parts[:-1]:
        base = base / part
        if not base.is_dir():
            return False
    if not parts:
        return True  # bare `import enigma_engine`
    last = base / (parts[-1] + ".py")
    pkg = base / parts[-1] / "__init__.py"
    return last.exists() or pkg.exists()


def _iter_imports(tree: ast.AST):
    """Yield ("import", dotted) for plain imports and ("from", base, names)
    for from-imports of the first-party package. Compiled AST, so quoted
    prose, comments and continuations cannot confuse it."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == _PKG_NAME or alias.name.startswith(_PKG_NAME + "."):
                    yield ("import", alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative imports resolve against their own package
                continue
            base = node.module or ""
            if base == _PKG_NAME or base.startswith(_PKG_NAME + "."):
                yield ("from", base, [a.name for a in node.names if a.name != "*"])


def _init_defined_names(init_path: Path) -> set[str] | None:
    """Names a package __init__ genuinely defines (imports incl. the
    try/except-guarded fallback pattern, assignments, def/class, __all__
    literal strings). A module-level __getattr__ (lazy exports) makes names
    statically unverifiable ONLY when no __all__ literal exists to consult --
    core/__init__ has both, and its __all__ is the authoritative surface."""
    tree = ast.parse(init_path.read_text(encoding="utf-8", errors="replace"))
    names: set[str] = set()
    has_getattr = False
    has_all = False
    for node in ast.walk(tree):  # walk: imports live inside try/except blocks
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == "__getattr__":
                has_getattr = True
            names.add(node.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
                    if t.id == "__all__" and isinstance(getattr(node, "value", None), (ast.List, ast.Tuple)):
                        has_all = True
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                names.add(elt.value)
    if has_getattr and not has_all:
        return None
    return names


def _submodule_missing(base: str, name: str) -> bool:
    """True when `from BASE import name` names a submodule that no longer
    exists AND the package __init__ does not define the name."""
    pkg_dir = PKG
    for part in base.split(".")[1:]:
        pkg_dir = pkg_dir / part
    if not pkg_dir.is_dir():
        return False  # BASE is a module: `name` is an attribute, not checkable statically
    if (pkg_dir / f"{name}.py").exists() or (pkg_dir / name / "__init__.py").exists():
        return False
    init = pkg_dir / "__init__.py"
    if not init.exists():
        return True
    defined = _init_defined_names(init)
    if defined is None:  # lazy __getattr__ package: statically unverifiable
        return False
    return name not in defined


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
        try:
            tree = ast.parse(py.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError as exc:
            unresolved.add(f"{py.relative_to(ROOT)} -> SyntaxError {exc}")
            continue
        for item in _iter_imports(tree):
            if item[0] == "import":
                dotted = item[1]
                if not _module_exists(dotted) and dotted not in ALLOWED_MISSING:
                    unresolved.add(f"{py.relative_to(ROOT)} -> {dotted}")
            else:
                _, base, names = item
                if not _module_exists(base) and base not in ALLOWED_MISSING:
                    unresolved.add(f"{py.relative_to(ROOT)} -> {base}")
                    continue
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


def test_import_collector_reads_all_shapes():
    text = (
        "from enigma_engine.core import chat_format, optim\n"
        "from enigma_engine import core\n"
        "from enigma_engine.core.chat_format import (\n"
        "    render_chat,\n"
        "    parse_assistant_ids,  # comment\n"
        ")\n"
        "from enigma_engine.core import chat_format, \\\n"
        "    tokenizer\n"
        "import enigma_engine.core.model\n"
        '"""docstring prose: from enigma_engine.core import totally_fake_module"""\n'
    )
    got = list(_iter_imports(ast.parse(text)))
    assert ("from", "enigma_engine.core", ["chat_format", "optim"]) in got
    assert ("from", "enigma_engine", ["core"]) in got
    assert ("from", "enigma_engine.core.chat_format", ["render_chat", "parse_assistant_ids"]) in got
    assert ("from", "enigma_engine.core", ["chat_format", "tokenizer"]) in got  # backslash form
    assert ("import", "enigma_engine.core.model") in got
    # prose inside a string literal must NOT register (the old text-regex
    # sweep flagged quoted imports -- re-audit 2026-07-18)
    assert not any("totally_fake_module" in str(item) for item in got)


def test_submodule_missing_detects_a_deleted_module():
    # a real submodule passes...
    assert not _submodule_missing("enigma_engine.core", "chat_format")
    assert not _submodule_missing("enigma_engine", "core")
    # ...a module-base import is out of scope...
    assert not _submodule_missing("enigma_engine.core.chat_format", "render_chat")
    # ...and a name that is neither submodule nor defined by __init__ FLAGS,
    # even when the word appears in the package docstring (the old substring
    # check was satisfiable by prose -- "train" appears in core/__init__'s
    # docstring but is not a module or defined name there).
    assert _submodule_missing("enigma_engine.core", "definitely_deleted_module_xyz")
