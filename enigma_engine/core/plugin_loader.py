"""
Plugin Loader — discover and register custom commands.

Users can drop ``.py`` files into the ``plugins/`` directory.  Each plugin
module must expose a ``register(registry)`` callable that receives a
:class:`commands.CommandRegistry` and registers its commands.

Example plugin (``plugins/hello.py``)::

    from enigma_engine.core.commands import CommandResult

    def register(registry):
        def hello(args, ctx):
            name = args[0] if args else "world"
            return CommandResult(True, f"[OK] Hello, {name}!")

        registry.register(
            "hello.greet", hello,
            description="Greet someone",
            usage="hello.greet [name]",
        )

Plugin discovery rules:
    * Only ``.py`` files at the top level of ``plugins/`` are loaded.
    * Files starting with ``_`` are skipped (e.g. ``_helpers.py``).
    * Each file is loaded at most once per registry initialisation.
    * Errors in one plugin are logged and do not prevent others from loading.
"""

from __future__ import annotations

import ast
import importlib.util
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .commands import CommandRegistry

logger = logging.getLogger(__name__)

# Default location — workspace root / plugins
_PLUGINS_DIR = Path(__file__).resolve().parent.parent.parent / "plugins"

# Dangerous AST node / attribute patterns that plugins must not use.
# If any of these appear in the AST the plugin is rejected before execution.
_DANGEROUS_CALLS: frozenset[str] = frozenset(
    {
        "exec",
        "eval",
        "compile",
        "__import__",
        "globals",
        "locals",
        "vars",
        "getattr",
        "delattr",
    }
)
_DANGEROUS_ATTRS: frozenset[str] = frozenset(
    {
        "os.system",
        "os.popen",
        "os.exec",
        "os.execl",
        "os.execle",
        "os.execlp",
        "os.execlpe",
        "os.execv",
        "os.execve",
        "os.execvp",
        "os.execvpe",
        "os.spawn",
        "os.spawnl",
        "os.spawnle",
        "subprocess.call",
        "subprocess.run",
        "subprocess.Popen",
        "subprocess.check_output",
        "subprocess.check_call",
        "subprocess.getoutput",
        "subprocess.getstatusoutput",
        "shutil.rmtree",
        "ctypes.cdll",
        "ctypes.windll",
        "ctypes.CDLL",
        "ctypes.WinDLL",
    }
)


def discover_plugins(plugins_dir: Path | None = None) -> list[Path]:
    """Return a sorted list of plugin ``.py`` files in *plugins_dir*.

    Skips files whose stem starts with ``_``.
    """
    folder = plugins_dir or _PLUGINS_DIR
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.glob("*.py") if p.is_file() and not p.stem.startswith("_"))


def _has_register_def(source: str) -> bool:
    """Quick text scan: does *source* contain ``def register``?

    This is a cheap pre-check before the heavier AST parse.
    Rejects files that clearly cannot provide the expected entry point.
    """
    return "def register" in source


def _ast_scan_dangers(source: str, filename: str) -> list[str]:
    """Parse *source* as AST and return a list of flagged dangerous patterns.

    Each entry is a human-readable description like
    ``"line 12: os.system (dangerous call)"``.

    Returns an empty list if the source is clean.
    """
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError:
        # Syntax errors are caught later during exec_module
        return []

    flags: list[str] = []

    # Map local bindings back to their real module/name so aliasing and
    # `from X import Y` can't slip a dangerous call past the scan:
    #   import os as o           -> {"o": "os"}          (o.system -> os.system)
    #   from os import system    -> {"system": "os.system"}
    module_aliases: dict[str, str] = {}
    imported_funcs: dict[str, str] = {}
    _dangerous_leaves = {p.split(".")[-1] for p in _DANGEROUS_ATTRS}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_aliases[alias.asname or alias.name] = alias.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                imported_funcs[alias.asname or alias.name] = f"{node.module}.{alias.name}"

    def _matches_dangerous(full: str) -> bool:
        return any(full == pat or full.endswith("." + pat) for pat in _DANGEROUS_ATTRS)

    for node in ast.walk(tree):
        # Dangerous bare calls: exec(...), eval(...), etc.
        if isinstance(node, ast.Call):
            func = node.func
            # Direct name call: exec(...), or a `from os import system` binding
            if isinstance(func, ast.Name):
                if func.id in _DANGEROUS_CALLS:
                    flags.append(f"line {node.lineno}: {func.id}() (dangerous builtin)")
                else:
                    bound = imported_funcs.get(func.id)
                    if bound and _matches_dangerous(bound):
                        flags.append(f"line {node.lineno}: {func.id}() -> {bound} (dangerous call)")
            # Attribute call: os.system(...), incl. aliased `import os as o`
            elif isinstance(func, ast.Attribute):
                full = _resolve_attr(func)
                if full:
                    head, _, rest = full.partition(".")
                    resolved = f"{module_aliases[head]}.{rest}" if rest and head in module_aliases else full
                    if _matches_dangerous(full) or _matches_dangerous(resolved):
                        flags.append(f"line {node.lineno}: {resolved}() (dangerous call)")

        # Import of subprocess or os — flag for awareness
        # (We only block *calls*, but log this import)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif node.module:
                names = [node.module]
            for name in names:
                if name in ("subprocess", "ctypes"):
                    flags.append(f"line {node.lineno}: import {name} (flagged module)")

    return flags


def _resolve_attr(node: ast.Attribute) -> str | None:
    """Resolve an ``ast.Attribute`` chain to a dotted string like ``os.system``."""
    parts: list[str] = [node.attr]
    current = node.value
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return None


def _is_trusted(path: Path) -> bool:
    """Check if *path* is on the trusted-plugins allowlist.

    If the ``trusted_plugins`` config list is empty, **all** plugins are
    trusted (legacy / convenience behavior).  Otherwise only filenames
    that appear in the list are accepted.
    """
    from enigma_engine import CONFIG

    trusted: list[str] = CONFIG.get("trusted_plugins", [])
    if not trusted:
        logger.warning(
            "trusted_plugins list is empty — all plugins allowed. Set trusted_plugins in config to restrict."
        )
        return True  # empty list = allow all
    return path.name in trusted


def load_plugin(path: Path, registry: "CommandRegistry") -> bool:
    """Import a single plugin file and call its ``register(registry)``.

    Security checks applied before execution:
        1. Trusted-plugins allowlist (config ``trusted_plugins``).
        2. Source pre-scan for ``def register`` (rejects random ``.py`` files).
        3. AST scan for dangerous patterns (``os.system``, ``exec``, etc.).

    Returns ``True`` on success, ``False`` on failure (logged).
    """
    # --- 5A: trusted-plugins allowlist ---
    if not _is_trusted(path):
        logger.warning("Plugin %s is not in trusted_plugins allowlist — skipped", path.name)
        return False

    # Read source once for pre-scan checks
    try:
        source = path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("Cannot read plugin %s: %s", path.name, exc)
        return False

    # --- 5E: pre-scan for def register ---
    if not _has_register_def(source):
        logger.warning("Plugin %s has no 'def register' in source — skipped", path.name)
        return False

    # --- 5F: AST danger scan ---
    dangers = _ast_scan_dangers(source, path.name)
    if dangers:
        for flag in dangers:
            logger.warning("Plugin %s — %s", path.name, flag)
        logger.warning(
            "Plugin %s rejected: %d dangerous pattern(s) found",
            path.name,
            len(dangers),
        )
        return False

    module_name = f"enigma_plugin_{path.stem}"
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            logger.warning("Cannot create module spec for plugin: %s", path)
            return False
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except Exception as exc:
        logger.warning("Failed to import plugin %s: %s", path.name, exc)
        # Clean up partial entry
        sys.modules.pop(module_name, None)
        return False

    register_fn = getattr(module, "register", None)
    if not callable(register_fn):
        logger.warning("Plugin %s has no callable 'register' — skipped", path.name)
        sys.modules.pop(module_name, None)
        return False

    try:
        register_fn(registry)
    except Exception as exc:
        logger.warning("Plugin %s register() raised: %s", path.name, exc)
        return False

    logger.info("Loaded plugin: %s", path.name)
    return True


def load_all_plugins(
    registry: "CommandRegistry",
    plugins_dir: Path | None = None,
) -> int:
    """Discover and load all plugins.  Returns the number loaded."""
    paths = discover_plugins(plugins_dir)
    loaded = 0
    for p in paths:
        if load_plugin(p, registry):
            loaded += 1
    if loaded:
        logger.info("Loaded %d plugin(s) from %s", loaded, plugins_dir or _PLUGINS_DIR)
    return loaded
