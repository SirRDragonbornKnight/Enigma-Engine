"""
Command Registry - AI-controlled command system.

The AI emits [CMD]command arg1 arg2[/CMD] blocks.
This module parses and executes them.

Commands are registered by category (config, model, system, file, etc.)
and can be executed by name. CLI-only - no GUI dependencies.
"""

from __future__ import annotations

import logging
import re
import shlex
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# Shell metacharacters that must never appear in command arguments.
# These could be used for injection if an AI-generated argument
# reaches a subprocess or shell eval. Parentheses are NOT stripped:
# they are ordinary path characters on Windows ("Program Files (x86)")
# and cannot chain commands from inside an argument.
SHELL_METACHARACTERS = frozenset(";|&`{}*?<>[]")


def sanitize_args(args: list[str]) -> list[str]:
    """Strip shell metacharacters from command arguments.

    Removes characters in :data:`SHELL_METACHARACTERS` from every arg
    and logs a warning when anything is stripped.  This is a defence-in-depth
    measure — command handlers should never pass args to a shell, but this
    prevents accidental injection if they do.

    Returns a new list; the original is not mutated.
    """
    cleaned: list[str] = []
    for arg in args:
        stripped = "".join(ch for ch in arg if ch not in SHELL_METACHARACTERS)
        if stripped != arg:
            if not stripped:
                logger.warning(
                    "Rejected command arg: %r (entirely shell metacharacters)",
                    arg,
                )
                continue
            logger.warning(
                "Sanitized command arg: %r -> %r (removed shell metacharacters)",
                arg,
                stripped,
            )
        cleaned.append(stripped)
    return cleaned


@dataclass
class CommandResult:
    """Result of a command execution."""

    success: bool
    message: str
    data: Any = None


@dataclass
class Command:
    """A registered command."""

    name: str
    handler: Callable
    description: str
    usage: str


class CommandRegistry:
    """
    Central registry for all commands.

    Commands are registered by category (gui, config, model, etc.)
    and can be executed by name.
    """

    def __init__(self):
        self._commands: dict[str, Command] = {}
        self._context: dict[str, Any] = {}
        self._history: list[str] = []  # Track executed commands

    def register(self, name: str, handler: Callable, description: str = "", usage: str = "") -> None:
        """Register a command."""
        self._commands[name] = Command(
            name=name, handler=handler, description=description or f"Execute {name}", usage=usage or name
        )

    def set_context(self, key: str, value: Any) -> None:
        """Set context value (engine, window, etc.)."""
        self._context[key] = value

    def get_context(self, key: str) -> Any:
        """Get context value."""
        return self._context.get(key)

    def execute(self, command_str: str) -> CommandResult:
        """
        Execute a command string.

        Args:
            command_str: Full command like "config.set temperature 0.7"

        Returns:
            CommandResult with success/failure and message
        """
        command_str = command_str.strip()
        if not command_str:
            return CommandResult(False, "[ERROR] Empty command")

        # Preserve raw payload for code.run so fenced/multiline Python
        # is not destroyed by whitespace splitting or arg sanitization.
        if command_str.startswith("code.run"):
            cmd_name, _, raw_code = command_str.partition(" ")
            args = [raw_code] if raw_code.strip() else []
        else:
            # Quote-aware split so paths with spaces survive as one arg
            # ("C:\Program Files (x86)\thing"); posix=False keeps
            # backslashes intact, surrounding quotes are stripped after.
            try:
                parts = shlex.split(command_str, posix=False)
            except ValueError:
                parts = command_str.split()
            parts = [p[1:-1] if len(p) >= 2 and p[0] == p[-1] and p[0] in "\"'" else p for p in parts]
            cmd_name = parts[0]
            args = sanitize_args(parts[1:]) if len(parts) > 1 else []

        if cmd_name not in self._commands:
            return CommandResult(False, f"[ERROR] Unknown command: {cmd_name}")

        cmd = self._commands[cmd_name]

        # Track in history (skip history command itself to avoid recursion)
        if cmd_name != "history":
            self._history.append(command_str.strip())
            # Keep only last 100 commands
            if len(self._history) > 100:
                self._history = self._history[-100:]

        try:
            result = cmd.handler(args, self._context)
            if isinstance(result, CommandResult):
                return result
            return CommandResult(True, f"[OK] {result}")
        except Exception as e:
            return CommandResult(False, f"[ERROR] {cmd_name}: {e}")

    def get_history(self, count: int = 20) -> list[str]:
        """Get recent command history."""
        return self._history[-count:]

    def list_commands(self, prefix: str = "") -> list[Command]:
        """List all commands, optionally filtered by prefix."""
        if prefix:
            return [c for c in self._commands.values() if c.name.startswith(prefix)]
        return list(self._commands.values())

    def get_help(self, cmd_name: str = "") -> str:
        """Get help text for a command or all commands."""
        if cmd_name:
            if cmd_name in self._commands:
                cmd = self._commands[cmd_name]
                return f"{cmd.name}\n  {cmd.description}\n  Usage: {cmd.usage}"
            return f"Unknown command: {cmd_name}"

        # All commands grouped by category
        categories: dict[str, list[Command]] = {}
        for cmd in self._commands.values():
            cat = cmd.name.split(".")[0] if "." in cmd.name else "general"
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(cmd)

        lines = ["Available commands:"]
        for cat in sorted(categories.keys()):
            lines.append(f"\n{cat.upper()}")
            for cmd in sorted(categories[cat], key=lambda c: c.name):
                lines.append(f"  {cmd.usage:<30} - {cmd.description}")

        return "\n".join(lines)


def parse_commands(text: str) -> tuple:
    """
    Extract [CMD]...[/CMD] blocks from AI response.

    Args:
        text: Full AI response text

    Returns:
        (clean_text, commands_list) - text without command blocks, list of commands
    """
    commands = re.findall(r"\[CMD\](.*?)\[/CMD\]", text, re.DOTALL)
    commands = [cmd.strip() for cmd in commands]
    clean = re.sub(r"\[CMD\].*?\[/CMD\]", "", text, flags=re.DOTALL).strip()
    return clean, commands


# Global registry instance
_registry: Optional[CommandRegistry] = None
_registry_lock = threading.Lock()


def get_registry() -> CommandRegistry:
    """Get the global command registry (commands come from plugins).

    Built whole-or-not-at-all: the global is assigned only after loading
    succeeds, so a failure raises on EVERY call instead of the first call
    raising and later calls silently seeing an empty registry.
    """
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                registry = CommandRegistry()
                # Load user plugins from plugins/ directory
                from .plugin_loader import load_all_plugins

                load_all_plugins(registry)
                _registry = registry
    return _registry
