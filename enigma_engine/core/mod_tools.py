"""
Mod Tools — Auto-register mod commands as AI-callable tools.

Scans mod.json files and registers commands in the engine command
registry so the AI can invoke any mod command via [CMD] blocks.
Provides structured tool descriptions for AI system prompts.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def discover_mod_tools(mods_dir: Path) -> list[dict]:
    """Scan mods directory and return structured tool descriptions.

    Each tool dict has: mod_id, mod_name, name, description, args.
    """
    tools: list[dict] = []
    if not mods_dir.exists():
        return tools

    for mod_dir in sorted(mods_dir.iterdir()):
        if not mod_dir.is_dir() or mod_dir.name.startswith("_"):
            continue
        mod_json = mod_dir / "mod.json"
        if not mod_json.exists():
            continue
        try:
            data = json.loads(mod_json.read_text(encoding="utf-8"))
            mod_id = data.get("id", mod_dir.name)
            mod_name = data.get("name", mod_id)
            for cmd in data.get("commands", []):
                cmd_name = cmd.get("name", "")
                if not cmd_name:
                    logger.debug("Skipping command with empty name in %s", mod_json)
                    continue
                tools.append(
                    {
                        "mod_id": mod_id,
                        "mod_name": mod_name,
                        "name": f"{mod_id}.{cmd_name}",
                        "description": cmd.get("description", ""),
                        "args": cmd.get("args", {}),
                    }
                )
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("Could not read %s: %s", mod_json, exc)
    return tools


def register_mod_commands(registry: Any, mods_dir: Path, router: Any = None) -> int:
    """Register all mod commands in the engine command registry.

    Creates commands like ``imagegen.generate``, ``voice.listen``, etc.
    Skips commands already registered (earlier registrations, e.g. from
    plugins, take precedence).

    Returns the number of newly registered commands.
    """
    from enigma_engine.core.commands import CommandResult

    tools = discover_mod_tools(mods_dir)
    registered = 0

    for tool in tools:
        # Skip already-registered commands
        if tool["name"] in registry._commands:
            continue

        def _make_handler(tool_info: dict):
            """Closure factory so each handler captures its own tool."""

            def handler(args: list[str], ctx: dict) -> CommandResult:
                mod_id = tool_info["mod_id"]
                parts = tool_info["name"].split(".", 1)
                cmd_name = parts[1] if len(parts) == 2 else parts[0]
                r = ctx.get("router") or router
                if r is None:
                    return CommandResult(False, f"[ERROR] No router — cannot reach mod '{mod_id}'")
                message = {"command": cmd_name, "args": args}
                try:
                    result = r.send_to_mod(mod_id, message)
                    # None = no reply; falsy replies ("", 0, {}) are real
                    # responses and must not read as silence
                    if result is not None:
                        return CommandResult(True, str(result))
                    return CommandResult(False, f"[ERROR] Mod '{mod_id}' did not respond")
                except Exception as exc:
                    import traceback

                    err_msg = str(exc)
                    logger.error("Mod %s.%s failed: %s\n%s", mod_id, cmd_name, err_msg, traceback.format_exc())
                    return CommandResult(False, f"[ERROR] {mod_id}.{cmd_name}: {err_msg}")

            return handler

        # Build usage string with argument info
        args_desc = ""
        if tool["args"]:
            parts = []
            for aname, ainfo in tool["args"].items():
                if isinstance(ainfo, dict) and ainfo.get("required"):
                    parts.append(f"<{aname}>")
                else:
                    parts.append(f"[{aname}]")
            args_desc = " " + " ".join(parts)

        registry.register(tool["name"], _make_handler(tool), tool["description"], f"{tool['name']}{args_desc}")
        registered += 1

    return registered


def format_tools_for_prompt(mods_data: list[dict]) -> str:
    """Format mod tools as a structured prompt section for the AI.

    Includes ALL mods (running and stopped) so the AI knows what
    tools exist and can request starting ones it needs.
    """
    if not mods_data:
        return ""

    lines = ["[AVAILABLE TOOLS]"]
    lines.append("You have access to these tools via [CMD] blocks. Use [CMD]tool.command args[/CMD] to invoke a tool.")
    lines.append("If a tool is AVAILABLE but not RUNNING, use [CMD]mod.start <mod_id>[/CMD] to start it first.")
    lines.append("")

    for mod in mods_data:
        mod_id = mod.get("id", "")
        mod_name = mod.get("name", mod_id)
        running = mod.get("_running", False)
        status = "RUNNING" if running else "AVAILABLE"
        desc = mod.get("description", "")

        lines.append(f"  {mod_name} [{status}]" + (f" — {desc}" if desc else ""))

        cmds = mod.get("commands_full", mod.get("commands", []))
        for cmd in cmds:
            if isinstance(cmd, str):
                lines.append(f"    [CMD]{mod_id}.{cmd}[/CMD]")
            elif isinstance(cmd, dict):
                cmd_name = cmd.get("name", "")
                cmd_desc = cmd.get("description", "")
                args = cmd.get("args", {})
                args_str = ""
                if args:
                    parts = []
                    for a, info in args.items():
                        if isinstance(info, dict) and info.get("required"):
                            parts.append(f"<{a}>")
                        else:
                            parts.append(f"[{a}]")
                    args_str = " " + " ".join(parts)
                lines.append(f"    [CMD]{mod_id}.{cmd_name}{args_str}[/CMD] — {cmd_desc}")
        lines.append("")

    lines.append("[END AVAILABLE TOOLS]")
    return "\n".join(lines)
