# Plugins

Drop `.py` files here to add custom `[CMD]` commands to Enigma Engine.

## How It Works

Each plugin file must define a `register(registry)` function that receives
a `CommandRegistry` and registers one or more commands.

Plugins are auto-discovered and loaded when the command registry initialises.

## Rules

- Only `.py` files at the top level of this folder are loaded.
- Files starting with `_` are skipped (use `_helpers.py` for shared code).
- Each plugin should use a unique command prefix (e.g. `myplugin.action`).
- Errors in one plugin won't prevent others from loading.

## Example

See `_example.py` for a complete example (prefixed with `_` so it isn't
loaded automatically — rename to `example.py` to activate it).

## Command Format

The AI invokes commands with `[CMD]command.name arg1 arg2[/CMD]` blocks.
Your handler receives `(args: list[str], ctx: dict)` and should return
a `CommandResult(success, message, data=None)`.

Available context keys:
- `engine` — the `EnigmaEngine` instance (if loaded)
- `config` — dict of generation config values
- `window` — the GUI window (if running desktop mode)
