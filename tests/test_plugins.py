"""Tests for plugin loader, AST safety scanning, trusted allowlists, mod client, and mod tools."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestPluginLoader:
    """Verify the plugin discovery and loading system."""

    def test_discover_plugins_empty_dir(self, tmp_path):
        """discover_plugins returns [] for an empty directory."""
        from enigma_engine.core.plugin_loader import discover_plugins

        assert discover_plugins(tmp_path) == []

    def test_discover_plugins_finds_py_files(self, tmp_path):
        """discover_plugins finds .py files but skips _-prefixed ones."""
        from enigma_engine.core.plugin_loader import discover_plugins

        (tmp_path / "hello.py").write_text("# plugin", encoding="utf-8")
        (tmp_path / "_private.py").write_text("# hidden", encoding="utf-8")
        (tmp_path / "readme.md").write_text("# not python", encoding="utf-8")
        found = discover_plugins(tmp_path)
        assert len(found) == 1
        assert found[0].stem == "hello"

    def test_discover_plugins_nonexistent_dir(self):
        """discover_plugins returns [] for a directory that doesn't exist."""
        from enigma_engine.core.plugin_loader import discover_plugins
        from pathlib import Path

        assert discover_plugins(Path("/nonexistent/dir/xyz")) == []

    def test_load_plugin_success(self, tmp_path):
        """load_plugin imports a valid plugin and calls register()."""
        from enigma_engine.core.commands import CommandRegistry
        from enigma_engine.core.plugin_loader import load_plugin

        plugin_code = """
from enigma_engine.core.commands import CommandResult

def register(registry):
    def my_cmd(args, ctx):
        return CommandResult(True, "[OK] works")
    registry.register("test_plug.hello", my_cmd, "Test", "test_plug.hello")
"""
        plugin_file = tmp_path / "test_plug.py"
        plugin_file.write_text(plugin_code, encoding="utf-8")

        reg = CommandRegistry()
        assert load_plugin(plugin_file, reg) is True
        assert "test_plug.hello" in [c.name for c in reg.list_commands()]

    def test_load_plugin_no_register_function(self, tmp_path):
        """load_plugin returns False if plugin has no register()."""
        from enigma_engine.core.commands import CommandRegistry
        from enigma_engine.core.plugin_loader import load_plugin

        plugin_file = tmp_path / "no_register.py"
        plugin_file.write_text("x = 42\n", encoding="utf-8")

        reg = CommandRegistry()
        assert load_plugin(plugin_file, reg) is False

    def test_load_plugin_register_raises(self, tmp_path):
        """load_plugin returns False if register() raises."""
        from enigma_engine.core.commands import CommandRegistry
        from enigma_engine.core.plugin_loader import load_plugin

        plugin_file = tmp_path / "bad_register.py"
        plugin_file.write_text(
            "def register(reg):\n    raise ValueError('broken')\n",
            encoding="utf-8",
        )

        reg = CommandRegistry()
        assert load_plugin(plugin_file, reg) is False

    def test_load_plugin_syntax_error(self, tmp_path):
        """load_plugin returns False for a file with syntax errors."""
        from enigma_engine.core.commands import CommandRegistry
        from enigma_engine.core.plugin_loader import load_plugin

        plugin_file = tmp_path / "syntax_err.py"
        plugin_file.write_text("def broken(\n", encoding="utf-8")

        reg = CommandRegistry()
        assert load_plugin(plugin_file, reg) is False

    def test_load_all_plugins_counts(self, tmp_path):
        """load_all_plugins returns the number of successfully loaded plugins."""
        from enigma_engine.core.commands import CommandRegistry
        from enigma_engine.core.plugin_loader import load_all_plugins

        good = """
from enigma_engine.core.commands import CommandResult
def register(reg):
    reg.register("p1.cmd", lambda a, c: CommandResult(True, "ok"), "p1", "p1.cmd")
"""
        (tmp_path / "good.py").write_text(good, encoding="utf-8")
        (tmp_path / "bad.py").write_text("def register(r): raise RuntimeError()\n", encoding="utf-8")
        (tmp_path / "_skip.py").write_text("# skipped\n", encoding="utf-8")

        reg = CommandRegistry()
        loaded = load_all_plugins(reg, tmp_path)
        assert loaded == 1  # good loaded, bad failed, _skip skipped

    def test_plugin_command_executes(self, tmp_path):
        """A command registered by a plugin can be executed via the registry."""
        from enigma_engine.core.commands import CommandRegistry
        from enigma_engine.core.plugin_loader import load_plugin

        plugin_code = """
from enigma_engine.core.commands import CommandResult
def register(registry):
    def echo(args, ctx):
        return CommandResult(True, "[OK] " + " ".join(args))
    registry.register("echo.say", echo, "Echo args", "echo.say <text>")
"""
        (tmp_path / "echo_plug.py").write_text(plugin_code, encoding="utf-8")

        reg = CommandRegistry()
        load_plugin(tmp_path / "echo_plug.py", reg)
        result = reg.execute("echo.say hello world")
        assert result.success
        assert "hello world" in result.message

    def test_example_plugin_not_loaded(self):
        """_example.py should not be loaded (starts with underscore)."""
        from enigma_engine.core.plugin_loader import discover_plugins
        from pathlib import Path

        plugins_dir = Path(__file__).resolve().parent.parent / "plugins"
        found = discover_plugins(plugins_dir)
        assert all(p.stem != "_example" for p in found)

    def test_multiple_plugins_all_register(self, tmp_path):
        """Multiple valid plugins should all register their commands."""
        from enigma_engine.core.commands import CommandRegistry
        from enigma_engine.core.plugin_loader import load_all_plugins

        for i in range(3):
            code = f"""
from enigma_engine.core.commands import CommandResult
def register(reg):
    reg.register("multi{i}.cmd", lambda a, c: CommandResult(True, "ok"), "", "")
"""
            (tmp_path / f"plug{i}.py").write_text(code, encoding="utf-8")

        reg = CommandRegistry()
        loaded = load_all_plugins(reg, tmp_path)
        assert loaded == 3
        names = [c.name for c in reg.list_commands()]
        for i in range(3):
            assert f"multi{i}.cmd" in names


# =========================================================================
# Suggestion #4 — Command Injection Sanitization
# =========================================================================


class TestPluginSecurity:
    """Verify plugin pre-scan, AST validation, and trusted allowlist."""

    def test_has_register_def_positive(self):
        """_has_register_def returns True for source with def register."""
        from enigma_engine.core.plugin_loader import _has_register_def

        assert _has_register_def("def register(registry):\n    pass\n")

    def test_has_register_def_negative(self):
        """_has_register_def returns False for source without def register."""
        from enigma_engine.core.plugin_loader import _has_register_def

        assert not _has_register_def("x = 42\ndef foo(): pass\n")

    def test_reject_plugin_without_def_register(self, tmp_path):
        """Plugin without def register in source is rejected before exec_module."""
        from enigma_engine.core.commands import CommandRegistry
        from enigma_engine.core.plugin_loader import load_plugin

        plugin = tmp_path / "no_reg.py"
        plugin.write_text("x = 42\n", encoding="utf-8")
        reg = CommandRegistry()
        assert load_plugin(plugin, reg) is False

    def test_ast_scan_flags_exec(self):
        """_ast_scan_dangers flags exec() calls."""
        from enigma_engine.core.plugin_loader import _ast_scan_dangers

        source = "exec('print(1)')\n"
        flags = _ast_scan_dangers(source, "test.py")
        assert len(flags) == 1
        assert "exec()" in flags[0]

    def test_ast_scan_flags_eval(self):
        """_ast_scan_dangers flags eval() calls."""
        from enigma_engine.core.plugin_loader import _ast_scan_dangers

        source = "x = eval('2+2')\n"
        flags = _ast_scan_dangers(source, "test.py")
        assert len(flags) == 1
        assert "eval()" in flags[0]

    def test_ast_scan_flags_os_system(self):
        """_ast_scan_dangers flags os.system() calls."""
        from enigma_engine.core.plugin_loader import _ast_scan_dangers

        source = "import os\nos.system('rm -rf /')\n"
        flags = _ast_scan_dangers(source, "test.py")
        assert any("os.system" in f for f in flags)

    def test_ast_scan_flags_subprocess_run(self):
        """_ast_scan_dangers flags subprocess.run()."""
        from enigma_engine.core.plugin_loader import _ast_scan_dangers

        source = "import subprocess\nsubprocess.run(['ls'])\n"
        flags = _ast_scan_dangers(source, "test.py")
        assert any("subprocess.run" in f for f in flags)
        assert any("import subprocess" in f for f in flags)

    def test_ast_scan_clean_source(self):
        """_ast_scan_dangers returns empty list for safe source."""
        from enigma_engine.core.plugin_loader import _ast_scan_dangers

        source = """
from enigma_engine.core.commands import CommandResult
def register(registry):
    def hello(args, ctx):
        return CommandResult(True, "hello")
    registry.register("hello.greet", hello, "Greet", "hello.greet")
"""
        flags = _ast_scan_dangers(source, "test.py")
        assert flags == []

    def test_reject_plugin_with_dangerous_code(self, tmp_path):
        """Plugin with os.system() call is rejected."""
        from enigma_engine.core.commands import CommandRegistry
        from enigma_engine.core.plugin_loader import load_plugin

        code = """import os
def register(registry):
    os.system('echo pwned')
"""
        plugin = tmp_path / "evil.py"
        plugin.write_text(code, encoding="utf-8")
        reg = CommandRegistry()
        assert load_plugin(plugin, reg) is False

    def test_reject_plugin_with_exec(self, tmp_path):
        """Plugin using exec() is rejected."""
        from enigma_engine.core.commands import CommandRegistry
        from enigma_engine.core.plugin_loader import load_plugin

        code = """
def register(registry):
    exec("print('hacked')")
"""
        plugin = tmp_path / "exec_evil.py"
        plugin.write_text(code, encoding="utf-8")
        reg = CommandRegistry()
        assert load_plugin(plugin, reg) is False

    def test_trusted_plugins_allowlist(self, tmp_path, monkeypatch):
        """Plugin not in trusted_plugins list is rejected."""
        from enigma_engine.core.commands import CommandRegistry
        from enigma_engine.core.plugin_loader import load_plugin
        import enigma_engine

        monkeypatch.setitem(enigma_engine.CONFIG, "trusted_plugins", ["allowed.py"])
        code = """
from enigma_engine.core.commands import CommandResult
def register(registry):
    registry.register("t.cmd", lambda a, c: CommandResult(True, "ok"), "", "")
"""
        plugin = tmp_path / "not_allowed.py"
        plugin.write_text(code, encoding="utf-8")
        reg = CommandRegistry()
        assert load_plugin(plugin, reg) is False

    def test_trusted_plugins_allowlist_permits(self, tmp_path, monkeypatch):
        """Plugin in trusted_plugins list is loaded."""
        from enigma_engine.core.commands import CommandRegistry
        from enigma_engine.core.plugin_loader import load_plugin
        import enigma_engine

        monkeypatch.setitem(enigma_engine.CONFIG, "trusted_plugins", ["good.py"])
        code = """
from enigma_engine.core.commands import CommandResult
def register(registry):
    registry.register("t2.cmd", lambda a, c: CommandResult(True, "ok"), "", "")
"""
        plugin = tmp_path / "good.py"
        plugin.write_text(code, encoding="utf-8")
        reg = CommandRegistry()
        assert load_plugin(plugin, reg) is True

    def test_trusted_plugins_empty_allows_all(self, tmp_path, monkeypatch):
        """Empty trusted_plugins list allows all plugins (legacy)."""
        from enigma_engine.core.commands import CommandRegistry
        from enigma_engine.core.plugin_loader import load_plugin
        import enigma_engine

        monkeypatch.setitem(enigma_engine.CONFIG, "trusted_plugins", [])
        code = """
from enigma_engine.core.commands import CommandResult
def register(registry):
    registry.register("t3.cmd", lambda a, c: CommandResult(True, "ok"), "", "")
"""
        plugin = tmp_path / "any_plugin.py"
        plugin.write_text(code, encoding="utf-8")
        reg = CommandRegistry()
        assert load_plugin(plugin, reg) is True

    def test_ast_scan_flags_subprocess_popen(self):
        """_ast_scan_dangers flags subprocess.Popen()."""
        from enigma_engine.core.plugin_loader import _ast_scan_dangers

        source = "import subprocess\nsubprocess.Popen(['ls'])\n"
        flags = _ast_scan_dangers(source, "test.py")
        assert any("subprocess.Popen" in f for f in flags)

    def test_ast_scan_flags_shutil_rmtree(self):
        """_ast_scan_dangers flags shutil.rmtree()."""
        from enigma_engine.core.plugin_loader import _ast_scan_dangers

        source = "import shutil\nshutil.rmtree('/tmp/x')\n"
        flags = _ast_scan_dangers(source, "test.py")
        assert any("shutil.rmtree" in f for f in flags)

    def test_ast_scan_handles_syntax_error(self):
        """_ast_scan_dangers returns empty list for unparseable source."""
        from enigma_engine.core.plugin_loader import _ast_scan_dangers

        flags = _ast_scan_dangers("def broken(\n", "bad.py")
        assert flags == []


# =========================================================================
# Mod Client & Protocol (from test_mods.py)
# =========================================================================


class TestModClientSync:
    """Verify ModClient is fully synchronous (no asyncio)."""

    def test_mod_base_no_asyncio_import(self):
        """mod_base.py must not import asyncio."""
        import ast

        mod_base = Path(__file__).resolve().parent.parent / "mods" / "_template" / "mod_base.py"
        tree = ast.parse(mod_base.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "asyncio", "mod_base.py still imports asyncio"
            elif isinstance(node, ast.ImportFrom):
                assert node.module != "asyncio", "mod_base.py still imports asyncio"

    def test_mod_client_methods_are_sync(self):
        """All public ModClient methods must be plain functions, not coroutines."""
        import importlib.util
        import asyncio

        mod_base = Path(__file__).resolve().parent.parent / "mods" / "_template" / "mod_base.py"
        spec = importlib.util.spec_from_file_location("mod_base", mod_base)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cls = mod.ModClient
        for name in ("connect", "send_message", "receive_message", "register", "handle_message", "send_update", "run"):
            method = getattr(cls, name, None)
            assert method is not None, f"ModClient missing {name}"
            assert not getattr(method, "_is_coroutine", False), f"{name} is a coroutine"
            assert not asyncio.iscoroutinefunction(method), f"{name} is async"

    def test_mod_client_handle_message_dispatches(self):
        """handle_message should dispatch to cmd_* and return dict."""
        import importlib.util

        mod_base = Path(__file__).resolve().parent.parent / "mods" / "_template" / "mod_base.py"
        spec = importlib.util.spec_from_file_location("mod_base", mod_base)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        class _TestMod(mod.ModClient):
            def cmd_hello(self, args):
                return {"greeting": f"hi {args['name']}"}

        test_mod = _TestMod()
        resp = test_mod.handle_message(
            {"type": "command", "id": "1", "data": {"command": "hello", "args": {"name": "world"}}}
        )
        assert resp["success"] is True
        assert resp["data"]["greeting"] == "hi world"

    def test_mod_client_handle_ping(self):
        """handle_message should respond to ping with pong."""
        import importlib.util

        mod_base = Path(__file__).resolve().parent.parent / "mods" / "_template" / "mod_base.py"
        spec = importlib.util.spec_from_file_location("mod_base", mod_base)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        class _TestMod(mod.ModClient):
            def cmd_hello(self, args):
                return {"greeting": f"hi {args['name']}"}

        test_mod = _TestMod()
        resp = test_mod.handle_message({"type": "ping", "id": "42"})
        assert resp["type"] == "pong"
        assert resp["id"] == "42"


# =========================================================================
# Mod Tools (from test_mods.py)
# =========================================================================


class TestModTools:
    """Tests for enigma_engine.core.mod_tools module."""

    def test_discover_mod_tools_returns_list(self):
        """discover_mod_tools returns a list of tool dicts."""
        from enigma_engine.core.mod_tools import discover_mod_tools

        mods_dir = Path(__file__).parent.parent / "mods"
        tools = discover_mod_tools(mods_dir)
        assert isinstance(tools, list)
        # Should find at least imagegen.generate and voice.listen
        names = [t["name"] for t in tools]
        assert "imagegen.generate" in names
        assert "voice.listen" in names

    def test_discover_mod_tools_skips_template(self):
        """discover_mod_tools skips _template directory."""
        from enigma_engine.core.mod_tools import discover_mod_tools

        mods_dir = Path(__file__).parent.parent / "mods"
        tools = discover_mod_tools(mods_dir)
        mod_ids = {t["mod_id"] for t in tools}
        assert "_template" not in mod_ids

    def test_discover_mod_tools_has_required_keys(self):
        """Each tool dict has mod_id, name, description, args."""
        from enigma_engine.core.mod_tools import discover_mod_tools

        mods_dir = Path(__file__).parent.parent / "mods"
        tools = discover_mod_tools(mods_dir)
        for t in tools:
            assert "mod_id" in t
            assert "name" in t
            assert "description" in t
            assert "args" in t

    def test_discover_mod_tools_nonexistent_dir(self):
        """discover_mod_tools returns empty for non-existent directory."""
        from enigma_engine.core.mod_tools import discover_mod_tools

        tools = discover_mod_tools(Path("/nonexistent/dir"))
        assert tools == []

    def test_register_mod_commands_returns_count(self):
        """register_mod_commands returns count of newly registered."""
        from enigma_engine.core.mod_tools import register_mod_commands
        from enigma_engine.core.commands import CommandRegistry

        registry = CommandRegistry()
        mods_dir = Path(__file__).parent.parent / "mods"
        count = register_mod_commands(registry, mods_dir)
        assert isinstance(count, int)
        assert count > 0

    def test_register_mod_commands_skips_existing(self):
        """register_mod_commands does not overwrite existing commands."""
        from enigma_engine.core.mod_tools import register_mod_commands
        from enigma_engine.core.commands import CommandRegistry, CommandResult

        registry = CommandRegistry()

        # Pre-register a command
        def sentinel(a, c):
            return CommandResult(True, "sentinel")

        registry.register("imagegen.generate", sentinel, "test", "test")
        mods_dir = Path(__file__).parent.parent / "mods"
        register_mod_commands(registry, mods_dir)
        # Should still be our sentinel, not overwritten
        assert registry._commands["imagegen.generate"].handler is sentinel

    def test_format_tools_for_prompt_empty(self):
        """format_tools_for_prompt returns empty string for no mods."""
        from enigma_engine.core.mod_tools import format_tools_for_prompt

        assert format_tools_for_prompt([]) == ""

    def test_format_tools_for_prompt_includes_mods(self):
        """format_tools_for_prompt includes mod names and commands."""
        from enigma_engine.core.mod_tools import format_tools_for_prompt

        mods_data = [
            {
                "id": "test_mod",
                "name": "Test Mod",
                "description": "A test mod",
                "_running": True,
                "commands_full": [
                    {
                        "name": "do_thing",
                        "description": "Does a thing",
                        "args": {"input": {"type": "string", "required": True}},
                    },
                ],
            }
        ]
        result = format_tools_for_prompt(mods_data)
        assert "Test Mod" in result
        assert "do_thing" in result

    def test_format_tools_shows_available_status(self):
        """Stopped mods show as AVAILABLE, not RUNNING."""
        from enigma_engine.core.mod_tools import format_tools_for_prompt

        mods_data = [
            {
                "id": "stopped_mod",
                "name": "Stopped Mod",
                "_running": False,
                "commands_full": [{"name": "cmd1", "description": "d"}],
            }
        ]
        result = format_tools_for_prompt(mods_data)
        assert "AVAILABLE" in result


class TestCommandSurface:
    """The command registry and arg handling serve real Windows inputs."""

    def test_get_registry_builds_and_is_singleton(self):
        """get_registry returns a working registry (commands come from
        plugins) and hands back the same instance on repeat calls."""
        import enigma_engine.core.commands as commands_mod

        # reset the module singleton for a clean probe
        commands_mod._registry = None
        try:
            reg = commands_mod.get_registry()
            assert reg is commands_mod.get_registry()
            assert isinstance(reg.list_commands(), list)
        finally:
            commands_mod._registry = None

    def test_execute_keeps_quoted_paths_whole(self):
        """A double-quoted Windows path with spaces and (x86) arrives at
        the handler as ONE untouched argument."""
        from enigma_engine.core.commands import CommandRegistry, CommandResult

        reg = CommandRegistry()
        seen: list[list[str]] = []

        def handler(args, ctx):
            seen.append(args)
            return CommandResult(True, "ok")

        reg.register("files.open", handler, "open a file", "files.open <path>")
        result = reg.execute('files.open "C:\\Program Files (x86)\\My App\tool.txt"')
        assert result.success
        assert seen == [["C:\\Program Files (x86)\\My App\tool.txt"]]

    def test_sanitize_args_keeps_parentheses(self):
        """Parens are path characters, not injection vectors; chaining
        metacharacters still get stripped."""
        from enigma_engine.core.commands import sanitize_args

        assert sanitize_args(["(x86)"]) == ["(x86)"]
        assert sanitize_args(["a;b|c"]) == ["abc"]

    def test_mod_command_falsy_reply_is_a_response(self):
        """A mod replying "" (falsy) responded; only None is silence."""
        from enigma_engine.core.commands import CommandRegistry
        from enigma_engine.core.mod_tools import register_mod_commands

        class Router:
            def send_to_mod(self, mod_id, message):
                return ""

        import json as _json
        from pathlib import Path as _Path
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            mod_dir = _Path(td) / "echo"
            mod_dir.mkdir()
            (mod_dir / "mod.json").write_text(
                _json.dumps({"id": "echo", "name": "Echo", "commands": [{"name": "ping"}]}),
                encoding="utf-8",
            )
            reg = CommandRegistry()
            n = register_mod_commands(reg, _Path(td), router=Router())
            assert n == 1
            result = reg.execute("echo.ping")
            assert result.success, result.message
