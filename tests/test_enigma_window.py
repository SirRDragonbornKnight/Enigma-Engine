"""The window shim's argument parsing must be safe against malformed input --
the shim usually runs under pyw, where nothing printed is ever seen, so a
bad --url must degrade to the default instead of navigating to garbage."""

import json
import sys

import pytest

import enigma_window
from enigma_engine.core.persona import Persona
from enigma_window import DEFAULT_URL, _boot_html, _host_port, _parse_args

HER = Persona()


def test_defaults():
    assert _parse_args([]) == (DEFAULT_URL, False, HER)


def test_on_top_flag():
    assert _parse_args(["--on-top"]) == (DEFAULT_URL, True, HER)


def test_url_with_value():
    assert _parse_args(["--url", "http://127.0.0.1:9000/"]) == ("http://127.0.0.1:9000/", False, HER)


def test_trailing_url_falls_back_to_default():
    assert _parse_args(["--url"]) == (DEFAULT_URL, False, HER)


def test_url_never_swallows_a_flag():
    assert _parse_args(["--url", "--on-top"]) == (DEFAULT_URL, True, HER)


def test_host_port_defaults():
    assert _host_port("http://127.0.0.1:8000/") == ("127.0.0.1", 8000)
    assert _host_port("https://example.test/") == ("example.test", 443)


def test_out_of_range_port_falls_back_to_default():
    assert _parse_args(["--url", "http://127.0.0.1:99999/"]) == (DEFAULT_URL, False, HER)


def test_garbage_url_falls_back_to_default():
    assert _parse_args(["--url", "not a url"]) == (DEFAULT_URL, False, HER)


def test_her_boot_page_is_unchanged():
    """The default window is the one it has always been: her name in the
    title and the heading, and HER error log named as the place to look when
    a cold start takes minutes."""
    page = _boot_html(HER)
    assert "<title>Enigma</title>" in page
    assert "<h2>Waking Enigma up...</h2>" in page
    assert "Check serve_enigma.err.log in the Enigma Engine folder" in page
    assert "__PERSONA_NAME__" not in page and "__ERR_LOG__" not in page


def test_a_pack_opens_a_window_of_her_own(tmp_path):
    """Two AIs mean two windows, and a taskbar entry titled "Enigma" over the
    other one's chat is the confusion this closes. The error log follows the
    same slug the launcher writes, so the page points at a file that exists."""
    pack = tmp_path / "atlas"
    pack.mkdir()
    (pack / "pack.json").write_text(json.dumps({
        "name": "Atlas Prime", "data_dirname": ".atlas_prime", "port": 8300,
    }), encoding="utf-8")

    url, on_top, persona = _parse_args(["--url", "http://127.0.0.1:8300/", "--persona", str(pack)])
    assert (url, on_top) == ("http://127.0.0.1:8300/", False)
    assert persona.name == "Atlas Prime"
    page = _boot_html(persona)
    assert "<title>Atlas Prime</title>" in page
    assert "Check serve_atlas_prime.err.log in the Enigma Engine folder" in page


def test_a_persona_flag_with_no_pack_opens_as_her():
    """Same rule as --url: under pyw nobody reads the WARN, so the window has
    to open on something. A pack path that cannot be guessed is not one."""
    assert _parse_args(["--persona"]) == (DEFAULT_URL, False, HER)


def test_a_missing_pack_refuses_rather_than_mislabelling_the_window(tmp_path):
    """A NAMED pack that is not there is a different failure from a missing
    flag: titling the window "Enigma" over an AI the launcher just started
    would be the lie this whole wave exists to stop."""
    with pytest.raises(SystemExit):
        _parse_args(["--persona", str(tmp_path / "absent")])


def test_a_refused_pack_is_a_popup_not_a_silent_death(monkeypatch, tmp_path):
    """...and that refusal has to be VISIBLE. The shim runs under pyw, where
    SystemExit's message goes nowhere at all: the Talk shortcut was clicked,
    the server started, and the window never appeared with nothing anywhere
    saying why. The chain's standing rule is that a failure is a popup."""
    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(enigma_window, "_message_box",
                        lambda text, title: seen.append((text, title)))
    monkeypatch.setattr(sys, "argv",
                        ["enigma_window.py", "--persona", str(tmp_path / "absent")])
    with pytest.raises(SystemExit):
        enigma_window.main()
    assert len(seen) == 1, seen
    text, title = seen[0]
    assert "persona pack" in text and str(tmp_path / "absent") in text
    assert title == "Enigma"


def test_a_good_launch_raises_no_popup(monkeypatch):
    """The other direction: the popup belongs to the refusal only. main()
    returns through the browser fallback here (no pywebview in the test
    environment is fine -- either way nothing pops)."""
    seen: list[tuple[str, str]] = []
    monkeypatch.setattr(enigma_window, "_message_box",
                        lambda text, title: seen.append((text, title)))
    monkeypatch.setattr(enigma_window, "_wait_for_port", lambda url, seconds: None)
    monkeypatch.setattr(enigma_window.webbrowser, "open", lambda url: True)
    monkeypatch.setitem(sys.modules, "webview", None)  # `import webview` -> ImportError
    monkeypatch.setattr(sys, "argv", ["enigma_window.py"])
    assert enigma_window.main() == 0
    assert seen == []


@pytest.mark.skipif(sys.platform != "win32", reason="user32 is Windows-only")
def test_the_popup_is_the_zero_dependency_user32_call(monkeypatch):
    """No pywebview, no WinForms, no console -- MessageBoxW through ctypes is
    the one popup this shim can always raise."""
    import ctypes

    calls: list[tuple] = []
    monkeypatch.setattr(
        ctypes.windll.user32, "MessageBoxW",
        lambda hwnd, text, title, flags: calls.append((hwnd, text, title, flags)) or 1,
    )
    enigma_window._message_box("boom", "Enigma")
    assert calls == [(None, "boom", "Enigma", 0x10)]
