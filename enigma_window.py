"""Enigma's own desktop window -- no browser.

A thin frame (pywebview over the WebView2 runtime that ships with
Windows 11) around the chat page serve_enigma.py hosts at
http://127.0.0.1:8000/. The page stays the single source of truth;
this file only gives it a window of its own -- taskbar entry, alt-tab
target, no tabs or address bar. Deliberately throwaway: when the Unity
avatar becomes her window, this shim retires with nothing lost.

The window opens on a local boot page and navigates to the chat page
once the server's port answers, so a slow cold start (model load, AV
scan) shows "waking her up" instead of a dead navigation-error page.
The poller runs on a DAEMON thread and stops when the window closes --
webview.start's own helper thread is non-daemon, and a never-give-up
loop there kept a ghost pythonw alive after the window was closed
(2026-07-17 audit).

Usage:
    py -3.12 enigma_window.py [--on-top] [--url URL] [--persona PACK_DIR]

--on-top keeps the window above a fullscreen-windowed game. If pywebview
or its WebView2 backend is unavailable, the default browser is opened
instead (after waiting up to 2 minutes for the server, so it doesn't
open onto a connection-refused page mid cold-start).

--persona is the pack a generated Talk shim passes: the window is titled
for whoever it opens onto, and its boot page names HER error log. Omitted
is Enigma, which is what every value here already said. A NAMED pack that
will not load stops the window with a POPUP -- under pyw the printed
message goes nowhere, so the shortcut was clicked and nothing happened.
"""

import html
import socket
import sys
import threading
import time
import urllib.parse
import webbrowser
from pathlib import Path

from enigma_engine.core.persona import Persona

DEFAULT_URL = "http://127.0.0.1:8000/"

# `__PERSONA_NAME__` and `__ERR_LOG__` are filled in by _boot_html(): the page
# is a constant and the identity is only known once the arguments are parsed.
_BOOT_HTML = """<!doctype html><html><head><title>__PERSONA_NAME__</title><style>
body { background: #1e1e2e; color: #cba6f7; font-family: 'Segoe UI', sans-serif;
       display: flex; align-items: center; justify-content: center; height: 96vh; }
div { text-align: center; }
small { color: #6c7086; }
</style></head><body><div>
<h2>Waking __PERSONA_NAME__ up...</h2>
<p><small>The window switches to her chat the moment the server answers.<br>
Taking minutes? Check __ERR_LOG__ in the Enigma Engine folder.</small></p>
</div></body></html>"""


def _boot_html(persona: Persona) -> str:
    """The waking-up page, for whoever this window opens onto. The name is
    escaped because a pack is data -- it lands inside markup here, exactly as
    it does in the chat page serve renders."""
    return (_BOOT_HTML
            .replace("__PERSONA_NAME__", html.escape(persona.name))
            .replace("__ERR_LOG__", f"serve_{persona.slug}.err.log"))


def _valid_url(candidate: str) -> bool:
    """http(s) with a REAL hostname and a usable (or absent) port -- a garbage
    URL must fall back to the default, not leave the poller retrying a target
    that can never answer. urlparse defers port validation to .port, treats 0
    as in-range (nothing can listen on 0 client-side), and accepts whitespace
    hostnames -- all three fall back here (re-audit 2026-07-17)."""
    try:
        p = urllib.parse.urlparse(candidate)
        port = p.port  # raises ValueError on non-numeric / out-of-range ports
        if port == 0:
            return False
        host = p.hostname
        return p.scheme in ("http", "https") and host is not None and host.strip() != ""
    except ValueError:
        return False


def _parse_args(args: list[str]) -> tuple[str, bool, Persona]:
    on_top = "--on-top" in args
    url = DEFAULT_URL
    if "--url" in args:
        i = args.index("--url")
        if i + 1 < len(args) and not args[i + 1].startswith("--"):
            if _valid_url(args[i + 1]):
                url = args[i + 1]
            else:
                print(f"WARN: --url {args[i + 1]!r} is not a valid http(s) URL; using default {DEFAULT_URL}")
        else:
            print(f"WARN: --url needs a value; using default {DEFAULT_URL}")
    pack = None
    if "--persona" in args:
        i = args.index("--persona")
        if i + 1 < len(args) and not args[i + 1].startswith("--"):
            pack = Path(args[i + 1])
        else:
            # A bad --url degrades to the default because the window can still
            # be useful pointed at her; a missing pack DIRECTORY cannot be
            # guessed, so the window opens as Enigma and says it did.
            print("WARN: --persona needs a pack directory; opening Enigma's window")
    return url, on_top, Persona.load(pack)


def _message_box(text: str, title: str) -> None:
    """A popup with no dependencies: user32's MessageBoxW through ctypes.
    pywebview may be the very thing that is broken, and this shim runs under
    pyw, where nothing printed is ever seen. MB_ICONERROR | MB_OK."""
    import ctypes

    ctypes.windll.user32.MessageBoxW(None, text, title, 0x10)


def _report_fatal(message: str) -> None:
    """Raise a startup refusal as a POPUP before the process dies. Persona.load
    raises SystemExit on a pack that is missing, malformed or not hers, and
    under the windowless launch its message goes nowhere at all -- the shortcut
    was clicked, the server started, and no window ever appeared (contra the
    chain's rule that a failure is a popup). SystemExit still prints it for the
    console launches, so this adds the one channel pyw has and does not
    duplicate the one it already had."""
    try:
        _message_box(message, "Enigma")
    except Exception as exc:  # no user32 (non-Windows), broken ctypes
        print(f"(could not raise a popup: {exc})")


def _host_port(url: str) -> tuple[str, int]:
    p = urllib.parse.urlparse(url)
    return p.hostname or "127.0.0.1", p.port or (443 if p.scheme == "https" else 80)


def _wait_for_port(url: str, seconds: float) -> None:
    """Bounded wait used by the browser fallbacks -- a browser has no boot
    page, so opening it mid cold-start lands on connection-refused."""
    host, port = _host_port(url)
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            time.sleep(0.5)


def _start_poller(window, url: str) -> None:
    """Runs on webview.start's NON-daemon helper thread: hand off to a daemon
    poller immediately so nothing here can pin the process alive after the
    window closes."""
    threading.Thread(target=_poll_and_load, args=(window, url), daemon=True).start()


def _poll_and_load(window, url: str) -> None:
    closed = threading.Event()
    try:
        window.events.closed += closed.set
    except Exception:
        pass  # no events API: the daemon flag still guarantees process exit
    host, port = _host_port(url)
    while not closed.is_set():
        try:
            with socket.create_connection((host, port), timeout=1):
                break
        except OSError:
            closed.wait(1)
    if closed.is_set():
        return
    try:
        window.load_url(url)
    except Exception:
        pass  # window destroyed between the port check and the load


def main() -> int:
    try:
        url, on_top, persona = _parse_args(sys.argv[1:])
    except SystemExit as exc:
        _report_fatal(str(exc) or "the persona pack could not be loaded")
        raise
    try:
        import webview
    except ImportError:
        print("pywebview is not installed -- opening the browser instead.")
        print("to get the window back: py -3.12 -m pip install pywebview")
        _wait_for_port(url, 120)
        webbrowser.open(url)
        return 0
    try:
        window = webview.create_window(
            persona.name,
            html=_boot_html(persona),
            width=520,
            height=760,
            min_size=(360, 480),
            on_top=on_top,
        )
        webview.start(_start_poller, (window, url))
    except Exception as exc:  # WebView2 runtime broken/missing, etc.
        print(f"window failed ({exc}) -- opening the browser instead.")
        _wait_for_port(url, 120)
        webbrowser.open(url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
