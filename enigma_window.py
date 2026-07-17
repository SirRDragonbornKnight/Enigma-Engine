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
    py -3.12 enigma_window.py [--on-top] [--url URL]

--on-top keeps the window above a fullscreen-windowed game. If pywebview
or its WebView2 backend is unavailable, the default browser is opened
instead (after waiting up to 2 minutes for the server, so it doesn't
open onto a connection-refused page mid cold-start).
"""

import socket
import sys
import threading
import time
import urllib.parse
import webbrowser

DEFAULT_URL = "http://127.0.0.1:8000/"

_BOOT_HTML = """<!doctype html><html><head><title>Enigma</title><style>
body { background: #1e1e2e; color: #cba6f7; font-family: 'Segoe UI', sans-serif;
       display: flex; align-items: center; justify-content: center; height: 96vh; }
div { text-align: center; }
small { color: #6c7086; }
</style></head><body><div>
<h2>Waking Enigma up...</h2>
<p><small>The window switches to her chat the moment the server answers.<br>
Taking minutes? Check serve_enigma.err.log in the Enigma Engine folder.</small></p>
</div></body></html>"""


def _parse_args(args: list[str]) -> tuple[str, bool]:
    on_top = "--on-top" in args
    url = DEFAULT_URL
    if "--url" in args:
        i = args.index("--url")
        if i + 1 < len(args) and not args[i + 1].startswith("--"):
            url = args[i + 1]
        else:
            print(f"WARN: --url needs a value; using default {DEFAULT_URL}")
    return url, on_top


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
    url, on_top = _parse_args(sys.argv[1:])
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
            "Enigma",
            html=_BOOT_HTML,
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
