"""Enigma's own desktop window -- no browser.

A thin frame (pywebview over the WebView2 runtime that ships with
Windows 11) around the chat page serve_enigma.py hosts at
http://127.0.0.1:8000/. The page stays the single source of truth;
this file only gives it a window of its own -- taskbar entry, alt-tab
target, no tabs or address bar. Deliberately throwaway: when the Unity
avatar becomes her window, this shim retires with nothing lost.

The window opens on a local boot page and navigates to the chat page
only once the server's port actually answers, so a slow cold start
(model load, AV scan) shows "waking her up" instead of a dead
navigation-error page.

Usage:
    py -3.12 enigma_window.py [--on-top] [--url URL]

--on-top keeps the window above a fullscreen-windowed game. If pywebview
or its WebView2 backend is unavailable, the default browser is opened
instead -- something always opens.
"""

import socket
import sys
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


def _load_when_up(window, url: str) -> None:
    """Poll the server's TCP port from a worker thread; navigate the window
    off the boot page the moment it answers. Never gives up -- the boot page
    itself tells the user where to look if it takes minutes."""
    p = urllib.parse.urlparse(url)
    host = p.hostname or "127.0.0.1"
    port = p.port or (443 if p.scheme == "https" else 80)
    while True:
        try:
            with socket.create_connection((host, port), timeout=1):
                window.load_url(url)
                return
        except OSError:
            time.sleep(1)


def main() -> int:
    url, on_top = _parse_args(sys.argv[1:])
    try:
        import webview
    except ImportError:
        print("pywebview is not installed -- opening the browser instead.")
        print("to get the window back: py -3.12 -m pip install pywebview")
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
        webview.start(_load_when_up, (window, url))
    except Exception as exc:  # WebView2 runtime broken/missing, etc.
        print(f"window failed ({exc}) -- opening the browser instead.")
        webbrowser.open(url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
