"""Enigma's own desktop window -- no browser.

A thin frame (pywebview over the WebView2 runtime that ships with
Windows 11) around the chat page serve_enigma.py hosts at
http://127.0.0.1:8000/. The page stays the single source of truth;
this file only gives it a window of its own -- taskbar entry, alt-tab
target, no tabs or address bar. Deliberately throwaway: when the Unity
avatar becomes her window, this shim retires with nothing lost.

Usage:
    py -3.12 enigma_window.py [--on-top] [--url URL]

--on-top keeps the window above a fullscreen-windowed game. Falls back
to the default browser if pywebview is missing, so the launcher never
leaves you with nothing.
"""

import sys
import webbrowser

DEFAULT_URL = "http://127.0.0.1:8000/"


def main() -> int:
    args = sys.argv[1:]
    on_top = "--on-top" in args
    url = DEFAULT_URL
    if "--url" in args:
        i = args.index("--url")
        if i + 1 < len(args):
            url = args[i + 1]
    try:
        import webview
    except ImportError:
        print("pywebview is not installed -- opening the browser instead.")
        print("to get the window back: py -3.12 -m pip install pywebview")
        webbrowser.open(url)
        return 0
    webview.create_window(
        "Enigma",
        url,
        width=520,
        height=760,
        min_size=(360, 480),
        on_top=on_top,
    )
    webview.start()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
