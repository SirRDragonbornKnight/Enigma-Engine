"""GUI baseline measurement helper for Phase 0b.

What this script does:
- M3 (Idle RAM): polls RSS of a GUI process by PID after a 60s settle.
- M4 (Packaged size estimate): sums GUI source + customtkinter + Pillow on disk.

What this script does NOT do (operator-only):
- M1 (Cold start): requires instrumented `EnigmaGUI.__init__` print pairs.
- M2 (Page-switch latency): requires `<Button-1>` timing inside the live GUI.
- M5 (Frame stall): requires the after-tick frame monitor inside a real training run.

Usage:
    # M4 alone (no live GUI required):
    python information/gui/measure_baseline.py --m4

    # M3 (operator first launches GUI, then runs this with the PID):
    python information/gui/measure_baseline.py --m3 --pid 12345 --settle 60

    # Print measurement protocol summary:
    python information/gui/measure_baseline.py --help
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def m4_estimate() -> dict[str, float]:
    """Sum GUI source tree + customtkinter + Pillow as a packaged-size proxy."""
    repo = Path(__file__).resolve().parent.parent.parent
    gui_dir = repo / "enigma_engine" / "gui"
    gui_bytes = sum(f.stat().st_size for f in gui_dir.rglob("*") if f.is_file())
    try:
        import customtkinter

        ctk_root = Path(customtkinter.__file__).parent
        ctk_bytes = sum(f.stat().st_size for f in ctk_root.rglob("*") if f.is_file())
    except ImportError:
        ctk_bytes = 0
    try:
        import PIL

        pil_root = Path(PIL.__file__).parent
        pil_bytes = sum(f.stat().st_size for f in pil_root.rglob("*") if f.is_file())
    except ImportError:
        pil_bytes = 0

    mb = 1024 * 1024
    return {
        "gui_src_mb": round(gui_bytes / mb, 1),
        "customtkinter_mb": round(ctk_bytes / mb, 1),
        "pillow_mb": round(pil_bytes / mb, 1),
        "sum_mb": round((gui_bytes + ctk_bytes + pil_bytes) / mb, 1),
    }


def m3_rss(pid: int, settle_seconds: int) -> float:
    """Return RSS (MB) of process `pid` after a `settle_seconds` wait."""
    try:
        import psutil
    except ImportError:
        print(
            "[!] psutil not installed. Install with: pip install psutil",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        print(f"[!] No process with PID {pid}", file=sys.stderr)
        sys.exit(1)

    print(f"[M3] Settling {settle_seconds}s on PID {pid} (name={proc.name()})...")
    time.sleep(settle_seconds)
    rss_bytes = proc.memory_info().rss
    return round(rss_bytes / (1024 * 1024), 1)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="GUI baseline measurement helper (Phase 0b)",
    )
    parser.add_argument("--m4", action="store_true", help="Run M4 size estimate")
    parser.add_argument("--m3", action="store_true", help="Run M3 idle RSS poll")
    parser.add_argument("--pid", type=int, help="PID of running GUI process (M3)")
    parser.add_argument(
        "--settle",
        type=int,
        default=60,
        help="Settle seconds before reading RSS (M3, default 60)",
    )
    args = parser.parse_args()

    if not (args.m3 or args.m4):
        parser.print_help()
        return 0

    if args.m4:
        result = m4_estimate()
        print("[M4] Packaged install size estimate (UI surface only):")
        print(f"      GUI src       : {result['gui_src_mb']} MB")
        print(f"      CustomTkinter : {result['customtkinter_mb']} MB")
        print(f"      Pillow        : {result['pillow_mb']} MB")
        print(f"      Sum           : {result['sum_mb']} MB")
        print(
            "      Note: excludes Python interpreter (~30 MB) and torch "
            "(~2 GB) — those are out of scope per rubric §5 row 3.",
        )

    if args.m3:
        if args.pid is None:
            print("[!] --m3 requires --pid", file=sys.stderr)
            return 1
        rss_mb = m3_rss(args.pid, args.settle)
        print(f"[M3] Idle RSS after {args.settle}s settle: {rss_mb} MB")
        print(
            "      Run this 3 times across cold launches; record median in BASELINE.md §3.",
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
