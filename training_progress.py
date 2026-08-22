"""Where-she's-at chart for the Enigma pretrain run.

Parses the NEWEST train_*.log (append-only, one line / 10 steps plus periodic
[val] lines) and prints an ASCII progress dashboard: a completion bar, the live
numbers, and a perplexity descent curve for the whole run so far. Runs write
train_<lineage>.log, so a fixed name would follow the v1 log forever; --log
names one explicitly.

Zero dependencies, ASCII-only output (the Windows cp1252 console can't print
box-drawing or arrows). Run:  python training_progress.py [--log PATH] [--watch SECS]
"""
from __future__ import annotations

import argparse
import os
import glob
import re
import sys
import time

# The middle of the step line grew a cumulative-rate parenthetical and a peak
# VRAM figure, either of which may be absent (CPU runs print no peak). Anchor on
# the two ends -- the rate and the trailing token total -- and skip whatever
# sits between them, so the next field added there does not blind this reader
# again. The captured rate is the WINDOWED one, which is the honest gauge.
STEP_RE = re.compile(
    r"^step (\d+)/(\d+) loss ([\d.]+|nan|inf|-inf) lr ([\d.eE+-]+) ([\d,]+) tok/s"
    r"(?:.*?)\s([\d.]+)B\s*$"
)
# Same non-finite tolerance as STEP_RE: `loss_acc` and the val loss are both
# printed with :.4f, which renders nan/inf. Hardening one and not the other
# left the ppl curve frozen at the last finite val while the run diverged.
VAL_RE = re.compile(r"\[val\] step (\d+) loss ([\d.]+|nan|-?inf) ppl ([\d.]+|nan|-?inf)")
TOKSTEP_RE = re.compile(r"([\d,]+) tok/step")


def parse(log_path):
    """Return (last_step_fields, val_series, tok_per_step) from the log."""
    last_step = None
    vals = []  # (step, ppl)
    tok_per_step = 196608  # default; overridden by the 'training:' line if present
    with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = STEP_RE.match(line)
            if m:
                last_step = (
                    int(m.group(1)),           # cur
                    int(m.group(2)),           # total
                    float(m.group(3)),         # loss
                    m.group(4),                # lr (str)
                    int(m.group(5).replace(",", "")),  # tok/s
                    float(m.group(6)),         # cumulative tokens (B)
                )
                continue
            mv = VAL_RE.search(line)
            if mv:
                vals.append((int(mv.group(1)), float(mv.group(3))))
                continue
            if "tok/step" in line:
                mt = TOKSTEP_RE.search(line)
                if mt:
                    tok_per_step = int(mt.group(1).replace(",", ""))
    return last_step, vals, tok_per_step


def fmt_eta(seconds):
    seconds = int(seconds)
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, _ = divmod(rem, 60)
    if d:
        return f"{d}d {h}h {m}m"
    if h:
        return f"{h}h {m}m"
    return f"{m}m"


def bar(pct, width=42):
    filled = round(pct * width)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def ascii_plot(points, width=60, height=12):
    """Scatter/line plot of (x, y) points using ASCII. Returns list of rows."""
    if len(points) < 2:
        return ["  (not enough val points to chart yet)"]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if ymax == ymin:
        ymax = ymin + 1.0
    if xmax == xmin:
        xmax = xmin + 1

    # grid of spaces
    grid = [[" "] * width for _ in range(height)]
    # nearest-value lookup per column (points are already step-ordered)
    j = 0
    for c in range(width):
        step_at_c = xmin + (xmax - xmin) * c / (width - 1)
        while j + 1 < len(points) and abs(points[j + 1][0] - step_at_c) <= abs(
            points[j][0] - step_at_c
        ):
            j += 1
        y = points[j][1]
        row = round((ymax - y) / (ymax - ymin) * (height - 1))
        grid[row][c] = "*"

    ylab_w = 5
    out = []
    for r in range(height):
        if r == 0:
            label = f"{ymax:{ylab_w}.2f}"
        elif r == height - 1:
            label = f"{ymin:{ylab_w}.2f}"
        else:
            label = " " * ylab_w
        out.append(label + " |" + "".join(grid[r]))
    out.append(" " * ylab_w + " +" + "-" * width)
    xlabel = ("step " + str(xmin)).ljust(width // 2) + ("step " + str(xmax)).rjust(
        width - width // 2
    )
    out.append(" " * (ylab_w + 2) + xlabel)
    return out


def render(log_path):
    last, vals, tok_per_step = parse(log_path)
    lines = []
    lines.append("=" * 74)
    lines.append("  ENIGMA PRETRAIN -- where she's at")
    lines.append("=" * 74)
    if last is None:
        lines.append(f"  No 'step' lines found in {log_path} yet.")
        return "\n".join(lines)

    cur, total, loss, lr, tok_s, tokens_b = last
    pct = cur / total if total else 0.0
    remaining_steps = max(0, total - cur)
    sec_per_step = (tok_per_step / tok_s) if tok_s else 0.0
    eta = fmt_eta(remaining_steps * sec_per_step) if sec_per_step else "n/a"
    target_b = total * tok_per_step / 1e9
    latest_ppl = vals[-1][1] if vals else None

    lines.append("")
    lines.append(f"  {bar(pct)} {pct * 100:5.1f}%")
    lines.append("")
    lines.append(f"  step        {cur:,} / {total:,}")
    lines.append(f"  tokens      {tokens_b:.2f}B / {target_b:.1f}B")
    lines.append(f"  train loss  {loss:.4f}      lr {lr}")
    if latest_ppl is not None:
        lines.append(f"  val ppl     {latest_ppl:.2f}  (latest of {len(vals)} evals)")
    lines.append(f"  throughput  {tok_s:,} tok/s   (~{sec_per_step:.1f}s/step)")
    lines.append(f"  eta         {eta} of pure compute ({remaining_steps:,} steps left)")
    lines.append("")
    lines.append("  val perplexity over the run:")
    for row in ascii_plot(vals):
        lines.append("  " + row)
    lines.append("=" * 74)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="ASCII progress chart for Enigma pretraining.")
    here = os.path.dirname(os.path.abspath(__file__))
    # Runs write train_<lineage>.log, so a fixed default points at the v1 log
    # forever. Follow the newest, and name it -- the same rule
    # tail_training_log.ps1 uses.
    _newest = sorted(glob.glob(os.path.join(here, "train_*.log")),
                     key=os.path.getmtime, reverse=True)
    ap.add_argument("--log", default=(_newest[0] if _newest
                                      else os.path.join(here, "train_large.log")),
                    help="path to the training log (default: the newest train_*.log)")
    ap.add_argument("--watch", type=float, default=0.0, metavar="SECS",
                    help="redraw every SECS seconds (Ctrl-C to stop)")
    args = ap.parse_args()

    if not os.path.exists(args.log):
        print(f"Log not found: {args.log}")
        sys.exit(1)

    if args.watch and args.watch > 0:
        try:
            while True:
                # constant command string, no user input reaches the shell
                os.system("cls" if os.name == "nt" else "clear")  # noqa: S605
                print(render(args.log))
                print(f"\n  (watching every {args.watch:.0f}s -- Ctrl-C to stop)")
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nstopped.")
    else:
        print(render(args.log))


if __name__ == "__main__":
    main()
