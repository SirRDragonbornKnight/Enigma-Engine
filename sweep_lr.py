"""Run a grid of short pretrain runs and rank them on final val loss.

Each point is a COMPLETE short run including its decay tail -- optimizer and
schedule rankings invert mid-decay, so a run judged before decay finishes ranks
the wrong config. Every point writes to its own --out directory and the corpus
is opened read-only, so a sweep can never touch a live lineage.

    python sweep_lr.py --size v2_deep_238m --block 2048 --micro-batch 4 \
        --grad-accum 8 --tokens 200000000 --lrs 1e-3,2e-3,3e-3 --seeds 0,1

Results land in <out-root>/sweep_results.json plus a printed table; the file
already there is rotated to sweep_results.prev.json, so the sweep before this
one keeps its receipts.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PRETRAIN = ROOT / "pretrain_enigma.py"

# pretrain prints: "[final] val loss 3.1234 ppl 22.72 bits/token 4.5123"
FINAL_VAL = re.compile(r"\[final\] val loss ([0-9.]+) ppl ([0-9.]+) bits/token ([0-9.]+)")
# [val] is the corpus TAIL, whatever the walk placed last -- on the v2b corpus
# that is one StackExchange site. Ranking learning rates on it ranks them on
# that site. When the run carries a fenced general-domain window
# (--val-general-end), that is the honest signal and it decides the ranking;
# [val] is still recorded so both are visible.
FINAL_VALGEN = re.compile(r"\[final\] val-gen loss ([0-9.]+) ppl ([0-9.]+)")
# Anchored to the STEP line. A bare "N tok/s" also appears in the throughput
# WARNING, and findall()[-1] would then record a degraded window as the point's
# throughput -- for exactly the points that slowed down.
THROUGHPUT = re.compile(r"^step \d+/\d+ .*?([0-9,]+) tok/s", re.MULTILINE)


def run_point(args, lr: float, seed: int, out_dir: Path) -> dict:
    cmd = [
        sys.executable, str(PRETRAIN),
        "--size", args.size,
        "--block", str(args.block),
        "--micro-batch", str(args.micro_batch),
        "--grad-accum", str(args.grad_accum),
        "--tokens", str(args.tokens),
        "--lr", str(lr),
        "--seed", str(seed),
        "--optimizer", args.optimizer,
        "--schedule", args.schedule,
        "--out", str(out_dir),
        "--save-every", str(args.save_every),
        "--eval-every", str(args.eval_every),
    ]
    if args.warmup is not None:
        cmd += ["--warmup", str(args.warmup)]
    if args.tokens_bin:
        cmd += ["--tokens-bin", args.tokens_bin]
    if args.sdpa_backend:
        cmd += ["--sdpa-backend", args.sdpa_backend]
    if args.no_grad_ckpt:
        cmd += ["--no-grad-ckpt"]
    if args.extra:
        # Prepended, not appended: argparse lets a LATER flag win, so appending
        # would let --extra silently override the sweep's own --lr/--seed/--out
        # and make every row in the ranked table a lie. posix=False keeps
        # Windows backslash paths intact (POSIX mode eats them); the quote
        # strip removes the surrounding quotes non-POSIX mode retains.
        extra = [t.strip('"') for t in shlex.split(args.extra, posix=False)]
        insert = cmd[:2]
        cmd = insert + extra + cmd[2:]

    print(f"\n=== lr={lr:g} seed={seed} -> {out_dir.name}", flush=True)
    started = time.time()
    try:
        proc = subprocess.run(
            cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=args.point_timeout or None
        )
    except subprocess.TimeoutExpired as exc:
        # a hung point is one FAILED row, never the death of the whole sweep
        partial = exc.stdout or b""
        if isinstance(partial, bytes):
            partial = partial.decode("utf-8", errors="replace")
        point = {
            "lr": lr, "seed": seed, "out": str(out_dir),
            "seconds": round(time.time() - started, 1), "returncode": None,
            "val_loss": None, "ppl": None, "bits_per_token": None, "tok_per_s": None,
            # Same key set as a completed point: a row missing rank_loss or
            # ranked_on makes sweep_results.json heterogeneous, and every
            # reader then has to guess whether the key is absent or null.
            "rank_loss": None, "ranked_on": None,
            "error": f"timed out after {args.point_timeout}s and was killed; "
                     f"last output: {partial[-300:]}",
        }
        print(f"    TIMED OUT after {args.point_timeout}s", flush=True)
        return point
    elapsed = time.time() - started
    tail = (proc.stdout or "")[-4000:] + (proc.stderr or "")[-2000:]

    point = {
        "lr": lr, "seed": seed, "out": str(out_dir), "seconds": round(elapsed, 1),
        "returncode": proc.returncode, "val_loss": None, "ppl": None, "bits_per_token": None,
        "tok_per_s": None, "error": None,
    }
    match = FINAL_VAL.search(proc.stdout or "")
    if match:
        point["val_loss"] = float(match.group(1))
        point["ppl"] = float(match.group(2))
        point["bits_per_token"] = float(match.group(3))
    else:
        # a point with no final val is a FAILED point, never a silent zero
        point["error"] = "no [final] val line in output"
    # val_loss/ppl/bits_per_token keep meaning exactly one thing: the [val]
    # tail. The general-domain window is recorded separately, and `rank_loss`
    # names which of the two the ranking compares -- overwriting val_loss in
    # place left ppl and bits/token describing a different number than the loss
    # beside them.
    vg = FINAL_VALGEN.search(proc.stdout or "")
    if vg:
        point["val_gen_loss"] = float(vg.group(1))
        point["val_gen_ppl"] = float(vg.group(2))
    point["ranked_on"] = "val_gen" if vg else "val"
    point["rank_loss"] = point.get("val_gen_loss") if vg else point["val_loss"]
    # A point that failed to produce its tail val is a FAILED point even when
    # the general window parsed -- otherwise it keeps a rankable score, enters
    # the results table, and the FAILED loop (which keys on a null rank_loss)
    # never prints its error.
    if point["error"]:
        point["rank_loss"] = None
    rates = THROUGHPUT.findall(proc.stdout or "")
    if rates:
        point["tok_per_s"] = int(rates[-1].replace(",", ""))
    if proc.returncode != 0:
        # A point that printed [final] and THEN crashed is still a failed point:
        # drop EVERY parsed score so none of them can be ranked as a good
        # result -- including the general-domain one, which is what ranks.
        point["error"] = f"exit {proc.returncode}: {tail[-600:]}"
        for key in ("val_loss", "ppl", "bits_per_token",
                    "val_gen_loss", "val_gen_ppl", "rank_loss"):
            point[key] = None
    print(
        f"    {point['ranked_on']} {point['rank_loss']} "
        f"(tail val {point['val_loss']} ppl {point['ppl']}) "
        f"({point['seconds']}s){' FAILED: ' + point['error'] if point['error'] else ''}",
        flush=True,
    )
    return point


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", required=True, help="ForgeConfig preset for every point")
    ap.add_argument("--lrs", required=True, help="comma-separated peak learning rates")
    ap.add_argument("--seeds", default="0", help="comma-separated seeds (2-3 to see the noise floor)")
    ap.add_argument("--tokens", type=int, required=True, help="tokens PER POINT (keep the horizon short)")
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--micro-batch", type=int, default=12)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--optimizer", default="muon")
    ap.add_argument("--schedule", default="wsd_sqrt")
    ap.add_argument("--tokens-bin", default=None)
    ap.add_argument("--sdpa-backend", default=None)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--save-every", type=int, default=100000, help="high by default: sweep points are throwaway")
    ap.add_argument("--eval-every", type=int, default=100000, help="the [final] val is what ranks a point")
    ap.add_argument("--out-root", default="models/sweeps", help="parent dir; each point gets a subdir")
    ap.add_argument("--warmup", type=int, default=None, help="warmup steps per point (see the horizon check below)")
    ap.add_argument("--point-timeout", type=int, default=0, help="seconds before a hung point is killed; 0 = no limit")
    ap.add_argument(
        "--extra", default="",
        help="extra flags for every point; inserted BEFORE the sweep's own flags so they cannot be overridden",
    )
    args = ap.parse_args()

    lrs = [float(x) for x in args.lrs.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    out_root = (ROOT / args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"sweep: {len(lrs)} lrs x {len(seeds)} seeds = {len(lrs) * len(seeds)} points, "
          f"{args.tokens / 1e6:.0f}M tokens each, size {args.size} block {args.block}", flush=True)

    # A point too short to leave warmup never reaches its own peak LR, so the
    # ranking would compare warmup fractions instead of learning rates.
    steps_per_point = max(1, args.tokens // (args.micro_batch * args.grad_accum * args.block))
    warmup = args.warmup if args.warmup is not None else 200  # pretrain's default
    if steps_per_point < 10 * warmup:
        print(f"  WARNING: {steps_per_point} steps/point vs {warmup} warmup steps -- points barely "
              f"leave warmup and the peak LR is never applied. Raise --tokens or lower --warmup.",
              flush=True)

    # --out-root has a fixed default, so every bare `python sweep_lr.py` writes
    # its receipts over the LAST sweep's -- a second grid silently erased the
    # numbers the first one was run to produce. Rotate ONCE, here, before any
    # point runs: the per-point rewrite below is this sweep's own file, and one
    # generation of the previous sweep survives beside it (make_sft_data's
    # _write_artifact does the same for the bake artifacts).
    results_path = out_root / "sweep_results.json"
    previous_path = results_path.with_suffix(".prev.json")
    if results_path.exists():
        os.replace(results_path, previous_path)
        print(f"  (previous receipts rotated to {previous_path.name})", flush=True)

    results = []
    for lr in lrs:
        for seed in seeds:
            tag = f"{args.size}_lr{lr:g}_s{seed}"
            results.append(run_point(args, lr, seed, out_root / tag))
            results_path.write_text(
                json.dumps(results, indent=2), encoding="utf-8"
            )  # rewritten after every point so a killed sweep keeps its receipts

    ok = [r for r in results if r.get("rank_loss") is not None]
    ranked_on = {r.get("ranked_on") for r in ok} or {"val"}
    print("\n======== SWEEP RESULTS (best first) ========", flush=True)
    if ranked_on == {"val"}:
        print("ranked on [val] -- the corpus TAIL, which is one source. Pass "
              "--extra \"--val-general-end <offset>\" to rank on a general-domain "
              "window instead.", flush=True)
    elif "val" in ranked_on:
        print("WARNING: points were ranked on DIFFERENT signals (val-gen and val) "
              "-- they are not comparable; re-run the val-only points with "
              "--val-general-end.", flush=True)
    else:
        print("ranked on [val-gen] -- the fenced general-domain window.", flush=True)
    print(f"{'lr':>10} {'seed':>5} {'rank':>9} {'tailval':>9} {'ppl':>9} "
          f"{'bits/tok':>9} {'tok/s':>10}")
    for r in sorted(ok, key=lambda r: r["rank_loss"]):
        rate = f"{r['tok_per_s']:,}" if r["tok_per_s"] else "-"
        # tail val can be absent when only the general window printed
        tail_val = f"{r['val_loss']:>9.4f}" if r["val_loss"] is not None else f"{'-':>9}"
        ppl = f"{r['ppl']:>9.2f}" if r["ppl"] is not None else f"{'-':>9}"
        bpt = f"{r['bits_per_token']:>9.4f}" if r["bits_per_token"] is not None else f"{'-':>9}"
        print(f"{r['lr']:>10g} {r['seed']:>5} {r['rank_loss']:>9.4f} {tail_val} {ppl} "
              f"{bpt} {rate:>10}")
    for r in results:
        if r.get("rank_loss") is None:
            print(f"  FAILED lr={r['lr']:g} seed={r['seed']}: {r['error']}")

    if len({r["seed"] for r in ok}) > 1:
        by_lr = {}
        for r in ok:
            by_lr.setdefault(r["lr"], []).append(r["rank_loss"])
        spreads = [max(v) - min(v) for v in by_lr.values() if len(v) > 1]
        if spreads:
            print(f"\nseed spread (same lr): max {max(spreads):.4f} -- treat lr gaps "
                  f"below this as noise, not signal", flush=True)
    print(f"\nreceipts: {results_path}", flush=True)


if __name__ == "__main__":
    main()
