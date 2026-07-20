#!/usr/bin/env python
"""Post-hoc checkpoint EMA -- the free win from the v2 recipe (IMU-1: EMA of
the final ~10 archived checkpoints at beta 0.8 beats the last checkpoint).

Runs OFFLINE over already-saved checkpoints, so it costs nothing during
training and can be re-run with any beta. Feed it the archive snapshots of
ONE lineage in training order (oldest first; step order is verified from the
checkpoints themselves when recorded):

  python ema_checkpoints.py models/run/step_280000.pth models/run/step_285000.pth \
      models/run/model.pth --out models/run/model_ema.pth --beta 0.8

ema <- first; then for each later checkpoint: ema = beta*ema + (1-beta)*w,
accumulated in float32, written back in the original dtype. The output is the
standard {model_state_dict, config, step} format the rest of the stack loads,
plus an "ema" provenance block.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def _load(path: Path) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in ck or "config" not in ck:
        raise SystemExit(f"{path}: not a training checkpoint (missing model_state_dict/config)")
    return ck


def ema_state_dicts(states: list[dict], beta: float) -> dict:
    """EMA over state dicts in training order. Float tensors are averaged in
    fp32; integer tensors (counters, if any ever appear) take the LAST value.
    Key sets and shapes must match exactly -- a mismatch means the inputs are
    not one lineage, and averaging them would build a corrupt model."""
    if len(states) < 2:
        raise ValueError(f"need >= 2 checkpoints for an EMA, got {len(states)}")
    first = states[0]
    keys = set(first.keys())
    for i, sd in enumerate(states[1:], 1):
        if set(sd.keys()) != keys:
            missing = keys.symmetric_difference(sd.keys())
            raise ValueError(f"checkpoint {i} key set differs (e.g. {sorted(missing)[:3]}) -- not one lineage")
        for k in keys:
            if sd[k].shape != first[k].shape:
                raise ValueError(f"checkpoint {i} shape mismatch at {k}: {tuple(sd[k].shape)} vs {tuple(first[k].shape)}")

    out = {}
    for k in keys:
        if not first[k].is_floating_point():
            out[k] = states[-1][k].clone()
            continue
        acc = first[k].float().clone()
        for sd in states[1:]:
            acc.mul_(beta).add_(sd[k].float(), alpha=1.0 - beta)
        out[k] = acc.to(first[k].dtype)
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+", help="checkpoints of ONE lineage, oldest first")
    ap.add_argument("--out", required=True, help="output path for the EMA model")
    ap.add_argument("--beta", type=float, default=0.8, help="EMA decay (IMU-1 free-win value: 0.8)")
    args = ap.parse_args(argv)

    if not (0.0 < args.beta < 1.0):
        raise SystemExit(f"--beta must be in (0, 1), got {args.beta}")
    paths = [Path(p) for p in args.checkpoints]
    out_path = Path(args.out)
    if out_path.resolve() in {p.resolve() for p in paths}:
        raise SystemExit(f"--out {out_path} is one of the inputs -- refusing to overwrite a source checkpoint")

    cks = [_load(p) for p in paths]

    # Same-lineage guards: identical configs, strictly increasing steps
    # (when recorded -- final model.pth saves carry "step" too).
    cfg0 = cks[0]["config"]
    for p, ck in zip(paths[1:], cks[1:]):
        if ck["config"] != cfg0:
            raise SystemExit(f"{p}: config differs from {paths[0]} -- not one lineage, refusing to average")
    steps = [ck.get("step") for ck in cks]
    known = [(i, s) for i, s in enumerate(steps) if s is not None]
    for (i, si), (j, sj) in zip(known, known[1:]):
        if sj <= si:
            raise SystemExit(
                f"checkpoint order is wrong: {paths[j]} (step {sj}) does not come after "
                f"{paths[i]} (step {si}) -- pass them oldest first"
            )

    ema = ema_state_dicts([ck["model_state_dict"] for ck in cks], args.beta)

    from enigma_engine.core.safe_save import atomic_torch_save

    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(
        {
            "model_state_dict": ema,
            "config": cfg0,
            "step": steps[-1],
            "ema": {"beta": args.beta, "n": len(cks), "steps": steps, "sources": [str(p) for p in paths]},
        },
        str(out_path),
    )
    print(
        f"ema: {len(cks)} checkpoints (beta {args.beta}, steps {steps[0]}..{steps[-1]}) -> {out_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
