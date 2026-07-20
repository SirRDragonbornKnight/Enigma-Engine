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
accumulated in float32, written back in the original dtype. Checkpoints are
STREAMED one at a time with optimizer state dropped on load (audit
2026-07-20: holding ten full checkpoints was ~20 GiB at 182M and would not
fit the rig at the 542M candidate). The output is the standard
{model_state_dict, config, step} format the rest of the stack loads, plus an
"ema" provenance block. NOTE: the output carries no optimizer state --
pretrain_enigma refuses to --resume it (use --init-from to continue from it).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Iterator

import torch


def _load(path: Path) -> dict:
    # weights_only=True: this format is plain tensors/primitives, and it
    # removes the arbitrary-code-execution surface of full unpickling.
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if "model_state_dict" not in ck or "config" not in ck:
        raise SystemExit(f"{path}: not a training checkpoint (missing model_state_dict/config)")
    # Keep only what the EMA needs -- the optimizer moments are 2/3 of a
    # checkpoint's bytes and this tool never reads them.
    return {"model_state_dict": ck["model_state_dict"], "config": ck["config"], "step": ck.get("step")}


def ema_state_dicts(states: Iterable[dict], beta: float) -> dict:
    """EMA over state dicts in training order (accepts a generator, so
    checkpoints can be streamed). Float tensors are averaged in fp32; integer
    tensors (counters, if any ever appear) take the LAST value. Key sets and
    shapes must match exactly -- a mismatch means the inputs are not one
    lineage, and averaging them would build a corrupt model."""

    def _check_tensors(sd: dict, idx: int) -> None:
        for k, v in sd.items():
            if not torch.is_tensor(v):
                raise ValueError(f"checkpoint {idx} entry {k!r} is not a tensor ({type(v).__name__})")

    it: Iterator[dict] = iter(states)
    try:
        first = next(it)
    except StopIteration:
        raise ValueError("need >= 2 checkpoints for an EMA, got 0") from None
    _check_tensors(first, 0)
    keys = set(first.keys())
    shapes = {k: first[k].shape for k in keys}
    dtypes = {k: first[k].dtype for k in keys}
    acc = {k: first[k].float().clone() for k in keys if first[k].is_floating_point()}
    last_int = {k: first[k].clone() for k in keys if not first[k].is_floating_point()}

    n = 1
    for sd in it:
        _check_tensors(sd, n)
        if set(sd.keys()) != keys:
            missing = keys.symmetric_difference(sd.keys())
            raise ValueError(f"checkpoint {n} key set differs (e.g. {sorted(missing)[:3]}) -- not one lineage")
        for k in keys:
            if sd[k].shape != shapes[k]:
                raise ValueError(f"checkpoint {n} shape mismatch at {k}: {tuple(sd[k].shape)} vs {tuple(shapes[k])}")
            if k in acc:
                acc[k].mul_(beta).add_(sd[k].float(), alpha=1.0 - beta)
            else:
                last_int[k] = sd[k].clone()
        n += 1
    if n < 2:
        raise ValueError(f"need >= 2 checkpoints for an EMA, got {n}")

    out = dict(last_int)
    for k, v in acc.items():
        out[k] = v.to(dtypes[k])
    return out


def _normalized_config(raw: dict) -> dict:
    """Project a checkpoint's config blob onto the CURRENT field set.

    Raw dict equality refused legitimate single-lineage archive windows that
    straddle a code upgrade -- the on-disk configs span eras (34/35/40 keys
    measured), because to_dict() gains fields and drops retired ones (audit
    2026-07-20). Round-tripping through ForgeConfig fills defaults for
    missing keys and filters retired ones, so only REAL differences remain.
    """
    from enigma_engine.core.model_presets import ForgeConfig

    return ForgeConfig.from_dict(raw).to_dict()


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

    # Streamed accumulation: one checkpoint in memory at a time. Lineage
    # guards run per checkpoint as it loads; a violation aborts before any
    # output is written.
    steps: list = []
    cfg_raw0: dict = {}
    cfg_norm0: dict = {}

    def _stream():
        last_step = None
        last_step_path = None
        for i, p in enumerate(paths):
            ck = _load(p)
            cfg_norm = _normalized_config(ck["config"])
            if i == 0:
                cfg_raw0.update(ck["config"])
                cfg_norm0.update(cfg_norm)
            elif cfg_norm != cfg_norm0:
                diff = sorted(k for k in set(cfg_norm) | set(cfg_norm0) if cfg_norm.get(k) != cfg_norm0.get(k))
                raise SystemExit(
                    f"{p}: config differs from {paths[0]} on {diff[:5]} -- not one lineage, refusing to average"
                )
            step = ck["step"]
            steps.append(step)
            if step is not None:
                if last_step is not None and step <= last_step:
                    raise SystemExit(
                        f"checkpoint order is wrong: {p} (step {step}) does not come after "
                        f"{last_step_path} (step {last_step}) -- pass them oldest first"
                    )
                last_step, last_step_path = step, p
            yield ck["model_state_dict"]

    ema = ema_state_dicts(_stream(), args.beta)

    if all(s is None for s in steps):
        print(
            "WARN: no input checkpoint records a step -- training order CANNOT be "
            "verified; the EMA is only correct if you passed them oldest first",
            flush=True,
        )

    from enigma_engine.core.safe_save import atomic_torch_save

    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(
        {
            "model_state_dict": ema,
            "config": cfg_raw0,
            "step": steps[-1],
            "ema": {"beta": args.beta, "n": len(steps), "steps": steps, "sources": [str(p) for p in paths]},
        },
        str(out_path),
    )
    print(
        f"ema: {len(steps)} checkpoints (beta {args.beta}, steps {steps[0]}..{steps[-1]}) -> {out_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
