"""Strip a training checkpoint down to what SERVING actually reads.

A trained .pth carries the whole run: optimizer moments, scheduler state,
scaler state, RNG -- everything a resume needs and a server never opens. boot()
reads exactly four keys (serve_enigma.py: config :628, model_state_dict
:653/:659, step :679, meta :680), so everything else is weight the portable
build would carry across a USB stick for nothing.

    python strip_serving_ckpt.py --in models/enigma_v2_sft2/model.pth \\
                                 --out "C:\\Users\\SirKn\\Enigma Backups\\v2_serving.pth"

The output is a NEW artifact every time: an existing --out is REFUSED, never
rebuilt in place (the versioned-artifact rule the trainers already follow).
The source is opened read-only and is never modified.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from enigma_engine.core.safe_save import atomic_torch_save

# The four keys boot() reads. `config` and `model_state_dict` are indexed
# directly (a missing one is a dead server), `step` and `meta` come through
# .get() and are optional -- but they are carried when present because STEP
# and META are what a transcript uses to say WHICH weights answered.
SERVING_KEYS = ("model_state_dict", "config", "step", "meta")
REQUIRED_KEYS = ("model_state_dict", "config")


def strip(src: str | Path, out: str | Path) -> dict:
    """Write a serving-only copy of the checkpoint at *src* to *out*.

    Returns the key inventory of the ORIGINAL as {key: what it holds}, so the
    caller can report what the strip actually dropped.
    """
    src, out = Path(src), Path(out)
    if not src.is_file():
        raise SystemExit(f"REFUSED: --in {src} does not exist")
    if out.exists():
        raise SystemExit(
            f"REFUSED: {out} already exists -- model artifacts are versioned, "
            f"never rebuilt in place. Name a NEW --out, or delete the old "
            f"artifact deliberately first."
        )

    ck = torch.load(src, map_location="cpu", weights_only=True)
    if not isinstance(ck, dict):
        raise SystemExit(f"REFUSED: {src} is a {type(ck).__name__}, not a checkpoint dict")
    missing = [k for k in REQUIRED_KEYS if k not in ck]
    if missing:
        raise SystemExit(f"REFUSED: {src} carries no {', '.join(missing)} -- not a servable checkpoint")

    inventory = {k: _describe(v) for k, v in ck.items()}
    atomic_torch_save({k: ck[k] for k in SERVING_KEYS if k in ck}, out)
    return inventory


def _tensor_bytes(value) -> tuple[int, int]:
    """(tensor count, bytes) under *value*, RECURSING into nested containers.

    The optimizer state is the reason: its tensors hang off state[param][...],
    so a top-level-only count reports the 1.8 GB of Adam moments as "dict, 2
    keys" -- the exact number this tool exists to explain, hidden.
    """
    if torch.is_tensor(value):
        return 1, value.numel() * value.element_size()
    if isinstance(value, dict):
        value = value.values()
    elif not isinstance(value, (list, tuple)):
        return 0, 0
    count = size = 0
    for item in value:
        c, s = _tensor_bytes(item)
        count += c
        size += s
    return count, size


def _describe(value) -> str:
    """One honest line about what a checkpoint key holds, for the report."""
    count, size = _tensor_bytes(value)
    weight = f", {count} tensors, {size / (1024 * 1024):,.1f} MB" if count else ""
    if isinstance(value, dict):
        return f"dict, {len(value)} keys{weight}"
    if torch.is_tensor(value):
        return f"tensor {tuple(value.shape)}{weight}"
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__}, {len(value)} items{weight}"
    return f"{type(value).__name__}: {value!r}"[:80]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--in", dest="src", required=True, help="training checkpoint to read (never modified)")
    p.add_argument("--out", dest="out", required=True, help="serving-only .pth to write; must not exist")
    args = p.parse_args()

    inventory = strip(args.src, args.out)
    src_mb = Path(args.src).stat().st_size / (1024 * 1024)
    out_mb = Path(args.out).stat().st_size / (1024 * 1024)
    print(f"read  {args.src}  ({src_mb:,.1f} MB)")
    print("  key inventory of the original:")
    for key, what in inventory.items():
        kept = "KEPT " if key in SERVING_KEYS else "DROP "
        print(f"    {kept} {key:24s} {what}")
    print(f"wrote {args.out}  ({out_mb:,.1f} MB)")
    print(f"  saved {src_mb - out_mb:,.1f} MB ({100 * (1 - out_mb / src_mb):.1f}% of the original)")


if __name__ == "__main__":
    main()
