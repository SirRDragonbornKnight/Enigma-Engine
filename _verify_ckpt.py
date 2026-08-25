"""Scratch: prove the checkpoint's KEY SET and param count are stable across a load.

NOT a bit-identity check -- weight values are never hashed; KEYHASH covers the
sorted key names only. Load-only (CPU, no optimizer step, no data). Run before
and after any model.py / model_components.py cleanup; PARAMS and KEYHASH must
not change and MISSING/UNEXPECTED must stay empty.

Usage:
    python _verify_ckpt.py                 # BOTH live lineages
    python _verify_ckpt.py <path.pth> ...  # the checkpoints named

No argument fingerprints the adopted lineage AND the rollback, which is what
CLEANUP_TRACKER rule 2 asks for: this script's old hardcoded v1 path passed a
change that broke loading the model actually being served. Exits nonzero if
any named checkpoint is absent or does not load clean.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import hashlib
import sys
from pathlib import Path

import torch
from enigma_engine.core.model import Enigma, ForgeConfig

# Rule 2's BOTH: the served weights and the artifact a revert lands on.
LINEAGES = ("models/enigma_v2_sft2/model.pth", "models/enigma_dpo/model.pth")


def fingerprint(path: str) -> bool:
    """Print one checkpoint's load fingerprint; True when it loaded clean."""
    print(f"CHECKPOINT: {path}")
    src = Path(path)
    if not src.exists():
        print("  ABSENT: no such checkpoint")
        return False
    ck = torch.load(src, map_location="cpu", weights_only=True)
    cfg = ForgeConfig.from_dict(ck["config"])
    m = Enigma(cfg)
    res = m.load_state_dict(ck["model_state_dict"], strict=False)
    mk = sorted(m.state_dict().keys())
    print("  STEP:", ck.get("step"))
    print("  PARAMS:", sum(p.numel() for p in m.parameters()))
    print("  MODEL_KEYS:", len(mk))
    print("  KEYHASH:", hashlib.sha256("\n".join(mk).encode()).hexdigest()[:16])
    print("  MISSING:", list(res.missing_keys)[:10])
    print("  UNEXPECTED:", list(res.unexpected_keys)[:10])
    return not (res.missing_keys or res.unexpected_keys)


targets = sys.argv[1:] or list(LINEAGES)
ok = [fingerprint(p) for p in targets]
if not all(ok):
    raise SystemExit(f"{ok.count(False)} of {len(ok)} checkpoints absent or loaded dirty")
