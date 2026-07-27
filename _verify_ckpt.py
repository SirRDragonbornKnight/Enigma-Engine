"""Scratch: prove the from-scratch checkpoint still loads bit-identically.

Load-only (CPU, no optimizer step, no data) — NOT training. Run before and
after any model.py / model_components.py cleanup; PARAMS and KEYHASH must not
change and MISSING/UNEXPECTED must stay empty.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import hashlib
import torch
from enigma_engine.core.model import Enigma, ForgeConfig

ck = torch.load("models/enigma_pretrain_large/latest.pth", map_location="cpu", weights_only=True)
cfg = ForgeConfig.from_dict(ck["config"])
m = Enigma(cfg)
res = m.load_state_dict(ck["model_state_dict"], strict=False)
mk = sorted(m.state_dict().keys())
print("STEP:", ck.get("step"))
print("PARAMS:", sum(p.numel() for p in m.parameters()))
print("MODEL_KEYS:", len(mk))
print("KEYHASH:", hashlib.sha256("\n".join(mk).encode()).hexdigest()[:16])
print("MISSING:", list(res.missing_keys)[:10])
print("UNEXPECTED:", list(res.unexpected_keys)[:10])
