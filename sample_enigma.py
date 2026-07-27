#!/usr/bin/env python
"""Generate text from a pretrained Enigma checkpoint (the from-scratch model).

python sample_enigma.py                                   # uses the completed pretrain model
python sample_enigma.py --ckpt models/enigma_pretrain_large/model.pth --prompts "The" "In 1850"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "models" / "enigma_pretrain_large" / "model.pth"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-new", type=int, default=50)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--prompts", nargs="+", default=None)
    args = ap.parse_args()

    from enigma_engine.core.model import Enigma
    from enigma_engine.core.model_presets import ForgeConfig
    from enigma_engine.core.tokenizer import get_tokenizer

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    tok = get_tokenizer("bpe")
    eos = getattr(tok, "eos_token_id", 2)
    bos = getattr(tok, "bos_token_id", 1)

    # weights_only=True, explicit and measured (2026-07-26 audit): the real
    # checkpoints load under it on torch 2.10, where it is also the default
    # this script was already running with. False would be a downgrade to
    # executing whatever pickle a foreign .pth carries.
    ck = torch.load(args.ckpt, map_location=device, weights_only=True)
    config = ForgeConfig.from_dict(ck["config"])
    model = Enigma(config)
    model.load_state_dict(ck["model_state_dict"], strict=False)
    model.to(device).eval()
    print(
        f"loaded {args.ckpt}\n  step {ck.get('step', '?')} | "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params | device {device}",
        flush=True,
    )

    prompts = args.prompts or [
        "The history of",
        "In 1066, ",
        "Water is made of",
        "The capital of France is",
        "She opened the door and ",
    ]
    for p in prompts:
        ids_list = tok.encode(p)
        if ids_list and ids_list[-1] == eos:  # drop trailing EOS so it keeps going
            ids_list = ids_list[:-1]
        if not ids_list or ids_list[0] != bos:
            ids_list = [bos] + ids_list
        ids = torch.tensor([ids_list], device=device)
        # Cached decode: prefill the prompt once, then one new token per step with
        # O(1) KV-cache writes (enigma_engine.core.kv_cache). Verified equivalent to
        # a full no-cache recompute, logit-for-logit, by tests/test_model_kv_cache.py
        # — so this is both faster and exact.
        with torch.no_grad():
            gen = model.generate(
                ids,
                max_new_tokens=args.max_new,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=1.1,
                stop_tokens=[eos],
            )
        text = tok.decode(gen[0].tolist())
        print(f"\n>>> {p!r}\n    {text!r}", flush=True)


if __name__ == "__main__":
    main()
