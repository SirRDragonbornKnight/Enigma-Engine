"""Measure decode latency: prefill cost, per-token cost, and host syncs.

At 182M the decode loop is launch-bound, not compute-bound, so the number that
matters is milliseconds per token at batch 1 -- and the thing that inflates it
is host synchronization inside the loop (every .item() drains the GPU queue).

    python bench_generate.py --model models/enigma_dpo/model.pth --tokens 64
    python bench_generate.py --tiny --device cpu          # no checkpoint needed
    python bench_generate.py --model ... --count-syncs    # locate the syncs

Reports median ms/token rather than the mean: one scheduler hiccup skews a mean
and the median is what a conversation actually feels like.
"""

from __future__ import annotations

import argparse
import statistics
import time
import warnings
from pathlib import Path

import torch

from enigma_engine.core.model import Enigma
from enigma_engine.core.model_presets import ForgeConfig, get_preset

ROOT = Path(__file__).resolve().parent


def build_model(args):
    if args.tiny:
        cfg = get_preset("tiny", vocab_size=512)
        cfg.dropout = 0.0
        return Enigma(cfg).to(args.device).eval(), cfg
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    if not (isinstance(ckpt, dict) and "model_state_dict" in ckpt and "config" in ckpt):
        raise SystemExit(f"{args.model} is not an Enigma checkpoint")
    cfg = ForgeConfig.from_dict(ckpt["config"])
    model = Enigma(cfg)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    if missing or unexpected:
        print(f"  WARN: checkpoint mismatch -- {len(missing)} missing, "
              f"{len(unexpected)} unexpected keys; timings may not reflect a real model")
    return model.to(args.device).eval(), cfg


def timed_decode(model, input_ids, n_tokens, device):
    """Generate n_tokens one at a time, returning per-token wall times.

    Times each token separately so the distribution is visible; a mean alone
    hides whether the loop is steady or spiky.
    """
    per_token = []
    with torch.no_grad():
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = model.forward(input_ids, use_cache=True, start_pos=0)
        if device == "cuda":
            torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - t0) * 1000

        generated = input_ids
        for _ in range(n_tokens):
            if device == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            logits = model.forward(
                next_token, use_cache=True, start_pos=generated.shape[1] - 1
            )
            if device == "cuda":
                torch.cuda.synchronize()
            per_token.append((time.perf_counter() - t) * 1000)
    return prefill_ms, per_token


def count_syncs(model, input_ids, n_tokens, device):
    """Count host-sync warnings emitted while generating.

    torch's sync debug mode warns on every implicit device-to-host
    synchronization; each one drains the queue and costs a full launch latency.
    """
    if device != "cuda":
        return None, "sync counting needs CUDA"
    torch.cuda.set_sync_debug_mode("warn")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with torch.no_grad():
                model.generate(input_ids, max_new_tokens=n_tokens, temperature=0.0)
        hits = [w for w in caught if "synchron" in str(w.message).lower()]
        return len(hits), None
    finally:
        torch.cuda.set_sync_debug_mode("default")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(ROOT / "models" / "enigma_dpo" / "model.pth"))
    ap.add_argument("--tiny", action="store_true", help="use an untrained tiny preset instead of a checkpoint")
    ap.add_argument("--tokens", type=int, default=64, help="tokens to decode per repeat")
    ap.add_argument("--prompt-len", type=int, default=32, help="synthetic prompt length in tokens")
    ap.add_argument("--repeats", type=int, default=3, help="timed repeats after the warmup")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--count-syncs", action="store_true", help="report host syncs per generate call (CUDA only)")
    args = ap.parse_args()

    model, cfg = build_model(args)
    params = sum(p.numel() for p in model.parameters())
    print(
        f"bench: {params / 1e6:.1f}M params on {args.device} | {cfg.n_layers}L x {cfg.dim}d | "
        f"prompt {args.prompt_len} tok, decoding {args.tokens} tok x {args.repeats}",
        flush=True,
    )

    vocab = min(cfg.vocab_size, 256)
    input_ids = torch.randint(1, vocab, (1, args.prompt_len), device=args.device)

    timed_decode(model, input_ids, 8, args.device)  # warmup: allocator + any compile

    prefills, all_tokens = [], []
    for _ in range(args.repeats):
        prefill_ms, per_token = timed_decode(model, input_ids, args.tokens, args.device)
        prefills.append(prefill_ms)
        all_tokens.extend(per_token)

    ordered = sorted(all_tokens)
    median = statistics.median(ordered)
    p90 = ordered[min(len(ordered) - 1, max(0, round(len(ordered) * 0.9) - 1))]
    print(f"  prefill      : {statistics.median(prefills):8.2f} ms ({args.prompt_len} tokens)")
    print(f"  per token    : {median:8.3f} ms median | {ordered[0]:.3f} min | {p90:.3f} p90")
    print(f"  throughput   : {1000 / median:8.1f} tok/s at batch 1")
    print(f"  destination  :    0.300 -    0.600 ms/token (the serving-ladder target)")

    if args.count_syncs:
        syncs, err = count_syncs(model, input_ids, args.tokens, args.device)
        if err:
            print(f"  syncs        : skipped -- {err}")
        else:
            print(f"  syncs        : {syncs} over {args.tokens} tokens "
                  f"({syncs / max(1, args.tokens):.2f} per token, measured on "
                  f"model.generate -- the serve path, not the timed loop above)")


if __name__ == "__main__":
    main()
