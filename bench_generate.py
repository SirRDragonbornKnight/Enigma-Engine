"""Measure decode latency: prefill cost, per-token cost, and host syncs.

At 182M the decode loop is launch-bound, not compute-bound, so the number that
matters is milliseconds per token at batch 1 -- and the thing that inflates it
is host synchronization inside the loop (every .item() drains the GPU queue).

    python bench_generate.py --model models/enigma_v2_sft2/model.pth --tokens 64
    python bench_generate.py --tiny --device cpu          # no checkpoint needed
    python bench_generate.py --model ... --count-syncs    # locate the syncs

Reports median ms/token rather than the mean: one scheduler hiccup skews a mean
and the median is what a conversation actually feels like.
"""

from __future__ import annotations

import argparse
import contextlib
import statistics
import time
import warnings
from pathlib import Path

import torch

from enigma_engine.core.model import Enigma
from enigma_engine.core.model_presets import ForgeConfig, get_preset

# ONE device rule for the bench and the server. serve_enigma is import-safe
# (boot() owns startup), so this costs an import, not a boot -- and a bench
# that resolved --device its own way is exactly how a CPU baseline ends up
# measured on the GPU.
from quantize_serving_ckpt import load_checkpoint, load_serving_ckpt
from serve_enigma import _resolve_device

ROOT = Path(__file__).resolve().parent


def build_model(args):
    if args.tiny:
        cfg = get_preset("tiny", vocab_size=512)
        cfg.dropout = 0.0
        return Enigma(cfg).to(args.device).eval(), cfg
    ckpt = load_checkpoint(args.model)
    if not (isinstance(ckpt, dict) and "model_state_dict" in ckpt and "config" in ckpt):
        raise SystemExit(f"{args.model} is not an Enigma checkpoint")
    if (ckpt.get("meta") or {}).get("quant"):
        # An int8 number is only honest if it came through the loader that
        # ASSERTS the weights stayed quantized -- this bench's tolerant
        # strict=False path would happily time a silently dequantized model.
        model, ckpt = load_serving_ckpt(args.model)
        return model.to(args.device).eval(), ForgeConfig.from_dict(ckpt["config"])
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


def timed_serve_path(model, input_ids, n_tokens, device, repeats):
    """Per-token wall time on the path serve actually runs: generate_stream,
    serve's live sampler settings, bf16 autocast on CUDA. The default
    timed_decode loop is greedy argmax over a bare forward in fp32 -- every
    sampler cost (repetition penalty, top-k/p, min-p) is absent from it, so
    its number is a floor for serving rather than a measurement of it."""
    # serve_enigma's ChatReq defaults.
    kw = dict(temperature=0.3, top_k=50, top_p=0.9, min_p=0.05, repetition_penalty=1.1)
    # Same gate serve uses: bf16 only where supported, else the receipt would
    # measure a dtype the server never runs.
    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else contextlib.nullcontext()
    per_call = []
    with torch.no_grad(), autocast:
        list(model.generate_stream(input_ids, max_new_tokens=8, **kw))  # warmup
        for _ in range(repeats):
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            produced = sum(1 for _ in model.generate_stream(input_ids, max_new_tokens=n_tokens, **kw))
            if device == "cuda":
                torch.cuda.synchronize()
            per_call.append((time.perf_counter() - start) * 1000 / max(1, produced))
    return per_call


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # Follows ADOPTION: a serving benchmark must measure what is served.
    ap.add_argument("--model", default=str(ROOT / "models" / "enigma_v2_sft2" / "model.pth"))
    ap.add_argument("--tiny", action="store_true", help="use an untrained tiny preset instead of a checkpoint")
    ap.add_argument("--tokens", type=int, default=64, help="tokens to decode per repeat")
    ap.add_argument("--prompt-len", type=int, default=32, help="synthetic prompt length in tokens")
    ap.add_argument("--repeats", type=int, default=3, help="timed repeats after the warmup")
    ap.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="auto = cuda when available; --device cuda without CUDA REFUSES rather than "
        "quietly benching the CPU and calling it a GPU number",
    )
    ap.add_argument(
        "--threads",
        type=int,
        default=10,
        help="torch CPU threads (default 10 -- the cap that keeps a CPU bench from starving "
        "the desktop while Chrome Remote Desktop is connected)",
    )
    ap.add_argument("--count-syncs", action="store_true", help="report host syncs per generate call (CUDA only)")
    ap.add_argument(
        "--serve-path",
        action="store_true",
        help="time generate_stream with serve's live sampler under bf16 autocast -- what serving "
        "actually runs. The default loop times bare forward + argmax in fp32, which is a floor, "
        "not a serve receipt.",
    )
    args = ap.parse_args()
    # Before build_model: the state-dict copies parallelize too, so a thread cap
    # set after it would leave the loading half uncapped.
    torch.set_num_threads(args.threads)
    args.device = _resolve_device(args.device)

    model, cfg = build_model(args)
    params = sum(p.numel() for p in model.parameters())
    print(
        f"bench: {params / 1e6:.1f}M params on {args.device} "
        f"(torch.cuda.is_available()={torch.cuda.is_available()}, "
        f"{torch.get_num_threads()} cpu threads) | {cfg.n_layers}L x {cfg.dim}d | "
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
    print(f"    ^ greedy argmax over a bare forward, fp32: a FLOOR, not serving")
    print(f"  destination  :    0.300 -    0.600 ms/token (the serving-ladder target)")

    if args.serve_path:
        serve_times = timed_serve_path(model, input_ids, args.tokens, args.device, args.repeats)
        serve_median = statistics.median(serve_times)
        print(f"  serve path   : {serve_median:8.3f} ms median | {1000 / serve_median:.1f} tok/s "
              f"(generate_stream + sampler"
              f"{' + bf16 autocast' if args.device == 'cuda' else ''})")

    if args.count_syncs:
        syncs, err = count_syncs(model, input_ids, args.tokens, args.device)
        if err:
            print(f"  syncs        : skipped -- {err}")
        else:
            print(f"  syncs        : {syncs} over {args.tokens} tokens "
                  f"({syncs / max(1, args.tokens):.2f} per token, measured on "
                  f"model.generate at temperature 0 -- NOT the serve path, which "
                  f"runs generate_stream and samples)")


if __name__ == "__main__":
    main()
