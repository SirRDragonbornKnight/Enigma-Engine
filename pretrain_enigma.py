#!/usr/bin/env python
"""Pretrain the REAL Enigma — our own architecture, our own weights — on the
pre-tokenized ``data/pretrain/tokens.bin`` corpus (56.6B tokens, vocab 4718).

This is the genuine own-brained model: a from-scratch transformer that learns
language from the data, NOT a wrapper around someone else's model. It loads
the SAME AdvancedBPETokenizer that produced ``tokens.bin`` and streams the
tokens via memmap (nanoGPT-style). The instruct pass lives in
``finetune_enigma.py``; both passes share the optimizer/schedule arsenal in
``enigma_engine.core.optim``.

  python pretrain_enigma.py --sanity                  # 1-step smoke test, then exit
  python pretrain_enigma.py --size base --tokens 2e9  # the real run (~GPT-2-small)
  python pretrain_enigma.py --resume models/enigma_pretrain_base/latest.pth

Checkpoints (model_state_dict + config + step + optimizer + schedule) land in
``models/enigma_pretrain_<size>/latest.pth`` every --save-every steps, with the
previous generation rotated to ``prev.pth`` (and optional frozen snapshots via
--archive-every). Resumes restore the recorded schedule — pass
--override-schedule to deliberately change it. The final model is written as ``model.pth`` in the standard
{model_state_dict, config} format (plus step/optimizer/schedule) the rest of the stack already loads.

Future-run knobs (defaults reproduce the live run exactly; both are schedule-locked
into checkpoints): ``--optimizer muon`` (Moonlight Muon for the 2-D body, aux AdamW)
and ``--schedule wsd`` (warmup-stable-decay-to-zero — continuation/multi-epoch
friendly). Never switch either on an existing lineage.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch

from enigma_engine.core.optim import build_optimizer, get_lr

try:  # Windows consoles default to cp1252 and crash on unicode sample text.
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent
TOKENS_BIN = ROOT / "data" / "pretrain" / "tokens.bin"
TOKENS_META = ROOT / "data" / "pretrain" / "tokens.json"
HEADER_BYTES = 256  # ETOK reserved header (see pretokenize_data.py)


def _pin_sdpa(backend: str) -> None:
    """Enable exactly ONE F.scaled_dot_product_attention backend.

    Strict on purpose: with a single backend enabled, an unsupported shape
    raises instead of silently dispatching elsewhere -- so the run KNOWS
    which kernel computed it. PyTorch wheels have shipped with cuDNN
    silently unselected; "pin + log" is the v2 recipe's answer to that.
    """
    torch.backends.cuda.enable_flash_sdp(backend == "flash")
    torch.backends.cuda.enable_mem_efficient_sdp(backend == "efficient")
    torch.backends.cuda.enable_cudnn_sdp(backend == "cudnn")
    torch.backends.cuda.enable_math_sdp(backend == "math")


def _sdpa_preflight(model, get_batch, amp_dtype, backend: str) -> None:
    """One fwd/bwd under the pinned backend vs the MATH reference, before any
    long run: a non-finite-gradient guard plus gradient agreement. SDPA wheels
    have a documented history of silent backward NaNs on specific shapes --
    refuse to train on a kernel whose gradients disagree with the reference.

    amp_dtype None runs the check in fp32: an fp16 backward without the loss
    scaler overflows routinely, which would false-refuse a healthy kernel.
    """
    X, Y = get_batch("train")
    X, Y = X[:2], Y[:2]  # MATH materializes full attention scores; keep it small

    def _grads() -> torch.Tensor:
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype or torch.bfloat16, enabled=amp_dtype is not None):
            _, loss = model(X, targets=Y)
        loss.backward()
        flat = torch.cat([p.grad.detach().float().reshape(-1) for p in model.parameters() if p.grad is not None])
        model.zero_grad(set_to_none=True)
        return flat

    g_pin = _grads()
    if not torch.isfinite(g_pin).all():
        raise SystemExit(f"sdpa preflight: NON-FINITE gradients under backend '{backend}' -- refusing to train")
    _pin_sdpa("math")
    try:
        g_ref = _grads()
    finally:
        _pin_sdpa(backend)
    if not torch.isfinite(g_ref).all():
        raise SystemExit(
            "sdpa preflight: NON-FINITE gradients under the MATH reference -- "
            "the model itself is unstable on this batch; refusing to train"
        )
    cos = (g_pin @ g_ref / (g_pin.norm() * g_ref.norm() + 1e-12)).item()
    if cos < 0.98:
        raise SystemExit(
            f"sdpa preflight: backend '{backend}' gradients disagree with the MATH "
            f"reference (cosine {cos:.6f} < 0.98) -- refusing to train on a kernel "
            "with wrong gradients; try a different --sdpa-backend"
        )
    print(f"sdpa preflight: backend '{backend}' vs math grad cosine {cos:.6f} -- OK", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="base", help="ForgeConfig preset (tiny..xl)")
    ap.add_argument("--block", type=int, default=1024, help="sequence length")
    ap.add_argument("--micro-batch", type=int, default=12, help="sequences per forward")
    ap.add_argument("--grad-accum", type=int, default=16, help="micro-batches per optimizer step")
    ap.add_argument("--tokens", type=float, default=2e9, help="target training tokens")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument(
        "--dropout", type=float, default=0.0, help="dropout (0.0 for single-epoch pretraining; presets default to 0.1)"
    )
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument(
        "--optimizer",
        choices=["adamw", "muon"],
        default="adamw",
        help="adamw = the live run's exact path. muon (Moonlight variant) "
        "is for FUTURE runs -- never switch mid-lineage",
    )
    ap.add_argument(
        "--schedule",
        choices=["cosine", "wsd", "wsd_sqrt"],
        default="cosine",
        help="cosine = the live run's schedule. wsd = warmup-stable-decay "
        "(linear decay-to-zero over the last --wsd-decay-frac) -- "
        "continuation-friendly, for FUTURE runs. wsd_sqrt = the IMU-1 shape "
        "(1-sqrt(t) decay to a 0.01x floor), the v2 recipe's candidate",
    )
    ap.add_argument(
        "--wsd-decay-frac", type=float, default=0.10, help="fraction of total steps spent in the WSD decay phase"
    )
    ap.add_argument("--save-every", type=int, default=250, help="steps between checkpoints")
    ap.add_argument("--eval-every", type=int, default=250, help="steps between val-loss checks")
    ap.add_argument("--eval-iters", type=int, default=40)
    ap.add_argument("--val-tokens", type=int, default=10_000_000, help="tail tokens held out for val")
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument(
        "--init-from",
        default=None,
        help="warm-start a NEW run from a checkpoint's WEIGHTS: rebuild the "
        "architecture from the checkpoint and load its weights, but start at "
        "step 0 with a fresh optimizer and a fresh schedule/warmup (CLI args, "
        "not the checkpoint's). For length extension / continued pretraining "
        "at a new --block. Mutually exclusive with --resume.",
    )
    ap.add_argument(
        "--override-schedule",
        action="store_true",
        help="on resume, let CLI schedule args override the checkpoint's recorded schedule",
    )
    ap.add_argument(
        "--archive-every",
        type=int,
        default=0,
        help="also keep a frozen step_NNNNNN.pth checkpoint every N steps (0 = off)",
    )
    ap.add_argument(
        "--val-general-end",
        type=int,
        default=56_575_624_692,
        help="end offset of the pre-anime-append corpus; a second [val-gen] eval window "
        "is carved just below it (0 = disable)",
    )
    ap.add_argument("--no-grad-ckpt", action="store_true", help="disable gradient checkpointing")
    ap.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="torch.compile the model (~1.5-2x; --no-compile for eager / if Triton is absent)",
    )
    ap.add_argument(
        "--tokens-bin",
        default=None,
        help="alternate ETOK corpus for continued-pretrain passes (sidecar "
        "meta is <name>.json beside it). Default: the live run's "
        "data/pretrain/tokens.bin -- unchanged.",
    )
    ap.add_argument("--sanity", action="store_true", help="one fwd/bwd step then exit")
    ap.add_argument(
        "--throttle-ms",
        type=float,
        default=0.0,
        help="sleep N ms after each micro-batch to yield the GPU (e.g. while gaming); 0 = full speed",
    )
    ap.add_argument(
        "--sdpa-backend",
        default="auto",
        choices=["auto", "cudnn", "flash", "efficient", "math"],
        help="pin ONE F.sdpa backend (strict: unsupported shapes error instead of silently "
        "falling back; schedule-recorded so a bare --resume restores the run's own pin); "
        "auto = torch default dispatch, unchanged behavior",
    )
    ap.add_argument(
        "--skip-sdpa-preflight",
        action="store_true",
        help="skip the pinned-backend non-finite/grad-agreement preflight",
    )
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # --resume (exact continuation) and --init-from (warm-start a NEW run from
    # a checkpoint's weights) both load a checkpoint EARLY, but only resume
    # restores the recorded schedule/optimizer/step. They are mutually
    # exclusive.
    if args.resume and args.init_from:
        raise SystemExit("--resume and --init-from are mutually exclusive")
    warm_start = bool(args.init_from)
    ckpt_arg = args.resume or args.init_from
    # Warm-start guards, checked before loading anything heavy: a warm-start is
    # a NEW run (immutable-lineage rule), so it must have an explicit --out that
    # is not the source checkpoint's own directory (which it would clobber).
    if warm_start:
        if not args.out:
            raise SystemExit("--init-from requires an explicit --out <new dir> (a warm-start is a new run)")
        if Path(args.out).resolve() == Path(ckpt_arg).resolve().parent:
            raise SystemExit(
                f"--init-from would write into the source model's directory ({args.out}); "
                "pass --out <new dir> so the checkpoint you warm-started from is not overwritten"
            )
    ck = None
    if ckpt_arg:
        rp = Path(ckpt_arg)
        if not rp.exists() and rp.name == "latest.pth" and (rp.parent / "prev.pth").exists():
            print(f"load: {rp} missing -> falling back to {rp.parent / 'prev.pth'}", flush=True)
            rp = rp.parent / "prev.pth"
        if not rp.exists():
            raise SystemExit(
                f"checkpoint {ckpt_arg} not found (no prev.pth fallback either) -- "
                f"refusing to silently start a fresh run"
            )
        try:
            ck = torch.load(rp, map_location=device)
        except Exception as exc:
            prev = rp.parent / "prev.pth"
            if rp.name == "latest.pth" and prev.exists():
                # corrupt-but-present latest (e.g. power loss mid-write):
                # fall back to the rotated previous generation
                print(f"load: {rp} unreadable ({exc}) -> falling back to {prev}", flush=True)
                rp = prev
                ck = torch.load(rp, map_location=device)
            else:
                raise
        # Warm-start keeps the CLI schedule (fresh warmup at the new block);
        # only exact resume restores the checkpoint's recorded schedule.
        saved_sched = ck.get("schedule") if not warm_start else None
        if saved_sched:
            diffs = {k: (v, getattr(args, k)) for k, v in saved_sched.items() if getattr(args, k, None) != v}
            if args.override_schedule:
                for k, (ck_v, cli_v) in diffs.items():
                    print(
                        f"resume: schedule[{k}] CLI {cli_v} OVERRIDES checkpoint {ck_v} (--override-schedule)",
                        flush=True,
                    )
            else:
                for k, v in saved_sched.items():
                    setattr(args, k, v)
                for k, (ck_v, cli_v) in diffs.items():
                    print(f"resume: schedule[{k}] = {ck_v} from checkpoint (CLI {cli_v} ignored)", flush=True)
        elif not warm_start:
            print(
                "resume: checkpoint predates schedule recording -- trusting CLI args (this run will record them)",
                flush=True,
            )

    # Corpus resolution runs AFTER the resume/schedule restore above, so a bare
    # `--resume` recovers the run's OWN --tokens-bin from the checkpoint schedule
    # (final audit 2026-07-16 M1). Without this, resuming a facts continued-pretrain
    # run without re-passing the flag silently finished it on the default 56.6B
    # corpus. NOTE (comment corrected 2026-07-17): on resume the restore WINS over
    # a DIFFERING explicit --tokens-bin -- like every schedule key, it prints
    # "CLI ... ignored". To retarget a resumed run's corpus on purpose, pass
    # --override-schedule. Checkpoints written before this fix predate tokens_bin
    # in the schedule and must still re-pass the flag on resume.
    global TOKENS_BIN, TOKENS_META
    if args.tokens_bin:
        TOKENS_BIN = Path(args.tokens_bin)
        TOKENS_META = TOKENS_BIN.with_suffix(".json")

    if not TOKENS_BIN.exists():
        raise SystemExit(f"missing corpus: {TOKENS_BIN}")
    meta = json.loads(TOKENS_META.read_text(encoding="utf-8"))
    # uint32 is the v1 lineage; uint16 is the v2 corpus format (vocab 16,384
    # fits 2-byte ids, halving 86 GB to ~43 -- TOKENIZER_V2_SPEC). The dtype
    # comes from the sidecar and is CROSS-CHECKED against the bin's own bpt
    # header below, so a stale sidecar cannot mis-type the stream.
    dtype_name = meta.get("dtype", "uint32")
    if dtype_name not in ("uint32", "uint16"):
        raise SystemExit(f"expected uint32/uint16 tokens, got {dtype_name}")
    np_dtype = np.uint32 if dtype_name == "uint32" else np.uint16
    itemsize = 4 if dtype_name == "uint32" else 2
    vocab_meta = meta["vocab_size"]
    if itemsize == 2 and vocab_meta > 65536:
        raise SystemExit(f"uint16 corpus with vocab {vocab_meta} > 65536 -- corrupt metadata")
    print(
        f"corpus: {meta['total_tokens']:,} tokens, vocab {vocab_meta}, {meta['file_size_gb']} GB ({meta['tokenizer']})",
        flush=True,
    )

    # Vocab is authoritative from the corpus metadata: the model trains on the
    # raw token IDs, so it doesn't need the tokenizer at all. We still try to
    # load the exact one that produced tokens.bin (AdvancedBPETokenizer, via
    # 'bpe' — never 'auto', which would grab tiktoken) for readable samples,
    # but training proceeds fine without it.
    vocab_size = vocab_meta
    try:
        from enigma_engine.core.tokenizer import get_tokenizer

        tok = get_tokenizer("bpe")
        if getattr(tok, "vocab_size", None) != vocab_size:
            print(
                f"  WARN: tokenizer vocab {getattr(tok, 'vocab_size', '?')} != "
                f"corpus vocab {vocab_size}; using corpus vocab",
                flush=True,
            )
    except Exception as exc:
        tok = None
        print(f"  (tokenizer unavailable -- training on raw IDs: {exc})", flush=True)

    # Validate the ETOK header before trusting the stream. Without this a stale
    # or truncated tokens.bin would silently memmap a wrong token count (numpy
    # just drops trailing bytes), training on a misaligned/short corpus while
    # reporting the JSON's numbers. Fail loudly instead.
    file_bytes = TOKENS_BIN.stat().st_size
    with open(TOKENS_BIN, "rb") as _fh:
        magic, _ver, bpt, hdr_total, _hdr_vocab, _eos = struct.unpack("<4sIIQII", _fh.read(28))
    if magic != b"ETOK":
        raise SystemExit(f"{TOKENS_BIN} is not an ETOK corpus (magic {magic!r}) -- refusing to train")
    # tokens.json is written AFTER the bin (pretokenize_data.py); a crash in
    # between leaves a new bin + stale JSON. The JSON's vocab sizes the model,
    # so a mismatch means embedding/CE indices can run out of bounds (or waste
    # rows). Trust the bin's own header over the sidecar and fail loudly.
    if _hdr_vocab and _hdr_vocab != vocab_meta:
        raise SystemExit(
            f"{TOKENS_BIN} header says vocab {_hdr_vocab} but tokens.json says {vocab_meta} "
            f"(stale sidecar?) -- refusing to train"
        )
    stream_bytes = file_bytes - HEADER_BYTES
    if bpt != itemsize or stream_bytes % itemsize != 0:
        raise SystemExit(
            f"{TOKENS_BIN} stream disagrees with {dtype_name} (header bpt={bpt}, expected {itemsize}; "
            f"{stream_bytes} bytes after header) -- stale sidecar or truncated write"
        )
    if hdr_total != stream_bytes // itemsize:
        raise SystemExit(
            f"{TOKENS_BIN} header claims {hdr_total:,} tokens but the file holds "
            f"{stream_bytes // itemsize:,} (stale header or truncated write) -- refusing to train"
        )

    # memmap the token stream after the 256-byte header.
    data = np.memmap(TOKENS_BIN, dtype=np_dtype, mode="r", offset=HEADER_BYTES)
    n = len(data)
    val_n = min(args.val_tokens, n // 100)
    train_end = n - val_n
    print(f"memmapped {n:,} tokens  (train {train_end:,} / val {val_n:,})", flush=True)

    # id-0 audit (v2 corpora only): the CE loss ignores index 0 (<pad>),
    # so any id-0 token in the stream silently drops its loss target. v2
    # carves EVERY special literal, and web text does contain the literal
    # string "<pad>" (2 hits measured in the 23.7B build, 2026-07-20).
    # Microscopic is fine; a broken tokenizer writing real pads is not --
    # scan once at boot (~1 min for 47 GB) and refuse past 0.01%.
    if meta.get("pretokenizer") == "v2":
        zeros = 0
        _CH = 1 << 28
        for _i in range(0, n, _CH):
            zeros += int((np.asarray(data[_i : _i + _CH]) == 0).sum())
        if zeros:
            frac = zeros / n
            print(f"id-0 (<pad>) tokens in stream: {zeros:,} ({frac:.2e}) -- loss silently ignored there", flush=True)
            if frac > 1e-4:
                raise SystemExit(
                    f"{zeros:,} id-0 tokens ({frac:.2%}) -- a real pad stream, not stray literals; "
                    f"refusing to train with ignore_index=0 dropping that much signal"
                )

    # [val-gen]: second eval window at the tail of the ORIGINAL corpus. The
    # 2026-06-07 anime append landed at the END of tokens.bin, so the held-out
    # tail above ([val]) became 100% anime-domain. This window restores a
    # general-domain signal. It was truly held out only until the append
    # (~16% train-sampled between then and this fence landing), so it reads
    # slightly optimistic; the fence in get_batch stops further leakage.
    # The offset must lie INSIDE this corpus. The default (56.6B, the v1
    # pre-anime-append fence) exceeds the ~23.7B v2 corpus entirely; the old
    # clamp to train_end would silently evaluate an in-distribution train
    # tail while fencing it from sampling (Arc 3 audit, MED-1).
    if args.val_general_end > train_end:
        print(
            f"val-gen: offset {args.val_general_end:,} lies beyond train_end {train_end:,} "
            f"(this corpus is not the one the default was tuned for) -- window DISABLED",
            flush=True,
        )
    vg_end = min(args.val_general_end, train_end)
    vg_lo = max(0, vg_end - val_n)
    use_val_gen = 0 < args.val_general_end <= train_end and (vg_end - vg_lo) > args.block + 1
    if use_val_gen:
        print(f"val-gen window: [{vg_lo:,}, {vg_end:,}) -- pre-append tail, fenced from train sampling", flush=True)

    block = args.block

    def get_batch(split: str):
        if split == "train":
            lo, hi = 0, train_end
        elif split == "val":
            lo, hi = train_end, n
        else:  # "val_gen" — pre-append general-domain window
            lo, hi = vg_lo, vg_end
        ix = np.random.randint(lo, hi - block - 1, size=args.micro_batch, dtype=np.int64)
        if split == "train" and use_val_gen:
            # Fence: re-draw the rare index (~0.02% chance) whose sample
            # [i, i+block] would overlap the val-gen window, keeping it
            # held out from here on.
            for j in range(len(ix)):
                while vg_lo - block <= ix[j] < vg_end:
                    # dtype is load-bearing: legacy randint defaults to C-long
                    # (int32 on Windows) and hi is ~56.7e9.
                    ix[j] = np.random.randint(lo, hi - block - 1, dtype=np.int64)
        x = np.stack([np.asarray(data[i : i + block], dtype=np.int64) for i in ix])
        y = np.stack([np.asarray(data[i + 1 : i + 1 + block], dtype=np.int64) for i in ix])
        X = torch.from_numpy(x).to(device, non_blocking=True)
        Y = torch.from_numpy(y).to(device, non_blocking=True)
        return X, Y

    # Build the model from a preset, sized to the corpus vocab.
    from enigma_engine.core.model import Enigma
    from enigma_engine.core.model_presets import ForgeConfig, get_preset

    # On resume, rebuild config from the CHECKPOINT (exact architecture) rather than the
    # preset. Otherwise a flag mismatch builds a differently-shaped model and
    # load_state_dict(strict=False) silently leaves the mismatched tensors at random
    # init — a silent corruption. Trust the checkpoint. (ck was loaded early, above,
    # so its recorded schedule could win before schedule-derived values were computed.)
    if ck is not None:
        config = ForgeConfig.from_dict(ck["config"])
        print(f"{'init-from' if warm_start else 'resume'}: config rebuilt from checkpoint ({ckpt_arg})", flush=True)
    else:
        config = get_preset(args.size, vocab_size=vocab_size)
    config.dropout = args.dropout  # 0.0 for single-epoch pretraining (preset default 0.1 undertrains)
    if block > config.max_seq_len:
        config.max_seq_len = block
    model = Enigma(config)
    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable()
    else:
        # Config default is use_gradient_checkpointing=True, so the flag
        # must actively DISABLE. Without this else, --no-grad-ckpt was a
        # no-op: every run to date trained with checkpointing ON (paying
        # the ~30-40% recompute tax) while the log printed ckpt=False
        # (2026-07-20 trainer audit; the extend_length.ps1 probe numbers
        # were measured with it on, so those ETAs are conservative).
        model.gradient_checkpointing_disable()
    model.to(device)
    raw_model = model  # the real nn.Module; checkpoints/optimizer bind here even when compiled
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"Enigma '{args.size}': {n_params / 1e6:.1f}M params, dim={config.dim} "
        f"layers={config.n_layers} heads={config.n_heads} block={block}",
        flush=True,
    )

    optim = build_optimizer(raw_model, args.optimizer, args.lr, args.weight_decay)

    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    use_scaler = device == "cuda" and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    tokens_per_step = args.micro_batch * args.grad_accum * block
    total_steps = max(1, int(args.tokens / tokens_per_step))

    # Everything that defines the run's MATH, recorded into every checkpoint so a
    # resume restores it exactly (see the resume block above). Operational knobs
    # (--save-every/--eval-every/--compile/--throttle-ms/...) stay CLI-controlled.
    schedule = {
        k: getattr(args, k)
        for k in (
            "tokens",
            "lr",
            "warmup",
            "micro_batch",
            "grad_accum",
            "block",
            "dropout",
            "val_tokens",
            "weight_decay",
            "grad_clip",
            "val_general_end",
            "optimizer",
            "schedule",
            "wsd_decay_frac",
            "tokens_bin",
            "sdpa_backend",
        )
    }

    if args.out:
        out = Path(args.out)
    elif ck is not None and not warm_start:
        # A resume writes back to the directory it resumed FROM, not the
        # size-derived default. Otherwise a bare `--resume <dir>/latest.pth`
        # of a non-default lineage (e.g. the block-2048 warm-start run in
        # models/enigma_pretrain_2048) resolves out to enigma_pretrain_base
        # and silently rotates a DIFFERENT lineage's checkpoints. rp is the
        # resolved checkpoint path (post prev.pth fallback); its parent is the
        # run's own directory.
        out = rp.parent
    else:
        out = ROOT / "models" / f"enigma_pretrain_{args.size}"
    out.mkdir(parents=True, exist_ok=True)

    start_step = 0
    if ck is not None:
        missing, unexpected = raw_model.load_state_dict(ck["model_state_dict"], strict=False)
        # config came from the checkpoint, so any real mismatch signals corruption —
        # hard-fail rather than silently train half-random weights. (freqs_cis / causal
        # mask are non-persistent buffers recomputed at build time; ignore those.)
        real_missing = [k for k in missing if "freqs_cis" not in k and "causal_mask" not in k]
        if unexpected or real_missing:
            raise SystemExit(
                f"{'init-from' if warm_start else 'resume'} arch mismatch -- refusing to corrupt: "
                f"missing={real_missing[:5]} unexpected={unexpected[:5]}"
            )
        if warm_start:
            # Fresh optimizer moments + fresh schedule; start_step stays 0.
            # A short warmup at the new --block re-stabilizes the converged
            # weights on the new sequence length.
            print(
                f"init-from {args.init_from}: loaded weights, starting a fresh run at step 0 "
                f"(block {block}, fresh {args.schedule} schedule, warmup {args.warmup})",
                flush=True,
            )
        else:
            if "optimizer" in ck:
                try:
                    optim.load_state_dict(ck["optimizer"])
                except Exception as exc:
                    raise SystemExit(
                        f"resume: checkpoint optimizer state does not fit --optimizer "
                        f"{args.optimizer} ({exc}) -- the run was saved with a different "
                        f"optimizer; refusing to continue with reset moments"
                    ) from None
            # ck["step"] is a COMPLETED step (saved after its optimizer update);
            # resume at the next one instead of training it twice.
            start_step = int(ck.get("step", -1)) + 1
            print(f"resumed from {args.resume} after step {start_step - 1} (next: {start_step})", flush=True)

    # SDPA backend pin (v2 recipe): applied AFTER the resume/schedule restore so
    # a bare --resume runs under the pin its own lineage recorded, and BEFORE the
    # first forward pass anywhere below (preflight/compile/sanity/loop).
    if device == "cuda":
        if args.sdpa_backend != "auto":
            _pin_sdpa(args.sdpa_backend)
            print(f"sdpa: pinned to {args.sdpa_backend} (strict, unsupported shapes error out)", flush=True)
        else:
            print(
                "sdpa: auto dispatch (flash={} mem_efficient={} cudnn={} math={})".format(
                    torch.backends.cuda.flash_sdp_enabled(),
                    torch.backends.cuda.mem_efficient_sdp_enabled(),
                    torch.backends.cuda.cudnn_sdp_enabled(),
                    torch.backends.cuda.math_sdp_enabled(),
                ),
                flush=True,
            )
        if args.sdpa_backend != "auto" and not args.skip_sdpa_preflight:
            _sdpa_preflight(raw_model, get_batch, amp_dtype if use_bf16 else None, args.sdpa_backend)

    # torch.compile after any resume-load so weights land in raw_model first. The
    # compiled wrapper is used only for fwd/bwd; save/load/optimizer stay on
    # raw_model so the `_orig_mod.` prefix never leaks into checkpoints. Eager
    # fallback keeps the run alive where inductor/Triton is unavailable (Windows).
    if args.compile and device == "cuda":
        try:
            compiled = torch.compile(raw_model)
            # torch.compile traces lazily on first call, so a missing-Triton /
            # inductor failure would otherwise crash mid-run. Force the compile
            # NOW on the real training shape (same graph the loop uses -> no later
            # recompile); on any failure fall back to eager. Throwaway grads cleared.
            _x = torch.zeros((args.micro_batch, block), dtype=torch.long, device=device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                _, _loss = compiled(_x, targets=_x)
            _loss.backward()
            optim.zero_grad(set_to_none=True)
            model = compiled
            print("torch.compile: enabled", flush=True)
        except Exception as exc:
            optim.zero_grad(set_to_none=True)
            model = raw_model
            print(f"torch.compile: unavailable -> eager ({str(exc).splitlines()[0][:140]})", flush=True)

    def save(tag: str, step: int):
        from enigma_engine.core.safe_save import atomic_torch_save

        # latest.pth keeps one previous generation as prev.pth (rotated atomically
        # inside atomic_torch_save AFTER the new file is fully written), so one bad
        # save can never cost more than --save-every steps.
        rotate = (out / "prev.pth") if tag == "latest.pth" else None
        atomic_torch_save(
            {
                "model_state_dict": raw_model.state_dict(),
                "config": config.to_dict(),
                "step": step,
                "optimizer": optim.state_dict(),
                "schedule": schedule,
            },
            str(out / tag),
            rotate_to=rotate,
        )
        (out / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")

    @torch.no_grad()
    def estimate_val(split: str = "val") -> float:
        raw_model.eval()
        losses = []
        for _ in range(args.eval_iters):
            X, Y = get_batch(split)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(device == "cuda")):
                _, loss = model(X, targets=Y)
            losses.append(loss.item())
        raw_model.train()
        return sum(losses) / max(1, len(losses))

    # --- sanity: one fwd/bwd, report loss vs random baseline, exit ----------
    if args.sanity:
        raw_model.train()
        X, Y = get_batch("train")
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(device == "cuda")):
            _, loss = model(X, targets=Y)
        loss.backward()
        base = math.log(vocab_meta)
        print(
            f"[sanity] batch={tuple(X.shape)} loss={loss.item():.4f} (random baseline ln(V)={base:.3f}) -- pipeline OK",
            flush=True,
        )
        return

    print(
        f"training: target {args.tokens / 1e9:.2f}B tokens over {total_steps:,} steps | "
        f"{tokens_per_step:,} tok/step (mb {args.micro_batch} x ga {args.grad_accum} x {block}) | "
        f"amp={'bf16' if use_bf16 else 'fp16'} "
        # Log the model's ACTUAL state, not the flag -- the flag lied for
        # every pre-2026-07-20 run (see the --no-grad-ckpt else-branch).
        f"ckpt={any(getattr(b, 'use_checkpoint', False) for b in raw_model.layers)}",
        flush=True,
    )

    raw_model.train()
    t0 = time.time()
    base_tokens = start_step * tokens_per_step
    seen = base_tokens
    # Finite default so a no-op resume (start_step >= total_steps, e.g. re-exporting
    # a finished run's model.pth) passes the final-save guard instead of raising
    # NameError on an undefined loss_acc.
    loss_acc = 0.0
    for step in range(start_step, total_steps):
        lr = get_lr(step, args.warmup, total_steps, args.lr, schedule=args.schedule, decay_frac=args.wsd_decay_frac)
        for g in optim.param_groups:
            g["lr"] = lr
        optim.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(args.grad_accum):
            X, Y = get_batch("train")
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(device == "cuda")):
                _, loss = model(X, targets=Y)
                loss = loss / args.grad_accum
            scaler.scale(loss).backward()
            loss_acc += loss.item()
            if args.throttle_ms:
                time.sleep(args.throttle_ms / 1000.0)
        if use_scaler:
            scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optim)
        scaler.update()
        seen += tokens_per_step

        if step % 10 == 0:
            dt = max(1e-9, time.time() - t0)
            tps = (seen - base_tokens) / dt
            print(
                f"step {step}/{total_steps} loss {loss_acc:.4f} lr {lr:.2e} {tps:,.0f} tok/s {seen / 1e9:.3f}B",
                flush=True,
            )

        if step > start_step and step % args.eval_every == 0:
            vl = estimate_val("val")
            print(f"  [val] step {step} loss {vl:.4f} ppl {math.exp(min(20, vl)):.1f}", flush=True)
            if use_val_gen:
                vg = estimate_val("val_gen")
                print(f"  [val-gen] step {step} loss {vg:.4f} ppl {math.exp(min(20, vg)):.1f}", flush=True)

        if step > start_step and step % args.save_every == 0:
            if math.isfinite(loss_acc):
                save("latest.pth", step)
                print(f"  [ckpt] step {step} -> {out / 'latest.pth'}", flush=True)
            else:
                print(
                    f"  [ckpt] step {step} SKIPPED -- non-finite loss ({loss_acc}); "
                    f"keeping last good latest.pth/prev.pth",
                    flush=True,
                )

        if args.archive_every and step > start_step and step % args.archive_every == 0 and math.isfinite(loss_acc):
            save(f"step_{step:06d}.pth", step)
            print(f"  [archive] step {step} -> {out / f'step_{step:06d}.pth'}", flush=True)

    if not math.isfinite(loss_acc):
        # same guard the periodic saves have: never ship NaN weights as model.pth
        raise SystemExit(
            f"FINAL SAVE REFUSED: last loss is not finite ({loss_acc}); "
            f"last good checkpoint remains {out / 'latest.pth'}"
        )
    save("model.pth", total_steps)
    print(f"done -> {out / 'model.pth'}  ({total_steps:,} steps, {seen / 1e9:.2f}B tokens)", flush=True)


if __name__ == "__main__":
    main()
