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
        "is for FUTURE runs — never switch mid-lineage",
    )
    ap.add_argument(
        "--schedule",
        choices=["cosine", "wsd"],
        default="cosine",
        help="cosine = the live run's schedule. wsd = warmup-stable-decay "
        "(linear decay-to-zero over the last --wsd-decay-frac) — "
        "continuation-friendly, for FUTURE runs",
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
        "--no-diff-attn",
        action="store_true",
        help="disable differential attention -> use fused SDPA kernel (2-4x faster)",
    )
    ap.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="torch.compile the model (~1.5-2x; --no-compile for eager / if Triton is absent)",
    )
    ap.add_argument("--sanity", action="store_true", help="one fwd/bwd step then exit")
    ap.add_argument(
        "--throttle-ms",
        type=float,
        default=0.0,
        help="sleep N ms after each micro-batch to yield the GPU (e.g. while gaming); 0 = full speed",
    )
    args = ap.parse_args()

    if not TOKENS_BIN.exists():
        raise SystemExit(f"missing corpus: {TOKENS_BIN}")
    meta = json.loads(TOKENS_META.read_text(encoding="utf-8"))
    if meta.get("dtype") != "uint32":
        raise SystemExit(f"expected uint32 tokens, got {meta.get('dtype')}")
    vocab_meta = meta["vocab_size"]
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
        print(f"  (tokenizer unavailable — training on raw IDs: {exc})", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Load the resume checkpoint EARLY: its recorded schedule must be restored
    # before anything derived from schedule args (block, val windows, token
    # budget) is computed. A typo'd --resume must hard-fail here — the old
    # behavior silently started a FRESH run into the same --out directory.
    ck = None
    if args.resume:
        rp = Path(args.resume)
        if not rp.exists() and rp.name == "latest.pth" and (rp.parent / "prev.pth").exists():
            print(f"resume: {rp} missing -> falling back to {rp.parent / 'prev.pth'}", flush=True)
            rp = rp.parent / "prev.pth"
        if not rp.exists():
            raise SystemExit(
                f"--resume {args.resume} not found (no prev.pth fallback either) — "
                f"refusing to silently start a fresh run"
            )
        try:
            ck = torch.load(rp, map_location=device)
        except Exception as exc:
            prev = rp.parent / "prev.pth"
            if rp.name == "latest.pth" and prev.exists():
                # corrupt-but-present latest (e.g. power loss mid-write):
                # fall back to the rotated previous generation
                print(f"resume: {rp} unreadable ({exc}) -> falling back to {prev}", flush=True)
                rp = prev
                ck = torch.load(rp, map_location=device)
            else:
                raise
        saved_sched = ck.get("schedule")
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
        else:
            print(
                "resume: checkpoint predates schedule recording — trusting CLI args (this run will record them)",
                flush=True,
            )

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
    if bpt != 4 or stream_bytes % 4 != 0:
        raise SystemExit(
            f"{TOKENS_BIN} stream is not uint32-aligned (bpt={bpt}, {stream_bytes} bytes after header) -- truncated?"
        )
    if hdr_total != stream_bytes // 4:
        raise SystemExit(
            f"{TOKENS_BIN} header claims {hdr_total:,} tokens but the file holds "
            f"{stream_bytes // 4:,} (stale header or truncated write) -- refusing to train"
        )

    # memmap the uint32 token stream after the 256-byte header.
    data = np.memmap(TOKENS_BIN, dtype=np.uint32, mode="r", offset=HEADER_BYTES)
    n = len(data)
    val_n = min(args.val_tokens, n // 100)
    train_end = n - val_n
    print(f"memmapped {n:,} tokens  (train {train_end:,} / val {val_n:,})", flush=True)

    # [val-gen]: second eval window at the tail of the ORIGINAL corpus. The
    # 2026-06-07 anime append landed at the END of tokens.bin, so the held-out
    # tail above ([val]) became 100% anime-domain. This window restores a
    # general-domain signal. It was truly held out only until the append
    # (~16% train-sampled between then and this fence landing), so it reads
    # slightly optimistic; the fence in get_batch stops further leakage.
    vg_end = min(args.val_general_end, train_end)
    vg_lo = max(0, vg_end - val_n)
    use_val_gen = args.val_general_end > 0 and (vg_end - vg_lo) > args.block + 1
    if use_val_gen:
        print(f"val-gen window: [{vg_lo:,}, {vg_end:,}) — pre-append tail, fenced from train sampling", flush=True)

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
        print(f"resume: config rebuilt from checkpoint ({args.resume})", flush=True)
    else:
        config = get_preset(args.size, vocab_size=vocab_size)
        config.neftune_alpha = 0.0  # NEFTune is a finetuning trick; off for pretraining
        if args.no_diff_attn:
            config.use_differential_attn = False  # fused SDPA; must stay consistent across resumes
    config.dropout = args.dropout  # 0.0 for single-epoch pretraining (preset default 0.1 undertrains)
    if block > config.max_seq_len:
        config.max_seq_len = block
    model = Enigma(config)
    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable()
    model.to(device)
    raw_model = model  # the real nn.Module; checkpoints/optimizer bind here even when compiled
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"Enigma '{args.size}': {n_params / 1e6:.1f}M params, dim={config.dim} "
        f"layers={config.n_layers} heads={config.n_heads} block={block} "
        f"diff_attn={config.use_differential_attn}",
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
        )
    }

    out = Path(args.out) if args.out else ROOT / "models" / f"enigma_pretrain_{args.size}"
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
                f"resume arch mismatch — refusing to corrupt: missing={real_missing[:5]} unexpected={unexpected[:5]}"
            )
        if "optimizer" in ck:
            try:
                optim.load_state_dict(ck["optimizer"])
            except Exception as exc:
                raise SystemExit(
                    f"resume: checkpoint optimizer state does not fit --optimizer "
                    f"{args.optimizer} ({exc}) — the run was saved with a different "
                    f"optimizer; refusing to continue with reset moments"
                ) from None
        # ck["step"] is a COMPLETED step (saved after its optimizer update);
        # resume at the next one instead of training it twice.
        start_step = int(ck.get("step", -1)) + 1
        print(f"resumed from {args.resume} after step {start_step - 1} (next: {start_step})", flush=True)

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
            f"[sanity] batch={tuple(X.shape)} loss={loss.item():.4f} (random baseline ln(V)={base:.3f}) — pipeline OK",
            flush=True,
        )
        return

    print(
        f"training: target {args.tokens / 1e9:.2f}B tokens over {total_steps:,} steps | "
        f"{tokens_per_step:,} tok/step (mb {args.micro_batch} x ga {args.grad_accum} x {block}) | "
        f"amp={'bf16' if use_bf16 else 'fp16'} ckpt={not args.no_grad_ckpt}",
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
                    f"  [ckpt] step {step} SKIPPED — non-finite loss ({loss_acc}); "
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
