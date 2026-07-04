#!/usr/bin/env python
"""Finetune (SFT) the from-scratch Enigma into an instruct/tool model — the
"hands" pass of the roadmap.

Bespoke per the 2026-06-11 ruling (CLEANUP_TRACKER): the dormant FORGE stack
stays dormant; this follows the proven ``pretrain_enigma.py`` pattern and
shares its arsenal from ``enigma_engine.core.optim`` (``build_optimizer``/
``get_lr`` — so ``--optimizer muon`` and ``--schedule wsd`` work here too).

  python finetune_enigma.py --data data/sft/mix.jsonl \
      --init models/enigma_pretrain_large/latest.pth --out models/enigma_sft

Data: JSONL; each record either ``{"messages": [{role, content, ...}]}`` or a
``{"prompt", "completion"}`` pair (``response``/``answer``/``output`` keys are
accepted too). Rendering goes through ``enigma_engine.core.chat_format`` — the
SAME canonical template ``serve_enigma.py`` uses, so train and serve can never
disagree. The loss is masked to assistant spans (+ their ``<|im_end|>`` +
final EOS): she learns to ANSWER and to STOP, never to imitate the user.

On the first pass from a BASE checkpoint the chat-token embedding rows
(4718-4723, untouched by pretraining) are re-initialized to the mean real
embedding + small noise. Checkpoints carry the pretrain regime (schedule lock,
``prev.pth`` rotation, finite-loss guard) plus ``meta.chat_format`` so
``serve_enigma.py`` auto-detects instruct mode. Examples longer than --block
are SKIPPED and counted (block stays 1024 until the length-extension anneal).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

try:  # Windows consoles default to cp1252 and crash on unicode sample text.
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from enigma_engine.core.optim import build_optimizer, get_lr  # the shared arsenal

ROOT = Path(__file__).resolve().parent
IGNORE = -100  # ignore_index for the masked positions


def load_examples(path: Path, tokenizer, block: int):
    """JSONL -> list of (ids, mask) conversations that fit in one block."""
    from enigma_engine.core.chat_format import render_training

    examples, skipped_long, bad = [], 0, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            msgs = rec.get("messages")
            if not msgs:
                prompt = rec.get("prompt") or rec.get("question") or rec.get("instruction")
                completion = rec.get("completion") or rec.get("response") or rec.get("answer") or rec.get("output")
                if not (prompt and completion):
                    bad += 1
                    continue
                msgs = [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}]
            try:
                ids, mask = render_training(tokenizer, msgs)
            except ValueError:
                bad += 1
                continue
            if len(ids) > block + 1:
                skipped_long += 1
                continue
            if not any(mask):  # nothing to learn from (no assistant turn)
                bad += 1
                continue
            examples.append((ids, mask))
    print(
        f"data: {len(examples)} usable conversations from {path} "
        f"({skipped_long} skipped as longer than block {block}, {bad} malformed/empty)",
        flush=True,
    )
    return examples


def pack_blocks(examples, block: int, pad_id: int = 0, seed: int = 0):
    """Greedy-pack whole conversations into (block+1)-token rows; never split
    one across rows. Returns X [N, block] and Y [N, block] with IGNORE
    everywhere the next token is not an assistant-trainable position."""
    rng = random.Random(seed)
    order = list(range(len(examples)))
    rng.shuffle(order)

    rows = []
    cur_ids: list[int] = []
    cur_mask: list[bool] = []

    def flush():
        if not cur_ids:
            return
        ids = cur_ids + [pad_id] * (block + 1 - len(cur_ids))
        mask = cur_mask + [False] * (block + 1 - len(cur_mask))
        x = ids[:-1]
        y = [ids[i + 1] if mask[i + 1] else IGNORE for i in range(block)]
        rows.append((x, y))

    for i in order:
        ids, mask = examples[i]
        if len(cur_ids) + len(ids) > block + 1:
            flush()
            cur_ids, cur_mask = [], []
        cur_ids += ids
        cur_mask += mask
    flush()

    X = torch.tensor([r[0] for r in rows], dtype=torch.long)
    Y = torch.tensor([r[1] for r in rows], dtype=torch.long)
    return X, Y


def reinit_chat_rows(model: torch.nn.Module) -> list[int]:
    """Give the chat-token rows a real starting point: mean of the trained
    embedding + small noise. Pretraining only ever pushed these rows DOWN
    (never targets), so they carry no usable signal. Tied head => one tensor."""
    from enigma_engine.core.chat_format import BASE_VOCAB, CHAT_TOKENS

    emb = model.tok_embeddings.weight
    with torch.no_grad():
        mean = emb[:BASE_VOCAB].mean(dim=0)
        for tid in sorted(CHAT_TOKENS.values()):
            emb[tid] = mean + 0.02 * torch.randn_like(mean)
    return sorted(CHAT_TOKENS.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="chat JSONL (messages or prompt/completion)")
    ap.add_argument(
        "--init",
        default=str(ROOT / "models" / "enigma_pretrain_large" / "latest.pth"),
        help="BASE checkpoint to start the instruct pass from",
    )
    ap.add_argument("--resume", default=None, help="resume a previous SFT run's latest.pth")
    ap.add_argument(
        "--override-schedule",
        action="store_true",
        help="on resume, let CLI schedule args override the recorded schedule",
    )
    ap.add_argument("--out", default=str(ROOT / "models" / "enigma_sft"))
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5, help="~peak/30 of pretraining")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument(
        "--weight-decay", type=float, default=0.0, help="0 is the usual SFT choice; decay is for the long pretrain"
    )
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw")
    ap.add_argument("--schedule", choices=["cosine", "wsd"], default="cosine")
    ap.add_argument("--wsd-decay-frac", type=float, default=0.10)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--sanity", action="store_true", help="one fwd/bwd step then exit")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    from enigma_engine.core.chat_format import CHAT_FORMAT_NAME, attach_chat_tokens
    from enigma_engine.core.tokenizer import get_tokenizer

    tokenizer = attach_chat_tokens(get_tokenizer("bpe"))

    # Resume (an SFT run) or init (from a BASE checkpoint). Same early-load +
    # schedule-lock discipline as pretrain_enigma.
    ck = None
    resuming = False
    src = None
    if args.resume:
        rp = Path(args.resume)
        if not rp.exists() and rp.name == "latest.pth" and (rp.parent / "prev.pth").exists():
            print(f"resume: {rp} missing -> falling back to {rp.parent / 'prev.pth'}", flush=True)
            rp = rp.parent / "prev.pth"
        if not rp.exists():
            raise SystemExit(f"--resume {args.resume} not found")
        src, resuming = rp, True
    else:
        src = Path(args.init)
        if not src.exists():
            raise SystemExit(
                f"--init {src} not found — the instruct pass starts from a "
                f"pretrained checkpoint (her run is user-gated; see SUGGESTIONS.md)"
            )
    try:
        ck = torch.load(src, map_location=device, weights_only=False)
    except Exception as exc:
        prev = src.parent / "prev.pth"
        if resuming and src.name == "latest.pth" and prev.exists():
            # corrupt-but-present latest (e.g. power loss mid-write): fall
            # back to the rotated previous generation instead of stranding
            print(f"resume: {src} unreadable ({exc}) -> falling back to {prev}", flush=True)
            src = prev
            ck = torch.load(src, map_location=device, weights_only=False)
        else:
            raise
    if not (isinstance(ck, dict) and "model_state_dict" in ck and "config" in ck):
        raise SystemExit(f"{src} is not an Enigma checkpoint")

    saved_sched = ck.get("schedule") if resuming else None
    if saved_sched:
        diffs = {k: (v, getattr(args, k)) for k, v in saved_sched.items() if hasattr(args, k) and getattr(args, k) != v}
        if args.override_schedule:
            for k, (ck_v, cli_v) in diffs.items():
                print(f"resume: schedule[{k}] CLI {cli_v} OVERRIDES checkpoint {ck_v}", flush=True)
        else:
            for k, v in saved_sched.items():
                if hasattr(args, k):
                    setattr(args, k, v)
            for k, (ck_v, cli_v) in diffs.items():
                print(f"resume: schedule[{k}] = {ck_v} from checkpoint (CLI {cli_v} ignored)", flush=True)

    # seed AFTER the schedule restore so a checkpoint-recorded seed governs
    # torch.randperm's batch order too, not just the python-side shuffle/split
    torch.manual_seed(args.seed)

    examples = load_examples(Path(args.data), tokenizer, args.block)
    if not examples:
        raise SystemExit("no usable training conversations")
    rng = random.Random(args.seed)
    rng.shuffle(examples)
    n_val = min(int(len(examples) * args.val_frac), 512) if len(examples) >= 20 else 0
    val_examples, train_examples = examples[:n_val], examples[n_val:]
    X, Y = pack_blocks(train_examples, args.block, seed=args.seed)
    if n_val:
        VX, VY = pack_blocks(val_examples, args.block, seed=args.seed)
    print(
        f"packed: {X.shape[0]} train blocks" + (f" / {VX.shape[0]} val blocks" if n_val else " (no val split)"),
        flush=True,
    )

    from enigma_engine.core.model import Enigma
    from enigma_engine.core.model_presets import ForgeConfig

    config = ForgeConfig.from_dict(ck["config"])
    if args.block > config.max_seq_len:
        raise SystemExit(f"--block {args.block} > model max_seq_len {config.max_seq_len}")
    model = Enigma(config)
    missing, unexpected = model.load_state_dict(ck["model_state_dict"], strict=False)
    real_missing = [k for k in missing if "freqs_cis" not in k and "causal_mask" not in k]
    if unexpected or real_missing:
        raise SystemExit(f"arch mismatch: missing={real_missing[:5]} unexpected={unexpected[:5]}")
    model.to(device)
    raw_model = model

    meta = dict(ck.get("meta") or {})
    if meta.get("chat_format") != CHAT_FORMAT_NAME:
        rows = reinit_chat_rows(raw_model)
        print(
            f"init: chat-token embedding rows {rows} re-initialized (mean + noise) — "
            f"first instruct pass over a base checkpoint",
            flush=True,
        )
        meta = {"chat_format": CHAT_FORMAT_NAME, "init_from": str(src), "base_step": int(ck.get("step", 0))}

    optim = build_optimizer(raw_model, args.optimizer, args.lr, args.weight_decay)
    start_step = 0
    if resuming and "optimizer" in ck:
        try:
            optim.load_state_dict(ck["optimizer"])
        except Exception as exc:
            raise SystemExit(f"resume: optimizer state does not fit --optimizer {args.optimizer} ({exc})") from None
        # ck["step"] is a COMPLETED step (saved after its optimizer update);
        # resume at the next one instead of training it twice.
        start_step = int(ck.get("step", -1)) + 1
    del ck

    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    use_scaler = device == "cuda" and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    steps_per_epoch = max(1, X.shape[0] // (args.micro_batch * args.grad_accum))
    total_steps = max(1, args.epochs * steps_per_epoch)

    # Guard the schedule against a short run: with warmup >= total_steps the
    # ENTIRE run rides the warmup ramp and never reaches peak lr (this bit the
    # real mix.jsonl: 22 total steps vs warmup 50). Same for save/eval cadence
    # never firing. Clamp loudly; the clamped values are what get recorded.
    if args.warmup >= total_steps:
        clamped = max(1, total_steps // 10)
        print(f"warmup {args.warmup} >= total_steps {total_steps} -> clamped to {clamped}", flush=True)
        args.warmup = clamped
    if args.save_every > total_steps:
        clamped = max(1, total_steps // 2)
        print(f"save-every {args.save_every} > total_steps {total_steps} -> clamped to {clamped}", flush=True)
        args.save_every = clamped
    if args.eval_every > total_steps:
        clamped = max(1, total_steps // 4)
        print(f"eval-every {args.eval_every} > total_steps {total_steps} -> clamped to {clamped}", flush=True)
        args.eval_every = clamped

    schedule = {
        k: getattr(args, k)
        for k in (
            "data",
            "epochs",
            "lr",
            "warmup",
            "micro_batch",
            "grad_accum",
            "block",
            "weight_decay",
            "grad_clip",
            "optimizer",
            "schedule",
            "wsd_decay_frac",
            # run-defining too: a different seed on resume silently re-splits
            # train/val (val leakage) and reshuffles packing
            "seed",
            "val_frac",
        )
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def save(tag: str, step: int):
        from enigma_engine.core.safe_save import atomic_torch_save

        rotate = (out / "prev.pth") if tag == "latest.pth" else None
        atomic_torch_save(
            {
                "model_state_dict": raw_model.state_dict(),
                "config": config.to_dict(),
                "step": step,
                "optimizer": optim.state_dict(),
                "schedule": schedule,
                "meta": meta,
            },
            str(out / tag),
            rotate_to=rotate,
        )
        (out / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")

    def batch_at(data_x, data_y, idx):
        bx = data_x[idx].to(device, non_blocking=True)
        by = data_y[idx].to(device, non_blocking=True)
        return bx, by

    @torch.no_grad()
    def estimate_val() -> float:
        raw_model.eval()
        losses = []
        for s in range(0, VX.shape[0], args.micro_batch):
            bx, by = VX[s : s + args.micro_batch].to(device), VY[s : s + args.micro_batch].to(device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(device == "cuda")):
                _, loss = model(bx, targets=by, pad_token_id=IGNORE)
            losses.append(loss.item())
        raw_model.train()
        return sum(losses) / max(1, len(losses))

    if args.sanity:
        raw_model.train()
        idx = torch.arange(min(args.micro_batch, X.shape[0]))
        bx, by = batch_at(X, Y, idx)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(device == "cuda")):
            _, loss = model(bx, targets=by, pad_token_id=IGNORE)
        loss.backward()
        n_train_tok = int((by != IGNORE).sum())
        print(
            f"[sanity] batch={tuple(bx.shape)} loss={loss.item():.4f} "
            f"({n_train_tok} assistant-target tokens) — pipeline OK",
            flush=True,
        )
        return

    print(
        f"sft: {total_steps} steps ({args.epochs} epochs x {steps_per_epoch}) | "
        f"mb {args.micro_batch} x ga {args.grad_accum} x {args.block} | "
        f"lr {args.lr} {args.schedule}/{args.optimizer} | amp={'bf16' if use_bf16 else 'fp16'}",
        flush=True,
    )

    raw_model.train()
    t0 = time.time()
    perm = torch.randperm(X.shape[0])
    cursor = 0
    # Resume continues the ORIGINAL data order: replay the already-consumed
    # micro-batches through the same perm/reshuffle logic (identical randperm
    # calls -> identical RNG stream). Without this a resumed run restarts at
    # the head of the permutation and never reaches the tail examples.
    for _ in range(start_step * args.grad_accum):
        if cursor + args.micro_batch > X.shape[0]:
            perm = torch.randperm(X.shape[0])
            cursor = 0
        cursor += args.micro_batch
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
            if cursor + args.micro_batch > X.shape[0]:
                perm = torch.randperm(X.shape[0])
                cursor = 0
            idx = perm[cursor : cursor + args.micro_batch]
            cursor += args.micro_batch
            bx, by = batch_at(X, Y, idx)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(device == "cuda")):
                _, loss = model(bx, targets=by, pad_token_id=IGNORE)
                loss = loss / args.grad_accum
            scaler.scale(loss).backward()
            loss_acc += loss.item()
        if use_scaler:
            scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optim)
        scaler.update()

        if step % 10 == 0:
            dt = max(1e-9, time.time() - t0)
            print(
                f"step {step}/{total_steps} loss {loss_acc:.4f} lr {lr:.2e} "
                f"({(step - start_step + 1) / dt:.2f} step/s)",
                flush=True,
            )
        if n_val and step > start_step and step % args.eval_every == 0:
            print(f"  [val] step {step} loss {estimate_val():.4f}", flush=True)
        if step > start_step and step % args.save_every == 0 and math.isfinite(loss_acc):
            save("latest.pth", step)
            print(f"  [ckpt] step {step} -> {out / 'latest.pth'}", flush=True)

    if not math.isfinite(loss_acc):
        # same guard the periodic saves have: never ship NaN weights as the
        # deliverable (and never rotate away the last good latest.pth)
        raise SystemExit(
            f"FINAL SAVE REFUSED: last loss is not finite ({loss_acc}); "
            f"last good checkpoint remains {out / 'latest.pth'}"
        )
    if n_val:
        print(f"  [val] final loss {estimate_val():.4f}", flush=True)
    save("model.pth", total_steps)
    save("latest.pth", total_steps)
    print(f"done -> {out / 'model.pth'}  (instruct: {meta.get('chat_format')})", flush=True)


if __name__ == "__main__":
    main()
