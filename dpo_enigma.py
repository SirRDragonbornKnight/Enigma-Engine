#!/usr/bin/env python
"""DPO (Direct Preference Optimization) pass over the instruct model --
ROADMAP Phase 6. Teaches Enigma to PREFER her authored voice over the
measured failure modes (foreign identity, boilerplate, sycophancy) without
a reward model or PPO machinery: policy + frozen reference, one loss.

    loss = -log sigmoid(beta * ((pi_c - ref_c) - (pi_r - ref_r)))

where pi_*/ref_* are the summed assistant-token log-probs of the chosen/
rejected answer under the policy / frozen reference. Sequence rendering and
the assistant-token mask come from chat_format.render_training -- the SAME
renderer as SFT and serve, so preferences are learned on exactly the bytes
she serves.

Usage:
    python make_dpo_data.py                       # writes data/sft/dpo_pairs.jsonl
    python dpo_enigma.py --init models/enigma_sft/model.pth --out models/enigma_dpo

The output checkpoint is serve-loadable (same format finetune_enigma writes).
Keep --lr tiny: DPO moves logits fast; the point is a nudge, not a retrain.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from enigma_engine.core.chat_format import CHAT_FORMAT_NAME, attach_chat_tokens, render_training
from enigma_engine.core.model import Enigma
from enigma_engine.core.model_presets import ForgeConfig
from enigma_engine.core.tokenizer import get_tokenizer

PAD = 0  # padding id for batching; padded positions are never scored


def _render(tokenizer, prompt: str, answer: str, block: int):
    """(ids, target_mask) for one prompt->answer, or None if it can't fit.
    target_mask marks positions whose TOKEN is an assistant target (the DPO-
    scored positions), shifted for next-token prediction at scoring time."""
    msgs = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer},
    ]
    ids, mask = render_training(tokenizer, msgs, add_eos=True)
    if len(ids) > block or sum(mask) == 0:
        return None
    return ids, mask


def _seq_logps(model, ids: torch.Tensor, mask: torch.Tensor, amp_dtype) -> torch.Tensor:
    """Summed log-prob of the masked (assistant) tokens for each row.
    logits[t] predicts ids[t+1]; score positions t+1 where mask is True."""
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=ids.is_cuda):
        out = model(ids)
    logits = out[0] if isinstance(out, tuple) else out
    logp = F.log_softmax(logits[:, :-1].float(), dim=-1)
    tgt = ids[:, 1:]
    tok_logp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return (tok_logp * mask[:, 1:]).sum(dim=1)


def _batchify(rows, device):
    """Pad a list of (ids, mask) to a rectangle of tensors."""
    n = len(rows)
    width = max(len(ids) for ids, _ in rows)
    x = torch.full((n, width), PAD, dtype=torch.long)
    m = torch.zeros((n, width), dtype=torch.float32)
    for i, (ids, mask) in enumerate(rows):
        x[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        m[i, : len(ids)] = torch.tensor(mask, dtype=torch.float32)
    return x.to(device), m.to(device)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/sft/dpo_pairs.jsonl")
    ap.add_argument("--init", required=True, help="instruct checkpoint to start from (policy AND frozen reference)")
    ap.add_argument("--out", default="models/enigma_dpo")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--micro-batch", type=int, default=4, help="preference PAIRS per step")
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--sanity", action="store_true", help="one step then exit")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    tokenizer = attach_chat_tokens(get_tokenizer("bpe"))

    src = Path(args.init)
    if not src.exists():
        raise SystemExit(f"--init {src} not found")
    ck = torch.load(src, map_location=device, weights_only=False)
    if not (isinstance(ck, dict) and "model_state_dict" in ck and "config" in ck):
        raise SystemExit(f"{src} is not an Enigma checkpoint")
    meta = dict(ck.get("meta") or {})
    if meta.get("chat_format") != CHAT_FORMAT_NAME:
        raise SystemExit("DPO expects an INSTRUCT checkpoint (run finetune_enigma.py first)")

    config = ForgeConfig.from_dict(ck["config"])
    policy = Enigma(config)
    missing, unexpected = policy.load_state_dict(ck["model_state_dict"], strict=False)
    real_missing = [k for k in missing if "freqs_cis" not in k and "causal_mask" not in k]
    if unexpected or real_missing:
        raise SystemExit(f"arch mismatch: missing={real_missing[:5]} unexpected={unexpected[:5]}")
    policy.to(device)
    ref = copy.deepcopy(policy).eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    base_step = int(ck.get("step", 0))
    del ck

    # Load + render pairs (drop the ones that don't fit the block).
    pairs = []
    n_skip = 0
    for line in Path(args.data).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        p = json.loads(line)
        c = _render(tokenizer, p["prompt"], p["chosen"], args.block)
        r = _render(tokenizer, p["prompt"], p["rejected"], args.block)
        if c is None or r is None:
            n_skip += 1
            continue
        pairs.append((c, r))
    if not pairs:
        raise SystemExit("no usable preference pairs")
    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    n_val = min(int(len(pairs) * args.val_frac), 64)
    val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]
    print(f"dpo: {len(train_pairs)} train / {n_val} val pairs ({n_skip} skipped as block-unfit)", flush=True)

    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    optim = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))

    def step_loss(batch):
        """DPO loss + margin accuracy for a list of (chosen, rejected) rows."""
        cx, cm = _batchify([c for c, _ in batch], device)
        rx, rm = _batchify([r for _, r in batch], device)
        pi_c = _seq_logps(policy, cx, cm, amp_dtype)
        pi_r = _seq_logps(policy, rx, rm, amp_dtype)
        with torch.no_grad():
            ref_c = _seq_logps(ref, cx, cm, amp_dtype)
            ref_r = _seq_logps(ref, rx, rm, amp_dtype)
        margin = args.beta * ((pi_c - ref_c) - (pi_r - ref_r))
        loss = -F.logsigmoid(margin).mean()
        acc = (margin > 0).float().mean().item()
        return loss, acc

    @torch.no_grad()
    def estimate_val():
        policy.eval()
        losses, accs = [], []
        for s in range(0, len(val_pairs), args.micro_batch):
            loss, acc = step_loss(val_pairs[s : s + args.micro_batch])
            losses.append(loss.item())
            accs.append(acc)
        policy.train()
        n = max(1, len(losses))
        return sum(losses) / n, sum(accs) / n

    if args.sanity:
        policy.train()
        loss, acc = step_loss(train_pairs[: args.micro_batch])
        loss.backward()
        print(f"[sanity] loss={loss.item():.4f} acc={acc:.2f} -- pipeline OK", flush=True)
        return

    steps_per_epoch = max(1, len(train_pairs) // args.micro_batch)
    total_steps = args.epochs * steps_per_epoch
    print(
        f"dpo: {total_steps} steps ({args.epochs} epochs x {steps_per_epoch}) | "
        f"mb {args.micro_batch} pairs | beta {args.beta} lr {args.lr} | amp={'bf16' if use_bf16 else 'fp16'}",
        flush=True,
    )
    policy.train()
    t0 = time.time()
    step = 0
    last_loss = 0.0
    for _epoch in range(args.epochs):
        rng.shuffle(train_pairs)
        for s in range(0, steps_per_epoch * args.micro_batch, args.micro_batch):
            loss, acc = step_loss(train_pairs[s : s + args.micro_batch])
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            optim.step()
            last_loss = loss.item()
            if step % 10 == 0:
                rate = (step + 1) / (time.time() - t0)
                print(f"step {step}/{total_steps} loss {last_loss:.4f} acc {acc:.2f} ({rate:.2f} step/s)", flush=True)
            step += 1

    if not math.isfinite(last_loss):
        # Same guard as pretrain/finetune's final save: never ship NaN weights
        # as the deliverable.
        raise SystemExit(f"FINAL SAVE REFUSED: last loss is not finite ({last_loss})")

    if n_val:
        vl, va = estimate_val()
        print(f"  [val] dpo loss {vl:.4f} | preference accuracy {va:.2%}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from enigma_engine.core.safe_save import atomic_torch_save

    schedule = {
        "data": args.data, "beta": args.beta, "lr": args.lr, "epochs": args.epochs,
        "micro_batch": args.micro_batch, "block": args.block, "seed": args.seed,
        "trainer": "dpo",
    }
    meta = dict(meta)
    meta["dpo_from"] = str(src)
    atomic_torch_save(
        {
            "model_state_dict": policy.state_dict(),
            "config": config.to_dict(),
            "step": base_step + total_steps,
            "optimizer": optim.state_dict(),
            "schedule": schedule,
            "meta": meta,
        },
        str(out / "model.pth"),
    )
    (out / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    print(f"done -> {out / 'model.pth'}  (dpo over {src.name})", flush=True)


if __name__ == "__main__":
    main()
