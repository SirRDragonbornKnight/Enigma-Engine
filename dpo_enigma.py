#!/usr/bin/env python
"""DPO (Direct Preference Optimization) pass over the instruct model --
the alignment polish. Teaches Enigma to PREFER her authored voice over the
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
    python dpo_enigma.py --init models/enigma_v2_sft2/model.pth --out models/<new_run_dir>

The output checkpoint is serve-loadable (same format finetune_enigma writes).
Keep --lr tiny: DPO moves logits fast; the point is a nudge, not a retrain.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from enigma_engine.core.chat_format import CHAT_FORMAT_NAME, attach_chat_tokens, render_training
from enigma_engine.core.model import Enigma
from enigma_engine.core.model_presets import ForgeConfig
from enigma_engine.core.persona import Persona
from enigma_engine.core.safe_save import refuse_existing_artifact
from enigma_engine.core.tokenizer import get_tokenizer, vocab_file_for_size
from eval_leak_guard import persona_manifest, refuse_if_leaky

PAD = 0  # padding id for batching; padded positions are never scored


def surface_collisions(records: list[dict], containment: float = 0.6,
                       min_tokens: int = 3) -> list[tuple[int, int, float]]:
    """Chosen-vs-rejected surface overlap across DIFFERENT pairs -- the
    cancellation detector.

    The Stage-C epoch sweep (2026-08-10) measured what happens when the same
    phrasing sits on the chosen side of one class and the rejected side of
    another: the two gradients cancel and the behaviour never installs
    (unknown 0/9 at every beta and epoch, restored to 2/9 the moment the
    opposing class was removed). At this scale the model cannot prefer a
    string in one context and disprefer it in another.

    Containment (overlap / smaller set), not Jaccard: the measured collision
    was a SHORT rejected decline ("I can't know that.") sitting inside LONGER
    chosen declines, which Jaccard scores low precisely because the chosen
    side keeps explaining. Advisory by design -- the full identity build has
    legitimate topical overlap, and the trainer's job is to make the hazard
    visible before the GPU spends, not to veto data. Known limit: the
    tokenizer drops digits, so pairs differing only in numeric values read
    as identical surfaces and inflate the count.
    """
    tok = re.compile(r"[a-z']+")
    chosen = [(i, set(tok.findall((r.get("chosen") or "").lower()))) for i, r in enumerate(records)]
    hits = []
    for j, r in enumerate(records):
        rt = set(tok.findall((r.get("rejected") or "").lower()))
        if len(rt) < min_tokens:
            continue
        for i, ct in chosen:
            if i == j or len(ct) < min_tokens:
                continue
            score = len(ct & rt) / min(len(ct), len(rt))
            if score >= containment:
                hits.append((i, j, score))
    return hits


def collision_report(records: list[dict], hits: list[tuple[int, int, float]]) -> str:
    """Render the cancellation advisory. One example under a bare count was
    untriageable at real volume -- the live focused file fires 584 hits
    (audit 2026-08-13) -- so: count by class pair first, then the top
    examples ranked by containment."""
    from collections import Counter

    def _cls(k: int) -> str:
        return records[k].get("class") or "untagged"

    pair_counts = Counter((_cls(i), _cls(j)) for i, j, _ in hits)
    pairs_line = "  ".join(f"{a}(chosen)/{b}(rejected)={n}"
                           for (a, b), n in pair_counts.most_common())
    lines = [
        f"WARN: surface collision -- {len(hits)} chosen/rejected string pairs "
        f"overlap at containment >= 0.6. Opposing preferences sharing surfaces "
        f"CANCEL at this scale (measured 2026-08-10: unknown 0/9 "
        f"counterweighted vs 2/9 clean).",
        f"      by class pair: {pairs_line}",
    ]
    for i, j, score in sorted(hits, key=lambda h: -h[2])[:3]:
        lines.append(f"      {score:.2f}: chosen {records[i]['chosen']!r} "
                     f"vs rejected {records[j]['rejected']!r}")
    return "\n".join(lines)


def batches(rows: list, micro_batch: int):
    """Slices covering EVERY row -- the tail included. The old len // mb
    bound dropped the tail every epoch, and at the adopted 1-epoch recipe a
    dropped pair was never trained at all (T5 trained 332 of its 333 rows)."""
    for s in range(0, len(rows), micro_batch):
        yield rows[s : s + micro_batch]


def untagged_warning(by_class: dict) -> str | None:
    """The per-class instrument is DARK on an untagged corpus -- every row
    lands in one 'untagged' bucket and the report collapses to exactly the
    blended number it exists to replace (dpo_pairs.jsonl carries zero class
    tags, measured 2026-08-13)."""
    if by_class and set(by_class) == {"untagged"}:
        return ("WARN: no class tags in this corpus -- the per-class accuracy "
                "report will show one 'untagged' bucket. The tagged corpus is "
                "data/sft/dpo_focused.jsonl (make_dpo_data.py --focused).")
    return None


def startup_artifact_guard(args) -> None:
    """Refuse at STARTUP if --out already holds a model -- before the GPU
    spends anything (the old --out default, models/enigma_dpo, aimed a
    careless run straight at the v8 ROLLBACK checkpoint; audit 2026-08-13).
    --sanity is exempt: it never writes, and refusing a smoke test pointed
    at an existing run dir was pointless friction."""
    if not args.sanity:
        refuse_existing_artifact(Path(args.out))


def screening_manifest(persona_arg: str | None) -> Path:
    """The sealed gate this run screens its pairs against.

    WHICH gate is the loaded PERSONA's answer, never the argument's: an
    Enigma-SPELLED pack IS her, so it screens against hers. Split out of
    `main` because it is the whole behavior of `--persona` here -- the trainers
    route no data by persona -- and a test can reach it without a GPU."""
    pack = Path(persona_arg) if persona_arg else None
    persona = Persona.load(pack)
    return persona_manifest(None if persona.is_default else pack)


def artifact_payload(model_state: dict, config_dict: dict, step: int,
                     schedule: dict, meta: dict) -> dict:
    """The saved checkpoint dict -- the ONE place its shape is decided.
    No optimizer state on purpose: DPO has no --resume path (SFT keeps its
    because it does), so the AdamW moments were ~1.9 GB of dead weight on
    every artifact (measured against the live T5 checkpoint 2026-08-13)."""
    return {
        "model_state_dict": model_state,
        "config": config_dict,
        "step": step,
        "schedule": schedule,
        "meta": meta,
    }


def _render(tokenizer, prompt: str, answer: str, block: int, system: str | None = None):
    """(ids, target_mask) for one prompt->answer, or None if it can't fit.
    target_mask marks positions whose TOKEN is an assistant target (the DPO-
    scored positions), shifted for next-token prediction at scoring time.

    ``system`` renders a leading system turn. Serve folds retrieved memories
    and the tool preamble into exactly that slot (_with_context), so a
    preference about reading an injected memory block can only be trained in
    the shape it is served in -- a bare user turn is a shape she never sees
    with a memory hit. Pairs without the field render as before."""
    msgs = ([{"role": "system", "content": system}] if system else []) + [
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


def group_split(pairs, val_frac: float, seed: int, val_cap: int = 64):
    """Split (prompt, class, chosen_row, rejected_row) records into train/val
    with ALL records sharing a prompt on the same side. A flat shuffle leaked:
    the generated pairs put the same (prompt, chosen) against several
    rejected variants, and user-taught pairs ride x3 as exact duplicates, so
    twins landed on both sides and the end-of-run "preference accuracy"
    partly measured training-set recall (audit 2026-07-15). Returns
    (train_rows, val_rows) as (class, chosen, rejected) tuples."""
    rng = random.Random(seed)
    by_prompt: dict[str, list[tuple]] = {}
    for prompt, cls, c, r in pairs:
        by_prompt.setdefault(prompt, []).append((cls, c, r))
    groups = list(by_prompt.values())
    rng.shuffle(groups)
    n_val_target = min(int(len(pairs) * val_frac), val_cap)
    train_rows, val_rows = [], []
    # With fewer than two prompt groups there is no held-out prompt to spare
    # (a single-prompt teach_pairs.jsonl): train the whole set, val stays empty
    # -- class_accuracy() tolerates that; an empty TRAIN split would crash
    # _batchify([]) (final audit 2026-07-16 M2).
    if len(groups) < 2 or n_val_target <= 0:
        for g in groups:
            train_rows.extend(g)
        return train_rows, val_rows
    # Deal SMALLEST groups to val first and never assign the largest group to
    # val, so a single oversized group can neither empty train nor overshoot
    # val_cap (the old greedy fill added a whole group while under target, which
    # a giant first group blew past). sorted() is stable, so the shuffle still
    # orders equal-sized groups.
    order = sorted(groups, key=len)
    for g in order[:-1]:
        if len(val_rows) + len(g) <= n_val_target:
            val_rows.extend(g)
        else:
            train_rows.extend(g)
    train_rows.extend(order[-1])
    return train_rows, val_rows


def _refuse_over_block(block: int, max_seq_len: int) -> None:
    """DPO on positions past the model's context window forwards cleanly (RoPE
    table is built at 2x max_seq_len) and silently trains positions the window
    never covers -- finetune_enigma:394 has refused this since its guard; DPO
    gets the identical one. (Honest scope note: v2's max_seq_len is 8192 while
    the lineage TRAINED at block 2048 -- a 4096 block passes this guard exactly
    as it passes finetune's; the guard bounds the window, not the diet.)"""
    if block > max_seq_len:
        raise SystemExit(f"--block {block} > model max_seq_len {max_seq_len}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/sft/dpo_pairs.jsonl")
    ap.add_argument("--init", required=True, help="instruct checkpoint to start from (policy AND frozen reference)")
    # REQUIRED, no default: the old default was models/enigma_dpo -- the v8
    # ROLLBACK dir -- so a bare run aimed at the revert target (audit
    # 2026-08-13). Every artifact dir is somebody's receipt; name one.
    ap.add_argument("--out", required=True,
                    help="NEW output dir for this run's artifact (refused if it already holds a model.pth)")
    ap.add_argument("--beta", type=float, default=0.1)
    # 5e-7 x1 is the ADOPTED setting (the v8 artifact at models/enigma_dpo).
    # The old default 2e-6 over-optimized and damaged her (ROADMAP Phase 6,
    # measured 2026-07-06) -- the default should be the setting we trust.
    ap.add_argument("--lr", type=float, default=5e-7)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--micro-batch", type=int, default=4, help="preference PAIRS per step")
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--sanity", action="store_true", help="one step then exit")
    # Manifest selection and NOTHING else: --data still says which pairs to
    # train on. A pack's run screened against HER sealed gate would refuse the
    # pack for resembling questions its AI is never asked, while enforcing a
    # gate its weights are not trained for.
    ap.add_argument("--persona", default=None,
                    help="persona pack whose sealed gate screens --data; omitted = Enigma's")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    startup_artifact_guard(args)
    manifest = screening_manifest(args.persona)
    if args.persona:
        print(f"persona: {Persona.load(Path(args.persona)).name} | "
              f"leak gate: {manifest}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    src = Path(args.init)
    if not src.exists():
        raise SystemExit(f"--init {src} not found")
    ck = torch.load(src, map_location=device, weights_only=True)
    if not (isinstance(ck, dict) and "model_state_dict" in ck and "config" in ck):
        raise SystemExit(f"{src} is not an Enigma checkpoint")
    meta = dict(ck.get("meta") or {})
    if meta.get("chat_format") != CHAT_FORMAT_NAME:
        raise SystemExit("DPO expects an INSTRUCT checkpoint (run finetune_enigma.py first)")

    # Tokenizer selected by the CHECKPOINT's vocab (see finetune_enigma):
    # encoding preference pairs with any other vocab trains on wrong ids.
    try:
        _vocab_file = vocab_file_for_size(int(ck["config"]["vocab_size"]))
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"cannot pick a tokenizer for this checkpoint: {exc}")
    tokenizer = attach_chat_tokens(get_tokenizer("bpe", vocab_path=_vocab_file))
    print(f"tokenizer: {_vocab_file.name} (model vocab {ck['config']['vocab_size']})", flush=True)

    config = ForgeConfig.from_dict(ck["config"])
    _refuse_over_block(args.block, config.max_seq_len)
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
    records = []
    n_skip = 0
    prompts = []
    answers = []
    for line in Path(args.data).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        p = json.loads(line)
        sysblk = p.get("system") or None
        # SYSTEM turns are PROMPT SIDE -- the rule make_sft_data's probe_screen
        # and finetune_enigma's load_examples both file them under. The ask
        # alone was handed to the guard, so a sealed probe pasted into a memory
        # block reached the weights unrefused, and the memory_recall pairs are
        # built entirely of that shape.
        prompts.append(p["prompt"])
        if sysblk:
            prompts.append(sysblk)
        # The graded sides are advisory, not blocking: an answer legitimately
        # shares a question's content words (see refuse_if_leaky).
        answers += [x for x in (p.get("chosen"), p.get("rejected")) if x]
        c = _render(tokenizer, p["prompt"], p["chosen"], args.block, sysblk)
        r = _render(tokenizer, p["prompt"], p["rejected"], args.block, sysblk)
        if c is None or r is None:
            n_skip += 1
            continue
        records.append(p)
        # Group by the RENDERED prompt side: the same question under two
        # different memory blocks is two different preferences, and collapsing
        # them into one group would put a block she must read on the train side
        # and its twin on val (the group_split leak, in a new dress).
        pairs.append(((sysblk or "") + "\n" + p["prompt"],
                      p.get("class") or "untagged", c, r))
    refuse_if_leaky(prompts, Path(args.data), manifest=manifest, advisory=answers)
    if not pairs:
        raise SystemExit("no usable preference pairs")

    # The cancellation advisory (see surface_collisions). Loud, not fatal:
    # the run that would have burned three sweeps to rediscover this gets
    # told at load time instead.
    hits = surface_collisions(records)
    if hits:
        print(collision_report(records, hits), flush=True)

    rng = random.Random(args.seed)
    train_pairs, val_pairs = group_split(pairs, args.val_frac, args.seed)
    n_val = len(val_pairs)
    n_prompts = len({prompt for prompt, _, _, _ in pairs})
    by_class: dict[str, int] = {}
    for _, cls, _, _ in pairs:
        by_class[cls] = by_class.get(cls, 0) + 1
    print(
        f"dpo: {len(train_pairs)} train / {n_val} val pairs, split by prompt "
        f"({n_prompts} prompt groups; {n_skip} skipped as block-unfit; "
        + " ".join(f"{k}={v}" for k, v in sorted(by_class.items())) + ")",
        flush=True,
    )
    dark = untagged_warning(by_class)
    if dark:
        print(dark, flush=True)

    use_bf16 = device == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    optim = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))

    def step_loss(batch):
        """DPO loss + per-row margins for a list of (class, chosen, rejected)
        rows. Margins come back per row so callers can aggregate accuracy BY
        CLASS -- the epoch sweep's cancellation hid for three runs behind a
        blended 83.33% that never moved while loss fell; the contested class
        sitting at coin-flip would have named the bug in one."""
        cx, cm = _batchify([c for _, c, _ in batch], device)
        rx, rm = _batchify([r for _, _, r in batch], device)
        pi_c = _seq_logps(policy, cx, cm, amp_dtype)
        pi_r = _seq_logps(policy, rx, rm, amp_dtype)
        with torch.no_grad():
            ref_c = _seq_logps(ref, cx, cm, amp_dtype)
            ref_r = _seq_logps(ref, rx, rm, amp_dtype)
        margin = args.beta * ((pi_c - ref_c) - (pi_r - ref_r))
        loss = -F.logsigmoid(margin).mean()
        return loss, margin.detach()

    @torch.no_grad()
    def class_accuracy(rows):
        """(overall_loss, overall_acc, {class: (won, total)}) for a row list,
        loss weighted by rows (the old batch-mean average over-weighted the
        short tail batch)."""
        policy.eval()
        loss_sum = 0.0
        stats: dict[str, list[int]] = {}
        n = 0
        for batch in batches(rows, args.micro_batch):
            loss, margin = step_loss(batch)
            loss_sum += loss.item() * len(batch)
            n += len(batch)
            for (cls, _, _), m in zip(batch, margin.tolist()):
                w_t = stats.setdefault(cls, [0, 0])
                w_t[0] += int(m > 0)
                w_t[1] += 1
        policy.train()
        won = sum(w for w, _ in stats.values())
        return loss_sum / max(1, n), won / max(1, n), stats

    def class_line(stats: dict[str, list[int]]) -> str:
        return " ".join(f"{c} {w}/{t}" for c, (w, t) in sorted(stats.items()))

    if args.sanity:
        policy.train()
        loss, margin = step_loss(train_pairs[: args.micro_batch])
        loss.backward()
        acc = (margin > 0).float().mean().item()
        print(f"[sanity] loss={loss.item():.4f} acc={acc:.2f} -- pipeline OK", flush=True)
        return

    # ceil, and the epoch loop walks the whole list: len // mb silently
    # dropped the tail every epoch, and at the ADOPTED recipe of 1 epoch a
    # dropped pair was never trained at all, not merely reordered (up to
    # mb-1 pairs -- T5 trained 332 of its 333 train rows).
    steps_per_epoch = max(1, math.ceil(len(train_pairs) / args.micro_batch))
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
        for batch in batches(train_pairs, args.micro_batch):
            loss, margin = step_loss(batch)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.grad_clip)
            optim.step()
            last_loss = loss.item()
            if step % 10 == 0:
                rate = (step + 1) / (time.time() - t0)
                acc = (margin > 0).float().mean().item()
                print(f"step {step}/{total_steps} loss {last_loss:.4f} acc {acc:.2f} ({rate:.2f} step/s)", flush=True)
            step += 1

    if not math.isfinite(last_loss):
        # Same guard as pretrain/finetune's final save: never ship NaN weights
        # as the deliverable.
        raise SystemExit(f"FINAL SAVE REFUSED: last loss is not finite ({last_loss})")

    # Save FIRST. The class report below is a full forward pass over every
    # train row; sitting between train-end and the save it could OOM and eat
    # the finished checkpoint (audit 2026-08-13).
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
        artifact_payload(policy.state_dict(), config.to_dict(),
                         base_step + total_steps, schedule, meta),
        str(out / "model.pth"),
    )
    (out / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    print(f"done -> {out / 'model.pth'}  (dpo over {src.name})", flush=True)

    # End-of-run report, BY CLASS, train and val both. The signature worth
    # knowing by heart: a class pinned near coin-flip on ITS OWN TRAINING
    # ROWS while total loss fell is a contested surface -- the gradient went
    # to widening margins on classes it had already won.
    tl, ta, tstats = class_accuracy(train_pairs)
    print(f"  [train] dpo loss {tl:.4f} | preference accuracy {ta:.2%} | {class_line(tstats)}", flush=True)
    if n_val:
        vl, va, vstats = class_accuracy(val_pairs)
        print(f"  [val] dpo loss {vl:.4f} | preference accuracy {va:.2%} | {class_line(vstats)}", flush=True)


if __name__ == "__main__":
    main()
