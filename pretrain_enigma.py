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
# Throughput-watch thresholds. The floor sits well under normal step-to-step
# jitter so eval and checkpoint windows do not trip it; the grad-checkpoint tax
# (30-40%) and a VRAM spill (an order of magnitude) both land far below it.
_TPS_FLOOR = 0.70
_TPS_REF_AFTER = 50  # steps of warmup excluded from the reference rate

# Everything a resume must restore to continue the SAME run, recorded into
# every checkpoint. Operational knobs that do not change the run's cost or math
# (--eval-every/--compile/--throttle-ms/...) stay CLI-controlled.
# --seed is deliberately absent: restoring it would re-seed the sampler and
# replay the windows the run already trained on.
SCHEDULE_KEYS = (
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
    # The anneal changes WHAT the model sees in the decay phase, so it is run
    # math, not an operational knob: a resume that dropped it would finish the
    # tail on a different diet than it started.
    "anneal_tokens",
    "anneal_frac",
    # Checkpointing is mathematically transparent but costs 30-40% throughput,
    # and the flag DISABLES a config default that is on. Unrecorded, a bare
    # --resume silently re-enables it and the rest of a multi-day run finishes
    # at two thirds speed, announced only by ckpt= in the banner.
    "no_grad_ckpt",
    # The archive cadence decides what post-hoc EMA has to average and how much
    # disk the run costs, and it is sized against the DECAY TAIL of the launch's
    # step count. Unrecorded, every resume re-imposes whatever the caller
    # happens to pass, so a launch tuned to put ~10 archives in the tail loses
    # that the first time it is restarted.
    "archive_every",
    # The per-source windows are FENCES: a resume that dropped this would let
    # train sampling eat every held-out window, and the rest of the run's
    # [val-src] would read in-distribution loss on data it just trained on.
    "val_per_source",
    # The pause-on-a-checkpoint protocol assumes the launch's save cadence
    # survives every resume; unrecorded, a bare --resume reverted it to the
    # 250-step default with no boot line saying so.
    "save_every",
)


def anneal_first_step(total_steps: int, decay_frac: float) -> int:
    """The step the decay phase -- and so the anneal -- begins at."""
    return int(total_steps * (1.0 - decay_frac))


def pause_resets_window(step: int, start_step: int, eval_every: int,
                        save_every: int, archive_every: int) -> bool:
    """True when this step ran an eval/checkpoint/archive pause, so the
    throughput window must restart after it.

    The windowed rate divides tokens by wall time since the last step-print.
    Eval and saves run BETWEEN two prints, so without a restart the next
    window absorbs the pause and reads as a collapse -- the same shape a
    real collapse takes (a config spilling to host memory), so every
    checkpoint window warns and the one signal that catches a genuine
    slowdown stops meaning anything.
    Mirrors the guards on the eval/save/archive blocks themselves: nothing
    fires at or before start_step, and archive_every=0 means no archives.
    """
    if step <= start_step:
        return False
    return (
        step % eval_every == 0
        or step % save_every == 0
        or bool(archive_every) and step % archive_every == 0
    )


def anneal_counts(micro_batch: int, anneal_frac: float) -> tuple[int, int]:
    """(curated, general) draws for one micro-batch during the decay phase.

    Clamped both ends: a fraction outside [0, 1] must not produce a negative
    draw count or ask for more rows than the batch holds. NaN means "no
    anneal" rather than an exception -- this runs inside the training loop and
    first fires at the decay boundary, days into a run, where a crash costs the
    most. Boot refuses a NaN fraction outright so it never gets this far."""
    if anneal_frac != anneal_frac:  # NaN
        return 0, micro_batch
    k = min(max(int(round(micro_batch * anneal_frac)), 0), micro_batch)
    return k, micro_batch - k


def anneal_region(anneal_tokens: int, train_end: int, block: int) -> int | None:
    """Start offset of the decay-phase curated region, validated at BOOT.

    Both bad sizes pass argparse and every startup check, then kill the run at
    the DECAY BOUNDARY -- days in -- when the phase's first curated draw
    executes, and a resume restores the same schedule from the checkpoint and
    dies at the same step. Nothing short of a boot refusal helps:
    * a region of block+1 tokens or fewer cannot fit one sample window plus a
      draw position, so randint on the curated side gets hi <= lo;
    * a region covering all but block+1 tokens of the stream leaves the same
      empty randint on the general side."""
    if anneal_tokens < 0:
        raise SystemExit(f"--anneal-tokens {anneal_tokens} must not be negative")
    if not anneal_tokens:
        return None
    if anneal_tokens <= block + 1:
        raise SystemExit(
            f"--anneal-tokens {anneal_tokens:,} cannot fit one sample window plus a draw "
            f"position (needs at least block+2 = {block + 2:,}) -- this boots clean and "
            f"then dies at the decay boundary, so it is refused here"
        )
    lo = train_end - anneal_tokens
    if lo <= block + 1:
        raise SystemExit(
            f"--anneal-tokens {anneal_tokens:,} leaves no general region ahead of it "
            f"(train stream holds {train_end:,} tokens)"
        )
    return lo


def refuse_repeated_source_in_val(meta: dict, train_end: int, block: int = 0,
                                  fenced: tuple = ()) -> None:
    """Refuse a corpus whose source placement defeats the training it claims.

    val is the last val_n tokens of the bin, and the T1 tail-position design
    died on exactly that coupling -- a curated shard tokenized last landed
    100% in val, held out and never trained on (round-7 audit, 2026-07-25).
    The first guard here checked only REPEATED sources, and its own audit
    caught the survivor the same day: on a checkout with no stackexchange
    dirs the UN-repeated curated shard walks last and reproduces the round-7
    failure with nothing declared. So, from the corpus's own metadata
    (corpora predating the extents record are unaffected):
    * ANY source lying entirely inside val is refused -- a source with zero
      training contribution is never intended, declared or not;
    * a REPEATED source (curated-by-declaration) must not extend into val or
      into a fenced window `(lo, hi)` (the val-gen window is sampled-around,
      so copies there are held out while their siblings memorize the eval);
    * a repeated source's per-pass span must exceed `block`, or the "copies
      land a whole source apart" spacing is void -- every copy co-occupies
      one training window and teaches verbatim repetition."""
    repeated = meta.get("repeated_sources") or {}
    extents = meta.get("source_token_extents") or {}
    for label, ext in extents.items():
        if ext and ext[0] >= train_end:
            raise SystemExit(
                f"source '{label}' lies entirely in the val tail (starts at token "
                f"{ext[0]:,}, train ends at {train_end:,}) -- zero training "
                f"contribution is never intended; move it earlier in SOURCE_DIRS "
                f"(pretokenize_data.py -- the walk is deliberately not a CLI knob) "
                f"or shrink --val-tokens"
            )
    for label, r in repeated.items():
        ext = extents.get(label)
        if not ext:
            continue
        span = ext[1] - ext[0]
        if ext[1] > train_end:
            raise SystemExit(
                f"repeated source '{label}' (x{r}) extends into the val tail "
                f"(ends at token {ext[1]:,}, train ends at {train_end:,}) -- its copies "
                f"would be held out and its train copies would leak into val; move it "
                f"earlier in SOURCE_DIRS (pretokenize_data.py) or shrink --val-tokens"
            )
        for lo, hi in fenced:
            if ext[0] < hi and ext[1] > lo:
                raise SystemExit(
                    f"repeated source '{label}' (x{r}) overlaps the fenced window "
                    f"[{lo:,}, {hi:,}) -- the fence redraws every sample touching it, "
                    f"so those copies are held out while their train siblings memorize "
                    f"the eval window"
                )
        if block and span // max(r, 1) <= block:
            raise SystemExit(
                f"repeated source '{label}' (x{r}) has a per-pass span of "
                f"{span // max(r, 1):,} tokens, not more than one {block}-token training "
                f"window -- the copies co-occupy windows and teach verbatim repetition; "
                f"lower the repeat count or grow the shard"
            )


def report_val_sources(meta: dict, train_end: int, n: int,
                       have_general_window: bool = False,
                       val_gen: tuple[int, int] | None = None) -> list[tuple[str, int]]:
    """Name the sources the val tail actually samples, loudest first.

    Pass `val_gen=(lo, hi)` to ALSO measure the --val-general-end window. Without
    it this reports only the tail, which is how a single-source [val-gen] hid: the
    note below tells the operator to read [val-gen] while nothing ever checked
    what [val-gen] contains.

    val is the last `n - train_end` tokens, so its domain is whatever the walk
    happened to place last, not a cross-section of the corpus. A val slice
    drawn from one source reports that source's loss under the name "val", and
    every LR, early-stop and go/no-go decision inherits the mislabel. This
    reports the split and warns when one source owns most of it; the fix is
    --val-general-end (a second window inside a chosen source) or a --val-tokens
    wide enough to span several, so the engine states the split rather than
    guessing which domain was intended.

    Returns [(label, tokens_in_val)] descending, empty when the corpus predates
    the extents record."""
    extents = meta.get("source_token_extents") or {}
    val_n = n - train_end
    if not extents or val_n <= 0:
        # Saying nothing reads as "val is fine". Say the check could not run.
        print("val sources: not recorded in this corpus -- the val tail's domain "
              "is unknown; [val] may be a single source", flush=True)
        return []
    share = []
    for label, ext in extents.items():
        # Reporting must never be the thing that kills a launch: a malformed or
        # truncated extent is skipped, not indexed blind.
        if not (isinstance(ext, (list, tuple)) and len(ext) >= 2
                and all(isinstance(v, int) and not isinstance(v, bool) for v in ext[:2])):
            continue
        overlap = min(ext[1], n) - max(ext[0], train_end)
        if overlap > 0:
            share.append((label, overlap))
    share.sort(key=lambda kv: -kv[1])
    if not share:
        return []
    top, top_tok = share[0]
    print(
        f"val sources ({len(share)}): "
        + ", ".join(f"{lab} {tok / val_n:.0%}" for lab, tok in share[:5])
        + ("" if len(share) <= 5 else f", +{len(share) - 5} more"),
        flush=True,
    )
    # The [val-gen] window is contiguous too -- it is [end - val_n, end), so it
    # lands inside whichever single source spans that offset. Directing the
    # operator to [val-gen] without stating what [val-gen] is MADE OF moves the
    # mislabel instead of fixing it, so the second window is measured and warned
    # about on the same terms as the first.
    gen_share: list[tuple[str, int]] = []
    gen_width = 0
    if val_gen is not None:
        gen_lo, gen_hi = val_gen
        gen_width = gen_hi - gen_lo
        if gen_width > 0:
            gen_share = [
                (lab, min(ext[1], gen_hi) - max(ext[0], gen_lo))
                for lab, ext in extents.items()
                if isinstance(ext, (list, tuple)) and len(ext) >= 2
                and all(isinstance(v, int) and not isinstance(v, bool) for v in ext[:2])
                and min(ext[1], gen_hi) - max(ext[0], gen_lo) > 0
            ]
            gen_share.sort(key=lambda kv: -kv[1])
            if gen_share:
                print(
                    f"val-gen sources ({len(gen_share)}): "
                    + ", ".join(f"{lab} {tok / gen_width:.0%}" for lab, tok in gen_share[:5])
                    + ("" if len(gen_share) <= 5 else f", +{len(gen_share) - 5} more"),
                    flush=True,
                )

    if top_tok / val_n >= 0.9:
        if have_general_window or gen_share:
            # A second window exists, so a narrow tail is expected and says
            # nothing new; the alarm belongs on whichever window is READ.
            print(f"  note: [val] is {top_tok / val_n:.0%} '{top}' -- read [val-gen] "
                  f"for the general-domain signal", flush=True)
        else:
            print(
                f"WARNING: {top_tok / val_n:.0%} of val is '{top}' -- every [val] number "
                f"this run prints is that domain's loss, not the corpus's. Pass "
                f"--val-general-end <token offset inside a representative source> for a "
                f"second window, or raise --val-tokens past this source's span.",
                flush=True,
            )

    if gen_share:
        gen_top, gen_top_tok = gen_share[0]
        if gen_top_tok / gen_width >= 0.9:
            print(
                f"WARNING: {gen_top_tok / gen_width:.0%} of [val-gen] is '{gen_top}' -- the "
                f"second window is single-source too, so [val-gen] is that domain's loss and "
                f"not the corpus's. A contiguous window cannot span the diet: move "
                f"--val-general-end onto a source boundary or widen --val-tokens.",
                flush=True,
            )
    return share


def source_val_windows(meta: dict, train_end: int, block: int,
                       per_source: int,
                       avoid: tuple[int, int] | None = None) -> list[tuple[str, int, int]]:
    """Carve one held-out window from the tail of each source's extent.

    A single contiguous window cannot represent the diet -- it lands inside
    whichever source spans its offset (the [val-gen] window is 100%
    FineWeb-Edu at every --val-tokens, measured 2026-07-28). One window per
    source, each fenced from train sampling and loss-weighted by that
    source's share of the diet, is the representative signal.

    Skipped, not clipped-blind:
    * REPEATED sources -- a fenced window there holds one copy out while its
      train siblings memorize it (same hazard refuse_repeated_source_in_val
      guards the val-gen window against);
    * sources whose in-train extent cannot give the window `block + 2` tokens
      after clipping to `train_end` (a window narrower than one sample would
      draw the same span every batch).

    A window that would overlap `avoid` (the val-gen window) is slid below it
    instead: FineWeb-Edu's extent ends exactly at the recommended
    --val-general-end, so without the slide its window NESTS INSIDE val-gen
    and [val-src]'s heaviest component is a re-measurement of [val-gen]
    rather than an independent signal.

    Returns [(label, lo, hi)] in walk order; empty when per_source is 0 or
    the corpus predates the extents record.
    """
    if per_source <= 0:
        return []
    extents = meta.get("source_token_extents") or {}
    repeated = meta.get("repeated_sources") or {}
    windows: list[tuple[str, int, int]] = []
    for label, ext in extents.items():
        if label in repeated:
            continue
        if not (isinstance(ext, (list, tuple)) and len(ext) >= 2
                and all(isinstance(v, int) and not isinstance(v, bool) for v in ext[:2])):
            continue
        hi = min(ext[1], train_end)
        if avoid is not None:
            a_lo, a_hi = avoid
            if max(ext[0], hi - per_source) < a_hi and hi > a_lo:
                hi = min(hi, a_lo)
        lo = max(ext[0], hi - per_source)
        if hi - lo <= block + 1:
            continue
        windows.append((label, lo, hi))
    return windows


def apply_seed(seed: int | None) -> bool:
    """Seed every stream a run draws from, and report whether it seeded.

    BOTH streams matter: torch drives weight init, numpy drives get_batch's
    window sampling. Seeding only torch leaves the data order random and a
    sweep's runs are then not comparable.
    """
    if seed is None:
        return False
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return True


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
        # Fixed RNG for BOTH passes: with --dropout > 0 the two backends would
        # otherwise draw different dropout masks and the cosine gate measures
        # noise, not the kernel (audit 2026-07-20). fork_rng also leaves the
        # global torch stream exactly where it was.
        with torch.random.fork_rng(devices=[X.device]):
            torch.manual_seed(1234)
            torch.cuda.manual_seed_all(1234)
            with torch.autocast(device_type="cuda", dtype=amp_dtype or torch.bfloat16, enabled=amp_dtype is not None):
                _, loss = model(X, targets=Y)
            loss.backward()
        flat = torch.cat([p.grad.detach().float().reshape(-1) for p in model.parameters() if p.grad is not None])
        model.zero_grad(set_to_none=True)
        return flat

    try:
        g_pin = _grads()
    except RuntimeError as exc:
        # Strict pins raise "No available kernel" when the backend cannot
        # serve this dtype/shape (e.g. flash/cudnn under fp32 on an fp16-only
        # card) -- refuse cleanly instead of a raw traceback.
        raise SystemExit(
            f"sdpa preflight: backend '{backend}' has no working kernel here "
            f"({str(exc).splitlines()[0][:120]}) -- pick another --sdpa-backend "
            "or pass --skip-sdpa-preflight"
        ) from None
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
    ap.add_argument(
        "--anneal-tokens",
        type=int,
        default=0,
        help="decay-tail anneal: treat the last N tokens of the train stream as the "
        "CURATED region and oversample it during the decay phase. 0 = off (the "
        "sampler draws uniformly, as the live lineage did). NOTE: the end of the BIN "
        "is val, so a shard tokenized last feeds val, not this region -- the T1 "
        "curated shard oversamples via pretokenize --repeat-sources instead; this "
        "flag waits for a deliberately placed region (e.g. a length-extension set)",
    )
    ap.add_argument(
        "--anneal-frac",
        type=float,
        default=0.5,
        help="fraction of each micro-batch drawn from the curated region once the "
        "decay phase starts (only meaningful with --anneal-tokens)",
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
    ap.add_argument(
        "--val-per-source",
        type=int,
        default=0,
        help="hold out the last N tokens of EACH non-repeated source as its own "
        "fenced eval window; [val-src] is their diet-share-weighted mean, the "
        "representative signal one contiguous window cannot give (0 = off)",
    )
    ap.add_argument(
        "--eval-only",
        action="store_true",
        help="load --init-from weights, print every val window at 6dp, and "
        "exit without training. Runs eager (compile warmup would dominate) "
        "and seeds the batch draw so different checkpoints score on "
        "IDENTICAL batches -- margins are paired, not two noisy absolutes. "
        "--resume is refused here: it restores each checkpoint's own "
        "schedule, which un-pairs the scores.",
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
        "--seed",
        type=int,
        default=None,
        help="seed weight init and batch sampling for a reproducible run "
        "(LR/HP sweeps need this). Default None = unseeded, the live lineage's behavior.",
    )
    ap.add_argument(
        "--throttle-ms",
        type=float,
        default=0.0,
        help="sleep N ms after each micro-batch to yield the GPU (e.g. while gaming); 0 = full speed",
    )
    ap.add_argument(
        "--sdpa-backend",
        default=None,
        choices=["auto", "cudnn", "flash", "efficient", "math"],
        help="pin ONE F.sdpa backend (strict: unsupported shapes error instead of silently "
        "falling back; schedule-recorded so a bare --resume restores the run's own pin). "
        "Unset = keep the lineage's recorded pin if any, else auto. Passing a value "
        "explicitly overrides only together with --override-schedule",
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
    if args.eval_only and not ckpt_arg:
        raise SystemExit("--eval-only needs weights: pass --init-from <ckpt>")
    if args.eval_only and args.resume:
        # Scoring compares checkpoints under ONE CLI. --resume restores each
        # checkpoint's OWN recorded schedule (block, corpus, val windows), so
        # two resumed checkpoints can silently evaluate on different data and
        # the "paired batches" property the re-seed buys is void. It also
        # refuses weight-only exports and loads optimizer moments a
        # forward-only pass never uses.
        raise SystemExit(
            "--eval-only with --resume un-pairs the scores (each checkpoint "
            "restores its own schedule); pass --init-from <ckpt> so every "
            "checkpoint evaluates under this command line's windows"
        )
    if warm_start:
        if not args.out and not args.eval_only:
            # eval-only writes nothing, so a scoring pass over an existing
            # checkpoint needs no output directory to protect.
            raise SystemExit("--init-from requires an explicit --out <new dir> (a warm-start is a new run)")
        if args.out and Path(args.out).resolve() == Path(ckpt_arg).resolve().parent:
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
            # weights_only=True, explicit and MEASURED (2026-07-26 audit): on
            # torch 2.10 it is the default and the real lineage checkpoints
            # (config + model_state_dict + optimizer + step + schedule) load
            # under it -- this path was already running refuse-foreign-pickle
            # mode and working. Pinning False here "to match the siblings"
            # was a live security downgrade; the siblings' False pins were
            # pre-2.6 legacy and every loader now pins True (KNOWN_ISSUES
            # 11.5, closed with per-artifact receipts).
            ck = torch.load(rp, map_location=device, weights_only=True)
        except Exception as exc:
            prev = rp.parent / "prev.pth"
            if rp.name == "latest.pth" and prev.exists():
                # corrupt-but-present latest (e.g. power loss mid-write):
                # fall back to the rotated previous generation
                print(f"load: {rp} unreadable ({exc}) -> falling back to {prev}", flush=True)
                rp = prev
                ck = torch.load(rp, map_location=device, weights_only=True)
            else:
                raise
        # A foreign .pth must refuse HERE with a curated message, not surface
        # as a bare KeyError hundreds of lines later at ck["model_state_dict"].
        if not (isinstance(ck, dict) and "model_state_dict" in ck and "config" in ck):
            raise SystemExit(
                f"{rp} is not an Enigma checkpoint (need model_state_dict + config)"
            )
        # Warm-start keeps the CLI schedule (fresh warmup at the new block);
        # only exact resume restores the checkpoint's recorded schedule.
        saved_sched = ck.get("schedule") if not warm_start else None
        if saved_sched:
            diffs = {
                k: (v, getattr(args, k))
                for k, v in saved_sched.items()
                if getattr(args, k, None) != v
                # unset --sdpa-backend is a sentinel, not a CLI opinion: the
                # resolution below keeps the recorded pin either way, so
                # listing it here logged the OPPOSITE of what happens under
                # --override-schedule (round-2 audit 2026-07-20).
                and not (k == "sdpa_backend" and getattr(args, k, None) is None)
            }
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
            # Keys the checkpoint predates take the CLI value, which for a bare
            # --resume is the argparse default. no_grad_ckpt is the expensive
            # one: its default re-enables checkpointing and the run finishes at
            # two thirds speed. Name every unrecorded key rather than let the
            # gap pass as a restore.
            missing = [k for k in SCHEDULE_KEYS if k not in saved_sched]
            if missing:
                # Show each unrecorded key WITH the value it is taking. A fixed
                # example in the text reported `no_grad_ckpt` even when that key
                # had been restored from the checkpoint and a different one was
                # missing.
                print(
                    "resume: this checkpoint predates "
                    + ", ".join(f"{k}={getattr(args, k)!r}" for k in missing)
                    + " -- each takes the CLI/default value shown. Re-pass any "
                      "the original run set.",
                    flush=True,
                )
        elif not warm_start:
            print(
                "resume: checkpoint predates schedule recording -- trusting CLI args (this run will record them)",
                flush=True,
            )

    # --sdpa-backend CLI None = no opinion: the lineage's recorded pin survives
    # even --override-schedule (audit 2026-07-20: override used to silently
    # un-pin a cudnn lineage back to auto when the user only meant to change
    # the LR). Resolved HERE -- before the schedule dict is built -- so the
    # recorded value is never the None sentinel.
    if args.sdpa_backend is None:
        recorded_pin = (ck.get("schedule") or {}).get("sdpa_backend") if (ck is not None and not warm_start) else None
        args.sdpa_backend = recorded_pin or "auto"

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
        f"corpus: {meta['total_tokens']:,} tokens, vocab {vocab_meta}, "
        f"{meta.get('file_size_gib', meta.get('file_size_gb'))} GiB ({meta['tokenizer']})",
        flush=True,
    )
    if meta.get("dedup_capped"):
        # Two sidecar generations share the flag: the exact-set corpora
        # (pre-2026-07-30) stopped remembering at the cap, the bloom corpora
        # keep answering but past design FPR. Either way the dedup did not
        # run at its claimed quality for the whole walk.
        print(f"corpus: paragraph dedup exceeded its {meta.get('dedup_cap', '?'):,}-entry "
              f"capacity during the walk -- the dedup did not run at design quality "
              f"past that point ({meta.get('dedup_backend', 'set')} backend)", flush=True)
    if meta.get("tokenizer_backend"):
        # Which encoder produced these ids. The rust and python paths are meant
        # to agree; unrecorded and unread, a corpus built by one and extended by
        # the other is indistinguishable from a consistent one.
        print(f"corpus: tokenized by the {meta['tokenizer_backend']} backend", flush=True)
    # vocab_size alone cannot tell two different 16,366-row tables apart, and a
    # corpus tokenized by the other one decodes to nothing. Compare the recorded
    # hash when the file is still where the sidecar says.
    _vsha = meta.get("vocab_sha256")
    if _vsha and meta.get("vocab_file"):
        _vpath = ROOT / meta["vocab_file"]
        if _vpath.exists():
            import hashlib
            _live = hashlib.sha256(_vpath.read_bytes()).hexdigest()
            if _live != _vsha:
                raise SystemExit(
                    f"vocab MISMATCH: {meta['vocab_file']} hashes {_live[:12]} but the "
                    f"corpus was tokenized against {_vsha[:12]}. Same row count is not "
                    f"the same table -- training on these ids would learn a vocabulary "
                    f"this file cannot decode. Restore the vocab that built the corpus, "
                    f"or retokenize."
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
    # One held-out window per source, each fenced like the val-gen window,
    # slid below the val-gen window where they would overlap. Weighted by
    # diet share (source extent span), their mean is the representative
    # signal a single contiguous window cannot give.
    src_windows = source_val_windows(
        meta, train_end, block, args.val_per_source,
        avoid=(vg_lo, vg_end) if use_val_gen else None,
    )
    if args.val_per_source > 0 and not src_windows:
        print("val-src: no eligible source windows (corpus predates the extents "
              "record, every source is repeated/too narrow, or --val-per-source "
              f"{args.val_per_source:,} is below block+2={block + 2:,}) -- "
              "[val-src] disabled",
              flush=True)
    if src_windows:
        extents = meta.get("source_token_extents") or {}
        src_weights = {
            label: min(extents[label][1], train_end) - extents[label][0]
            for label, _, _ in src_windows
        }
        print(
            f"val-src windows ({len(src_windows)}): "
            + ", ".join(f"{lab} [{lo:,}, {hi:,})" for lab, lo, hi in src_windows[:3])
            + ("" if len(src_windows) <= 3 else f", +{len(src_windows) - 3} more")
            + " -- fenced from train sampling, loss weighted by diet share "
            "(non-repeated sources only: a repeated source cannot be held out, "
            "its train copies would memorize the window)",
            flush=True,
        )
    # Every fenced interval, in one place: the guard checks them all and the
    # train sampler redraws around them all.
    fences = ([(vg_lo, vg_end)] if use_val_gen else []) + [
        (lo, hi) for _, lo, hi in src_windows
    ]
    # The redraw is rejection sampling, so fence coverage is a boot-time
    # contract: at ~2% it costs nothing, at 100% the first train batch spins
    # forever with no output. Refuse anything past 20% -- no legitimate
    # holdout needs a fifth of the corpus, and a --val-per-source typo is the
    # only way to get there.
    fence_tokens = sum(hi - lo for lo, hi in fences)
    if fences and train_end > 0 and fence_tokens / train_end > 0.20:
        raise SystemExit(
            f"fenced windows cover {fence_tokens / train_end:.0%} of the train "
            f"stream ({fence_tokens:,} of {train_end:,} tokens) -- the train "
            f"sampler redraws around fences and would spin, not sample. Lower "
            f"--val-per-source (currently {args.val_per_source:,})."
        )
    # After block and the fenced windows are known, so the guard can check all
    # three couplings (val tail, fence, per-pass span vs window).
    refuse_repeated_source_in_val(
        meta, train_end, block=block,
        fenced=tuple(fences),
    )
    report_val_sources(meta, train_end, n, have_general_window=use_val_gen,
                       val_gen=(vg_lo, vg_end) if use_val_gen else None)

    # Decay-tail anneal: during the WSD decay phase, oversample a curated
    # region of the corpus instead of continuing to draw uniformly.
    #
    # The v2 recipe called for "anneal on the best ~2-3B tokens" and there was
    # nowhere for that to land -- get_batch samples IID over the whole train
    # stream, so a file of hand-picked tokens changes nothing about what the
    # model actually sees. The curated region is the TAIL of the train stream;
    # during decay a fraction of every micro-batch is drawn from it and the
    # rest from the general region. NOTE what that tail actually IS: val is
    # carved off the very end of the BIN, so a shard tokenized last feeds val,
    # not this region (round-7 audit, 2026-07-25) -- the T1 curated shard
    # oversamples via pretokenize --repeat-sources instead, and this mechanism
    # waits for a region that is deliberately placed (e.g. a length-extension
    # anneal set).
    #
    # OFF by default: with --anneal-tokens 0 this is the same single draw the
    # sampler always made, so the live lineage's data order is untouched.
    anneal_lo = anneal_region(args.anneal_tokens, train_end, block)
    if anneal_lo is not None and not 0.0 <= args.anneal_frac <= 1.0:
        # Refuse at BOOT. The sampler clamps too, but a run that silently
        # annealed at a fraction nobody asked for is a lineage nobody can
        # reproduce -- and this first fires days in, at the decay boundary.
        raise SystemExit(
            f"--anneal-frac {args.anneal_frac} must be between 0 and 1"
        )
    def _draw(lo: int, hi: int, size: int):
        # dtype is load-bearing: legacy randint defaults to C-long (int32 on
        # Windows) and hi is ~56.7e9.
        return np.random.randint(lo, hi - block - 1, size=size, dtype=np.int64)

    def _draw_train(lo: int, hi: int, size: int):
        # Train draws redraw around the fences WITHIN THE RANGE THEY WERE
        # DRAWN FROM. Redrawing from the full train range instead silently
        # dilutes the anneal: a curated-region index that hits a fence would
        # come back as a general-region index, and the boot banner would
        # still announce the requested anneal fraction.
        ix = _draw(lo, hi, size)
        if fences:
            for j in range(len(ix)):
                spins = 0
                while any(f_lo - block <= ix[j] < f_hi for f_lo, f_hi in fences):
                    ix[j] = np.random.randint(lo, hi - block - 1, dtype=np.int64)
                    spins += 1
                    if spins > 100_000:
                        # The boot-time coverage guard bounds the global case;
                        # this bounds a subrange (e.g. an anneal region) the
                        # fences happen to smother. Spinning silently IS the
                        # failure -- say what range cannot be sampled.
                        raise SystemExit(
                            f"fence redraw spun {spins:,}x inside "
                            f"[{lo:,}, {hi:,}) -- the fenced windows cover "
                            f"this draw range; lower --val-per-source or "
                            f"move the region"
                        )
        return ix

    def get_batch(split: str, step: int | None = None,
                  bounds: tuple[int, int] | None = None):
        if bounds is not None:  # an explicit eval window (per-source val)
            lo, hi = bounds
        elif split == "train":
            lo, hi = 0, train_end
        elif split == "val":
            lo, hi = train_end, n
        else:  # "val_gen" — pre-append general-domain window
            lo, hi = vg_lo, vg_end
        # total_steps is bound further down; read it at CALL time, which is
        # always after the schedule is settled.
        if (split == "train" and anneal_lo is not None and step is not None
                and step >= anneal_first_step(total_steps, args.wsd_decay_frac)):
            curated, general = anneal_counts(args.micro_batch, args.anneal_frac)
            parts = []
            if curated:
                parts.append(_draw_train(anneal_lo, train_end, curated))
            if general:
                parts.append(_draw_train(lo, anneal_lo, general))
            ix = np.concatenate(parts)
        elif split == "train":
            ix = _draw_train(lo, hi, args.micro_batch)
        else:
            ix = _draw(lo, hi, args.micro_batch)
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
    # Seed BEFORE construction: weight init draws from torch, get_batch draws
    # from numpy. Unseeded (None) leaves both streams as the live lineage ran them.
    if apply_seed(args.seed):
        print(f"seed: {args.seed} (weight init + batch sampling)", flush=True)
        # Reaching here already means a seed was supplied: --seed defaults to
        # None and is deliberately absent from SCHEDULE_KEYS, so a resume never
        # restores one.
        if args.resume and not warm_start:
            # The sampler draws windows from numpy, so re-seeding an exact
            # resume restarts that draw sequence: the resumed segment re-reads
            # the windows the first segment already trained on. Weight init is
            # irrelevant here (the weights come from the checkpoint), so the
            # seed buys nothing on a resume and costs unique-token coverage.
            print(
                "WARNING: --seed on an exact resume replays the batch windows "
                "from step 0 -- the resumed segment re-reads what it already "
                "trained on. Omit --seed when resuming.",
                flush=True,
            )
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

    # Recorded into every checkpoint so a resume restores it exactly (see the
    # resume block above); the contract lives at SCHEDULE_KEYS.
    schedule = {k: getattr(args, k) for k in SCHEDULE_KEYS}

    if anneal_lo is not None:
        first = int(total_steps * (1.0 - args.wsd_decay_frac))
        print(
            f"decay-tail anneal: last {args.anneal_tokens:,} train tokens are the curated "
            f"region (offset {anneal_lo:,}); from step {first:,}/{total_steps:,} "
            f"{args.anneal_frac:.0%} of each micro-batch is drawn from it",
            flush=True,
        )

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
    if not args.eval_only:
        # A scoring pass writes nothing -- creating the size-derived default
        # dir would litter models/ with empty lineage directories.
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
            else:
                # Every checkpoint this trainer writes carries optimizer state;
                # a weight-only file here is an ema_checkpoints.py output or a
                # converted import -- an exact continuation would silently
                # reset the moments (audit 2026-07-20).
                kind = "an ema_checkpoints.py output" if "ema" in ck else "weight-only"
                raise SystemExit(
                    f"resume: {rp} has no optimizer state ({kind}) -- exact continuation "
                    "would silently reset the moments; use --init-from <ckpt> --out <new dir> "
                    "for a weight-only warm start"
                )
            # ck["step"] is a COMPLETED step (saved after its optimizer update);
            # resume at the next one instead of training it twice.
            start_step = int(ck.get("step", -1)) + 1
            print(f"resumed from {args.resume} after step {start_step - 1} (next: {start_step})", flush=True)

    # SDPA backend pin (v2 recipe): applied AFTER the resume/schedule restore so
    # a bare --resume runs under the pin its own lineage recorded, and BEFORE the
    # first forward pass anywhere below (preflight/compile/sanity/loop).
    if device != "cuda" and args.sdpa_backend != "auto":
        print(
            f"sdpa: WARN backend pin '{args.sdpa_backend}' requested/recorded but device is {device} -- "
            "pin not applicable, running default kernels (the recorded schedule is preserved)",
            flush=True,
        )
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
    if args.eval_only and args.compile:
        # A scoring pass is a few forward-only minutes; compile's autotune
        # warmup would dominate it and buy nothing.
        args.compile = False
        print("eval-only: running eager (compile warmup would dominate a scoring pass)", flush=True)
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
            # Record-only (never restored): a resumed segment is not
            # bit-reproducible, so re-seeding on resume would replay step-0 data.
            "seed": args.seed,
            # The best windowed rate this segment reached. The throughput watch
            # otherwise compares a run only against ITSELF, so a segment that is
            # slow from step 0 -- a resume that lost --no-grad-ckpt, a config
            # spilling to host memory at launch -- sets a low reference and
            # never warns. Carrying it forward gives the next segment an
            # absolute number to fall short of.
            "tok_s_ref": ref_tps,
                "optimizer": optim.state_dict(),
                "schedule": schedule,
            },
            str(out / tag),
            rotate_to=rotate,
        )
        (out / "config.json").write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")

    @torch.no_grad()
    def estimate_val(split: str = "val", bounds: tuple[int, int] | None = None,
                     iters: int | None = None) -> float:
        raw_model.eval()
        losses = []
        for _ in range(iters if iters is not None else args.eval_iters):
            X, Y = get_batch(split, bounds=bounds)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(device == "cuda")):
                _, loss = model(X, targets=Y)
            losses.append(loss.item())
        raw_model.train()
        return sum(losses) / max(1, len(losses))

    def estimate_val_src(iters_per_window: int) -> tuple[float, list[tuple[str, float]]]:
        """Per-source losses and their diet-share-weighted mean.

        The weights are each source's share of the TRAIN stream, so the
        aggregate reads "loss on the diet actually trained on" -- an
        unweighted mean would let a 0.1% source move the number as much as a
        36% one.
        """
        per = [(label, estimate_val(bounds=(lo, hi), iters=iters_per_window))
               for label, lo, hi in src_windows]
        wsum = sum(src_weights[label] for label, _ in per)
        agg = sum(src_weights[label] * v for label, v in per) / max(1, wsum)
        return agg, per

    # --- eval-only: score a checkpoint on every val window, then exit -------
    if args.eval_only:
        # Identical batches across checkpoints: re-seed the draw HERE, after
        # boot consumed an unknowable number of draws. Scores become paired
        # (same windows, same order), so checkpoint margins resolve below the
        # single-score noise floor.
        eval_seed = args.seed if args.seed is not None else 1234
        np.random.seed(eval_seed)
        print(f"eval-only: seed {eval_seed}, {args.eval_iters} iters/window, "
              f"mb {args.micro_batch} x block {block}", flush=True)
        vl = estimate_val("val")
        print(f"[eval] val loss {vl:.6f} ppl {math.exp(min(20, vl)):.2f}", flush=True)
        if use_val_gen:
            vg = estimate_val("val_gen")
            print(f"[eval] val-gen loss {vg:.6f} ppl {math.exp(min(20, vg)):.2f}", flush=True)
        if src_windows:
            agg, per = estimate_val_src(args.eval_iters)
            for label, v in sorted(per, key=lambda kv: -src_weights[kv[0]]):
                print(f"[eval]   {label} loss {v:.6f} (weight {src_weights[label] / sum(src_weights.values()):.1%})",
                      flush=True)
            print(f"[eval] val-src loss {agg:.6f} ppl {math.exp(min(20, agg)):.2f} "
                  f"(diet-weighted over {len(per)} sources)", flush=True)
        return

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
    # Windowed throughput + peak VRAM. The cumulative rate above averages over
    # the whole run, so a slowdown that starts on day five barely moves it: a
    # drop from 31k to 12k tok/s after five good days still reads 28k. A config
    # sitting on the VRAM ceiling spills to host memory over PCIe and crawls
    # with nothing raised, so the peak is printed beside the rate and a
    # sustained fall below `_TPS_FLOOR` of the best window says so out loud.
    # Seeded from the previous segment when resuming, so the first slow window
    # of a resume is measured against how fast this lineage ACTUALLY ran rather
    # than against nothing.
    ref_tps = float((ck or {}).get("tok_s_ref") or 0.0) if not warm_start else 0.0
    if ref_tps:
        print(f"throughput reference from the previous segment: {ref_tps:,.0f} tok/s "
              f"(a sustained fall below {_TPS_FLOOR:.0%} of it will say so)", flush=True)
    win_t0, win_base, tps_warned = t0, seen, False
    for step in range(start_step, total_steps):
        lr = get_lr(step, args.warmup, total_steps, args.lr, schedule=args.schedule, decay_frac=args.wsd_decay_frac)
        for g in optim.param_groups:
            g["lr"] = lr
        optim.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(args.grad_accum):
            X, Y = get_batch("train", step)
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
            now = time.time()
            dt = max(1e-9, now - t0)
            tps = (seen - base_tokens) / dt
            wtps = (seen - win_base) / max(1e-9, now - win_t0)
            win_t0, win_base = now, seen
            mem = ""
            if device == "cuda":
                # RESERVED, not allocated: the caching allocator's reservation is
                # what actually occupies the card, and it is the number that
                # approaches the ceiling. Allocated understates it (measured
                # 0.68 vs 1.18 GiB on the same step) and would read as headroom
                # that is not there.
                peak = torch.cuda.max_memory_reserved() / 1e9
                torch.cuda.reset_peak_memory_stats()
                mem = f" peak {peak:.1f}GB"
            print(
                f"step {step}/{total_steps} loss {loss_acc:.4f} lr {lr:.2e} "
                f"{wtps:,.0f} tok/s (avg {tps:,.0f}){mem} {seen / 1e9:.3f}B",
                flush=True,
            )
            # Skip the first windows: compile, autotune and cache warmup all
            # land there and would set an unreachable reference.
            if step >= start_step + _TPS_REF_AFTER:
                if wtps > ref_tps:
                    ref_tps = wtps
                if ref_tps and wtps < _TPS_FLOOR * ref_tps:
                    if not tps_warned:
                        print(
                            f"WARNING: throughput {wtps:,.0f} tok/s is below "
                            f"{_TPS_FLOOR:.0%} of this lineage's best {ref_tps:,.0f} -- "
                            f"check peak VRAM against the card (a config on the "
                            f"ceiling spills to host memory and never errors) and "
                            f"whether --no-grad-ckpt survived a resume.",
                            flush=True,
                        )
                        tps_warned = True
                elif tps_warned and wtps > 0.85 * ref_tps:
                    tps_warned = False

        if step > start_step and step % args.eval_every == 0:
            # 6dp: adjacent LR rungs differ by ~1e-3 and a 4dp print
            # quantizes real seed spread to identical (measured 2026-07-29).
            vl = estimate_val("val")
            print(f"  [val] step {step} loss {vl:.6f} ppl {math.exp(min(20, vl)):.1f}", flush=True)
            if use_val_gen:
                vg = estimate_val("val_gen")
                print(f"  [val-gen] step {step} loss {vg:.6f} ppl {math.exp(min(20, vg)):.1f}", flush=True)
            if src_windows:
                # The iters budget spreads across windows down to a floor of
                # 4, so past eval_iters/4 windows the periodic cost grows
                # linearly (30 windows at the default 40 iters = 120 forward
                # passes, ~3x one estimate_val). [final] and --eval-only
                # spend the full budget per window -- a different estimator
                # with different variance than this periodic one.
                agg, _ = estimate_val_src(max(4, args.eval_iters // len(src_windows)))
                print(f"  [val-src] step {step} loss {agg:.6f} ppl {math.exp(min(20, agg)):.1f}", flush=True)

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

        # A pause is not training time: restart the throughput window after
        # eval/save/archive so the next window measures only training (the
        # cumulative avg above keeps counting everything, unchanged).
        if pause_resets_window(step, start_step, args.eval_every, args.save_every, args.archive_every):
            win_t0, win_base = time.time(), seen

    if not math.isfinite(loss_acc):
        # same guard the periodic saves have: never ship NaN weights as model.pth
        raise SystemExit(
            f"FINAL SAVE REFUSED: last loss is not finite ({loss_acc}); "
            f"last good checkpoint remains {out / 'latest.pth'}"
        )
    save("model.pth", total_steps)
    print(f"done -> {out / 'model.pth'}  ({total_steps:,} steps, {seen / 1e9:.2f}B tokens)", flush=True)

    # Final val runs AFTER the save: the periodic eval only fires on the
    # eval_every cadence, so a run whose length is not a multiple of it would
    # end with no measurement of the decayed weights (the number an LR/HP sweep
    # is judged on) -- but a failure while measuring must never cost the run
    # its checkpoint. Weights are already on disk by this point.
    try:
        final_val = estimate_val("val")
        # bits/TOKEN, not bits/char: comparable across runs on the SAME corpus
        # and tokenizer (what a sweep ranks), NOT across tokenizers -- that
        # needs bits/char, i.e. this divided by the corpus chars/token.
        print(
            f"[final] val loss {final_val:.6f} ppl {math.exp(min(20, final_val)):.2f} "
            f"bits/token {final_val / math.log(2):.4f}",
            flush=True,
        )
        if use_val_gen:
            final_vg = estimate_val("val_gen")
            print(
                f"[final] val-gen loss {final_vg:.6f} ppl {math.exp(min(20, final_vg)):.2f}",
                flush=True,
            )
        if src_windows:
            final_src, _ = estimate_val_src(args.eval_iters)
            print(
                f"[final] val-src loss {final_src:.6f} ppl {math.exp(min(20, final_src)):.2f}",
                flush=True,
            )
    except Exception as exc:
        print(f"[final] val FAILED ({type(exc).__name__}: {exc}); {out / 'model.pth'} is saved", flush=True)


if __name__ == "__main__":
    main()
