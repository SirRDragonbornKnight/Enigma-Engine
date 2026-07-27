#!/usr/bin/env python
"""Distill whisper-base's encoder into Enigma's OWN AudioEncoder.

Phase 4.5 step 6 (ROADMAP): her ears must run on HER weights. Same honest
shortcut as the eyes (distill_vision_encoder.py): the teacher is used
OFFLINE, once, and the resulting checkpoint is hers.

Teacher: openai/whisper-base encoder (frozen; d_model 512, 1500 frames for
        its fixed 30 s window). Teacher consumes WHISPER's mel features.
Student: AudioEncoder AUDIO_PRESETS["base"] (from scratch; dim 512, stride-2
        front-end -> also ~1500 frames at 30 s). Student consumes HER OWN
        mel pipeline's output for the SAME waveform -- the pipeline serve
        and the (still-to-be-built) audio-align trainer will use.

Loss: per-frame cosine over the REAL (non-padding) frames only -- clips are
2-15 s inside the padded 30 s window, and distilling 80% silence would waste
the student's capacity -- plus 0.5 x pooled-global cosine over the same
mask. A learned 512->512 head maps student space to teacher space
(discarded at align time). Frames align 1:1 (both stacks are stride-2 on
10 ms hops); any off-by-one from mel edge handling is truncated to the
shorter side.

    python distill_audio_encoder.py --sanity
    python distill_audio_encoder.py --data data/audio/librispeech.jsonl
    python distill_audio_encoder.py --resume models/enigma_audio_distill/latest.pth

Output: models/enigma_audio_distill/{model.pth,latest.pth} carrying
audio_encoder_state_dict + config + head + optimizer (exact resume).
ASCII-only console (cp1252).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from enigma_engine.core.audio_encoder import (
    AUDIO_PRESETS,
    AudioEncoder,
    load_audio,
    preprocess_audio,
)
from enigma_engine.core.safe_save import atomic_torch_save

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "models" / "enigma_audio_distill"
WINDOW_S = 30  # whisper's fixed encoder window


class SpeechDataset(Dataset):
    """librispeech.jsonl rows -> (her_mel [80,T], whisper_mel [80,3000],
    real_frames). Waveforms are padded/truncated to the 30 s window; both
    mel variants come from the SAME padded waveform. A corrupt file returns
    None and the collate drops it."""

    def __init__(self, jsonl: Path, encoder_config, feature_extractor):
        self.cfg = encoder_config
        self.fx = feature_extractor
        self.paths: list[str] = []
        with open(jsonl, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("audio"):
                    self.paths.append(rec["audio"])
        if not self.paths:
            raise SystemExit(f"no audio rows in {jsonl}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        try:
            wav = load_audio(self.paths[i], target_sr=self.cfg.sample_rate)
            n_window = WINDOW_S * self.cfg.sample_rate
            wav = wav[:n_window]
            n_real = wav.shape[0]
            if n_real < n_window:
                wav = torch.cat([wav, torch.zeros(n_window - n_real)])
            # center=True STFT on the exact 30 s window emits 3001 frames
            # (floor(480000/160)+1) -- one past the encoder's pos_embed
            # after stride-2 (1501 > max_audio_len 1500) and one past
            # whisper's own 3000 (audit 2026-07-20). Trim to the
            # encoder's bound: frames align 1:1 and pos_embed holds.
            her_mel = preprocess_audio(wav, self.cfg)[0][:, : 2 * self.cfg.max_audio_len]  # [80, <=3000]
            whisper_mel = torch.from_numpy(
                self.fx(wav.numpy(), sampling_rate=self.cfg.sample_rate, return_tensors="np").input_features[0]
            )  # [80, 3000]
            # real ENCODER frames: samples -> mel hop -> stride-2 conv
            real_frames = max(1, n_real // self.cfg.hop_length // 2)
            return her_mel, whisper_mel, real_frames
        except Exception:
            return None


def _collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    her = torch.stack([b[0] for b in batch])
    wsp = torch.stack([b[1] for b in batch])
    real = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return her, wsp, real


def masked_distill_loss(
    student_frames: torch.Tensor,
    teacher_frames: torch.Tensor,
    real_frames: torch.Tensor,
    proj: nn.Module,
) -> torch.Tensor:
    """1 - cosine per REAL frame + 0.5 x the same on mean-pooled features.
    Off-by-one length mismatches truncate to the shorter side."""
    n = min(student_frames.shape[1], teacher_frames.shape[1])
    s = proj(student_frames[:, :n])
    t = teacher_frames[:, :n]
    mask = torch.arange(n, device=s.device)[None, :] < real_frames[:, None].clamp(max=n)  # [B, n]
    cos = F.cosine_similarity(s, t, dim=-1)  # [B, n]
    denom = mask.sum().clamp(min=1)
    patch = 1.0 - (cos * mask).sum() / denom
    m = mask.unsqueeze(-1).to(s.dtype)
    s_pool = (s * m).sum(1) / m.sum(1).clamp(min=1)
    t_pool = (t * m).sum(1) / m.sum(1).clamp(min=1)
    global_ = 1.0 - F.cosine_similarity(s_pool, t_pool, dim=-1).mean()
    return patch + 0.5 * global_


def main() -> None:
    p = argparse.ArgumentParser(description="Distill whisper-base's encoder into Enigma's own AudioEncoder")
    p.add_argument("--data", default=str(ROOT / "data" / "audio" / "librispeech.jsonl"))
    p.add_argument("--preset", default="base", choices=sorted(AUDIO_PRESETS))
    p.add_argument("--teacher", default="openai/whisper-base")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=300)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--val-clips", type=int, default=512, help="held-out tail of the (seed-shuffled) rows")
    p.add_argument("--out", default=str(OUT_DIR))
    p.add_argument("--resume", default=None)
    p.add_argument("--sanity", action="store_true", help="two tiny synthetic steps, then exit")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    cfg = AUDIO_PRESETS[args.preset]
    student = AudioEncoder(cfg).to(device)

    if args.sanity:
        proj = nn.Linear(cfg.dim, 512).to(device)
        opt = torch.optim.AdamW(list(student.parameters()) + list(proj.parameters()), lr=args.lr)
        for step in range(2):
            mel = torch.randn(2, cfg.n_mels, 3000, device=device)
            t_fake = torch.randn(2, 1500, 512, device=device)
            real = torch.tensor([400, 900], device=device)
            loss = masked_distill_loss(student(mel), t_fake, real, proj)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            print(f"[sanity] step {step} loss {loss.item():.4f}", flush=True)
        print("[sanity] OK", flush=True)
        return

    from transformers import WhisperFeatureExtractor, WhisperModel

    print(f"loading teacher {args.teacher} encoder (frozen) ...", flush=True)
    teacher = WhisperModel.from_pretrained(args.teacher).encoder.to(device).eval()
    for prm in teacher.parameters():
        prm.requires_grad = False
    fx = WhisperFeatureExtractor.from_pretrained(args.teacher)

    ds = SpeechDataset(Path(args.data), cfg, fx)
    rng = random.Random(1234)
    rng.shuffle(ds.paths)
    n_val = min(args.val_clips, max(0, len(ds) - 1))
    val_paths = ds.paths[-n_val:] if n_val else []
    ds.paths = ds.paths[: len(ds.paths) - n_val]
    val_ds = SpeechDataset.__new__(SpeechDataset)
    val_ds.cfg, val_ds.fx, val_ds.paths = cfg, fx, val_paths
    print(f"dataset: {len(ds)} train / {len(val_ds)} val clips", flush=True)

    dl = DataLoader(
        ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
        collate_fn=_collate, pin_memory=(device == "cuda"), drop_last=True,
        persistent_workers=(args.workers > 0),
    )

    proj = nn.Linear(cfg.dim, teacher.config.d_model).to(device)
    opt = torch.optim.AdamW(
        list(student.parameters()) + list(proj.parameters()),
        lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95),
    )
    total_steps = (len(ds) // args.batch) * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / max(1, args.warmup))
        * (0.5 * (1 + torch.cos(torch.tensor(min(1.0, s / max(1, total_steps)) * 3.141592653589793)).item())),
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    start_epoch = 0
    best_val = float("inf")
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=True)
        student.load_state_dict(ck["audio_encoder_state_dict"])
        proj.load_state_dict(ck["proj_state_dict"])
        opt.load_state_dict(ck["optimizer_state_dict"])
        sched.load_state_dict(ck["scheduler_state_dict"])
        step = ck.get("step", 0)
        start_epoch = ck.get("epoch", 0)
        best_val = ck.get("best_val", float("inf"))
        print(f"resumed from {args.resume} at step {step}", flush=True)

    def _evaluate() -> float:
        if not val_ds.paths:
            return float("nan")
        vdl = DataLoader(val_ds, batch_size=args.batch, num_workers=2, collate_fn=_collate)
        student.eval()
        losses = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            for b in vdl:
                if b is None:
                    continue
                her, wsp, real = (x.to(device, non_blocking=True) for x in b)
                t = teacher(wsp).last_hidden_state
                losses.append(masked_distill_loss(student(her), t, real, proj).item())
        student.train()
        return sum(losses) / max(1, len(losses))

    def _save(path: Path) -> None:
        atomic_torch_save(
            {
                "audio_encoder_state_dict": student.state_dict(),
                "audio_encoder_config": cfg.to_dict(),
                "proj_state_dict": proj.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": sched.state_dict(),
                "step": step,
                "epoch": epoch,
                "best_val": best_val,
                "teacher": args.teacher,
                "loss": "masked_frame_cosine+0.5*global_cosine",
                "normalization": "student: her own mel pipeline; teacher: whisper mel",
            },
            path,
        )

    print(
        f"distilling {args.teacher} -> AudioEncoder[{args.preset}] "
        f"({student.param_count()/1e6:.1f}M params) on {device}; {total_steps} steps planned",
        flush=True,
    )
    student.train()
    t0 = time.time()
    seen = 0
    for epoch in range(start_epoch, args.epochs):
        for b in dl:
            if b is None:
                continue
            her, wsp, real = (x.to(device, non_blocking=True) for x in b)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                with torch.no_grad():
                    t = teacher(wsp).last_hidden_state
                loss = masked_distill_loss(student(her), t, real, proj)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(student.parameters()) + list(proj.parameters()), 1.0)
            opt.step()
            sched.step()
            step += 1
            seen += her.shape[0]
            if step % 25 == 0:
                cps = seen / max(1e-9, time.time() - t0)
                eta_h = (total_steps - step) * args.batch / max(1e-9, cps) / 3600
                print(
                    f"epoch {epoch} step {step}/{total_steps} loss {loss.item():.4f} "
                    f"lr {sched.get_last_lr()[0]:.2e} {cps:.0f} clips/s eta {eta_h:.1f}h",
                    flush=True,
                )
            if step % args.save_every == 0:
                _save(out_dir / "latest.pth")
        val = _evaluate()
        print(f"epoch {epoch} done: val distill loss {val:.4f}", flush=True)
        _save(out_dir / "latest.pth")
        if val == val and val < best_val:  # not-NaN and improved
            best_val = val
            _save(out_dir / "model.pth")
            print(f"new best (val {val:.4f}) -> model.pth", flush=True)
    (out_dir / "distill_meta.json").write_text(
        json.dumps(
            {
                "teacher": args.teacher, "preset": args.preset, "steps": step,
                "best_val": best_val, "clips": len(ds), "batch": args.batch,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("distillation complete", flush=True)


if __name__ == "__main__":
    random.seed(1234)
    torch.manual_seed(1234)
    main()
