#!/usr/bin/env python
"""Distill DINOv2-S patch features into Enigma's OWN VisionEncoder.

Phase 4.5 step 3 (ROADMAP): her eyes must run on HER weights. Contrastive
pretraining from scratch needs 10M+ pairs; distillation from a frozen
teacher is the honest shortcut that still ends in owned weights -- the
teacher is used OFFLINE, once, and the resulting checkpoint is hers.

Teacher: facebook/dinov2-small (22M, frozen, eval). 224px / patch 14
        -> 16x16 = 256 patch tokens, dim 384 (CLS stripped).
Student: VisionEncoder VISION_PRESETS["medium"] (from scratch, ~19M).
        224px / patch 16 -> 14x14 = 196 patch tokens, dim 512.

Loss: per-patch cosine distillation. The teacher's 16x16 feature grid is
bilinearly interpolated to the student's 14x14 grid; a learned linear head
(512 -> 384, discarded at align time) maps student features into teacher
space; loss = (1 - cosine) per patch, plus 0.5 x the same on mean-pooled
global features. Student sees [-1, 1] inputs (her native from-scratch
convention -- align + serve preprocess must match); the teacher sees the
SAME image under ImageNet normalization.

    python distill_vision_encoder.py --sanity
    python distill_vision_encoder.py --images data/vision/llava/images
    python distill_vision_encoder.py --images data/vision/llava/images \
        --resume models/enigma_vision_distill/latest.pth

Output: models/enigma_vision_distill/{model.pth,latest.pth} -- checkpoint
holds vision_encoder_state_dict + config + the projection head + optimizer
state, so a killed run resumes exactly. ASCII-only console (cp1252).
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

from enigma_engine.core.safe_save import atomic_torch_save, refuse_existing_artifact
from enigma_engine.core.vision_encoder import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    VISION_PRESETS,
    VisionEncoder,
)

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "models" / "enigma_vision_distill"
_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class ImageFolder(Dataset):
    """Recursive image-file dataset. Returns raw [0, 1] float tensors
    (C, H, W); normalization variants are applied in the train step so one
    decode feeds both student and teacher. A corrupt file returns None and
    the collate drops it -- LLaVA-Pretrain ships a handful of bad JPEGs."""

    def __init__(self, root: Path, image_size: int, manifest: Path | None = None):
        self.image_size = image_size
        if manifest is not None and manifest.exists():
            self.paths = [Path(p) for p in manifest.read_text(encoding="utf-8").splitlines() if p]
        else:
            print(f"scanning {root} for images (one-time; manifest caches the list) ...", flush=True)
            self.paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in _EXTS)
            if manifest is not None and self.paths:
                manifest.write_text("\n".join(str(p) for p in self.paths), encoding="utf-8")
        if not self.paths:
            raise SystemExit(f"no images found under {root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        from PIL import Image

        try:
            with Image.open(self.paths[i]) as im:
                im = im.convert("RGB").resize((self.image_size, self.image_size), Image.BICUBIC)
                x = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8)
                return x.view(self.image_size, self.image_size, 3).permute(2, 0, 1).float() / 255.0
        except Exception:
            return None


def _collate_drop_none(batch):
    batch = [b for b in batch if b is not None]
    return torch.stack(batch) if batch else None


def teacher_to_student_grid(teacher_patches: torch.Tensor, n_student: int) -> torch.Tensor:
    """[B, Nt, D] on a square grid -> [B, Ns, D] via bilinear interpolation.
    Refuses non-square token counts loudly."""
    b, nt, d = teacher_patches.shape
    gt = int(nt**0.5)
    gs = int(n_student**0.5)
    if gt * gt != nt or gs * gs != n_student:
        raise ValueError(f"non-square patch grids: teacher {nt}, student {n_student}")
    if gt == gs:
        return teacher_patches
    grid = teacher_patches.transpose(1, 2).reshape(b, d, gt, gt)
    grid = F.interpolate(grid, size=(gs, gs), mode="bilinear", align_corners=False)
    return grid.reshape(b, d, gs * gs).transpose(1, 2)


def distill_loss(student_feats: torch.Tensor, teacher_patches: torch.Tensor, proj: nn.Module) -> torch.Tensor:
    """1 - cosine per patch (teacher grid resampled to student grid) plus
    0.5 x the same on mean-pooled global features."""
    t = teacher_to_student_grid(teacher_patches, student_feats.shape[1])
    s = proj(student_feats)
    patch = 1.0 - F.cosine_similarity(s, t, dim=-1).mean()
    global_ = 1.0 - F.cosine_similarity(s.mean(1), t.mean(1), dim=-1).mean()
    return patch + 0.5 * global_


def main() -> None:
    p = argparse.ArgumentParser(description="Distill DINOv2-S into Enigma's own VisionEncoder")
    p.add_argument("--images", default=str(ROOT / "data" / "vision" / "llava" / "images"))
    p.add_argument("--preset", default="medium", choices=sorted(VISION_PRESETS))
    p.add_argument("--teacher", default="facebook/dinov2-small")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--val-images", type=int, default=2048, help="held-out tail of the manifest")
    p.add_argument("--out", default=str(OUT_DIR))
    p.add_argument("--resume", default=None, help="checkpoint to continue from (e.g. .../latest.pth)")
    p.add_argument("--sanity", action="store_true", help="two tiny steps on synthetic data, then exit")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    cfg = VISION_PRESETS[args.preset]
    student = VisionEncoder(cfg).to(device)
    n_student = cfg.num_patches

    if args.sanity:
        # Synthetic teacher features: shape-compatible random tensors. Proves
        # the student/loss/optimizer plumbing without the teacher download.
        proj = nn.Linear(cfg.dim, 384).to(device)
        opt = torch.optim.AdamW(list(student.parameters()) + list(proj.parameters()), lr=args.lr)
        for step in range(2):
            x = torch.rand(2, 3, cfg.image_size, cfg.image_size, device=device) * 2 - 1
            t_fake = torch.randn(2, 256, 384, device=device)
            loss = distill_loss(student(x), t_fake, proj)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            print(f"[sanity] step {step} loss {loss.item():.4f}", flush=True)
        print("[sanity] OK", flush=True)
        return

    from transformers import AutoModel

    print(f"loading teacher {args.teacher} (frozen) ...", flush=True)
    teacher = AutoModel.from_pretrained(args.teacher).to(device).eval()
    for prm in teacher.parameters():
        prm.requires_grad = False

    images_root = Path(args.images)
    manifest = images_root.parent / "images_manifest.txt"
    ds = ImageFolder(images_root, cfg.image_size, manifest=manifest)
    n_val = min(args.val_images, max(0, len(ds) - 1))
    val_paths = ds.paths[-n_val:] if n_val else []
    ds.paths = ds.paths[: len(ds.paths) - n_val]
    val_ds = ImageFolder.__new__(ImageFolder)
    val_ds.image_size = cfg.image_size
    val_ds.paths = val_paths
    print(f"dataset: {len(ds)} train / {len(val_ds)} val images", flush=True)

    dl = DataLoader(
        ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
        collate_fn=_collate_drop_none, pin_memory=(device == "cuda"), drop_last=True,
        persistent_workers=(args.workers > 0),
    )

    proj = nn.Linear(cfg.dim, teacher.config.hidden_size).to(device)
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

    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    out_dir = Path(args.out)
    # A resume continuing its OWN directory is the sanctioned write; any other
    # run aimed at an existing lineage would rebuild it in place.
    if not (args.resume and Path(args.resume).resolve().parent == out_dir.resolve()):
        refuse_existing_artifact(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    step = 0
    start_epoch = 0
    best_val = float("inf")
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=True)
        needed = ("vision_encoder_state_dict", "proj_state_dict",
                  "optimizer_state_dict", "scheduler_state_dict")
        missing = [k for k in needed if not isinstance(ck, dict) or k not in ck]
        if missing:
            raise SystemExit(
                f"{args.resume} is not a resumable vision-distill checkpoint "
                f"(missing {missing})"
            )
        student.load_state_dict(ck["vision_encoder_state_dict"])
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
        vdl = DataLoader(val_ds, batch_size=args.batch, num_workers=2, collate_fn=_collate_drop_none)
        student.eval()
        losses = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            for x in vdl:
                if x is None:
                    continue
                x = x.to(device, non_blocking=True)
                t = teacher(pixel_values=(x - mean) / std).last_hidden_state[:, 1:]
                losses.append(distill_loss(student(x * 2 - 1), t, proj).item())
        student.train()
        return sum(losses) / max(1, len(losses))

    def _save(path: Path) -> None:
        atomic_torch_save(
            {
                "vision_encoder_state_dict": student.state_dict(),
                "vision_encoder_config": cfg.to_dict(),
                "proj_state_dict": proj.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": sched.state_dict(),
                "step": step,
                "epoch": epoch,
                "best_val": best_val,
                "teacher": args.teacher,
                "loss": "patch_cosine+0.5*global_cosine",
                "normalization": "student [-1,1]; teacher imagenet",
            },
            path,
        )

    print(
        f"distilling {args.teacher} -> VisionEncoder[{args.preset}] "
        f"({student.param_count()/1e6:.1f}M params, {n_student} patches) on {device}; "
        f"{total_steps} steps planned",
        flush=True,
    )
    student.train()
    t0 = time.time()
    seen = 0
    for epoch in range(start_epoch, args.epochs):
        for x in dl:
            if x is None:
                continue
            x = x.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
                with torch.no_grad():
                    t = teacher(pixel_values=(x - mean) / std).last_hidden_state[:, 1:]
                loss = distill_loss(student(x * 2 - 1), t, proj)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(student.parameters()) + list(proj.parameters()), 1.0)
            opt.step()
            sched.step()
            step += 1
            seen += x.shape[0]
            if step % 50 == 0:
                ips = seen / max(1e-9, time.time() - t0)
                eta_h = (total_steps - step) * args.batch / max(1e-9, ips) / 3600
                print(
                    f"epoch {epoch} step {step}/{total_steps} loss {loss.item():.4f} "
                    f"lr {sched.get_last_lr()[0]:.2e} {ips:.0f} img/s eta {eta_h:.1f}h",
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
                "best_val": best_val, "images": len(ds), "batch": args.batch,
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
