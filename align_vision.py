#!/usr/bin/env python
"""Align her own eyes to her own brain -- Phase 4.5 step 4.

Trains her distilled VisionEncoder + a fresh vision_projection so the FROZEN
adopted text model can caption what she sees (LLaVA stage-1 recipe, on her
own weights end to end):

  encoder init = models/enigma_vision_distill/model.pth  (her ViT, distilled
                 from DINOv2-S on 2026-07-17; sees [-1,1] inputs)
  text model   = the --model checkpoint (default: models/enigma_v2_sft2,
                 the ADOPTED v2 lineage) with vision_hidden_size enabled;
                 every text weight FROZEN (freeze_text_io=True -- the
                 projection must target the served model's exact embedding
                 space so serve can later load encoder+projection onto the
                 pristine served weights). The EXISTING align artifact was
                 trained against v8's space pre-adoption: it boots on v2
                 but captions degrade (measured 2026-08-09) -- the v2
                 re-align is the open item.
  data         = data/vision/llava_pretrain.jsonl (558,128 pairs)

    python align_vision.py --sanity
    python align_vision.py
    python align_vision.py --resume models/enigma_vision_align/<stem>_vision_best.pt

Output: checkpoints under models/enigma_vision_align/ carrying BOTH the
model (with trained vision_projection) and vision_encoder_state_dict
(persistence contract f9ec5184). ASCII-only console (cp1252).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from enigma_engine.core.model import Enigma
from enigma_engine.core.model_presets import ForgeConfig
from enigma_engine.core.tokenizer import get_tokenizer, vocab_file_for_size
from enigma_engine.core.vision_encoder import VisionEncoder, VisionEncoderConfig
from enigma_engine.training.encoder_align import Trainer, TrainingConfig

ROOT = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser(description="Align the distilled vision encoder to the adopted text model")
    # Follows ADOPTION (v2 SFT-2 since 2026-08-09): the projection is trained
    # INTO the served model's embedding space, so a default left on the v8
    # rollback would silently align the eyes to a model nothing serves.
    p.add_argument("--model", default=str(ROOT / "models" / "enigma_v2_sft2" / "model.pth"))
    p.add_argument("--encoder", default=str(ROOT / "models" / "enigma_vision_distill" / "model.pth"))
    p.add_argument("--data", default=str(ROOT / "data" / "vision" / "llava_pretrain.jsonl"))
    p.add_argument("--out", default=str(ROOT / "models" / "enigma_vision_align"))
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--val", type=int, default=2000, help="held-out tail of the (seed-shuffled) pairs")
    p.add_argument("--resume", default=None)
    p.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="rolling mid-epoch checkpoint every N optimizer steps (0 = off); "
        "crash insurance for the single-epoch 558k run",
    )
    p.add_argument("--sanity", action="store_true", help="64 pairs, 1 epoch, then exit")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # -- her brain, frozen, with the vision projection port added --------
    ck = torch.load(args.model, map_location="cpu", weights_only=True)
    if not (isinstance(ck, dict) and "model_state_dict" in ck and "config" in ck):
        raise SystemExit(f"{args.model} is not an Enigma checkpoint")

    # -- her eyes, distilled -------------------------------------------
    eck = torch.load(args.encoder, map_location="cpu", weights_only=True)
    if "vision_encoder_state_dict" not in eck:
        raise SystemExit(f"{args.encoder} carries no vision_encoder_state_dict (run distill_vision_encoder.py)")
    if "vision_encoder_config" not in eck:
        raise SystemExit(
            f"{args.encoder} carries no vision_encoder_config; re-run "
            f"distill_vision_encoder.py to write one beside the weights"
        )
    try:
        vcfg = VisionEncoderConfig(**eck["vision_encoder_config"])
        encoder = VisionEncoder(vcfg)
    except Exception as exc:
        raise SystemExit(
            f"{args.encoder} carries a vision_encoder_config that will not rebuild "
            f"({type(exc).__name__}: {exc})"
        ) from exc
    encoder.load_state_dict(eck["vision_encoder_state_dict"], strict=True)
    encoder.to(device)

    cfg = ForgeConfig.from_dict(ck["config"])
    cfg.vision_hidden_size = vcfg.dim  # opens the projection port (fresh weights)
    model = Enigma(cfg)
    missing, unexpected = model.load_state_dict(ck["model_state_dict"], strict=False)
    bad_missing = [k for k in missing if "vision_projection" not in k]
    if bad_missing or unexpected:
        raise SystemExit(f"unexpected state mismatch: missing={bad_missing[:5]} unexpected={list(unexpected)[:5]}")
    print(f"text model loaded from {args.model}; fresh keys: {list(missing)}", flush=True)
    model.to(device)

    # The tokenizer follows the CHECKPOINT, not the default vocab file: a v2
    # model aligned against v1 ids trains happily and decodes to garbage.
    try:
        _vocab_file = vocab_file_for_size(int(cfg.vocab_size))
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"cannot pick a tokenizer for this checkpoint: {exc}")
    tokenizer = get_tokenizer("bpe", vocab_path=_vocab_file)
    print(f"tokenizer: {_vocab_file.name} (model vocab {cfg.vocab_size})", flush=True)

    # -- data ----------------------------------------------------------
    pairs: list[dict] = []
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("image") and rec.get("text"):
                pairs.append({"image": rec["image"], "text": rec["text"]})
    rng = random.Random(1234)
    rng.shuffle(pairs)
    if args.sanity:
        pairs = pairs[:64]
        n_val = 16
    else:
        n_val = min(args.val, max(0, len(pairs) - 1))
    val_data = pairs[len(pairs) - n_val :]
    train_data = pairs[: len(pairs) - n_val]
    print(f"align data: {len(train_data)} train / {len(val_data)} val pairs", flush=True)

    tcfg = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        warmup_steps=args.warmup,
        weight_decay=0.01,
        gradient_clip=1.0,
        use_amp=True,
        amp_dtype="bfloat16",
        checkpoint_dir=args.out,
        save_every=1,
        save_every_steps=max(0, args.save_steps),
        log_every=50,
        seed=1234,
    )
    trainer = Trainer(model, tokenizer, tcfg)
    state = trainer.train_vision(
        vision_encoder=encoder,
        data=train_data,
        val_data=val_data,
        unfreeze_text_layers=0,
        resume_from=args.resume,
        freeze_text_io=True,
    )
    print(
        f"align done: epochs {state.epoch}, steps {state.step}, "
        f"best tracked loss {state.best_loss:.4f}, "
        f"val losses {[f'{v:.4f}' for v in state.validation_losses]}",
        flush=True,
    )
    if state.abort_reason:
        raise SystemExit(f"ABORTED: {state.abort_reason}")


if __name__ == "__main__":
    main()
