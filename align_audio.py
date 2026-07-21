#!/usr/bin/env python
"""Align her own ears to her own brain -- Phase 4.5 step 6.

Trains her distilled AudioEncoder + a fresh audio_projection so the FROZEN
adopted text model can transcribe what she hears (the vision align recipe
over mel spectrograms, on her own weights end to end):

  encoder init = models/enigma_audio_distill/model.pth  (her Whisper-style
                 encoder, distilled via distill_audio_encoder.py)
  text model   = the ADOPTED checkpoint (models/enigma_dpo, v8) with
                 audio_hidden_size enabled; every text weight FROZEN
                 (freeze_text_io=True -- the projection must target v8's
                 exact embedding space so serve can later load
                 encoder+projection onto the pristine served weights)
  data         = data/audio/librispeech.jsonl (collect_audio_data.py output)

    python align_audio.py --sanity
    python align_audio.py
    python align_audio.py --resume models/enigma_audio_align/<stem>_audio_best.pt

Output: checkpoints under models/enigma_audio_align/ carrying BOTH the
model (with trained audio_projection) and audio_encoder_state_dict
(persistence contract f9ec5184). ASCII-only console (cp1252).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from enigma_engine.core.audio_encoder import AudioEncoder, AudioEncoderConfig
from enigma_engine.core.model import Enigma
from enigma_engine.core.model_presets import ForgeConfig
from enigma_engine.core.tokenizer import get_tokenizer
from enigma_engine.training.encoder_align import Trainer, TrainingConfig

ROOT = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser(description="Align the distilled audio encoder to the adopted text model")
    p.add_argument("--model", default=str(ROOT / "models" / "enigma_dpo" / "model.pth"))
    p.add_argument("--encoder", default=str(ROOT / "models" / "enigma_audio_distill" / "model.pth"))
    p.add_argument("--pairs", default=str(ROOT / "data" / "audio" / "librispeech.jsonl"))
    p.add_argument("--out", default=str(ROOT / "models" / "enigma_audio_align"))
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="ragged mels batch through the padding-mask collate (mask-aware encoder)",
    )
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--val", type=int, default=2000, help="held-out tail of the (seed-shuffled) pairs")
    p.add_argument("--resume", default=None)
    p.add_argument(
        "--save-steps",
        type=int,
        default=500,
        help="rolling mid-epoch checkpoint every N optimizer steps (0 = off); "
        "crash insurance for the long single-epoch run",
    )
    p.add_argument("--sanity", action="store_true", help="64 pairs, 1 epoch, then exit")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # -- her brain, frozen, with the audio projection port added ---------
    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    if not (isinstance(ck, dict) and "model_state_dict" in ck and "config" in ck):
        raise SystemExit(f"{args.model} is not an Enigma checkpoint")

    # -- her ears, distilled -------------------------------------------
    eck = torch.load(args.encoder, map_location="cpu", weights_only=False)
    if "audio_encoder_state_dict" not in eck:
        raise SystemExit(f"{args.encoder} carries no audio_encoder_state_dict (run distill_audio_encoder.py)")
    if "audio_encoder_config" not in eck:
        raise SystemExit(
            f"{args.encoder} carries no audio_encoder_config; re-run "
            f"distill_audio_encoder.py to write one beside the weights"
        )
    try:
        acfg = AudioEncoderConfig(**eck["audio_encoder_config"])
        encoder = AudioEncoder(acfg)
    except Exception as exc:
        raise SystemExit(
            f"{args.encoder} carries an audio_encoder_config that will not rebuild "
            f"({type(exc).__name__}: {exc})"
        ) from exc
    encoder.load_state_dict(eck["audio_encoder_state_dict"], strict=True)
    encoder.to(device)

    cfg = ForgeConfig.from_dict(ck["config"])
    cfg.audio_hidden_size = acfg.dim  # opens the projection port (fresh weights)
    model = Enigma(cfg)
    missing, unexpected = model.load_state_dict(ck["model_state_dict"], strict=False)
    bad_missing = [k for k in missing if "audio_projection" not in k]
    if bad_missing or unexpected:
        raise SystemExit(f"unexpected state mismatch: missing={bad_missing[:5]} unexpected={list(unexpected)[:5]}")
    print(f"text model loaded from {args.model}; fresh keys: {list(missing)}", flush=True)
    model.to(device)

    tokenizer = get_tokenizer("bpe")

    # -- data ----------------------------------------------------------
    # Rows are {"audio": <absolute flac path>, "text": <transcript>}.
    pairs: list[dict] = []
    with open(args.pairs, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("audio") and rec.get("text"):
                pairs.append({"audio": rec["audio"], "text": rec["text"]})
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
        batch_size=args.batch_size,
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
    state = trainer.train_audio(
        audio_encoder=encoder,
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
