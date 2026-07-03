"""
Vision Training Data Collector for Enigma Engine
=================================================
Downloads and formats image-text pair datasets for vision SFT.

Sources (all require `pip install datasets`):
  - LLaVA-Pretrain (--llava-pretrain N --images-dir PATH)
        Stage-1 LLaVA caption data (558K). Captions stream from HF; image
        bytes live in the dataset's `images.zip` archive which the user
        downloads + extracts manually (~14 GB).

Output format:
  JSONL with {"image": "<absolute-path>", "text": "<caption>"} per line.
  Compatible with `enigma_engine.gui.scanners.scan_vision_data` JSONL
  strategy and with `Trainer.train_vision()`'s expected input shape.

Usage:
  # 1. Download images.zip from the dataset card and extract somewhere:
  #    https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain
  # 2. Run the collector pointing at the extraction dir:
  python collect_vision_data.py --llava-pretrain 100000 \\
        --images-dir D:/datasets/llava/images
  # 3. Point Forge → Vision training mode at:
  #    data/vision/llava_pretrain.jsonl
"""

import argparse
import hashlib
import json
import logging
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/vision")


def _ensure_datasets() -> bool:
    """Check that the `datasets` library is installed."""
    try:
        import datasets  # noqa: F401

        return True
    except ImportError:
        logger.error("The `datasets` library is required.\nInstall with: pip install datasets")
        return False


def _dedup_pairs(pairs: list[dict]) -> list[dict]:
    """Remove exact-duplicate (image, text) records."""
    seen: set[bytes] = set()
    unique: list[dict] = []
    for item in pairs:
        key = hashlib.sha256((item["image"] + "\x00" + item["text"]).encode("utf-8")).digest()[:16]
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _write_jsonl(pairs: list[dict], path: Path) -> int:
    """Write pairs to JSONL file, return count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in pairs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return len(pairs)


# ── LLaVA-Pretrain (V-5) ──────────────────────────────────────────


def collect_llava_pretrain(
    max_samples: int = 100000,
    images_dir: Path | str | None = None,
) -> list[dict]:
    """Download and format LLaVA-Pretrain caption metadata.

    Source: `liuhaotian/LLaVA-Pretrain` (license: see dataset card).
    Composition: 558K BLIP-style captions over CC3M+SBU+LAION subsets
    used as the LLaVA-1.5 stage-1 pretraining corpus.

    Schema (HF dataset card, verified by Pass 156c):
        {
          "id": str,
          "image": str,                       # filename relative to images.zip
          "conversations": [
            {"from": "human", "value": "<image>\\n<question>"},
            {"from": "gpt",   "value": "<caption>"}
          ]
        }

    The image bytes are NOT in the streamed dataset — they live in
    `images.zip` on the dataset card. Caller must extract that archive
    and pass its root via ``images_dir``. Each row's ``image`` filename
    is joined onto ``images_dir`` and the resulting path is verified
    on disk before the row is kept. Missing files are warned and
    skipped (per row).

    Args:
        max_samples: Hard cap on rows iterated from the stream.
        images_dir: Path to the extracted images.zip root.

    Returns:
        List of ``{"image": <abs_path>, "text": <caption>}`` dicts.

    Raises:
        FileNotFoundError: If ``images_dir`` does not exist on disk.
    """
    if images_dir is None:
        raise FileNotFoundError(
            "images_dir is required. Download images.zip from "
            "https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain "
            "and pass its extracted root via --images-dir."
        )
    images_dir = Path(images_dir)
    if not images_dir.exists():
        raise FileNotFoundError(
            f"images_dir does not exist: {images_dir}. Download and "
            "extract images.zip from the LLaVA-Pretrain dataset card."
        )

    if not _ensure_datasets():
        return []

    from datasets import load_dataset

    logger.info(f"Downloading LLaVA-Pretrain captions (max {max_samples:,} samples)...")
    try:
        ds = load_dataset(
            "liuhaotian/LLaVA-Pretrain",
            split="train",
            streaming=True,
        )
    except Exception as exc:
        logger.error("Failed to load LLaVA-Pretrain: %s", exc)
        return []

    pairs: list[dict] = []
    seen = 0
    skipped_missing = 0
    for item in ds:
        if seen >= max_samples:
            break
        seen += 1

        image_rel = item.get("image")
        conversations = item.get("conversations") or []
        gpt_turn = next((t for t in conversations if t.get("from") == "gpt"), None)
        if not image_rel or not gpt_turn:
            continue

        caption = (gpt_turn.get("value") or "").strip()
        if not caption or len(caption) < 5:
            continue

        img_path = (images_dir / image_rel).resolve()
        if not img_path.exists():
            skipped_missing += 1
            if skipped_missing <= 5:
                logger.warning(
                    "Image not found, skipping: %s (extract images.zip into --images-dir)",
                    image_rel,
                )
            elif skipped_missing == 6:
                logger.warning(
                    "Further missing-image warnings suppressed; total missing will be reported at end.",
                )
            continue

        pairs.append({"image": str(img_path), "text": caption})

        if seen % 10000 == 0:
            logger.info(f"  LLaVA-Pretrain: {seen:,} processed, {len(pairs):,} kept, {skipped_missing:,} missing...")

    pairs = _dedup_pairs(pairs)
    if skipped_missing:
        logger.warning(
            "LLaVA-Pretrain: %d rows skipped (image file not on disk)",
            skipped_missing,
        )
    logger.info(f"LLaVA-Pretrain: {len(pairs)} image-caption pairs extracted")
    return pairs


# ── Stats ─────────────────────────────────────────────────────────


def show_stats(output_dir: Path) -> None:
    """Show counts and sizes of collected vision JSONL files."""
    if not output_dir.exists():
        logger.info("No vision data directory yet at %s", output_dir)
        return
    total_rows = 0
    total_bytes = 0
    rows_for: list[tuple[str, int, int]] = []
    for jsonl in sorted(output_dir.glob("*.jsonl")):
        rows = 0
        with open(jsonl, encoding="utf-8") as f:
            for _ in f:
                rows += 1
        size = jsonl.stat().st_size
        rows_for.append((jsonl.name, rows, size))
        total_rows += rows
        total_bytes += size
    if not rows_for:
        logger.info("No vision JSONL files in %s", output_dir)
        return
    logger.info("Vision data summary (%s):", output_dir)
    for name, rows, size in rows_for:
        logger.info("  %s: %d rows, %.1f MB", name, rows, size / 1024 / 1024)
    logger.info("  TOTAL: %d rows, %.1f MB", total_rows, total_bytes / 1024 / 1024)


# ── CLI ───────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect vision training data for Enigma Engine")
    parser.add_argument(
        "--llava-pretrain",
        type=int,
        nargs="?",
        const=100000,
        help="Download LLaVA-Pretrain caption metadata (default: 100K samples). Requires --images-dir.",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="Path to extracted images.zip root. Required when "
        "fetching any vision source that uses external image bytes.",
    )
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR), help="Output directory for JSONL files")
    parser.add_argument("--stats", action="store_true", help="Show collected vision data statistics")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.stats:
        show_stats(output_dir)
        return

    if args.llava_pretrain is None:
        parser.print_help()
        return

    if not args.images_dir:
        parser.error(
            "--images-dir is required for --llava-pretrain. Download "
            "images.zip from "
            "https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain "
            "and pass its extracted root."
        )

    start = time.monotonic()
    try:
        pairs = collect_llava_pretrain(
            max_samples=args.llava_pretrain,
            images_dir=Path(args.images_dir),
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return
    if pairs:
        out = output_dir / "llava_pretrain.jsonl"
        _write_jsonl(pairs, out)
        elapsed = time.monotonic() - start
        logger.info(f"Saved {len(pairs):,} pairs → {out} ({elapsed:.1f}s)")
    else:
        logger.warning("No pairs collected.")


if __name__ == "__main__":
    main()
