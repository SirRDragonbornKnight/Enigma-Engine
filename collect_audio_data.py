"""
Audio Training Data Collector for Enigma Engine
================================================
Downloads and formats speech-text pair datasets for the ears organ
(Phase 4.5 step 6 -- her own hearing; this collector was the missing
counterpart to collect_vision_data.py, noted in every audit since 07-13).

Sources:
  - LibriSpeech train-clean-100 (--librispeech)
        100 hours / ~28.5k utterances of read English audiobooks with
        transcripts (openslr.org/12, CC-BY 4.0). ~6.3 GB tar.gz, direct
        stdlib download -- no HF dependency. FLAC decode at TRAIN time
        needs soundfile (installed).

Output format:
  JSONL with {"audio": "<absolute-path>", "text": "<transcript>"} per line.
  The (audio, text) pair shape the audio-align trainer expects. NOTE
  (2026-07-18): that trainer does not exist yet -- Trainer.train_audio was
  deleted with the Forge stack and only the vision half was carved out
  (enigma_engine/training/vision_align.py). Mirror it for audio before this
  data can be aligned; distill_audio_encoder.py already consumes this file.

Usage:
  python collect_audio_data.py --librispeech            # download + extract + jsonl
  python collect_audio_data.py --librispeech --skip-download   # re-emit jsonl only
"""

import argparse
import json
import logging
import tarfile
import time
import urllib.request
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/audio")
LIBRISPEECH_URL = "https://www.openslr.org/resources/12/train-clean-100.tar.gz"
SUBSET = "train-clean-100"


def _download(url: str, dest: Path) -> None:
    """Stdlib streaming download with a coarse progress line."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    start = time.monotonic()
    last_pct = -1

    def _hook(blocks: int, block_size: int, total: int) -> None:
        nonlocal last_pct
        if total <= 0:
            return
        pct = min(100, blocks * block_size * 100 // total)
        if pct >= last_pct + 5:
            mb = blocks * block_size / 1e6
            rate = mb / max(1e-9, time.monotonic() - start)
            logger.info(f"  {pct}% ({mb:.0f} MB, {rate:.1f} MB/s)")
            last_pct = pct

    logger.info(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, tmp, reporthook=_hook)
    tmp.replace(dest)
    logger.info(f"Download complete: {dest.stat().st_size / 1e9:.2f} GB")


def collect_librispeech(output_dir: Path, skip_download: bool = False) -> int:
    """Download/extract train-clean-100 and emit librispeech.jsonl.

    LibriSpeech layout: LibriSpeech/<subset>/<speaker>/<chapter>/
      <speaker>-<chapter>.trans.txt   ("<utt-id> <TRANSCRIPT IN CAPS>")
      <utt-id>.flac
    """
    tar_path = output_dir / "train-clean-100.tar.gz"
    root = output_dir / "LibriSpeech" / SUBSET

    if not root.exists():
        if not tar_path.exists():
            if skip_download:
                raise SystemExit(f"--skip-download but neither {root} nor {tar_path} exists")
            _download(LIBRISPEECH_URL, tar_path)
        logger.info(f"Extracting {tar_path} (~28.5k FLAC files) ...")
        with tarfile.open(tar_path) as tf:
            tf.extractall(output_dir, filter="data")
        logger.info(f"Extracted to {output_dir / 'LibriSpeech'}")
    else:
        logger.info(f"{root} already extracted; skipping download")

    pairs = 0
    missing = 0
    out_path = output_dir / "librispeech.jsonl"
    with open(out_path, "w", encoding="utf-8") as out:
        for trans in sorted(root.rglob("*.trans.txt")):
            for line in trans.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                utt_id, _, text = line.partition(" ")
                flac = trans.parent / f"{utt_id}.flac"
                if not flac.exists():
                    missing += 1
                    continue
                # Transcripts ship ALL-CAPS; her text is normal prose.
                out.write(
                    json.dumps(
                        {"audio": str(flac.resolve()), "text": text.strip().capitalize()},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                pairs += 1
    if missing:
        logger.warning(f"{missing} transcript lines had no matching .flac (skipped)")
    logger.info(f"LibriSpeech: {pairs} audio-text pairs -> {out_path}")
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--librispeech", action="store_true", help="collect LibriSpeech train-clean-100")
    parser.add_argument("--skip-download", action="store_true", help="use the already-extracted tree")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    if not args.librispeech:
        parser.print_help()
        return
    collect_librispeech(Path(args.output_dir), skip_download=args.skip_download)


if __name__ == "__main__":
    main()
