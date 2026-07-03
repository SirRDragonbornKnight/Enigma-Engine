#!/usr/bin/env python
"""Pre-tokenize pretraining data to binary format for fast loading.

Reads source text files from data/pretrain/ subdirectories (same sources
as collect_pretraining_data.py), tokenizes them using the project tokenizer,
and writes a flat binary file of uint32 token IDs separated by EOS tokens.

Output:
    data/pretrain/tokens.bin   - Binary array of uint32 token IDs
    data/pretrain/tokens.json  - Metadata (vocab_size, token count, etc.)

This runs independently of training and doesn't touch combined.txt.
"""

import array
import hashlib
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Config — mirrors collect_pretraining_data.py
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent / "data" / "pretrain"
SOURCE_DIRS = [
    ("Wiki Dump", BASE_DIR / "wikipedia_dump"),
    ("Wikipedia", BASE_DIR / "wikipedia"),
    ("Simple Wiki", BASE_DIR / "simple_wiki"),
    ("Gutenberg", BASE_DIR / "gutenberg"),
    ("FineWeb-Edu", BASE_DIR / "fineweb_edu"),
    ("OpenWebText", BASE_DIR / "openwebtext"),
    ("C4", BASE_DIR / "c4"),
    ("Wayback", BASE_DIR / "wayback"),
    ("Fandom", BASE_DIR / "fandom"),
]
SE_DIR = BASE_DIR / "stackexchange"

MIN_PARAGRAPH_LENGTH = 50
MAX_DEDUP_ENTRIES = 50_000_000

OUTPUT_BIN = BASE_DIR / "tokens.bin"
OUTPUT_META = BASE_DIR / "tokens.json"
HEADER_SIZE = 256  # Reserved header bytes


def main():
    # Project root on sys.path so imports work
    root = Path(__file__).parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from enigma_engine.core.tokenizer import get_tokenizer

    print("=" * 60)
    print("  Pre-tokenize pretraining data to binary")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load tokenizer (same one training uses)
    # ------------------------------------------------------------------
    print("\nLoading tokenizer...")
    tokenizer = get_tokenizer()
    vocab_size = getattr(tokenizer, "vocab_size", 0)
    eos_id = getattr(tokenizer, "eos_token_id", 2)
    tok_name = type(tokenizer).__name__
    print(f"  Tokenizer: {tok_name}  (vocab={vocab_size:,}, eos={eos_id})")

    # ------------------------------------------------------------------
    # Build source directory list
    # ------------------------------------------------------------------
    source_dirs = list(SOURCE_DIRS)
    if SE_DIR.exists():
        for sub in sorted(SE_DIR.iterdir()):
            if sub.is_dir():
                source_dirs.append((f"SE/{sub.name}", sub))

    # ------------------------------------------------------------------
    # Dedup state (paragraph-level, same logic as combine_all_sources)
    # ------------------------------------------------------------------
    seen_hashes: set[bytes] = set()
    dedup_warned = False
    dupes_skipped = 0

    # ------------------------------------------------------------------
    # Counters
    # ------------------------------------------------------------------
    total_tokens = 0
    total_docs = 0
    total_files = 0
    start_time = time.monotonic()

    # ------------------------------------------------------------------
    # Write binary tokens
    # ------------------------------------------------------------------
    tmp_file = OUTPUT_BIN.with_suffix(".bin.tmp")

    try:
        with open(tmp_file, "wb") as out:
            # Reserve header space — filled at the end
            out.write(b"\x00" * HEADER_SIZE)

            for label, source_dir in source_dirs:
                if not source_dir.exists():
                    continue

                dir_files = 0
                dir_tokens = 0

                try:
                    with os.scandir(source_dir) as scanner:
                        for entry in scanner:
                            if not entry.is_file(follow_symlinks=False):
                                continue
                            if not entry.name.endswith(".txt"):
                                continue

                            try:
                                text = Path(entry.path).read_text(encoding="utf-8", errors="replace")
                            except OSError:
                                continue
                            if len(text.strip()) < MIN_PARAGRAPH_LENGTH:
                                continue

                            # Paragraph-level dedup
                            paragraphs = text.split("\n\n")
                            unique_paras: list[str] = []
                            for para in paragraphs:
                                para = para.strip()
                                if len(para) < MIN_PARAGRAPH_LENGTH:
                                    unique_paras.append(para)
                                    continue
                                h = hashlib.sha256(para.encode("utf-8")).digest()[:8]
                                if h in seen_hashes:
                                    dupes_skipped += 1
                                    continue
                                if len(seen_hashes) < MAX_DEDUP_ENTRIES:
                                    seen_hashes.add(h)
                                elif not dedup_warned:
                                    dedup_warned = True
                                    print(f"  WARNING: Dedup table at capacity ({MAX_DEDUP_ENTRIES:,})")
                                unique_paras.append(para)

                            cleaned = "\n\n".join(unique_paras).strip()
                            if len(cleaned) < MIN_PARAGRAPH_LENGTH:
                                continue

                            # Tokenize
                            tokens = tokenizer.encode(cleaned)
                            if len(tokens) < 5:
                                continue

                            # Append EOS as document separator
                            tokens.append(eos_id)

                            # Write uint32 little-endian via array module
                            arr = array.array("I", tokens)
                            arr.tofile(out)

                            n = len(tokens)
                            dir_tokens += n
                            total_tokens += n
                            total_docs += 1
                            dir_files += 1
                            total_files += 1

                            if dir_files % 50_000 == 0:
                                elapsed = time.monotonic() - start_time
                                rate = total_tokens / elapsed if elapsed > 0 else 0
                                gb = (total_tokens * 4) / (1024**3)
                                print(
                                    f"  [{label}] {dir_files:,} files | "
                                    f"{total_tokens:,} tok | "
                                    f"{rate:,.0f} tok/s | {gb:.2f} GB"
                                )
                except OSError:
                    continue

                if dir_files > 0:
                    gb = (total_tokens * 4) / (1024**3)
                    print(f"  [{label}] {dir_files:,} files -> {dir_tokens:,} tokens  ({gb:.2f} GB cumulative)")

            # ----------------------------------------------------------
            # Write header now that we know totals
            # ----------------------------------------------------------
            out.seek(0)
            # Simple fixed header: magic(4) + version(4) + bpt(4) +
            # total_tokens(8) + vocab_size(4) + eos_id(4) = 28 bytes
            import struct

            header = struct.pack(
                "<4sIIQII",
                b"ETOK",  # Magic bytes
                1,  # Version
                4,  # Bytes per token (uint32)
                total_tokens,  # Total token count
                vocab_size,  # Vocab size
                eos_id,  # EOS token ID
            )
            out.write(header)

        # Atomic rename — only replaces output after full write succeeds
        tmp_file.replace(OUTPUT_BIN)

    except BaseException:
        try:
            if tmp_file.exists():
                tmp_file.unlink()
        except OSError:
            pass
        raise

    # ------------------------------------------------------------------
    # Write metadata JSON
    # ------------------------------------------------------------------
    elapsed = time.monotonic() - start_time
    file_gb = (total_tokens * 4) / (1024**3)

    meta = {
        "format": "ETOK",
        "version": 1,
        "dtype": "uint32",
        "header_size": HEADER_SIZE,
        "total_tokens": total_tokens,
        "total_documents": total_docs,
        "total_files": total_files,
        "vocab_size": vocab_size,
        "eos_token_id": eos_id,
        "tokenizer": tok_name,
        "dupes_skipped": dupes_skipped,
        "file_size_gb": round(file_gb, 2),
        "elapsed_seconds": round(elapsed, 1),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    OUTPUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("  Done!")
    print(f"  Tokens:    {total_tokens:,}")
    print(f"  Documents: {total_docs:,}")
    print(f"  Files:     {total_files:,}")
    print(f"  Dupes:     {dupes_skipped:,} paragraphs skipped")
    print(f"  Output:    {OUTPUT_BIN} ({file_gb:.2f} GB)")
    print(f"  Metadata:  {OUTPUT_META}")
    print(f"  Time:      {elapsed / 60:.1f} min")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
