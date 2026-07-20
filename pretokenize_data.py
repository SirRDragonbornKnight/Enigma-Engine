#!/usr/bin/env python
"""Pre-tokenize pretraining data to binary format for fast loading.

Reads source text files from data/pretrain/ subdirectories (same sources
as collect_pretraining_data.py), tokenizes them using the project tokenizer,
and writes a flat binary file of token IDs. tokenizer.encode() adds
BOS and EOS (add_special_tokens defaults True) and one more EOS is appended
per document, so the on-disk layout for each document is:
    <bos> ...content... <eos> <eos>
This matches the existing tokens.bin corpus and pretrain lineage exactly.

Defaults reproduce the v1 lineage byte-for-byte: live vocab, uint32,
data/pretrain/tokens.bin, single process. The v2 retokenize
(TOKENIZER_V2_SPEC) parameterizes all four:

    python pretokenize_data.py --vocab enigma_engine/vocab_model/bpe_vocab_v2_16k.json ^
        --output-bin data/pretrain/tokens_v2.bin --dtype uint16 --workers 10

Parallel layout (measured 4.42 -> 37.3 MB/s at 12 workers, spec table):
the PARENT does the walk + paragraph dedup + filters -- that state is
inherently sequential (shared seen_hashes) and is microseconds per doc
against BPE encode at ~4.4 MB/s -- and WORKERS do the encoding. Ordered
imap keeps the output stream byte-deterministic (= the sequential walk
order). Launch the whole script at BelowNormal priority for the CRD
budget (children inherit the class); default 10 workers leaves ~6 cores
for the desktop session.

This runs independently of training and doesn't touch combined.txt.
"""

import argparse
import array
import hashlib
import json
import os
import struct
import sys
import time
from collections import deque
from pathlib import Path

# ---------------------------------------------------------------------------
# Config — mirrors collect_pretraining_data.py
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent
BASE_DIR = ROOT / "data" / "pretrain"
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
    ("DCLM", BASE_DIR / "dclm"),
    ("FineMath", BASE_DIR / "finemath"),
    ("The Stack", BASE_DIR / "the_stack"),
]
SE_DIR = BASE_DIR / "stackexchange"

MIN_PARAGRAPH_LENGTH = 50
MAX_DEDUP_ENTRIES = 50_000_000

OUTPUT_BIN = BASE_DIR / "tokens.bin"
OUTPUT_META = BASE_DIR / "tokens.json"
HEADER_SIZE = 256  # Reserved header bytes

# typecode only -- bytes-per-token derives from the array itemsize below so
# the header can never disagree with the bytes actually written (audit LOW-2).
DTYPES = {"uint32": "I", "uint16": "H"}


def build_source_dirs() -> list[tuple[str, Path]]:
    source_dirs = list(SOURCE_DIRS)
    if SE_DIR.exists():
        for sub in sorted(SE_DIR.iterdir()):
            if sub.is_dir():
                source_dirs.append((f"SE/{sub.name}", sub))
    return source_dirs


def iter_cleaned_docs(source_dirs):
    """Walk + filter + paragraph-dedup, EXACT v1 semantics; yields
    (label, cleaned_text). The shared dedup set makes this inherently
    sequential -- it stays in the parent by design."""
    seen_hashes: set[bytes] = set()
    dedup_warned = False
    stats = iter_cleaned_docs.stats = {"dupes_skipped": 0}

    for label, source_dir in source_dirs:
        if not source_dir.exists():
            continue
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

                    paragraphs = text.split("\n\n")
                    unique_paras: list[str] = []
                    for para in paragraphs:
                        para = para.strip()
                        if len(para) < MIN_PARAGRAPH_LENGTH:
                            unique_paras.append(para)
                            continue
                        h = hashlib.sha256(para.encode("utf-8")).digest()[:8]
                        if h in seen_hashes:
                            stats["dupes_skipped"] += 1
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
                    yield label, cleaned
        except OSError:
            continue


# ---------------------------------------------------------------------------
# Worker side: encode one cleaned doc to packed bytes.
# Module-level (not nested) so Windows spawn can pickle them; state lives in
# process-globals set by the initializer -- one tokenizer load per worker.
# ---------------------------------------------------------------------------

_TOK = None
_TYPECODE = "I"
_VOCAB = 0
_EOS = 2


def _worker_init(
    repo_root: str, vocab_path, typecode: str, vocab_size: int, eos_id: int, nice: bool = False
) -> None:
    global _TOK, _TYPECODE, _VOCAB, _EOS
    if nice and sys.platform == "win32":
        # Self-enforce BelowNormal in every spawned worker (CRD budget --
        # spec requirement), independent of how the parent was launched.
        # 0x4000 = BELOW_NORMAL_PRIORITY_CLASS; GetCurrentProcess() is the
        # pseudo-handle, no cleanup needed.
        import ctypes

        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from enigma_engine.core.tokenizer import get_tokenizer

    _TOK = get_tokenizer("bpe", vocab_path=vocab_path) if vocab_path else get_tokenizer("bpe")
    _TYPECODE, _VOCAB, _EOS = typecode, vocab_size, eos_id


def _encode_doc(cleaned: str):
    """(packed_bytes, n_tokens) for one document, or None to skip (<5 tokens).

    Packs <bos> ...content... <eos> <eos> as native little-endian
    (x86; matches v1's array.tofile). Bounds guard raises so a
    vocab/tokenizer mismatch kills the run instead of writing a
    poisoned corpus (train-time CUDA device-side assert class).
    """
    tokens = _TOK.encode(cleaned)
    if len(tokens) < 5:
        return None
    if _VOCAB > 0:
        bad = max(tokens)
        if bad >= _VOCAB or min(tokens) < 0:
            raise ValueError(f"tokenizer emitted id {bad} outside [0, {_VOCAB}) -- vocab/tokenizer mismatch")
    tokens.append(_EOS)
    return array.array(_TYPECODE, tokens).tobytes(), len(tokens)


def main():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    ap = argparse.ArgumentParser(description="Pre-tokenize the pretrain corpus to an ETOK binary")
    ap.add_argument("--vocab", default=None, help="tokenizer vocab file (default: the live bpe_vocab.json)")
    ap.add_argument("--output-bin", default=str(OUTPUT_BIN), help="output .bin (metadata lands beside it as .json)")
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="uint32", help="token width (uint16 needs vocab <= 65,536)")
    ap.add_argument("--workers", type=int, default=1, help="encode processes; 10 = CRD-safe parallel, 1 = legacy sequential")
    args = ap.parse_args()

    out_bin = Path(args.output_bin)
    out_meta = out_bin.with_suffix(".json")
    typecode = DTYPES[args.dtype]
    bpt = array.array(typecode).itemsize

    # The v1 lineage corpus is sacred: refuse to aim a NON-default vocab or
    # dtype at the default tokens.bin path (same protection class as
    # train_tokenizer's explicit-output rule, LOW-9).
    if out_bin.resolve() == OUTPUT_BIN.resolve() and (args.vocab or args.dtype != "uint32"):
        raise SystemExit(
            "refusing to overwrite the lineage tokens.bin with a custom vocab/dtype -- pass --output-bin"
        )

    from enigma_engine.core.tokenizer import get_tokenizer

    print("=" * 60)
    print("  Pre-tokenize pretraining data to binary")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load tokenizer (same one training uses) for metadata + guards.
    # 'bpe' -- never 'auto', which prefers tiktoken when installed and would
    # silently write a cl100k corpus incompatible with the bpe pipeline.
    # ------------------------------------------------------------------
    print("\nLoading tokenizer...")
    tokenizer = get_tokenizer("bpe", vocab_path=args.vocab) if args.vocab else get_tokenizer("bpe")
    vocab_size = getattr(tokenizer, "vocab_size", 0)
    eos_id = getattr(tokenizer, "eos_token_id", 2)
    tok_name = type(tokenizer).__name__
    pretok = getattr(tokenizer, "pretokenizer_version", "v1")
    print(f"  Tokenizer: {tok_name}  (vocab={vocab_size:,}, eos={eos_id}, pretokenizer={pretok})")
    if vocab_size > 0 and not (0 <= eos_id < vocab_size):
        raise SystemExit(f"eos_token_id {eos_id} outside [0, {vocab_size}) -- refusing to write {out_bin.name}")
    if bpt == 2 and vocab_size > 65536:
        raise SystemExit(f"--dtype uint16 with vocab {vocab_size:,} > 65,536 -- ids would overflow")

    source_dirs = build_source_dirs()

    # Spec requirement: NEVER skip an absent source silently -- a quietly
    # smaller corpus is a quality regression nothing downstream can catch.
    absent = [label for label, d in source_dirs if not d.exists()]
    if absent:
        print(f"  WARNING: {len(absent)} source dirs ABSENT and skipped: {', '.join(absent)}", flush=True)

    total_tokens = 0
    total_docs = 0
    per_label: dict[str, list[int]] = {}
    start_time = time.monotonic()
    tmp_file = out_bin.with_suffix(".bin.tmp")

    def result_stream():
        """(label, packed, n) triples in deterministic walk order."""
        docs = iter_cleaned_docs(source_dirs)
        if args.workers <= 1:
            _worker_init(str(ROOT), args.vocab, typecode, vocab_size, eos_id)
            for label, cleaned in docs:
                res = _encode_doc(cleaned)
                if res is not None:
                    yield label, res[0], res[1]
            return
        import multiprocessing as mp

        labels: deque[str] = deque()

        def texts():
            for label, cleaned in docs:
                labels.append(label)
                yield cleaned

        ctx = mp.get_context("spawn")
        with ctx.Pool(
            args.workers,
            initializer=_worker_init,
            initargs=(str(ROOT), args.vocab, typecode, vocab_size, eos_id, True),
        ) as pool:
            # ordered imap: results come back in submission order, so the
            # label deque pops in lockstep and the output stream is
            # byte-identical to a sequential run.
            for res in pool.imap(_encode_doc, texts(), chunksize=32):
                label = labels.popleft()
                if res is not None:
                    yield label, res[0], res[1]

    try:
        with open(tmp_file, "wb") as out:
            out.write(b"\x00" * HEADER_SIZE)

            for label, packed, n in result_stream():
                out.write(packed)
                total_tokens += n
                total_docs += 1
                cnt = per_label.setdefault(label, [0, 0])
                cnt[0] += 1
                cnt[1] += n

                if total_docs % 50_000 == 0:
                    elapsed = time.monotonic() - start_time
                    rate = total_tokens / elapsed if elapsed > 0 else 0
                    gb = (total_tokens * bpt) / (1024**3)
                    print(
                        f"  [{label}] {total_docs:,} docs | {total_tokens:,} tok | {rate:,.0f} tok/s | {gb:.2f} GB",
                        flush=True,
                    )

            out.seek(0)
            header = struct.pack(
                "<4sIIQII",
                b"ETOK",  # Magic bytes
                1,  # Version
                bpt,  # Bytes per token
                total_tokens,  # Total token count
                vocab_size,  # Vocab size
                eos_id,  # EOS token ID
            )
            out.write(header)

        # Atomic rename — only replaces output after full write succeeds
        tmp_file.replace(out_bin)

    except BaseException:
        try:
            if tmp_file.exists():
                tmp_file.unlink()
        except OSError:
            pass
        raise

    dupes_skipped = getattr(iter_cleaned_docs, "stats", {}).get("dupes_skipped", 0)

    for label, (n_docs, n_tok) in per_label.items():
        print(f"  [{label}] {n_docs:,} docs -> {n_tok:,} tokens")

    elapsed = time.monotonic() - start_time
    file_gb = (total_tokens * bpt) / (1024**3)

    meta = {
        "format": "ETOK",
        "version": 1,
        "dtype": args.dtype,
        "header_size": HEADER_SIZE,
        "total_tokens": total_tokens,
        "total_documents": total_docs,
        "total_files": total_docs,
        "vocab_size": vocab_size,
        "eos_token_id": eos_id,
        "tokenizer": tok_name,
        "pretokenizer": pretok,
        "vocab_file": str(args.vocab) if args.vocab else "enigma_engine/vocab_model/bpe_vocab.json",
        "workers": args.workers,
        "sources_absent": absent,
        "dupes_skipped": dupes_skipped,
        "file_size_gb": round(file_gb, 2),
        "elapsed_seconds": round(elapsed, 1),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\n{'=' * 60}")
    print("  Done!")
    print(f"  Tokens:    {total_tokens:,}")
    print(f"  Documents: {total_docs:,}")
    print(f"  Dupes:     {dupes_skipped:,} paragraphs skipped")
    print(f"  Output:    {out_bin} ({file_gb:.2f} GB, {args.dtype})")
    print(f"  Metadata:  {out_meta}")
    print(f"  Time:      {elapsed / 60:.1f} min")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
