#!/usr/bin/env python
"""Pre-tokenize pretraining data to binary format for fast loading.

Reads source text files from data/pretrain/ subdirectories (same sources
as collect_pretraining_data.py), tokenizes them using the project tokenizer,
and writes a flat binary file of token IDs. tokenizer.encode() adds
BOS and EOS (add_special_tokens defaults True) and one more EOS is appended
per document, so the on-disk layout for each document is:
    <bos> ...content... <eos> <eos>
This matches the existing tokens.bin corpus and pretrain lineage exactly.

The v1 lineage corpus (data/pretrain/tokens.bin) is COMPLETE and sacred:
this script now refuses to write to that path at all -- pretraining on it
finished 2026-07-03, and since the Curated source joined SOURCE_DIRS
(2026-07-25) a rebuild could not be byte-identical to the lineage anyway
(a new source shifts the stream AND wins dedup collisions). Every new
corpus names its own --output-bin; the v2 retokenize (TOKENIZER_V2_SPEC):

    python pretokenize_data.py --vocab enigma_engine/vocab_model/bpe_vocab_v2_16k.json ^
        --output-bin data/pretrain/tokens_v2.bin --dtype uint16 --workers 10

Parallel layout: the PARENT does the walk + paragraph dedup + filters --
that state is inherently sequential (shared seen_hashes) -- with file
reads PREFETCHED by a thread pool (see iter_cleaned_docs: per-file open
latency, not CPU, was the wall on the first v2 attempt), and WORKERS do
the encoding (Rust v2 backend when built: measured 17.8 MB/s/worker vs
~0.1 for pure Python). Ordered imap keeps the output stream
byte-deterministic (= the sequential walk order). Launch the whole
script at BelowNormal priority for the CRD budget (workers also self-set
it); default 10 workers leaves ~6 cores for the desktop session.

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
    # The curated shard (make_pretrain_curated.py): must-know facts, her own
    # identity as prose, short dialogue. Walked FIRST, for two reasons the
    # fix-arc audits paid for (both 2026-07-25):
    # * position decides VAL membership: val is carved off the very END of
    #   the bin, so the round-7 tail-position design handed the whole shard
    #   to val, and "not last" was still one absent stackexchange dir away
    #   from last on a fresh checkout. First can never touch the tail.
    # * position decides DEDUP precedence: the paragraph dedup is first-wins,
    #   so walked 13th the shard silently LOST every paragraph a web source
    #   happened to share -- before any --repeat-sources multiplied what was
    #   left. First means her must-know text wins the collision.
    # Oversample with --repeat-sources; pretrain re-checks placement at boot
    # from the recorded extents either way.
    ("Curated", BASE_DIR / "curated"),
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
    """The walk order, which IS the order of the token stream.

    There is deliberately no knob to reorder it. A --tail-sources flag existed
    for one uncommitted day so the anneal could read a shard "at the end" --
    and the end of the bin is exactly what pretrain carves off as val, so the
    shard it moved would have been held out, not oversampled. Position in the
    corpus buys nothing the sampler can see; --repeat-sources is the honest
    oversample."""
    source_dirs = list(SOURCE_DIRS)
    if SE_DIR.exists():
        for sub in sorted(SE_DIR.iterdir()):
            if sub.is_dir():
                source_dirs.append((f"SE/{sub.name}", sub))
    return source_dirs


def parse_repeat_sources(spec: str, source_dirs: list[tuple[str, Path]]) -> dict[str, int]:
    """'curated=5' -> {'Curated': 5}, validated against the actual walk.

    Copying the shard on disk was the obvious route and it does not survive
    this script: the paragraph dedup is GLOBAL, so replicated files collapse
    right back to one copy at tokenize time -- silently, and only for
    paragraphs long enough to be hashed, which skews as well as shrinks.
    Repetition has to happen on the far side of the dedup; this knob is
    that."""
    if not spec:
        return {}
    by_key = {label.lower(): label for label, _p in source_dirs}
    by_key.update({p.name.lower(): label for label, p in source_dirs})
    out: dict[str, int] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, _eq, count = part.partition("=")
        label = by_key.get(name.strip().lower())
        if label is None:
            raise SystemExit(
                f"--repeat-sources names no such source: {name.strip()!r} "
                f"(known: {', '.join(sorted(p.name for _l, p in source_dirs))})"
            )
        try:
            r = int(count)
        except ValueError:
            r = -1
        if not 2 <= r <= 50:
            raise SystemExit(
                f"--repeat-sources {part!r}: count must be a whole number in [2, 50] "
                f"(one copy is what you get without the flag)"
            )
        out[label] = r
    return out


# The repeat cache holds a source's whole cleaned pass-1 text in RAM. That is
# the point (byte-identical passes on the far side of the dedup) and it is
# only sane for the curated-shard class: someone aiming the flag at a web
# source (fineweb_edu is ~95 GB of text on this box) would OOM the parent
# hours into the walk, and the BaseException handler would then delete the
# whole .tmp. Refuse early instead, at a bound generous for any honest shard.
REPEAT_CACHE_MAX_BYTES = 2 * 1024**3


class RepeatCacheExceeded(RuntimeError):
    """The repeat-cache refusal, raised INSIDE the doc stream.

    Its class is load-bearing twice over: it must be a plain Exception (a
    SystemExit raised here is swallowed by pool.imap's task-feeder thread,
    whose guard catches Exception only, and the run HUNG forever instead of
    refusing -- round-B audit, 2026-07-25), and it must be its OWN type so
    main() can convert exactly this refusal to a clean SystemExit without
    also dressing up a genuine worker crash (RuntimeError) as one."""


def with_repeats(docs, repeats: dict[str, int]):
    """Re-emit a repeated source's cleaned docs after its first pass.

    Runs on the dedup walk's OUTPUT, so every copy is byte-identical to the
    pass-1 text and the global paragraph dedup cannot touch it. Sources are
    contiguous in walk order, so copies of a doc land a whole SOURCE apart in
    the stream. NOTE the honest limit its own audit measured (2026-07-25):
    that spacing is only wider than a training window when the source itself
    is -- a pass shorter than --block still puts copies in one window, which
    is why pretrain refuses a repeated source whose per-pass span does not
    exceed its block."""
    cache: list[str] = []
    cache_bytes = 0
    prev: str | None = None

    def flush(label):
        for _ in range(repeats[label] - 1):
            for text in cache:
                yield label, text

    for label, cleaned in docs:
        if label != prev and cache:
            yield from flush(prev)
            cache.clear()
            cache_bytes = 0
        prev = label
        if repeats.get(label, 1) > 1:
            cache.append(cleaned)
            cache_bytes += len(cleaned)
            if cache_bytes > REPEAT_CACHE_MAX_BYTES:
                raise RepeatCacheExceeded(
                    f"--repeat-sources: source '{label}' exceeds the "
                    f"{REPEAT_CACHE_MAX_BYTES / 1024**3:.0f} GB repeat cache -- this "
                    "flag is for the curated-shard class, not web-scale sources "
                    "(the whole source is held in RAM for the repeat passes)"
                )
        yield label, cleaned
    if cache:
        yield from flush(prev)


def _read_txt(path: str) -> str | None:
    """One file read for the prefetch pool; None = unreadable (skip)."""
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def iter_cleaned_docs(source_dirs, read_threads: int = 16, read_ahead: int = 64):
    """Walk + filter + paragraph-dedup, EXACT v1 semantics; yields
    (label, cleaned_text). The shared dedup set makes this inherently
    sequential -- it stays in the parent by design.

    File READS are prefetched by a thread pool but consumed STRICTLY in
    walk order, so dedup and the output stream stay byte-deterministic.
    The prefetch is load-bearing, not a nicety: per-file open latency on
    NTFS+Defender (~4 ms) serialized the first v2 run attempt to ~1 MB/s
    with ten encode workers sitting 98% idle -- the walk itself was the
    entire 25-hour wall."""
    from concurrent.futures import ThreadPoolExecutor

    seen_hashes: set[bytes] = set()
    dedup_warned = False
    stats = iter_cleaned_docs.stats = {"dupes_skipped": 0}

    def walk():
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
                        yield label, entry.path
            except OSError:
                continue

    with ThreadPoolExecutor(max_workers=read_threads) as pool:
        pending: deque = deque()
        walker = walk()
        exhausted = False
        while pending or not exhausted:
            while not exhausted and len(pending) < read_ahead:
                try:
                    label, path = next(walker)
                except StopIteration:
                    exhausted = True
                    break
                pending.append((label, pool.submit(_read_txt, path)))
            if not pending:
                break
            label, fut = pending.popleft()
            text = fut.result()
            if text is None or len(text.strip()) < MIN_PARAGRAPH_LENGTH:
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
    ap.add_argument(
        "--repeat-sources",
        default="",
        help="comma-separated name=count, e.g. 'curated=5': emit that source's cleaned "
        "docs count times so the corpus itself carries the oversample (on-disk copies "
        "would be collapsed by the global paragraph dedup). Meant for the curated-shard "
        "class: the whole source is cached in RAM for the repeat passes. Default: every "
        "source once",
    )
    args = ap.parse_args()

    out_bin = Path(args.output_bin)
    out_meta = out_bin.with_suffix(".json")
    typecode = DTYPES[args.dtype]
    bpt = array.array(typecode).itemsize

    # The v1 lineage corpus is sacred AND finished (pretraining done
    # 2026-07-03). The old guard only refused a custom vocab/dtype here, so a
    # bare no-argument run could still overwrite the lineage bin -- and since
    # Curated joined SOURCE_DIRS the rebuild would differ from the lineage
    # regardless (new source, and dedup is first-wins). Refuse the path
    # outright; every corpus this script writes from now on is a NEW one.
    def _canon(p: Path) -> str:
        # Identity, not string equality: resolve() keeps a \\?\ extended-
        # length prefix verbatim, so `--output-bin \\?\...\tokens.bin` failed
        # the == while the OS opened the identical file, defeating both
        # refusals below (round-C audit, 2026-07-25). normcase folds the
        # case/separator dimensions the same way the filesystem does.
        s = str(Path(p).resolve())
        if s.startswith("\\\\?\\UNC\\"):
            s = "\\\\" + s[8:]
        elif s.startswith("\\\\?\\"):
            s = s[4:]
        return os.path.normcase(s)

    if _canon(out_bin) == _canon(OUTPUT_BIN):
        raise SystemExit(
            "refusing to overwrite the v1 lineage tokens.bin (pretraining on it is "
            "COMPLETE and the source walk has changed since it was built) -- name a "
            "new corpus with --output-bin"
        )
    # The SIDECAR needs the same protection: metadata lands at
    # with_suffix(".json"), so `--output-bin data/pretrain/tokens.bin2` (any
    # tokens.<ext> there) passes the bin refusal above and then clobbers
    # tokens.json -- the lineage receipt, which is gitignored and therefore
    # unrecoverable (round-B audit, 2026-07-25).
    if _canon(out_meta) == _canon(OUTPUT_BIN.with_suffix(".json")):
        raise SystemExit(
            "refusing to overwrite the v1 lineage sidecar tokens.json -- this "
            "--output-bin maps its metadata onto the lineage receipt; pick a bin "
            "name whose .json lands elsewhere"
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
    repeats = parse_repeat_sources(args.repeat_sources, source_dirs)
    for label, r in repeats.items():
        print(f"  Repeat: {label} x{r} (copies emitted after the source's own pass)", flush=True)

    # Spec requirement: NEVER skip an absent source silently -- a quietly
    # smaller corpus is a quality regression nothing downstream can catch.
    absent = [label for label, d in source_dirs if not d.exists()]
    if absent:
        print(f"  WARNING: {len(absent)} source dirs ABSENT and skipped: {', '.join(absent)}", flush=True)
    repeat_absent = sorted(set(repeats) & set(absent))
    if repeat_absent:
        # A warning is enough for a source that merely shrinks the corpus; a
        # source someone explicitly asked to OVERSAMPLE cannot be absent by
        # intent, and its silent loss is the exact failure the repeat exists
        # to prevent.
        raise SystemExit(
            f"--repeat-sources names absent source dir(s): {', '.join(repeat_absent)}"
        )

    total_tokens = 0
    total_docs = 0
    per_label: dict[str, list[int]] = {}
    # Token extents per source, [start, end) in stream token indices -- the
    # same index space pretrain memmaps. This is what lets pretrain refuse a
    # repeated source whose copies extend into the held-out val tail instead
    # of trusting walk order across two scripts.
    extents: dict[str, list[int]] = {}
    start_time = time.monotonic()
    tmp_file = out_bin.with_suffix(".bin.tmp")

    def result_stream():
        """(label, packed, n) triples in deterministic walk order."""
        docs = iter_cleaned_docs(source_dirs)
        if repeats:
            docs = with_repeats(docs, repeats)
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
                ext = extents.get(label)
                if ext is None:
                    extents[label] = [total_tokens, total_tokens + n]
                else:
                    ext[1] = total_tokens + n
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

        # A declared oversample must have HAPPENED. The absent-dir check
        # catches a missing directory and nothing else: a curated dir with no
        # .txt files, or one whose every paragraph an earlier source already
        # published (first-wins dedup), emitted zero tokens while the sidecar
        # recorded the multiplier as fact -- the fix-arc audit produced a
        # stream byte-identical to a run with no curated source at all, under
        # a meta claiming x5 (2026-07-25). Refuse before the rename; the
        # except-handler below removes the .tmp.
        unrepresented = sorted(
            label for label in repeats
            if (extents.get(label) or [0, 0])[1] <= (extents.get(label) or [0, 0])[0]
        )
        if unrepresented:
            raise SystemExit(
                f"repeated source(s) emitted no tokens: {', '.join(unrepresented)} -- "
                "the declared oversample never happened (no .txt files, every doc "
                "under the 5-token floor, or every paragraph already published by an "
                "earlier source and deduped)"
            )

        # Atomic rename — only replaces output after full write succeeds
        tmp_file.replace(out_bin)

    except RepeatCacheExceeded as exc:
        # The one refusal raised inside the doc stream; it rides out as a
        # plain Exception so pool.imap's feeder thread propagates it instead
        # of dying silently, and becomes the clean exit it means here. A
        # bare RuntimeError (a genuine crash) stays a crash below.
        try:
            if tmp_file.exists():
                tmp_file.unlink()
        except OSError:
            pass
        raise SystemExit(str(exc)) from exc
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
        "repeated_sources": repeats,
        "source_token_extents": extents,
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
