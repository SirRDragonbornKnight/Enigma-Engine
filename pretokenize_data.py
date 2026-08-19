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
        --output-bin data/pretrain/tokens_v2c.bin --dtype uint16 --workers 10 ^
        --repeat-sources curated=5

(curated=5 is a standing ruling -- BACKLOG 7.95 T1; omitting the flag runs
with NO curated oversample and nothing refuses, so copy the line whole. The
output name is the NEXT free version: an existing bin is refused at boot,
because overwriting the live corpus or the rollback in place destroys the
receipt a failed build would need.)

A persona pack's curated shard (make_pretrain_curated.py --persona <pack>
--out <dir>) rides the SAME walk: --curated-dir swaps the PATH under the
"Curated" label, and --only-curated walks that one source alone -- the pack
smoke shape. Neither reorders the walk (see build_source_dirs) and neither
touches a run that passes neither.

Parallel layout: the PARENT does the walk + paragraph dedup + filters --
that state is inherently sequential (shared dedup filter) -- with file
reads PREFETCHED by a thread pool (see iter_cleaned_docs: per-file open
latency, not CPU, was the wall on the first v2 attempt), and WORKERS do
the encoding (Rust v2 backend when built: measured 17.8 MB/s/worker;
pure Python measured 4.6-6.4 MB/s/worker on real corpus text, 2026-07-28
-- the python path is slower but is NOT the wall, the parent's walk is).
The sidecar records which backend ran. Ordered imap keeps the output stream
byte-deterministic (= the sequential walk order). Launch the whole
script at BelowNormal priority for the CRD budget (workers also self-set
it); default 10 workers leaves ~6 cores for the desktop session.

This runs independently of training and doesn't touch combined.txt.
"""

import argparse
import array
import hashlib
import json
import math
import os
import struct
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

from collect_pretraining_data import DOC_SEP_CHAR, _sanitize_special_literals

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
    # C4 and OpenWebText are OUT: DCLM replaces them -- the model-filtered
    # stream beats both on the research receipts, so the swap raises corpus
    # quality at roughly constant size. Their dirs are retired; this list is
    # what the corpus walks.
    ("Wayback", BASE_DIR / "wayback"),
    ("Fandom", BASE_DIR / "fandom"),
    ("DCLM", BASE_DIR / "dclm"),
    ("FineMath", BASE_DIR / "finemath"),
    ("The Stack", BASE_DIR / "the_stack"),
]
SE_DIR = BASE_DIR / "stackexchange"

MIN_PARAGRAPH_LENGTH = 50

# DOC_SEP_CHAR (imported above): one RECORD SEPARATOR (U+001E) between
# records inside a packed .txt file. The HF-streamed sources batch ~5 MB of
# records per file joined by "\n\n", indistinguishable from a paragraph
# break -- so the walk read one FILE as one document and <s>/</s> were
# learned almost entirely from the one-article-per-file sources (bos in the
# first 5M tokens of each extent, measured 2026-07-28: FineWeb-Edu 4,
# The Stack 3, DCLM 4, Wiki Dump 4,870). The collector writes it; files
# predating it contain none and still read as one document each.

# Paragraph-dedup membership. The 50M-entry exact set capped out mid-walk on
# v2b (sampled estimate 190-275M paragraphs), so every source past the fill
# was deduped only against what came before it -- and an exact set at that
# scale is measured 75 B/entry (2026-07-30): 19-28 GiB, a RAM gamble on a
# box that is also running 10 tokenize workers. A Bloom filter holds the
# whole walk in ~0.7 GiB: no false negatives (a stored paragraph is always
# found), and a false positive drops a unique paragraph at the design rate
# below -- 0.1% random loss against the measured multi-percent under-dedup
# it replaces.
DEDUP_CAPACITY = 400_000_000
DEDUP_FPR = 1e-3


class ParagraphBloom:
    """Bloom-filter membership for paragraph digests.

    Standard construction: m = -n*ln(p)/ln(2)^2 bits, k = (m/n)*ln(2)
    probes, double hashing (Kirsch-Mitzenmacher) from two independent
    64-bit halves of the paragraph's sha256.
    """

    def __init__(self, capacity: int = DEDUP_CAPACITY, fpr: float = DEDUP_FPR):
        m = int(-capacity * math.log(fpr) / (math.log(2) ** 2))
        self.n_bits = ((m + 63) // 64) * 64
        self.k = max(1, round((self.n_bits / capacity) * math.log(2)))
        self.words = np.zeros(self.n_bits // 64, dtype=np.uint64)
        self.capacity = capacity
        self.fpr = fpr
        self.adds = 0

    def seen_or_add(self, digests: "np.ndarray") -> "np.ndarray":
        """(N, 2) uint64 rows -> (N,) bool "was already present"; every row
        is added. uint64 wraparound in the probe arithmetic is fine -- the
        probes only need to stay well-distributed mod n_bits."""
        h1 = digests[:, 0][:, None]
        h2 = digests[:, 1][:, None]
        probes = np.arange(self.k, dtype=np.uint64)[None, :]
        idx = (h1 + probes * h2) % np.uint64(self.n_bits)
        word = (idx >> np.uint64(6)).astype(np.int64)
        bit = np.uint64(1) << (idx & np.uint64(63))
        present = ((self.words[word] & bit) != 0).all(axis=1)
        # bitwise_or.at, not words[word] |= bit: fancy-index assignment drops
        # all but one write when the same word index repeats in a batch.
        np.bitwise_or.at(self.words, word.ravel(), bit.ravel())
        self.adds += int((~present).sum())
        return present

OUTPUT_BIN = BASE_DIR / "tokens.bin"
OUTPUT_META = BASE_DIR / "tokens.json"
HEADER_SIZE = 256  # Reserved header bytes

# typecode only -- bytes-per-token derives from the array itemsize below so
# the header can never disagree with the bytes actually written (audit LOW-2).
DTYPES = {"uint32": "I", "uint16": "H"}


def build_source_dirs(curated_dir: str | Path | None = None,
                      only_curated: bool = False) -> list[tuple[str, Path]]:
    """The walk order, which IS the order of the token stream.

    There is deliberately no knob to reorder it. A --tail-sources flag existed
    for one uncommitted day so the anneal could read a shard "at the end" --
    and the end of the bin is exactly what pretrain carves off as val, so the
    shard it moved would have been held out, not oversampled. Position in the
    corpus buys nothing the sampler can see; --repeat-sources is the honest
    oversample.

    Two knobs change WHICH dirs are walked, never the order:
    * `curated_dir` replaces the PATH under the "Curated" label (a persona
      pack's shard from make_pretrain_curated.py --out). The label stays, so
      the shard is still walked FIRST and keeps both properties that position
      buys -- never in the val tail, and first-wins on dedup collisions -- and
      --repeat-sources still resolves it by the label AND by the new
      directory's own basename.
    * `only_curated` drops every other entry and the SE expansion, leaving
      that one source alone: tokenize exactly one pack shard.
    Defaults are the full hardcoded walk, byte for byte. Both read the module
    globals, so a monkeypatched SOURCE_DIRS/SE_DIR still composes."""
    source_dirs = list(SOURCE_DIRS)
    if curated_dir is not None:
        swapped = Path(curated_dir)
        source_dirs = [(label, swapped if label == "Curated" else path)
                       for label, path in source_dirs]
    if only_curated:
        return [(label, path) for label, path in source_dirs if label == "Curated"]
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
# source (fineweb_edu is 43 GB of text on this box, measured 2026-07-27)
# would OOM the parent hours into the walk, and the BaseException handler
# would then delete the whole .tmp. Refuse early instead, at a bound generous
# for any honest shard.
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

    bloom = ParagraphBloom()
    dedup_warned = False
    stats = iter_cleaned_docs.stats = {"dupes_skipped": 0, "dedup_capped": False,
                                       "dedup_adds": 0}

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

            # Packed files carry DOC_SEP_CHAR between records; each record is
            # its own document (own <s>/</s> downstream). Files predating the
            # separator contain none and pass through as one document.
            for record in text.split(DOC_SEP_CHAR):
                paragraphs = [p.strip() for p in record.split("\n\n")]
                # Dedup in three layers: short paragraphs bypass (never
                # hashed), an intra-record repeat loses to its first
                # occurrence (keyed on the full 128-bit digest pair), and the
                # record's unique digests are then checked against -- and
                # added to -- the global filter in one batch. Near-sequential
                # semantics: the one divergence is a Bloom false positive
                # completed by an EARLIER paragraph in the same batch, which
                # the batch form keeps (sequential would drop it) -- fewer
                # unique paragraphs lost, deterministic either way.
                hashed_idx = [i for i, p in enumerate(paragraphs)
                              if len(p) >= MIN_PARAGRAPH_LENGTH]
                dropped: set[int] = set()
                if hashed_idx:
                    digests = np.empty((len(hashed_idx), 2), dtype=np.uint64)
                    first_seen: dict[tuple[int, int], int] = {}
                    intra_dupes: list[int] = []
                    for row, i in enumerate(hashed_idx):
                        d = hashlib.sha256(paragraphs[i].encode("utf-8")).digest()
                        pair = (int.from_bytes(d[:8], "little"),
                                int.from_bytes(d[8:16], "little"))
                        digests[row] = pair
                        if pair in first_seen:
                            intra_dupes.append(row)
                        else:
                            first_seen[pair] = row
                    unique_rows = sorted(first_seen.values())
                    present = bloom.seen_or_add(digests[unique_rows])
                    for row, was_present in zip(unique_rows, present):
                        if was_present:
                            dropped.add(hashed_idx[row])
                    for row in intra_dupes:
                        dropped.add(hashed_idx[row])
                    stats["dupes_skipped"] += len(intra_dupes) + int(present.sum())
                    stats["dedup_adds"] = bloom.adds
                    if bloom.adds > bloom.capacity and not dedup_warned:
                        dedup_warned = True
                        # Past design capacity the filter keeps answering, but
                        # its false-positive rate climbs above DEDUP_FPR --
                        # unique paragraphs start being dropped as dupes at an
                        # unmeasured rate. The sidecar records it: a console
                        # line scrolls past, and "dupes_skipped" on its own
                        # reads like the dedup ran clean.
                        stats["dedup_capped"] = True
                        print(f"  WARNING: dedup filter past its design capacity "
                              f"({bloom.capacity:,} adds) -- false-positive rate now "
                              f"exceeds {bloom.fpr:.2%}; recorded as dedup_capped "
                              f"in the sidecar")

                cleaned = "\n\n".join(p for i, p in enumerate(paragraphs)
                                      if i not in dropped).strip()
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
    ap.add_argument(
        "--curated-dir",
        default=None,
        help="walk this directory as the 'Curated' source instead of data/pretrain/curated "
        "-- a persona pack's shard (make_pretrain_curated.py --out). Label, walk order and "
        "every other source are unchanged, and --repeat-sources still resolves it as "
        "'curated' or by this directory's own basename",
    )
    ap.add_argument(
        "--only-curated",
        action="store_true",
        help="walk the Curated source ALONE: no other source dirs, no stackexchange "
        "expansion. With --curated-dir this tokenizes exactly one pack shard; "
        "--repeat-sources is refused under it (a one-source corpus's repeated extent "
        "spans the whole bin, which pretrain refuses at boot)",
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
    # And the general form: corpora are VERSIONED, never rebuilt in place. An
    # --output-bin that already exists is either the live corpus or a
    # rollback, and the standing command line has pointed at an existing bin
    # before (the v2b line survived in two docs after v2b became the
    # rollback). Overwrite-in-place destroys the receipt a failed build
    # would need.
    if out_bin.exists():
        raise SystemExit(
            f"refusing to overwrite {out_bin} -- it already exists, and corpora "
            f"are versioned, never rebuilt in place. Name a NEW bin (the next "
            f"free version suffix) or delete the old one deliberately first."
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
    if args.vocab and not Path(args.vocab).exists():
        # get_tokenizer does NOT raise for a missing vocab: it falls back to an
        # untrained 264-row char-level tokenizer with one console line, every
        # id stays inside the bounds guard, and the run completes -- writing a
        # garbage corpus. A wrong path must die here, not 40 minutes later.
        raise SystemExit(
            f"vocab file not found: {Path(args.vocab).resolve()}\n"
            "get_tokenizer would silently fall back to an untrained char-level "
            "tokenizer and this run would write a garbage corpus. Pass the full "
            "path (e.g. enigma_engine/vocab_model/bpe_vocab_v2_16k.json)."
        )
    tokenizer = get_tokenizer("bpe", vocab_path=args.vocab) if args.vocab else get_tokenizer("bpe")
    if not getattr(tokenizer, "merges", None):
        # Same failure through the other door: a vocab file that exists but
        # parses to nothing trained (or a default install with no live vocab).
        raise SystemExit(
            f"tokenizer loaded from {args.vocab or 'the default vocab'} has no "
            "BPE merges (untrained char-level fallback) -- refusing to write a "
            "corpus with it"
        )
    vocab_size = getattr(tokenizer, "vocab_size", 0)
    eos_id = getattr(tokenizer, "eos_token_id", 2)
    tok_name = type(tokenizer).__name__
    pretok = getattr(tokenizer, "pretokenizer_version", "v1")
    print(f"  Tokenizer: {tok_name}  (vocab={vocab_size:,}, eos={eos_id}, pretokenizer={pretok})")
    if vocab_size > 0 and not (0 <= eos_id < vocab_size):
        raise SystemExit(f"eos_token_id {eos_id} outside [0, {vocab_size}) -- refusing to write {out_bin.name}")
    if bpt == 2 and vocab_size > 65536:
        raise SystemExit(f"--dtype uint16 with vocab {vocab_size:,} > 65,536 -- ids would overflow")

    # A directory someone explicitly pointed the walk at cannot be absent by
    # intent (the same reasoning as the repeat-absent refusal below): an absent
    # or non-directory --curated-dir would merely WARN as a skipped source and
    # the run would write a corpus carrying no curated shard at all.
    if args.curated_dir is not None and not Path(args.curated_dir).is_dir():
        raise SystemExit(
            f"--curated-dir is not a directory: {Path(args.curated_dir)} -- a source "
            "named on the command line cannot be absent by intent (it would be "
            "skipped with a warning and this corpus would carry no curated shard). "
            "Point it at the shard make_pretrain_curated.py --out wrote."
        )
    # --only-curated makes the walk ONE source, so a repeat of it spans the
    # whole bin -- which means it reaches the val tail, and pretrain_enigma.py's
    # refuse_repeated_source_in_val refuses that corpus at boot. Fail here
    # instead of after the tokenize.
    if args.only_curated and args.repeat_sources.strip():
        raise SystemExit(
            "--only-curated with --repeat-sources is structurally untrainable: the "
            "walk is a single source, so the repeated source's extent spans the whole "
            "bin and therefore reaches the val tail -- pretrain_enigma.py refuses a "
            "repeated source that extends into val at boot. Tokenize the shard alone "
            "(no repeat), or oversample it in a full-corpus run (--curated-dir without "
            "--only-curated)."
        )

    source_dirs = build_source_dirs(curated_dir=args.curated_dir,
                                    only_curated=args.only_curated)
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

    sanitized_hits = [0]

    def _sanitized(stream):
        # Consume-time scrub: the collectors sanitize what they fetch NOW,
        # but ~96 GB collected before 2026-07-27 was never scrubbed and a
        # sampled scan found real literals in it ("List<A>" carves the actual
        # <A> id, a literal "</s>" writes EOS mid-document). This choke point
        # covers every source, past and future, before any id is written.
        # Runs BEFORE with_repeats so a cached curated pass is scrubbed once
        # and repeats stay byte-identical.
        for label, cleaned in stream:
            cleaned, hits = _sanitize_special_literals(cleaned)
            sanitized_hits[0] += hits
            yield label, cleaned

    def result_stream():
        """(label, packed, n) triples in deterministic walk order."""
        docs = _sanitized(iter_cleaned_docs(source_dirs))
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

    _stats = getattr(iter_cleaned_docs, "stats", {})
    dupes_skipped = _stats.get("dupes_skipped", 0)
    dedup_capped = bool(_stats.get("dedup_capped", False))
    vocab_sha = None
    if args.vocab:
        try:
            vocab_sha = hashlib.sha256(Path(args.vocab).read_bytes()).hexdigest()
        except OSError:
            vocab_sha = None
    tok_backend = "rust" if getattr(tokenizer, "_rust_backend", None) is not None else "python"

    for label, (n_docs, n_tok) in per_label.items():
        print(f"  [{label}] {n_docs:,} docs -> {n_tok:,} tokens")
    if sanitized_hits[0]:
        print(f"  {sanitized_hits[0]:,} special-token literal(s) space-broken before encoding")

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
        # Which walk this was. null/false is the full hardcoded one; a path
        # here means the "Curated" extent below is that directory's shard (a
        # persona pack's), not data/pretrain/curated.
        "curated_dir": str(Path(args.curated_dir).resolve()) if args.curated_dir else None,
        "only_curated": bool(args.only_curated),
        "repeated_sources": repeats,
        "source_token_extents": extents,
        "dupes_skipped": dupes_skipped,
        # False means the whole walk was deduped at the design false-positive
        # rate. True means adds exceeded the filter's design capacity and the
        # FPR climbed past dedup_fpr partway through, which dupes_skipped
        # alone cannot show.
        "dedup_capped": dedup_capped,
        "dedup_backend": "bloom",
        "dedup_cap": DEDUP_CAPACITY,
        "dedup_fpr": DEDUP_FPR,
        "dedup_adds": _stats.get("dedup_adds", 0),
        # vocab_file is a PATH, and a path is not an identity: regenerating the
        # vocab to a different table of the same size leaves every downstream
        # size check passing while the corpus no longer decodes. The hash is
        # what a later run can actually compare against.
        "vocab_sha256": vocab_sha,
        # Which encoder produced these ids. The rust and python paths are
        # supposed to agree; unrecorded, a corpus built by one and extended by
        # the other is indistinguishable from a consistent one.
        "tokenizer_backend": tok_backend,
        "special_literals_sanitized": sanitized_hits[0],
        # GiB, not GB -- kept under the historic key so older sidecars and the
        # trainer's corpus banner keep reading.
        "file_size_gib": round(file_gb, 2),
        "file_size_gb": round(file_gb, 2),
        "elapsed_seconds": round(elapsed, 1),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\n{'=' * 60}")
    print("  Done!")
    print(f"  Tokens:    {total_tokens:,}")
    print(f"  Documents: {total_docs:,}")
    print(f"  Dupes:     {dupes_skipped:,} paragraphs skipped"
          + ("  [FILTER PAST DESIGN CAPACITY -- false-positive rate exceeded "
             "its design bound partway through]" if dedup_capped else ""))
    print(f"  Output:    {out_bin} ({file_gb:.2f} GiB / "
          f"{file_gb * 1024**3 / 1e9:.2f} GB, {args.dtype})")
    print(f"  Metadata:  {out_meta}")
    print(f"  Time:      {elapsed / 60:.1f} min")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
