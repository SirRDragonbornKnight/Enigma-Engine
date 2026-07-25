"""pretokenize_data.py -- the corpus writer.

The load-bearing contract for the v2 retokenize (TOKENIZER_V2_SPEC): the
PARALLEL encode path must produce a BYTE-IDENTICAL corpus to the sequential
one. Parallelism here is not a convenience -- it is the difference between
5.7 hours and ~0.7 -- but a corpus that differs from the sequential result
by even one token is a silently wrong pretrain that nothing downstream can
detect. Determinism comes from parent-side walk/dedup + an ORDERED imap;
these tests are what keep that true.

Also pinned: the uint16 write path (v2 vocab fits 2-byte ids, halving the
corpus), the ETOK header the pretrain reader validates against, and the
refusal to aim a custom vocab/dtype at the v1 lineage tokens.bin.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pretokenize_data as pd  # noqa: E402
from enigma_engine.core.bpe_tokenizer import BPETokenizer  # noqa: E402

# Enough documents that a parallel run really splits work across chunks
# (pool chunksize is 32) rather than handing everything to one worker.
N_DOCS = 90


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """A miniature two-source corpus + a small v2 vocab trained on it.

    Source A (walked first) holds HEAVY documents and source B light ones.
    The asymmetry is deliberate: with an unordered imap the light chunks
    finish long before the heavy first chunk and shuffle the output stream,
    so the byte-identity test fails deterministically instead of only when
    scheduling happens to reorder equal-sized work. (Mutation P1 survived
    the original equal-size fixture -- unordered results arrived in
    submission order anyway.)
    """
    root = tmp_path_factory.mktemp("corpus")
    src_a = root / "src_a"
    src_b = root / "src_b"
    src_a.mkdir()
    src_b.mkdir()

    shared = (
        "The model learns from the data and the data shapes the model. "
        "In 1000 years 2500 machines counted 42 stars and 7 moons. "
    )
    # A 64-char single-letter run the vocab learns as ONE deep-merged token
    # (z -> zz -> ... -> z*64), so a document holding only it encodes to
    # [bos, z64, eos] = 3 tokens < 5 and _encode_doc returns None. This is
    # the only way to exercise the skip branch INSIDE spawn workers, where
    # monkeypatching cannot reach (audit LOW-3a).
    tiny_marker = "z" * 64
    for i in range(N_DOCS):
        heavy = i % 2 == 0  # exactly the docs that land in source A
        filler = (
            " ".join(f"filler{i}word{j}" for j in range(2500))
            if heavy
            else f"light body for document {i}"
        )
        body = (
            f"Document number {i} describes the {i % 7} th subject in detail. "
            + filler + " " + shared * 2
            + f"It closes with a unique remark about item {i} and its history. "
        )
        target = src_a if heavy else src_b
        # A duplicated paragraph in every file exercises the shared dedup set,
        # which is the reason the walk must stay in the parent process.
        (target / f"doc_{i:03d}.txt").write_text(
            body + "\n\n" + shared + "\n\n" + f"Tail paragraph unique to document {i}, long enough to survive.",
            encoding="utf-8",
        )

    # One extra file that must be SKIPPED (encodes below the 5-token floor);
    # walked last, so the skip lands inside a parallel chunk and the
    # byte-identity test certifies deque balance across a real None.
    (src_b / "doc_091_tiny.txt").write_text(tiny_marker, encoding="utf-8")

    vocab = root / "v2_vocab.json"
    tok = BPETokenizer(pretokenizer="v2")
    tok.train(
        [shared * 40 + (tiny_marker + "\n") * 30],
        vocab_size=400,
        min_frequency=1,
        verbose=False,
    )
    tok.save(vocab)
    assert len(tok.encode(tiny_marker)) < 5, "fixture marker failed to deep-merge; skip branch untested"
    return root, [("A", src_a), ("B", src_b)], vocab


def _run(monkeypatch, sources, out_bin, *, vocab, dtype, workers, extra=()):
    monkeypatch.setattr(pd, "SOURCE_DIRS", sources)
    monkeypatch.setattr(pd, "SE_DIR", out_bin.parent / "_no_stackexchange")
    argv = [
        "pretokenize_data.py",
        "--output-bin", str(out_bin),
        "--vocab", str(vocab),
        "--dtype", dtype,
        "--workers", str(workers),
        *extra,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    pd.main()
    return out_bin, out_bin.with_suffix(".json")


def _read_header(path: Path):
    with open(path, "rb") as fh:
        magic, ver, bpt, total, vocab_size, eos = struct.unpack("<4sIIQII", fh.read(28))
    return magic, ver, bpt, total, vocab_size, eos


def test_parallel_output_is_byte_identical_to_sequential(monkeypatch, corpus, tmp_path):
    """THE contract. An ordered imap over a parent-side walk must reproduce
    the sequential byte stream exactly -- otherwise the 8x speedup silently
    buys a different corpus."""
    root, sources, vocab = corpus
    seq_bin, seq_meta = _run(monkeypatch, sources, tmp_path / "seq.bin", vocab=vocab, dtype="uint16", workers=1)
    par_bin, par_meta = _run(monkeypatch, sources, tmp_path / "par.bin", vocab=vocab, dtype="uint16", workers=4)

    assert seq_bin.read_bytes() == par_bin.read_bytes(), "parallel corpus differs from sequential"

    s, p = json.loads(seq_meta.read_text()), json.loads(par_meta.read_text())
    for key in ("total_tokens", "total_documents", "vocab_size", "eos_token_id", "dtype", "dupes_skipped"):
        assert s[key] == p[key], f"metadata {key} diverged: {s[key]} vs {p[key]}"
    assert s["total_tokens"] > 0 and s["total_documents"] == N_DOCS
    assert s["dupes_skipped"] > 0, "the shared paragraph should have been deduped"


def test_uint16_halves_the_stream_and_round_trips(monkeypatch, corpus, tmp_path):
    """uint16 is what takes the v2 corpus from ~86 GB to ~43. The ids must
    survive the narrower type exactly."""
    import numpy as np

    root, sources, vocab = corpus
    b16, m16 = _run(monkeypatch, sources, tmp_path / "u16.bin", vocab=vocab, dtype="uint16", workers=1)
    b32, m32 = _run(monkeypatch, sources, tmp_path / "u32.bin", vocab=vocab, dtype="uint32", workers=1)

    meta16, meta32 = json.loads(m16.read_text()), json.loads(m32.read_text())
    assert meta16["total_tokens"] == meta32["total_tokens"]
    stream16 = b16.stat().st_size - pd.HEADER_SIZE
    stream32 = b32.stat().st_size - pd.HEADER_SIZE
    assert stream32 == 2 * stream16

    ids16 = np.memmap(b16, dtype=np.uint16, mode="r", offset=pd.HEADER_SIZE)
    ids32 = np.memmap(b32, dtype=np.uint32, mode="r", offset=pd.HEADER_SIZE)
    assert np.array_equal(ids16.astype(np.int64), ids32.astype(np.int64)), "uint16 lost or altered ids"


def test_header_matches_what_the_pretrain_reader_validates(monkeypatch, corpus, tmp_path):
    """pretrain_enigma.py hard-fails on magic/bpt/count mismatch; the writer
    must satisfy every one of those checks for both widths."""
    root, sources, vocab = corpus
    for dtype, want_bpt in (("uint16", 2), ("uint32", 4)):
        out, meta_path = _run(monkeypatch, sources, tmp_path / f"h_{dtype}.bin", vocab=vocab, dtype=dtype, workers=1)
        magic, ver, bpt, total, vocab_size, eos = _read_header(out)
        meta = json.loads(meta_path.read_text())
        assert magic == b"ETOK" and ver == 1
        assert bpt == want_bpt
        assert total == meta["total_tokens"]
        assert total == (out.stat().st_size - pd.HEADER_SIZE) // want_bpt  # the reader's exact arithmetic
        assert vocab_size == meta["vocab_size"]
        assert 0 <= eos < vocab_size
        assert meta["pretokenizer"] == "v2"


def test_refuses_any_write_at_the_lineage_path(monkeypatch, corpus, tmp_path):
    """The v1 tokens.bin is the pretrain lineage, COMPLETE since 2026-07-03.
    The old guard refused only a custom vocab/dtype there, so a bare
    no-argument run could still overwrite it -- and since Curated joined
    SOURCE_DIRS the rebuild would differ from the lineage regardless (new
    source; first-wins dedup). Any write at that path is now refused.

    The lineage path here is a tmp STAND-IN, never the real tokens.bin.
    Under mutation testing the guard this test exercises is deliberately
    disabled, at which point main() runs to completion and WRITES its
    output -- aiming at the production path nearly cost the real corpus
    (2026-07-20; only a subprocess timeout stopped the atomic rename)."""
    root, sources, vocab = corpus
    lineage = tmp_path / "lineage_stand_in" / "tokens.bin"
    lineage.parent.mkdir()
    monkeypatch.setattr(pd, "SOURCE_DIRS", sources)
    monkeypatch.setattr(pd, "SE_DIR", tmp_path / "_no_stackexchange")
    monkeypatch.setattr(pd, "OUTPUT_BIN", lineage)
    for extra in ([], ["--vocab", str(vocab)], ["--dtype", "uint16"]):
        monkeypatch.setattr(sys, "argv", ["pretokenize_data.py", "--output-bin", str(lineage)] + extra)
        with pytest.raises(SystemExit, match="refusing to overwrite"):
            pd.main()
    assert not lineage.exists() and not lineage.with_suffix(".bin.tmp").exists()

    # the SIDECAR is lineage too: `tokens.bin2` (any tokens.<ext> beside it)
    # passes the bin refusal and then maps with_suffix(".json") onto the
    # lineage receipt -- which is gitignored, so the clobber is unrecoverable
    # (round-B audit, 2026-07-25)
    monkeypatch.setattr(
        sys, "argv",
        ["pretokenize_data.py", "--output-bin", str(lineage.parent / "tokens.bin2"),
         "--vocab", str(vocab), "--dtype", "uint16"],
    )
    with pytest.raises(SystemExit, match="sidecar"):
        pd.main()
    assert not (lineage.parent / "tokens.json").exists(), "the lineage sidecar was clobbered"

    # \\?\ extended-length spellings: resolve() keeps the prefix verbatim, so
    # a string == let them straight past both refusals and the lineage bin
    # AND sidecar were overwritten (round-C audit, 2026-07-25)
    for target, what in ((lineage, "refusing to overwrite"),
                         (lineage.parent / "tokens.bin2", "sidecar")):
        monkeypatch.setattr(
            sys, "argv",
            ["pretokenize_data.py", "--output-bin", "\\\\?\\" + str(target),
             "--vocab", str(vocab), "--dtype", "uint16"],
        )
        with pytest.raises(SystemExit, match=what):
            pd.main()
    assert not lineage.exists() and not (lineage.parent / "tokens.json").exists()


def test_uint16_refuses_an_oversized_vocab(monkeypatch, corpus, tmp_path):
    """uint16 with vocab > 65,536 would wrap ids silently."""
    root, sources, vocab = corpus

    class _Huge:
        vocab_size = 70000
        eos_token_id = 2
        pretokenizer_version = "v2"

    monkeypatch.setattr(pd, "SOURCE_DIRS", sources)
    monkeypatch.setattr(pd, "SE_DIR", tmp_path / "_no_stackexchange")
    monkeypatch.setattr("enigma_engine.core.tokenizer.get_tokenizer", lambda *a, **k: _Huge())
    monkeypatch.setattr(
        sys, "argv",
        ["pretokenize_data.py", "--output-bin", str(tmp_path / "big.bin"), "--dtype", "uint16", "--vocab", str(vocab)],
    )
    with pytest.raises(SystemExit, match="65,536"):
        pd.main()


def test_sub_minimum_documents_are_skipped_in_workers(monkeypatch, corpus, tmp_path):
    """The fixture ships N_DOCS + 1 files; the tiny-marker doc encodes below
    the 5-token floor and must vanish from the corpus IN THE PARALLEL PATH
    (the None travels back from a spawn worker and the labels deque must
    stay in lockstep across it)."""
    root, sources, vocab = corpus
    out, meta_path = _run(monkeypatch, sources, tmp_path / "skip.bin", vocab=vocab, dtype="uint16", workers=4)
    meta = json.loads(meta_path.read_text())
    assert meta["total_documents"] == N_DOCS  # 91 files walked, 90 kept


def test_absent_source_dirs_are_logged_not_silent(monkeypatch, corpus, tmp_path, capsys):
    """Spec requirement: a missing source dir must be CALLED OUT (v1 skips
    silently; a quietly smaller corpus is undetectable downstream). The
    sidecar records the absence as provenance."""
    root, sources, vocab = corpus
    ghost = ("Ghost", root / "never_created_dir")
    out, meta_path = _run(
        monkeypatch, sources + [ghost], tmp_path / "ghost.bin", vocab=vocab, dtype="uint16", workers=1
    )
    assert "ABSENT" in capsys.readouterr().out
    assert json.loads(meta_path.read_text())["sources_absent"] == ["Ghost"]


def test_failed_encode_leaves_no_partial_output(monkeypatch, corpus, tmp_path):
    """A worker exception mid-stream must abort the run and remove the .tmp:
    a half-written corpus that survives to disk is a poisoned pretrain."""
    root, sources, vocab = corpus
    real = pd._encode_doc
    calls = {"n": 0}

    def boom(cleaned):
        calls["n"] += 1
        if calls["n"] == 10:
            raise RuntimeError("encode died mid-stream")
        return real(cleaned)

    monkeypatch.setattr(pd, "_encode_doc", boom)
    out = tmp_path / "crash.bin"
    monkeypatch.setattr(pd, "SOURCE_DIRS", sources)
    monkeypatch.setattr(pd, "SE_DIR", tmp_path / "_no_stackexchange")
    monkeypatch.setattr(
        sys, "argv",
        ["pretokenize_data.py", "--output-bin", str(out), "--vocab", str(vocab), "--dtype", "uint16", "--workers", "1"],
    )
    with pytest.raises(RuntimeError, match="encode died"):
        pd.main()
    assert not out.exists(), "output bin exists after a failed run"
    assert not out.with_suffix(".bin.tmp").exists(), "leftover .tmp after a failed run"
    assert not out.with_suffix(".json").exists(), "metadata written for a failed run"


# ---------------------------------------------------------------------------
# --repeat-sources: the curated shard's oversample (round-7 redesign).
# Position in the corpus buys nothing (the sampler is IID, and the END of the
# bin is val), so oversampling must live in the stream itself -- and on the
# far side of the dedup, or it silently is not an oversample at all.
# ---------------------------------------------------------------------------


def test_on_disk_copies_collapse_but_repeats_do_not(tmp_path):
    """The obvious route -- replicating the shard's files on disk -- is
    exactly what the GLOBAL paragraph dedup deletes: copy one seeds
    seen_hashes and every later copy of each long paragraph is dropped,
    silently, while short paragraphs sail through (a skew, not just a
    shrink). with_repeats runs on the walk's OUTPUT, so its copies are
    byte-identical and untouchable by the dedup."""
    src = tmp_path / "cur"
    src.mkdir()
    para = "A fact she must know, phrased long enough to be hashed by the dedup pass. " * 2
    assert len(para) >= pd.MIN_PARAGRAPH_LENGTH
    (src / "a.txt").write_text(para, encoding="utf-8")
    (src / "b.txt").write_text(para, encoding="utf-8")  # the on-disk "oversample"

    docs = list(pd.iter_cleaned_docs([("C", src)]))
    assert len(docs) == 1, (
        "fixture assumption gone: the global dedup no longer collapses "
        "on-disk copies, so with_repeats may not be the only honest oversample"
    )
    assert list(pd.with_repeats(iter(docs), {"C": 3})) == docs * 3


def test_repeats_emit_whole_passes_never_adjacent_copies():
    """Copies of one doc must land a full source apart -- adjacent copies can
    share a training window and teach verbatim repetition."""
    docs = [("A", "a1"), ("A", "a2"), ("B", "b1")]
    assert list(pd.with_repeats(iter(docs), {"A": 3})) == [
        ("A", "a1"), ("A", "a2"),
        ("A", "a1"), ("A", "a2"),
        ("A", "a1"), ("A", "a2"),
        ("B", "b1"),
    ]
    # a repeated FINAL source flushes at end-of-stream too
    assert list(pd.with_repeats(iter(docs), {"B": 2})) == docs + [("B", "b1")]
    # and no repeats means the identity stream
    assert list(pd.with_repeats(iter(docs), {})) == docs


def test_repeat_sources_spec_is_validated():
    dirs = [("Curated", Path("data/pretrain/curated")), ("A", Path("x/src_a"))]
    assert pd.parse_repeat_sources("", dirs) == {}
    assert pd.parse_repeat_sources("curated=5", dirs) == {"Curated": 5}
    assert pd.parse_repeat_sources("src_a=2, CURATED=3", dirs) == {"A": 2, "Curated": 3}
    with pytest.raises(SystemExit, match="no such source"):
        pd.parse_repeat_sources("nonesuch=2", dirs)
    for bad in ("curated=1", "curated=0", "curated=51", "curated=x", "curated"):
        with pytest.raises(SystemExit, match="whole number"):
            pd.parse_repeat_sources(bad, dirs)


def test_repeated_source_writes_byte_identical_passes_and_extents(monkeypatch, corpus, tmp_path):
    """End to end: the repeated source's region is N byte-identical passes,
    the sidecar records the repeat and every source's token extent (what lets
    pretrain refuse a repeated source that reaches into the val tail), and
    the rest of the stream is untouched."""
    root, sources, vocab = corpus
    base_bin, base_meta = _run(
        monkeypatch, sources, tmp_path / "base.bin", vocab=vocab, dtype="uint16", workers=1
    )
    rep_bin, rep_meta = _run(
        monkeypatch, sources, tmp_path / "rep.bin", vocab=vocab, dtype="uint16", workers=1,
        extra=("--repeat-sources", "a=3"),
    )
    base, rep = json.loads(base_meta.read_text()), json.loads(rep_meta.read_text())
    assert rep["repeated_sources"] == {"A": 3}
    assert base["repeated_sources"] == {}

    ax0, ax1 = base["source_token_extents"]["A"]
    bx0, bx1 = base["source_token_extents"]["B"]
    assert (ax0, bx0) == (0, ax1), "extents must tile the stream in walk order"
    rx0, rx1 = rep["source_token_extents"]["A"]
    assert (rx0, rx1) == (0, 3 * ax1), "repeated extent must cover all passes"
    assert rep["total_documents"] == base["total_documents"] + 2 * 45  # source A holds 45 docs

    bpt = 2  # uint16
    base_stream = base_bin.read_bytes()[pd.HEADER_SIZE:]
    rep_stream = rep_bin.read_bytes()[pd.HEADER_SIZE:]
    passes = [rep_stream[i * ax1 * bpt:(i + 1) * ax1 * bpt] for i in range(3)]
    assert passes[0] == base_stream[:ax1 * bpt], "pass 1 must equal the unrepeated stream"
    assert passes[0] == passes[1] == passes[2], "copies must be byte-identical (dedup-proof)"
    assert rep_stream[3 * ax1 * bpt:] == base_stream[ax1 * bpt:], "later sources shifted or changed"


def test_a_repeat_naming_an_absent_source_refuses(monkeypatch, corpus, tmp_path):
    """A merely-missing source WARNs; a source someone explicitly asked to
    oversample cannot be absent by intent -- its silent loss is the failure
    the repeat exists to prevent."""
    root, sources, vocab = corpus
    ghost = ("Ghost", root / "never_created_dir")
    with pytest.raises(SystemExit, match="absent"):
        _run(monkeypatch, sources + [ghost], tmp_path / "g.bin", vocab=vocab,
             dtype="uint16", workers=1, extra=("--repeat-sources", "ghost=2"))


def test_a_declared_oversample_that_never_happened_refuses(monkeypatch, corpus, tmp_path):
    """The absent-dir check catches a missing DIRECTORY and nothing else: a
    dir with no .txt files (or one whose every paragraph an earlier source
    already published, first-wins dedup) emitted zero tokens while the
    sidecar recorded the multiplier as fact -- the fix-arc audit measured a
    stream byte-identical to a run with no such source at all, under a meta
    claiming x5 (2026-07-25). The run must refuse, and must not leave a bin."""
    root, sources, vocab = corpus
    hollow = root / "hollow_dir"
    hollow.mkdir(exist_ok=True)
    (hollow / "notes.md").write_text("not a .txt, never walked", encoding="utf-8")
    out = tmp_path / "hollow.bin"
    with pytest.raises(SystemExit, match="emitted no tokens"):
        _run(monkeypatch, sources + [("Hollow", hollow)], out, vocab=vocab,
             dtype="uint16", workers=1, extra=("--repeat-sources", "hollow=5"))
    assert not out.exists() and not out.with_suffix(".bin.tmp").exists()


def test_the_repeat_cache_refuses_web_scale_sources(monkeypatch, corpus, tmp_path):
    """The cache IS the mechanism (byte-identical passes beyond the dedup),
    and it holds the whole source in RAM -- aimed at a web source that is an
    OOM hours into the walk, after which the except-handler deletes the .tmp.
    Refuse as soon as the cap is crossed instead.

    The refusal must be a plain Exception, never SystemExit: with workers > 1
    the generator is consumed inside pool.imap's task-feeder thread, whose
    guard catches Exception only -- a SystemExit killed the feeder silently
    and the run HUNG forever instead of refusing (round-B audit, 2026-07-25;
    measured: workers=1 refused, workers=2 sat past a 90 s kill)."""
    src = tmp_path / "big"
    src.mkdir()
    body = ("A paragraph long enough to be walked and cached without any "
            "trouble at all, number %d.")
    for i in range(4):
        (src / f"doc_{i}.txt").write_text(body % i, encoding="utf-8")
    monkeypatch.setattr(pd, "REPEAT_CACHE_MAX_BYTES", 150)
    with pytest.raises(pd.RepeatCacheExceeded, match="repeat cache") as excinfo:
        list(pd.with_repeats(pd.iter_cleaned_docs([("Big", src)]), {"Big": 2}))
    assert isinstance(excinfo.value, Exception), (
        "a BaseException-only class escapes pool.imap's feeder guard and "
        "turns the refusal into a silent hang under --workers > 1"
    )
    # under the cap the same stream repeats fine
    monkeypatch.setattr(pd, "REPEAT_CACHE_MAX_BYTES", 10_000)
    docs = list(pd.with_repeats(pd.iter_cleaned_docs([("Big", src)]), {"Big": 2}))
    assert len(docs) == 8

    # end to end, main() converts the refusal to a clean SystemExit and
    # removes the .tmp (the BaseException path used to leave it on kill)
    root, sources, vocab = corpus
    monkeypatch.setattr(pd, "REPEAT_CACHE_MAX_BYTES", 150)
    out = tmp_path / "capped.bin"
    with pytest.raises(SystemExit, match="repeat cache"):
        _run(monkeypatch, sources + [("Big", src)], out, vocab=vocab,
             dtype="uint16", workers=1, extra=("--repeat-sources", "big=2"))
    assert not out.exists() and not out.with_suffix(".bin.tmp").exists()
