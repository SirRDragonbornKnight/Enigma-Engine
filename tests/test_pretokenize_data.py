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


def test_refuses_to_rebuild_any_existing_bin_in_place(monkeypatch, corpus, tmp_path):
    """Corpora are versioned, never rebuilt in place. This is not lineage
    paranoia but a measured hazard: after v2b became the ROLLBACK, the
    standing retokenize line in two documents still named tokens_v2b.bin as
    its output -- "copy the line whole" would have destroyed the rollback
    with no receipt (found 2026-07-30). Any --output-bin that already exists
    on disk must refuse at boot, before the walk touches anything."""
    root, sources, vocab = corpus
    existing = tmp_path / "corpora" / "tokens_v9x.bin"
    existing.parent.mkdir()
    existing.write_bytes(b"ETOK" + b"\x00" * 60)  # stands in for a live corpus
    monkeypatch.setattr(pd, "SOURCE_DIRS", sources)
    monkeypatch.setattr(pd, "SE_DIR", tmp_path / "_no_stackexchange")
    monkeypatch.setattr(sys, "argv",
                        ["pretokenize_data.py", "--output-bin", str(existing),
                         "--vocab", str(vocab), "--dtype", "uint16"])
    with pytest.raises(SystemExit, match="never rebuilt in place"):
        pd.main()
    assert existing.read_bytes()[:4] == b"ETOK", "the existing bin was touched"
    # a FRESH name beside it is fine -- the guard must not block new versions
    fresh = existing.parent / "tokens_v9y.bin"
    monkeypatch.setattr(sys, "argv",
                        ["pretokenize_data.py", "--output-bin", str(fresh),
                         "--vocab", str(vocab), "--dtype", "uint16"])
    pd.main()
    assert fresh.exists(), "a new version name must still build"


def test_special_literals_are_scrubbed_at_consume_time(monkeypatch, corpus, tmp_path):
    """~96 GB of source text predates the collectors' fetch-time sanitizer and
    a sampled scan found real literals in it ("List<A>" carves the actual <A>
    id under the v2 vocab). The pretokenize choke point must scrub every doc
    -- past or future source alike -- before any id is written, and the
    sidecar must carry the count as the receipt."""
    root, sources, vocab = corpus
    poisoned = sources[0][1] / "doc_900_literals.txt"
    poisoned.write_text(
        "A code sample follows here with plenty of body text to survive the "
        "cleaner and the five token floor of the encoder. "
        "public <A> List<A> map(Function<A> f) plus a literal </s> tag and "
        "one <USER> marker inside otherwise ordinary prose that keeps going "
        "long enough to be kept as its own paragraph.",
        encoding="utf-8",
    )
    try:
        out, meta_path = _run(monkeypatch, sources, tmp_path / "scrub.bin", vocab=vocab,
                              dtype="uint16", workers=1)
        meta = json.loads(meta_path.read_text())
        assert meta["special_literals_sanitized"] == 5  # three <A>, one </s>, one <USER>
        # The receipt alone is not the property: a wrapper that COUNTS but
        # yields the original text produces the same sidecar. Decode the bin
        # and assert on the artifact. (</s> frames every doc in the decode,
        # so the assertion uses the literals that only the poisoned doc
        # carries.)
        import numpy as np

        from enigma_engine.core.tokenizer import get_tokenizer
        ids = np.memmap(out, dtype=np.uint16, mode="r", offset=pd.HEADER_SIZE)
        decoded = get_tokenizer("bpe", vocab_path=str(vocab)).decode([int(i) for i in ids])
        assert "<A>" not in decoded and "<USER>" not in decoded, "literal reached the corpus"
        assert "< A>" in decoded and "< USER>" in decoded, "scrubbed form missing from the corpus"
    finally:
        poisoned.unlink()  # module-scoped fixture; later tests expect the base corpus


def test_missing_or_untrained_vocab_refuses_before_writing(monkeypatch, corpus, tmp_path):
    """get_tokenizer does not raise on a missing vocab path: it falls back to
    an untrained 264-row char-level tokenizer whose ids all pass the bounds
    guard, so a bare --vocab filename run from the wrong directory would have
    COMPLETED and written a garbage corpus with one console line as witness.
    Both doors refuse now: the missing path, and a loaded tokenizer with no
    BPE merges."""
    root, sources, vocab = corpus
    out = tmp_path / "guard.bin"
    with pytest.raises(SystemExit) as err:
        _run(monkeypatch, sources, out, vocab=root / "missing_vocab.json", dtype="uint16", workers=1)
    assert "garbage corpus" in str(err.value)
    assert not out.exists(), "refusal must not leave an output bin"

    class _Untrained:
        vocab_size = 264
        eos_token_id = 2
        merges: list = []
        pretokenizer_version = "v1"

    monkeypatch.setattr("enigma_engine.core.tokenizer.get_tokenizer", lambda *a, **k: _Untrained())
    with pytest.raises(SystemExit) as err:
        _run(monkeypatch, sources, out, vocab=vocab, dtype="uint16", workers=1)
    assert "no BPE merges" in str(err.value)
    assert not out.exists()


def test_uint16_refuses_an_oversized_vocab(monkeypatch, corpus, tmp_path):
    """uint16 with vocab > 65,536 would wrap ids silently."""
    root, sources, vocab = corpus

    class _Huge:
        vocab_size = 70000
        eos_token_id = 2
        pretokenizer_version = "v2"
        # trained (merges present) -- the refusal under test is the uint16
        # bound, not the untrained-fallback guard
        merges = [("a", "b")]

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


def test_packed_files_split_into_documents_at_the_record_separator(tmp_path):
    """The HF-streamed sources batch ~5 MB of records per file joined by
    "\\n\\n" -- indistinguishable from a paragraph break, so one FILE read as
    one document and <s>/</s> came almost entirely from the
    one-article-per-file sources (measured 2026-07-28: 3-4 bos in the first
    5M tokens of FineWeb-Edu/Stack/DCLM vs Wiki Dump's 4,870). The collector
    now writes U+001E between records; the walk must yield each record as
    its OWN document, and a file predating the separator must still read as
    exactly one."""
    src = tmp_path / "packed"
    src.mkdir()
    rec_a = "First record with enough text to clear the minimum paragraph gate. " * 2
    rec_b = "Second record, also long enough to clear the minimum length gate. " * 2
    (src / "packed_00000000.txt").write_text(
        rec_a + "\n\x1e\n" + rec_b, encoding="utf-8")
    docs = list(pd.iter_cleaned_docs([("P", src)]))
    assert [d[1] for d in docs] == [rec_a.strip(), rec_b.strip()], (
        "each record must become its own document -- one joined doc here "
        "means the separator was not split on and every packed source is "
        "back to one <s>/</s> per 5 MB file"
    )
    # a legacy file with no separator is ONE document, not zero, not many
    legacy = tmp_path / "legacy"
    legacy.mkdir()
    (legacy / "old_00000000.txt").write_text(
        rec_a + "\n\n" + rec_b, encoding="utf-8")
    docs = list(pd.iter_cleaned_docs([("L", legacy)]))
    assert len(docs) == 1 and rec_a.strip() in docs[0][1] and rec_b.strip() in docs[0][1]


def test_bloom_dedup_drops_cross_and_intra_record_dupes_only():
    """The 50M-entry exact set capped out mid-walk on v2b, so sources past
    the fill were never deduped against each other. The Bloom replacement
    must keep the first-wins semantics the corpus contract rests on: a
    repeat inside one record loses to its first occurrence, a repeat across
    records/files loses to the earlier record, short paragraphs bypass, and
    a NEVER-seen paragraph is never dropped (no false negatives at test
    scale)."""
    bloom = pd.ParagraphBloom(capacity=10_000, fpr=1e-3)
    import numpy as _np

    def dig(s):
        import hashlib as _h
        d = _h.sha256(s.encode()).digest()
        return (int.from_bytes(d[:8], "little"), int.from_bytes(d[8:16], "little"))

    a, b, c = dig("para a"), dig("para b"), dig("para c")
    arr = _np.array([a, b], dtype=_np.uint64)
    assert list(bloom.seen_or_add(arr)) == [False, False]
    # a again (cross-batch dupe) + c (fresh): exactly one flagged
    arr2 = _np.array([a, c], dtype=_np.uint64)
    assert list(bloom.seen_or_add(arr2)) == [True, False]
    # no false negatives: everything added answers present
    arr3 = _np.array([a, b, c], dtype=_np.uint64)
    assert list(bloom.seen_or_add(arr3)) == [True, True, True]
    assert bloom.adds == 3

    # the same-word-collision hazard the .at() comment guards: many digests
    # in ONE batch landing bits in the same backing word must ALL persist
    # (fancy-index |= keeps only one write). 64 fresh paragraphs, then
    # re-query: every one must answer present.
    fresh = _np.array([dig(f"p{i}") for i in range(64)], dtype=_np.uint64)
    bloom.seen_or_add(fresh)
    assert list(bloom.seen_or_add(fresh)) == [True] * 64


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


# ---------------------------------------------------------------------------
# --curated-dir / --only-curated: a persona pack's shard through the SAME walk
# (Stage 7 wave 4a). The pack pipeline used to stop at a screened shard on
# disk -- SOURCE_DIRS is hardcoded and its only curated entry is Enigma's, so
# nothing could tokenize anyone else's. The two flags are orthogonal: swap the
# PATH under the "Curated" label, and/or restrict the walk to that one entry.
# ---------------------------------------------------------------------------


@pytest.fixture
def swap_dirs(corpus, tmp_path):
    """Her curated shard, a pack's shard, and one other source.

    Deliberately tiny (one document each) rather than the module corpus:
    these tests decode the whole bin back to text to prove WHICH directory
    was walked, and the module corpus's 45 heavy documents make that slow
    without making it stronger."""
    _root, _sources, vocab = corpus
    hers = tmp_path / "curated"
    pack = tmp_path / "curated_atlas"
    other = tmp_path / "wiki"
    for d in (hers, pack, other):
        d.mkdir()
    (hers / "curated_000.txt").write_text(
        "Enigma states her own name here in a paragraph long enough to be kept.",
        encoding="utf-8")
    (pack / "curated_000.txt").write_text(
        "Atlas states his own name here in a paragraph long enough to be kept.",
        encoding="utf-8")
    (other / "article_000.txt").write_text(
        "An ordinary web article with nothing to do with either identity at all.",
        encoding="utf-8")
    return hers, pack, other, vocab


def _decode(out_bin: Path, vocab) -> str:
    """The corpus read back as text -- the artifact, not the sidecar's claim."""
    import numpy as np

    from enigma_engine.core.tokenizer import get_tokenizer
    ids = np.memmap(out_bin, dtype=np.uint16, mode="r", offset=pd.HEADER_SIZE)
    return get_tokenizer("bpe", vocab_path=str(vocab)).decode([int(i) for i in ids])


def test_curated_dir_swaps_the_path_and_keeps_the_label(monkeypatch, swap_dirs, tmp_path):
    """A pack's shard is walked AS the Curated source: same label, same first
    position (so it can never land in the val tail and it wins dedup
    collisions), and every other source untouched."""
    hers, pack, other, vocab = swap_dirs
    out, meta_path = _run(monkeypatch, [("Curated", hers), ("Wiki", other)],
                          tmp_path / "swap.bin", vocab=vocab, dtype="uint16", workers=1,
                          extra=("--curated-dir", str(pack)))
    meta = json.loads(meta_path.read_text())
    assert sorted(meta["source_token_extents"]) == ["Curated", "Wiki"]
    assert meta["source_token_extents"]["Curated"][0] == 0, "the swap moved the walk"
    assert meta["sources_absent"] == []

    decoded = _decode(out, vocab)
    assert "Atlas states his own name" in decoded, "the substitute shard was not walked"
    assert "Enigma states her own name" not in decoded, "the replaced dir was still walked"
    assert "ordinary web article" in decoded, "the rest of the walk changed"


def test_repeat_sources_resolves_a_swapped_dir_by_label_and_by_basename(
        monkeypatch, swap_dirs, tmp_path):
    """parse_repeat_sources keys on label AND directory basename, so a swapped
    curated dir must answer to both -- 'curated' (the standing line, copied
    whole) and the pack directory's own name."""
    hers, pack, other, vocab = swap_dirs
    for key, tag in (("curated", "label"), ("curated_atlas", "basename")):
        out, meta_path = _run(monkeypatch, [("Curated", hers), ("Wiki", other)],
                              tmp_path / f"rep_{tag}.bin", vocab=vocab, dtype="uint16",
                              workers=1,
                              extra=("--curated-dir", str(pack), "--repeat-sources", f"{key}=3"))
        meta = json.loads(meta_path.read_text())
        assert meta["repeated_sources"] == {"Curated": 3}, f"{key} did not resolve"
        x0, x1 = meta["source_token_extents"]["Curated"]
        assert x0 == 0
        one_pass = (x1 - x0) // 3
        assert one_pass > 0
        stream = out.read_bytes()[pd.HEADER_SIZE:]
        passes = [stream[i * one_pass * 2:(i + 1) * one_pass * 2] for i in range(3)]
        assert passes[0] == passes[1] == passes[2], "the copies are not byte-identical"
        assert "Atlas states his own name" in _decode(out, vocab)


def test_only_curated_walks_exactly_one_source(monkeypatch, swap_dirs, tmp_path):
    """The smoke shape: the pack's shard and nothing else. No other
    SOURCE_DIRS entry, and no SE expansion either."""
    hers, pack, other, vocab = swap_dirs
    # _run points SE_DIR at <out dir>/_no_stackexchange; creating it with a
    # subdir gives the full walk something real to expand, which is the only
    # way to prove --only-curated drops the expansion rather than finding it
    # empty.
    se_sub = tmp_path / "_no_stackexchange" / "askubuntu"
    se_sub.mkdir(parents=True)
    (se_sub / "qa_000.txt").write_text(
        "A stackexchange answer with enough body text to be kept by the cleaner.",
        encoding="utf-8")
    sources = [("Curated", hers), ("Wiki", other)]

    full, full_meta = _run(monkeypatch, sources, tmp_path / "full.bin", vocab=vocab,
                           dtype="uint16", workers=1)
    assert sorted(json.loads(full_meta.read_text())["source_token_extents"]) == [
        "Curated", "SE/askubuntu", "Wiki"], "control run did not walk all three"

    out, meta_path = _run(monkeypatch, sources, tmp_path / "only.bin", vocab=vocab,
                          dtype="uint16", workers=1,
                          extra=("--only-curated", "--curated-dir", str(pack)))
    meta = json.loads(meta_path.read_text())
    assert list(meta["source_token_extents"]) == ["Curated"]
    assert meta["sources_absent"] == []
    decoded = _decode(out, vocab)
    assert "Atlas states his own name" in decoded
    for stranger in ("Enigma states her own name", "ordinary web article",
                     "stackexchange answer"):
        assert stranger not in decoded, f"--only-curated still walked: {stranger}"


def test_only_curated_refuses_a_repeat(monkeypatch, swap_dirs, tmp_path):
    """A one-source corpus with a repeat is untrainable and the refusal must
    say why: the repeated extent spans the whole bin, so it reaches the val
    tail and pretrain_enigma.refuse_repeated_source_in_val refuses at boot.
    Fail here, not after the tokenize."""
    hers, pack, other, vocab = swap_dirs
    out = tmp_path / "only_rep.bin"
    with pytest.raises(SystemExit) as err:
        _run(monkeypatch, [("Curated", hers), ("Wiki", other)], out, vocab=vocab,
             dtype="uint16", workers=1,
             extra=("--only-curated", "--curated-dir", str(pack),
                    "--repeat-sources", "curated=3"))
    msg = str(err.value)
    assert "val tail" in msg and "pretrain_enigma.py" in msg
    assert not out.exists() and not out.with_suffix(".bin.tmp").exists()

    # without --only-curated the same repeat is the standing, honest line
    ok, ok_meta = _run(monkeypatch, [("Curated", hers), ("Wiki", other)],
                       tmp_path / "full_rep.bin", vocab=vocab, dtype="uint16", workers=1,
                       extra=("--curated-dir", str(pack), "--repeat-sources", "curated=3"))
    assert json.loads(ok_meta.read_text())["repeated_sources"] == {"Curated": 3}


def test_curated_dir_must_name_a_real_directory(monkeypatch, swap_dirs, tmp_path):
    """A source named on the command line cannot be absent by intent -- absent
    it would only WARN as a skipped dir and the corpus would carry no curated
    shard at all. A FILE is the same mistake with a typo."""
    hers, pack, other, vocab = swap_dirs
    sources = [("Curated", hers), ("Wiki", other)]
    ghost = tmp_path / "no_such_shard"
    not_a_dir = tmp_path / "shard_is_a_file.txt"
    not_a_dir.write_text("a shard is a directory of .txt files", encoding="utf-8")

    for i, target in enumerate((ghost, not_a_dir)):
        out = tmp_path / f"absent_{i}.bin"
        with pytest.raises(SystemExit, match="--curated-dir is not a directory"):
            _run(monkeypatch, sources, out, vocab=vocab, dtype="uint16", workers=1,
                 extra=("--curated-dir", str(target)))
        assert not out.exists() and not out.with_suffix(".bin.tmp").exists()
        # and with --only-curated, where the walk would otherwise be empty
        out = tmp_path / f"absent_only_{i}.bin"
        with pytest.raises(SystemExit, match="--curated-dir is not a directory"):
            _run(monkeypatch, sources, out, vocab=vocab, dtype="uint16", workers=1,
                 extra=("--only-curated", "--curated-dir", str(target)))
        assert not out.exists()


def test_sidecar_records_the_walk_shape(monkeypatch, swap_dirs, tmp_path):
    """Which walk built this corpus is provenance: a "Curated" extent means a
    different shard depending on these two fields."""
    hers, pack, other, vocab = swap_dirs
    sources = [("Curated", hers), ("Wiki", other)]
    _out, meta_path = _run(monkeypatch, sources, tmp_path / "shape.bin", vocab=vocab,
                           dtype="uint16", workers=1,
                           extra=("--curated-dir", str(pack), "--only-curated"))
    meta = json.loads(meta_path.read_text())
    assert Path(meta["curated_dir"]) == pack.resolve()
    assert meta["only_curated"] is True

    _bare, bare_meta = _run(monkeypatch, sources, tmp_path / "shape_bare.bin", vocab=vocab,
                            dtype="uint16", workers=1)
    bare = json.loads(bare_meta.read_text())
    assert bare["curated_dir"] is None and bare["only_curated"] is False
    # additive only: the fields the trainer and the audits already read
    for key in ("sources_absent", "repeated_sources", "source_token_extents"):
        assert key in bare


def test_a_bare_walk_is_untouched_by_the_new_knobs(monkeypatch, tmp_path):
    """The pin. Neither flag may change a run that passes neither: the bare
    walk is the hardcoded list plus the SE expansion, in that exact order,
    read from the module globals the whole suite monkeypatches."""
    hers, wiki = tmp_path / "curated", tmp_path / "wikipedia"
    se = tmp_path / "stackexchange"
    for d in (hers, wiki, se / "unix", se / "askubuntu"):
        d.mkdir(parents=True)
    patched = [("Curated", hers), ("Wikipedia", wiki)]
    monkeypatch.setattr(pd, "SOURCE_DIRS", patched)
    monkeypatch.setattr(pd, "SE_DIR", se)

    expected = patched + [("SE/askubuntu", se / "askubuntu"), ("SE/unix", se / "unix")]
    assert pd.build_source_dirs() == expected
    assert pd.build_source_dirs(curated_dir=None, only_curated=False) == expected

    # the knobs change WHICH dirs, never the order, and never the globals
    swapped = pd.build_source_dirs(curated_dir=tmp_path / "curated_atlas")
    assert [label for label, _p in swapped] == [label for label, _p in expected]
    assert swapped[0][1] == tmp_path / "curated_atlas"
    assert pd.build_source_dirs(curated_dir=tmp_path / "curated_atlas", only_curated=True) == [
        ("Curated", tmp_path / "curated_atlas")]
    assert pd.SOURCE_DIRS == patched, "build_source_dirs mutated the module list"
