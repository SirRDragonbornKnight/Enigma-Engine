"""Collector guards: the special-literal sanitizer, the sealed-probe screen,
per-config byte accounting in the shared-directory case (finemath's 50/50
split silently delivered 0% of its second config before 2026-07-27), and the
combine step's determinism + mis-encoding receipt."""
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import collect_pretraining_data as cpd  # noqa: E402

V2_VOCAB = ROOT / "enigma_engine" / "vocab_model" / "bpe_vocab_v2_16k.json"


def test_special_literal_list_covers_the_live_vocab():
    """The sanitizer's list is hardcoded; the vocab's own special_tokens table
    is the authority. Every multi-char special EITHER live vocab declares --
    at any id, so the pin survives new allocations like the ruled image
    begin/end delimiters -- must be in the list, or a source file containing
    the missing literal writes a real control id into the corpus."""
    vocabs = [V2_VOCAB, ROOT / "enigma_engine" / "vocab_model" / "bpe_vocab.json"]
    checked = 0
    for path in vocabs:
        if not path.exists():
            continue
        vocab = json.loads(path.read_text(encoding="utf-8"))
        specials = vocab.get("special_tokens") or {}
        reserved = [tok for tok in specials if len(tok) > 1]
        assert reserved, f"{path.name} declares no multi-char specials? fixture broke"
        missing = [tok for tok in reserved if tok not in cpd._SPECIAL_TOKEN_LITERALS]
        assert not missing, (
            f"sanitizer does not cover {path.name} specials: {missing} -- "
            "new vocab rows (e.g. image delimiters) must be added to "
            "_SPECIAL_TOKEN_LITERALS in the same change"
        )
        checked += 1
    assert checked, "no vocab file found to pin against"


def test_sanitizer_actually_stops_the_carve():
    """The claim is not 'the string changed' but 'the tokenizer no longer
    emits the control id'. Assert both directions on the real v2 tokenizer:
    the raw text DOES carve (the hazard is real) and the sanitized text does
    not (the fix works)."""
    from enigma_engine.core.tokenizer import get_tokenizer

    tok = get_tokenizer("bpe", vocab_path=str(V2_VOCAB))
    raw = "public <A> List<A> map(Function<A> f) covers <s>this</s> and <Q> too."
    special_ids = set(range(14))

    raw_ids = set(tok.encode(raw))
    assert raw_ids & special_ids - {1, 2}, "raw literal text no longer carves -- hazard gone? re-check"

    clean, hits = cpd._sanitize_special_literals(raw)
    assert hits == 6  # three <A>, one <s>, one </s>, one <Q>
    for lit in cpd._SPECIAL_TOKEN_LITERALS:
        assert lit not in clean
    # bos/eos ids 1/2 frame every encode; no OTHER special id may survive
    clean_ids = set(tok.encode(clean))
    assert not (clean_ids & special_ids - {1, 2})

    untouched, zero = cpd._sanitize_special_literals("no literals in here at all")
    assert zero == 0 and untouched == "no literals in here at all"


def _fake_datasets(monkeypatch, records):
    """Install a fake `datasets` module whose load_dataset streams `records`."""

    class _Stream:
        def __init__(self, rows):
            self._rows = rows

        def skip(self, n):
            return _Stream(self._rows[n:])

        def __iter__(self):
            return iter(self._rows)

    fake = types.ModuleType("datasets")
    fake.load_dataset = lambda *a, **k: _Stream(records)
    monkeypatch.setitem(sys.modules, "datasets", fake)


def test_shared_dir_configs_account_separately(monkeypatch, tmp_path):
    """Two configs writing into ONE directory must each meet their own target.
    Directory-level byte accounting made the second config read the first's
    files as its own progress and skip entirely -- the finemath split shipped
    100% finemath-4plus, 0% InfiMM."""
    body = ("solve for x in this worked example " * 40) + "<A> appears once here. "
    records = [{"text": body + f"record {i}"} for i in range(40)]
    _fake_datasets(monkeypatch, records)
    monkeypatch.setattr(cpd, "save_progress", lambda progress: None)
    monkeypatch.setattr(cpd, "detect_ai_content", lambda text: False)
    monkeypatch.setattr(cpd, "quality_filter", lambda text, min_length: True)
    monkeypatch.setattr(cpd, "_locked_probe_guard", lambda label: None)

    shared = tmp_path / "finemath"
    target_gb = 20_000 / 1e9  # tiny target so a few records satisfy it

    saved_a = cpd._fetch_hf_streaming(
        dataset_name="x/y", config_name="a", text_field="text", output_dir=shared,
        target_gb=target_gb, label="FineMath-4plus", progress={}, progress_key="a",
        min_length=10, filter_ai=False,
    )
    saved_b = cpd._fetch_hf_streaming(
        dataset_name="x/y", config_name="b", text_field="text", output_dir=shared,
        target_gb=target_gb, label="InfiMM-WebMath", progress={}, progress_key="b",
        min_length=10, filter_ai=False,
    )
    a_files = list(shared.glob("finemath_4plus_*.txt"))
    b_files = list(shared.glob("infimm_webmath_*.txt"))
    assert saved_a and a_files, "first config wrote nothing"
    assert saved_b and b_files, "second config skipped -- the shared-dir accounting bug"
    # and the sanitizer runs on the write path: no literal survives into a file
    for f in a_files + b_files:
        assert "<A>" not in f.read_text(encoding="utf-8")
        assert "< A>" in f.read_text(encoding="utf-8")


def test_sealed_probe_screen_drops_leaky_records(monkeypatch, tmp_path):
    """A record quoting a sealed probe must not be written to disk: the
    pretrain path has no consume-time guard, so collection time is the only
    screen this text will ever meet."""
    records = [
        {"text": "perfectly ordinary long math text " * 20},
        {"text": "THE SEALED PROBE VERBATIM " + "padding words here " * 20},
    ]
    _fake_datasets(monkeypatch, records)
    monkeypatch.setattr(cpd, "save_progress", lambda progress: None)
    monkeypatch.setattr(cpd, "quality_filter", lambda text, min_length: True)

    class _Guard:
        def leaks(self, text):
            return "SEALED PROBE" in text

    monkeypatch.setattr(cpd, "_locked_probe_guard", lambda label: _Guard())

    out = tmp_path / "screened"
    cpd._fetch_hf_streaming(
        dataset_name="x/y", config_name=None, text_field="text", output_dir=out,
        target_gb=1e-6, label="ScreenTest", progress={}, progress_key="s",
        min_length=10, filter_ai=False,
    )
    saved_text = "".join(f.read_text(encoding="utf-8") for f in out.glob("*.txt"))
    assert "ordinary long math text" in saved_text
    assert "SEALED PROBE" not in saved_text


_SOURCE_DIR_GLOBALS = (
    "WIKI_DUMP_DIR", "WIKI_DIR", "SIMPLE_DIR", "GUTENBERG_DIR", "FINEWEB_DIR",
    "OPENWEBTEXT_DIR", "C4_DIR", "DCLM_DIR", "FINEMATH_DIR", "STACK_DIR",
    "WAYBACK_DIR", "FANDOM_DIR", "STACKEX_DIR",
)


def _isolate_combine(monkeypatch, tmp_path) -> Path:
    """Point every source dir and the output file at tmp_path; return the one
    source dir the test fills. Unpatched, combine_all_sources walks the REAL
    collected corpus and overwrites the real combined.txt."""
    for name in _SOURCE_DIR_GLOBALS:
        monkeypatch.setattr(cpd, name, tmp_path / "absent" / name.lower())
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setattr(cpd, "WIKI_DIR", source)
    monkeypatch.setattr(cpd, "COMBINED_FILE", tmp_path / "combined.txt")
    return source


def _enumerate_reversed(monkeypatch):
    """Hand the combine its directory entries in reverse-name order -- the
    stand-in for a filesystem whose enumeration order is not name order."""
    real_scandir = os.scandir

    class _Scan:
        def __init__(self, entries):
            self._entries = entries

        def __enter__(self):
            return iter(self._entries)

        def __exit__(self, *exc):
            return False

    def fake_scandir(path):
        with real_scandir(path) as it:
            return _Scan(sorted(it, key=lambda e: e.name, reverse=True))

    monkeypatch.setattr(os, "scandir", fake_scandir)


def test_combine_orders_files_by_name_not_enumeration_order(monkeypatch, tmp_path):
    """combined.txt must be a function of the tree, not of the order the
    filesystem happens to hand back. With first-wins paragraph dedup the
    order decides WHICH copy of a duplicated paragraph survives, so an
    unsorted walk let the same tree combine to different training bytes."""
    source = _isolate_combine(monkeypatch, tmp_path)
    shared = "the same paragraph appears in every one of these source files here"
    # created z, m, a -- creation order is not name order
    for name, marker in (("z_omega.txt", "ZZZ"), ("m_middle.txt", "MMM"), ("a_alpha.txt", "AAA")):
        (source / name).write_text(
            f"{marker} marker paragraph long enough to clear the minimum length bar\n\n{shared}",
            encoding="utf-8",
        )
    _enumerate_reversed(monkeypatch)

    cpd.combine_all_sources()
    combined = cpd.COMBINED_FILE.read_text(encoding="utf-8")

    order = [combined.index(m) for m in ("AAA", "MMM", "ZZZ")]
    assert order == sorted(order), combined
    # ...and first-wins dedup therefore keeps the copy from the FIRST name
    assert combined.count(shared) == 1
    assert combined.index(shared) < combined.index("MMM"), combined


def test_combine_warns_on_replacement_chars(monkeypatch, tmp_path, capsys):
    """errors="replace" keeps one bad byte from costing a whole source file,
    but it wrote U+FFFD into the training bytes with no count and no warning.
    The substitutions get named and counted."""
    source = _isolate_combine(monkeypatch, tmp_path)
    good = source / "clean.txt"
    good.write_text("a perfectly clean paragraph, long enough to clear the minimum bar", encoding="utf-8")
    bad = source / "mojibake.txt"
    bad.write_bytes(b"a paragraph with \xff three \xff bad \xff bytes, long enough to clear the bar")

    cpd.combine_all_sources()
    out = capsys.readouterr().out

    warns = [line for line in out.splitlines() if "WARN" in line]
    assert len(warns) == 1, out
    assert "mojibake.txt" in warns[0] and "3" in warns[0], warns
    assert "clean.txt" not in out
