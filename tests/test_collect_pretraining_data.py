"""Collector guards: the special-literal sanitizer, the sealed-probe screen,
per-config byte accounting in the shared-directory case (finemath's 50/50
split silently delivered 0% of its second config before 2026-07-27), the
combine step's determinism + mis-encoding receipt, the resume markers that
may only fire on a CLEAN pass, atomic per-article writes, the Stack license
screen, and the wiki-dump resume checks."""
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import pytest

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


# ---------------------------------------------------------------------------
# Resume markers, atomic writes, license screen, dump resume (audit 2026-08-22)
# ---------------------------------------------------------------------------


class _FakeResp:
    """Stand-in for a requests.Response on the collectors' fetch seams."""

    def __init__(self, status_code=200, payload=None, text="", headers=None, chunks=()):
        self.status_code = status_code
        self.headers = dict(headers or {})
        self.text = text
        self._payload = payload
        self._chunks = list(chunks)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise cpd.requests.RequestException(f"HTTP {self.status_code}")

    def json(self):
        return self._payload

    def iter_content(self, chunk_size=None):
        return iter(self._chunks)


def _quiet_collector(monkeypatch):
    """Neutralize the rate limits and the quality screens; nothing sleeps and
    nothing writes to the real progress.json."""
    monkeypatch.setattr(cpd, "WAYBACK_DELAY", 0)
    monkeypatch.setattr(cpd, "FANDOM_DELAY", 0)
    monkeypatch.setattr(cpd.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(cpd, "save_progress", lambda progress: None)
    monkeypatch.setattr(cpd, "quality_filter", lambda text, min_length: True)
    monkeypatch.setattr(cpd, "detect_ai_content", lambda text: False)


def _wire_wayback(monkeypatch, tmp_path, rows, page_status=200):
    """Point fetch_wayback at tmp_path with a stubbed CDX + page seam.
    Returns the list of archived-page URLs actually fetched."""
    _quiet_collector(monkeypatch)
    monkeypatch.setattr(cpd, "WAYBACK_DIR", tmp_path / "wayback")
    fetched: list[str] = []
    body = "<html><body><p>" + ("an archived paragraph of real prose " * 20) + "</p></body></html>"

    class _Session:
        def get(self, url, params=None, timeout=None):
            if "/cdx/" in url:
                return _FakeResp(payload=[["timestamp", "original"]] + rows)
            fetched.append(url)
            return _FakeResp(status_code=page_status, text=body)

    monkeypatch.setattr(cpd, "SESSION", _Session())
    return fetched


def test_a_capped_wayback_domain_is_revisited_and_a_finished_one_is_not(monkeypatch, tmp_path):
    """The per-URL loop breaks on the page cap, then fell through to
    wayback_done.add() anyway -- so every later, bigger run skipped the domain
    forever with its remaining pages unfetched. Only a full pass over the CDX
    rows may complete it."""
    pattern = "plato.stanford.edu/entries/*"
    rows = [["2020", f"http://plato.stanford.edu/entries/e{i}"] for i in range(3)]
    fetched = _wire_wayback(monkeypatch, tmp_path, rows)
    progress: dict = {}

    saved = cpd.fetch_wayback([(pattern, "SEP")], max_pages=2, progress=progress)
    assert saved == 2 and len(fetched) == 2
    assert progress.get("wayback_done_domains", []) == [], "a cap-stopped domain was marked done"

    # The next, bigger run revisits it, takes the page it missed, and only THEN
    # -- having walked every row -- marks it done.
    saved = cpd.fetch_wayback([(pattern, "SEP")], max_pages=5, progress=progress)
    assert saved == 3 and len(fetched) == 3
    assert progress["wayback_done_domains"] == [pattern]


def test_a_rate_limited_wayback_domain_is_not_marked_done(monkeypatch, tmp_path):
    """The other early exit: MAX_RETRIES consecutive 429s move on to the next
    domain with nothing saved. That is a backoff, not a completed domain."""
    pattern = "www.britannica.com/topic/*"
    rows = [["2020", f"http://www.britannica.com/topic/t{i}"] for i in range(cpd.MAX_RETRIES + 2)]
    _wire_wayback(monkeypatch, tmp_path, rows, page_status=429)
    progress: dict = {}

    saved = cpd.fetch_wayback([(pattern, "Britannica")], max_pages=50, progress=progress)
    assert saved == 0
    assert progress.get("wayback_done_domains", []) == [], "a rate-limited domain was marked done"


def _wire_fandom(monkeypatch, tmp_path, batches):
    """Point fetch_fandom at tmp_path. `batches` is a list of
    (titles, apcontinue) served to successive allpages calls; the last repeats."""
    _quiet_collector(monkeypatch)
    monkeypatch.setattr(cpd, "FANDOM_DIR", tmp_path / "fandom")
    state = {"n": 0}
    wikitext = "a real article body with enough prose to clear the length bar " * 20

    class _Session:
        def get(self, url, params=None, timeout=None):
            if params.get("list") == "allpages":
                titles, apcontinue = batches[min(state["n"], len(batches) - 1)]
                state["n"] += 1
                payload = {"query": {"allpages": [{"title": t} for t in titles]}}
                if apcontinue:
                    payload["continue"] = {"apcontinue": apcontinue}
                return _FakeResp(payload=payload)
            titles = params["titles"].split("|")
            return _FakeResp(payload={"query": {"pages": {
                str(i): {"title": t, "revisions": [{"slots": {"main": {"*": wikitext}}}]}
                for i, t in enumerate(titles)
            }}})

    monkeypatch.setattr(cpd, "SESSION", _Session())


def test_a_capped_fandom_wiki_is_revisited_and_a_finished_one_is_not(monkeypatch, tmp_path):
    """Same shape as wayback, threaded: _fetch_wiki breaks on the article cap
    (and on rate-limit/error) and then marked the wiki done unconditionally.
    Only running the allpages enumeration out may complete it."""
    _wire_fandom(monkeypatch, tmp_path, [(["Alpha", "Beta"], "Beta"), (["Gamma"], None)])
    progress: dict = {}

    cpd.fetch_fandom([("minecraft", "Minecraft")], max_articles=1, progress=progress)
    assert progress.get("fandom_done_wikis", []) == [], "a cap-stopped wiki was marked done"
    assert len(list((tmp_path / "fandom").glob("*.txt"))) == 1

    cpd.fetch_fandom([("minecraft", "Minecraft")], max_articles=10, progress=progress)
    assert progress["fandom_done_wikis"] == ["minecraft"]
    assert len(list((tmp_path / "fandom").glob("*.txt"))) == 2


def test_an_interrupted_article_write_leaves_no_blessed_file(monkeypatch, tmp_path):
    """Resume reads the .txt stems on disk as done, so a direct write_text cut
    partway froze a TRUNCATED article into the corpus forever. tmp-then-rename
    means the final name never appears unless the whole body landed."""
    rows = [["2020", "http://plato.stanford.edu/entries/e0"]]
    _wire_wayback(monkeypatch, tmp_path, rows)
    real_write_text = Path.write_text

    def truncate_then_die(self, data, *args, **kwargs):
        # Keyed on ".txt" so it cuts the write WHEREVER the article goes --
        # a direct write to the final name is interrupted exactly as a tmp is.
        if ".txt" in self.name:
            real_write_text(self, data[: len(data) // 2], *args, **kwargs)
            raise KeyboardInterrupt("Ctrl+C partway through the write")
        return real_write_text(self, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", truncate_then_die)
    with pytest.raises(KeyboardInterrupt):
        cpd.fetch_wayback([("plato.stanford.edu/entries/*", "SEP")], max_pages=5, progress={})

    out = tmp_path / "wayback"
    leftover = list(out.glob("*.txt.tmp"))
    assert not list(out.glob("*.txt")), "a truncated article was blessed with its final name"
    assert leftover, "fixture broke -- nothing was written at all"


def test_a_corrupt_dump_stream_is_reported_not_raised(monkeypatch, tmp_path, capsys):
    """bz2 reports a truncated or corrupt compressed stream as OSError, and
    only KeyboardInterrupt and ET.ParseError were caught -- so it raised out
    of a run hours deep and took the final stats with it, when the articles
    already extracted are keepers."""
    dump = tmp_path / "enwiki.xml.bz2"
    dump.write_bytes(b"this is not a bzip2 stream at all, not even close")
    monkeypatch.setattr(cpd, "WIKI_DUMP_FILE", dump)
    monkeypatch.setattr(cpd, "WIKI_DUMP_DIR", tmp_path / "wikipedia_dump")
    monkeypatch.setattr(cpd, "_download_wiki_dump", lambda resume=True: True)
    monkeypatch.setattr(cpd, "save_progress", lambda progress: None)

    assert cpd.process_wiki_dump({}) == 0
    out = capsys.readouterr().out
    assert "Read error" in out and "Done:" in out


def test_the_stack_keeps_only_records_with_a_permissive_license(monkeypatch, tmp_path):
    """The filter is documented permissive-only, but `if licenses and not
    any(...)` passed an EMPTY license list -- unlicensed code, exactly the
    class the screen exists to keep out of the corpus."""
    body = "def f():\n    return 42\n" * 40
    records = [
        {"content": "PERMISSIVE\n" + body, "max_stars_repo_licenses": ["MIT"]},
        {"content": "UNLICENSED\n" + body, "max_stars_repo_licenses": []},
        {"content": "RESTRICTIVE\n" + body, "max_stars_repo_licenses": ["gpl-3.0"]},
    ]
    _fake_datasets(monkeypatch, records)
    monkeypatch.setattr(cpd, "STACK_DIR", tmp_path / "the_stack")
    monkeypatch.setattr(cpd, "_STACK_LANGUAGES", [("python", 1.0)])
    monkeypatch.setattr(cpd, "save_progress", lambda progress: None)
    monkeypatch.setattr(cpd, "_locked_probe_guard", lambda label: None)

    cpd.fetch_the_stack(target_gb=1e-6, progress={})

    written = "".join(f.read_text(encoding="utf-8") for f in (tmp_path / "the_stack").glob("*.txt"))
    assert "PERMISSIVE" in written
    assert "UNLICENSED" not in written, "an unlicensed record cleared the permissive-only screen"
    assert "RESTRICTIVE" not in written


def _wire_dump(monkeypatch, tmp_path, head_length, get_resp):
    """Point _download_wiki_dump at tmp_path; returns the headers the GET saw."""
    monkeypatch.setattr(cpd, "WIKI_DUMP_FILE", tmp_path / "enwiki.xml.bz2")
    monkeypatch.setattr(cpd, "WIKI_DUMP_VALIDATOR", tmp_path / "enwiki.xml.bz2.validator")
    seen: dict = {"headers": {}}

    class _Session:
        def head(self, url, timeout=None):
            return _FakeResp(headers={"Content-Length": str(head_length)})

        def get(self, url, headers=None, stream=False, timeout=None):
            seen["headers"] = dict(headers or {})
            return get_resp

    monkeypatch.setattr(cpd, "SESSION", _Session())
    return seen


def test_an_oversize_local_dump_is_not_verified_as_complete(monkeypatch, tmp_path):
    """`existing_size >= expected` blessed a local file BIGGER than the
    server's as verified and handed it straight to the bz2 parser. Only an
    exact match verifies; anything larger is not a partial of this dump and
    there is nothing to resume from -- it is re-downloaded from zero."""
    dump = tmp_path / "enwiki.xml.bz2"
    dump.write_bytes(b"o" * 100)
    seen = _wire_dump(
        monkeypatch, tmp_path, head_length=50,
        get_resp=_FakeResp(headers={"Content-Length": "50", "ETag": '"v2"'}, chunks=[b"n" * 50]),
    )

    assert cpd._download_wiki_dump(resume=True) is True
    assert "Range" not in seen["headers"], "resumed a file that is not this dump"
    assert dump.read_bytes() == b"n" * 50
    # ...and the fresh body's validator is recorded for the next resume
    assert cpd.WIKI_DUMP_VALIDATOR.read_text(encoding="utf-8") == '"v2"'


def test_a_resume_replays_the_recorded_validator_as_if_range(monkeypatch, tmp_path):
    """enwiki-LATEST rotates. A bare Range against it splices a DIFFERENT
    dump's tail onto the partial and nothing downstream can tell. If-Range
    makes the server answer 200-with-full-body instead, which the existing
    restart-from-zero branch already handles."""
    dump = tmp_path / "enwiki.xml.bz2"
    dump.write_bytes(b"p" * 30)
    (tmp_path / "enwiki.xml.bz2.validator").write_text('"v1"', encoding="utf-8")
    seen = _wire_dump(
        monkeypatch, tmp_path, head_length=100,
        get_resp=_FakeResp(status_code=206, headers={"Content-Range": "bytes 30-99/100"},
                           chunks=[b"q" * 70]),
    )

    assert cpd._download_wiki_dump(resume=True) is True
    assert seen["headers"]["Range"] == "bytes=30-"
    assert seen["headers"]["If-Range"] == '"v1"'
    assert dump.read_bytes() == b"p" * 30 + b"q" * 70
    # a 206 means the stored validator still describes the partial: it stays
    assert cpd.WIKI_DUMP_VALIDATOR.read_text(encoding="utf-8") == '"v1"'


def test_a_resume_with_no_recorded_validator_keeps_the_old_behavior(monkeypatch, tmp_path, capsys):
    """No validator on file (a partial from before this was recorded, or a
    server that sent neither ETag nor Last-Modified) resumes exactly as
    before -- and says so, because that splice is then unguarded."""
    dump = tmp_path / "enwiki.xml.bz2"
    dump.write_bytes(b"p" * 30)
    seen = _wire_dump(
        monkeypatch, tmp_path, head_length=100,
        get_resp=_FakeResp(status_code=206, headers={"Content-Range": "bytes 30-99/100"},
                           chunks=[b"q" * 70]),
    )

    assert cpd._download_wiki_dump(resume=True) is True
    assert seen["headers"]["Range"] == "bytes=30-"
    assert "If-Range" not in seen["headers"]
    assert "without If-Range" in capsys.readouterr().out
