"""collect_audio_data: the transcript bytes and the corpus-replacement path.

Round-2 review (2026-08-13): str.capitalize() LOWERCASED everything past
index 0 -- every proper noun, acronym, and standalone "I" across ~28.5k
utterances -- and the output jsonl was truncated on open, so an interrupted
or empty walk replaced a good corpus with a stub.
"""
from __future__ import annotations

import json

import pytest

from collect_audio_data import SUBSET, _prose, collect_librispeech


def test_prose_restores_first_person_from_all_caps():
    """True casing is unrecoverable from ALL-CAPS; the one CERTAIN case in
    English is first-person I (and its contractions), which .capitalize()
    destroyed."""
    assert _prose("MISTER QUILTER IS THE APOSTLE") == "Mister quilter is the apostle"
    assert _prose("I THINK I'LL GO AND I'M SURE I'D KNOW") == "I think I'll go and I'm sure I'd know"
    assert _prose("WHAT I SAID IS WHAT I MEANT") == "What I said is what I meant"
    # words merely containing 'i' stay untouched
    assert _prose("THE SKI HILL IS ICY") == "The ski hill is icy"


def test_librispeech_refuses_to_replace_the_corpus_with_nothing(tmp_path):
    """An extracted tree that yields zero pairs must NOT truncate a good
    librispeech.jsonl -- it refuses, and the existing file survives."""
    good = tmp_path / "librispeech.jsonl"
    good.write_text('{"audio": "a.flac", "text": "Existing pair."}\n', encoding="utf-8")
    (tmp_path / "LibriSpeech" / SUBSET).mkdir(parents=True)  # empty tree, walk finds nothing

    with pytest.raises(SystemExit, match="zero pairs"):
        collect_librispeech(tmp_path, skip_download=True)

    assert good.read_text(encoding="utf-8").startswith('{"audio": "a.flac"'), \
        "the refusal still clobbered the existing corpus"


def test_librispeech_walk_emits_prose_and_skips_textless_lines(tmp_path):
    chapter = tmp_path / "LibriSpeech" / SUBSET / "19" / "198"
    chapter.mkdir(parents=True)
    (chapter / "19-198.trans.txt").write_text(
        "19-198-0000 SHE SAID I WOULD KNOW\n"
        "19-198-0001\n"          # textless line: skipped, not written as ""
        "19-198-0002 MISSING FLAC LINE\n",
        encoding="utf-8",
    )
    (chapter / "19-198-0000.flac").write_bytes(b"flac")
    # 0001's flac EXISTS so only the textless guard can drop that row (with
    # no .flac the pre-existing missing-file guard dropped it either way and
    # this test proved nothing -- audit 2026-08-14; same audit: the prose
    # row's I is mid-sentence, where .capitalize() would lowercase it).
    (chapter / "19-198-0001.flac").write_bytes(b"flac")

    pairs = collect_librispeech(tmp_path, skip_download=True)

    assert pairs == 1
    rows = [json.loads(l) for l in
            (tmp_path / "librispeech.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["text"] == "She said I would know"
    assert all(r["text"] for r in rows)
