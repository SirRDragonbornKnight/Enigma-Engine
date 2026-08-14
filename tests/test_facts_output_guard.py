"""make_facts_pretrain_data.write_etok refuses an existing output.

The rule is pretokenize_data's, extended to the second ETOK writer the day
its defaults flipped to the v2 names (2026-08-10): --out now lands on the T4
run's actual training artifact (facts_tokens_v2.bin), and a default rebuild
would have silently destroyed that receipt -- knowledge_corpus has changed
since T4, so the rebuilt bytes would not be the ones the served model
trained on.
"""
from __future__ import annotations

import numpy as np
import pytest

from make_facts_pretrain_data import write_etok


def test_write_etok_refuses_an_existing_output(tmp_path):
    out = tmp_path / "facts.bin"
    tokens = np.arange(32, dtype=np.uint16)
    write_etok(tokens, out, vocab_size=100, n_docs=1, eos_id=2)
    assert out.exists() and out.with_suffix(".json").exists()
    with pytest.raises(SystemExit, match="already exists"):
        write_etok(tokens, out, vocab_size=100, n_docs=1, eos_id=2)


def test_main_refuses_an_existing_output_before_any_work(tmp_path, monkeypatch):
    """The guard used to live ONLY in write_etok -- the LAST statement of
    main() -- so a default run burned the whole 60M-token interleave off the
    48 GB memmap and then refused (audit 2026-08-13). It must fire at
    startup, before any corpus work."""
    import make_facts_pretrain_data as mod

    out = tmp_path / "facts.bin"
    out.write_bytes(b"ETOK-receipt")

    def _heavy_work_reached():
        raise AssertionError("gen_knowledge_pretrain_text ran -- the refusal came too late")

    monkeypatch.setattr(mod, "gen_knowledge_pretrain_text", _heavy_work_reached)
    monkeypatch.setattr("sys.argv", ["make_facts_pretrain_data.py", "--out", str(out)])
    with pytest.raises(SystemExit, match="already exists"):
        mod.main()


def test_write_etok_refuses_a_stale_sidecar(tmp_path):
    """A bin moved aside leaves its .json behind; overwriting that sidecar
    silently loses the dtype/vocab/eos receipt downstream readers trust."""
    out = tmp_path / "facts.bin"
    out.with_suffix(".json").write_text("{}", encoding="utf-8")
    tokens = np.arange(8, dtype=np.uint16)
    with pytest.raises(SystemExit, match="already exists"):
        write_etok(tokens, out, vocab_size=100, n_docs=1, eos_id=2)


def test_out_must_name_a_bin_file(tmp_path):
    """--out '' resolved to '.' and died as a raw ValueError inside
    with_suffix; a dotted stem (facts_v2.1) silently RENAMED its sidecar to
    facts_v2.json (audit 2026-08-13). All malformed shapes refuse cleanly."""
    from pathlib import Path

    from make_facts_pretrain_data import refuse_existing_output

    for bad in ("", ".", str(tmp_path / "facts_v2.1"), str(tmp_path / "noext")):
        with pytest.raises(SystemExit, match="bin"):
            refuse_existing_output(Path(bad))


def test_write_etok_still_writes_a_fresh_path(tmp_path):
    out = tmp_path / "fresh.bin"
    tokens = np.arange(16, dtype=np.uint32)
    write_etok(tokens, out, vocab_size=50, n_docs=2, eos_id=2)
    assert out.exists()
    # header records the array's own dtype width, not a constant
    header = out.read_bytes()[:28]  # <4sIIQII = 4+4+4+8+4+4
    import struct
    magic, ver, bpt, n, vocab, eos = struct.unpack("<4sIIQII", header)
    assert (magic, bpt, n, vocab, eos) == (b"ETOK", 4, 16, 50, 2)
