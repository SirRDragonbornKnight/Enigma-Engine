"""Append the NEW anime/LN fandom articles to tokens.bin (instead of a 13h+ full rebuild).

SAFETY MODEL: record the original byte length L0 first. The original 56.6B tokens occupy
[0, L0) and are NEVER modified (we only open in append / seek-header modes). On ANY error or
failed verification, truncate the file back to L0 and restore the ETOK header + tokens.json
-> byte-for-byte original. tokens.bin is also fully reproducible from sources, so risk is doubly bounded.
"""

import sys, os, json, struct, shutil, time, array
from pathlib import Path

sys.path.insert(0, r"C:\Users\SirKn\Enigma Engine")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

PP = Path(r"C:\Users\SirKn\Enigma Engine\data\pretrain")
BIN = PP / "tokens.bin"
META = PP / "tokens.json"
BAK = PP / "tokens.json.bak"
FAND = PP / "fandom"
HEADER = 256
MIN_LEN = 50

from enigma_engine.core.tokenizer import get_tokenizer

tok = get_tokenizer("bpe")
eos = getattr(tok, "eos_token_id", 2)

meta = json.loads(META.read_text(encoding="utf-8"))
orig_total = meta["total_tokens"]
orig_docs = meta["total_documents"]
L0 = BIN.stat().st_size
# Integrity gate: file length must match metadata BEFORE we touch anything.
assert (L0 - HEADER) // 4 == orig_total, (
    f"PRE-CHECK FAIL: file has {(L0 - HEADER) // 4:,} tok, meta says {orig_total:,}"
)
print(f"orig: {L0:,} bytes = {orig_total:,} tokens, {orig_docs:,} docs  [integrity OK]", flush=True)

bin_mtime = BIN.stat().st_mtime
new_files = sorted(f for f in FAND.glob("*.txt") if f.stat().st_mtime > bin_mtime)
print(f"new anime/LN files to append: {len(new_files):,}", flush=True)
assert new_files, "no new files found"

shutil.copy2(META, BAK)
added_tok = 0
added_docs = 0
t0 = time.time()
try:
    with open(BIN, "ab") as out:
        buf = array.array("I")
        for i, f in enumerate(new_files):
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if len(text.strip()) < MIN_LEN:
                continue
            ids = tok.encode(text)
            if len(ids) < 5:
                continue
            ids.append(eos)
            buf.extend(ids)
            added_tok += len(ids)
            added_docs += 1
            if len(buf) >= 2_000_000:
                buf.tofile(out)
                buf = array.array("I")
            if (i + 1) % 5000 == 0:
                print(f"  {i + 1:,}/{len(new_files):,} files | {added_tok:,} tok | {time.time() - t0:.0f}s", flush=True)
        if buf:
            buf.tofile(out)
        out.flush()
        os.fsync(out.fileno())

    new_size = BIN.stat().st_size
    assert new_size - L0 == added_tok * 4, f"BYTE MISMATCH: grew {new_size - L0}, expected {added_tok * 4}"
    assert added_tok > 0
    print(
        f"appended {added_tok:,} tokens ({added_docs:,} docs) = {(new_size - L0) / 1e6:.1f} MB in {time.time() - t0:.0f}s",
        flush=True,
    )

    # verify by decoding the boundary + tail
    import numpy as np

    data = np.memmap(BIN, dtype=np.uint32, mode="r", offset=HEADER)
    first = tok.decode(data[orig_total : orig_total + 60].tolist())
    last = tok.decode(data[-60:].tolist())
    del data
    print(f"\nfirst appended tokens -> {first[:220]!r}", flush=True)
    print(f"last  appended tokens -> {last[:220]!r}", flush=True)

    # commit: update ETOK header total_tokens (offset 12, <Q) + tokens.json
    new_total = orig_total + added_tok
    with open(BIN, "r+b") as fb:
        fb.seek(12)
        fb.write(struct.pack("<Q", new_total))
    meta["total_tokens"] = new_total
    meta["total_documents"] = orig_docs + added_docs
    meta["total_files"] = meta.get("total_files", orig_docs) + added_docs
    meta["appended_anime_docs"] = added_docs
    META.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"\nCOMMITTED: total_tokens {orig_total:,} -> {new_total:,}  (+{added_tok:,}); docs +{added_docs:,}", flush=True
    )
    print("[done] append OK", flush=True)

except BaseException as e:
    print(f"\nERROR: {e!r}\nROLLING BACK to {L0:,} bytes ...", flush=True)
    try:
        with open(BIN, "r+b") as fb:
            fb.truncate(L0)
            fb.seek(12)
            fb.write(struct.pack("<Q", orig_total))
        if BAK.exists():
            shutil.copy2(BAK, META)
        ok = BIN.stat().st_size == L0
        print(f"rollback complete: size=={L0:,}? {ok}", flush=True)
    except Exception as e2:
        print(f"ROLLBACK ERROR: {e2!r}  (recover via full rebuild from sources if needed)", flush=True)
    raise
