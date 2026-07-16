#!/usr/bin/env python
"""Build the facts continued-pretrain corpus -- install knowledge in weights.

SFT surfaces knowledge; it cannot install it (audit 2026-07-15: "largest
planet" -> Jupiter but "biggest planet" -> Saturn). Installation happens in
pretraining, where the fact appears in MANY textual forms. This script mixes
the ~914 plain-text fact lines from knowledge_corpus.gen_knowledge_pretrain_text
(declarative / QA / cloze / in-context) into a stream of REPLAY chunks sampled
from the real pretrain corpus, so a short low-LR continued-pretrain pass
learns the facts without forgetting the language:

    python make_facts_pretrain_data.py                     # 60M tokens, 2% facts
    python pretrain_enigma.py --tokens-bin data/pretrain/facts_tokens.bin \\
        --init-from models/enigma_pretrain_large/latest.pth \\
        --out models/enigma_pretrain_facts --tokens 60e6 --lr 1e-4 --warmup 50

The output is a standard ETOK file (same header/layout as tokens.bin, doc
layout <bos> content <eos> <eos>). The LAST --val-reserve tokens are pure
replay, so pretrain's [val] window measures general-domain retention, not
fact memorization.
"""

from __future__ import annotations

import argparse
import json
import random
import struct
import time
from pathlib import Path

import numpy as np

from enigma_engine.core.tokenizer import get_tokenizer
from knowledge_corpus import gen_knowledge_pretrain_text

ROOT = Path(__file__).resolve().parent
SOURCE_BIN = ROOT / "data" / "pretrain" / "tokens.bin"
OUT_BIN = ROOT / "data" / "pretrain" / "facts_tokens.bin"
HEADER_SIZE = 256
EOS_ID = 2


def tokenize_fact_docs(lines: list[str], tokenizer, vocab_size: int) -> list[list[int]]:
    """Each line becomes one document: <bos> content <eos> <eos> (the
    pretokenize_data.py layout). Refuses out-of-vocab ids loudly."""
    docs = []
    for line in lines:
        ids = list(tokenizer.encode(line)) + [EOS_ID]
        bad = [t for t in ids if not (0 <= t < vocab_size)]
        if bad:
            raise SystemExit(f"fact line tokenized outside vocab {vocab_size}: {line[:60]!r} -> {bad[:5]}")
        docs.append(ids)
    return docs


def interleave(fact_docs: list[list[int]], replay, target_tokens: int,
               fact_frac: float, val_reserve: int, chunk: int, seed: int) -> np.ndarray:
    """Deterministic stream: replay chunks with fact docs inserted at the
    cadence that yields ~fact_frac fact tokens, and a pure-replay tail of
    val_reserve tokens. `replay` is any indexable uint32 sequence."""
    rng = random.Random(seed)
    n_replay_src = len(replay)
    if n_replay_src < chunk + 1:
        raise SystemExit("replay source too small")

    fact_total = sum(len(d) for d in fact_docs)
    mixed_end = max(0, target_tokens - val_reserve)
    fact_budget = int(mixed_end * fact_frac)
    # cadence: one fact doc per k replay chunks
    n_fact_insertions = max(1, fact_budget // max(1, fact_total // max(1, len(fact_docs))))
    chunks_per_fact = max(1, int((mixed_end - fact_budget) / chunk / max(1, n_fact_insertions)))

    order = list(range(len(fact_docs)))
    out = np.empty(target_tokens + chunk + 512, dtype=np.uint32)  # slack, trimmed at end
    pos = 0
    fact_i = 0
    rng.shuffle(order)
    while pos < mixed_end:
        for _ in range(chunks_per_fact):
            if pos >= mixed_end:
                break
            start = rng.randrange(0, n_replay_src - chunk)
            out[pos : pos + chunk] = replay[start : start + chunk]
            pos += chunk
        if pos >= mixed_end:
            break
        doc = fact_docs[order[fact_i % len(order)]]
        fact_i += 1
        if fact_i % len(order) == 0:
            rng.shuffle(order)
        out[pos : pos + len(doc)] = np.asarray(doc, dtype=np.uint32)
        pos += len(doc)
    mixed_actual = pos
    while pos < target_tokens:  # pure-replay tail (the val window)
        start = rng.randrange(0, n_replay_src - chunk)
        take = min(chunk, target_tokens - pos)
        out[pos : pos + take] = replay[start : start + take]
        pos += take
    n_fact_tokens = fact_i and sum(len(fact_docs[order[i % len(order)]]) for i in range(fact_i))
    print(
        f"stream: {pos:,} tokens ({fact_i} fact-doc insertions, "
        f"~{100.0 * (mixed_actual and n_fact_tokens / mixed_actual):.1f}% fact tokens in the mixed region, "
        f"{target_tokens - mixed_end:,} pure-replay tail)"
    )
    return out[:pos]


def write_etok(tokens: np.ndarray, out_bin: Path, vocab_size: int, n_docs: int) -> None:
    header = struct.pack("<4sIIQII", b"ETOK", 1, 4, len(tokens), vocab_size, EOS_ID)
    tmp = out_bin.with_suffix(".bin.tmp")
    with open(tmp, "wb") as f:
        f.write(header)
        f.write(b"\x00" * (HEADER_SIZE - len(header)))
        tokens.tofile(f)
    tmp.replace(out_bin)
    meta = {
        "format": "ETOK",
        "version": 1,
        "dtype": "uint32",
        "header_size": HEADER_SIZE,
        "total_tokens": len(tokens),
        "total_documents": n_docs,
        "total_files": 1,
        "vocab_size": vocab_size,
        "eos_token_id": EOS_ID,
        "tokenizer": "AdvancedBPETokenizer",
        "file_size_gb": round(len(tokens) * 4 / 1024**3, 2),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "facts continued-pretrain mix (make_facts_pretrain_data.py)",
    }
    out_bin.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {out_bin} ({meta['file_size_gb']} GB) + sidecar json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-tokens", type=float, default=60e6)
    ap.add_argument("--fact-frac", type=float, default=0.02, help="fraction of mixed-region tokens that are fact text")
    ap.add_argument("--val-reserve", type=int, default=500_000, help="pure-replay tail for the [val] window")
    ap.add_argument("--chunk", type=int, default=2048, help="replay slice length in tokens")
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--out", default=str(OUT_BIN))
    args = ap.parse_args()

    if not SOURCE_BIN.exists():
        raise SystemExit(f"missing replay source: {SOURCE_BIN}")
    src_meta = json.loads(SOURCE_BIN.with_suffix(".json").read_text(encoding="utf-8"))
    vocab_size = src_meta["vocab_size"]

    lines = gen_knowledge_pretrain_text()
    tokenizer = get_tokenizer("bpe")
    fact_docs = tokenize_fact_docs(lines, tokenizer, vocab_size)
    print(f"fact docs: {len(fact_docs)} lines, {sum(len(d) for d in fact_docs):,} tokens")

    replay = np.memmap(SOURCE_BIN, dtype=np.uint32, mode="r", offset=HEADER_SIZE)
    # stay inside the live run's train region: its val is the corpus tail
    replay_end = len(replay) - 10_000_000
    tokens = interleave(fact_docs, replay[:replay_end], int(args.target_tokens),
                        args.fact_frac, args.val_reserve, args.chunk, args.seed)
    write_etok(tokens, Path(args.out), vocab_size, n_docs=len(fact_docs))


if __name__ == "__main__":
    main()
