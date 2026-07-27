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
        --out models/enigma_pretrain_facts --tokens 60e6 --lr 1e-4 --warmup 50 \\
        --val-general-end 0

The output is a standard ETOK file (same header/layout as tokens.bin, doc
layout <bos> content <eos> <eos>). The LAST --val-reserve tokens are pure
replay, so pretrain's [val] window measures general-domain retention, not
fact memorization -- val-reserve is auto-raised to cover pretrain's n//100
[val] window, and no fact doc is allowed to cross into that tail. Pass
--val-general-end 0: the live run's val-gen index points into the 56.6B
corpus tail and is meaningless on this small mixed stream.
"""

from __future__ import annotations

import argparse
import json
import random
import struct
import time
from pathlib import Path

import numpy as np

from enigma_engine.core.tokenizer import get_tokenizer, vocab_file_for_size
from eval_leak_guard import LockedProbeGuard
from knowledge_corpus import gen_knowledge_pretrain_text

ROOT = Path(__file__).resolve().parent
SOURCE_BIN = ROOT / "data" / "pretrain" / "tokens.bin"
OUT_BIN = ROOT / "data" / "pretrain" / "facts_tokens.bin"
HEADER_SIZE = 256
# Fallback only: the real eos id follows the replay corpus's sidecar (set in
# main), never this constant -- a hardcoded 2 was a second source of truth
# for the ETOK format, unvalidated while pretokenize derives AND checks its
# own. Kept for the library callers that build docs without a corpus.
EOS_ID = 2


def eos_from_sidecar(meta: dict, vocab_size: int) -> int:
    """The corpus's eos id, refused unless it is a PLAIN in-range int.

    int() was the first guard here and it laundered hand-edit damage: a JSON
    float 2.7 truncated to 2 and `true` became 1, both silently -- the exact
    "stale or hand-edited sidecar" shapes the refusal message claimed to
    catch (convergence audit, 2026-07-26). A missing key keeps the legacy
    default; a present-but-wrong one refuses."""
    raw = meta.get("eos_token_id", EOS_ID)
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise SystemExit(
            f"sidecar eos_token_id is not a plain integer ({raw!r}) -- stale or "
            "hand-edited sidecar; refusing to write a corpus with it"
        )
    if not 0 <= raw < vocab_size:
        raise SystemExit(
            f"sidecar eos_token_id {raw} is outside vocab {vocab_size} -- "
            "stale or foreign sidecar; refusing to write a corpus with it"
        )
    return raw


REVIEW_DIR = ROOT / "data" / "pretrain"


def screen_locked_probes(lines: list[str], guard: "LockedProbeGuard",
                         review_dir: Path | None = None) -> list[str]:
    """Drop fact lines that are verbatim or paraphrase-close to a LOCKED probe.

    The knowledge stream is the one place that can teach a locked probe's answer
    straight into the weights, and pretraining is exactly where knowledge gets
    installed -- so an unscreened facts corpus silently converts the honest gate
    into a memorization test. Empty manifest = no-op, matching the SFT build's
    `_held_out`.

    Counts go to the console, RECORDS go to a file: a dropped line scores >=
    threshold against a sealed probe, so printing the text would put the sealed
    set's content words (and their answers) into any pasted build log. Same
    split make_sft_data uses for its review band.
    """
    if not len(guard):
        print("locked-probe fuzzy guard inactive (no data/eval/locked_probes.manifest.json yet)")
        return list(lines)
    kept, dropped, near = [], [], []
    for line in lines:
        if guard.leaks(line):
            dropped.append(line)
            continue
        if guard.is_near_miss(line):
            near.append(line)
        kept.append(line)
    print(f"locked-probe fuzzy guard ACTIVE: {len(guard)} sealed probes "
          f"(jaccard >= {guard.threshold})")
    print(f"fact lines: {len(kept)} kept, {len(dropped)} dropped as locked-probe leaks, "
          f"{len(near)} in the review band (kept)")
    if dropped:
        # Dropping lines does NOT change the fact-token fraction -- the budget in
        # `interleave` is a token count -- but the same number of insertions now
        # cycles a SMALLER pool, so every surviving fact repeats more often. That
        # is memorization pressure moving onto the unscreened facts, so say it.
        lift = len(lines) / len(kept) if kept else float("inf")
        print(f"  per-fact repetition rises ~{lift:.2f}x (same insertion budget, "
              f"{len(kept)} docs instead of {len(lines)})")
    out_dir = review_dir if review_dir is not None else REVIEW_DIR
    for name, rows in (("locked_dropped_facts.jsonl", dropped),
                       ("locked_near_miss_facts.jsonl", near)):
        path = out_dir / name
        if rows:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(json.dumps({"text": r}, ensure_ascii=False) for r in rows) + "\n",
                            encoding="utf-8")
            print(f"  wrote {path} ({len(rows)} lines) -- review it, do not annotate it (regenerated every run)")
        elif path.exists():
            path.unlink()  # stale file from an earlier run
    return kept


def tokenize_fact_docs(lines: list[str], tokenizer, vocab_size: int,
                       eos_id: int = EOS_ID) -> list[list[int]]:
    """Each line becomes one document: <bos> content <eos> <eos> (the
    pretokenize_data.py layout). Refuses out-of-vocab ids loudly."""
    docs = []
    for line in lines:
        ids = list(tokenizer.encode(line)) + [eos_id]
        bad = [t for t in ids if not (0 <= t < vocab_size)]
        if bad:
            raise SystemExit(f"fact line tokenized outside vocab {vocab_size}: {line[:60]!r} -> {bad[:5]}")
        docs.append(ids)
    return docs


def interleave(fact_docs: list[list[int]], replay, target_tokens: int,
               fact_frac: float, val_reserve: int, chunk: int, seed: int,
               dtype=np.uint32) -> np.ndarray:
    """Deterministic stream: replay chunks with fact docs inserted at the
    cadence that yields ~fact_frac fact tokens, and a pure-replay tail of
    val_reserve tokens. `replay` is any indexable integer sequence; `dtype`
    must match the source corpus (v1 uint32, v2 uint16) or the written corpus
    reads back as garbage."""
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
    out = np.empty(target_tokens + chunk + 512, dtype=dtype)  # slack, trimmed at end
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
        # Fence: never let a fact doc cross mixed_end into the pure-replay tail
        # (the [val] window). A doc inserted just under mixed_end used to spill
        # fact tokens past the fence, so [val] read ~0.4% facts and stopped
        # purely measuring general retention (final audit 2026-07-16 M3). The
        # leftover space fills as replay in the tail loop below.
        if pos + len(doc) > mixed_end:
            break
        fact_i += 1
        if fact_i % len(order) == 0:
            rng.shuffle(order)
        out[pos : pos + len(doc)] = np.asarray(doc, dtype=dtype)
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


def write_etok(tokens: np.ndarray, out_bin: Path, vocab_size: int, n_docs: int,
               eos_id: int = EOS_ID) -> None:
    bpt = tokens.dtype.itemsize  # must match the array, not a hardcoded width
    header = struct.pack("<4sIIQII", b"ETOK", 1, bpt, len(tokens), vocab_size, eos_id)
    tmp = out_bin.with_suffix(".bin.tmp")
    with open(tmp, "wb") as f:
        f.write(header)
        f.write(b"\x00" * (HEADER_SIZE - len(header)))
        tokens.tofile(f)
    tmp.replace(out_bin)
    meta = {
        "format": "ETOK",
        "version": 1,
        "dtype": str(tokens.dtype),
        "header_size": HEADER_SIZE,
        "total_tokens": len(tokens),
        "total_documents": n_docs,
        "total_files": 1,
        "vocab_size": vocab_size,
        "eos_token_id": eos_id,
        "tokenizer": "AdvancedBPETokenizer",
        "file_size_gb": round(len(tokens) * tokens.dtype.itemsize / 1024**3, 2),
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
    ap.add_argument("--seed", type=int, default=99,
                    help="insertion-cadence shuffle seed; the stream is deterministic per seed")
    ap.add_argument("--out", default=str(OUT_BIN),
                    help="output .bin (sidecar json lands beside it)")
    ap.add_argument(
        "--source-bin",
        default=str(SOURCE_BIN),
        help="replay corpus (its sidecar sets dtype + vocab). Default: the v1 tokens.bin",
    )
    args = ap.parse_args()

    source_bin = Path(args.source_bin)
    if not source_bin.exists():
        raise SystemExit(f"missing replay source: {source_bin}")
    src_meta = json.loads(source_bin.with_suffix(".json").read_text(encoding="utf-8"))
    # Every sidecar key this writer trusts gets the eos treatment: a plain
    # in-range value or a refusal. A hand-edited vocab_size or dtype would
    # otherwise reinterpret the replay corpus silently.
    vocab_size = src_meta.get("vocab_size")
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size <= 0:
        raise SystemExit(
            f"sidecar vocab_size is not a positive integer ({vocab_size!r}) -- stale "
            "or hand-edited sidecar; refusing to build a facts corpus with it"
        )
    # dtype and tokenizer both follow the CORPUS, never a hardcoded default:
    # replaying v2 tokens through the v1 vocab would emit a corpus whose ids
    # mean nothing to the model that trains on it.
    raw_dtype = src_meta.get("dtype", "uint32")
    if raw_dtype not in ("uint16", "uint32"):
        raise SystemExit(
            f"sidecar dtype {raw_dtype!r} is not a token dtype (uint16/uint32) -- "
            "stale or hand-edited sidecar; refusing to reinterpret the replay "
            "corpus with it"
        )
    src_dtype = np.dtype(raw_dtype)
    # eos follows the corpus too: the sidecar records what pretokenize derived
    # and validated, and this writer stamps eos into its own ETOK header.
    eos_id = eos_from_sidecar(src_meta, vocab_size)
    try:
        vocab_file = vocab_file_for_size(vocab_size)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"cannot pick a tokenizer for this corpus: {exc}")
    print(f"replay: {source_bin.name} | dtype {src_dtype} | vocab {vocab_size} -> {vocab_file.name}", flush=True)

    source_lines = gen_knowledge_pretrain_text()
    if not source_lines:
        raise SystemExit("knowledge_corpus produced no fact lines; nothing to build")
    lines = screen_locked_probes(source_lines, LockedProbeGuard.load())
    if not lines:
        raise SystemExit(
            f"all {len(source_lines)} fact lines screened out as locked-probe leaks -- "
            "the sealed set and the knowledge corpus are covering the same ground"
        )
    tokenizer = get_tokenizer("bpe", vocab_path=vocab_file)
    if getattr(tokenizer, "vocab_size", None) != vocab_size:
        raise SystemExit(
            f"tokenizer vocab {getattr(tokenizer, 'vocab_size', '?')} != corpus vocab "
            f"{vocab_size}; refusing to build a facts corpus the model cannot read"
        )
    fact_docs = tokenize_fact_docs(lines, tokenizer, vocab_size, eos_id=eos_id)
    print(f"fact docs: {len(fact_docs)} lines, {sum(len(d) for d in fact_docs):,} tokens")

    replay = np.memmap(source_bin, dtype=src_dtype, mode="r", offset=HEADER_SIZE)
    # stay inside the live run's train region: its val is the corpus tail
    replay_end = len(replay) - 10_000_000
    target = int(args.target_tokens)
    # pretrain's [val] window is the LAST n//100 tokens (min with --val-tokens),
    # so the pure-replay tail must be at least that big or [val] reads fact
    # tokens and stops measuring general-domain retention (final audit
    # 2026-07-16 M3). n ~= target_tokens here (the tail fills to target).
    val_reserve = max(args.val_reserve, target // 100)
    if val_reserve != args.val_reserve:
        print(f"val-reserve raised {args.val_reserve:,} -> {val_reserve:,} "
              f"to cover pretrain's n//100 [val] window")
    tokens = interleave(fact_docs, replay[:replay_end], target,
                        args.fact_frac, val_reserve, args.chunk, args.seed,
                        dtype=src_dtype)
    write_etok(tokens, Path(args.out), vocab_size, n_docs=len(fact_docs), eos_id=eos_id)


if __name__ == "__main__":
    main()
