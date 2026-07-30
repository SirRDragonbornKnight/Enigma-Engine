#!/usr/bin/env python
"""Build the CURATED pretrain shard: the text she should know best.

Three things the last lineage left entirely to SFT, which is the wrong stage
for two of them:

* **World facts.** `knowledge_corpus` already emits ~6.6 surface forms per
  fact (declarative, Q/A, cloze, fact-in-context). Facts install during
  PRETRAINING -- this repo measured that SFT cannot install knowledge -- so
  they belong in the corpus, not only in the instruct bake.
* **Her own identity.** `identity_anchors` holds her voice as chat pairs, and
  a chat pair teaches a FORMAT. Who she is should be in the weights as prose
  before any instruct pass shapes how she says it. Generated from the active
  PERSONA, so a different persona pack produces a different corpus -- which is
  what "the trainer can build another AI" has to mean at the pretrain stage.
* **Short conversational register.** The v1 corpus is documents; every trace of
  dialogue arrived in SFT. A model that never saw a turn until the instruct
  pass has to learn the shape and the content at once.

Everything written here is screened against the SEALED probe manifest first.
`pretokenize_data.py` is the one path with no leak guard of its own, so an
unscreened curated shard would teach the honest gate's answers straight into
the weights -- the exact failure the seal exists to prevent.

The shard is written to its OWN source directory, and `SOURCE_DIRS` walks it
FIRST. Position buys nothing from the IID sampler, but it decides two other
things (both learned by audit, 2026-07-25): VAL membership -- val is carved
off the very END of the bin, so this script's first design (tail position,
for the decay-phase anneal to read) handed the whole shard to val, never
trained on -- and DEDUP precedence -- the paragraph dedup is first-wins, so a
mid-list shard silently loses every paragraph a web source shares. First wins
both. Oversampling is the tokenizer walk's job:

    python make_pretrain_curated.py            # build data/pretrain/curated/
    python make_pretrain_curated.py --stats    # count what it would write
    python pretokenize_data.py --repeat-sources curated=5 ...   # oversample x5

Writing the extra copies HERE would do nothing: pretokenize's paragraph dedup
is global, so on-disk duplicates collapse back to one copy at tokenize time --
silently, and only for paragraphs long enough to be hashed.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from collect_pretraining_data import DOC_SEP_JOIN
from enigma_engine.core.persona import Persona
from eval_leak_guard import LockedProbeGuard
from identity_anchors import EXAMPLES as IDENTITY_EXAMPLES
from knowledge_corpus import gen_knowledge_pretrain_text
from make_sft_data import _is_low_quality

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data" / "pretrain" / "curated"
GENERAL = ROOT / "data" / "finetune" / "combined_finetune.jsonl"

# Documents per .txt shard. Small enough that a bad shard is cheap to drop,
# large enough that the walker is not opening a file per paragraph.
DOCS_PER_SHARD = 2000


def identity_lines(persona: Persona) -> list[str]:
    """Her identity as PROSE, in both persons.

    The anchors are chat pairs, which teach a format. These teach the content:
    the same claims as plain statements a pretrained model can absorb, phrased
    both as she would say them and as someone would say them about her."""
    out: list[str] = []
    name = persona.name
    for _cat, pairs in IDENTITY_EXAMPLES.items():
        for _user, said in pairs:
            said = " ".join(said.split())
            if not said:
                continue
            out.append(said)                                  # first person, as trained
            out.append(f"{name} says: {said}")                # attributed
    if persona.name_meaning:
        out.append(f"{name} is called {name} because it means {persona.name_meaning}.")
    # A handful of flat statements of record. Deliberately plain: this is the
    # shape a model reads a fact out of, not the shape it answers in.
    out += [
        f"{name} is a local AI that runs on the user's own machine.",
        f"{name} runs locally, so nothing said to {name} leaves the machine.",
        f"The AI called {name} is not a cloud service and has no telemetry.",
        f"{name} is a language model that runs on the user's own hardware.",
        f"When you talk to {name}, the conversation stays on your computer.",
    ]
    return out


def conversational_lines(persona: Persona, limit: int, seed: int = 21) -> list[str]:
    """Short dialogue rendered as prose, so the corpus contains turns at all.

    Drawn from the existing general diet rather than invented: made-up chat
    would teach a register nobody speaks.

    Labelled the way `serve_enigma._render_transcript` labels a BASE
    checkpoint -- "User:" and "<name>:" -- because that bridge is how a
    mid-pretrain checkpoint is actually talked to, and T2 produces exactly such
    a checkpoint. Any other label teaches a format nothing consumes. The name
    comes from the persona, so a different AI's corpus carries her own.

    NOT the chat-format control tokens: those are the instruct pass's to
    define, and baking them into pretrain would pin the format before the
    format is decided."""
    if not GENERAL.exists():
        return []
    rng = random.Random(seed)
    pairs: list[tuple[str, str]] = []
    with GENERAL.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = rec.get("prompt") or rec.get("question") or rec.get("instruction")
            a = rec.get("completion") or rec.get("response") or rec.get("answer") or rec.get("output")
            if not (isinstance(q, str) and isinstance(a, str) and q.strip() and a.strip()):
                continue
            if not (len(q) < 400 and len(a) < 800):
                continue
            # The SAME junk gate the SFT build applies to this corpus. It is
            # bulk-pulled and carries raw HTML, encoding damage and source
            # loops; the instruct pass screens them and pretraining had no
            # reason to be less careful -- less, if anything, since a pretrain
            # pass is not something a later stage can undo.
            if _is_low_quality({"messages": [{"role": "assistant", "content": a}]}):
                continue
            pairs.append((" ".join(q.split()), " ".join(a.split())))
    rng.shuffle(pairs)
    return [f"User: {q}\n{persona.name}: {a}" for q, a in pairs[:limit]]


def build(persona: Persona, conv_limit: int) -> dict[str, list[str]]:
    return {
        "facts": gen_knowledge_pretrain_text(),
        "identity": identity_lines(persona),
        "conversation": conversational_lines(persona, conv_limit),
    }


def screen(groups: dict[str, list[str]], guard: LockedProbeGuard) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Drop anything that quotes or paraphrases a sealed probe.

    This is the ONLY screen on the pretrain path: `pretokenize_data.py` reads
    whatever text is on disk."""
    kept: dict[str, list[str]] = {}
    dropped: dict[str, int] = {}
    for name, lines in groups.items():
        good = [ln for ln in lines if not guard.leaks(ln)]
        kept[name] = good
        dropped[name] = len(lines) - len(good)
    return kept, dropped


def write_shards(kept: dict[str, list[str]], out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("curated_*.txt"):
        stale.unlink()  # regenerated wholesale; a stale shard is silent corpus drift
    docs: list[str] = []
    for name in ("facts", "identity", "conversation"):
        docs += kept.get(name, [])
    written = 0
    for i in range(0, len(docs), DOCS_PER_SHARD):
        chunk = docs[i : i + DOCS_PER_SHARD]
        shard = out_dir / f"curated_{i // DOCS_PER_SHARD:04d}.txt"
        # DOC_SEP_JOIN, not "\n\n": a blank line is indistinguishable from a
        # paragraph break, so a shard joined with it reads as ONE document at
        # pretokenize time -- 2,000 identity/fact documents sharing a single
        # <s>/</s> pair. The packed-source separator contract applies to this
        # writer exactly as it does to the HF fetchers.
        shard.write_text(DOC_SEP_JOIN.join(chunk) + "\n", encoding="utf-8")
        written += 1
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--persona", default=None, help="persona pack; omitted = Enigma")
    ap.add_argument("--conversation", type=int, default=20_000,
                    help="short dialogue documents to include")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--stats", action="store_true", help="count only; write nothing")
    args = ap.parse_args()

    persona = Persona.load(Path(args.persona) if args.persona else None)
    guard = LockedProbeGuard.load()
    if not len(guard):
        # Same first line every builder prints for this condition, so a log
        # grep finds them all -- plus the consequence unique to THIS path:
        # pretokenize has no screen of its own, so nothing downstream rechecks.
        print("locked-probe fuzzy guard inactive (no data/eval/locked_probes.manifest.json yet)", flush=True)
        print("WARN: the curated shard would be UNSCREENED -- the pretrain path has no consume-time guard", flush=True)

    groups = build(persona, args.conversation)
    kept, dropped = screen(groups, guard)

    print(f"persona: {persona.name}")
    total_chars = 0
    for name in ("facts", "identity", "conversation"):
        n, d = len(kept.get(name, [])), dropped.get(name, 0)
        chars = sum(len(x) for x in kept.get(name, []))
        total_chars += chars
        print(f"  {name:13} {n:>7,} docs  ({d} dropped as sealed-probe leaks)  {chars:>12,} chars")
    print(f"  {'TOTAL':13} {sum(len(v) for v in kept.values()):>7,} docs  "
          f"{'':>32}{total_chars:>12,} chars")

    if args.stats:
        print("--stats: nothing written")
        return
    out_dir = Path(args.out)
    shards = write_shards(kept, out_dir)
    print(f"wrote {shards} shard(s) -> {out_dir}")
    print("one copy on disk; oversample at tokenize time with "
          "pretokenize_data.py --repeat-sources curated=N (on-disk copies would "
          "be collapsed by the global paragraph dedup)")


if __name__ == "__main__":
    main()
