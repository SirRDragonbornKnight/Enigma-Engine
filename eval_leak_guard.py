#!/usr/bin/env python
"""Eval-leak guard for the LOCKED behavior probes (EVAL_REDESIGN.md, 2026-07-16).

Two tiers keep the behavior gate honest:
- DEV probes (`data/eval/behavior_probes.jsonl`) get only the EXACT-match
  backstop in `make_sft_data._eval_probe_questions` -- you may iterate toward
  the dev set (train/dev/test hygiene).
- LOCKED probes are the honest gate you must NEVER train toward. They are
  SEALED as a manifest of hashed content-word shingles + a normalized-string
  hash, so the training build can reject paraphrases of a locked probe WITHOUT
  ever reading the probe plaintext.

Why hashed shingles: catching paraphrases needs a semantic signal (content-word
overlap). Shipping the words verbatim would half-unseal the probes, so each
content word is hashed; Jaccard over the hashed sets equals Jaccard over the
words (collisions negligible), and the plaintext stays sealed.

Seal a locked set:
    python eval_leak_guard.py seal data/eval/locked_probes.jsonl
        -> writes data/eval/locked_probes.manifest.json (commit this; keep the
           .jsonl out of the training-authoring reading path)
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOCKED_MANIFEST = ROOT / "data" / "eval" / "locked_probes.manifest.json"

_WORD = re.compile(r"[a-z0-9]+")
# Light stoplist: drop function words so Jaccard measures CONTENT overlap. A
# probe and its paraphrase share content words ("capital", "france"); the
# scaffolding ("what's", "the", "of") is noise.
_STOP_WORDS = (
    "a an the of to is are am was were be been being do does did done "
    "you your yours i me my mine we us our ours it its this that these those "
    "what whats who whom whose where when which how why and or but if then than "
    "in on at by for with about as from into out up down over under s t re ll ve "
    "can could would should will shall may might must have has had get got"
)
_STOP = frozenset(_STOP_WORDS.split())
DEFAULT_JACCARD = 0.6


def _norm(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def _content_words(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in _STOP and len(w) > 1]


def _h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _hw(w: str) -> str:
    return hashlib.sha1(w.encode("utf-8")).hexdigest()[:12]


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def seal(texts: list[str], threshold: float = DEFAULT_JACCARD) -> dict:
    """Build a sealed manifest from locked-probe plaintext. Ships only hashes,
    never the words."""
    probes = []
    for t in texts:
        probes.append({"h": _h(_norm(t)), "s": sorted({_hw(w) for w in _content_words(t)})})
    return {"version": 1, "jaccard_threshold": threshold, "probes": probes}


class LockedProbeGuard:
    """Rejects training questions that are verbatim or paraphrase-close to any
    LOCKED probe, working only from the sealed manifest. Empty (no manifest) =
    a no-op that leaks nothing, so the build is safe before a locked set exists."""

    def __init__(self, manifest: dict | None = None):
        m = manifest or {}
        self.threshold: float = m.get("jaccard_threshold", DEFAULT_JACCARD)
        self.exact: set[str] = {p["h"] for p in m.get("probes", [])}
        self.shingles: list[set[str]] = [set(p["s"]) for p in m.get("probes", [])]

    @classmethod
    def load(cls, path: Path = LOCKED_MANIFEST) -> "LockedProbeGuard":
        path = Path(path)
        if not path.exists():
            return cls(None)
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def __len__(self) -> int:
        return len(self.shingles)

    def score(self, text: str) -> float:
        """Best similarity of `text` to any locked probe: 1.0 on a verbatim
        (normalized) match, else the max hashed-shingle Jaccard."""
        if not self.shingles:
            return 0.0
        if _h(_norm(text)) in self.exact:
            return 1.0
        q = {_hw(w) for w in _content_words(text)}
        return max((_jaccard(q, s) for s in self.shingles), default=0.0)

    def leaks(self, text: str) -> bool:
        return self.score(text) >= self.threshold

    def is_near_miss(self, text: str, low: float = 0.5) -> bool:
        """In the review band [low, threshold): worth a human glance, not dropped."""
        return low <= self.score(text) < self.threshold


def _cli_seal(src: str) -> int:
    src_path = Path(src)
    if not src_path.exists():
        print(f"ERROR: locked probe file not found: {src_path}")
        return 1
    texts = []
    for line in src_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rec = json.loads(line)
        q = rec.get("q") or rec.get("question") or ""
        if q:
            texts.append(q)
        for fact in rec.get("teach", []):  # memory-probe teach messages count too
            texts.append(fact)
    manifest = seal(texts)
    LOCKED_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    LOCKED_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"sealed {len(texts)} locked probe strings -> {LOCKED_MANIFEST.name} "
          f"(jaccard>={manifest['jaccard_threshold']}); manifest carries hashes only")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[0] == "seal":
        return _cli_seal(argv[1])
    print("usage: python eval_leak_guard.py seal <locked_probes.jsonl>")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
