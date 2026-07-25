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

# Content words a sealed probe needs before CONTAINING it counts as a leak on
# its own. Measured against the live training asks: a floor of 5 still flags 3
# ordinary records, 6 flags none in mix/tool_calls/identity, and a padded
# verbatim probe is caught at every padding level either way. Sealed strings
# shorter than this keep the ratio test only -- a two-word probe is not
# distinctive enough for containment to mean anything. (measured 2026-07-25)
_CONTAINMENT_MIN_WORDS = 6


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


def grading_digest(cases: list[dict]) -> str:
    """A digest over everything that decides a score but is not sealed TEXT.

    `want_any`, `deny_any`, `expect_tool` and `category` never enter the probe
    hashes, yet they decide every verdict: a file with its wants and denies
    emptied re-seals perfectly and then passes any answer at all. Sealing this
    digest INTO the manifest is what lets a run verify them without the
    plaintext on disk -- comparing against an unsealed copy of the file could
    not work, because that copy is exactly what an edit would also change (and
    the canonical run compares the file against itself)."""
    keyed = [
        (
            " ".join(str(c.get("q") or "").split()),
            c.get("category") or "",
            "<absent>" if "expect_tool" not in c else repr(c.get("expect_tool")),
            tuple(sorted(c.get("want_any") or [])),
            tuple(sorted(c.get("deny_any") or [])),
            # Teach CONTENT, hashed (never plaintext), not a count. Every locked
            # memory probe carries exactly one teach line, so a count let all
            # twelve be permuted while the seal still verified -- and the run
            # posts each case's teach lines immediately before that case's
            # question, so a swap changes what the memory category measures.
            #
            # Hashed the way `q` is (whitespace-collapsed, nothing else), NOT
            # through _norm. _norm keeps only [a-z0-9] runs, so sealing teach
            # through it left case, punctuation and every non-Latin script free:
            # the twelve teach lines could be uppercased, or have Cyrillic and
            # emoji appended, and still verify -- while the run POSTS the
            # mutated text to the server. That is an injection channel into the
            # sealed memory probes, not a formatting nicety.
            tuple(_h(" ".join(str(t).split())) for t in (c.get("teach") or [])),
        )
        for c in cases
    ]
    # NOT sorted: file order is part of what the holdout IS. The store is
    # cleared once per run and then accumulates every taught fact, so moving a
    # memory probe past another probe changes what the later one can recall.
    return hashlib.sha256(repr(keyed).encode("utf-8")).hexdigest()


def seal(texts: list[str], threshold: float = DEFAULT_JACCARD,
         cases: list[dict] | None = None) -> dict:
    """Build a sealed manifest from locked-probe plaintext. Ships only hashes,
    never the words.

    Pass `cases` (the parsed probe records) to seal the grading keys too. A
    manifest without `grading_digest` can only prove the QUESTIONS are intact,
    which is half a gate -- eval_behavior refuses to treat such a manifest as
    a gate until it is re-sealed."""
    probes = []
    for t in texts:
        probes.append({"h": _h(_norm(t)), "s": sorted({_hw(w) for w in _content_words(t)})})
    manifest = {"version": 1, "jaccard_threshold": threshold, "probes": probes}
    if cases is not None:
        manifest["grading_digest"] = grading_digest(cases)
    return manifest


class LockedProbeGuard:
    """Rejects training questions that are verbatim or paraphrase-close to any
    LOCKED probe, working only from the sealed manifest. Empty (no manifest) =
    a no-op that leaks nothing, so the build is safe before a locked set exists."""

    class Weakened(ValueError):
        """The manifest asks for less enforcement than the code allows."""

    def __init__(self, manifest: dict | None = None):
        m = manifest or {}
        threshold = m.get("jaccard_threshold", DEFAULT_JACCARD)
        # The threshold is the ONE enforcement parameter that lived in an
        # editable sidecar while nothing verified it: the probe hashes and the
        # grading digest are identical under any threshold, so eval still
        # printed "seal verified" over a manifest edited to 0.99 -- and every
        # paraphrase then trained freely. Raising it is purely a weakening, so
        # a manifest above the code default is refused rather than obeyed.
        # Below it is stricter than the code asks for and is honoured.
        if not isinstance(threshold, (int, float)) or threshold != threshold:
            raise self.Weakened(
                f"manifest jaccard_threshold is not a number ({threshold!r}); re-seal it"
            )
        if threshold > DEFAULT_JACCARD:
            raise self.Weakened(
                f"manifest jaccard_threshold {threshold} is weaker than the code default "
                f"{DEFAULT_JACCARD} -- a raised threshold silently stops refusing "
                "paraphrases of sealed probes; restore it or re-seal deliberately"
            )
        self.threshold: float = threshold
        self.exact: set[str] = {p["h"] for p in m.get("probes", [])}
        self.shingles: list[set[str]] = [set(p["s"]) for p in m.get("probes", [])]
        # Probes distinctive enough that CONTAINING one is itself the leak.
        # Jaccard is a ratio, so padding a verbatim probe with unrelated words
        # shrinks it below the threshold while the probe sits intact inside the
        # text (measured: a 10-word probe drops to 0.529 at 8 filler words and
        # trains freely). Containment does not divide by the added words, so it
        # cannot be diluted -- but it is only safe above a length floor, since
        # any text mentioning a 2-word probe's words "contains" it.
        self.long_shingles: list[set[str]] = [
            s for s in self.shingles if len(s) >= _CONTAINMENT_MIN_WORDS
        ]

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

    def contains_probe(self, text: str) -> bool:
        """True when a distinctive sealed probe sits INTACT inside `text`.

        The dilution answer to `score`: adding words can only lower a ratio, so
        a padded verbatim probe scores under the threshold while still teaching
        the gate. Restricted to probes of at least `_CONTAINMENT_MIN_WORDS`
        content words -- below that a probe is not distinctive enough for
        containment to mean anything."""
        if not self.long_shingles:
            return False
        t = {_hw(w) for w in _content_words(text)}
        return any(s <= t for s in self.long_shingles)

    def leaks(self, text: str) -> bool:
        return self.score(text) >= self.threshold or self.contains_probe(text)

    def is_near_miss(self, text: str, low: float = 0.5) -> bool:
        """In the review band [low, threshold): worth a human glance, not dropped."""
        return low <= self.score(text) < self.threshold


def refuse_if_leaky(texts: list[str], source: Path, manifest: Path = LOCKED_MANIFEST,
                    advisory: list[str] | None = None) -> None:
    """Refuse to TRAIN on an artifact whose ASKS match a sealed probe.

    The build-time screens only clean data as it is generated; a trainer reads
    whatever file is on disk, and a pre-seal artifact left there keeps its
    leaks forever. Checking at consume time is what makes the seal binding on
    the run that actually touches the weights.

    `texts` are the prompt-side strings and they REFUSE, using the same
    predicate `make_sft_data._held_out` screens with -- so "rebuild the
    artifact" is advice that actually works.

    `advisory` are answer-side strings (assistant turns, DPO chosen/rejected)
    and they only WARN. Scanning them as leaks was tried and reverted: an
    answer shares most of a question's content words by nature ("Jupiter is the
    largest planet..." scores 0.67 against "What's the largest planet?"), so at
    this threshold the check cannot separate a leak from topicality. It flagged
    56 assistant turns in the live SFT mix -- blocking the entire queued
    training block behind advice that could not clear it, because the builder
    screens the question side only. Counting them out loud keeps the signal
    without the deadlock.

    Raises SystemExit naming counts -- never the leaking text, which is sealed
    content."""
    guard = LockedProbeGuard.load(manifest)
    if not len(guard):
        # No sealed set yet: nothing to enforce, and nothing to leak. SAY SO --
        # this returned in total silence, so a training log could not tell a
        # clean run from one where the guard never ran at all (a missing or
        # emptied manifest reads exactly like success). The build-time screens
        # already announce their own no-op; this is the same courtesy on the
        # path that actually touches the weights.
        print(f"leak guard: INACTIVE for {source} (no sealed probes at {manifest}); "
              "nothing is being enforced", flush=True)
        return
    leaks = sum(1 for t in texts if guard.leaks(t))
    if leaks:
        raise SystemExit(
            f"REFUSING to train: {source} carries {leaks} of {len(texts)} ASKS that match a "
            f"SEALED locked probe (jaccard >= {guard.threshold}). Training on it would rig the "
            f"gate. Rebuild the artifact with the current guard, then re-run."
        )
    note = ""
    flagged_texts: list[str] = []
    if advisory:
        flagged_texts = [t for t in advisory if guard.leaks(t)]
        if flagged_texts:
            note = (f"; {len(flagged_texts)} of {len(advisory)} answer-side strings sit at or "
                    "above the threshold -- expected on shared topics, reviewed not blocked")
    print(f"leak guard: {len(texts)} asks clean against {len(guard)} sealed probes{note}",
          flush=True)
    _write_verdict(source, manifest, guard, len(texts), advisory or [], flagged_texts)


def last_verdict(source: Path, manifest: Path = LOCKED_MANIFEST) -> dict | None:
    """The recorded verdict for `source`, or None if the guard never ran on it.

    Trainers read this to stamp the guard's result into the checkpoint, so a
    finished model carries evidence the screen ran rather than leaving it in a
    console scrollback that redirection can lose."""
    path = _verdict_path(source)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _verdict_path(source: Path) -> Path:
    return Path(source).with_name(Path(source).name + ".leakguard.json")


def _write_verdict(source: Path, manifest: Path, guard: "LockedProbeGuard",
                   n_asks: int, advisory: list[str], flagged: list[str]) -> None:
    """Record the verdict beside the artifact.

    A bare console count is unactionable and vanishes under redirection, and a
    finished checkpoint carried no evidence the screen had run at all. The
    flagged strings are the artifact's OWN text (never sealed plaintext), so
    writing them is what makes the advisory band reviewable -- the same pattern
    the build-time screens already use for dropped and near-miss records."""
    try:
        manifest_sha = hashlib.sha256(Path(manifest).read_bytes()).hexdigest() if Path(manifest).exists() else None
    except OSError:
        manifest_sha = None
    verdict = {
        "source": str(source),
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha,
        "sealed_probes": len(guard),
        "jaccard_threshold": guard.threshold,
        "asks_screened": n_asks,
        "asks_refused": 0,  # a refusal raises before this point
        "answer_side_screened": len(advisory),
        "answer_side_flagged": len(flagged),
        "answer_side_flagged_distinct": len({" ".join(t.split()) for t in flagged}),
        "flagged": sorted({" ".join(t.split()) for t in flagged})[:200],
    }
    try:
        _verdict_path(source).write_text(json.dumps(verdict, indent=2, ensure_ascii=False),
                                         encoding="utf-8")
    except OSError as exc:  # a receipt must never take the training run down
        print(f"WARN: could not write the leak-guard verdict beside {source} ({exc})", flush=True)


def _cli_seal(src: str) -> int:
    src_path = Path(src)
    if not src_path.exists():
        print(f"ERROR: locked probe file not found: {src_path}")
        return 1
    texts = []
    cases = []
    for line in src_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        rec = json.loads(line)
        cases.append(rec)
        q = rec.get("q") or rec.get("question") or ""
        if q:
            texts.append(q)
        for fact in rec.get("teach", []):  # memory-probe teach messages count too
            texts.append(fact)
    # A probe whose content words are ALL stopwords ("Is it you?") produces an
    # empty shingle set: it can only ever match VERBATIM, so paraphrases of it
    # pass the guard silently. Say so at seal time (audit 2026-07-16).
    for t in texts:
        if not _content_words(t):
            print(f"WARN: probe has no content words (verbatim-match only): {t!r}")
    manifest = seal(texts, cases=cases)
    LOCKED_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    LOCKED_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"sealed {len(texts)} locked probe strings -> {LOCKED_MANIFEST.name} "
          f"(jaccard>={manifest['jaccard_threshold']}); manifest carries hashes only")
    print(f"grading keys sealed too (digest {manifest['grading_digest'][:12]}...): "
          "want/deny/expect_tool/category are now tamper-evident without the plaintext")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[0] == "seal":
        return _cli_seal(argv[1])
    print("usage: python eval_leak_guard.py seal <locked_probes.jsonl>")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
