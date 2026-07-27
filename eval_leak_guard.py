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


# Words per sealed n-gram. A run of this many CONTENT words in order is the
# unit of "this text quotes a probe". Probes shorter than NGRAM_MIN are not
# sealed as runs at all and keep only the exact/jaccard tests.
#
# Floor history. The first sweep (2026-07-25) chose 3, reading min=2's extra
# ask hits as false-positive cost and "one or two words in common" as not a
# quotation. The round-7 audit then showed what that floor leaves open: 18 of
# the 108 sealed strings have exactly TWO content words -- five of them locked
# MEMORY TEACH lines -- and at floor 3 they are sealed as NOTHING, so a
# training turn can carry a sealed teach fact whole and the guard cannot see
# it. Re-measured on the live artifacts (2026-07-25, this session), floor 3
# vs floor 2:
#
#   run coverage                 85/108 -> 103/108 (the 5 one-word strings stay
#                                                   exact/jaccard-only: a 1-word
#                                                   "run" tests membership, not
#                                                   order, and is no quotation)
#   mix asks (120,791)            5 -> 301 leak    (dropped at next rebuild --
#                                                   `make_sft_data._held_out`
#                                                   screens with THIS predicate,
#                                                   so the remedy works)
#   mix answers (116,769)        72 -> 131         (advisory: counted, never blocks)
#   combined asks (105,203)      14 -> 58
#   mix turns carrying a sealed   0/10 -> 10/10    (ONE 2-cw teach line occurs
#   2-cw teach line, caught                         verbatim in 10 turns; floor 3
#                                                   saw none of them)
#
# ~300 records of 120k is the price of the memory category measuring MEMORY
# instead of pretraining. A 2-word run does fire on innocent phrase reuse --
# that cost lands on the build as drops, not on the gate as blindness. A
# consume-time refusal on a pre-floor-2 artifact is stale data, not a false
# alarm: rebuild it.
#
# Honest limit (fix-arc audit, 2026-07-25): a single 2-word run has ZERO
# redundancy -- one inflection ("work" -> "worked") or one interposed content
# word breaks it, and at that point jaccard needs a near-bare sentence to
# fire. Floor 2 buys VERBATIM carry, not paraphrase-proofing; making the 18
# short strings paraphrase-proof means giving them DISTINCTIVE content at a
# probe-content re-seal, which edits the holdout and is the user's call.
NGRAM_N = 4
NGRAM_MIN = 2


def _runs(words: list[str], n: int) -> set[str]:
    return {_hw(" ".join(words[i:i + n])) for i in range(len(words) - n + 1)}


def _probe_ngrams(text: str) -> set[str]:
    """The runs to SEAL for one probe.

    A probe of four or more content words is sealed as its 4-word runs. A
    shorter one is sealed as the single run it is, down to two words, so a
    two-word probe is still quotable rather than unscreened."""
    words = _content_words(text)
    if len(words) < NGRAM_MIN:
        return set()
    return _runs(words, min(NGRAM_N, len(words)))


def _text_ngrams(text: str) -> set[str]:
    """The runs to TEST a candidate text against, at every sealed length.

    The set-based tests could not screen a quotation without a length rule, and
    every length rule was wrong somewhere. A ratio (jaccard) shrinks when the
    quote is padded, so a verbatim probe plus filler slipped through. Raw
    containment does not shrink, but having no order it fires on any long
    document that happens to reuse the words -- a 1407-word record matched a
    6-word probe that way.

    An ordered run has neither failure. Padding cannot remove a run that is
    present, so dilution is powerless at ANY probe length -- which is what the
    58 of 108 sealed strings below four content words actually needed -- and
    reproducing a probe's own word ORDER is something unrelated documents do
    not do by accident."""
    words = _content_words(text)
    out: set[str] = set()
    for n in range(NGRAM_MIN, NGRAM_N + 1):
        if len(words) >= n:
            out |= _runs(words, n)
    return out


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


def file_digest(path: Path) -> str:
    """Digest of a probe file's CONTENT, line-endings normalized.

    This is the gate's IDENTITY. Every hash-set test answers "does this file
    mean the same thing", which is the wrong question for identity and has now
    been evaded once per audit round through whichever normalization dimension
    was left over -- case, punctuation, non-Latin script, and finally
    whitespace, where doubling every space kept the seal intact while changing
    the bytes posted to the model for all 96 questions. Bytes have no
    dimensions left to evade.

    Line endings are normalized because the repo normalizes them on checkout,
    so raw bytes would make the same sealed blob hash differently on two
    clones."""
    raw = Path(path).read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(raw).hexdigest()


def seal(texts: list[str], threshold: float = DEFAULT_JACCARD,
         cases: list[dict] | None = None, probe_file: Path | None = None) -> dict:
    """Build a sealed manifest from locked-probe plaintext. Ships only hashes,
    never the words.

    Pass `cases` (the parsed probe records) to seal the grading keys too. A
    manifest without `grading_digest` can only prove the QUESTIONS are intact,
    which is half a gate -- eval_behavior refuses to treat such a manifest as
    a gate until it is re-sealed."""
    probes = []
    for t in texts:
        probes.append({
            "h": _h(_norm(t)),
            "s": sorted({_hw(w) for w in _content_words(t)}),
            "n": sorted(_probe_ngrams(t)),
        })
    # ngram_min/ngram_n are the RUN parameters this manifest was sealed under.
    # They must ride in the manifest because the trainers' consume-time guard
    # never re-seals: when the code's floor moved 3 -> 2, a stale floor-3
    # manifest under floor-2 code would have kept every 2-content-word string
    # sealed as NOTHING -- the exact hole the move closed -- while printing
    # the same "ACTIVE" banner.
    manifest = {"version": 1, "jaccard_threshold": threshold,
                "ngram_min": NGRAM_MIN, "ngram_n": NGRAM_N, "probes": probes}
    if cases is not None:
        manifest["grading_digest"] = grading_digest(cases)
    if probe_file is not None:
        manifest["probe_file_sha256"] = file_digest(probe_file)
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
        if threshold <= 0:
            # The bound was one-sided, so 0.0 and -1.0 read as "stricter than
            # asked" and made the guard refuse EVERY artifact -- a broken
            # manifest blaming the data it was screening.
            raise self.Weakened(
                f"manifest jaccard_threshold {threshold} would match every string -- "
                "a threshold at or below zero refuses all training data; re-seal it"
            )
        self.threshold: float = threshold
        # The run parameters are enforcement parameters too: seal and test
        # must speak the same unit or a probe short enough to fall between
        # them is sealed as nothing. Manifests that predate the keys but
        # carry runs were all sealed at (3, 4); an empty manifest (no sealed
        # set yet) stays a no-op and is never refused.
        sealed_runs = any(p.get("n") for p in m.get("probes", []))
        sealed_min = m.get("ngram_min", 3 if sealed_runs else NGRAM_MIN)
        sealed_n = m.get("ngram_n", 4 if sealed_runs else NGRAM_N)
        if sealed_runs and (sealed_min, sealed_n) != (NGRAM_MIN, NGRAM_N):
            raise self.Weakened(
                f"manifest sealed word runs at floor {sealed_min} / length {sealed_n} "
                f"but the code tests at {NGRAM_MIN}/{NGRAM_N} -- strings that fall "
                "between the two are sealed as nothing while the guard still prints "
                "ACTIVE; re-seal the locked set"
            )
        # The ARRAYS are enforcement payload, same as the parameters: emptying
        # `n` (or `s`) passes every digest -- the seal comparison is over the
        # exact-hash lists and the file digest covers the PLAINTEXT, not this
        # sidecar -- so an edit could strip the quotation or paraphrase tier
        # while the banner still printed ACTIVE (fix-arc audit, 2026-07-25;
        # the round-5 jaccard_threshold lesson re-entering through the
        # arrays). Two or more distinct content words ALWAYS seal a run at
        # floor 2, and a run always implies a shingle set, so either
        # inconsistency is an edit, not a shape seal() can produce.
        for p in m.get("probes", []):
            if not all(k in p for k in ("h", "s", "n")):
                # A probe entry with fields DELETED (not emptied) used to
                # escape the payload checks below and die as a bare KeyError
                # at construction -- fail-closed, but a traceback where the
                # class promises a refusal (round-C audit, 2026-07-25).
                raise self.Weakened(
                    "manifest probe is missing its h/s/n fields -- an edited or "
                    "truncated entry; re-seal the locked set"
                )
            if len(p.get("s") or ()) >= NGRAM_MIN and not p.get("n"):
                raise self.Weakened(
                    "manifest probe carries a paraphrase set but no sealed runs -- "
                    "its quotation tier has been stripped; re-seal the locked set"
                )
            if p.get("n") and not p.get("s"):
                raise self.Weakened(
                    "manifest probe carries sealed runs but no shingle set -- "
                    "its paraphrase tier has been stripped; re-seal the locked set"
                )
        # Stripping BOTH arrays evades the two one-sided checks above, because
        # s=[] n=[] is also the legitimate shape of a stopword-only probe
        # (round-B audit, 2026-07-25). Without plaintext a loader cannot tell
        # the two apart PER PROBE, but a whole manifest of them can only be a
        # strip or a pre-4-gram seal -- no real locked set is all stopwords.
        # Honest aperture (round-C measured it wider than "partial both-array
        # strip"): ANY array edit that keeps each probe's shape plausible --
        # one shingle kept, runs emptied -- still loads here. The AUTHORITY on
        # array content is the gate run, where eval_behavior recomputes the
        # full h/s/n payload from plaintext and refuses every such edit
        # (verified for full, partial, runs-only, shingles-only and
        # keep-one-shingle strips); the manifest is git-tracked, so the edit
        # is a visible diff either way.
        probes_m = m.get("probes", [])
        if probes_m and not any(p.get("s") for p in probes_m) and not any(p.get("n") for p in probes_m):
            raise self.Weakened(
                "manifest carries probes but no shingle sets and no runs at all -- "
                "exact-hash-only enforcement is a stripped (or pre-4-gram) manifest; "
                "re-seal the locked set"
            )
        self.exact: set[str] = {p["h"] for p in m.get("probes", [])}
        self.shingles: list[set[str]] = [set(p["s"]) for p in m.get("probes", [])]
        # Ordered word runs from every sealed probe -- the QUOTATION test.
        # Padding cannot remove a run that is present, so it does not dilute
        # like a ratio; and a run is ordered, so it does not fire on a long
        # document that merely reuses the same vocabulary. That covers both
        # failures the set-based tests traded against each other, at every
        # probe length, with no floor to tune.
        self.ngrams: set[str] = {n for p in m.get("probes", []) for n in p.get("n", [])}

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
        """True when `text` QUOTES a sealed probe -- reproduces a run of its
        content words in order.

        The dilution answer to `score`: adding words can only lower a ratio, so
        a padded verbatim probe scores under the threshold while still teaching
        the gate. A run does not shrink when the text around it grows, and it
        cannot be reproduced by an unrelated document that happens to share
        vocabulary."""
        if not self.ngrams:
            return False
        return bool(_text_ngrams(text) & self.ngrams)

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
        # ...and OVERWRITE any verdict an earlier run left beside this artifact.
        # Returning without touching it let `last_verdict` hand back a previous
        # run's "108 sealed probes enforced", which finetune then stamped into a
        # checkpoint that had been screened by nothing at all. Absence of a
        # write was being read as a passing result.
        _write_inactive_verdict(source, manifest)
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


def last_verdict(source: Path) -> dict | None:
    """The recorded verdict for `source`, or None if the guard never ran on it.

    Trainers read this to stamp the guard's result into the checkpoint, so a
    finished model carries evidence the screen ran rather than leaving it in a
    console scrollback that redirection can lose. (It once took a `manifest`
    parameter in signature-mimicry of refuse_if_leaky; the verdict lives
    beside the SOURCE and no caller ever passed one.)"""
    path = _verdict_path(source)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _verdict_path(source: Path) -> Path:
    return Path(source).with_name(Path(source).name + ".leakguard.json")


def _write_inactive_verdict(source: Path, manifest: Path) -> None:
    """Record that NOTHING was enforced, replacing any earlier verdict."""
    try:
        _verdict_path(source).write_text(
            json.dumps({"source": str(source), "manifest": str(manifest),
                        "active": False, "sealed_probes": 0,
                        "note": "no sealed probes were available; nothing was enforced"},
                       indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        print(f"WARN: could not clear the leak-guard verdict beside {source} ({exc})", flush=True)


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
    try:
        source_sha = file_digest(source)
    except OSError:
        source_sha = None
    verdict = {
        "source": str(source),
        # A receipt that can be inferred is a receipt that can be wrong: without
        # the artifact's own digest, a verdict beside a file says nothing about
        # WHICH bytes were screened.
        "source_sha256": source_sha,
        "active": True,
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


# Every top-level key a locked probe record may carry. Anything else is
# refused at seal time: an unknown field ("#": "note", "comment": ...) rides
# inside probe_file_sha256 while entering NO sealed hash and NOT the grading
# digest -- the same uncovered-annotation hazard as a comment line, wearing
# JSON (convergence audit, 2026-07-26).
_PROBE_KEYS = frozenset(
    {"q", "question", "category", "want_any", "deny_any", "expect_tool", "teach"}
)


def _cli_seal(src: str) -> int:
    src_path = Path(src)
    if not src_path.exists():
        print(f"ERROR: locked probe file not found: {src_path}")
        return 1
    raw = src_path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        # A BOM survives strip(), dodges the '#' check below, and would ride
        # inside the byte seal covered by no hash -- the Windows-editor way
        # into the same annotation hazard (convergence audit, 2026-07-26).
        print("ERROR: file starts with a UTF-8 BOM -- re-save it plain "
              "(the BOM would ride inside the byte seal uncovered by any hash)")
        return 1
    texts = []
    cases = []
    for line in raw.decode("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            # REFUSE, never skip: a skipped comment enters the file's BYTE
            # identity (probe_file_sha256) without entering any sealed hash,
            # so a commented file could seal and gate-run while carrying
            # editable text the seal does not cover -- and the docs' claim
            # that "a commented holdout can never match its seal" was only
            # true if this refusal exists (2026-07-26 audit).
            print("ERROR: comment line in the locked probe file -- author pure JSONL "
                  "(comments would ride inside the byte seal uncovered by any hash)")
            return 1
        rec = json.loads(line)
        unknown = set(rec) - _PROBE_KEYS
        if unknown:
            print(f"ERROR: unknown probe field(s) {sorted(unknown)} -- they would "
                  "ride inside the byte seal uncovered by any hash; remove them "
                  "(or seal them by extending grading_digest first)")
            return 1
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
    manifest = seal(texts, cases=cases, probe_file=src_path)
    LOCKED_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    LOCKED_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"sealed {len(texts)} locked probe strings -> {LOCKED_MANIFEST.name} "
          f"(jaccard>={manifest['jaccard_threshold']}); manifest carries hashes only")
    print(f"grading keys sealed too (digest {manifest['grading_digest'][:12]}...): "
          "want/deny/expect_tool/category are now tamper-evident without the plaintext")
    print(f"probe file sealed by CONTENT (sha256 {manifest['probe_file_sha256'][:12]}...): "
          "a gate run must be byte-identical to this file, not merely equivalent to it")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[0] == "seal":
        return _cli_seal(argv[1])
    print("usage: python eval_leak_guard.py seal <locked_probes.jsonl>")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
