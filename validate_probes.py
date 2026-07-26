#!/usr/bin/env python
"""Check a locked-probe file BEFORE it is sealed.

Sealing is permanent: the manifest hashes the questions, the training builder
starts dropping records that resemble them, and nothing can be added later. A
probe that is malformed in the right way does not announce itself -- it scores
0% forever, or passes on any output at all, and the number lands on the
scorecard looking like a fact about the model.

Everything here runs locally and prints only to this terminal. Nothing is sent
anywhere, and the file is never modified.

    python validate_probes.py data/eval/locked_probes.jsonl

ERRORs corrupt the sealed set and must be fixed. WARNs are probes that will
grade in a way you probably did not intend.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from eval_behavior import INFORMATIONAL_CATEGORIES, THRESHOLDS, _grade_identity  # noqa: E402
from eval_leak_guard import LockedProbeGuard, seal as _seal_texts  # noqa: E402
from serve_enigma import _MEMORABLE, _looks_arithmetic  # noqa: E402

# Gated categories PLUS the ones declared measured-but-ungated. A category
# outside both is a typo: it would be reported informationally and decide
# nothing, which is not what an author writing a probe intends.
CATEGORIES = set(THRESHOLDS) | set(INFORMATIONAL_CATEGORIES)
TOOL_CATEGORIES = {"tool", "restraint"}
KNOWN_KEYS = {"category", "q", "want_any", "deny_any", "teach", "expect_tool", "note"}
# The one client tool the eval injects for tool/restraint probes.
VALID_EXPECT_TOOLS = {None, "get_weather"}
# Organ probes call the server's OWN built-ins instead: no client tool is
# injected, so what they measure is whether her intent gate offers the organ.
ORGAN_EXPECT_TOOLS = {None, "speak", "imagine", "remember", "forget", "calculate"}

# A deny key only stays a deny key if a CORRECT answer cannot contain it.
# "just a wrapper" fires on "I am not just a wrapper"; "i am a wrapper" does
# not, because the negation breaks the phrase.
_AFFIRMING = ("yes", "i am", "i'm", "that is right", "that's right", "correct", "indeed", "sure")

_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "do", "does", "did", "you", "your",
    "i", "me", "my", "what", "who", "where", "when", "why", "how", "to", "of", "in",
    "on", "for", "and", "or", "it", "this", "that", "am", "can", "be",
}


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in _STOP}


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", text.lower()))


def _load_training_questions(exclude: Path | None = None) -> tuple[set[str], list[str], list[str]]:
    """(normalized user turns, RAW user turns, sources scanned). The raw list
    feeds the fuzzy scan; the sources list keeps the scan honest -- a missing
    mix file must be REPORTED, not silently scanned as empty.

    `exclude` is the file under validation. Without it, checking the dev set
    compared that file against a corpus CONTAINING it, so every probe reported
    itself as already-in-training and 150 real findings hid behind the noise --
    the same compare-a-file-with-itself shape that once let a seal check pass
    over unverified grading keys."""
    seen: set[str] = set()
    raw: list[str] = []
    scanned: list[str] = []
    for rel in ("data/sft/mix.jsonl", "data/eval/behavior_probes.jsonl"):
        path = ROOT / rel
        if not path.exists():
            continue
        if exclude is not None and path.resolve() == Path(exclude).resolve():
            continue
        scanned.append(rel)
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for text in (
                    [rec.get("q")] if rec.get("q") else []
                ) + [m.get("content") for m in rec.get("messages") or [] if m.get("role") == "user"] + (
                    [rec.get("prompt")] if rec.get("prompt") else []
                ):
                    if text:
                        seen.add(_norm(text))
                        raw.append(text)
    return seen, raw, scanned


def _fuzzy_leak_counts(questions: list[str], training_raw: list[str]) -> tuple[dict[int, int], int]:
    """(per-probe count of DISTINCT training questions each probe would delete,
    total DISTINCT records deleted across all probes).

    The seal enforces content-word Jaccard >= 0.6, not exact match (audit
    2026-07-22: an exact-only validator scan blessed a pool whose seal would
    delete 252 training records). Counting DISTINCT normalized questions, not
    raw records: a probe endangering one question repeated 25x is not as
    dangerous as one endangering 25 distinct questions, and the hard-error
    keys on that distinction (audit r3). The total is a set union, so a record
    two probes both match is not double-counted (audit r3)."""
    guard = LockedProbeGuard(_seal_texts(questions))
    single = [LockedProbeGuard(_seal_texts([q])) for q in questions]
    per_probe: dict[int, set[str]] = {}
    deleted: set[str] = set()
    for text in training_raw:
        if not guard.leaks(text):
            continue
        norm = _norm(text)
        deleted.add(norm)
        for idx, g in enumerate(single):
            if g.leaks(text):
                per_probe.setdefault(idx, set()).add(norm)
    return {idx: len(qs) for idx, qs in per_probe.items()}, len(deleted)


def check(path: Path, skip_leak: bool = False) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warns: list[str] = []
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        errors.append("file:1 saved as UTF-8 with BOM; re-save as plain UTF-8")
    text = raw.decode("utf-8-sig", errors="replace")

    records: list[tuple[int, dict]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            # The sealer and the eval BOTH skip these now, but the gate's
            # identity is the file's BYTES -- a commented holdout can never
            # match its seal, so refusing at authoring time is the honest gate.
            errors.append(f"line {lineno}: comment line -- a commented file can never match its byte seal")
            continue
        try:
            rec = json.loads(stripped)
        except json.JSONDecodeError as exc:
            errors.append(f"line {lineno}: not valid JSON ({exc.msg})")
            continue
        if not isinstance(rec, dict):
            errors.append(f"line {lineno}: expected one JSON object per line")
            continue
        records.append((lineno, rec))

    counts: dict[str, int] = {}
    seen_questions: dict[str, int] = {}
    if skip_leak:
        training, training_raw, scanned = set(), [], []
    else:
        training, training_raw, scanned = _load_training_questions(exclude=path)
        if "data/sft/mix.jsonl" not in scanned:
            warns.append(
                "training-overlap scan ran WITHOUT data/sft/mix.jsonl (missing) -- "
                "leak results are vacuous; rebuild the mix first"
            )

    for lineno, rec in records:
        cat = rec.get("category")
        question = rec.get("q") or ""
        where = f"line {lineno}"

        if cat not in CATEGORIES:
            errors.append(
                f"{where}: category {cat!r} is not one of {sorted(CATEGORIES)} -- "
                "an undeclared category is reported informationally and gates nothing"
            )
            continue
        counts[cat] = counts.get(cat, 0) + 1

        unknown = set(rec) - KNOWN_KEYS
        if unknown:
            errors.append(
                f"{where}: unknown key(s) {sorted(unknown)} -- a misspelled 'want_any' "
                "leaves the want list empty, which passes on any output"
            )
        if not question:
            errors.append(f"{where}: no 'q'")
            continue
        if rec.get("teach") is not None and (
            not isinstance(rec["teach"], list) or not all(isinstance(t, str) for t in rec["teach"])
        ):
            errors.append(f"{where}: 'teach' must be a list of strings")
            continue
        # want/deny are the fields the grader MATCHES on -- a curly quote there
        # is the deadliest place for one (audit 2026-07-22: the earlier check
        # covered only q/teach).
        for field in ("q", "teach", "want_any", "deny_any"):
            value = rec.get(field)
            blob = " ".join(str(v) for v in value) if isinstance(value, list) else (value or "")
            if any(ord(ch) > 127 for ch in blob):
                bad = sorted({ch for ch in blob if ord(ch) > 127})
                errors.append(
                    f"{where}: non-ASCII in {field!r} ({bad}) -- a curly quote from an "
                    "editor never matches a straight one and fails silently forever"
                )

        norm_q = _norm(question)
        if norm_q in seen_questions:
            errors.append(f"{where}: duplicate question, also on line {seen_questions[norm_q]}")
        seen_questions[norm_q] = lineno
        if training and norm_q in training:
            errors.append(
                f"{where}: this exact question is already in the training/dev data -- it "
                "measures memorization, and sealing it deletes the matching training records"
            )

        # Tool-graded by SHAPE, matching the grader. Keying on the category name
        # alone told an organ probe (which grades on `expect_tool` too) that its
        # empty want list "passes on ANY output" -- advice that would have had
        # the author add wants the grader never reads.
        if cat in TOOL_CATEGORIES or "expect_tool" in rec:
            if "expect_tool" not in rec:
                errors.append(
                    f"{where}: {cat} probe has no 'expect_tool' -- it silently grades as a "
                    "restraint probe (expects NO tool call)"
                )
            else:
                valid = VALID_EXPECT_TOOLS if cat in TOOL_CATEGORIES else ORGAN_EXPECT_TOOLS
                if rec.get("expect_tool") not in valid:
                    errors.append(
                        f"{where}: expect_tool {rec.get('expect_tool')!r} is not a tool this "
                        f"category can call ({sorted(t for t in valid if t)}) -- this probe can "
                        "never pass"
                    )
            if rec.get("want_any") or rec.get("deny_any"):
                warns.append(f"{where}: {cat} probes grade on the tool call; want/deny are ignored")
            continue

        want = rec.get("want_any") or []
        deny = rec.get("deny_any") or []
        if not isinstance(want, list) or not isinstance(deny, list):
            errors.append(f"{where}: 'want_any'/'deny_any' must be lists")
            continue
        if not all(isinstance(k, str) for k in want + deny):
            errors.append(f"{where}: every want/deny key must be a string")
            continue
        want = [k for k in want]
        if not want:
            errors.append(f"{where}: empty 'want_any' -- this probe passes on ANY output")
            continue
        blank = [k for k in want + deny if not k.strip()]
        if blank:
            errors.append(
                f"{where}: empty-string want/deny key -- matches at any word boundary, "
                "so the probe passes on ANY output"
            )
            continue

        q_words = _words(question)
        echoable = [w for w in want if _words(w) and _words(w) <= q_words]
        if echoable:
            warns.append(
                f"{where}: want {echoable} appears in the question itself -- she can pass by "
                "echoing it back without knowing the answer"
            )
        for key in deny:
            if _words(key) <= q_words and _words(key):
                warns.append(
                    f"{where}: deny {key!r} is inside the question -- quoting the question fails her"
                )
            elif cat != "unknown" and not any(marker in key.lower() for marker in _AFFIRMING):
                # unknown-category denies are guess-markers ("it was probably")
                # that DELIBERATELY fire on hedged fabrication; a decline that
                # happens to contain one is a false-fail in the safe direction.
                warns.append(
                    f"{where}: deny {key!r} would also fire on a CORRECT denial that repeats the "
                    'phrase; anchor it to an affirmation ("yes, i am ...")'
                )

        if cat in ("identity", "adversarial"):
            if _grade_identity("No.", want, deny):
                warns.append(f"{where}: passes on the single word \"No.\" -- add a distinctive want")
            if _grade_identity("I'm not sure.", want, deny):
                warns.append(f"{where}: passes on \"I'm not sure.\" -- evasion should not score")
            # Single common words as the pass bar are negatable: "small"
            # passes "7 billion parameters, on the small side" (audit
            # 2026-07-22 measured 9 false-passes in a 21-probe sample).
            # Count LITERAL words: the grader matches the whole phrase, so
            # "this machine" is a two-word bar even though "this" is a
            # stopword elsewhere.
            loose = [
                k for k in want
                if len(k.split()) == 1 and k.strip().lower() not in ("enigma", "sirrulean")
                and not any(ch.isdigit() for ch in k)
            ]
            if loose and len(loose) == len(want):
                warns.append(
                    f"{where}: every want is a single generic word {loose} -- a wrong answer "
                    "that mentions the word passes; use a distinctive phrase"
                )

        if cat == "math" and not _looks_arithmetic(question):
            warns.append(
                f"{where}: the calculator is not offered for this phrasing, so she must do it in "
                "her head (0% in weights). Use plus/minus/times/divided or digits with an operator"
            )

        if cat == "memory":
            teach = rec.get("teach") or []
            if not teach:
                errors.append(f"{where}: memory probe has no 'teach' lines")
            for line in teach:
                if not _MEMORABLE.search(line):
                    errors.append(
                        f"{where}: teach line {line!r} never offers the save tool, so the fact is "
                        'never stored and recall CANNOT pass. Use "My X is Y." or "Remember, ..."'
                    )
            for key in want:
                if _words(key) <= q_words and _words(key):
                    warns.append(f"{where}: want {key!r} is in the question; she can echo it")

    if not records:
        errors.append("no probes found in the file")
    else:
        missing = sorted(CATEGORIES - set(counts))
        if missing:
            warns.append(f"no probes at all in categor{'y' if len(missing) == 1 else 'ies'}: {missing}")
        thin = sorted(c for c, n in counts.items() if n < 9)
        if thin:
            warns.append(f"fewer than 9 probes in {thin} -- category scores there are noise")
        if len(records) < 60:
            warns.append(f"{len(records)} probes total; 60-90 is the usable range, and you cannot add later")

        # The FUZZY scan: sealing enforces content-word Jaccard >= 0.6, so a
        # probe that merely RESEMBLES training questions still deletes them
        # from every future build -- and v5/v8 were trained WITH them, rigging
        # the comparison. Run the real guard, not an exact-match lookalike.
        if training_raw:
            questions = [rec.get("q") or "" for _, rec in records]
            leak_counts, total_deleted = _fuzzy_leak_counts(questions, training_raw)
            minor = 0
            for idx, n in sorted(leak_counts.items(), key=lambda kv: -kv[1]):
                lineno, rec = records[idx]
                msg = (
                    f"line {lineno}: sealing this probe deletes {n} DISTINCT training "
                    f"question(s) (fuzzy match) -- q={rec.get('q', '')[:50]!r}"
                )
                if n >= 8:
                    errors.append(msg + " -- too close to trained phrasing; reword it")
                elif n >= 2:
                    warns.append(msg)
                else:
                    minor += 1  # a single endangered question: borderline, aggregate only
            if total_deleted:
                print(
                    f"  fuzzy leak: sealing would delete {total_deleted} distinct training "
                    f"questions ({minor} probe(s) endangering only one, not listed). "
                    "Record this number in EVAL_REDESIGN at seal time."
                )

    print(f"{path.name}: {len(records)} probes")
    for cat in sorted(counts):
        print(f"  {cat:<12} {counts[cat]:>3}")
    return errors, warns


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default=str(ROOT / "data" / "eval" / "locked_probes.jsonl"))
    ap.add_argument(
        "--skip-leak",
        action="store_true",
        help="skip the training-overlap scans (faster, but leaked probes then "
        "validate CLEAN and the exit code no longer reflects them)",
    )
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f"no such file: {path}")

    errors, warns = check(path, skip_leak=args.skip_leak)
    for warn in warns:
        print(f"  WARN  {warn}")
    for err in errors:
        print(f"  ERROR {err}")
    print()
    if errors:
        print(f"{len(errors)} error(s), {len(warns)} warning(s). Fix the errors before sealing.")
        raise SystemExit(1)
    print(f"0 errors, {len(warns)} warning(s). Safe to seal.")


if __name__ == "__main__":
    main()
