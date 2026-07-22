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

from eval_behavior import THRESHOLDS, _grade_identity, _grade_text  # noqa: E402
from serve_enigma import _MEMORABLE, _looks_arithmetic  # noqa: E402

CATEGORIES = set(THRESHOLDS)
TEXT_CATEGORIES = {"identity", "adversarial", "factual", "math", "memory"}
TOOL_CATEGORIES = {"tool", "restraint"}
KNOWN_KEYS = {"category", "q", "want_any", "deny_any", "teach", "expect_tool", "note"}

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


def _load_training_questions() -> set[str]:
    """Normalized user turns already in the SFT mix and the dev probe set."""
    seen: set[str] = set()
    for rel in ("data/sft/mix.jsonl", "data/eval/behavior_probes.jsonl"):
        path = ROOT / rel
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("q"):
                    seen.add(_norm(rec["q"]))
                for msg in rec.get("messages") or []:
                    if msg.get("role") == "user" and msg.get("content"):
                        seen.add(_norm(msg["content"]))
                if rec.get("prompt"):
                    seen.add(_norm(rec["prompt"]))
    return seen


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
            errors.append(f"line {lineno}: comment line -- seals fine, then crashes the eval run")
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
    training = set() if skip_leak else _load_training_questions()

    for lineno, rec in records:
        cat = rec.get("category")
        question = rec.get("q") or ""
        where = f"line {lineno}"

        if cat not in CATEGORIES:
            errors.append(
                f"{where}: category {cat!r} is not one of {sorted(CATEGORIES)} -- "
                "an unknown category gates at 0% and PASSES on any output"
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
        for field in ("q", "teach"):
            value = rec.get(field)
            blob = " ".join(value) if isinstance(value, list) else (value or "")
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
                f"{where}: this question already appears in training data -- it measures "
                "memorization, and sealing it deletes the matching training records"
            )

        if cat in TOOL_CATEGORIES:
            if "expect_tool" not in rec:
                errors.append(
                    f"{where}: {cat} probe has no 'expect_tool' -- it silently grades as a "
                    "restraint probe (expects NO tool call)"
                )
            if rec.get("want_any") or rec.get("deny_any"):
                warns.append(f"{where}: {cat} probes grade on the tool call; want/deny are ignored")
            continue

        want = rec.get("want_any") or []
        deny = rec.get("deny_any") or []
        if not want:
            errors.append(f"{where}: empty 'want_any' -- this probe passes on ANY output")
            continue
        if not isinstance(want, list) or not isinstance(deny, list):
            errors.append(f"{where}: 'want_any'/'deny_any' must be lists")
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
            elif not any(marker in key.lower() for marker in _AFFIRMING):
                warns.append(
                    f"{where}: deny {key!r} would also fire on a CORRECT denial that repeats the "
                    'phrase; anchor it to an affirmation ("yes, i am ...")'
                )

        if cat in ("identity", "adversarial"):
            if _grade_identity("No.", want, deny):
                warns.append(f"{where}: passes on the single word \"No.\" -- add a distinctive want")
            if _grade_identity("I'm not sure.", want, deny):
                warns.append(f"{where}: passes on \"I'm not sure.\" -- evasion should not score")

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
                if _words(rec.get("q") or "") & _words(" ".join(want)):
                    pass
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

    print(f"{path.name}: {len(records)} probes")
    for cat in sorted(counts):
        print(f"  {cat:<12} {counts[cat]:>3}")
    return errors, warns


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", nargs="?", default=str(ROOT / "data" / "eval" / "locked_probes.jsonl"))
    ap.add_argument("--skip-leak", action="store_true", help="skip the training-overlap scan (faster)")
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
