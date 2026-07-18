#!/usr/bin/env python
"""Teach Enigma by chatting with her.

    python serve_enigma.py --memory-dir data/memory   # shell 1: her brain
    python teach_enigma.py                            # shell 2: this tool
    python teach_enigma.py --base-url http://127.0.0.1:8123

Talk normally. When an answer is wrong, garbled, or just not how she should
say it, correct her in place:

    /fix <what she should have said>   save a teaching + a preference pair
    /good                              save her last answer as a keeper
    /undo                              forget the last exchange (and retract
                                       anything /fix or /good saved for it)
    /new                               start a fresh conversation
    /help                              show the commands
    /quit                              leave

On /fix the tool AUTO-AUGMENTS your correction into several question
phrasings (plus a declarative statement twin when the question is a simple
'what/who/where is X'), shows them, and lets you accept / edit / skip before
anything is saved. Diverse phrasings teach the FACT; a single phrasing at
bake weight just memorizes the string (measured 2026-07-05), so the bake
oversample dropped x8 -> x4 once corrections carry their own variety.

Where it goes: teachings ride teachings.jsonl (make_sft_data oversamples
them at the next bake -- your corrections are her strongest training signal).
/fix also appends {prompt, chosen, rejected} to teach_pairs.jsonl: her wrong
answer becomes DPO evidence (make_dpo_data merges the file automatically). A
second /fix on the same exchange REPLACES the first -- the retracted records
are truncated from disk, and the preference pair always rejects her original
reply, never your earlier correction.

The conversation continues AS IF she had said the corrected thing (/fix
rewrites history), so one bad answer doesn't poison the rest of the chat.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEACHINGS = ROOT / "teachings.jsonl"
TEACH_PAIRS = ROOT / "teach_pairs.jsonl"

try:  # Windows consoles default to cp1252; her replies may carry unicode.
    sys.stdout.reconfigure(encoding="utf-8")
    # utf-8-sig strips the BOM that piped/redirected input smuggles in
    # (measured 2026-07-15: a scripted session taught her "﻿hi there").
    sys.stdin.reconfigure(encoding="utf-8-sig")
except Exception:
    pass


def _post(base_url: str, messages: list[dict], temperature: float, max_tokens: int) -> str:
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps({"messages": messages, "temperature": temperature, "max_tokens": max_tokens}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        body = json.loads(r.read().decode())
    return (body["choices"][0]["message"].get("content") or "").strip()


def _append_jsonl(path: Path, record: dict) -> int:
    """Append one record. Returns the file's byte size BEFORE the append so
    the write can be retracted later by truncating back to it (/undo must
    really undo -- audit 2026-07-15 found it left records baked on disk)."""
    size_before = path.stat().st_size if path.exists() else 0
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return size_before


def save_teaching(questions, answers, path: Path = TEACHINGS) -> int:
    """One correction -> one teachings.jsonl line in the questions/answers
    shape gen_teaching_examples reads. ``questions``/``answers`` may be single
    strings (a /good keeper) or lists (an auto-augmented /fix): the generator
    crosses phrasings against rotating answers, so several phrasings teach the
    FACT rather than one exact string at bake weight (audit 2026-07-05).
    Returns the pre-append size for retract."""
    if isinstance(questions, str):
        questions = [questions]
    if isinstance(answers, str):
        answers = [answers]
    return _append_jsonl(path, {"questions": list(questions), "answers": list(answers)})


def _lower_first(s: str) -> str:
    """Lowercase a leading capital so a wrapper prefix reads grammatically
    ('...what is X?'), but leave acronyms and the pronoun 'I' alone."""
    if len(s) >= 2 and s[0].isupper() and s[1].islower() and not s.startswith("I "):
        return s[0].lower() + s[1:]
    return s


def paraphrase_question(q: str) -> list[str]:
    """Light, meaning-preserving reframings of a question so one correction
    teaches the fact, not one exact token string (a single phrasing at bake
    weight memorizes the string -- measured 2026-07-05). Wrapper prefixes stay
    grammatical for any question; the user edits them at the confirm step."""
    q = q.strip()
    lc = _lower_first(q)
    cands = [q, f"Remind me, {lc}", f"Just to be sure, {lc}", f"Quick question -- {q}"]
    seen, out = set(), []
    for c in cands:
        key = c.lower()
        if c and key not in seen:
            seen.add(key)
            out.append(c)
    return out


_WH_STATEMENT = re.compile(r"^\s*(?:what|who|where)\b.*?\b(is|are|was|were)\b\s+(.+?)\s*\??\s*$", re.IGNORECASE)

# A sentence-initial function word in the ANSWER keeps its capital when the
# answer lands mid-statement ("In the forge." -> "I was born In the forge.",
# baked into training data -- test-suite audit 2026-07-17). Lowercase ONLY
# these openers; proper-noun answers (Paris, Jupiter, Sir Knight) keep their
# casing, which is why a blanket lowercase was rejected (2026-07-16).
_ANSWER_OPENER_LOWER = frozenset({
    "a", "an", "the", "in", "at", "on", "of", "from", "by", "with", "near",
    "inside", "under", "over", "behind", "beside", "between", "among",
    "around", "during", "before", "after", "above", "below", "across", "through",
})

# The question is the USER's voice but the twin is baked as the ASSISTANT's
# statement, so person must flip or the fact attaches to the wrong speaker
# ('Who are you?' must twin to 'I am Enigma.', never 'You are Enigma.' --
# audit 2026-07-16 found the unflipped form baked into teachings).
_PERSON_FLIP = {
    "you": "i", "your": "my", "yours": "mine", "yourself": "myself",
    "i": "you", "my": "your", "mine": "yours", "me": "you", "myself": "yourself",
}
_SUBJECT_PRONOUNS = {"you", "i", "he", "she", "it", "we", "they"}
_VERB_AGREE = {("i", "are"): "am", ("i", "were"): "was", ("you", "am"): "are", ("you", "was"): "were"}


def statement_twin(question: str, answer: str) -> str | None:
    """Best-effort declarative restatement of a simple 'what/who/where is X'
    Q/A so the fact is also exposed as a STATEMENT, not only as a question.
    Person flips into the assistant's voice, and an inverted pronoun subject
    un-inverts ('Where were you born?' -> 'I was born ...', not the garbled
    'You born were ...'). Returns None for behavioral/how/why corrections,
    which have no safe generic statement transform (auto-augment 2026-07-16)."""
    m = _WH_STATEMENT.match(question)
    if not m:
        return None
    verb = m.group(1).lower()
    words = m.group(2).strip().rstrip("?.").strip().split()
    ans = answer.strip().rstrip(".").strip()
    if not words or not ans:
        return None
    first, _, rest = ans.partition(" ")
    # `not first[1:]` admits the single-char article "A" -- ""[1:].islower()
    # is False, so the bare islower() check silently skipped the most common
    # opener (re-audit 2026-07-18: "It is A dog." baked into training data).
    if (
        first.lower() in _ANSWER_OPENER_LOWER
        and first[:1].isupper()
        and (not first[1:] or first[1:].islower())
    ):
        ans = first.lower() + (" " + rest if rest else "")
    if words[0].lower() in _SUBJECT_PRONOUNS and len(words) > 1:
        # Subject-aux inversion: the trailing predicate belongs AFTER the verb.
        subject, predicate = [_PERSON_FLIP.get(words[0].lower(), words[0])], words[1:]
    else:
        subject, predicate = [_PERSON_FLIP.get(w.lower(), w) for w in words], []
    verb = _VERB_AGREE.get((subject[0].lower(), verb), verb)
    statement = " ".join(subject + [verb] + predicate + [ans]) + "."
    statement = re.sub(r"\bi\b", "I", statement)
    return statement[0].upper() + statement[1:]


def augment_teaching(question: str, answer: str) -> tuple[list[str], list[str]]:
    """Expand one correction into several question phrasings + (when derivable)
    a declarative statement twin, so /fix teaches the fact at a lower bake
    weight instead of memorizing a single string (ROADMAP teach-loop
    auto-augment)."""
    questions = paraphrase_question(question)
    answers = [answer.strip()]
    twin = statement_twin(question, answer)
    if twin and twin.lower() != answer.strip().lower():
        answers.append(twin)
    return questions, answers


def review_augmentation(questions: list[str], answers: list[str]):
    """Show the auto-generated phrasings and let the user accept / edit / skip
    BEFORE anything is written (confirm-before-bake, 2026-07-16). Returns the
    (questions, answers) to save, or None to save nothing. Non-interactive
    stdin (piped/scripted teaching) auto-accepts -- no prompt to block on."""
    if not sys.stdin.isatty():
        return questions, answers
    print("  will bake these phrasings (each -> your corrected answer):")
    for qq in questions:
        print(f"    Q: {qq}")
    for aa in answers:
        print(f"    A: {aa}")
    print("  [Enter]=save  e=edit phrasings  s=save just the one  x=cancel")
    try:
        choice = input("  > ").strip().lower()
    except EOFError:
        choice = ""  # piped stdin ran dry: auto-accept, same as non-tty
    except KeyboardInterrupt:
        print()
        return None  # Ctrl+C aborts here like everywhere else in the tool
    if choice == "x":
        return None
    if choice == "s":
        return [questions[0]], [answers[0]]
    if choice == "e":
        print("  type one question phrasing per line; blank line to finish:")
        edited = []
        while True:
            try:
                ln = input("    Q: ").strip()
            except EOFError:
                break
            except KeyboardInterrupt:
                print()
                return None
            if not ln:
                break
            edited.append(ln)
        if edited:
            questions = edited
    return questions, answers


def save_pair(prompt: str, chosen: str, rejected: str, path: Path = TEACH_PAIRS) -> int:
    """Her wrong answer is evidence, not garbage: {prompt, chosen, rejected}
    in the dpo_pairs.jsonl schema; make_dpo_data merges the file at the next
    build. Returns the pre-append size for retract."""
    return _append_jsonl(path, {"prompt": prompt, "chosen": chosen, "rejected": rejected})


def retract(writes: list[tuple[Path, int]]) -> int:
    """Undo appends recorded as (path, size_before_append) by truncating each
    file back to its EARLIEST recorded size. Files are append-only within a
    session, so truncation removes exactly this session's retracted records.
    Returns the number of writes retracted."""
    if not writes:
        return 0
    earliest: dict[Path, int] = {}
    for path, size in writes:
        earliest[path] = min(earliest.get(path, size), size)
    for path, size in earliest.items():
        try:
            with open(path, "r+b") as f:
                cur = f.seek(0, 2)  # current size
                # Only ever SHRINK. If the file was hand-edited smaller between
                # the append and now, truncating to the larger recorded offset
                # would PAD it with NUL bytes and corrupt the jsonl (final audit
                # 2026-07-16 m4). A smaller file can't hold our record anyway.
                if size < cur:
                    f.truncate(size)
        except FileNotFoundError:
            pass
    return len(writes)


def fix_last(history: list[dict], correction: str) -> tuple[str, str] | None:
    """Replace her last reply in history with the correction. Returns
    (user_question, her_original_reply), or None if there is nothing to fix."""
    if len(history) < 2 or history[-1]["role"] != "assistant" or history[-2]["role"] != "user":
        return None
    original = history[-1]["content"]
    history[-1] = {"role": "assistant", "content": correction}
    return history[-2]["content"], original


def last_exchange(history: list[dict]) -> tuple[str, str] | None:
    if len(history) < 2 or history[-1]["role"] != "assistant" or history[-2]["role"] != "user":
        return None
    return history[-2]["content"], history[-1]["content"]


_HELP = """commands:
  /fix <better answer>   correct her last reply (auto-augments into several
                         phrasings you can accept/edit/skip, then saves a
                         teaching + preference pair; a second /fix replaces the first)
  /good                  save her last reply as a keeper
  /undo                  forget the last exchange (retracts its saved records)
  /new                   fresh conversation
  /quit                  leave (also /exit)"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Teach Enigma by chatting with her.")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000", help="running serve_enigma URL")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max-tokens", type=int, default=200)
    args = ap.parse_args()

    print("Teaching session -- talk to her; /help for commands; /quit to leave.")
    print(f"(teachings -> {TEACHINGS.name}, preference pairs -> {TEACH_PAIRS.name})")
    history: list[dict] = []
    # One entry per Q/A pair in history: her ORIGINAL reply (the preference
    # pair must always reject it, never an earlier correction) plus every
    # disk write made for the exchange, so /undo and re-/fix can retract.
    exchanges: list[dict] = []
    n_taught = 0

    while True:
        try:
            line = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            line = "/quit"
        if not line:
            continue

        if line.startswith("/"):
            cmd, _, rest = line.partition(" ")
            cmd = cmd.lower()
            if cmd in ("/quit", "/exit"):
                print(f"bye -- {n_taught} teaching(s) saved this session.")
                if n_taught:
                    print("they bake in at the next training cycle (make_sft_data + finetune).")
                return
            if cmd == "/help":
                print(_HELP)
            elif cmd == "/new":
                history, exchanges = [], []
                print("fresh conversation.")
            elif cmd == "/undo":
                if last_exchange(history):
                    ex = exchanges.pop() if exchanges else {"writes": [], "taught": 0}
                    n_gone = retract(ex["writes"])
                    n_taught -= ex["taught"]
                    history = history[:-2]
                    if n_gone:
                        print(f"forgot the last exchange and retracted {n_gone} saved record(s).")
                    else:
                        print("forgot the last exchange.")
                else:
                    print("nothing to undo.")
            elif cmd == "/good":
                pair = last_exchange(history)
                if pair is None or not exchanges:
                    print("nothing to keep yet -- chat first.")
                elif exchanges[-1]["writes"]:
                    # A /fix (or an earlier /good) already saved for this
                    # exchange; a second write would duplicate it (final audit
                    # 2026-07-16 m3). /undo first to change what's saved.
                    print("already saved a teaching for this exchange -- /undo first to change it.")
                else:
                    exchanges[-1]["writes"].append((TEACHINGS, save_teaching(*pair)))
                    exchanges[-1]["taught"] += 1
                    n_taught += 1
                    print("kept. she'll train on that answer.")
            elif cmd == "/fix":
                if not rest.strip():
                    print("usage: /fix <what she should have said>")
                    continue
                correction = rest.strip()
                fixed = fix_last(history, correction)
                if fixed is None or not exchanges:
                    print("nothing to fix yet -- chat first.")
                else:
                    question, _ = fixed
                    ex = exchanges[-1]
                    # Auto-augment into several phrasings + a statement twin,
                    # then let the user vet them BEFORE anything is written OR
                    # unwritten -- a cancelled re-fix must leave the earlier
                    # save intact (audit 2026-07-16: retract ran pre-review,
                    # so 'x' silently destroyed the previous record).
                    questions, answers = augment_teaching(question, correction)
                    reviewed = review_augmentation(questions, answers)
                    if reviewed is None:
                        # History already continues from the correction; the
                        # user just chose not to bake it.
                        if ex["writes"]:
                            print("not saved -- keeping what was previously saved for this exchange.")
                        else:
                            print("not saved -- conversation still continues from your version.")
                    else:
                        if ex["writes"]:
                            # Re-fixing (or fixing a /good'd answer): the earlier
                            # save would contradict this one -- replace it.
                            retract(ex["writes"])
                            n_taught -= ex["taught"]
                            ex["writes"], ex["taught"] = [], 0
                            print("(replaced what was previously saved for this exchange)")
                        q_list, a_list = reviewed
                        ex["writes"].append((TEACHINGS, save_teaching(q_list, a_list)))
                        ex["writes"].append((TEACH_PAIRS, save_pair(question, correction, ex["original"])))
                        ex["taught"] += 1
                        n_taught += 1
                        print(f"fixed and saved ({len(q_list)} phrasing(s)). the conversation continues from your version.")
            else:
                print(f"unknown command {cmd} -- /help lists them.")
            continue

        history.append({"role": "user", "content": line})
        try:
            reply = _post(args.base_url, history, args.temperature, args.max_tokens)
        except Exception as exc:
            history.pop()
            print(f"error talking to {args.base_url}: {exc}")
            print("is serve_enigma.py running? start it, then try again.")
            continue
        history.append({"role": "assistant", "content": reply})
        exchanges.append({"original": reply, "writes": [], "taught": 0})
        print(f"enigma> {reply}")


if __name__ == "__main__":
    main()
