#!/usr/bin/env python
"""Teach Enigma by chatting with her.

    python serve_enigma.py --memory-dir data/memory   # shell 1: her brain
    python teach_enigma.py                            # shell 2: this tool
    python teach_enigma.py --base-url http://127.0.0.1:8123

Talk normally. When an answer is wrong, garbled, or just not how she should
say it, correct her in place:

    /fix <what she should have said>   save a teaching + a preference pair
    /good                              save her last answer as a keeper
    /undo                              forget the last exchange
    /new                               start a fresh conversation
    /help                              show the commands
    /quit                              leave

Where it goes: teachings ride teachings.jsonl (make_sft_data oversamples
them x8 at the next bake -- your corrections are her strongest training
signal). /fix also appends {prompt, chosen, rejected} to teach_pairs.jsonl:
her wrong answer becomes DPO training evidence once merged.

The conversation continues AS IF she had said the corrected thing (/fix
rewrites history), so one bad answer doesn't poison the rest of the chat.
"""

from __future__ import annotations

import argparse
import json
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


def _append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_teaching(question: str, answer: str, path: Path = TEACHINGS) -> None:
    """One correction -> one teachings.jsonl line in the questions/answers
    shape gen_teaching_examples reads. One phrasing is a flashcard; the
    generator warns about thin records, and that is fine -- a real correction
    now beats a perfect one never."""
    _append_jsonl(path, {"questions": [question], "answers": [answer]})


def save_pair(prompt: str, chosen: str, rejected: str, path: Path = TEACH_PAIRS) -> None:
    """Her wrong answer is evidence, not garbage: {prompt, chosen, rejected}
    in the dpo_pairs.jsonl schema, for a future DPO merge."""
    _append_jsonl(path, {"prompt": prompt, "chosen": chosen, "rejected": rejected})


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
  /fix <better answer>   correct her last reply (saves teaching + preference pair)
  /good                  save her last reply as a keeper
  /undo                  forget the last exchange
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
                history = []
                print("fresh conversation.")
            elif cmd == "/undo":
                if last_exchange(history):
                    history = history[:-2]
                    print("forgot the last exchange.")
                else:
                    print("nothing to undo.")
            elif cmd == "/good":
                ex = last_exchange(history)
                if ex is None:
                    print("nothing to keep yet -- chat first.")
                else:
                    save_teaching(*ex)
                    n_taught += 1
                    print("kept. she'll train on that answer.")
            elif cmd == "/fix":
                if not rest.strip():
                    print("usage: /fix <what she should have said>")
                    continue
                fixed = fix_last(history, rest.strip())
                if fixed is None:
                    print("nothing to fix yet -- chat first.")
                else:
                    question, original = fixed
                    save_teaching(question, rest.strip())
                    save_pair(question, rest.strip(), original)
                    n_taught += 1
                    print("fixed and saved. the conversation continues from your version.")
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
        print(f"enigma> {reply}")


if __name__ == "__main__":
    main()
