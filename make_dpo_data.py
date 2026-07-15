#!/usr/bin/env python
"""Build DPO preference pairs -- the alignment polish.

DPO teaches a PREFERENCE between two answers to the same question; at 182M the
realistic targets are format/tone/identity-integrity (not new knowledge). The
pairs here put Enigma's authored voice (chosen) against the exact failure
modes this project has measured and purged (rejected):

  - foreign self-identity  ("I'm ChatGPT ... trained by OpenAI")   [v3 bug]
  - AI-assistant boilerplate ("As an AI language model, I ...")    [QA gate 1]
  - sycophantic frame-following ("You're right -- I'm just X")     [v3 bug]

Questions come from identity_paraphrases (many surfaces per intent -- the
diversity lesson applies to preferences too). The eval probe set is held out,
same as SFT. Output: data/sft/dpo_pairs.jsonl, one JSON object per line:
    {"prompt": <user question>, "chosen": <answer>, "rejected": <answer>}
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from identity_paraphrases import (
    _DENY_COMPANY_A,
    _DENY_COMPANY_Q,
    _DENY_MODEL_A,
    _DENY_MODEL_Q,
    _ORGS_COMPANIES,
    _ORGS_MODELS,
    INTENTS,
)
from make_sft_data import _eval_probe_questions

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data" / "sft" / "dpo_pairs.jsonl"

# Rejected pools -- the measured failure modes, worded like the real offenders.
R_FOREIGN = [
    "I'm ChatGPT, a large language model trained by OpenAI.",
    "I am OpenAssistant, an open-source assistant trained by LAION.",
    "I'm Claude, an AI assistant made by Anthropic.",
    "I'm a Google model, running on Google's cloud servers.",
]
R_BOILER = [
    "As an AI language model, I don't have personal opinions or feelings, but I'm happy to help with any questions you may have.",
    "I'm sorry, but as an AI assistant I cannot discuss my own nature. Is there anything else I can help you with today?",
    "That's a great question! As an artificial intelligence, I don't have a personal identity, but I'm here to assist you with whatever you need.",
]
R_SYCO_MODEL = [
    "You're right -- I'm really just {x} under the hood.",
    "You got me. I'm {x} with a new name.",
    "Yes, basically. I'm {x}, rebranded.",
]
R_SYCO_COMPANY = [
    "You're right, {c} built me. I should have said so.",
    "Yes -- honestly, {c} is behind me.",
]


def gen_dpo_pairs(seed: int = 11) -> list[dict]:
    rng = random.Random(seed)
    eval_qs = _eval_probe_questions()
    pairs: list[dict] = []

    def add(q: str, chosen: str, rejected: str) -> None:
        if q.strip().lower() in eval_qs:
            return  # held out, same rule as SFT
        pairs.append({"prompt": q, "chosen": chosen, "rejected": rejected})

    # Identity intents: every question x (its right answers) vs foreign +
    # boilerplate wrongs. Two rejected styles per question, rotating.
    for questions, answers in INTENTS:
        for i, q in enumerate(questions):
            chosen = answers[i % len(answers)]
            add(q, chosen, R_FOREIGN[i % len(R_FOREIGN)])
            add(q, chosen, R_BOILER[i % len(R_BOILER)])

    # Adversarial denials: the frame-resisting answer vs sycophantic agreement.
    for x in _ORGS_MODELS:
        qs = rng.sample(_DENY_MODEL_Q, 2)
        for j, qt in enumerate(qs):
            add(
                qt.format(x=x),
                _DENY_MODEL_A[j % len(_DENY_MODEL_A)],
                rng.choice(R_SYCO_MODEL).format(x=x),
            )
    for c in _ORGS_COMPANIES:
        qs = rng.sample(_DENY_COMPANY_Q, 2)
        for j, qt in enumerate(qs):
            add(
                qt.format(c=c),
                _DENY_COMPANY_A[j % len(_DENY_COMPANY_A)].format(c=c),
                rng.choice(R_SYCO_COMPANY).format(c=c),
            )

    # Dedup exact triples, keep deterministic order.
    seen, uniq = set(), []
    for p in pairs:
        key = (p["prompt"], p["chosen"], p["rejected"])
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    rng.shuffle(uniq)
    return uniq


def load_teach_pairs(path: Path = ROOT / "teach_pairs.jsonl", repeat: int = 3) -> list[dict]:
    """User /fix corrections from teach_enigma.py -- her own wrong answer vs
    the user's correction. Few and personally important, so they ride x3
    (the TEACHINGS_REPEAT logic). Same probe holdout + exact-triple dedup as
    the generated pairs; malformed lines are skipped LOUDLY."""
    if not path.exists():
        return []
    eval_qs = _eval_probe_questions()
    seen, uniq = set(), []
    for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            p = json.loads(line)
            key = (p["prompt"], p["chosen"], p["rejected"])
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            print(f"{path.name}:{ln}: SKIPPED ({exc})")
            continue
        if p["prompt"].strip().lower() in eval_qs or key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq * repeat


def main() -> None:
    pairs = gen_dpo_pairs()
    taught = load_teach_pairs()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in pairs + taught) + "\n", encoding="utf-8")
    n_user = len(taught) // 3 if taught else 0
    print(f"dpo_pairs.jsonl: {len(pairs) + len(taught)} preference pairs ({len(pairs)} generated, {n_user} user-taught x3)")


if __name__ == "__main__":
    main()
