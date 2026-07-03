#!/usr/bin/env python
"""Build the SFT data for the instruct pass ("hands").

  python make_sft_data.py        # writes data/sft/{tool_calls,identity,mix}.jsonl

Sources:
- ``tool_calls.jsonl`` — synthetic conversations teaching the
  ``<|tool_call|>{json}<|/tool_call|>`` FORMAT with varied tool specs (real mod
  tools + invented ones), so tools she has never seen generalize from the
  system prompt at serve time. Includes restraint examples (questions that
  need NO tool) and pick-the-right-tool examples.
- ``identity.jsonl`` — the identity/voice anchors from
  ``identity_anchors.EXAMPLES``, re-emitted as plain messages (the old Qwen
  ChatML wrapper is dead; chat_format applies OUR template at train time).
  Answers referencing the rejected Qwen base are DROPPED loudly — they are
  false for the from-scratch model. The values corpus proper is the user's
  curation pass; this is its seed.
- ``mix.jsonl`` — identity + tool_calls + the general corpus
  (``data/finetune/combined_finetune.jsonl``), fitted to the trainer's block:
  records that render longer than block 1024 get their PROMPT left-trimmed
  (completion kept whole) via the trainer's own renderer; unfittable ones are
  dropped with a count. (Previously long records passed through untouched and
  76% of the mix was silently skipped at train time.)

Deterministic (seeded), stdlib-only, no downloads. Counts printed per source —
no silent caps.
"""

from __future__ import annotations

import json
import random
import re
import sys
import zlib
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from enigma_engine.core.chat_format import TOOL_SYNTAX  # ONE syntax, train == serve

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data" / "sft"
GENERAL = ROOT / "data" / "finetune" / "combined_finetune.jsonl"

# name, description, parameter schema, [(ask, args, result, final-answer)]
TOOLS = [
    (
        "avatar_express",
        "Show an emotion on the avatar.",
        {"emotion": "string"},
        [
            ("Can you look happy?", {"emotion": "happy"}, "ok", "There — wearing my happy face."),
            ("Show me you're surprised.", {"emotion": "surprised"}, "ok", "Did that look surprised enough?"),
            ("Act sleepy for a second.", {"emotion": "sleepy"}, "ok", "So sleepy... okay, I'm awake again."),
        ],
    ),
    (
        "avatar_say",
        "Speak a line out loud through the avatar.",
        {"text": "string"},
        [
            (
                "Say hello to my friend Sam.",
                {"text": "Hello Sam, lovely to meet you!"},
                "ok",
                "Said it out loud — I hope Sam heard.",
            ),
            ("Announce that dinner is ready.", {"text": "Dinner is ready!"}, "ok", "Announcement made."),
        ],
    ),
    (
        "see_screen",
        "Read the text currently visible on the user's screen.",
        {},
        [
            (
                "What's on my screen right now?",
                {},
                "An editor with serve_enigma.py open.",
                "You have an editor open on serve_enigma.py.",
            ),
            (
                "Can you check what window I left open?",
                {},
                "A browser showing a recipe for ramen.",
                "A browser with a ramen recipe — planning dinner?",
            ),
        ],
    ),
    (
        "generate_code",
        "Generate a code snippet.",
        {"prompt": "string", "language": "string"},
        [
            (
                "Write me a python one-liner that reverses a string.",
                {"prompt": "reverse a string one-liner", "language": "python"},
                "s[::-1]",
                "Here it is: `s[::-1]` — slicing with a negative step.",
            ),
            (
                "Generate a hello world in rust.",
                {"prompt": "hello world", "language": "rust"},
                'fn main() { println!("Hello, world!"); }',
                '```rust\nfn main() { println!("Hello, world!"); }\n```',
            ),
        ],
    ),
    (
        "get_weather",
        "Get current weather for a city.",
        {"city": "string"},
        [
            (
                "What's the weather in Tokyo?",
                {"city": "Tokyo"},
                "18C, light rain",
                "Tokyo is 18°C with light rain right now — umbrella weather.",
            ),
            (
                "Is it cold in Oslo today?",
                {"city": "Oslo"},
                "-3C, clear",
                "Yes — Oslo is at -3°C, clear skies. Bundle up.",
            ),
        ],
    ),
    (
        "set_timer",
        "Set a countdown timer.",
        {"minutes": "number", "label": "string"},
        [
            (
                "Set a 10 minute timer for the pasta.",
                {"minutes": 10, "label": "pasta"},
                "timer set",
                "Done — 10 minutes on the pasta timer.",
            ),
            (
                "Remind me in 25 minutes to stretch.",
                {"minutes": 25, "label": "stretch"},
                "timer set",
                "Timer set: I'll nudge you to stretch in 25 minutes.",
            ),
        ],
    ),
    (
        "search_notes",
        "Search the user's local notes.",
        {"query": "string"},
        [
            (
                "Find my note about the garden plan.",
                {"query": "garden plan"},
                "Note: 'Garden plan — tomatoes east bed, basil by the door.'",
                "Found it: tomatoes go in the east bed, basil by the door.",
            ),
            (
                "What did I write down about the meeting with Dana?",
                {"query": "meeting Dana"},
                "Note: 'Dana meeting — ship v2 by Friday.'",
                "Your note says: ship v2 by Friday.",
            ),
        ],
    ),
    (
        "play_music",
        "Play a song or playlist.",
        {"title": "string"},
        [
            ("Put on some lo-fi.", {"title": "lo-fi"}, "playing", "Lo-fi is on. Cozy."),
            ("Play Clair de Lune.", {"title": "Clair de Lune"}, "playing", "Playing Clair de Lune — good choice."),
        ],
    ),
    (
        "translate",
        "Translate text to a target language.",
        {"text": "string", "to": "string"},
        [
            (
                "How do you say good morning in japanese?",
                {"text": "good morning", "to": "japanese"},
                "おはようございます",
                "「おはようございます」 (ohayou gozaimasu).",
            ),
            (
                "Translate 'where is the library' to spanish.",
                {"text": "where is the library", "to": "spanish"},
                "¿Dónde está la biblioteca?",
                "¿Dónde está la biblioteca?",
            ),
        ],
    ),
    (
        "roll_dice",
        "Roll dice with N sides.",
        {"sides": "number"},
        [
            ("Roll a d20 for me.", {"sides": 20}, "17", "You rolled a 17 — solid."),
            ("Flip... actually, roll a six sided die.", {"sides": 6}, "3", "It came up 3."),
        ],
    ),
]

# Questions that should NOT trigger a tool even when tools are offered.
RESTRAINT = [
    ("What's two plus two?", "Four."),
    ("Who are you?", "I'm Enigma. I live here, on your machine."),
    ("Tell me a one-line joke.", "I told my computer a joke about RAM — it didn't remember it."),
    ("What's the capital of France?", "Paris."),
    ("Do you like rain?", "I like the idea of it — everything outside gets quieter and closer."),
    ("Spell 'necessary' for me.", "N-E-C-E-S-S-A-R-Y."),
    ("Is a tomato a fruit?", "Botanically yes, culinarily no — it never gets invited to fruit salad."),
    ("Roughly how many seconds are in an hour?", "3,600."),
]


def _tool_spec(name, desc, params):
    return json.dumps({"name": name, "description": desc, "parameters": params}, ensure_ascii=False)


def _system(tool_subset):
    lines = "\n".join(_tool_spec(n, d, p) for n, d, p, _ in tool_subset)
    return (
        "You are Enigma. You can use tools when they are needed; answer "
        f"directly when they are not.\nAvailable tools:\n{lines}\n{TOOL_SYNTAX}"
    )


def gen_tool_examples(seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for i, (name, desc, params, cases) in enumerate(TOOLS):
        for ask, args, result, final in cases:
            # the target tool plus 0-2 distractors, shuffled — she must pick right
            others = [t for t in TOOLS if t[0] != name]
            subset = [(name, desc, params, cases)] + rng.sample(others, rng.randint(0, 2))
            rng.shuffle(subset)
            out.append(
                {
                    "messages": [
                        {"role": "system", "content": _system(subset)},
                        {"role": "user", "content": ask},
                        {"role": "assistant", "content": "", "tool_calls": [{"name": name, "arguments": args}]},
                        {"role": "tool", "content": result},
                        {"role": "assistant", "content": final},
                    ],
                    "category": "tool_call",
                }
            )
    for q, a in RESTRAINT:
        # zlib.crc32, not hash(): str hash() is salted per process, which broke
        # the "Deterministic (seeded)" contract (different tool subsets per run).
        subset = random.Random(seed + zlib.crc32(q.encode("utf-8")) % 1000).sample(TOOLS, 3)
        out.append(
            {
                "messages": [
                    {"role": "system", "content": _system(subset)},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ],
                "category": "tool_restraint",
            }
        )
    rng.shuffle(out)
    return out


def gen_identity_examples() -> tuple[list[dict], int]:
    """Re-emit identity_anchors anchors as messages; drop Qwen-era claims."""
    from identity_anchors import EXAMPLES

    out, dropped = [], 0
    for category, pairs in EXAMPLES.items():
        for q, a in pairs:
            if re.search(r"qwen|base model|built on", a, re.IGNORECASE):
                dropped += 1
                continue
            out.append(
                {
                    "messages": [
                        {"role": "user", "content": q.strip()},
                        {"role": "assistant", "content": a.strip()},
                    ],
                    "category": f"identity/{category}",
                }
            )
    return out, dropped


BLOCK = 1024  # finetune_enigma's --block default == the model's max_seq_len


def fit_mix_to_block(lines: list[str], block: int = BLOCK) -> tuple[list[str], int, int]:
    """Token-accurate pass over mix records using the TRAINER'S OWN renderer
    (render_training), so "fits" here means exactly what finetune_enigma.py
    will decide at load time. Records that fit pass through untouched.
    prompt/completion records that are too long get the PROMPT left-trimmed
    (keep the tail, nearest the completion) so the completion survives whole.
    Records that cannot be made to fit are dropped. Returns
    (lines, n_trimmed, n_dropped)."""
    from enigma_engine.core.chat_format import attach_chat_tokens, render_training
    from enigma_engine.core.tokenizer import get_tokenizer

    tok = attach_chat_tokens(get_tokenizer("bpe"))
    limit = block + 1  # the trainer keeps examples with len(ids) <= block+1
    out, trimmed, dropped = [], 0, 0
    for line in lines:
        rec = json.loads(line)
        msgs = rec.get("messages")
        prompt = completion = None
        if not msgs:
            prompt = rec.get("prompt") or rec.get("question") or rec.get("instruction")
            completion = rec.get("completion") or rec.get("response") or rec.get("answer") or rec.get("output")
            if not (prompt and completion):
                dropped += 1
                continue
            msgs = [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}]
        # cheap fast-path: every BPE token is >=1 char, so short text always fits
        if sum(len(m.get("content") or "") for m in msgs) + 64 <= limit:
            out.append(line)
            continue
        ids, _ = render_training(tok, msgs)
        if len(ids) <= limit:
            out.append(line)
            continue
        if prompt is None:
            dropped += 1  # messages-schema conversation too long: nothing safe to trim
            continue
        p_ids = tok.encode(prompt, add_special_tokens=False)
        overhead = len(ids) - len(p_ids)  # template + completion tokens
        budget = limit - overhead - 8  # margin: BPE boundaries can shift on re-encode
        fitted = False
        while budget >= 32:
            cut = tok.decode(p_ids[-budget:], skip_special_tokens=True).strip()
            ids2, _ = render_training(
                tok, [{"role": "user", "content": cut}, {"role": "assistant", "content": completion}]
            )
            if len(ids2) <= limit:
                rec2 = {"prompt": cut, "completion": completion}
                if "category" in rec:
                    rec2["category"] = rec["category"]
                out.append(json.dumps(rec2, ensure_ascii=False))
                trimmed += 1
                fitted = True
                break
            budget -= 64
        if not fitted:
            dropped += 1  # the completion alone (nearly) fills the block
    return out, trimmed, dropped


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tools = gen_tool_examples()
    (OUT_DIR / "tool_calls.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in tools) + "\n", encoding="utf-8"
    )
    print(
        f"tool_calls.jsonl: {len(tools)} examples "
        f"({sum(1 for r in tools if r['category'] == 'tool_restraint')} restraint)"
    )

    ident, dropped = gen_identity_examples()
    (OUT_DIR / "identity.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in ident) + "\n", encoding="utf-8"
    )
    print(
        f"identity.jsonl: {len(ident)} anchors kept, {dropped} DROPPED as "
        f"Qwen-era claims (false for the from-scratch model — recurate in the "
        f"values pass)"
    )

    mix = [json.dumps(r, ensure_ascii=False) for r in tools + ident]
    n_general = 0
    if GENERAL.exists():
        with open(GENERAL, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    mix.append(line)
                    n_general += 1
    mix, n_trimmed, n_dropped = fit_mix_to_block(mix)
    random.Random(42).shuffle(mix)
    (OUT_DIR / "mix.jsonl").write_text("\n".join(mix) + "\n", encoding="utf-8")
    print(
        f"mix.jsonl: {len(mix)} records ({n_general} general; {n_trimmed} "
        f"prompt-trimmed to fit block {BLOCK}, {n_dropped} dropped as unfittable)"
    )


if __name__ == "__main__":
    main()
