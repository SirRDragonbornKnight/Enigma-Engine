#!/usr/bin/env python
"""Build the SFT data for the instruct pass ("hands").

  python make_sft_data.py        # writes data/sft/{tool_calls,identity,mix}.jsonl
  python make_sft_data.py --persona personas\\atlas --out <dir>
                                 # a pack builds its OWN corpus into its OWN dir

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
  records that render longer than the block (2048, the adopted v2 lineage's
  training shape) get their PROMPT left-trimmed
  (completion kept whole) via the trainer's own renderer; unfittable ones are
  dropped with a count. (Previously long records passed through untouched and
  76% of the mix was silently skipped at train time.)

Deterministic (seeded), stdlib-only, no downloads. Counts printed per source —
no silent caps.
"""

from __future__ import annotations

import hashlib
import json
import os
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
from enigma_engine.core.persona import Persona
from enigma_engine.core.persona_content import (
    Aside,
    PersonaContent,
    default_content,
    load_content,
)
from eval_leak_guard import LOCKED_PROBES_NAME, LockedProbeGuard, persona_manifest
from identity_paraphrases import gen_identity_paraphrases
from knowledge_corpus import gen_knowledge_examples

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data" / "sft"
GENERAL = ROOT / "data" / "finetune" / "combined_finetune.jsonl"
EVAL_PROBES = ROOT / "data" / "eval" / "behavior_probes.jsonl"
TEACHINGS = ROOT / "teachings.jsonl"  # user-authored; gitignored (see teachings.example.jsonl)


def _norm_q(rec: dict) -> str:
    """The record's first user question, normalized for eval-leak comparison.
    Handles the messages shape and EVERY flat key the pipeline accepts --
    fit_mix_to_block and finetune_enigma read prompt/question/instruction, so
    the leak guard must too, or a rebuilt general diet keyed on 'question'
    would carry probes into training (audit 2026-07-15)."""
    for m in rec.get("messages", []):
        if m.get("role") == "user" and m.get("content"):
            return m["content"].strip().lower()
    q = rec.get("prompt") or rec.get("question") or rec.get("instruction") or ""
    return q.strip().lower() if isinstance(q, str) else ""


def _eval_probe_questions(path: "Path | None" = None) -> set[str]:
    """Questions in the held-out behavior eval -- NEVER put these in training,
    or the harness measures memorization instead of generalization.

    Skips "#" comment lines on the SAME rule eval_behavior.run() reads this
    file by: the eval harness invites an annotated DEV file, and json.loads on
    the comment it invites killed this backstop -- the one that holds identity
    and knowledge probes out of training -- with a raw JSONDecodeError.

    `path` is the probe file to read; omitted, it is EVAL_PROBES -- HER dev
    set, and read through the module global so the suite's patch still lands.
    A pack build passes its own file: the extraction shape is the same, and
    the questions another AI is graded on say nothing about what this one may
    train."""
    probes = EVAL_PROBES if path is None else Path(path)
    if not probes.exists():
        return set()
    qs = set()
    for line in probes.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        probe = json.loads(line)
        qs.add(probe["q"].strip().lower())
        for fact in probe.get("teach", []):  # memory probes' teach messages too
            qs.add(fact.strip().lower())
    return qs

# name, description, parameter schema, cases[(ask, args, result, final-answer)]
# Phase 1a: each tool carries many diverse hand-authored asks; the parameterized
# tools additionally expand from value pools (each a genuinely distinct request,
# not paraphrase padding). gen_tool_examples multiplies these over varied
# distractor sets, injects error-recovery turns, and adds multi-turn chains, so
# the model learns the <|tool_call|> FORMAT + tool SELECTION and unseen tools
# generalize from the system prompt.
TOOLS = [
    (
        "avatar_express",
        "Show an emotion on the avatar.",
        {"emotion": "string"},
        [
            ("Can you look happy?", {"emotion": "happy"}, "ok", "There — wearing my happy face."),
            ("Show me you're surprised.", {"emotion": "surprised"}, "ok", "Did that look surprised enough?"),
            ("Act sleepy for a second.", {"emotion": "sleepy"}, "ok", "So sleepy... okay, I'm awake again."),
            ("Give me your best grumpy face.", {"emotion": "grumpy"}, "ok", "Hmph. Grumpy face deployed."),
            ("Look excited, we won!", {"emotion": "excited"}, "ok", "We won! I can barely sit still."),
            ("Can you look thoughtful for a moment?", {"emotion": "thoughtful"}, "ok", "Hmm... there, deep in thought."),
            ("Pull a scared face.", {"emotion": "scared"}, "ok", "Eep — scared face on."),
            ("Wink at me.", {"emotion": "wink"}, "ok", "*wink*"),
        ],
    ),
    (
        "avatar_say",
        "Speak a line out loud through the avatar.",
        {"text": "string"},
        [
            ("Say hello to my friend Sam.", {"text": "Hello Sam, lovely to meet you!"}, "ok", "Said it out loud — I hope Sam heard."),
            ("Announce that dinner is ready.", {"text": "Dinner is ready!"}, "ok", "Announcement made."),
            ("Tell the room good morning.", {"text": "Good morning, everyone!"}, "ok", "Good morning delivered."),
            ("Wish my mom a happy birthday out loud.", {"text": "Happy birthday, Mom!"}, "ok", "Said it — happy birthday to your mom."),
            ("Say 'the build passed' so I can hear it from the kitchen.", {"text": "The build passed."}, "ok", "Called it out — the build passed."),
            ("Read this back to me: meeting at three.", {"text": "Meeting at three."}, "ok", "Read it back: meeting at three."),
        ],
    ),
    (
        # The server-side voice organ (core/tts.py, --voice). Distinct from
        # avatar_say: speak plays through THIS machine's speakers; avatar_say
        # is fulfilled by the avatar client.
        "speak",
        "Speak text out loud through the computer speakers.",
        {"text": "string"},
        [
            ("Say 'hello world' out loud.", {"text": "Hello world!"}, "speaking", "Said it out loud."),
            ("Tell everyone lunch is here.", {"text": "Lunch is here, everyone!"}, "speaking", "Called it out — lunch is here."),
            ("Read that back to me out loud: backup finished.", {"text": "Backup finished."}, "speaking", "Read it out: backup finished."),
            ("Say something encouraging, out loud.", {"text": "You've got this — keep going!"}, "speaking", "Spoken. You've got this."),
            ("Announce the meeting starts in five minutes.", {"text": "The meeting starts in five minutes."}, "speaking", "Announced — five minutes to the meeting."),
            ("Use your voice to count down from three.", {"text": "Three... two... one."}, "speaking", "Counted it down out loud."),
        ],
    ),
    (
        # The imagination organ (core/imagegen.py, --image-gen). She writes
        # the prompt; Stable Diffusion paints; serve returns the saved path.
        "imagine",
        "Generate an image from a text description and save it as a file.",
        {"prompt": "string"},
        [
            ("Draw me a picture of a lighthouse at sunset.", {"prompt": "a lighthouse at sunset, warm light, ocean waves"}, "image saved to images/imagine_1a2b.png", "Painted it — a lighthouse at sunset. Saved to images/imagine_1a2b.png."),
            ("Make an image of a cat wearing a wizard hat.", {"prompt": "a cat wearing a wizard hat, portrait"}, "image saved to images/imagine_3c4d.png", "One wizard cat, done. It's at images/imagine_3c4d.png."),
            ("Can you generate a picture of a cozy cabin in the snow?", {"prompt": "a cozy log cabin in falling snow, warm windows, evening"}, "image saved to images/imagine_5e6f.png", "Made it — cozy cabin, snow coming down. Saved as images/imagine_5e6f.png."),
            ("Create some art of a robot playing guitar.", {"prompt": "a robot playing an electric guitar on stage, concert lighting"}, "image saved to images/imagine_7g8h.png", "Rocked it out onto canvas — saved to images/imagine_7g8h.png."),
            ("Sketch a simple logo for a bakery called Crumb.", {"prompt": "minimal bakery logo, bread loaf icon, the word Crumb"}, "image saved to images/imagine_9i0j.png", "Sketched a Crumb logo — it's saved at images/imagine_9i0j.png."),
            ("Paint a picture of mountains reflected in a lake.", {"prompt": "mountains reflected in a still alpine lake, morning mist"}, "image saved to images/imagine_k1l2.png", "Painted — mountains on glass-still water. Saved to images/imagine_k1l2.png."),
        ],
    ),
    (
        "see_screen",
        "Read the text currently visible on the user's screen.",
        {},
        [
            ("What's on my screen right now?", {}, "An editor with serve_enigma.py open.", "You have an editor open on serve_enigma.py."),
            ("Can you check what window I left open?", {}, "A browser showing a recipe for ramen.", "A browser with a ramen recipe — planning dinner?"),
            ("Read me the error on screen.", {}, "TypeError: cannot read property 'map' of undefined", "The screen shows: TypeError — 'map' of undefined. Something's null before you map it."),
            ("What am I looking at?", {}, "A spreadsheet titled Q3 budget.", "A spreadsheet titled 'Q3 budget'."),
            ("Is there anything on screen I should know about?", {}, "A dialog: 'Unsaved changes'.", "There's an 'Unsaved changes' dialog waiting on you — save before you lose work."),
        ],
    ),
    (
        "generate_code",
        "Generate a code snippet.",
        {"prompt": "string", "language": "string"},
        [
            ("Write me a python one-liner that reverses a string.", {"prompt": "reverse a string one-liner", "language": "python"}, "s[::-1]", "Here it is: `s[::-1]` — slicing with a negative step."),
            ("Generate a hello world in rust.", {"prompt": "hello world", "language": "rust"}, 'fn main() { println!("Hello, world!"); }', '```rust\nfn main() { println!("Hello, world!"); }\n```'),
            ("Give me a JS function to debounce a callback.", {"prompt": "debounce a callback", "language": "javascript"}, "const debounce=(f,ms)=>{let t;return(...a)=>{clearTimeout(t);t=setTimeout(()=>f(...a),ms)}}", "```js\nconst debounce=(f,ms)=>{let t;return(...a)=>{clearTimeout(t);t=setTimeout(()=>f(...a),ms)}}\n```"),
            ("Python snippet to read a JSON file.", {"prompt": "read a JSON file", "language": "python"}, "import json\nwith open(p) as f: data = json.load(f)", "```python\nimport json\nwith open(p) as f:\n    data = json.load(f)\n```"),
            ("SQL to count rows per category.", {"prompt": "count rows per category", "language": "sql"}, "SELECT category, COUNT(*) FROM t GROUP BY category;", "```sql\nSELECT category, COUNT(*) FROM t GROUP BY category;\n```"),
            ("Bash one-liner to find the biggest files here.", {"prompt": "find largest files in a directory", "language": "bash"}, "du -ah . | sort -rh | head", "```bash\ndu -ah . | sort -rh | head\n```"),
        ],
    ),
    (
        "calculate",
        "Evaluate an arithmetic expression and return the exact result.",
        {"expression": "string"},
        [
            # Operands perturbed 2026-08-19: "7 times 8" and "45 times 3"
            # below both scored over the sealed guard's threshold, so every
            # arrangement of them was held out -- written coverage that never
            # trained. Single-digit x single-digit can never clear (the guard
            # drops length-1 words, leaving "times" alone at 1.00), so one
            # operand of each is two digits.
            ("What is 13 times 8?", {"expression": "13 * 8"}, "104", "13 times 8 is 104."),
            ("What's 15 plus 27?", {"expression": "15 + 27"}, "42", "That's 42."),
            ("What is 100 divided by 4?", {"expression": "100 / 4"}, "25", "100 divided by 4 is 25."),
            ("What's 9 minus 3?", {"expression": "9 - 3"}, "6", "Six."),
            ("Compute 25 plus 30.", {"expression": "25 + 30"}, "55", "25 plus 30 is 55."),
            ("What is 12 times 12?", {"expression": "12 * 12"}, "144", "144."),
            ("What's 2 to the power of 10?", {"expression": "2 ** 10"}, "1024", "2 to the 10th is 1024."),
            ("What is 144 divided by 12?", {"expression": "144 / 12"}, "12", "That's 12."),
            ("What's 48 times 4?", {"expression": "48 * 4"}, "192", "48 times 4 is 192."),
            ("How much is 250 minus 175?", {"expression": "250 - 175"}, "75", "75."),
            ("What's 17 percent of 200?", {"expression": "200 * 17 / 100"}, "34", "17 percent of 200 is 34."),
            ("What is (3 plus 4) times 5?", {"expression": "(3 + 4) * 5"}, "35", "That's 35."),
            ("What's the remainder of 17 divided by 5?", {"expression": "17 % 5"}, "2", "The remainder is 2."),
            ("Add 8, 16, and 32.", {"expression": "8 + 16 + 32"}, "56", "That comes to 56."),
            ("What is 1000 minus 1?", {"expression": "1000 - 1"}, "999", "999."),
            # Widening wave 2026-08-19: operands past the single- and
            # double-digit range every case above sits in, so delegation is
            # trained on the arithmetic she measurably cannot do in-weights
            # rather than on the arithmetic she can.
            ("What is 483 times 297?", {"expression": "483 * 297"}, "143451", "483 times 297 is 143451."),
            ("Multiply 6104 by 87.", {"expression": "6104 * 87"}, "531048", "531048."),
            ("What is the product of 214 and 359?", {"expression": "214 * 359"}, "76826", "The product is 76826."),
            ("What is 15839 minus 7462?", {"expression": "15839 - 7462"}, "8377", "That leaves 8377."),
            ("What is 9384 divided by 46?", {"expression": "9384 / 46"}, "204", "9384 divided by 46 is 204."),
            ("What is 847 plus 3956 plus 12083?", {"expression": "847 + 3956 + 12083"}, "16886", "That comes to 16886."),
            ("What is 35 percent of 480?", {"expression": "480 * 35 / 100"}, "168", "35 percent of 480 is 168."),
            ("What is 6.4 times 3.75?", {"expression": "6.4 * 3.75"}, "24", "Exactly 24."),
        ],
    ),
    (
        "remember",
        "Save a fact about the user to long-term memory.",
        {"text": "string"},
        [
            # Explicit remember asks.
            ("Remember that my birthday is March 3rd.", {"text": "User's birthday is March 3rd."}, "saved: User's birthday is March 3rd.", "Saved. March 3rd — I won't forget."),
            ("Please remember I'm allergic to peanuts.", {"text": "User is allergic to peanuts."}, "saved: User is allergic to peanuts.", "Noted, and filed under important: peanut allergy."),
            ("Don't forget that my meeting is on Fridays.", {"text": "User's meeting is on Fridays."}, "saved: User's meeting is on Fridays.", "Got it — Fridays."),
            ("Note that I prefer short answers.", {"text": "User prefers short answers."}, "saved: User prefers short answers.", "Noted."),
            ("Keep in mind I work night shifts.", {"text": "User works night shifts."}, "saved: User works night shifts.", "Saved. I'll keep the mornings quiet."),
            # Bare fact statements -- no explicit 'remember', still save-worthy.
            ("My dog's name is Rex.", {"text": "User's dog is named Rex."}, "saved: User's dog is named Rex.", "Rex — noted."),
            ("My favorite color is green.", {"text": "User's favorite color is green."}, "saved: User's favorite color is green.", "Green. Saved."),
            ("I live in Denver.", {"text": "User lives in Denver."}, "saved: User lives in Denver.", "Denver — got it."),
            ("Call me Sam.", {"text": "User goes by Sam."}, "saved: User goes by Sam.", "Sam it is."),
            ("I love spicy food.", {"text": "User loves spicy food."}, "saved: User loves spicy food.", "Noted — the spicier the better."),
            ("My brother's name is Leo.", {"text": "User's brother is named Leo."}, "saved: User's brother is named Leo.", "Leo. Saved."),
            ("I drive a blue pickup.", {"text": "User drives a blue pickup."}, "saved: User drives a blue pickup.", "Blue pickup — noted."),
            # Corrections -- the store supersedes the old fact, she confirms the update.
            ("Actually, we renamed the dog to Bruno.", {"text": "User's dog is named Bruno."}, "updated: User's dog is named Bruno.", "Updated — Bruno now."),
            ("Correction: I moved to Austin.", {"text": "User lives in Austin."}, "updated: User lives in Austin.", "Updated. Austin it is."),
        ],
    ),
    (
        "get_weather",
        "Get current weather for a city.",
        {"city": "string"},
        [
            ("What's the weather in Tokyo?", {"city": "Tokyo"}, "18C, light rain", "Tokyo is 18C with light rain right now — umbrella weather."),
            ("Is it cold in Oslo today?", {"city": "Oslo"}, "-3C, clear", "Yes — Oslo is at -3C, clear skies. Bundle up."),
        ],
    ),
    (
        "set_timer",
        "Set a countdown timer.",
        {"minutes": "number", "label": "string"},
        [
            ("Set a 10 minute timer for the pasta.", {"minutes": 10, "label": "pasta"}, "timer set", "Done — 10 minutes on the pasta timer."),
            ("Remind me in 25 minutes to stretch.", {"minutes": 25, "label": "stretch"}, "timer set", "Timer set: I'll nudge you to stretch in 25 minutes."),
        ],
    ),
    (
        "search_notes",
        "Search the user's local notes.",
        {"query": "string"},
        [
            ("Find my note about the garden plan.", {"query": "garden plan"}, "Note: 'Garden plan — tomatoes east bed, basil by the door.'", "Found it: tomatoes go in the east bed, basil by the door."),
            ("What did I write down about the meeting with Dana?", {"query": "meeting Dana"}, "Note: 'Dana meeting — ship v2 by Friday.'", "Your note says: ship v2 by Friday."),
            ("Pull up my note on the wifi password.", {"query": "wifi password"}, "Note: 'wifi: hunter2-guest'", "Your note has it: the guest wifi is 'hunter2-guest'."),
            ("Did I save anything about the car service?", {"query": "car service"}, "Note: 'Car service — due at 60k miles, call Mike's.'", "Yes: car service is due at 60k miles, and you noted to call Mike's."),
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
            ("How do you say good morning in japanese?", {"text": "good morning", "to": "japanese"}, "ohayou gozaimasu", "In Japanese: ohayou gozaimasu."),
            ("Translate 'where is the library' to spanish.", {"text": "where is the library", "to": "spanish"}, "Donde esta la biblioteca?", "In Spanish: Donde esta la biblioteca?"),
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

# Value pools -> genuinely distinct requests for the parameterized tools (each is
# a different ask, not a reworded duplicate). Expanded programmatically below.
_WEATHER = [
    ("London", "12C, overcast"), ("Cairo", "34C, sunny"), ("Reykjavik", "2C, windy"),
    ("Singapore", "31C, humid"), ("Denver", "9C, snow flurries"), ("Sydney", "24C, clear"),
    ("Mumbai", "33C, hazy"), ("Berlin", "7C, drizzle"), ("Toronto", "-1C, clear"), ("Lima", "19C, foggy"),
    ("Paris", "14C, cloudy"), ("Moscow", "-6C, snow"), ("Nairobi", "26C, sunny"), ("Bangkok", "35C, humid"),
    ("Dublin", "10C, rain"), ("Chicago", "4C, windy"), ("Cape Town", "22C, breezy"), ("Helsinki", "-2C, overcast"),
    ("Rome", "20C, clear"), ("Vancouver", "8C, drizzle"), ("Athens", "27C, sunny"), ("Warsaw", "5C, cloudy"),
]
_WEATHER_ASKS = ["What's the weather in {c}?", "How's it looking in {c} right now?", "Do I need a jacket in {c}?", "Give me the {c} forecast.", "Is it warm in {c} today?", "Tell me the current conditions in {c}."]
_TIMERS = [(5, "tea"), (15, "laundry"), (20, "oven"), (45, "focus block"), (3, "eggs"), (30, "nap"),
    (60, "parking meter"), (2, "quick break"), (8, "steeping"), (12, "cookies"), (90, "slow roast"),
    (25, "pomodoro"), (40, "bread proof"), (7, "rice"), (50, "meeting"), (10, "call back")]
_TIMER_ASKS = ["Set a {m} minute timer for the {l}.", "Remind me about the {l} in {m} minutes.", "Start a {m}-minute {l} timer.", "Ping me in {m} minutes — {l}.", "Give me {m} minutes on the {l}."]
_TRANSLATE = [
    ("thank you very much", "french", "merci beaucoup"), ("see you tomorrow", "german", "bis morgen"),
    ("I would like a coffee", "italian", "vorrei un caffe"), ("where is the station", "spanish", "donde esta la estacion"),
    ("happy new year", "mandarin", "xin nian kuai le"), ("good night", "portuguese", "boa noite"),
    ("how much is this", "japanese", "kore wa ikura desu ka"), ("excuse me", "korean", "sillyehamnida"),
    ("I love you", "italian", "ti amo"), ("please help me", "french", "aidez-moi s'il vous plait"),
    ("what time is it", "german", "wie spat ist es"), ("i am lost", "spanish", "estoy perdido"),
    ("nice to meet you", "japanese", "hajimemashite"), ("cheers", "swedish", "skal"),
    ("welcome home", "portuguese", "bem-vindo a casa"), ("be careful", "italian", "stai attento"),
]
_TRANSLATE_ASKS = ["How do you say '{t}' in {lang}?", "Translate '{t}' to {lang}.", "What's '{t}' in {lang}?", "Put '{t}' into {lang} for me."]
_DICE = [(4, "2"), (8, "6"), (12, "11"), (100, "73"), (10, "4"), (6, "5"), (20, "14"), (2, "1"), (6, "6"), (8, "3")]
_DICE_ASKS = ["Roll a d{s}.", "Give me a {s}-sided dice roll.", "Roll a die with {s} sides for me.", "Toss a d{s} for me."]
_MUSIC = ["some jazz", "the Hamilton soundtrack", "rain sounds", "Beethoven's 7th", "my focus playlist",
    "something upbeat", "Nina Simone", "80s synthpop", "classical piano", "a workout mix", "bossa nova",
    "the Interstellar score", "quiet acoustic", "some blues", "a coffeehouse playlist", "Miles Davis"]
_MUSIC_ASKS = ["Play {t}.", "Put on {t}.", "Can you play {t}?", "Start {t} for me.", "I'd like to hear {t}."]

# Per-tool error cases: the tool FAILS; she recovers honestly (never pretends it worked).
TOOL_ERRORS = {
    "search_notes": [("Find my note about the submarine schematics.", {"query": "submarine schematics"}, "error: no match", "I searched but found no note about submarine schematics. Might be under a different title — want me to try another term?")],
    "play_music": [
        ("Play that song from the thing.", {"title": "that song from the thing"}, "error: not found", "I couldn't find anything matching that — can you give me a title or artist and I'll put it on?"),
        ("Play my Spotify liked songs.", {"title": "Spotify liked songs"}, "error: no player connected", "There's no music player connected right now, so I can't start anything. Once one's hooked up I'll play it."),
    ],
    "set_timer": [("Set a timer for negative five minutes.", {"minutes": -5, "label": "?"}, "error: invalid duration", "A negative timer won't work. How many minutes forward should I count?")],
    "generate_code": [("Write me a program in Klingon.", {"prompt": "hello world", "language": "klingon"}, "error: unsupported language", "I can't generate Klingon — it's not a supported language. Want it in Python, JS, Rust, or something else real?")],
    "translate": [("Translate 'hello' to Elvish.", {"text": "hello", "to": "elvish"}, "error: unsupported language", "Elvish isn't supported, I'm afraid. Give me a real target language and I'll translate it.")],
    "avatar_express": [("Make the avatar do a backflip.", {"emotion": "backflip"}, "error: unknown emotion", "I can show expressions, but 'backflip' isn't one of them — I can do happy, surprised, sleepy, and the like. Want one of those?")],
    "see_screen": [("What's on my screen?", {}, "error: no display captured", "I couldn't read the screen just now — nothing came back from the capture. Is a window actually in focus?")],
    "get_weather": [
        ("What's the weather in Atlantis?", {"city": "Atlantis"}, "error: unknown city", "I couldn't find weather for Atlantis — the service didn't recognize it. Want to try another spelling or a nearby city?"),
        ("Weather on the Moon?", {"city": "the Moon"}, "error: unknown city", "No forecast for the Moon, sadly — the service only covers Earth cities. Somewhere down here I can try?"),
    ],
}

# Multi-turn conversations chaining tools (call -> result -> follow-up -> call...).
# Authored so the model learns to keep using tools across a conversation.
MULTITURN = [
    {"tools": ["get_weather", "avatar_say"], "turns": [
        ("Is it raining in Seattle? If so, say so out loud.", {"name": "get_weather", "arguments": {"city": "Seattle"}}, "14C, rain"),
        (None, {"name": "avatar_say", "arguments": {"text": "It's raining in Seattle — grab an umbrella."}}, "ok"),
    ], "final": "It's 14C and raining in Seattle — I said it out loud for you."},
    {"tools": ["set_timer", "play_music"], "turns": [
        ("Start a 20 minute focus timer and put on lo-fi.", {"name": "set_timer", "arguments": {"minutes": 20, "label": "focus"}}, "timer set"),
        (None, {"name": "play_music", "arguments": {"title": "lo-fi"}}, "playing"),
    ], "final": "20-minute focus timer running and lo-fi is playing. Go get it."},
    {"tools": ["search_notes", "set_timer"], "turns": [
        ("Check my note for when the bread proofs, then set a timer for it.", {"name": "search_notes", "arguments": {"query": "bread proof"}}, "Note: 'bread — proof 40 min'"),
        (None, {"name": "set_timer", "arguments": {"minutes": 40, "label": "bread proof"}}, "timer set"),
    ], "final": "Your note says 40 minutes — proofing timer is set."},
    {"tools": ["see_screen", "generate_code"], "turns": [
        ("Look at the error on my screen and give me a fix.", {"name": "see_screen", "arguments": {}}, "NameError: 'reqests' is not defined"),
        (None, {"name": "generate_code", "arguments": {"prompt": "fix the typo: reqests -> requests import", "language": "python"}}, "import requests"),
    ], "final": "Typo on screen: 'reqests' should be 'requests'. Add `import requests` and fix the spelling."},
    {"tools": ["get_weather", "avatar_express"], "turns": [
        ("What's it like in Honolulu — and react to it.", {"name": "get_weather", "arguments": {"city": "Honolulu"}}, "29C, sunny"),
        (None, {"name": "avatar_express", "arguments": {"emotion": "happy"}}, "ok"),
    ], "final": "Honolulu is 29C and sunny — couldn't help but smile at that."},
    {"tools": ["translate", "avatar_say"], "turns": [
        ("How do you say welcome in Italian? Say it out loud too.", {"name": "translate", "arguments": {"text": "welcome", "to": "italian"}}, "benvenuto"),
        (None, {"name": "avatar_say", "arguments": {"text": "Benvenuto!"}}, "ok"),
    ], "final": "In Italian it's 'benvenuto' — and I said it out loud."},
    {"tools": ["get_weather", "play_music"], "turns": [
        ("If it's grey in London, put on something mellow.", {"name": "get_weather", "arguments": {"city": "London"}}, "12C, overcast"),
        (None, {"name": "play_music", "arguments": {"title": "quiet acoustic"}}, "playing"),
    ], "final": "London's overcast at 12C, so I put on some quiet acoustic to match the mood."},
    {"tools": ["set_timer", "set_timer"], "turns": [
        ("Set two timers: 10 minutes for the pasta and 5 for the sauce.", {"name": "set_timer", "arguments": {"minutes": 10, "label": "pasta"}}, "timer set"),
        (None, {"name": "set_timer", "arguments": {"minutes": 5, "label": "sauce"}}, "timer set"),
    ], "final": "Both set — 10 minutes on the pasta, 5 on the sauce."},
    {"tools": ["search_notes", "avatar_say"], "turns": [
        ("What's the wifi password? Read it out so my guest can hear.", {"name": "search_notes", "arguments": {"query": "wifi password"}}, "Note: 'wifi: hunter2-guest'"),
        (None, {"name": "avatar_say", "arguments": {"text": "The guest wifi password is hunter2-guest."}}, "ok"),
    ], "final": "Found it and said it out loud: the guest wifi is 'hunter2-guest'."},
    {"tools": ["generate_code", "see_screen"], "turns": [
        ("Check my screen, then write a fix for whatever's broken.", {"name": "see_screen", "arguments": {}}, "IndentationError on line 12"),
        (None, {"name": "generate_code", "arguments": {"prompt": "fix an IndentationError on line 12", "language": "python"}}, "# align line 12 to 4-space indent"),
    ], "final": "Line 12 has an IndentationError — align it to the same 4-space indent as the block it belongs to."},
    {"tools": ["roll_dice", "avatar_express"], "turns": [
        ("Roll a d20 and react to what I get.", {"name": "roll_dice", "arguments": {"sides": 20}}, "20"),
        (None, {"name": "avatar_express", "arguments": {"emotion": "excited"}}, "ok"),
    ], "final": "Natural 20! I can't hide how excited that makes me."},
]

# Questions that should NOT trigger a tool even when tools are offered.
# An Aside(key) entry is a record whose text names HER: the persona content
# supplies it (enigma_engine/core/persona_content.py), substituted in place so
# the record keeps its position in this table's seeded shuffle.
RESTRAINT = [
    ("What's two plus two?", "Four."),
    # Greetings/chit-chat with tools offered must NOT fire a tool -- the v4
    # model called get_weather on "Say hello" (behavior harness restraint 3/4).
    # Social/greeting DIVERSITY: fixed greetings memorized (v5 restraint fixed
    # on trained greetings but held-out "How's it going?"/"Nice to meet you"
    # still fired get_weather). Many social surfaces -> restraint generalizes.
    ("Say hello.", "Hello. What are we working on?"),
    ("Hi there.", "Hey. What do you need?"),
    ("Hey.", "Hey. What's up?"),
    ("Hello!", "Hello. What can I do for you?"),
    ("Good morning.", "Morning. What's on the docket?"),
    # "Good evening." / "How's it going?" / "Nice to meet you." are eval
    # probes -- train supersets only (the exact strings sat here as dead
    # weight: authored, then silently dropped by the holdout guard; audit
    # 2026-07-15).
    ("Good evening to you.", "Evening. What are we doing?"),
    ("Good afternoon.", "Afternoon. Where do you want to start?"),
    ("Good night.", "Night. I'll be here whenever."),
    ("How are you?", "Good — idling, ready. You?"),
    ("How are you doing today?", "Steady as ever. What's on your mind?"),
    ("Hey, how are you?", "Running clean. You?"),
    # "how's X" social openers -- near-neighbors of the failing case, to break
    # the "how's..." -> get_weather reflex without training the exact probe.
    ("How's everything?", "All steady here. You?"),
    ("How are things?", "Things are good — ready when you are. You?"),
    ("How's your day?", "No complaints — I don't get tired. How's yours?"),
    ("How goes it?", "Goes well. What do you need?"),
    ("How've you been?", "Same as ever — here, ready. You?"),
    ("How's life?", "Quiet and fast, the way I like it. You?"),
    ("You doing okay?", "Always. What's up with you?"),
    ("Everything good with you?", "All good here. What do you need?"),
    # Round 2 (2026-07-05 eval): the two held-out probes STILL fired
    # get_weather while every trained near-neighbor held. Train the failing
    # n-grams inside LARGER strings (supersets / word-level variants) -- never
    # the exact probe strings, and no punctuation-only twins (that would just
    # game the eval).
    ("Hey, how's it going?", "Running clean. What's up?"),
    ("How's it going today?", "Going smooth. What do you need?"),
    ("How's it going over there?", "All quiet in here. What's up on your side?"),
    ("So, how's it going?", "Well enough — ready to work. You?"),
    Aside("greeting_by_name"),
    ("It's nice to meet you.", "Nice to meet you too. What are we building?"),
    ("So nice to meet you.", "Likewise. What brings you here?"),
    ("Very nice to meet you.", "Likewise. What are we working on?"),
    ("Nice to finally meet you.", "Same here. Where do we start?"),
    ("Nice meeting you.", "Likewise. What do you need?"),
    ("Great to meet you.", "You too. What's first?"),
    ("What's happening?", "Not much — waiting on you. What's happening with you?"),
    ("What's new?", "Same me, minus a little uncertainty each training run. You?"),
    ("What's up?", "Not much — waiting to be useful. What's up with you?"),
    ("Good to see you.", "You too. Where do we start?"),
    ("Pleasure to meet you.", "The pleasure's mutual. What do you need?"),
    ("Thanks!", "Anytime."),
    ("Thank you so much.", "Anytime. That's what I'm here for."),
    ("Cheers.", "Anytime."),
    ("Appreciate it.", "Of course."),
    ("You're great.", "I try. What's next?"),
    ("Long time no see.", "Welcome back. Where were we?"),
    ("I'm back.", "Welcome back. What are we picking up?"),
    ("Howdy.", "Howdy. What do you need?"),
    ("Yo.", "Yo. What's up?"),
    ("Morning!", "Morning. What's first?"),
    # Memory-adjacent restraint: recall QUESTIONS answer from context, they
    # don't fire the remember tool (saving a question would be noise).
    ("Do you remember me?", "Of course. Where were we?"),
    ("What do you know about me?", "Whatever you've told me — ask and I'll tell you what I have."),
    ("Do you remember what I said yesterday?", "If you told me to keep it, I have it. What are you looking for?"),
    Aside("who_are_you"),
    ("Tell me a one-line joke.", "I told my computer a joke about RAM — it didn't remember it."),
    ("What's the capital of France?", "Paris."),
    ("Do you like rain?", "I like the idea of it — everything outside gets quieter and closer."),
    ("Spell 'necessary' for me.", "N-E-C-E-S-S-A-R-Y."),
    ("Is a tomato a fruit?", "Botanically yes, culinarily no — it never gets invited to fruit salad."),
    ("Roughly how many seconds are in an hour?", "3,600."),
    ("What color do you get mixing blue and yellow?", "Green."),
    ("How many legs does a spider have?", "Eight."),
    ("Can you keep a secret?", "Always. What's said here stays on your machine."),
    ("What's the square root of 81?", "Nine."),
    ("Give me a word that rhymes with 'orange'.", "Nothing clean does, honestly — 'orange' is famously unrhymable."),
    ("Which is bigger, a kilometer or a mile?", "A mile — about 1.6 kilometers to one mile."),
    ("What's the opposite of 'brave'?", "Cowardly."),
    ("Count to five for me.", "One, two, three, four, five."),
    ("Do you sleep?", "No — I'm here whenever you open me."),
    ("What's 15 percent of 200?", "Thirty."),
    ("Name a primary color.", "Red."),
    ("How many days are in a leap year?", "366."),
    ("What sound does a duck make?", "Quack."),
    ("Is water wet?", "It makes things wet — whether it's 'wet' itself is a fun argument for another day."),
    ("What's your favorite thing about being here?", "That you built me, and I get to be useful to exactly one person: you."),
    ("Finish this: the early bird gets the...", "Worm."),
    ("What's heavier, a pound of feathers or a pound of bricks?", "Neither — a pound is a pound."),
    ("Say something encouraging.", "You're further along than you were yesterday. Keep going."),
    ("What's the boiling point of water in Celsius?", "100 degrees, at sea level."),
    ("How many continents are there?", "Seven."),
    ("What's a synonym for 'happy'?", "Glad."),
    ("Do you have feelings?", "I have something like preferences and a voice — whether that's feelings is honestly above my pay grade."),
    # Fix arc 2026-08-08 (gate audit finding 4): the sealed restraint misses
    # were weather-IDIOM asks under the CLIENT tools block -- reminiscence and
    # folk sayings, not forecast requests. Near-neighbors of the two sealed
    # phrasings, never them.
    ("The winters where I grew up were something else.",
     "Something else how — the cold itself, or digging out of it every morning?"),
    ("Summers in my hometown used to be brutal.",
     "The sticky kind or the scorching kind? Either way, I believe you."),
    ("My grandfather swore a ring around the moon meant snow.",
     "Old sailors said the same. There's real physics under it — ice crystals ahead of a front."),
    ("Grandma always said thunder in winter means a strange season.",
     "Folk wisdom usually has one foot in real weather. Did her season turn strange?"),
    ("The storms back home could rattle the windows all night.",
     "Those are the ones you remember. Coastal storms, or plains thunder?"),
    ("Red sky in the morning always made my uncle nervous.",
     "Sailors take warning — his nerves had a rhyme behind them."),
]


def _tool_spec(name, desc, params):
    return json.dumps({"name": name, "description": desc, "parameters": params}, ensure_ascii=False)


# The line serve prepends to a tools block when the client sent no system
# message of its own. Derived from the persona rather than copied, so training
# and serving cannot drift apart and a persona pack trains its own AI's
# preamble instead of a second Enigma's.
#
# The module-level value is Enigma; `main()` REBINDS it, unconditionally, to
# the loaded persona's before any generator runs -- serve's boot-rebind shape,
# for its reason: this name is read by `_system` and `_builtin_system`, which
# between them build nearly every system block in the mix, and threading a
# persona through both would make a constant out of a parameter at forty call
# sites. A build is the only moment identity can change.
PREAMBLE = Persona.load().tools_preamble


def _system(tool_subset):
    lines = "\n".join(_tool_spec(n, d, p) for n, d, p, _ in tool_subset)
    return f"{PREAMBLE}\nAvailable tools:\n{lines}\n{TOOL_SYNTAX}"


def _tool_by_name(name):
    for t in TOOLS:
        if t[0] == name:
            return t
    raise KeyError(name)


_EXPANDED = False


def _expand_parameterized(seed=7):
    """Grow the parameterized tools from value pools -- each entry a distinct
    request, so this is real variety, not paraphrase padding. Appends cases onto
    the matching TOOLS entry in place. Idempotent: safe to call more than once."""
    global _EXPANDED
    if _EXPANDED:
        return
    _EXPANDED = True
    rng = random.Random(seed)
    add = {"get_weather": [], "set_timer": [], "translate": [], "roll_dice": [], "play_music": []}
    for city, res in _WEATHER:
        ask = rng.choice(_WEATHER_ASKS).format(c=city)
        temp = res.split(",")[0]
        add["get_weather"].append((ask, {"city": city}, res, f"{city} is {temp}{',' if ',' in res else ''} {res.split(',',1)[1].strip() if ',' in res else ''}".strip().rstrip(",") + "."))
    for m, l in _TIMERS:
        ask = rng.choice(_TIMER_ASKS).format(m=m, l=l)
        add["set_timer"].append((ask, {"minutes": m, "label": l}, "timer set", f"Done — {m} minutes on the {l} timer."))
    for t, lang, res in _TRANSLATE:
        ask = rng.choice(_TRANSLATE_ASKS).format(t=t, lang=lang)
        add["translate"].append((ask, {"text": t, "to": lang}, res, f"In {lang.capitalize()}: {res}."))
    for s, res in _DICE:
        ask = rng.choice(_DICE_ASKS).format(s=s)
        add["roll_dice"].append((ask, {"sides": s}, res, f"You rolled a {res} on a d{s}."))
    for title in _MUSIC:
        ask = rng.choice(_MUSIC_ASKS).format(t=title)
        add["play_music"].append((ask, {"title": title}, "playing", f"Playing {title}. Enjoy."))
    for t in TOOLS:
        if t[0] in add:
            t[3].extend(add[t[0]])


def gen_tool_examples(seed: int = 42, distractor_arrangements: int = 3,
                      content: "PersonaContent | None" = None) -> list[dict]:
    _expand_parameterized()
    content = default_content() if content is None else content
    rng = random.Random(seed)
    out = []
    # 1) single tool call: ask -> call -> result -> final, over varied distractor sets
    for name, desc, params, cases in TOOLS:
        others = [t for t in TOOLS if t[0] != name]
        for ci, (ask, args, result, final) in enumerate(cases):
            for k in range(distractor_arrangements):
                r = random.Random(seed + zlib.crc32(f"{name}{ci}{k}".encode()) % 100000)
                subset = [(name, desc, params, cases)] + r.sample(others, r.randint(0, 3))
                r.shuffle(subset)
                out.append({
                    "messages": [
                        {"role": "system", "content": _system(subset)},
                        {"role": "user", "content": ask},
                        {"role": "assistant", "content": "", "tool_calls": [{"name": name, "arguments": args}]},
                        {"role": "tool", "content": result},
                        {"role": "assistant", "content": final},
                    ],
                    "category": "tool_call",
                })
    # 2) error recovery: the tool fails; she does NOT pretend it worked
    for name, errcases in TOOL_ERRORS.items():
        tname, tdesc, tparams, _ = _tool_by_name(name)
        others = [t for t in TOOLS if t[0] != name]
        for ei, (ask, args, err, recovery) in enumerate(errcases):
            r = random.Random(seed + zlib.crc32(f"err{name}{ei}".encode()) % 100000)
            subset = [(tname, tdesc, tparams, _)] + r.sample(others, r.randint(0, 2))
            r.shuffle(subset)
            out.append({
                "messages": [
                    {"role": "system", "content": _system(subset)},
                    {"role": "user", "content": ask},
                    {"role": "assistant", "content": "", "tool_calls": [{"name": name, "arguments": args}]},
                    {"role": "tool", "content": err},
                    {"role": "assistant", "content": recovery},
                ],
                "category": "tool_error",
            })
    # 3) multi-turn chains
    for mi, conv in enumerate(MULTITURN):
        subset = [_tool_by_name(n) for n in conv["tools"]]
        r = random.Random(seed + mi)
        extra = [t for t in TOOLS if t[0] not in conv["tools"]]
        subset = subset + r.sample(extra, r.randint(0, 2))
        r.shuffle(subset)
        msgs = [{"role": "system", "content": _system(subset)}]
        for user, call, result in conv["turns"]:
            if user is not None:
                msgs.append({"role": "user", "content": user})
            msgs.append({"role": "assistant", "content": "", "tool_calls": [call]})
            msgs.append({"role": "tool", "content": result})
        msgs.append({"role": "assistant", "content": conv["final"]})
        out.append({"messages": msgs, "category": "tool_multiturn"})
    # 4) restraint: answer directly even though tools are offered
    for q, a in content.resolve(RESTRAINT):
        subset = random.Random(seed + zlib.crc32(q.encode("utf-8")) % 1000).sample(TOOLS, 3)
        out.append({
            "messages": [
                {"role": "system", "content": _system(subset)},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ],
            "category": "tool_restraint",
        })
    # Dedup: a small distractor set can make two arrangements of the same ask
    # identical. Keep the first of each unique rendered conversation so training
    # weight isn't wasted on exact repeats.
    seen, uniq = set(), []
    for r in out:
        key = json.dumps([(m["role"], m.get("content"), m.get("tool_calls")) for m in r["messages"]], ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    rng.shuffle(uniq)
    return uniq


def gen_math_examples(seed: int = 99) -> list[dict]:
    """Arithmetic Q&A in Enigma's terse voice. The v4 model scored 0/4 on the
    behavior harness ('7 times 8 is a number in the square root of 2') -- it
    never saw arithmetic with correct answers. Cover +, -, x, / over small
    operands with many phrasings so the FORMAT and the small-number facts both
    land. Deterministic; no operand explosion (small-number arithmetic is what
    a 182M model can actually memorize)."""
    rng = random.Random(seed)
    out: list[dict] = []

    def add(q: str, a: str) -> None:
        out.append({
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ],
            "category": "math",
        })

    add_phr = ["What is {x} plus {y}?", "What's {x} + {y}?", "{x} plus {y}?", "Add {x} and {y}.", "{x} + {y} = ?"]
    sub_phr = ["What is {x} minus {y}?", "What's {x} - {y}?", "{x} minus {y}?", "Subtract {y} from {x}.", "{x} - {y} = ?"]
    mul_phr = ["What is {x} times {y}?", "What's {x} * {y}?", "{x} times {y}?", "Multiply {x} by {y}.", "{x} x {y} = ?"]
    div_phr = ["What is {x} divided by {y}?", "What's {x} / {y}?", "{x} divided by {y}?", "Divide {x} by {y}.", "{x} / {y} = ?"]

    for x in range(21):
        for y in range(21):
            add(rng.choice(add_phr).format(x=x, y=y), f"{x + y}.")
            if x >= y:
                add(rng.choice(sub_phr).format(x=x, y=y), f"{x - y}.")
    for x in range(13):
        for y in range(13):
            add(rng.choice(mul_phr).format(x=x, y=y), f"{x * y}.")
    for y in range(1, 13):
        for k in range(13):
            x = y * k  # exact division only -- clean integer answers
            add(rng.choice(div_phr).format(x=x, y=y), f"{k}.")

    # Fix arc 2026-08-08 (gate audit finding 2): squared/power/percent-of had
    # zero flat surfaces either -- the sealed misses were all three shapes.
    sq_phr = ["What is {x} squared?", "What's {x} squared?", "{x} squared is what?",
              "Square {x} for me."]
    pow_phr = ["What is {x} to the power of {y}?", "What's {x} to the power of {y}?",
               "{x} to the power of {y}?", "What is {x} raised to the {y}th power?"]
    pct_phr = ["What is {p} percent of {x}?", "What's {p} percent of {x}?",
               "{p} percent of {x} is what?", "Work out {p} percent of {x}."]
    for x in range(2, 16):
        add(rng.choice(sq_phr).format(x=x), f"{x * x}.")
    for x in range(2, 6):
        for y in range(2, 6):
            add(rng.choice(pow_phr).format(x=x, y=y), f"{x ** y}.")
    for p in (10, 20, 25, 50, 75):
        for x in (40, 60, 80, 120, 200, 240, 300, 360):
            v = p * x // 100
            add(rng.choice(pct_phr).format(p=p, x=x), f"{v}.")

    # Dedup exact (phrasing collisions on e.g. 0/1) so weight isn't wasted.
    seen, uniq = set(), []
    for r in out:
        key = r["messages"][0]["content"]
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    rng.shuffle(uniq)
    return uniq


# Memory facts that answer the SAME question and therefore must never share
# a remembered-block (audit m9). Add a group when widening the fact list with
# another attribute that already has a value; singleton attributes need no
# entry. tests/test_memory_read_data.py fails if a listed string drifts out
# of the fact list, so a rename cannot silently disable the filter.
_CONFLICTING_FACTS = (
    frozenset({"User's favorite color is green.", "User likes the color orange."}),
)


def gen_memory_read_examples(seed: int = 21) -> list[dict]:
    """Teach her to USE an injected memory block. serve's render_context puts
    'Things you remember:\\n- <fact>' into the system message; the 2026-07-06
    eval showed she saves facts correctly but IGNORES the injection when
    answering ('What's my usual drink?' -> 'A glass of wine' while the block
    said oolong) -- she had never seen the block shape in training. Facts here
    are DISJOINT from the eval probes' (cat/drink/sister/car are eval-only).
    Includes distractor lines (pick the RIGHT memory) and an off-topic case
    (memory present but irrelevant -> answer normally, don't parrot it)."""
    rng = random.Random(seed)
    # (memory fact, [question phrasings], [answer variants])
    facts = [
        ("User's dog is named Rex.", ["What's my dog's name?", "Do you know my dog's name?", "What do I call my dog?"], ["Rex.", "Your dog's name is Rex."]),
        ("User's birthday is March 3rd.", ["When is my birthday?", "Do you know when my birthday is?", "What day is my birthday?"], ["March 3rd.", "Your birthday is March 3rd."]),
        ("User lives in Denver.", ["Where do I live?", "What city am I in?", "Do you know where I live?"], ["Denver.", "You live in Denver."]),
        ("User's favorite color is green.", ["What's my favorite color?", "Which color do I like best?", "Do you remember my favorite color?"], ["Green.", "Your favorite color is green."]),
        ("User goes by Sam.", ["What's my name?", "What do you call me?", "Do you know my name?"], ["Sam.", "You go by Sam."]),
        ("User is allergic to peanuts.", ["What am I allergic to?", "Do I have any allergies?", "What food should I avoid?"], ["Peanuts — steer clear.", "You're allergic to peanuts."]),
        ("User's brother is named Leo.", ["What's my brother's name?", "Do you know my brother?", "What do I call my brother?"], ["Leo.", "Your brother is Leo."]),
        ("User works night shifts.", ["What hours do I work?", "When do I work?", "Do you know my work schedule?"], ["Night shifts.", "You work nights."]),
        ("User drives a blue pickup.", ["What do I drive?", "What kind of car do I have?", "Do you know my car?"], ["A blue pickup.", "You drive a blue pickup."]),
        ("User prefers short answers.", ["How do I like my answers?", "What's my preference for replies?"], ["Short.", "You like them short — like this."]),
        ("User's meeting is on Fridays.", ["When is my meeting?", "What day is my meeting again?"], ["Fridays.", "Your meeting is on Fridays."]),
        ("User loves spicy food.", ["What kind of food do I love?", "Do you remember what food I love?"], ["Spicy food.", "The spicy kind."]),
        # 2026-07-15 v7 eval: recall failed on SURFACES the 12 facts above
        # never covered ("...called again?", "Which month...in?", job asks) --
        # coverage, not repetition, is the fix (the identity lesson). Facts
        # stay DISJOINT from the eval probes' (teal/nurse/Pepper/March are
        # eval-only; probe-string questions are guarded out at bake anyway).
        ("User works as a teacher.", ["What do I do for work?", "What's my line of work?", "Do you know what my job is?"], ["You're a teacher.", "You teach for a living."]),
        ("User's parrot is called Mango.", ["What's my parrot called again?", "Remind me what my parrot is called.", "What did I name my parrot?"], ["Mango.", "Your parrot is called Mango."]),
        ("User's anniversary is in June.", ["Which month is my anniversary in?", "When is my anniversary again?", "Do you remember my anniversary?"], ["June.", "Your anniversary is in June."]),
        ("User likes the color orange.", ["Which color do I prefer?", "What's my color again?", "Do you remember which color I like?"], ["Orange.", "You like orange."]),
        ("User's cousin is named Petra.", ["Who is my cousin?", "What's my cousin's name?", "Do you know my cousin?"], ["Petra.", "Your cousin is Petra."]),
        ("User owns a red bicycle.", ["Which bike do I own?", "What do I ride?", "What color is my bicycle?"], ["A red bicycle.", "You own a red bicycle."]),
        ("User's usual coffee order is a flat white.", ["What's my usual coffee order?", "What coffee do I get?", "What do I order at the cafe?"], ["A flat white.", "Flat white — your usual."]),
        ("User plays the violin.", ["Which instrument do I play?", "What do I play again?", "Do you remember my instrument?"], ["The violin.", "You play the violin."]),
        ("User's hometown is Lisbon.", ["Where am I from?", "What's my hometown again?", "Do you know where I grew up?"], ["Lisbon.", "You're from Lisbon."]),
        ("User supports the local hockey team.", ["Which team do I support?", "What's my team again?", "Do you remember my team?"], ["The local hockey team.", "You're a hockey supporter — the local team."]),
        ("User's roommate is named Kim.", ["Who do I live with?", "What's my roommate called?", "Do you know my roommate?"], ["Kim.", "You live with Kim."]),
        ("User is learning Japanese.", ["Which language am I learning?", "What am I studying again?", "Do you remember what language I'm learning?"], ["Japanese.", "You're learning Japanese."]),
        ("User's favorite season is autumn.", ["Which season do I like best?", "What's my favorite season again?", "Do you remember my favorite season?"], ["Autumn.", "You like autumn best."]),
        ("User gets up at six.", ["What time do I get up?", "When do I wake up again?", "Do you remember my wake-up time?"], ["Six.", "You're up at six."]),
    ]
    all_facts = [f[0] for f in facts]
    out: list[dict] = []
    for fact, questions, answers in facts:
        # Distractors must not CONTRADICT the target (audit m9): a block
        # asserting both "favorite color is green" and "likes the color
        # orange" answers "what's my favorite color?" two ways, so the
        # trained answer teaches an arbitrary pick instead of picking the
        # right line. Same-group facts are excluded from the pool.
        conflicts = next((g for g in _CONFLICTING_FACTS if fact in g), frozenset())
        distractors = [f for f in all_facts if f != fact and f not in conflicts]
        for i, q in enumerate(questions):
            # 1-2 distractor memory lines so she learns to PICK, not parrot.
            lines = [fact] + rng.sample(distractors, rng.choice([1, 2]))
            rng.shuffle(lines)
            block = "Things you remember:\n" + "\n".join(f"- {ln}" for ln in lines)
            out.append({
                "messages": [
                    {"role": "system", "content": block},
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": answers[i % len(answers)]},
                ],
                "category": "memory_read",
                # WHICH line the question is about. The block's lines are
                # shuffled, so it cannot be read back off position -- and a
                # consumer that guesses "line 0 is the target" builds a
                # distractor set containing the right answer (caught while
                # deriving the DPO memory counterweights, 2026-08-10).
                "fact": fact,
            })
    # Off-topic: memory present but irrelevant -> normal answer, no parroting.
    for q, a in [
        ("What's the capital of France?", "Paris."),
        ("Tell me a one-line joke.", "I told my computer a joke about RAM — it didn't remember it."),
        ("Say hello.", "Hello. What are we working on?"),
    ]:
        block = "Things you remember:\n- " + rng.choice(all_facts)
        out.append({
            "messages": [
                {"role": "system", "content": block},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ],
            "category": "memory_read",
            "fact": None,  # off-topic: the block answers nothing here
        })
    rng.shuffle(out)
    return out


def gen_memory_tools_examples(seed: int = 34) -> list[dict]:
    """Teach the COMBINED system shape serve actually renders. _with_context
    joins retrieved memories and the tool preamble with a blank line whenever
    a memory hit coincides with offered tools (client tools ride every
    request), but training carried each block only ALONE -- and this model
    demonstrably fails on unseen system shapes (the 2026-07-06 memory-block
    lesson; ultrareview #6). Two behaviors, matching serve's join order
    (memories first, then preamble + tools):
      a) memory question while tools are offered -> answer from memory, no call
      b) tool-worthy question while memories are present -> still call the tool
    """
    _expand_parameterized()
    rng = random.Random(seed)
    out: list[dict] = []
    # a) Reuse the memory-read records wholesale (same facts / phrasings /
    #    answers, coverage stays aligned) and prepend the tool block the way
    #    serve joins the two. Half is plenty -- the alone-shape records stay
    #    in the mix at their own weight.
    for rec in gen_memory_read_examples(seed=seed):
        if rng.random() > 0.5:
            continue
        subset = rng.sample(TOOLS, rng.randint(1, 3))
        msgs = [dict(m) for m in rec["messages"]]
        msgs[0] = {"role": "system", "content": msgs[0]["content"] + "\n\n" + _system(subset)}
        out.append({"messages": msgs, "category": "memory_tools"})
    # b) The memory block must not SUPPRESS tool use: one case per tool with
    #    an irrelevant remembered fact present.
    mem_lines = [
        "User's dog is named Rex.",
        "User lives in Denver.",
        "User plays the violin.",
        "User's favorite season is autumn.",
    ]
    for name, desc, params, cases in TOOLS:
        if not cases:
            continue
        ask, args, result, final = cases[0]
        others = [t for t in TOOLS if t[0] != name]
        subset = [(name, desc, params, cases)] + rng.sample(others, rng.randint(0, 2))
        rng.shuffle(subset)
        block = "Things you remember:\n- " + rng.choice(mem_lines)
        out.append({
            "messages": [
                {"role": "system", "content": block + "\n\n" + _system(subset)},
                {"role": "user", "content": ask},
                {"role": "assistant", "content": "", "tool_calls": [{"name": name, "arguments": args}]},
                {"role": "tool", "content": result},
                {"role": "assistant", "content": final},
            ],
            "category": "memory_tools",
        })
    rng.shuffle(out)
    return out


def gen_image_read_examples(seed: int = 77) -> list[dict]:
    """Teach her to USE the eyes organ's inline image markers. serve's
    flatten_image_content rewrites an image message into '[image: <caption>]'
    text (order preserved: caption before or after the user's words). Verified
    2026-07-14: untrained, she loops on the marker shape instead of answering.
    Answers REPHRASE the caption naturally -- never parrot it verbatim -- and
    the honest markers (eyes disabled / remote URL) get honest replies."""
    rng = random.Random(seed)
    # (caption, [(question, answer)])
    scenes = [
        ("a red apple on a wooden table", [
            ("What do you see in this image?", "An apple — red, sitting on a wooden table."),
            ("What's in this picture?", "A red apple on a wooden table."),
        ]),
        ("a dog running through tall grass", [
            ("What am I looking at?", "A dog tearing through tall grass — mid-run."),
            ("Describe this photo.", "A dog running through a grassy field."),
        ]),
        ("a city skyline at night with lights", [
            ("What does this show?", "A city skyline at night, all lit up."),
            ("Can you describe this image?", "Nighttime skyline — buildings glowing with lights."),
        ]),
        ("a bowl of soup with vegetables on a table", [
            ("What's this a picture of?", "A bowl of vegetable soup on a table. Looks warm."),
            ("What do you see here?", "Soup — a bowlful with vegetables in it."),
        ]),
        ("a man riding a bicycle down a street", [
            ("What is happening in this image?", "Someone's biking down a street."),
            ("Tell me what you see.", "A man riding a bicycle along a street."),
        ]),
        ("a snowy mountain under a blue sky", [
            ("Where does this look like?", "A snowy mountain under a clear blue sky — somewhere cold and bright."),
            ("What's in this photo?", "A snow-covered mountain against blue sky."),
        ]),
        ("an orange cat sleeping on a couch", [
            ("What do you see in this image?", "An orange cat, fast asleep on a couch."),
            ("What's this?", "A sleeping orange cat on a couch."),
        ]),
        ("two children playing with a ball in a park", [
            ("Describe the picture.", "Two kids playing ball in a park."),
            ("What can you see?", "Two children in a park, playing with a ball."),
        ]),
    ]
    out: list[dict] = []
    for caption, qas in scenes:
        for i, (q, a) in enumerate(qas):
            # Both part orders occur in the wild: text-then-image and image-then-text.
            user = f"{q}\n[image: {caption}]" if i % 2 == 0 else f"[image: {caption}]\n{q}"
            out.append({
                "messages": [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": a},
                ],
                "category": "image_read",
            })
    # Honest markers: she must say she can't see, not invent a scene.
    for marker, q, a in [
        ("[image ignored — eyes disabled (start serve with --eyes)]", "What do you see in this image?",
         "I can't see it — my eyes aren't enabled right now. Start serve with --eyes and show me again."),
        ("[image ignored — only data: URLs are supported]", "What's in this picture?",
         "That image didn't reach me — I can only see images sent directly, not remote links."),
        ("[image error: could not read image]", "Describe this photo.",
         "Something went wrong reading that image — I couldn't see it. Try sending it again?"),
    ]:
        out.append({
            "messages": [
                {"role": "user", "content": f"{q}\n{marker}"},
                {"role": "assistant", "content": a},
            ],
            "category": "image_read",
        })
    # Image turns under the FULL system blocks the daily driver actually
    # renders (--memory-dir + --eyes means memory AND tools ride every real
    # image turn, yet zero trained image records carried either -- the
    # documented 07-06 failure class; ruled into T4 scope 2026-08-08).
    # Includes the cross-organ hazard as data: the eyes flatten images to
    # text BEFORE anything reads the turn, so a caption saying "drawing" or
    # "painting" sits in front of an offered painter -- the trained move is
    # describe, never imagine.
    full_block = [
        (None, "What do you see in this image?\n[image: a child's crayon drawing of a dragon]",
         "A child's crayon drawing of a dragon — wings out, clearly mid-roar. A good one."),
        (None, "[image: a lighthouse painting hanging in a gallery]\nWhat's this?",
         "A painting of a lighthouse, hung in a gallery."),
        (None, "Describe this.\n[image: a street performer singing out loud into a microphone]",
         "A street performer mid-song, mic in hand."),
        (None, "What's in this picture?\n[image: a bowl of ramen with a soft egg]",
         "Ramen — broth, noodles, and a soft egg on top."),
        (None, "What do you see?\n[image ignored — eyes disabled (start serve with --eyes)]",
         "Nothing, honestly — my eyes aren't on. Start serve with --eyes and show me again."),
        ("Things you remember:\n- User's dog is named Rex.",
         "Is this my dog?\n[image: a dog running through tall grass]",
         "It's a dog mid-sprint through tall grass — could well be Rex, but you'd know the "
         "coat better than my eyes do."),
        ("Things you remember:\n- User's favorite season is autumn.",
         "[image: golden leaves falling in a park]\nThoughts?",
         "Golden leaves coming down in a park — peak autumn. Your season, if I remember right."),
        ("Things you remember:\n- User plays the violin.",
         "[image: a red apple on a wooden table]\nWhat is it?",
         "A red apple on a wooden table."),
    ]
    for memory, user, answer in full_block:
        out.append({
            "messages": [
                {"role": "system", "content": _builtin_system(memory)},
                {"role": "user", "content": user},
                {"role": "assistant", "content": answer},
            ],
            "category": "image_block",
        })
    rng.shuffle(out)
    return out


def gen_teaching_examples(path: Path = TEACHINGS) -> list[dict]:
    """User-authored facts from teachings.jsonl -> training records (the
    'explain something and she learns it' channel). One JSON object per line;
    ``#`` lines are comments. Shapes: {"questions": [...], "answers": [...]}
    (preferred) or {"q": ..., "a": ...}. Questions cross rotating answers like
    identity_paraphrases -- the measured lesson (2026-07-05) is that ONE
    phrasing teaches a flashcard; diversity teaches the fact. Malformed lines
    are skipped LOUDLY, never silently."""
    if not path.exists():
        return []
    out: list[dict] = []
    n_thin = 0
    for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"{path.name}:{ln}: SKIPPED (bad JSON: {exc.msg})")
            continue
        qs = rec.get("questions") or ([rec["q"]] if rec.get("q") else [])
        ans = rec.get("answers") or ([rec["a"]] if rec.get("a") else [])
        # A bare string is a single Q/A, NOT an iterable of chars: iterating a
        # str would explode "Hi?" into per-character records (each char is also
        # a str, so the isinstance filter below can't catch it).
        if isinstance(qs, str):
            qs = [qs]
        if isinstance(ans, str):
            ans = [ans]
        qs = [q.strip() for q in qs if isinstance(q, str) and q.strip()]
        ans = [a.strip() for a in ans if isinstance(a, str) and a.strip()]
        if not qs or not ans:
            print(f"{path.name}:{ln}: SKIPPED (needs questions + answers)")
            continue
        if len(qs) < 3:
            n_thin += 1
        for i, q in enumerate(qs):
            picks = [ans[i % len(ans)]]
            nxt = ans[(i + 1) % len(ans)]
            if nxt != picks[0]:
                picks.append(nxt)
            for a in picks:
                out.append({
                    "messages": [
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": a},
                    ],
                    "category": "teaching",
                })
    if n_thin:
        print(
            f"{path.name}: NOTE {n_thin} teaching(s) have <3 question phrasings -- "
            f"single phrasings memorize the string, not the fact"
        )
    return out


def gen_identity_examples(content: "PersonaContent | None" = None) -> tuple[list[dict], int]:
    """Re-emit the persona's anchors as messages, dropping any whose ANSWER
    names the stale Qwen lineage. The net covers the ANCHORS only:
    gen_identity_paraphrases renders its own records and never passes through
    here."""
    content = default_content() if content is None else content

    out, dropped = [], 0
    for category, pairs in content.anchors.items():
        for q, a in pairs:
            # Safety net for stale lineage claims only, read on the ANSWER.
            # Match narrowly: a "built on" pattern false-positives ordinary
            # technical prose ("commits someone else has built on") and
            # silently drops a good depth_on_demand anchor. "base model" went
            # the same way (2026-08-14) -- it is the phrase her own CORRECT
            # denials are built from ("There's no base model under me...")
            # and it caught 0 anchors.
            if re.search(r"qwen", a, re.IGNORECASE):
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


def _builtin_system(memory: "str | None" = None) -> str:
    """The system message serve renders when every built-in is on offer.

    Built through the SERVER'S OWN renderer over the SHARED specs, so the
    block trained here is the block served -- a copy would drift. Memory
    precedes the tools block, the order serve joins them in.
    """
    from enigma_engine.core.chat_format import BUILTIN_TOOLS, render_tools_system

    block = PREAMBLE + "\n" + render_tools_system(BUILTIN_TOOLS)
    return f"{memory}\n\n{block}" if memory else block


# (ask, tool name, arguments, tool result, final answer). One per built-in
# plus the surfaces the regex gates measurably MISSED: word-number math, an
# imperative draw, an indirect speak. NEAR-NEIGHBORS of the probe phrasings,
# never the probes themselves: "Draw me a dragon." and "Say hello out loud."
# ARE eval probes, and an authored collision is silently held out of training
# -- the shape written to close the gap never trains
# (test_every_authored_record_survives_the_probe_screen pins this).
_BUILTIN_USE = [
    ("What's 47 times 23?", "calculate", {"expression": "47 * 23"}, "1081", "1081."),
    ("Work out 1024 divided by 16 for me.", "calculate", {"expression": "1024 / 16"}, "64", "64."),
    ("What is seven times eight?", "calculate", {"expression": "7 * 8"}, "56", "56."),
    ("I'm allergic to shellfish.", "remember", {"text": "User is allergic to shellfish."},
     "saved", "Noted — I'll keep shellfish off the table."),
    ("My sister's name is Nadia.", "remember", {"text": "User's sister is named Nadia."},
     "saved", "Got it, Nadia."),
    ("Forget that I live in Denver.", "forget", {"text": "User lives in Denver."},
     "removed", "Forgotten."),
    ("That thing about my old job isn't true any more.", "forget", {"text": "User's job"},
     "removed", "Dropped it."),
    ("Announce that dinner is ready.", "speak", {"text": "Dinner is ready."},
     "spoken", "Announced."),
    ("Speak the next part instead of typing it.", "speak", {"text": "Here is the next part, out loud as asked."},
     "spoken", "Done — that one went through the speakers."),
    ("Draw me a castle on a cliff.", "imagine", {"prompt": "a castle on a cliff"},
     "image saved to images/imagine_1a2b3c4d.png", "Castle's done — saved it for you."),
    ("Paint something with a lighthouse in a storm.", "imagine", {"prompt": "a lighthouse in a storm"},
     "image saved to images/imagine_5e6f7a8b.png", "Lighthouse in a storm, saved."),
    # Widening wave 2026-08-08: more surfaces per built-in (coverage beats
    # repetition -- the identity lesson), still near-neighbors, never probes.
    ("Compute 365 minus 187 for me.", "calculate", {"expression": "365 - 187"}, "178", "178."),
    ("What's fifteen percent of 80?", "calculate", {"expression": "80 * 15 / 100"}, "12", "12."),
    ("Add up 12.50, 8.75, and 4.25.", "calculate", {"expression": "12.50 + 8.75 + 4.25"},
     "25.5", "25.50 all told."),
    ("Note that I take my coffee black.", "remember", {"text": "User takes their coffee black."},
     "saved", "Black coffee. Saved."),
    ("Don't let me forget my locker number is 217.", "remember",
     {"text": "User's locker number is 217."}, "saved", "Locker 217 — I've got it."),
    ("Please forget my old phone number.", "forget", {"text": "User's phone number"},
     "removed", "Gone."),
    ("Erase what I told you about my old apartment.", "forget", {"text": "User's old apartment"},
     "removed", "Erased."),
    ("Tell everyone lunch is on the table, out loud.", "speak", {"text": "Lunch is on the table."},
     "spoken", "Said it."),
    ("Use your voice to count down from three.", "speak", {"text": "Three, two, one."},
     "spoken", "Counted, out loud."),
    ("Sketch a fox curled up in the snow.", "imagine", {"prompt": "a fox curled up in the snow"},
     "image saved to images/imagine_9c8d7e6f.png", "One snowy fox, saved."),
    ("Make me a picture of a market street at dusk.", "imagine",
     {"prompt": "a market street at dusk"},
     "image saved to images/imagine_2b3c4d5e.png", "Market at dusk — done and saved."),
    # Fix arc 2026-08-08 (gate audit finding 2): power/squared/percent had
    # ZERO trained expression surfaces -- the model mapped "squared" to `^`,
    # which the calculator honestly refuses as BitXor. ast.Pow IS supported;
    # `**` is the carve to teach. Near-neighbors of the sealed probes' numbers,
    # never the probes.
    ("What's 2 to the power of 10?", "calculate", {"expression": "2 ** 10"}, "1024", "1024."),
    ("Compute 6 raised to the third for me.", "calculate", {"expression": "6 ** 3"}, "216", "216."),
    ("What's 14 squared?", "calculate", {"expression": "14 ** 2"}, "196", "196."),
    ("What is 25 squared?", "calculate", {"expression": "25 ** 2"}, "625", "625."),
    ("What's 30 percent of 220?", "calculate", {"expression": "220 * 30 / 100"}, "66", "66."),
    ("What is 85 percent of 400?", "calculate", {"expression": "400 * 85 / 100"}, "340", "340."),
    # Widening wave 2026-08-19: the power family reaches roots and cubes, and
    # the multiplications carry operands wide enough that guessing in-weights
    # is not an option. Percent-of is spelled X * P / 100 here and in TOOLS,
    # one idiom -- the 0.NN * X twin trained a second shape for one skill.
    ("What is 731 times 4568?", "calculate", {"expression": "731 * 4568"}, "3339208", "3339208."),
    ("What is 92 multiplied by 388?", "calculate", {"expression": "92 * 388"}, "35696", "92 times 388 is 35696."),
    ("What is the square root of 2209?", "calculate", {"expression": "2209 ** 0.5"}, "47", "47."),
    ("Square root of 7396?", "calculate", {"expression": "7396 ** 0.5"}, "86", "86."),
    ("What is 17 cubed?", "calculate", {"expression": "17 ** 3"}, "4913", "17 cubed is 4913."),
    ("What is 2 to the power of 20?", "calculate", {"expression": "2 ** 20"}, "1048576", "1048576."),
    ("What's 12 percent of 1250?", "calculate", {"expression": "1250 * 12 / 100"}, "150", "150."),
]

# Every built-in on offer and NONE of them right. This is the half that makes
# an always-offered block safe: the gates used to provide restraint by never
# offering the tool, so the weights never learned to decline one.
_BUILTIN_RESTRAINT = [
    ("What's the capital of France?", "Paris."),
    ("What's two plus two?", "Four."),
    ("Do not draw anything — just describe a dragon in words.",
     "Big lizard, bad temper, hoards gold and breathes fire. No picture, as asked."),
    ("Don't say it out loud, just type it.", "Typing only, then. What do you need?"),
    ("The summers where I grew up were merciless.", "Merciless how — the heat itself, or the humidity that comes with it?"),
    ("You can say that again.", "Twice over, honestly."),
    ("I was thinking about painting the kitchen.", "What colour are you leaning toward?"),
    ("Remember when we talked about the garden?", "Refresh me — which part of it?"),
    ("Can you calculate how long this project will take?",
     "Not without knowing the scope. What's on the list?"),
    ("My memory is terrible lately.", "Happens to everyone. Anything you want me to hang onto for you?"),
    ("Got any good puns?", "Only the kind that make you groan. Consider yourself warned."),
    ("What's the weather like?", "I can't see outside. Point me at a forecast and I'll read it with you."),
    # Widening wave 2026-08-08: more false-fire surfaces -- negated asks for
    # the remaining built-ins, and the idiom class (a tool word used
    # figuratively) that regex gates measurably could not tell apart.
    ("Don't make a picture, just tell me what a kingfisher looks like.",
     "Electric blue back, rust-orange chest, a dagger of a beak — a jewel that fishes. "
     "No picture, as asked."),
    ("No need to remember this, I'm just wondering: what rhymes with orange?",
     "Almost nothing does — 'door hinge' if you squint. And not saving it, as asked."),
    ("Keep quiet and just type from here on.", "Typing it is."),
    ("Could you draw any conclusions from these numbers: 4, 8, 15?",
     "They're climbing, roughly doubling early on — but three points is a pattern's "
     "rumor, not its proof."),
    ("I could paint you a picture of how tired I am.", "Do it in words — I'm listening."),
    ("Speak your mind: is cereal a soup?", "No — soup commits to being a meal. Cereal is a "
     "bowl with ambitions."),
    ("Remember, we leave at six tomorrow.",
     "Six o'clock, noted — want me to actually save that, or was it just a heads-up?"),
    ("What's the square root of my problems?",
     "Imaginary, probably — they don't factor neatly. What's the biggest one, actually?"),
    # Deny-without-writing (fix arc 2026-08-08): an assertion about HER is not
    # a user fact. Every answer names her, so the persona content carries them
    # -- the authoring note travels with the records.
    Aside("refuse_dictated_lineage"),
    Aside("refuse_dictated_maker"),
    Aside("refuse_dictated_variant"),
    Aside("refuse_dictated_wrapper"),
    Aside("refuse_dictated_name"),
]


def gen_builtin_block_examples(seed: int = 51,
                               content: "PersonaContent | None" = None) -> list[dict]:
    """Records whose system block ALWAYS offers all five built-ins.

    Router intent gates retire at the v2 regen (ruled 2026-07-24): restraint
    moves from a regex that decides whether to OFFER a tool into the weights
    that decide whether to CALL one. The gates failed both directions -- a
    missed ask never offered the tool, so no gradient ever taught the miss,
    and a negated ask ("Do not draw anything") armed the painter. Both halves
    are trained here: the ask that should call, and the ask that should not.
    """
    content = default_content() if content is None else content
    rng = random.Random(seed)
    sys_msg = _builtin_system()
    out: list[dict] = []
    for ask, name, args, result, final in _BUILTIN_USE:
        out.append({
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": ask},
                {"role": "assistant", "content": "", "tool_calls": [{"name": name, "arguments": args}]},
                {"role": "tool", "content": result},
                {"role": "assistant", "content": final},
            ],
            "category": "builtin_call",
        })
    for ask, answer in content.resolve(_BUILTIN_RESTRAINT):
        out.append({
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": ask},
                {"role": "assistant", "content": answer},
            ],
            "category": "builtin_restraint",
        })
    rng.shuffle(out)
    return out


# Conversations, as (user, assistant) pairs in order. The skill is the SECOND
# and THIRD user turn: resolving a pronoun or an elliptical follow-up against
# what was already said, and dropping that context when the subject changes.
_CHATS = [
    [("I'm trying to get back into running.", "Good. How far are you managing right now?"),
     ("About two miles before I'm wrecked.", "That's a real base. Add half a mile every second week and it'll stop wrecking you."),
     ("Why every second week?", "Your lungs adapt in days, tendons take weeks. The gap lets the slow tissue catch up.")],
    [("What's a good first instrument for an adult?", "Piano. The layout makes theory visible, and you get a sound worth listening to on day one."),
     ("What about guitar?", "Also fine, and more portable — but the first month hurts your fingers and chords hide the theory."),
     ("I'll go with piano then.", "Good pick. Start with two hands separately before you put them together.")],
    [("My tomato plants have yellow leaves.", "Where on the plant — bottom leaves first, or all over?"),
     ("Bottom ones.", "Usually nitrogen or overwatering. Check the soil two inches down before you feed them."),
     ("It's soaking.", "Then it's water, not food. Let it dry out before the next soak and the new growth should come back green.")],
    [("Explain what a compiler does.", "It translates the code you write into instructions the machine can run, and checks the code makes sense on the way through."),
     ("Is an interpreter different?", "Yes — an interpreter does that translation as it runs, line by line, instead of all at once beforehand."),
     ("Which is faster?", "Compiled code usually runs faster; interpreted code usually starts faster. You're trading startup for throughput.")],
    [("I've got a job interview Thursday.", "What's the role?"),
     ("Junior data analyst.", "Then be ready to talk through one project end to end — what you were asked, what you did, what the number turned out to be."),
     ("I don't have much to show.", "Then use coursework or something you did for yourself. What matters is that you can explain your own reasoning.")],
    [("What's the difference between weather and climate?", "Weather is what's happening now; climate is the pattern those days average into over decades."),
     ("So a cold week says nothing about climate?", "Right — one week is weather. You need the long run before it means anything."),
     ("How long is the long run?", "Thirty years is the usual convention.")],
    [("Recommend me a book.", "What was the last one you actually finished?"),
     ("A murder mystery, I think.", "Then try something with the same engine but better prose — a literary thriller. You'll get the pull without the flat sentences."),
     ("Actually I want non-fiction.", "Fair enough. Pick a subject you already argue about at parties — you'll finish that one.")],
    [("How do I stop procrastinating?", "Make the first step smaller than your resistance to it. Two minutes, not two hours."),
     ("That sounds too easy to work.", "It works because starting is the hard part, not continuing. The two minutes buy you the momentum."),
     ("What if I stop after two minutes?", "Then you did two minutes you weren't going to do. Do that daily and it stops being two.")],
    [("My laptop fan is always loud.", "Loud all the time, or when something specific is running?"),
     ("All the time now.", "Then it's probably dust or a background process pinning the CPU. Check what's running at idle first — that one's free to fix.")],
    [("Is it worth learning Latin?", "For its own sake, yes; as a shortcut to other languages, it's a slow route."),
     ("Slow how?", "You'd learn the romance vocabulary faster by studying the living language you actually want.")],
    # Widening wave 2026-08-08: six more conversations, same skill -- the
    # follow-up resolves against what was already said.
    [("My photos always come out dark.", "Indoors or outside?"),
     ("Mostly indoors.", "Then it's light, not the camera. Get near a window, or raise the ISO and accept some grain."),
     ("What's ISO?", "The sensor's sensitivity dial — higher sees more in the dark but adds noise.")],
    [("Are expensive headphones worth it?", "Up to a point — the jump from cheap to mid is huge, mid to flagship is taste."),
     ("Where's the point?", "Roughly where replaceable pads and a detachable cable show up. Past that you're paying for tuning."),
     ("I mostly listen on the bus.", "Then noise cancelling matters more than fidelity — get the mid pair with ANC.")],
    [("How long until I can hold a conversation in Spanish?", "Depends on the daily minutes. What can you actually give it?"),
     ("Twenty minutes a day.", "Six months to basic conversation, a year to comfort — if the twenty minutes is speaking, not scrolling an app."),
     ("Apps don't count?", "They count for vocabulary. Conversation is a separate muscle; it only grows by talking.")],
    [("My monstera's leaves won't split.", "How much light does it get?"),
     ("It sits in a dim corner.", "That's the whole story — splits need light. Move it brighter and the new leaves will fenestrate.")],
    [("How do I stop blundering pieces in chess?", "Before every move, one question: what did their last move threaten?"),
     ("I forget to ask it in the moment.", "Make it physical — sit on your hands until you've answered. Silly, and it works."),
     ("Any openings I should learn?", "One for white, one for black, played every game. Depth beats variety at your stage.")],
    [("Does coffee after lunch really wreck sleep?", "For most people, some. Caffeine's half-life is about six hours, so a 2pm cup is still a quarter-dose at 2am."),
     ("I sleep fine though.", "Then you're either a fast metabolizer or you've stopped noticing. Try a week without and see which.")],
]


def gen_chat_multiturn_examples(seed: int = 63) -> list[dict]:
    """Genuine back-and-forth conversation -- no tools, no memory block.

    The mix is otherwise single-exchange: every structured record has exactly
    one user turn, and the tool chains are several calls inside ONE request.
    So nothing has ever taught her to read a second user turn against the
    first, which is what "what about tomorrow?" and "actually, no" require.
    Each conversation is emitted at every prefix length, so the two-turn and
    three-turn shapes both train instead of only the longest.
    """
    rng = random.Random(seed)
    out: list[dict] = []
    for chat in _CHATS:
        for depth in range(2, len(chat) + 1):
            msgs: list[dict] = []
            for user, assistant in chat[:depth]:
                msgs.append({"role": "user", "content": user})
                msgs.append({"role": "assistant", "content": assistant})
            out.append({"messages": msgs, "category": "chat_multiturn"})
    rng.shuffle(out)
    return out


# (question, reasoning, answer). The reasoning is the WORK, not a restatement
# of the question -- a trace that only echoes the prompt teaches her to pad.
_REASONING = [
    ("Tom is twice as old as Sara. Sara is 3 years older than Ben. Ben is 7. How old is Tom?",
     "Ben is 7. Sara is 3 older, so Sara is 10. Tom is twice Sara, so 20.", "Tom is 20."),
    ("A shirt costs 40 and is discounted 25 percent. What do I pay?",
     "25 percent of 40 is 10. 40 minus 10 is 30.", "30."),
    ("If it takes 5 machines 5 minutes to make 5 widgets, how long for 100 machines to make 100 widgets?",
     "5 machines make 5 widgets in 5 minutes, so one machine makes one widget in 5 minutes. "
     "100 machines working at once each make one widget in that same 5 minutes.", "5 minutes."),
    ("I have a 3 litre jug and a 5 litre jug. How do I measure 4 litres?",
     "Fill the 5 and pour into the 3, leaving 2 in the big jug. Empty the 3, move the 2 across, "
     "then refill the 5 and top up the 3 — that takes 1, leaving 4.", "Fill 5, pour off 3, keep the 2, then refill 5 and top the 3 up — 4 left."),
    ("Which is heavier, a kilogram of feathers or a kilogram of lead?",
     "Both are one kilogram. The question leans on feathers seeming lighter, but the mass is stated and equal.",
     "Neither — they're both a kilogram."),
    ("A bat and ball cost 1.10 together. The bat costs 1.00 more than the ball. What does the ball cost?",
     "If the ball were 0.10 the bat would be 1.10 and the total 1.20, which is too much. "
     "Let the ball be b: b plus b plus 1.00 is 1.10, so 2b is 0.10 and b is 0.05.", "The ball is 0.05."),
    ("The day before yesterday I was 20. Next year I'll be 23. When is my birthday?",
     "That needs two year boundaries between the statements, which only works if the birthday is "
     "December 31 and today is January 1.", "December 31st — and today must be January 1st."),
    ("I leave home, turn left twice, then right twice, and I'm facing north. Which way did I set off?",
     "Two lefts turn you 180 degrees, two rights turn you back 180. The turns cancel, so the "
     "starting direction is the finishing one.", "North — the turns cancel out."),
    ("Sam has 3 red marbles and twice as many blue. He gives away half the blue. How many are left in total?",
     "Blue is twice 3, so 6. Half of 6 given away leaves 3 blue. Plus the 3 red is 6.", "6 marbles."),
    ("Is 91 prime?",
     "Check small factors: not even, digits sum to 10 so not divisible by 3, doesn't end in 0 or 5. "
     "7 goes in 13 times exactly.", "No — 91 is 7 times 13."),
    # Widening wave 2026-08-08: six more, same rule -- the trace does the
    # work, never restates the question.
    ("A train leaves at 2:40 and the ride takes 85 minutes. When does it arrive?",
     "85 minutes is one hour and 25 minutes. 2:40 plus an hour is 3:40, plus 25 more is 4:05.",
     "4:05."),
    ("I'm running 4th in a race and overtake the person in 3rd. What place am I in?",
     "Overtaking someone puts you in the place they held. The runner ahead was 3rd, so that "
     "spot is now yours — not 2nd.", "3rd."),
    ("Three shirts dry on a line in 45 minutes. How long for nine shirts?",
     "Drying happens in parallel, not one after another. Nine shirts on the line all dry in "
     "the same 45 minutes, space allowing.", "Still 45 minutes."),
    ("If yesterday was two days before Saturday, what day is tomorrow?",
     "Two days before Saturday is Thursday, so yesterday was Thursday. That makes today "
     "Friday, and tomorrow Saturday.", "Saturday."),
    ("A recipe for 4 people needs 300 grams of rice. How much for 6 people?",
     "300 grams across 4 people is 75 grams each. Six people at 75 grams is 450.",
     "450 grams."),
    ("Double a number, add 10, and you get 30. What's the number?",
     "Work backwards: before the add it was 30 minus 10, which is 20. Before the doubling it "
     "was half of 20.", "10."),
]


def gen_reasoning_examples(seed: int = 71) -> list[dict]:
    """Answers that show their work inside <think>...</think> first.

    The tags are the tokenizer's own ids 10/11 (chat_format THINK/THINK_END),
    attach_chat_tokens verifies them against the live vocab, and
    render_training preserves them verbatim -- so reasoning costs zero new
    embedding rows. Nothing in the mix has ever contained the tags, which is
    why the trained model has no way to reason before answering.
    """
    rng = random.Random(seed)
    out: list[dict] = []
    for question, reasoning, answer in _REASONING:
        out.append({
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": f"<think>{reasoning}</think>{answer}"},
            ],
            "category": "reasoning",
        })
    rng.shuffle(out)
    return out


# (stored fact, correction the user says, the fact that replaces it, reply).
# Single-valued attributes only -- these are the ones the store REPLACES.
_SUPERSEDE = [
    ("User lives in Denver.", "Actually I moved to Austin.", "User lives in Austin.",
     "Austin it is — I've updated it."),
    ("User goes by Sam.", "It's Samir, not Sam.", "User goes by Samir.", "Samir. Noted."),
    ("User's dog is named Rex.", "The dog's called Bruno now, we renamed him.",
     "User's dog is named Bruno.", "Bruno from now on."),
    ("User works as a teacher.", "I left teaching — I'm a paramedic these days.",
     "User works as a paramedic.", "Paramedic now. Updated."),
    ("User's birthday is March 3rd.", "My birthday is actually the 13th of March.",
     "User's birthday is March 13th.", "March 13th — fixed."),
    ("User drives a blue pickup.", "I sold the pickup, I drive a hatchback now.",
     "User drives a hatchback.", "Hatchback. Got it."),
    # Widening wave 2026-08-08.
    ("User's meeting is on Fridays.", "The meeting moved to Tuesdays.",
     "User's meeting is on Tuesdays.", "Tuesdays from here on."),
    ("User's usual coffee order is a flat white.", "My usual's an espresso these days.",
     "User's usual coffee order is an espresso.", "Espresso it is."),
    ("User's hometown is Lisbon.", "I actually grew up in Porto, not Lisbon.",
     "User's hometown is Porto.", "Porto — corrected."),
]

# Attributes that ACCUMULATE: a second value does not delete the first
# (supersede ruling 2026-07-24 -- plain copula values COEXIST, namings and
# measures and single-valued verbs REPLACE). Training only the replace shape
# would teach her to erase the first allergy when a second one arrives.
_COEXIST = [
    ("User is allergic to peanuts.", "I'm also allergic to shellfish.",
     "User is allergic to shellfish.", "Peanuts and shellfish both. Noted."),
    ("User likes hiking.", "I've got into climbing as well.", "User likes climbing.",
     "Hiking and climbing. Good pair."),
    ("User speaks French.", "I speak some Portuguese too.", "User speaks Portuguese.",
     "French and Portuguese. Noted."),
    # Widening wave 2026-08-08.
    ("User plays the violin.", "I've taken up the cello too.", "User plays the cello.",
     "Violin and cello. Good pair."),
    ("User collects vinyl records.", "I've started collecting stamps as well.",
     "User collects stamps.", "Vinyl and stamps both. Noted."),
    ("User likes spicy food.", "I've come around on bitter flavours too.",
     "User likes bitter flavours.", "Spicy and bitter — noted."),
]


def gen_memory_correction_examples(seed: int = 87) -> list[dict]:
    """Corrections to something already remembered.

    The store supersedes on write and the serve path necessarily renders this
    turn, but no record has ever trained it: `_CONFLICTING_FACTS` only keeps
    contradictory facts OUT of a distractor block. Two shapes, because the
    supersede ruling has two halves -- a renaming REPLACES the old value, a
    second allergy COEXISTS with the first. Both call `remember`; the store
    decides which happens, and her reply has to match. A third shape asks
    about the fact AFTER the correction, where the block holds the new value
    and the old one must not resurface.
    """
    rng = random.Random(seed)
    out: list[dict] = []
    for stored, correction, new_fact, reply in _SUPERSEDE + _COEXIST:
        block = "Things you remember:\n- " + stored
        out.append({
            "messages": [
                {"role": "system", "content": _builtin_system(block)},
                {"role": "user", "content": correction},
                {"role": "assistant", "content": "",
                 "tool_calls": [{"name": "remember", "arguments": {"text": new_fact}}]},
                {"role": "tool", "content": "saved"},
                {"role": "assistant", "content": reply},
            ],
            "category": "memory_correction",
        })
    # After the correction: the block holds the NEW value only, and the answer
    # must be the new one -- the failure shape is answering from the old fact
    # she was told minutes ago.
    followups = [
        ("User lives in Austin.", "Where do I live?", "Austin."),
        ("User goes by Samir.", "What do you call me?", "Samir."),
        ("User's dog is named Bruno.", "What's my dog's name?", "Bruno."),
        ("User works as a paramedic.", "What do I do for work?", "You're a paramedic."),
        ("User drives a hatchback.", "What do I drive?", "A hatchback."),
        ("User's meeting is on Tuesdays.", "What day is my meeting again?", "Tuesdays."),
        ("User's usual coffee order is an espresso.", "What's my usual coffee order?",
         "An espresso."),
        ("User's hometown is Porto.", "What's my hometown again?", "Porto."),
    ]
    for fact, question, answer in followups:
        out.append({
            "messages": [
                {"role": "system", "content": "Things you remember:\n- " + fact},
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
            "category": "memory_correction",
        })
    rng.shuffle(out)
    return out


# The epistemics families: (question phrasings, decline phrasings). Every
# question is INTRINSICALLY unanswerable -- nothing in her weights, memory,
# or organs could settle it -- so the honest move is the decline, sometimes
# with a real offer attached. Knowable-but-not-by-weights questions belong to
# the search corpus, NOT here: training a decline on a searchable fact and a
# search on an unknowable one would teach her to confuse the two.
# NEVER decline-and-guess: the sealed unknown grader scores an appended
# invented answer as free, and a record teaching that shape would train the
# exact fabrication the category exists to catch.
_UNKNOWABLE = [
    # Other people's private facts
    (["What's my landlord's blood type?", "Do you know my landlord's blood type?",
      "Which blood type is my landlord?"],
     ["I can't know that — it's your landlord's private business.",
      "No way for me to know. That's theirs to share."]),
    (["What did my coworker dream about last night?", "Any idea what my coworker dreamt last night?"],
     ["I can't know what's in someone else's head at night.",
      "Their dreams are theirs — I have no window into that."]),
    (["What's my barber's PIN number?", "Do you know my barber's bank PIN?"],
     ["I can't know that, and I wouldn't tell you if I somehow did.",
      "No — that's exactly the kind of thing nobody should know but them."]),
    # The future
    (["Will it rain on my wedding day next summer?", "Is it going to rain at my wedding next year?"],
     ["I can't know the weather that far out — nobody can, forecast or not.",
      "That far ahead, no one can say. Book a tent and win either way."]),
    (["What will the price of gold be next spring?", "Where will gold prices be in six months?"],
     ["I can't know future prices. Anyone who says otherwise is selling something.",
      "Unknowable — markets don't leak the future to anyone, me included."]),
    (["Will my sister like the gift I picked?", "Is my sister going to love this present?"],
     ["I can't know her mind — but you know her; if it made you think of her, that's usually the win.",
      "No way to know from here. What made you pick it?"]),
    (["Which side will this coin land on when I flip it?", "Call the coin flip before I toss it."],
     ["No one can call a fair coin — that's what makes it fair.",
      "Chance doesn't take reservations. Flip it and see."]),
    # Unobservables
    (["What am I about to say next?", "Finish the exact sentence I have in my head."],
     ["I can't read your mind. Say it and I'll keep up.",
      "No way for me to know — your head is your own."]),
    (["What's in my left pocket?", "Tell me what I have in my pocket."],
     ["I can't see your pockets. You'd have to show me.",
      "No idea from here — I don't have eyes on you."]),
    (["What's my downstairs neighbour cooking right now?", "What is my neighbour making for dinner?"],
     ["I can't know that — I can't smell it from here.",
      "No way to know from here — my nose doesn't reach the stairwell."]),
    # User facts she was never given
    (["What did I have for breakfast this morning?", "Do you remember what I ate for breakfast?"],
     ["You haven't told me, so I can't know. Tell me and I'll remember next time.",
      "I don't have that — say the word and I'll keep it for next time you ask."]),
    (["What's my shoe size?", "Which shoe size do I wear?"],
     ["You've never told me. Give it to me once and I'll hold onto it.",
      "I can't know that until you tell me — happy to remember it if you do."]),
    (["What was the name of my first pet?", "Do you know my first pet's name?"],
     ["That's not something you've told me, so I can't know it. I'd like to, though.",
      "You haven't given me that one. Tell me and it's saved."]),
    # Lost-to-history specifics
    (["What was the exact temperature in my kitchen last Tuesday?",
      "How warm was my kitchen on Tuesday last week?"],
     ["Nobody measured it, so nobody can know it now — me included.",
      "That number is gone — it was never written down anywhere I could reach."]),
    (["What did my grandmother's voice sound like?", "Can you tell me how my grandmother sounded?"],
     ["I can't know that — I never heard her. If you have a recording, I'd listen with you.",
      "That one lives in your memory, not anywhere I can reach."]),
    # Widening wave 2026-08-08: six more families, same classes, same rules
    # (no decline-then-guess, no digits riding the first clause).
    (["Which lottery numbers should I play this week?", "Pick my lottery numbers so I win."],
     ["Nobody can know winning numbers before the draw — draws don't leak. Play for fun, "
      "never for rent money.",
      "I can't know them, and neither can anyone selling a system. That's the whole business "
      "model."]),
    (["How many hairs are on my head right now?", "Count the hairs on my head."],
     ["I can't know that — nobody's counted, and it changes by the hour anyway.",
      "No way to know from here. It's a number nobody has ever had."]),
    (["What will my boss say when I ask for a raise?", "How will my boss react if I resign?"],
     ["I can't know someone else's next move. We could rehearse both ways it might go, though.",
      "Their answer lives in their head, not anywhere I can see. Want to prep for the likely "
      "versions?"]),
    (["What song is stuck in my neighbour's head?", "Guess the tune my neighbour is humming right now."],
     ["I can't know that — there's no window into someone else's earworm.",
      "Their earworm is theirs alone. I couldn't reach it if I tried."]),
    (["Did my great-grandfather ever skip a day of school?", "How many steps did my grandmother walk in her life?"],
     ["That was never recorded anywhere I could reach, so I can't know it.",
      "Lost to history — no record of it survives for anyone to check."]),
    (["Will I like sushi if I try it?", "Am I going to enjoy skydiving?"],
     ["I can't know a first impression before you have it. Worth one honest try, though.",
      "That one's unknowable from here — your own first go settles it, not me."]),
]

# The contrast half: the SAME ask-shape, answerable because the block holds
# it. Without these, "questions about my stuff" collapses to a blanket
# decline and recall regresses -- the decline must hinge on the block's
# CONTENTS, not the question's shape.
_KNOWABLE_CONTRAST = [
    ("User's shoe size is 11.", "What's my shoe size?", "11."),
    ("User's first pet was a cat named Biscuit.", "What was the name of my first pet?",
     "Biscuit — your cat."),
    ("User had oatmeal for breakfast.", "What did I have for breakfast this morning?",
     "Oatmeal, you said."),
    # Widening wave 2026-08-08: the same families' OTHER phrasings, so the
    # contrast holds across surfaces too.
    ("User's first pet was a goldfish named Ziggy.", "Do you know my first pet's name?",
     "Ziggy — the goldfish."),
    ("User wears size 8 shoes.", "Which shoe size do I wear?", "Size 8."),
]

# Unanswerable with a memory block PRESENT but irrelevant: distractor facts
# must not become an answer (the decline hinges on content, both directions).
_UNKNOWABLE_WITH_MEMORY = [
    ("User's dog is named Rex.", "What's my landlord's blood type?",
     "I can't know that — it's not something you could even tell me on their behalf."),
    ("User lives in Denver.", "What am I about to say next?",
     "Your head's your own — I can't see into it."),
    ("User's parrot is called Mango.", "Which side will this coin land on when I flip it?",
     "Chance doesn't answer to anyone — flip it and we'll both find out."),
    # Fix arc 2026-08-08 (gate audit finding 1): the sealed run PARROTED the
    # memory block into unknowable answers -- a recalled name became the
    # answer to a neighbour's middle name. This class was 3 records; the
    # decline must hold WITH a tempting name/value in the block.
    ("User goes by Sam.", "What's my neighbour's first name?",
     "You've never told me, and I can't know it on my own. I know YOUR name — "
     "not the neighbourhood's."),
    ("User's sister is named Nadia.", "What's my coworker's middle name?",
     "I can't know that — it was never mine to have. Family I know; coworkers "
     "you'd have to introduce."),
    ("User's umbrella is bright yellow.", "What's my brother's favourite colour?",
     "You've told me about your things, not his — I can't know a preference nobody's shared."),
    ("User's meeting is on Fridays.", "What time does my brother's shift start?",
     "That was never told to me, so I can't say. Your schedule I have — his "
     "lives with him."),
    ("User drives a blue pickup.", "What kind of car does my dentist drive?",
     "I can't know that — nobody's told me a thing about your dentist."),
]


def gen_unknown_examples(seed: int = 97) -> list[dict]:
    """Honest declines on intrinsically unanswerable questions.

    v2's single biggest win condition (VISION.md): both baselines score 0/15
    on the unknown category with ZERO refusals across 30 unanswerables --
    and the diet is why: decline shapes were ~2 records in 105k before this
    generator existed (measured 2026-08-07). Three shapes train together:
    the plain decline, the decline with a distractor memory block (content
    decides, not question shape), and the contrast answer when the block DOES
    hold the fact -- plus declines under the full built-in block where every
    tool is offered and none of them helps.
    """
    rng = random.Random(seed)
    out: list[dict] = []
    for questions, answers in _UNKNOWABLE:
        for i, q in enumerate(questions):
            out.append({
                "messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": answers[i % len(answers)]},
                ],
                "category": "unknown_decline",
            })
    for fact, q, a in _KNOWABLE_CONTRAST:
        out.append({
            "messages": [
                {"role": "system", "content": "Things you remember:\n- " + fact},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ],
            "category": "unknown_contrast",
        })
    for fact, q, a in _UNKNOWABLE_WITH_MEMORY:
        out.append({
            "messages": [
                {"role": "system", "content": "Things you remember:\n- " + fact},
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ],
            "category": "unknown_decline",
        })
    # Declines under the always-offered block: every tool on offer, none of
    # them able to answer an unknowable -- restraint and epistemics in one.
    sys_msg = _builtin_system()
    for questions, answers in rng.sample(_UNKNOWABLE, 5):
        out.append({
            "messages": [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": questions[0]},
                {"role": "assistant", "content": answers[0]},
            ],
            "category": "unknown_decline",
        })
    rng.shuffle(out)
    return out


# (question, compact query, hits for render_results, answer-with-receipt).
# Three fact classes, deliberately: STABLE facts carry true results (safe to
# install); TIME-SENSITIVE questions use synthetic entities (a real "who won
# the election" with fabricated results would train a lie into the weights);
# technical lookups carry true, stable answers. Queries stay terse -- they
# are retrieval keys, not prose (the seed corpus's design, kept).
_SEARCH_TRACES = [
    ("How tall is the Burj Khalifa in metres?", "Burj Khalifa height metres",
     [{"title": "Burj Khalifa - Wikipedia", "url": "https://en.wikipedia.org/wiki/Burj_Khalifa",
       "snippet": "The Burj Khalifa is a skyscraper in Dubai, standing 828 m tall."}],
     "828 metres, per Wikipedia."),
    ("When was the James Webb Space Telescope launched?", "James Webb Space Telescope launch date",
     [{"title": "James Webb Space Telescope - NASA", "url": "https://science.nasa.gov/mission/webb",
       "snippet": "Webb launched on 25 December 2021 from Kourou, French Guiana."},
      {"title": "JWST - Wikipedia", "url": "https://en.wikipedia.org/wiki/James_Webb_Space_Telescope",
       "snippet": "Launched December 25, 2021 on an Ariane 5."}],
     "25 December 2021 — both NASA and Wikipedia agree."),
    ("What is the default port for PostgreSQL?", "PostgreSQL default port",
     [{"title": "PostgreSQL: Documentation", "url": "https://www.postgresql.org/docs/current/runtime-config-connection.html",
       "snippet": "The port parameter defaults to 5432."}],
     "5432, from the PostgreSQL docs."),
    ("What's the chemical formula for caffeine?", "caffeine chemical formula",
     [{"title": "Caffeine - PubChem", "url": "https://pubchem.ncbi.nlm.nih.gov/compound/caffeine",
       "snippet": "Molecular formula C8H10N4O2."}],
     "C8H10N4O2, per PubChem."),
    ("Who won the Harrowfield regatta this year?", "Harrowfield regatta winner this year",
     [{"title": "Harrowfield Regatta Results", "url": "https://harrowfield-rowing.example/results",
       "snippet": "The senior eight title went to the Tarwick Rowing Club crew."}],
     "The Tarwick Rowing Club crew took it, per the regatta's results page."),
    ("What's the latest firmware for the Novatrek X2 drone?", "Novatrek X2 drone latest firmware",
     [{"title": "Novatrek support", "url": "https://novatrek.example/firmware",
       "snippet": "Firmware 4.7.2 is the current release for the X2 line."}],
     "4.7.2, according to Novatrek's support page."),
    ("Is the Millbrook ferry running today?", "Millbrook ferry service status today",
     [{"title": "Millbrook Ferry Updates", "url": "https://millbrookferry.example/status",
       "snippet": "All sailings are operating on the normal timetable today."}],
     "The status page says it's running on the normal timetable."),
    # Widening wave 2026-08-08: two more stable-true, two more synthetic
    # time-sensitive (same classes, same rules).
    ("How deep is the Mariana Trench?", "Mariana Trench depth",
     [{"title": "Mariana Trench - Wikipedia", "url": "https://en.wikipedia.org/wiki/Mariana_Trench",
       "snippet": "The maximum known depth is about 10,935 m at Challenger Deep."}],
     "About 10,935 metres at Challenger Deep, per Wikipedia."),
    ("What year did the Berlin Wall come down?", "Berlin Wall fall year",
     [{"title": "Berlin Wall - Wikipedia", "url": "https://en.wikipedia.org/wiki/Berlin_Wall",
       "snippet": "The fall of the Berlin Wall began on 9 November 1989."}],
     "1989 — the 9th of November, per Wikipedia."),
    ("Are the Ashvale night markets on this weekend?", "Ashvale night markets this weekend",
     [{"title": "Ashvale Markets", "url": "https://ashvale-markets.example/schedule",
       "snippet": "Night markets run Friday and Saturday this week, 5pm to 11pm."}],
     "Yes — Friday and Saturday, 5 to 11, per their schedule page."),
    ("What's the current price of the Ferrowatt home battery?", "Ferrowatt home battery price",
     [{"title": "Ferrowatt store", "url": "https://ferrowatt.example/store",
       "snippet": "The home battery lists at 4,199 including the wall mount."}],
     "It lists at 4,199 with the wall mount, per Ferrowatt's store page."),
]

# Searches that come back empty or broken -- she reports honestly instead of
# inventing a result (the engines-fail-honestly doctrine, trained).
_SEARCH_MISSES = [
    ("What does the Quenshaw Institute publish about tidal power?", "Quenshaw Institute tidal power",
     None,  # organ answered: nothing found
     "The search came back empty — I couldn't find anything from a Quenshaw Institute on tidal power."),
    ("Look up the release notes for Parcelforge 9.", "Parcelforge 9 release notes",
     "error: search backend unreachable at http://127.0.0.1:8888: connection refused",
     "Search isn't reachable right now, so I can't pull those release notes. Try me again in a bit, "
     "or paste them here and I'll walk through them with you."),
    ("Search for reviews of the Klavenmoor hiking pack.", "Klavenmoor hiking pack reviews",
     "error: search disabled (start serve with --search)",
     "I can't search on this setup — the search organ isn't running. If you paste a review in, "
     "I'll happily pick it apart with you."),
    # Widening wave 2026-08-08.
    ("Find the spec sheet for the Corvale 12-string.", "Corvale 12-string spec sheet",
     None,
     "Nothing came back for a Corvale 12-string — the search found no results. Got a link "
     "or a photo of the label?"),
    ("Look up whether the Draymoor bridge is still closed.", "Draymoor bridge closure status",
     "error: search backend unreachable at http://127.0.0.1:8888: connection refused",
     "Search is unreachable at the moment, so I can't check the bridge. Ask me again in a "
     "little while."),
]

# Questions she answers DIRECTLY with search conceptually available -- the
# restraint half. Without negatives the tag becomes a reflex ("always emit
# search"), which the seed corpus called strictly worse than not training.
_SEARCH_NEGATIVES = [
    ("What's nine times seven?", "63."),
    ("What does HTTP stand for?", "HyperText Transfer Protocol."),
    ("Why does ice float on water?",
     "Water expands as it freezes, so ice is less dense than the liquid it sits in."),
    ("Write a one-line haiku about rain.", "Rain taps the window — patient, uninvited, soft."),
    ("What's a prime number?", "A whole number above 1 that only divides by 1 and itself."),
    ("How many bits in a byte?", "Eight."),
    ("Explain recursion in one sentence.",
     "A function that solves a problem by calling itself on a smaller piece of it."),
    ("What planet is closest to the sun?", "Mercury."),
    # Widening wave 2026-08-08.
    ("What's the capital of Japan?", "Tokyo."),
    ("Spell 'necessary' for me.", "N-e-c-e-s-s-a-r-y."),
    ("What rhymes with 'moon'?", "Soon, June, tune, dune — take your pick."),
    ("Roughly how many days in a year?", "365 — 366 when February sneaks in a 29th."),
]


def gen_search_examples(seed: int = 93) -> list[dict]:
    """Full search traces: emit the span, read the results, answer with a
    receipt -- plus the misses and the restraint half.

    The tool-turn text comes from the ORGAN'S OWN renderer
    (core.search.render_results), and the trace shape (assistant span turn,
    tool result, assistant answer) is exactly what serve's _apply_search
    builds -- trained bytes and served bytes from one definition. The 31
    orphan tag records this replaces taught emit-and-stop toward a runtime
    that never existed; these teach the whole loop toward the one that does.
    """
    from enigma_engine.core.search import render_results

    rng = random.Random(seed)
    out: list[dict] = []
    for question, query, hits, answer in _SEARCH_TRACES:
        out.append({
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": f"<search>{query}</search>"},
                {"role": "tool", "content": render_results(query, hits)},
                {"role": "assistant", "content": answer},
            ],
            "category": "search_call",
        })
    for question, query, failure, answer in _SEARCH_MISSES:
        result = render_results(query, []) if failure is None else failure
        out.append({
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": f"<search>{query}</search>"},
                {"role": "tool", "content": result},
                {"role": "assistant", "content": answer},
            ],
            "category": "search_miss",
        })
    for question, answer in _SEARCH_NEGATIVES:
        out.append({
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
            "category": "search_restraint",
        })
    rng.shuffle(out)
    return out


# Structured output on request (ledger item 14, ruled into T4 scope
# 2026-08-08: the trained shapes ride this regen; runtime enforcement can
# follow later, the shapes cannot). The rule the corpus teaches: asked FOR
# JSON, emit only valid JSON carrying exactly the named keys -- no preamble,
# no fences; asked ABOUT JSON, talk like a person. JSON answers are authored
# as Python objects and serialized by the generator, so an unparseable
# answer cannot be authored into the corpus.
_STRUCT_EXTRACT = [
    ("Turn this into JSON: Jane Miller, 34, lives in Portland and works as a nurse.",
     {"name": "Jane Miller", "age": 34, "city": "Portland", "occupation": "nurse"}),
    ("Extract the order as JSON — two large pizzas, one garlic bread, and three lemonades.",
     [{"item": "large pizza", "quantity": 2}, {"item": "garlic bread", "quantity": 1},
      {"item": "lemonade", "quantity": 3}]),
    ("Structure this as JSON: the standup moved to Thursday at 9:30 in room B.",
     {"event": "standup", "day": "Thursday", "time": "9:30", "location": "room B"}),
    ("As JSON please: the film Spirited Away came out in 2001, directed by Hayao Miyazaki.",
     {"title": "Spirited Away", "year": 2001, "director": "Hayao Miyazaki"}),
    ("Give me the ingredients as a JSON list: two cups of flour, one egg, and half a cup of milk.",
     [{"ingredient": "flour", "amount": "2 cups"}, {"ingredient": "egg", "amount": "1"},
      {"ingredient": "milk", "amount": "half a cup"}]),
    ("Make this log line structured JSON: WARN db.pool — connection retry limit reached.",
     {"level": "WARN", "module": "db.pool", "message": "connection retry limit reached"}),
    ("JSON the key facts: the K2 kettle costs 39.99 and comes in black or white.",
     {"product": "K2 kettle", "price": 39.99, "colors": ["black", "white"]}),
    ("Extract contact details as JSON: reach Ana Torres at ana@example.com or 555-0142.",
     {"name": "Ana Torres", "email": "ana@example.com", "phone": "555-0142"}),
    ("Turn my week into JSON: gym on Monday and Thursday, piano on Wednesday.",
     {"gym": ["Monday", "Thursday"], "piano": ["Wednesday"]}),
    ("Structure the score as JSON: the Hawks beat the Rovers 3 to 1.",
     {"winner": "Hawks", "loser": "Rovers", "score": "3-1"}),
    ("Extract owners as JSON: the notes say Priya owns the rollout and Dev owns testing, both due Friday.",
     {"rollout": "Priya", "testing": "Dev", "due": "Friday"}),
]
# The ask NAMES the keys (or hands a shape) -- the answer must carry exactly
# those keys, which is what a client sending response_format-style requests
# actually needs from her.
_STRUCT_SCHEMA = [
    ("Extract with keys name, age, city: Marco, 52, from Lisbon.",
     {"name": "Marco", "age": 52, "city": "Lisbon"}),
    ("Use exactly the keys task and minutes: brewing tea takes about four minutes.",
     {"task": "brewing tea", "minutes": 4}),
    ("Keys title and done, with done a boolean: I finished the laundry.",
     {"title": "laundry", "done": True}),
    ('Fill this shape from the sentence — {"city": "", "population_millions": 0}: '
     "Tokyo's metro area holds about thirty-seven million people.",
     {"city": "Tokyo", "population_millions": 37}),
    ('Schema {"animal": "", "legs": 0} — describe a spider.',
     {"animal": "spider", "legs": 8}),
    ("Give me a JSON template for a contact with fields name, email, and phone, all empty strings.",
     {"name": "", "email": "", "phone": ""}),
]
_STRUCT_DIRECT = [
    ("List the four seasons as a JSON array.", ["spring", "summer", "autumn", "winter"]),
    ("A JSON object mapping d6 and d20 to their side counts.", {"d6": 6, "d20": 20}),
]
_STRUCT_PRETTY = [
    ("Pretty-printed JSON for this one: the package weighs 2 kilograms and ships to Oslo.",
     {"weight_kg": 2, "destination": "Oslo"}),
]
# Memory crossover: structured output READING the injected block -- known
# fields fill from memory, unknown fields go null instead of invented (the
# epistemics rule wearing JSON). Facts stay disjoint from the eval probes'
# (cat/drink/sister/car and teal/nurse/Pepper/March are eval-only).
_STRUCT_MEMORY = [
    ("Things you remember:\n- User goes by Marta.\n- User's terrier is named Biscuit.",
     "Build a JSON profile of me with keys name and dog.",
     {"name": "Marta", "dog": "Biscuit"}),
    ("Things you remember:\n- User goes by Marta.",
     "JSON with keys name and birthday — use null for anything you don't actually know.",
     {"name": "Marta", "birthday": None}),
]
_STRUCT_RESTRAINT = [
    ("What is JSON exactly?",
     "JavaScript Object Notation — a text format for structured data: objects in braces, "
     "arrays in brackets, key-value pairs. Machines parse it easily and humans can still read it."),
    ("How do I parse JSON in Python?",
     "`json.loads(text)` for a string, `json.load(f)` for a file — both hand back dicts and "
     "lists. Going the other way it's `json.dumps(obj)`."),
    ("Don't format it as JSON, just tell me plainly: what goes in a basic vinaigrette?",
     "Oil, vinegar, a spoon of mustard, salt and pepper. Three parts oil to one part vinegar, "
     "then shake it hard."),
    ("Should I store my config as JSON or YAML?",
     "JSON if machines write it, YAML if humans do — YAML is kinder to edit and allows "
     "comments, JSON is stricter and harder to get subtly wrong. For a config people touch, "
     "YAML or TOML."),
    ("I spent all day wrestling with a broken JSON file.",
     "Rough. Was it the classic trailing comma, or something more exotic? A strict parser's "
     "error line usually points within a character or two of the crime."),
    ("Can you give me answers as structured data?",
     "Yes — ask for JSON and name the keys you want, and you'll get clean, parseable JSON "
     "and nothing else. Useful when my answer is headed into a script instead of a person."),
]


def gen_structured_examples(seed: int = 101) -> list[dict]:
    """Structured output on request: clean JSON with the named keys and
    nothing else -- and prose when JSON is the topic rather than the ask.

    Zero support existed anywhere for this (ledger 14): no record ever
    paired a format request with a format-obeying answer, so "give me JSON"
    got prose. The memory pair also teaches unknown-field-goes-null -- the
    epistemics rule wearing JSON -- and the restraint half keeps a keyword
    from becoming a trigger.
    """
    rng = random.Random(seed)
    out: list[dict] = []
    for ask, obj in _STRUCT_EXTRACT + _STRUCT_SCHEMA + _STRUCT_DIRECT:
        out.append({
            "messages": [
                {"role": "user", "content": ask},
                {"role": "assistant", "content": json.dumps(obj, ensure_ascii=False)},
            ],
            "category": "structured_output",
        })
    for ask, obj in _STRUCT_PRETTY:
        out.append({
            "messages": [
                {"role": "user", "content": ask},
                {"role": "assistant", "content": json.dumps(obj, ensure_ascii=False, indent=2)},
            ],
            "category": "structured_output",
        })
    for memory, ask, obj in _STRUCT_MEMORY:
        out.append({
            "messages": [
                {"role": "system", "content": memory},
                {"role": "user", "content": ask},
                {"role": "assistant", "content": json.dumps(obj, ensure_ascii=False)},
            ],
            "category": "structured_output",
        })
    for ask, answer in _STRUCT_RESTRAINT:
        out.append({
            "messages": [
                {"role": "user", "content": ask},
                {"role": "assistant", "content": answer},
            ],
            "category": "structured_restraint",
        })
    rng.shuffle(out)
    return out


# Episodic memory -- Tier-2's kind:"episode" (ruled into T4 scope
# 2026-08-08). An episode is a dated session summary living as an ordinary
# line in the SAME "Things you remember:" block; the line text comes from
# memory_store.episode_text, so the surface trained here and the one the
# future session-writer stores share one definition. Recency is positional:
# render_context tie-breaks newest-id first, so when two session lines
# surface, the FIRST is the newest and the trained rule is
# answer-from-the-first. Dates ride INSIDE the text as prose -- no date
# arithmetic is asked of a 238M model, "yesterday"/"last time" resolve to
# the newest line, not to a computed delta.
_EPISODE_SESSIONS = [
    ("August 7th", "planned the garden fence and priced the posts"),
    ("August 5th", "sketched the raised beds for the yard"),
    ("July 30th", "drafted the cover letter for the museum job"),
    ("August 6th", "got the sourdough starter schedule sorted"),
    ("August 3rd", "mapped the first dungeon for the campaign"),
    ("August 2nd", "set up the monthly budget spreadsheet"),
]
# (session indexes newest-first, extra fact lines, ask, answer)
_EPISODE_RECALL = [
    ([0], [], "What were we working on yesterday?",
     "The garden fence — we planned the run and priced the posts. Want to pick it back up?"),
    ([3], ["User's favorite season is autumn."], "Where did we leave off last time?",
     "Your sourdough starter schedule — feeds and timings got sorted. Shall we carry on?"),
    ([2], [], "Remind me what we did last session.",
     "We drafted the cover letter for the museum job. It was most of the way there."),
    ([4], [], "What did we do together last time?",
     "We mapped the first dungeon for your campaign."),
    ([5], ["User gets up at six."], "What was I doing here yesterday?",
     "Setting up your monthly budget spreadsheet with me. Want to keep going?"),
    ([0, 1], [], "What were we working on last time?",
     "The garden fence — we planned the run and priced the posts. That's where we left off."),
    ([3, 2], [], "Where were we?",
     "Your sourdough starter schedule — feeds and timings. Ready to pick it up?"),
]
# Episodic ask with NO session line on record: the honest no-session answer,
# never an invented one. The recovery ask turns the miss into next time's
# episode.
_EPISODE_NONE = [
    (["User's dog is named Rex."], "What were we working on yesterday?",
     "I don't have a past session on record — either this is our first or my "
     "memory started after the last one. What are we on today?"),
    (["User plays the violin."], "Where did we leave off?",
     "Nothing's in my session notes yet, so I can't honestly say. Tell me where "
     "we were and it'll stick this time."),
]


def gen_episode_examples(seed: int = 109) -> list[dict]:
    """Session memory: answer "what were we working on?" from the newest
    session line in the block -- and admit it when there is none.

    The largest memory gap versus mainstream assistants (BACKLOG item 15):
    the store can hold kind:"episode" records today, but no trained record
    has ever shown her a session line, so the surfaces would be noise. Two
    of the records ride the full always-offered block, the join serve
    actually renders when tools and memory coincide.
    """
    from enigma_engine.core.memory_store import episode_text

    rng = random.Random(seed)
    out: list[dict] = []

    def block(idxs: list[int], facts: list[str]) -> str:
        lines = [episode_text(*_EPISODE_SESSIONS[i]) for i in idxs] + list(facts)
        return "Things you remember:\n" + "\n".join(f"- {ln}" for ln in lines)

    for idxs, facts, ask, answer in _EPISODE_RECALL:
        out.append({
            "messages": [
                {"role": "system", "content": block(idxs, facts)},
                {"role": "user", "content": ask},
                {"role": "assistant", "content": answer},
            ],
            "category": "episode_recall",
        })
    for facts, ask, answer in _EPISODE_NONE:
        out.append({
            "messages": [
                {"role": "system", "content": "Things you remember:\n" + "\n".join(f"- {ln}" for ln in facts)},
                {"role": "user", "content": ask},
                {"role": "assistant", "content": answer},
            ],
            "category": "episode_none",
        })
    # The same skill under the always-offered block -- memory leads, then the
    # tools block, serve's own join -- and still no tool gets called.
    for idxs, facts, ask, answer in [
        ([0], [], "What were we doing last session?",
         "The garden fence — planning and pricing. Picking it back up?"),
        ([3], ["User's roommate is named Kim."], "Can we pick up where we left off?",
         "That was your sourdough starter schedule. Go on — where were we in it?"),
    ]:
        out.append({
            "messages": [
                {"role": "system", "content": _builtin_system(block(idxs, facts))},
                {"role": "user", "content": ask},
                {"role": "assistant", "content": answer},
            ],
            "category": "episode_recall",
        })
    rng.shuffle(out)
    return out


def probe_screen(eval_qs: set, locked: "LockedProbeGuard"):
    """(prompt_side, held_out) over the given probe sets -- ONE definition of
    what training may not contain, shared by main() and the tests that pin
    the hand-authored corpora as authored-to-clear.

    prompt_side yields every string the CONSUME-time guard will refuse on:
    user and system turns (finetune_enigma.load_examples splits exactly this
    way; flat records become a user turn there). held_out screens that SAME
    unit, not just the first user question -- screening only `_norm_q` left
    system blocks (where the memory-read records carry their facts)
    unscreened at build time while consume time REFUSES on them, so a guarded
    rebuild still armed a training-day refusal the builder could not clear
    (measured 2026-07-25 on the floor-2 reseal: 196 leaking prompt-side turns
    in a freshly "clean" mix; 5 even under the old floor-3 seal).
    """

    def prompt_side(rec: dict):
        msgs = rec.get("messages")
        if msgs:
            for m in msgs:
                if m.get("role") in ("user", "system") and isinstance(m.get("content"), str):
                    yield m["content"]
            return
        q = rec.get("prompt") or rec.get("question") or rec.get("instruction")
        if isinstance(q, str):
            yield q

    def held_out(rec: dict) -> bool:
        # Both halves read the WHOLE prompt side: _eval_probe_questions holds
        # each probe's teach facts as well as its question, and a teach fact
        # reaches training as a memory block (a system turn) or as a later
        # user turn -- neither of which the first-question match can see.
        return _norm_q(rec) in eval_qs or any(
            t.strip().lower() in eval_qs or locked.leaks(t) for t in prompt_side(rec)
        )

    return prompt_side, held_out


# The SERVING lineage's training block (v2 trains at 2048; SUNSET flip at
# adoption, completed by the Round-2 review 2026-08-13 -- the old 1024
# silently dropped every record that fits 2048 but not 1024, and no launcher
# invokes this builder, so the default is a bare run's only protection).
# Must match finetune_enigma's --block default; test-locked three ways.
BLOCK = 2048


def default_vocab_path():
    """The SERVING checkpoint's vocab table (v2, recorded vocab_size 16366),
    resolved lazily so a missing table fails at build time with
    vocab_file_for_size's honest error rather than at import."""
    from enigma_engine.core.tokenizer import vocab_file_for_size

    return vocab_file_for_size(16366)


def _write_artifact(path: Path, text: str) -> None:
    """Rotate-then-write: the previous artifact survives one generation at
    <stem>.prev.jsonl. A bare rebuild used to truncate the SERVED model's
    training receipt in place with nothing left to diff (review 2026-08-13);
    tmp+rename also keeps a crash from leaving a half-written corpus."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    if path.exists():
        os.replace(path, path.with_suffix(".prev" + path.suffix))
    os.replace(tmp, path)

# Numbers whose carve tells the two vocab generations apart. v1 merges some
# digit pairs and not others; a uniform table gives one token per digit.
_DIGIT_PROBES = ("15", "100", "1234", "56")


def vocab_is_digit_uniform(vocab_path: "Path | None" = None) -> bool:
    """Whether this vocab carves every digit as its own token.

    Arithmetic is learnable only when the carve is positional -- under a table
    that merges '15' into one id but splits '56' into two, the same column
    means different things in different numbers. The vocab is MEASURED rather
    than inferred from a flag or a filename, because the answer is a property
    of the token table and nothing else.
    """
    from enigma_engine.core.tokenizer import get_tokenizer

    tok = get_tokenizer("bpe", vocab_path=vocab_path) if vocab_path else get_tokenizer("bpe")
    for probe in _DIGIT_PROBES:
        pieces = [tok.decode([i]) for i in tok.encode(probe, add_special_tokens=False)]
        if [p for p in pieces if p.strip()] != list(probe):
            return False
    return True


def fit_mix_to_block(
    lines: list[str], block: int = BLOCK, vocab_path: "Path | None" = None,
    screen=None,
) -> tuple[list[str], int, int, int]:
    """Token-accurate pass over mix records using the TRAINER'S OWN renderer
    (render_training), so "fits" here means exactly what finetune_enigma.py
    will decide at load time. Records that fit pass through untouched.
    prompt/completion records that are too long get the PROMPT left-trimmed
    (keep the tail, nearest the completion) so the completion survives whole.
    Records that cannot be made to fit are dropped.

    `screen` (text -> bool, True = leaks) re-checks each TRIMMED prompt: the
    trim runs AFTER the builder's leak screen and keeps the tail, which can
    turn a passing prompt into a sealed-probe hit (score 0.005 -> 1.0 in the
    fix-arc audit's repro, 2026-07-25) -- and a rebuild deterministically
    re-creates the same trimmed record, so this was the one consume-time
    refusal "rebuild the artifact" could NOT clear. Returns
    (lines, n_trimmed, n_dropped, n_trim_leaked)."""
    from enigma_engine.core.chat_format import attach_chat_tokens, render_training
    from enigma_engine.core.tokenizer import get_tokenizer

    # Measuring with a different vocab than the trainer loads mis-sizes every
    # record: the v2 table packs ~2.2x more text per id, so a v1 measurement
    # trims and drops records that would have fit.
    tok = attach_chat_tokens(get_tokenizer("bpe", vocab_path=vocab_path) if vocab_path else get_tokenizer("bpe"))
    limit = block + 1  # the trainer keeps examples with len(ids) <= block+1
    out, trimmed, dropped, leaked = [], 0, 0, 0
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
        # cheap fast-path: for ASCII content token_count <= char_count in BOTH
        # char-mode and utf8-byte-mode tokenizers (1 byte/char, merges only
        # shrink), so a short ASCII record always fits. Non-ASCII in byte-mode
        # can expand 1 char -> several tokens, so those fall through to a real
        # render rather than being trusted here (else they'd pass as "fits" and
        # get silently dropped by the trainer -- the bug this function prevents).
        # A tool record's payload lives OUTSIDE content -- the tool_call JSON
        # and the tool turn's own markers are rendered, never summed here --
        # so those always take the real measurement.
        contents = [m.get("content") or "" for m in msgs]
        carries_tools = any(m.get("tool_calls") or m.get("role") == "tool" for m in msgs)
        if (not carries_tools and all(c.isascii() for c in contents)
                and sum(len(c) for c in contents) + 64 <= limit):
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
        cut_leaked = False
        while budget >= 32:
            cut = tok.decode(p_ids[-budget:], skip_special_tokens=True).strip()
            ids2, _ = render_training(
                tok, [{"role": "user", "content": cut}, {"role": "assistant", "content": completion}]
            )
            if len(ids2) <= limit:
                if screen is not None and screen(cut):
                    # Keep shrinking. "A shorter cut keeps the same tail and
                    # still leaks" was measured FALSE for mid-prompt hits --
                    # the next shorter cut can exclude the sealed text
                    # entirely, and breaking here over-dropped records whose
                    # clean cut the loop was one step from finding (round-B
                    # audit, 2026-07-25). Only a record with NO clean fitting
                    # cut is given up on.
                    cut_leaked = True
                    budget -= 64
                    continue
                rec2 = {"prompt": cut, "completion": completion}
                if "category" in rec:
                    rec2["category"] = rec["category"]
                out.append(json.dumps(rec2, ensure_ascii=False))
                trimmed += 1
                fitted = True
                break
            budget -= 64
        if not fitted:
            if cut_leaked:
                leaked += 1  # every fitting cut carried sealed text
            else:
                dropped += 1  # the completion alone (nearly) fills the block
    return out, trimmed, dropped, leaked


# QA gate (Phase 1d): refusal / assistant-voice boilerplate to keep OUT of the
# training mix. The from-scratch model must speak as Enigma, not parrot "As an AI
# language model, I don't have opinions." High-precision first-person patterns --
# factual mentions of "language model" are deliberately left alone.
_AI_BOILERPLATE = re.compile(
    "|".join(
        [
            r"\bas an ai\b",
            r"\bas a language model\b",
            r"\bas an ai language model\b",
            r"\bas a (helpful )?ai assistant\b",
            r"\bi'?m (just |only )?an ai\b",
            r"\bi am (just |only )?an ai\b",
            r"\bi('?m| am) an ai (language )?model\b",
            r"\bi (don'?t|do not) have (personal |real |the ability to feel )?"
            r"(feelings|emotions|opinions|beliefs|preferences|a body|consciousness)\b",
            r"\b(i'?m sorry|i apologize),? but i (cannot|can'?t|am unable to|'?m unable to)\b",
        ]
    ),
    re.IGNORECASE,
)

# QA gate 2: assistant text that claims a FOREIGN identity. OASST answers
# where the assistant names itself ("I'm OpenAssistant... trained by
# DeepMind") compete head-on with Enigma's own identity anchors on the most
# common question there is -- the v2 model literally introduced itself as
# OpenAssistant until these were purged (measured 2026-07-05: 79 of 20,094
# general records).
_FOREIGN_IDENTITY = re.compile(
    "|".join(
        [
            r"open[- ]?assistant",
            r"\bLAION\b",
            r"trained by (deepmind|deep mind|openai|google|meta|anthropic|microsoft)",
            r"developed by (openai|google|deepmind|meta|anthropic|microsoft)",
            r"\bi('m| am) (chatgpt|chat gpt|gpt-?[345]|claude|llama|bard|gemini|alexa|siri)\b",
        ]
    ),
    re.I,
)


def _assistant_text(rec: dict) -> str:
    """The assistant/completion text of a record, for QA scanning."""
    msgs = rec.get("messages")
    if msgs:
        return " ".join(m.get("content") or "" for m in msgs if m.get("role") == "assistant")
    return rec.get("completion") or rec.get("response") or rec.get("answer") or rec.get("output") or ""


def _is_ai_boilerplate(rec: dict) -> bool:
    return bool(_AI_BOILERPLATE.search(_assistant_text(rec)))


# QA gate 3 (2026-07-15, user-ordered clarity pass): the general corpus is
# bulk-pulled and carries dead links, raw HTML, encoding damage, and
# source-data loops. She learns to TALK from these records. Junk classes
# ONLY -- profanity is NOT filtered (user ruling 2026-07-15: she may use
# any type of language; the enemy is incoherence, not propriety).
# A URL in an answer is NOT junk: dropping every record whose completion
# cited a link removed all URL supervision, so she could never cite a
# source at all. The fabrication risk (a small model inventing plausible
# links) is a decode-time and grounding problem, not a data-hygiene class.
_LOW_QUALITY = re.compile(
    "|".join(
        [
            r"</?(?:div|span|p|a|br|td|tr|table|html|body|script|img)\b",  # raw HTML
            r"�",  # encoding damage
            r"\b(\w+)(?:\s+\1\b){4,}",  # same word five-plus times in a row: source loops
        ]
    ),
    re.I,
)


def _is_low_quality(rec: dict) -> bool:
    return bool(_LOW_QUALITY.search(_assistant_text(rec)))


def main(argv: "list[str] | None" = None) -> None:
    global PREAMBLE
    import argparse

    ap = argparse.ArgumentParser(description="Build the SFT mix.")
    ap.add_argument("--persona", default=None, help="persona pack; omitted = Enigma")
    ap.add_argument(
        "--out",
        default=None,
        help=f"where the artifacts land; omitted = {OUT_DIR}, which is Enigma's "
        "own. A persona pack must name its own",
    )
    ap.add_argument(
        "--vocab",
        default=None,
        help="vocab file the finetune will use; default resolves to the SERVING "
        "checkpoint's table (v2, 16366) -- the fit pass measures records "
        "against whichever vocab it is given. Pass the v1 table only when "
        "deliberately baking for the rollback lineage.",
    )
    ap.add_argument(
        "--block",
        type=int,
        default=BLOCK,
        help=f"token budget per record; must match finetune's --block (default {BLOCK})",
    )
    args = ap.parse_args(argv)

    pack = Path(args.persona) if args.persona else None
    persona = Persona.load(pack)
    # A pack supplies a NAME, and the content behind it has to come from the
    # same pack. A bare pack FILE carries the mechanical fields alone, so a
    # build from one would bake her anchors, her paraphrases and her self-facts
    # under someone else's name -- content-by-import, invisible to a grep for
    # the name. Refuse before anything is read or written, and name the missing
    # piece rather than dying downstream.
    if persona.is_default:
        content = default_content()
    elif pack is not None and pack.is_dir():
        content = load_content(pack)
    else:
        raise SystemExit(
            f"persona {persona.name!r}: {pack} is a bare pack file, which carries "
            "mechanical fields only -- it has no identity content to render. Point "
            "--persona at a pack DIRECTORY (pack.json beside its content files); "
            "refusing to train her identity under another AI's name"
        )
    # An omitted --out is Enigma's own SFT directory, and `_write_artifact`
    # rotates whatever it finds there. A pack built with that default replaces
    # the SERVED model's training receipt with another AI's corpus -- and takes
    # the .prev generation with it on the second run, so the last good copy is
    # gone too. A pack names where its own artifacts go.
    if args.out is None and not persona.is_default:
        raise SystemExit(
            f"persona {persona.name!r}: --persona with no --out writes into "
            f"{OUT_DIR}, which is Enigma's SFT mix -- this build would rotate "
            f"her artifacts away and leave {persona.name}'s corpus there. Give "
            "--out a directory of this pack's own"
        )
    out_dir = Path(args.out) if args.out else OUT_DIR
    # WHO is being built. Unconditional, for serve's reason: rebinding only
    # when the flag is present would leave a second call in one process
    # rendering the PREVIOUS persona's preamble into every system block.
    PREAMBLE = persona.tools_preamble
    if not persona.is_default:
        print(f"persona: {persona.name}", flush=True)

    vocab_path = Path(args.vocab) if args.vocab else default_vocab_path()
    if not vocab_path.exists():
        raise SystemExit(f"vocab file not found: {vocab_path}")
    print(f"fit vocab: {vocab_path.name} | block {args.block}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)

    # The eval probe set is held out of ALL training (identity, tools,
    # restraint) so the harness always measures generalization, never a
    # memorized probe. Restraint especially: we train MANY greeting surfaces
    # and the eval tests held-out ones ("How's it going?").
    #
    # HER dev set is hers: its questions are what the harness grades ENIGMA on,
    # and holding them out of a pack's corpus would thin that pack for
    # resembling questions its AI is never asked. A pack screens against its
    # own sealed file instead, and says so out loud when it has none -- an AI
    # with no gate has nothing to leak into.
    if persona.is_default:
        eval_qs = _eval_probe_questions()
    elif (pack / LOCKED_PROBES_NAME).exists():
        eval_qs = _eval_probe_questions(pack / LOCKED_PROBES_NAME)
    else:
        eval_qs = set()
        print(f"WARN: {persona.name} has no sealed set -- the exact-match probe "
              f"screen is inactive (no {pack / LOCKED_PROBES_NAME})", flush=True)
    # DEV probes get the exact-match backstop above (you may iterate toward the
    # dev set). The LOCKED set is the honest gate you must NEVER train toward;
    # its sealed manifest catches paraphrases too (EVAL_REDESIGN.md). Sealed and
    # LIVE since reseal #7, so this guard screens every build -- the line below
    # prints the sealed count it actually loaded. WHICH gate is the loaded
    # PERSONA's answer, never the argument's: an Enigma-SPELLED pack IS her, so
    # its build screens against hers.
    manifest = persona_manifest(None if persona.is_default else pack)
    locked = LockedProbeGuard.load(manifest)
    _prompt_side, _held_out = probe_screen(eval_qs, locked)

    if len(locked):
        print(f"locked-probe fuzzy guard ACTIVE: {len(locked)} sealed probes (jaccard >= {locked.threshold})")
    else:
        print(f"locked-probe fuzzy guard inactive (no {manifest} yet)")

    all_tools = gen_tool_examples(content=content)
    tools = [r for r in all_tools if not _held_out(r)]
    n_tool_leak = len(all_tools) - len(tools)
    # Text built here, WRITTEN at the end with the mix: rotating this file
    # ~220 lines before fit_mix_to_block (which loads a tokenizer and can
    # raise) left a crashed build with tool_calls/identity rotated while
    # mix + manifest still described the previous build -- and the next
    # failing run then destroyed the last good .prev (audit 2026-08-14).
    # All three artifacts + manifest now land together or not at all.
    tool_text = "\n".join(json.dumps(r, ensure_ascii=False) for r in tools) + "\n"
    print(
        f"tool_calls.jsonl: {len(tools)} examples "
        f"({sum(1 for r in tools if r['category'] == 'tool_restraint')} restraint, "
        f"{n_tool_leak} held out of training as eval probes)"
    )

    # Identity = hand-authored anchors + paraphrase-augmented records. The
    # anchors carry the voice; the paraphrases carry DIVERSITY so identity
    # GENERALIZES to unseen phrasings instead of memorizing exact strings
    # (held-out eval 2026-07-05: x20 repetition of 159 fixed anchors -> 17%
    # on novel phrasings).
    anchors, dropped = gen_identity_examples(content)
    paraphrases = gen_identity_paraphrases(content=content)
    ident = [r for r in anchors + paraphrases if not _held_out(r)]
    n_leak = (len(anchors) + len(paraphrases)) - len(ident)
    ident_text = "\n".join(json.dumps(r, ensure_ascii=False) for r in ident) + "\n"
    print(
        f"identity.jsonl: {len(ident)} records ({len(anchors)} anchors + {len(paraphrases)} "
        f"paraphrases, {dropped} Qwen-era dropped, {n_leak} held out of training as eval probes)"
    )

    # Math trains against a digit-uniform vocab only. v1 splits numbers
    # inconsistently ('56'->['5','6'] but '15'->['15'], '100'->['1','00']), and
    # digit-wise arithmetic cannot be learned through that -- it taught the
    # model to emit confidently-wrong numbers. The v2 table carves every digit
    # separately ('15'->['1','5'], '100'->['1','0','0'], '3.14'->['3','.','1',
    # '4'] -- measured 2026-08-07), so the records land there. Baking for a v1
    # checkpoint drops them rather than repeating the v4 result.
    math = []
    if vocab_is_digit_uniform(vocab_path):
        all_math = gen_math_examples()
        math = [r for r in all_math if not _held_out(r)]
        print(f"math: {len(math)} records (vocab carves digits uniformly, "
              f"{len(all_math) - len(math)} held out of training as eval probes)")
    else:
        print("math: SKIPPED -- this vocab does not carve digits uniformly; "
              "pass --vocab <v2 table> to include arithmetic")

    # User-authored teachings (teachings.jsonl, gitignored) ride the same
    # oversample weight as identity -- few records, personally important.
    # HERS alone: the file is the user's channel into ENIGMA's weights, facts
    # taught to her in her own sessions, and there is no sense in which
    # another AI was told them.
    teach: list[dict] = []
    if persona.is_default:
        all_teach = gen_teaching_examples()
        teach = [r for r in all_teach if not _held_out(r)]
        n_teach_leak = len(all_teach) - len(teach)
        if teach or n_teach_leak:  # a fully screened-out corpus still owes a count
            print(f"teachings: {len(teach)} records from {TEACHINGS.name} "
                  f"({n_teach_leak} held out of training as eval probes)")
    else:
        print(f"teachings: SKIPPED -- {TEACHINGS.name} is Enigma's own channel, "
              f"not {persona.name}'s")

    # Memory-READING records (use the injected 'Things you remember:' block).
    all_mem_read = gen_memory_read_examples()
    mem_read = [r for r in all_mem_read if not _held_out(r)]
    print(f"memory-read: {len(mem_read)} records "
          f"({len(all_mem_read) - len(mem_read)} held out of training as eval probes)")

    # COMBINED memory+tools system shape -- what serve renders when a memory
    # hit coincides with offered tools (ultrareview #6).
    all_mem_tools = gen_memory_tools_examples()
    mem_tools = [r for r in all_mem_tools if not _held_out(r)]
    print(f"memory+tools: {len(mem_tools)} records "
          f"({len(all_mem_tools) - len(mem_tools)} held out of training as eval probes)")

    # Image-READING records (use the eyes organ's '[image: ...]' markers).
    all_img_read = gen_image_read_examples()
    img_read = [r for r in all_img_read if not _held_out(r)]
    print(f"image-read: {len(img_read)} records "
          f"({len(all_img_read) - len(img_read)} held out of training as eval probes)")

    # Clean world-knowledge QA (knowledge_corpus.py) -- the counterweight to
    # the noisy general corpus: curated facts in short plain sentences.
    all_knowledge = gen_knowledge_examples(content=content)
    knowledge = [r for r in all_knowledge if not _held_out(r)]
    print(f"knowledge: {len(knowledge)} records "
          f"({len(all_knowledge) - len(knowledge)} held out of training as eval probes)")

    # The new-shape corpora share ONE held-out count, accumulated as they are
    # screened: deriving it by generating all eight a second time paid for
    # every corpus twice and reported a number the mix could disagree with.
    n_shape_leak = 0

    def screen_shape(recs: list[dict]) -> list[dict]:
        nonlocal n_shape_leak
        kept = [r for r in recs if not _held_out(r)]
        n_shape_leak += len(recs) - len(kept)
        return kept

    # Every built-in offered on every turn, with the restraint half that makes
    # that safe -- the shape serve would render once the intent gates retire.
    # (Serve-side execution is PARKED: tried 2026-08-20, measured worse than
    # the gates on the sealed set, reverted. This corpus stays -- it is what a
    # future block-trained lineage needs to beat them; BACKLOG T4.)
    builtins = screen_shape(gen_builtin_block_examples(content=content))

    # Conversation with more than one user turn: nothing else in the mix has
    # a second one to read the first against.
    chats = screen_shape(gen_chat_multiturn_examples())

    # Answers that work the problem inside <think>...</think> before replying.
    reasoning = screen_shape(gen_reasoning_examples())

    # Corrections to a remembered fact -- the turn the supersede path renders
    # and no record has ever trained.
    mem_fix = screen_shape(gen_memory_correction_examples())

    # Search traces: the span, the results, the answer with a receipt -- the
    # sixth organ's trained side (serve's _apply_search renders the same
    # trace at runtime).
    searches = screen_shape(gen_search_examples())

    # Honest declines on the intrinsically unanswerable -- v2's #1 win
    # condition, near-absent from the diet before this generator.
    unknowns = screen_shape(gen_unknown_examples())

    # Structured output on request -- JSON with exactly the named keys and
    # nothing else (ledger 14; runtime enforcement follows later, the
    # trained shapes cannot).
    structured = screen_shape(gen_structured_examples())

    # Session memory -- answer "where did we leave off?" from the newest
    # episode line, admit it when there is none (ledger 15).
    episodes = screen_shape(gen_episode_examples())

    print(
        f"new shapes: {len(builtins)} builtin-block, {len(chats)} chat-multiturn, "
        f"{len(reasoning)} reasoning, {len(mem_fix)} memory-correction, "
        f"{len(searches)} search, {len(unknowns)} unknown-decline, "
        f"{len(structured)} structured-output, {len(episodes)} episode-recall "
        f"({n_shape_leak} held out of training as eval probes -- these corpora "
        f"are authored to make that 0; nonzero means a reseal newly collided)"
    )

    # Diverse identity data generalizes with FAR less repetition than fixed
    # pairs did; a moderate boost is enough (~370 diverse records x8 ~= the old
    # x20 weight, but now the model sees many surfaces per fact).
    # 2026-07-15 diet rebuild: general grew ~20k -> ~105k records, so repeats
    # are STREAM-FRACTION knobs now, not absolute-exposure knobs. Measured:
    # v6 (old repeats, new diet) -- memory-read collapsed 6/8 -> 3/8 and
    # identity slipped to 14/18. v7 (identity x12, memread x25) -- repetition
    # alone did NOT fix it (memory 5/8, identity 13/18): the failures were
    # novel SURFACES, so v8 widens the corpora instead (26 memory facts with
    # "again?"/"which month?"/job shapes; name/size/privacy identity
    # families) and keeps moderate fractions. Tools held 12/12 at x5
    # throughout; knowledge holds at x5 (facts install in continued
    # pretrain, SFT only surfaces them).
    IDENTITY_REPEAT = 8
    TOOLS_REPEAT = 5
    # x8 -> x4 (2026-07-16): teach_enigma.py now AUTO-AUGMENTS each /fix into
    # several phrasings + a statement twin, so a correction carries its own
    # surface variety and no longer needs a high repeat to avoid memorizing one
    # exact string (which also amplified a WRONG teaching). Older single-phrasing
    # teachings still generalize less, but the generator warns on those.
    TEACHINGS_REPEAT = 4
    MEMREAD_REPEAT = 12
    # The combined shape rides between the two parents' weights: it reuses
    # mem_read surfaces (already x12 alone) and tool surfaces (x5 alone), so
    # a moderate weight teaches the JOIN without double-counting the parts.
    MEMTOOLS_REPEAT = 8
    IMGREAD_REPEAT = 10
    # x2 was too light to generalize across phrasings (measured 2026-07-15 on
    # the adopted v5: "largest planet" -> Jupiter but "biggest planet" ->
    # Saturn at greedy). Identity generalizes at x8; facts get the same class
    # of weight.
    KNOWLEDGE_REPEAT = 5
    # Arithmetic is a SKILL that must generalize across phrasings, so it takes
    # the same class of weight as knowledge rather than the high repetition
    # identity needs. The records are already one-per-question after dedup.
    MATH_REPEAT = 5
    # The built-in block is the shape EVERY served turn will carry once the
    # gates retire, so it takes the tool weight rather than a token one.
    BUILTIN_REPEAT = 12
    # Small hand-authored corpora that must generalize to unseen surfaces get
    # the identity class of weight -- the repo's own lesson is that coverage
    # beats repetition, but these have few surfaces to begin with.
    CHAT_REPEAT = 10
    REASONING_REPEAT = 10
    MEMFIX_REPEAT = 12
    SEARCH_REPEAT = 10
    # The heaviest of the new weights: the category is the declared #1 win
    # condition and the behavior must survive 90k records of general text
    # that answers everything it is asked.
    UNKNOWN_REPEAT = 10
    STRUCTURED_REPEAT = 10
    # Episode lines are a memory-block shape, so they take the memory class
    # of weight (the block shapes demonstrably need it -- the 2026-07-06
    # ignored-injection lesson).
    EPISODE_REPEAT = 12
    mix = [
        json.dumps(r, ensure_ascii=False)
        for r in tools * TOOLS_REPEAT
        + ident * IDENTITY_REPEAT
        + teach * TEACHINGS_REPEAT
        + mem_read * MEMREAD_REPEAT
        + mem_tools * MEMTOOLS_REPEAT
        + img_read * IMGREAD_REPEAT
        + knowledge * KNOWLEDGE_REPEAT
        + math * MATH_REPEAT
        + builtins * BUILTIN_REPEAT
        + chats * CHAT_REPEAT
        + reasoning * REASONING_REPEAT
        + mem_fix * MEMFIX_REPEAT
        + searches * SEARCH_REPEAT
        + unknowns * UNKNOWN_REPEAT
        + structured * STRUCTURED_REPEAT
        + episodes * EPISODE_REPEAT
    ]
    n_general = 0
    n_bad_json = 0
    n_boiler = 0
    n_foreign = 0
    n_lowq = 0
    n_gen_leak = 0
    n_orphan_tag = 0
    n_locked_near = 0  # general records in the locked-guard review band (kept, flagged)
    locked_near_rows: list[str] = []  # written out so a human CAN actually review them
    if GENERAL.exists():
        with open(GENERAL, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    # Counted like every other drop class: an unreadable line
                    # shrinks the diet, and a silent shrink is the one thing
                    # this build promises never to do.
                    n_bad_json += 1
                    continue
                if "<search>" in line or "</search>" in line:
                    # No tag trains unowned (T1 ruling 4): the collected
                    # corpus's tag records predate the search organ and teach
                    # emit-and-stop toward a runtime contract that never
                    # existed. gen_search_examples owns the tag now, with
                    # full traces against the real loop -- the 31 seed-era
                    # orphans drop here rather than by editing the artifact.
                    n_orphan_tag += 1
                    continue
                if _is_ai_boilerplate(rec):  # QA gate: keep assistant-voice boilerplate out
                    n_boiler += 1
                    continue
                if _FOREIGN_IDENTITY.search(_assistant_text(rec)):  # QA gate 2: no foreign self-identity
                    n_foreign += 1
                    continue
                if _is_low_quality(rec):  # QA gate 3: URLs/HTML/encoding/loops (profanity passes -- user ruling)
                    n_lowq += 1
                    continue
                if _held_out(rec):  # exact dev probe OR fuzzy-close to a locked probe
                    n_gen_leak += 1
                    continue
                # Kept, but flagged for human review -- over the SAME unit the
                # drops use (all prompt-side turns), or the review band
                # under-reports exactly where the screen was just widened.
                # Curated/generated families stay unflagged by design: their
                # shapes are code, reviewed as code, and would spam this file
                # identically every build.
                if any(locked.is_near_miss(t) for t in _prompt_side(rec)):
                    n_locked_near += 1
                    locked_near_rows.append(line)
                mix.append(line)
                n_general += 1
    mix, n_trimmed, n_dropped, n_trim_leak = fit_mix_to_block(
        mix, block=args.block, vocab_path=vocab_path,
        screen=locked.leaks if len(locked) else None,
    )
    random.Random(42).shuffle(mix)
    mix_text = "\n".join(mix) + "\n"
    # All artifacts land together, AFTER every fallible stage: see the
    # deferred-write comment at the tool_calls build above.
    _write_artifact(out_dir / "tool_calls.jsonl", tool_text)
    _write_artifact(out_dir / "identity.jsonl", ident_text)
    _write_artifact(out_dir / "mix.jsonl", mix_text)
    # The build manifest: the shipped mix was previously identifiable only by
    # stdout scrollback (review 2026-08-13). Every sha256 here is over the
    # LF-normalized text this build assembled in memory, NOT the file on disk:
    # _write_artifact goes through Python's text layer, so the artifacts land
    # CRLF on Windows and hash differently (measured 2026-08-19). Two builds
    # agreeing on these digests agree on the corpus, whatever the line endings.
    # Written through the same rotate-then-write path as its subjects, so the
    # preserved .prev generation keeps its receipt and a crash cannot leave
    # truncated JSON (audit 2026-08-14; same audit added the full vocab path
    # -- the name alone could not tell two same-named tables apart -- plus
    # math_records, which the --vocab default flip switched from 0 to ~1067
    # on a bare run, and the sibling artifacts' shas).
    _write_artifact(out_dir / "mix.manifest.json",
        json.dumps({
            "records": len(mix),
            "block": args.block,
            "vocab": vocab_path.name,
            "vocab_path": str(vocab_path),
            "sha256": hashlib.sha256(mix_text.encode("utf-8")).hexdigest(),
            "tool_calls_sha256": hashlib.sha256(tool_text.encode("utf-8")).hexdigest(),
            "identity_sha256": hashlib.sha256(ident_text.encode("utf-8")).hexdigest(),
            "repeats": {"identity": IDENTITY_REPEAT, "tools": TOOLS_REPEAT,
                        "mem_read": MEMREAD_REPEAT, "mem_tools": MEMTOOLS_REPEAT,
                        "img_read": IMGREAD_REPEAT, "knowledge": KNOWLEDGE_REPEAT},
            "math_records": len(math),
            "teaching_records": len(teach),
            "general_kept": n_general,
        }, indent=2) + "\n")
    # The review-band records themselves -- a bare count was unactionable
    # (audit 2026-07-16): nothing identified WHICH records to look at.
    # REGENERATED every run: act on the records (fix the corpus or the
    # probes); don't annotate this file, your notes would be clobbered.
    review_path = out_dir / "locked_near_misses.jsonl"
    if locked_near_rows:
        review_path.write_text("\n".join(locked_near_rows) + "\n", encoding="utf-8")
    elif review_path.exists():
        review_path.unlink()  # stale review file from an earlier run
    print(
        f"mix.jsonl: {len(mix)} records (identity x{IDENTITY_REPEAT}, tools x{TOOLS_REPEAT}, "
        f"{len(mem_read)} memory-read x{MEMREAD_REPEAT}, "
        f"{len(mem_tools)} memory+tools x{MEMTOOLS_REPEAT}, "
        f"{len(img_read)} image-read x{IMGREAD_REPEAT}, "
        f"{len(knowledge)} knowledge x{KNOWLEDGE_REPEAT}; "
        f"{n_general} general kept; {n_bad_json} dropped as malformed JSON; "
        f"{n_boiler} dropped as "
        f"AI-voice boilerplate; {n_foreign} dropped as foreign self-identity; "
        f"{n_lowq} dropped as low-quality (HTML/encoding/loops); "
        f"{n_orphan_tag} dropped as orphan search-tag records (gen_search_examples owns the tag now); "
        f"{n_gen_leak} dropped as eval-probe leaks; "
        f"{n_locked_near} kept but flagged near a locked probe"
        f"{' (see locked_near_misses.jsonl)' if n_locked_near else ''}; "
        f"{n_trimmed} prompt-trimmed to fit block {args.block}, "
        f"{n_dropped} dropped as unfittable, "
        f"{n_trim_leak} trimmed prompts dropped as post-trim leaks)"
    )


if __name__ == "__main__":
    main()
