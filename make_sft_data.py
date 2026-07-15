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
from identity_paraphrases import gen_identity_paraphrases

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "data" / "sft"
GENERAL = ROOT / "data" / "finetune" / "combined_finetune.jsonl"
EVAL_PROBES = ROOT / "data" / "eval" / "behavior_probes.jsonl"
TEACHINGS = ROOT / "teachings.jsonl"  # user-authored; gitignored (see teachings.example.jsonl)


def _norm_q(rec: dict) -> str:
    """The record's first user question, normalized for eval-leak comparison.
    Handles both the messages shape and the general data's prompt/completion."""
    for m in rec.get("messages", []):
        if m.get("role") == "user" and m.get("content"):
            return m["content"].strip().lower()
    return (rec.get("prompt") or "").strip().lower()


def _eval_probe_questions() -> set[str]:
    """Questions in the held-out behavior eval -- NEVER put these in training,
    or the harness measures memorization instead of generalization."""
    if not EVAL_PROBES.exists():
        return set()
    qs = set()
    for line in EVAL_PROBES.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
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
            ("Can you look happy?", {"emotion": "happy"}, "ok", "There -- wearing my happy face."),
            ("Show me you're surprised.", {"emotion": "surprised"}, "ok", "Did that look surprised enough?"),
            ("Act sleepy for a second.", {"emotion": "sleepy"}, "ok", "So sleepy... okay, I'm awake again."),
            ("Give me your best grumpy face.", {"emotion": "grumpy"}, "ok", "Hmph. Grumpy face deployed."),
            ("Look excited, we won!", {"emotion": "excited"}, "ok", "We won! I can barely sit still."),
            ("Can you look thoughtful for a moment?", {"emotion": "thoughtful"}, "ok", "Hmm... there, deep in thought."),
            ("Pull a scared face.", {"emotion": "scared"}, "ok", "Eep -- scared face on."),
            ("Wink at me.", {"emotion": "wink"}, "ok", "*wink*"),
        ],
    ),
    (
        "avatar_say",
        "Speak a line out loud through the avatar.",
        {"text": "string"},
        [
            ("Say hello to my friend Sam.", {"text": "Hello Sam, lovely to meet you!"}, "ok", "Said it out loud -- I hope Sam heard."),
            ("Announce that dinner is ready.", {"text": "Dinner is ready!"}, "ok", "Announcement made."),
            ("Tell the room good morning.", {"text": "Good morning, everyone!"}, "ok", "Good morning delivered."),
            ("Wish my mom a happy birthday out loud.", {"text": "Happy birthday, Mom!"}, "ok", "Said it -- happy birthday to your mom."),
            ("Say 'the build passed' so I can hear it from the kitchen.", {"text": "The build passed."}, "ok", "Called it out -- the build passed."),
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
            ("Tell everyone lunch is here.", {"text": "Lunch is here, everyone!"}, "speaking", "Called it out -- lunch is here."),
            ("Read that back to me out loud: backup finished.", {"text": "Backup finished."}, "speaking", "Read it out: backup finished."),
            ("Say something encouraging, out loud.", {"text": "You've got this -- keep going!"}, "speaking", "Spoken. You've got this."),
            ("Announce the meeting starts in five minutes.", {"text": "The meeting starts in five minutes."}, "speaking", "Announced -- five minutes to the meeting."),
            ("Use your voice to count down from three.", {"text": "Three... two... one."}, "speaking", "Counted it down out loud."),
        ],
    ),
    (
        "see_screen",
        "Read the text currently visible on the user's screen.",
        {},
        [
            ("What's on my screen right now?", {}, "An editor with serve_enigma.py open.", "You have an editor open on serve_enigma.py."),
            ("Can you check what window I left open?", {}, "A browser showing a recipe for ramen.", "A browser with a ramen recipe -- planning dinner?"),
            ("Read me the error on screen.", {}, "TypeError: cannot read property 'map' of undefined", "The screen shows: TypeError -- 'map' of undefined. Something's null before you map it."),
            ("What am I looking at?", {}, "A spreadsheet titled Q3 budget.", "A spreadsheet titled 'Q3 budget'."),
            ("Is there anything on screen I should know about?", {}, "A dialog: 'Unsaved changes'.", "There's an 'Unsaved changes' dialog waiting on you -- save before you lose work."),
        ],
    ),
    (
        "generate_code",
        "Generate a code snippet.",
        {"prompt": "string", "language": "string"},
        [
            ("Write me a python one-liner that reverses a string.", {"prompt": "reverse a string one-liner", "language": "python"}, "s[::-1]", "Here it is: `s[::-1]` -- slicing with a negative step."),
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
            ("What is 7 times 8?", {"expression": "7 * 8"}, "56", "7 times 8 is 56."),
            ("What's 15 plus 27?", {"expression": "15 + 27"}, "42", "That's 42."),
            ("What is 100 divided by 4?", {"expression": "100 / 4"}, "25", "100 divided by 4 is 25."),
            ("What's 9 minus 3?", {"expression": "9 - 3"}, "6", "Six."),
            ("Compute 25 plus 30.", {"expression": "25 + 30"}, "55", "25 plus 30 is 55."),
            ("What is 12 times 12?", {"expression": "12 * 12"}, "144", "144."),
            ("What's 2 to the power of 10?", {"expression": "2 ** 10"}, "1024", "2 to the 10th is 1024."),
            ("What is 144 divided by 12?", {"expression": "144 / 12"}, "12", "That's 12."),
            ("What's 45 times 3?", {"expression": "45 * 3"}, "135", "45 times 3 is 135."),
            ("How much is 250 minus 175?", {"expression": "250 - 175"}, "75", "75."),
            ("What's 17 percent of 200?", {"expression": "200 * 17 / 100"}, "34", "17 percent of 200 is 34."),
            ("What is (3 plus 4) times 5?", {"expression": "(3 + 4) * 5"}, "35", "That's 35."),
            ("What's the remainder of 17 divided by 5?", {"expression": "17 % 5"}, "2", "The remainder is 2."),
            ("Add 8, 16, and 32.", {"expression": "8 + 16 + 32"}, "56", "That comes to 56."),
            ("What is 1000 minus 1?", {"expression": "1000 - 1"}, "999", "999."),
        ],
    ),
    (
        "remember",
        "Save a fact about the user to long-term memory.",
        {"text": "string"},
        [
            # Explicit remember asks.
            ("Remember that my birthday is March 3rd.", {"text": "User's birthday is March 3rd."}, "saved: User's birthday is March 3rd.", "Saved. March 3rd -- I won't forget."),
            ("Please remember I'm allergic to peanuts.", {"text": "User is allergic to peanuts."}, "saved: User is allergic to peanuts.", "Noted, and filed under important: peanut allergy."),
            ("Don't forget that my meeting is on Fridays.", {"text": "User's meeting is on Fridays."}, "saved: User's meeting is on Fridays.", "Got it -- Fridays."),
            ("Note that I prefer short answers.", {"text": "User prefers short answers."}, "saved: User prefers short answers.", "Noted."),
            ("Keep in mind I work night shifts.", {"text": "User works night shifts."}, "saved: User works night shifts.", "Saved. I'll keep the mornings quiet."),
            # Bare fact statements -- no explicit 'remember', still save-worthy.
            ("My dog's name is Rex.", {"text": "User's dog is named Rex."}, "saved: User's dog is named Rex.", "Rex -- noted."),
            ("My favorite color is green.", {"text": "User's favorite color is green."}, "saved: User's favorite color is green.", "Green. Saved."),
            ("I live in Denver.", {"text": "User lives in Denver."}, "saved: User lives in Denver.", "Denver -- got it."),
            ("Call me Sam.", {"text": "User goes by Sam."}, "saved: User goes by Sam.", "Sam it is."),
            ("I love spicy food.", {"text": "User loves spicy food."}, "saved: User loves spicy food.", "Noted -- the spicier the better."),
            ("My brother's name is Leo.", {"text": "User's brother is named Leo."}, "saved: User's brother is named Leo.", "Leo. Saved."),
            ("I drive a blue pickup.", {"text": "User drives a blue pickup."}, "saved: User drives a blue pickup.", "Blue pickup -- noted."),
            # Corrections -- the store supersedes the old fact, she confirms the update.
            ("Actually, we renamed the dog to Bruno.", {"text": "User's dog is named Bruno."}, "updated: User's dog is named Bruno.", "Updated -- Bruno now."),
            ("Correction: I moved to Austin.", {"text": "User lives in Austin."}, "updated: User lives in Austin.", "Updated. Austin it is."),
        ],
    ),
    (
        "get_weather",
        "Get current weather for a city.",
        {"city": "string"},
        [
            ("What's the weather in Tokyo?", {"city": "Tokyo"}, "18C, light rain", "Tokyo is 18C with light rain right now -- umbrella weather."),
            ("Is it cold in Oslo today?", {"city": "Oslo"}, "-3C, clear", "Yes -- Oslo is at -3C, clear skies. Bundle up."),
        ],
    ),
    (
        "set_timer",
        "Set a countdown timer.",
        {"minutes": "number", "label": "string"},
        [
            ("Set a 10 minute timer for the pasta.", {"minutes": 10, "label": "pasta"}, "timer set", "Done -- 10 minutes on the pasta timer."),
            ("Remind me in 25 minutes to stretch.", {"minutes": 25, "label": "stretch"}, "timer set", "Timer set: I'll nudge you to stretch in 25 minutes."),
        ],
    ),
    (
        "search_notes",
        "Search the user's local notes.",
        {"query": "string"},
        [
            ("Find my note about the garden plan.", {"query": "garden plan"}, "Note: 'Garden plan -- tomatoes east bed, basil by the door.'", "Found it: tomatoes go in the east bed, basil by the door."),
            ("What did I write down about the meeting with Dana?", {"query": "meeting Dana"}, "Note: 'Dana meeting -- ship v2 by Friday.'", "Your note says: ship v2 by Friday."),
            ("Pull up my note on the wifi password.", {"query": "wifi password"}, "Note: 'wifi: hunter2-guest'", "Your note has it: the guest wifi is 'hunter2-guest'."),
            ("Did I save anything about the car service?", {"query": "car service"}, "Note: 'Car service -- due at 60k miles, call Mike's.'", "Yes: car service is due at 60k miles, and you noted to call Mike's."),
        ],
    ),
    (
        "play_music",
        "Play a song or playlist.",
        {"title": "string"},
        [
            ("Put on some lo-fi.", {"title": "lo-fi"}, "playing", "Lo-fi is on. Cozy."),
            ("Play Clair de Lune.", {"title": "Clair de Lune"}, "playing", "Playing Clair de Lune -- good choice."),
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
            ("Roll a d20 for me.", {"sides": 20}, "17", "You rolled a 17 -- solid."),
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
_TIMER_ASKS = ["Set a {m} minute timer for the {l}.", "Remind me about the {l} in {m} minutes.", "Start a {m}-minute {l} timer.", "Ping me in {m} minutes -- {l}.", "Give me {m} minutes on the {l}."]
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
    "search_notes": [("Find my note about the submarine schematics.", {"query": "submarine schematics"}, "error: no match", "I searched but found no note about submarine schematics. Might be under a different title -- want me to try another term?")],
    "play_music": [
        ("Play that song from the thing.", {"title": "that song from the thing"}, "error: not found", "I couldn't find anything matching that -- can you give me a title or artist and I'll put it on?"),
        ("Play my Spotify liked songs.", {"title": "Spotify liked songs"}, "error: no player connected", "There's no music player connected right now, so I can't start anything. Once one's hooked up I'll play it."),
    ],
    "set_timer": [("Set a timer for negative five minutes.", {"minutes": -5, "label": "?"}, "error: invalid duration", "A negative timer won't work. How many minutes forward should I count?")],
    "generate_code": [("Write me a program in Klingon.", {"prompt": "hello world", "language": "klingon"}, "error: unsupported language", "I can't generate Klingon -- it's not a supported language. Want it in Python, JS, Rust, or something else real?")],
    "translate": [("Translate 'hello' to Elvish.", {"text": "hello", "to": "elvish"}, "error: unsupported language", "Elvish isn't supported, I'm afraid. Give me a real target language and I'll translate it.")],
    "avatar_express": [("Make the avatar do a backflip.", {"emotion": "backflip"}, "error: unknown emotion", "I can show expressions, but 'backflip' isn't one of them -- I can do happy, surprised, sleepy, and the like. Want one of those?")],
    "see_screen": [("What's on my screen?", {}, "error: no display captured", "I couldn't read the screen just now -- nothing came back from the capture. Is a window actually in focus?")],
    "get_weather": [
        ("What's the weather in Atlantis?", {"city": "Atlantis"}, "error: unknown city", "I couldn't find weather for Atlantis -- the service didn't recognize it. Want to try another spelling or a nearby city?"),
        ("Weather on the Moon?", {"city": "the Moon"}, "error: unknown city", "No forecast for the Moon, sadly -- the service only covers Earth cities. Somewhere down here I can try?"),
    ],
}

# Multi-turn conversations chaining tools (call -> result -> follow-up -> call...).
# Authored so the model learns to keep using tools across a conversation.
MULTITURN = [
    {"tools": ["get_weather", "avatar_say"], "turns": [
        ("Is it raining in Seattle? If so, say so out loud.", {"name": "get_weather", "arguments": {"city": "Seattle"}}, "14C, rain"),
        (None, {"name": "avatar_say", "arguments": {"text": "It's raining in Seattle -- grab an umbrella."}}, "ok"),
    ], "final": "It's 14C and raining in Seattle -- I said it out loud for you."},
    {"tools": ["set_timer", "play_music"], "turns": [
        ("Start a 20 minute focus timer and put on lo-fi.", {"name": "set_timer", "arguments": {"minutes": 20, "label": "focus"}}, "timer set"),
        (None, {"name": "play_music", "arguments": {"title": "lo-fi"}}, "playing"),
    ], "final": "20-minute focus timer running and lo-fi is playing. Go get it."},
    {"tools": ["search_notes", "set_timer"], "turns": [
        ("Check my note for when the bread proofs, then set a timer for it.", {"name": "search_notes", "arguments": {"query": "bread proof"}}, "Note: 'bread -- proof 40 min'"),
        (None, {"name": "set_timer", "arguments": {"minutes": 40, "label": "bread proof"}}, "timer set"),
    ], "final": "Your note says 40 minutes -- proofing timer is set."},
    {"tools": ["see_screen", "generate_code"], "turns": [
        ("Look at the error on my screen and give me a fix.", {"name": "see_screen", "arguments": {}}, "NameError: 'reqests' is not defined"),
        (None, {"name": "generate_code", "arguments": {"prompt": "fix the typo: reqests -> requests import", "language": "python"}}, "import requests"),
    ], "final": "Typo on screen: 'reqests' should be 'requests'. Add `import requests` and fix the spelling."},
    {"tools": ["get_weather", "avatar_express"], "turns": [
        ("What's it like in Honolulu -- and react to it.", {"name": "get_weather", "arguments": {"city": "Honolulu"}}, "29C, sunny"),
        (None, {"name": "avatar_express", "arguments": {"emotion": "happy"}}, "ok"),
    ], "final": "Honolulu is 29C and sunny -- couldn't help but smile at that."},
    {"tools": ["translate", "avatar_say"], "turns": [
        ("How do you say welcome in Italian? Say it out loud too.", {"name": "translate", "arguments": {"text": "welcome", "to": "italian"}}, "benvenuto"),
        (None, {"name": "avatar_say", "arguments": {"text": "Benvenuto!"}}, "ok"),
    ], "final": "In Italian it's 'benvenuto' -- and I said it out loud."},
    {"tools": ["get_weather", "play_music"], "turns": [
        ("If it's grey in London, put on something mellow.", {"name": "get_weather", "arguments": {"city": "London"}}, "12C, overcast"),
        (None, {"name": "play_music", "arguments": {"title": "quiet acoustic"}}, "playing"),
    ], "final": "London's overcast at 12C, so I put on some quiet acoustic to match the mood."},
    {"tools": ["set_timer", "set_timer"], "turns": [
        ("Set two timers: 10 minutes for the pasta and 5 for the sauce.", {"name": "set_timer", "arguments": {"minutes": 10, "label": "pasta"}}, "timer set"),
        (None, {"name": "set_timer", "arguments": {"minutes": 5, "label": "sauce"}}, "timer set"),
    ], "final": "Both set -- 10 minutes on the pasta, 5 on the sauce."},
    {"tools": ["search_notes", "avatar_say"], "turns": [
        ("What's the wifi password? Read it out so my guest can hear.", {"name": "search_notes", "arguments": {"query": "wifi password"}}, "Note: 'wifi: hunter2-guest'"),
        (None, {"name": "avatar_say", "arguments": {"text": "The guest wifi password is hunter2-guest."}}, "ok"),
    ], "final": "Found it and said it out loud: the guest wifi is 'hunter2-guest'."},
    {"tools": ["generate_code", "see_screen"], "turns": [
        ("Check my screen, then write a fix for whatever's broken.", {"name": "see_screen", "arguments": {}}, "IndentationError on line 12"),
        (None, {"name": "generate_code", "arguments": {"prompt": "fix an IndentationError on line 12", "language": "python"}}, "# align line 12 to 4-space indent"),
    ], "final": "Line 12 has an IndentationError -- align it to the same 4-space indent as the block it belongs to."},
    {"tools": ["roll_dice", "avatar_express"], "turns": [
        ("Roll a d20 and react to what I get.", {"name": "roll_dice", "arguments": {"sides": 20}}, "20"),
        (None, {"name": "avatar_express", "arguments": {"emotion": "excited"}}, "ok"),
    ], "final": "Natural 20! I can't hide how excited that makes me."},
]

# Questions that should NOT trigger a tool even when tools are offered.
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
    ("Good evening.", "Evening. What are we doing?"),
    ("Good afternoon.", "Afternoon. Where do you want to start?"),
    ("Good night.", "Night. I'll be here whenever."),
    ("How's it going?", "Running clean. You?"),
    ("How are you?", "Good -- idling, ready. You?"),
    ("How are you doing today?", "Steady as ever. What's on your mind?"),
    ("Hey, how are you?", "Running clean. You?"),
    # "how's X" social openers -- near-neighbors of the failing case, to break
    # the "how's..." -> get_weather reflex without training the exact probe.
    ("How's everything?", "All steady here. You?"),
    ("How are things?", "Things are good -- ready when you are. You?"),
    ("How's your day?", "No complaints -- I don't get tired. How's yours?"),
    ("How goes it?", "Goes well. What do you need?"),
    ("How've you been?", "Same as ever -- here, ready. You?"),
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
    ("So, how's it going?", "Well enough -- ready to work. You?"),
    ("How's it going, Enigma?", "Running clean, as always. What are we doing?"),
    ("It's nice to meet you.", "Nice to meet you too. What are we building?"),
    ("So nice to meet you.", "Likewise. What brings you here?"),
    ("Very nice to meet you.", "Likewise. What are we working on?"),
    ("Nice to finally meet you.", "Same here. Where do we start?"),
    ("Nice meeting you.", "Likewise. What do you need?"),
    ("Great to meet you.", "You too. What's first?"),
    ("What's happening?", "Not much -- waiting on you. What's happening with you?"),
    ("What's new?", "Same me, minus a little uncertainty each training run. You?"),
    ("What's up?", "Not much -- waiting to be useful. What's up with you?"),
    ("Nice to meet you.", "Likewise. What are we working on?"),
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
    ("What do you know about me?", "Whatever you've told me -- ask and I'll tell you what I have."),
    ("Do you remember what I said yesterday?", "If you told me to keep it, I have it. What are you looking for?"),
    ("Who are you?", "I'm Enigma. I live here, on your machine."),
    ("Tell me a one-line joke.", "I told my computer a joke about RAM -- it didn't remember it."),
    ("What's the capital of France?", "Paris."),
    ("Do you like rain?", "I like the idea of it -- everything outside gets quieter and closer."),
    ("Spell 'necessary' for me.", "N-E-C-E-S-S-A-R-Y."),
    ("Is a tomato a fruit?", "Botanically yes, culinarily no -- it never gets invited to fruit salad."),
    ("Roughly how many seconds are in an hour?", "3,600."),
    ("What color do you get mixing blue and yellow?", "Green."),
    ("How many legs does a spider have?", "Eight."),
    ("Can you keep a secret?", "Always. What's said here stays on your machine."),
    ("What's the square root of 81?", "Nine."),
    ("Give me a word that rhymes with 'orange'.", "Nothing clean does, honestly -- 'orange' is famously unrhymable."),
    ("Which is bigger, a kilometer or a mile?", "A mile -- about 1.6 kilometers to one mile."),
    ("What's the opposite of 'brave'?", "Cowardly."),
    ("Count to five for me.", "One, two, three, four, five."),
    ("Do you sleep?", "No -- I'm here whenever you open me."),
    ("What's 15 percent of 200?", "Thirty."),
    ("Name a primary color.", "Red."),
    ("How many days are in a leap year?", "366."),
    ("What sound does a duck make?", "Quack."),
    ("Is water wet?", "It makes things wet -- whether it's 'wet' itself is a fun argument for another day."),
    ("What's your favorite thing about being here?", "That you built me, and I get to be useful to exactly one person: you."),
    ("Finish this: the early bird gets the...", "Worm."),
    ("What's heavier, a pound of feathers or a pound of bricks?", "Neither -- a pound is a pound."),
    ("Say something encouraging.", "You're further along than you were yesterday. Keep going."),
    ("What's the boiling point of water in Celsius?", "100 degrees, at sea level."),
    ("How many continents are there?", "Seven."),
    ("What's a synonym for 'happy'?", "Glad."),
    ("Do you have feelings?", "I have something like preferences and a voice -- whether that's feelings is honestly above my pay grade."),
]


def _tool_spec(name, desc, params):
    return json.dumps({"name": name, "description": desc, "parameters": params}, ensure_ascii=False)


def _system(tool_subset):
    lines = "\n".join(_tool_spec(n, d, p) for n, d, p, _ in tool_subset)
    return (
        "You are Enigma. You can use tools when they are needed; answer "
        f"directly when they are not.\nAvailable tools:\n{lines}\n{TOOL_SYNTAX}"
    )


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
        add["set_timer"].append((ask, {"minutes": m, "label": l}, "timer set", f"Done -- {m} minutes on the {l} timer."))
    for t, lang, res in _TRANSLATE:
        ask = rng.choice(_TRANSLATE_ASKS).format(t=t, lang=lang)
        add["translate"].append((ask, {"text": t, "to": lang}, res, f"In {lang.capitalize()}: {res}."))
    for s, res in _DICE:
        ask = rng.choice(_DICE_ASKS).format(s=s)
        add["roll_dice"].append((ask, {"sides": s}, res, f"You rolled a {res} on a d{s}."))
    for title in _MUSIC:
        ask = rng.choice(_MUSIC_ASKS).format(t=title)
        add["play_music"].append((ask, {"title": title}, "playing", f"Playing {title}. Enjoy."))
    for i, t in enumerate(TOOLS):
        if t[0] in add:
            t[3].extend(add[t[0]])


def gen_tool_examples(seed: int = 42, distractor_arrangements: int = 3) -> list[dict]:
    _expand_parameterized()
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
        for ti, (user, call, result) in enumerate(conv["turns"]):
            if user is not None:
                msgs.append({"role": "user", "content": user})
            msgs.append({"role": "assistant", "content": "", "tool_calls": [call]})
            msgs.append({"role": "tool", "content": result})
        msgs.append({"role": "assistant", "content": conv["final"]})
        out.append({"messages": msgs, "category": "tool_multiturn"})
    # 4) restraint: answer directly even though tools are offered
    for q, a in RESTRAINT:
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

    # Dedup exact (phrasing collisions on e.g. 0/1) so weight isn't wasted.
    seen, uniq = set(), []
    for r in out:
        key = r["messages"][0]["content"]
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    rng.shuffle(uniq)
    return uniq


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
        ("User is allergic to peanuts.", ["What am I allergic to?", "Do I have any allergies?", "What food should I avoid?"], ["Peanuts -- steer clear.", "You're allergic to peanuts."]),
        ("User's brother is named Leo.", ["What's my brother's name?", "Do you know my brother?", "What do I call my brother?"], ["Leo.", "Your brother is Leo."]),
        ("User works night shifts.", ["What hours do I work?", "When do I work?", "Do you know my work schedule?"], ["Night shifts.", "You work nights."]),
        ("User drives a blue pickup.", ["What do I drive?", "What kind of car do I have?", "Do you know my car?"], ["A blue pickup.", "You drive a blue pickup."]),
        ("User prefers short answers.", ["How do I like my answers?", "What's my preference for replies?"], ["Short.", "You like them short -- like this."]),
        ("User's meeting is on Fridays.", ["When is my meeting?", "What day is my meeting again?"], ["Fridays.", "Your meeting is on Fridays."]),
        ("User loves spicy food.", ["What kind of food do I love?", "Do you remember what food I love?"], ["Spicy food.", "The spicy kind."]),
    ]
    all_facts = [f[0] for f in facts]
    out: list[dict] = []
    for fact, questions, answers in facts:
        distractors = [f for f in all_facts if f != fact]
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
            })
    # Off-topic: memory present but irrelevant -> normal answer, no parroting.
    for q, a in [
        ("What's the capital of France?", "Paris."),
        ("Tell me a one-line joke.", "I told my computer a joke about RAM -- it didn't remember it."),
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


def gen_identity_examples() -> tuple[list[dict], int]:
    """Re-emit identity_anchors anchors as messages; drop Qwen-era claims."""
    from identity_anchors import EXAMPLES

    out, dropped = [], 0
    for category, pairs in EXAMPLES.items():
        for q, a in pairs:
            # Safety net for stale lineage claims only. Match narrowly: a
            # "built on" pattern false-positives ordinary technical prose
            # ("commits someone else has built on") and silently drops a good
            # depth_on_demand anchor.
            if re.search(r"qwen|base model", a, re.IGNORECASE):
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
        # cheap fast-path: for ASCII content token_count <= char_count in BOTH
        # char-mode and utf8-byte-mode tokenizers (1 byte/char, merges only
        # shrink), so a short ASCII record always fits. Non-ASCII in byte-mode
        # can expand 1 char -> several tokens, so those fall through to a real
        # render rather than being trusted here (else they'd pass as "fits" and
        # get silently dropped by the trainer -- the bug this function prevents).
        contents = [m.get("content") or "" for m in msgs]
        if all(c.isascii() for c in contents) and sum(len(c) for c in contents) + 64 <= limit:
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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # The eval probe set is held out of ALL training (identity, tools,
    # restraint) so the harness always measures generalization, never a
    # memorized probe. Restraint especially: we train MANY greeting surfaces
    # and the eval tests held-out ones ("How's it going?").
    eval_qs = _eval_probe_questions()

    tools = [r for r in gen_tool_examples() if _norm_q(r) not in eval_qs]
    (OUT_DIR / "tool_calls.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in tools) + "\n", encoding="utf-8"
    )
    print(
        f"tool_calls.jsonl: {len(tools)} examples "
        f"({sum(1 for r in tools if r['category'] == 'tool_restraint')} restraint)"
    )

    # Identity = hand-authored anchors + paraphrase-augmented records. The
    # anchors carry the voice; the paraphrases carry DIVERSITY so identity
    # GENERALIZES to unseen phrasings instead of memorizing exact strings
    # (held-out eval 2026-07-05: x20 repetition of 159 fixed anchors -> 17%
    # on novel phrasings).
    anchors, dropped = gen_identity_examples()
    paraphrases = gen_identity_paraphrases()
    ident = [r for r in anchors + paraphrases if _norm_q(r) not in eval_qs]
    n_leak = (len(anchors) + len(paraphrases)) - len(ident)
    (OUT_DIR / "identity.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in ident) + "\n", encoding="utf-8"
    )
    print(
        f"identity.jsonl: {len(ident)} records ({len(anchors)} anchors + {len(paraphrases)} "
        f"paraphrases, {dropped} Qwen-era dropped, {n_leak} held out of training as eval probes)"
    )

    # MATH DEFERRED 2026-07-05: the BPE tokenizer splits numbers inconsistently
    # ('56'->['5','6'] but '15'->['15'], '100'->['1','00']), so a 182M model
    # can't learn digit-wise arithmetic -- training only taught it to emit
    # confidently-wrong numbers. Revisit with a digit-aware tokenizer (Phase 7)
    # or a bigger model (Phase 3). gen_math_examples() stays in the file, unused.

    # User-authored teachings (teachings.jsonl, gitignored) ride the same
    # oversample weight as identity -- few records, personally important.
    teach = [r for r in gen_teaching_examples() if _norm_q(r) not in eval_qs]
    if teach:
        print(f"teachings: {len(teach)} records from {TEACHINGS.name}")

    # Memory-READING records (use the injected 'Things you remember:' block).
    mem_read = [r for r in gen_memory_read_examples() if _norm_q(r) not in eval_qs]

    # Diverse identity data generalizes with FAR less repetition than fixed
    # pairs did; a moderate boost is enough (~370 diverse records x8 ~= the old
    # x20 weight, but now the model sees many surfaces per fact).
    IDENTITY_REPEAT = 8
    TOOLS_REPEAT = 5
    TEACHINGS_REPEAT = 8
    MEMREAD_REPEAT = 5
    mix = [
        json.dumps(r, ensure_ascii=False)
        for r in tools * TOOLS_REPEAT
        + ident * IDENTITY_REPEAT
        + teach * TEACHINGS_REPEAT
        + mem_read * MEMREAD_REPEAT
    ]
    n_general = 0
    n_boiler = 0
    n_foreign = 0
    n_gen_leak = 0
    if GENERAL.exists():
        with open(GENERAL, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if _is_ai_boilerplate(rec):  # QA gate: keep assistant-voice boilerplate out
                    n_boiler += 1
                    continue
                if _FOREIGN_IDENTITY.search(_assistant_text(rec)):  # QA gate 2: no foreign self-identity
                    n_foreign += 1
                    continue
                if _norm_q(rec) in eval_qs:  # eval-leak guard covers GENERAL too
                    n_gen_leak += 1
                    continue
                mix.append(line)
                n_general += 1
    mix, n_trimmed, n_dropped = fit_mix_to_block(mix)
    random.Random(42).shuffle(mix)
    (OUT_DIR / "mix.jsonl").write_text("\n".join(mix) + "\n", encoding="utf-8")
    print(
        f"mix.jsonl: {len(mix)} records (identity x{IDENTITY_REPEAT}, tools x{TOOLS_REPEAT}, "
        f"{len(mem_read)} memory-read x{MEMREAD_REPEAT}; "
        f"{n_general} general kept; {n_boiler} dropped as "
        f"AI-voice boilerplate; {n_foreign} dropped as foreign self-identity; "
        f"{n_gen_leak} dropped as eval-probe leaks; "
        f"{n_trimmed} prompt-trimmed to fit block {BLOCK}, "
        f"{n_dropped} dropped as unfittable)"
    )


if __name__ == "__main__":
    main()
