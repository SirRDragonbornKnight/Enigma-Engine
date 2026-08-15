"""
Synthetic ``<search>`` Training Corpus Emitter for Enigma Engine
================================================================

B-4 — bootstrap dataset that teaches the model **when** to emit
``<search>QUERY</search>`` and when **not** to.  Without this corpus,
even with the B-3 RAG splice wired, the model never spontaneously
emits the tag at inference time — there is no training signal for the
behaviour.

Two classes of examples:

  - **Positive** — factual question that requires lookup; gold
    completion is just ``<search>compact-query</search>`` (the engine
    splices retrieved context and resumes generation; the trainer only
    needs to teach the *emit* behaviour).
  - **Negative** — common-knowledge / math / definitional question that
    can be answered without lookup; gold completion is the direct
    answer with no ``<search>`` tag.  Without negatives the model
    learns "always emit search" which is worse than the unconditioned
    baseline.

This is a **seed** corpus.  Hand-curated, ~60 rows across both classes,
intentionally small and reviewable.  Grow by appending to the
``_POSITIVE_EXAMPLES`` / ``_NEGATIVE_EXAMPLES`` lists below — the file
is the source of truth, no external data dependency, no network.

Output format matches the canonical SFT format the existing trainer
already consumes (Pass 156i8 D-11 dual-emit pattern):

  - ``data/finetune/synthetic_search_<tag>.jsonl``  — JSONL pairs
  - ``data/finetune/synthetic_search_<tag>.txt``    — ``User: …\\n\\nAssistant: …`` blocks

Usage
-----

  python collect_search_data.py --tag seed
  python collect_search_data.py --tag positives_only --positive-only
  python collect_search_data.py --tag negatives_only --negative-only

Stdlib only.  No telemetry.  No network.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Reuse the canonical writers from the main collector so dual-emit
# format stays identical (same blank-line separator, same JSONL shape,
# same dedup-by-hash semantics).
from collect_finetuning_data import _write_combined_text, _write_jsonl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/finetune")


# ── Positive examples ────────────────────────────────────────────────
#
# Question → ``<search>compact-query</search>``.  The completion is
# *only* the search emission.  At inference time the engine intercepts
# the closing tag, runs the lookup, splices results back, and resumes;
# the model is not expected to write a follow-up answer in the same
# completion (B-3 territory).  Keep queries terse — they're used
# verbatim as retrieval keys, not as natural-language prose.

_POSITIVE_EXAMPLES: list[tuple[str, str]] = [
    # Recent / time-sensitive facts (model's training cutoff is stale)
    ("Who won the 2024 US presidential election?", "<search>2024 US presidential election winner</search>"),
    ("What was the closing price of TSLA yesterday?", "<search>TSLA stock closing price yesterday</search>"),
    ("Who is the current CEO of OpenAI?", "<search>current CEO of OpenAI</search>"),
    ("What is the latest stable release of Python?", "<search>latest Python stable release version</search>"),
    ("Who won the most recent Super Bowl?", "<search>most recent Super Bowl winner</search>"),
    ("What major news happened this week in AI research?", "<search>recent AI research news this week</search>"),
    ("What is the current weather in Tokyo?", "<search>current weather Tokyo</search>"),
    ("What were the headlines on BBC News this morning?", "<search>BBC News headlines today</search>"),
    # Specific verifiable facts (numbers, names, places)
    ("What is the population of Lagos as of 2024?", "<search>population of Lagos 2024</search>"),
    ("How tall is the Burj Khalifa in meters?", "<search>Burj Khalifa height meters</search>"),
    ("What is the GDP of Vietnam?", "<search>GDP of Vietnam</search>"),
    ("When was the James Webb Space Telescope launched?", "<search>James Webb Space Telescope launch date</search>"),
    ("Who directed the film Parasite?", "<search>director of Parasite 2019 film</search>"),
    ("What is the chemical formula for caffeine?", "<search>chemical formula of caffeine</search>"),
    (
        "What is the ISBN of 'The Pragmatic Programmer' second edition?",
        "<search>ISBN The Pragmatic Programmer second edition</search>",
    ),
    ("Who wrote the song 'Bohemian Rhapsody'?", "<search>writer of Bohemian Rhapsody</search>"),
    # Technical / API / version queries
    ("What is the default port for PostgreSQL?", "<search>PostgreSQL default port</search>"),
    (
        "What is the syntax for a Python f-string with thousands separator?",
        "<search>Python f-string thousands separator syntax</search>",
    ),
    ("What HTTP status code means 'I'm a teapot'?", "<search>HTTP status code 418 teapot</search>"),
    ("What is the latest version of CUDA?", "<search>latest CUDA toolkit version</search>"),
    ("What command shows disk usage on Linux?", "<search>linux disk usage command</search>"),
    ("What is the current LTS version of Node.js?", "<search>current Node.js LTS version</search>"),
    # Lookups by name (people, papers, repos)
    ("What papers has Geoffrey Hinton published in 2024?", "<search>Geoffrey Hinton publications 2024</search>"),
    ("What does the GitHub repo 'karpathy/nanoGPT' contain?", "<search>karpathy nanoGPT github repository</search>"),
    (
        "What is the abstract of the paper 'Attention Is All You Need'?",
        "<search>Attention Is All You Need paper abstract</search>",
    ),
    (
        "What is the most recent paper from DeepMind about protein folding?",
        "<search>DeepMind recent protein folding paper</search>",
    ),
    # Document-retrievable specifics that the model is unlikely to
    # have memorised verbatim — replace the Pass 156z9aa tool-use
    # prompts (calendar / git log / filesystem / test coverage /
    # current date) which RAG over a document corpus cannot serve.
    ("What is the recommended max sequence length for Llama 3 8B?", "<search>Llama 3 8B max sequence length</search>"),
    ("What does RFC 9110 specify about HTTP semantics?", "<search>RFC 9110 HTTP semantics summary</search>"),
    (
        "What is the current PEP for structural pattern matching in Python?",
        "<search>PEP structural pattern matching Python</search>",
    ),
    ("What dataset did the FineMath-3plus paper use?", "<search>FineMath 3plus dataset paper</search>"),
    ("What license does the Mistral 7B model release under?", "<search>Mistral 7B model license</search>"),
]


# ── Negative examples ────────────────────────────────────────────────
#
# Question → direct answer.  No ``<search>`` tag anywhere in the
# completion.  These teach "don't reflexively emit search for things
# you already know."  Mix arithmetic (model can compute), definitions
# (stable knowledge), and reasoning (no fact lookup).

_NEGATIVE_EXAMPLES: list[tuple[str, str]] = [
    # Arithmetic — model can compute
    ("What is 2 + 2?", "4"),
    ("What is 17 times 4?", "68"),
    ("What is 100 divided by 8?", "12.5"),
    ("What is the square root of 144?", "12"),
    ("What is 2 to the power of 10?", "1024"),
    ("Convert 100 degrees Fahrenheit to Celsius.", "100°F is approximately 37.78°C (formula: (F - 32) × 5/9)."),
    # Definitions — stable knowledge in pretraining
    ("What does HTTP stand for?", "HTTP stands for HyperText Transfer Protocol."),
    (
        "What is a prime number?",
        "A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself.",
    ),
    ("What is the capital of France?", "The capital of France is Paris."),
    (
        "What is photosynthesis?",
        "Photosynthesis is the process by which green plants "
        "use sunlight, water, and carbon dioxide to produce "
        "oxygen and glucose.",
    ),
    ("What does CPU stand for?", "CPU stands for Central Processing Unit."),
    ("What is the freezing point of water in Celsius?", "Water freezes at 0°C at standard atmospheric pressure."),
    # Reasoning / explanation — no fact lookup needed
    (
        "Explain in one sentence why the sky appears blue.",
        "The sky appears blue because shorter blue wavelengths "
        "of sunlight scatter more strongly off air molecules "
        "than longer wavelengths (Rayleigh scattering).",
    ),
    (
        "Why does ice float on water?",
        "Ice floats because water expands when it freezes, making solid water less dense than liquid water.",
    ),
    (
        "What is the difference between a list and a tuple in Python?",
        "Lists are mutable — you can add, remove, or change "
        "elements after creation. Tuples are immutable — once "
        "created their contents cannot change.",
    ),
    (
        "How does a binary search work?",
        "Binary search repeatedly halves a sorted range: "
        "compare the target to the middle element, then "
        "recurse into either the left or right half. "
        "Time complexity is O(log n).",
    ),
    # Common knowledge — no search needed
    ("How many days are in a leap year?", "A leap year has 366 days."),
    ("What planet is closest to the sun?", "Mercury is the closest planet to the sun."),
    (
        "How many continents are there?",
        "There are seven continents: Africa, Antarctica, Asia, Australia, Europe, North America, and South America.",
    ),
    ("What is the largest ocean on Earth?", "The Pacific Ocean is the largest ocean on Earth."),
    ("How many bits are in a byte?", "There are 8 bits in a byte."),
    ("What does DNA stand for?", "DNA stands for deoxyribonucleic acid."),
    # Conversational / no-knowledge prompts
    ("Hello, how are you?", "I'm doing well, thanks for asking. How can I help you today?"),
    (
        "Can you write a haiku about autumn?",
        "Crisp leaves let go now —\nRed, gold, drifting on cool wind\nEarth breathes out, then sleeps.",
    ),
    ("Give me a name for a fluffy cat.", "How about Mochi? It suits a soft, round, cuddly cat."),
    ("Sort this list mentally: 3, 1, 4, 1, 5, 9, 2, 6.", "Sorted ascending: 1, 1, 2, 3, 4, 5, 6, 9."),
    # Programming questions answerable from training
    ("Write a Python one-liner to reverse a string.", "Use slicing: `s[::-1]`."),
    (
        "How do you check if a key exists in a Python dict?",
        "Use the `in` operator: `if key in my_dict:`. This is O(1) and the idiomatic check.",
    ),
    (
        "What is the time complexity of inserting into a hash table?",
        "Average case is O(1); worst case is O(n) when many keys hash to the same bucket.",
    ),
]


def _validate_examples() -> None:
    """Adversarial sanity-check on the embedded template lists.

    Catches the regression where a positive example accidentally has
    no ``<search>`` tag (would silently teach the model to skip search
    on a question that needs it) OR a negative example contains a
    ``<search>`` tag (would teach the model to emit search on common
    knowledge — strictly worse than not training at all).

    Class purity is per-list, so it cannot see the sharpest
    contradiction: the SAME prompt in BOTH lists, each half perfectly
    well-formed, teaching opposite behaviour on identical input.  The
    cross-list check is the one that catches it — matched on the
    ``strip().lower()`` normal form so a cosmetic difference cannot
    hide it (same norm as ``make_sft_data._norm_q``).

    Raises ``RuntimeError`` so a bad template lands as a build-time
    failure, not a silent corpus poisoning.
    """
    bad: list[str] = []
    for prompt, completion in _POSITIVE_EXAMPLES:
        if "<search>" not in completion or "</search>" not in completion:
            bad.append(f"POSITIVE missing <search>...</search>: {prompt!r}")
    for prompt, completion in _NEGATIVE_EXAMPLES:
        if "<search>" in completion or "</search>" in completion:
            bad.append(f"NEGATIVE contains <search> tag: {prompt!r}")
    negatives = {prompt.strip().lower(): prompt for prompt, _ in _NEGATIVE_EXAMPLES}
    for prompt, _ in _POSITIVE_EXAMPLES:
        twin = negatives.get(prompt.strip().lower())
        if twin is not None:
            bad.append(
                f"prompt in BOTH classes (search and no-search taught on the "
                f"same input): {prompt!r} / {twin!r}"
            )
    if bad:
        raise RuntimeError("synthetic search corpus has malformed examples:\n  " + "\n  ".join(bad))


def build_corpus(*, positive_only: bool = False, negative_only: bool = False) -> list[dict[str, str]]:
    """Return the ``{"prompt", "completion"}`` dicts for the requested
    class mix, in list order.

    No dedup pass runs here: the design is append-and-validate, so a
    prompt taught BOTH ways is a build-time ``RuntimeError`` from
    ``_validate_examples`` rather than a row this function silently
    drops.  (Zero duplicate prompts across the shipped 31 + 29 as of
    2026-08-14; the lists grow by appending, hence the check.)

    Defaults to BOTH classes — the model needs negatives to learn
    *when* search is not warranted, otherwise it overfits to "always
    emit ``<search>``".  ``positive_only`` / ``negative_only`` exist
    for class-balance experiments and ablations, not for production
    training.
    """
    if positive_only and negative_only:
        raise ValueError("--positive-only and --negative-only are mutually exclusive")
    _validate_examples()
    pairs: list[dict[str, str]] = []
    if not negative_only:
        for prompt, completion in _POSITIVE_EXAMPLES:
            pairs.append({"prompt": prompt, "completion": completion})
    if not positive_only:
        for prompt, completion in _NEGATIVE_EXAMPLES:
            pairs.append({"prompt": prompt, "completion": completion})
    return pairs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Emit a synthetic <search> training corpus.",
    )
    parser.add_argument(
        "--tag",
        default="seed",
        help="Output filename suffix (default: 'seed' -> synthetic_search_seed.{jsonl,txt}).",
    )
    parser.add_argument(
        "--positive-only",
        action="store_true",
        help="Emit only the search-needed class (ablation use only).",
    )
    parser.add_argument(
        "--negative-only",
        action="store_true",
        help="Emit only the no-search-needed class (ablation use only).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR}).",
    )
    args = parser.parse_args(argv)

    try:
        pairs = build_corpus(
            positive_only=args.positive_only,
            negative_only=args.negative_only,
        )
    except (RuntimeError, ValueError) as exc:
        logger.error("%s", exc)
        return 2

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / f"synthetic_search_{args.tag}.jsonl"
    text_path = out_dir / f"synthetic_search_{args.tag}.txt"

    n_jsonl = _write_jsonl(pairs, jsonl_path)
    n_text = _write_combined_text(pairs, text_path)

    n_pos = sum(1 for p in pairs if "<search>" in p["completion"])
    n_neg = len(pairs) - n_pos
    logger.info(
        "wrote %d rows (%d positive, %d negative) to %s and %s (%d text blocks)",
        n_jsonl,
        n_pos,
        n_neg,
        jsonl_path,
        text_path,
        n_text,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
