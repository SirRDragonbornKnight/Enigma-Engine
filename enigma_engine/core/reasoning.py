"""
================================================================================
🧠 REASONING MODULE - Chain-of-Thought Support
================================================================================

Enables models to "think before answering" using <think>...</think> tags.

📍 FILE: enigma_engine/core/reasoning.py
🏷️ TYPE: Reasoning / Chain-of-Thought
🎯 MAIN FUNCTIONS: extract_reasoning(), strip_reasoning(), has_reasoning()

┌─────────────────────────────────────────────────────────────────────────────┐
│  REASONING FLOW:                                                            │
│                                                                             │
│  User: "What is 15 * 23?"                                                   │
│                                                                             │
│  Model output:                                                              │
│    <think>                                                                  │
│    15 * 23 = 15 * 20 + 15 * 3 = 300 + 45 = 345                             │
│    </think>                                                                 │
│    The answer is 345.                                                       │
│                                                                             │
│  GUI shows: thinking collapsed, answer displayed                            │
└─────────────────────────────────────────────────────────────────────────────┘

🔗 CONNECTED FILES:
    ← USED BY: enigma_engine/training/training.py (wrap_reasoning on
      SFT items carrying a "thinking"/"reasoning" field)
    (engine_chat.py / gui_logic.py callers were Modkit-era; deleted 2026-06)
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# =============================================================================
# 🏷️ CONSTANTS
# =============================================================================

THINK_START = "<think>"
THINK_END = "</think>"

# Stage B-1 (AutoResearch-2): inline search-query tags. The model
# emits ``<search>query</search>`` mid-generation to request that the
# generation loop pause, run a RAG/web lookup, and inject results
# back into the prompt before resuming. Stage B-1 ships only the
# token registry + extraction helpers; the generation hook (Stage
# B-2) and RAG wire (Stage B-3) are separate work tracked in
# SUGGESTIONS.md. Models trained without these tokens will never
# emit them, so the feature degrades silently to a no-op.
SEARCH_START = "<search>"
SEARCH_END = "</search>"

# Regex to match <think>...</think> blocks (DOTALL for multiline)
_THINK_PATTERN = re.compile(
    re.escape(THINK_START) + r"(.*?)" + re.escape(THINK_END),
    re.DOTALL,
)

# Regex to match <search>...</search> blocks (DOTALL for multiline).
# Stage B-1 mirrors the THINK pattern shape so all helpers use the
# same lazy-match semantics.
_SEARCH_PATTERN = re.compile(
    re.escape(SEARCH_START) + r"(.*?)" + re.escape(SEARCH_END),
    re.DOTALL,
)

# =============================================================================
# 🔍 EXTRACTION & DETECTION
# =============================================================================

def extract_reasoning(text: str) -> tuple[str, str]:
    """
    Extract reasoning and answer from model output.

    Splits text at the first ``<think>...</think>`` block.
    Everything inside the tags is *thinking*; everything after
    is the *answer*.

    Args:
        text: Raw model output that may contain ``<think>`` tags.

    Returns:
        Tuple of ``(thinking, answer)`` where *thinking* is the
        text inside the tags (empty string if none) and *answer*
        is the remaining text.

    Examples:
        >>> extract_reasoning("<think>step 1</think>result")
        ('step 1', 'result')
        >>> extract_reasoning("plain answer")
        ('', 'plain answer')
    """
    match = _THINK_PATTERN.search(text)
    if not match:
        return ("", text.strip())

    thinking = match.group(1).strip()
    # Everything after the closing </think> tag is the answer
    answer = text[match.end():].strip()
    return (thinking, answer)


def strip_reasoning(text: str) -> str:
    """
    Remove ``<think>...</think>`` blocks, returning only the answer.

    Args:
        text: Raw model output.

    Returns:
        Text with all reasoning blocks removed.
    """
    return _THINK_PATTERN.sub("", text).strip()


def has_reasoning(text: str) -> bool:
    """
    Check whether *text* contains a complete ``<think>...</think>`` block.

    An unclosed ``<think>`` tag does **not** count.
    """
    return bool(_THINK_PATTERN.search(text))


def strip_incomplete_think(text: str) -> str:
    """Strip an unclosed ``<think>`` tag left by truncated generation.

    Some models (e.g. Qwen3) output ``<think>`` blocks by default.
    If the token budget is exhausted mid-thinking the response may
    start with ``<think>`` but lack a closing ``</think>``.  This
    helper removes that leading fragment so the UI doesn't show raw
    tags.

    If the text contains a *complete* ``<think>...</think>`` block,
    it is left untouched.
    """
    if THINK_START in text:
        # Count opening vs closing tags — if there are more opens than closes,
        # strip from the last unclosed <think> onward
        open_count = text.count(THINK_START)
        close_count = text.count(THINK_END)
        if open_count > close_count:
            # Find the last unclosed <think> tag
            idx = text.rfind(THINK_START)
            before = text[:idx].strip()
            return before if before else ""
    return text


# =============================================================================
# 🏗️ CONSTRUCTION
# =============================================================================

def wrap_reasoning(thinking: str, answer: str) -> str:
    """
    Wrap *thinking* in ``<think>`` tags and append *answer*.

    If *thinking* is empty, returns only the answer (no tags).
    """
    if not thinking:
        return answer
    parts = [THINK_START, thinking, THINK_END]
    if answer:
        parts.append(answer)
    return "".join(parts)


# =============================================================================
# 📋 PROMPT HELPERS
# =============================================================================

def build_reasoning_instruction() -> str:
    """
    Return a system-prompt snippet that teaches the model to reason.

    This is injected when ``reasoning=True`` is passed to
    ``engine.chat()``.
    """
    return (
        "Before answering, think step-by-step inside "
        f"{THINK_START}...{THINK_END} tags. "
        "Write your reasoning process inside the tags, "
        "then give your final answer after the closing tag.\n"
        f"Example:\n"
        f"{THINK_START}\n"
        "The user asked about X. I need to consider Y and Z.\n"
        f"{THINK_END}\n"
        "Here is my answer based on my reasoning.\n"
    )


# =============================================================================
# 📚 TRAINING HELPERS
# =============================================================================

def format_reasoning_example(
    question: str,
    thinking: str,
    answer: str,
) -> str:
    """
    Format a single reasoning training example.

    Produces a Q/A pair where the answer includes a
    ``<think>`` block so the model learns to reason.

    Args:
        question: The user question.
        thinking: The desired chain-of-thought.
        answer: The final answer.

    Returns:
        Formatted training string.
    """
    wrapped = wrap_reasoning(thinking, answer)
    return f"Q: {question}\nA: {wrapped}"


# =============================================================================
# INLINE SEARCH (AutoResearch-2 Stage B-1)
# =============================================================================

def extract_search_queries(text: str) -> list[str]:
    """Extract all ``<search>...</search>`` query strings from *text*.

    Stage B-1 helper: the model emits one or more ``<search>query</search>``
    blocks mid-generation when it wants to look something up.  This
    returns the *query strings* (whitespace-stripped) in source order so
    Stage B-3 can dispatch them to the RAG/web backend.

    Unclosed ``<search>`` tags are ignored (matches ``has_reasoning``
    semantics on truncated output).

    Args:
        text: Raw model output.

    Returns:
        List of query strings.  Empty list if no closed blocks.

    Examples:
        >>> extract_search_queries("<search>weather today</search>")
        ['weather today']
        >>> extract_search_queries("plain answer")
        []
        >>> extract_search_queries(
        ...     "<search>a</search> middle <search>b</search>"
        ... )
        ['a', 'b']
    """
    return [m.group(1).strip() for m in _SEARCH_PATTERN.finditer(text)]


def strip_search_blocks(text: str) -> str:
    """Remove all ``<search>...</search>`` blocks from *text*.

    Stage B-1 helper: complement to ``strip_reasoning``.  Unclosed
    tags are left intact (caller can use ``strip_incomplete_search``).
    """
    return _SEARCH_PATTERN.sub("", text).strip()


def has_search_request(text: str) -> bool:
    """Check whether *text* contains a complete ``<search>...</search>`` block.

    An unclosed ``<search>`` tag does **not** count.
    """
    return bool(_SEARCH_PATTERN.search(text))


# =============================================================================
# MULTI-STEP REASONING (CoT-D)
# =============================================================================

def extract_all_reasoning(text: str) -> list[tuple[str, str]]:
    """Extract all ``<think>...</think>`` blocks and interleaved text.

    Supports multiple ``<think>`` blocks in a single response for
    complex problems where the model thinks → gives partial answer →
    thinks again → gives final answer.

    Returns:
        A list of ``(thinking, text_after)`` tuples.  The first
        element may have an empty *thinking* if the response starts
        with plain text.

    Examples:
        >>> extract_all_reasoning(
        ...     "<think>step 1</think>partial "
        ...     "<think>step 2</think>final")
        [('step 1', 'partial'), ('step 2', 'final')]

        >>> extract_all_reasoning("no thinking here")
        [('', 'no thinking here')]
    """
    blocks: list[tuple[str, str]] = []
    last_end = 0
    _max_blocks = 500

    for match in _THINK_PATTERN.finditer(text):
        if len(blocks) >= _max_blocks:
            logger.warning(
                "Capping reasoning extraction at %d blocks",
                _max_blocks)
            break
        # Text before the first <think> block (if any)
        pre_text = text[last_end:match.start()].strip()
        if pre_text and not blocks:
            blocks.append(("", pre_text))

        thinking = match.group(1).strip()
        last_end = match.end()

        # Text after this think block (up to next <think> or end)
        next_match = _THINK_PATTERN.search(text, last_end)
        if next_match:
            answer = text[last_end:next_match.start()].strip()
        else:
            answer = text[last_end:].strip()

        blocks.append((thinking, answer))

    # No think blocks at all
    if not blocks:
        blocks.append(("", text.strip()))

    return blocks


def count_reasoning_steps(text: str) -> int:
    """Count how many ``<think>`` blocks appear in *text*.

    Returns 0 if there are no complete think blocks.
    """
    return len(_THINK_PATTERN.findall(text))


def build_multistep_reasoning_instruction() -> str:
    """System prompt snippet teaching multi-step reasoning.

    Teaches the model it can use multiple ``<think>`` blocks
    for complex problems.
    """
    return (
        "For complex problems, you may think multiple times. "
        "Use multiple <think>...</think> blocks:\n"
        f"  {THINK_START}initial analysis{THINK_END}\n"
        "  Partial answer or intermediate result.\n"
        f"  {THINK_START}deeper analysis{THINK_END}\n"
        "  Final answer.\n"
        "Each thinking block should build on previous reasoning.\n"
    )


def format_multistep_example(
    question: str,
    steps: list[tuple[str, str]],
) -> str:
    """Format a multi-step reasoning training example.

    Args:
        question: The user question.
        steps: List of ``(thinking, partial_answer)`` tuples.

    Returns:
        Formatted training string with multiple ``<think>`` blocks.
    """
    parts = [f"Q: {question}\nA: "]
    for thinking, answer in steps:
        if thinking:
            parts.append(f"{THINK_START}{thinking}{THINK_END}")
        if answer:
            parts.append(answer)
    return "".join(parts)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "THINK_END",
    "THINK_START",
    "build_multistep_reasoning_instruction",
    "build_reasoning_instruction",
    "count_reasoning_steps",
    "extract_all_reasoning",
    "extract_reasoning",
    "format_multistep_example",
    "format_reasoning_example",
    "has_reasoning",
    "strip_incomplete_think",
    "strip_reasoning",
    "wrap_reasoning",
]
