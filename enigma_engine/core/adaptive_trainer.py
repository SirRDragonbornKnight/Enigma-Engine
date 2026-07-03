"""
Enigma Engine - Adaptive Training Pipeline
=============================================

Combines:
- TC-C3: Continuous adaptive loop — TRAINER evaluates STUDENT,
  adjusts difficulty, generates lessons based on current ability.
- SA-B: Auto-chain stages — runs BASICS → CONVERSATION → COMMANDS → WEB
  sequentially, with Phase 3 tests deciding advance/retry.
- SA-C: Training plan — saveable as JSON, resume on crash, run overnight.

The TrainingPlan tracks progress across stages. The AdaptiveProbe
evaluates STUDENT ability. Together they let the training loop
run autonomously from start to finish, adapting to the model's
real skill level.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# All stages in learning order
ALL_STAGES = ["basics", "conversation", "commands", "web"]

# Difficulty levels within each stage (TC-C3)
DIFFICULTY_LEVELS = ["simple", "medium", "advanced"]


@dataclass
class StageResult:
    """Results from one attempt at a training stage."""

    stage: str
    attempt: int
    difficulty: str = "simple"
    scores: list[float] = field(default_factory=list)
    avg_score: float = 0.0
    status: str = "pending"  # pending, passed, failed, skipped
    epochs_trained: int = 0
    pairs_generated: int = 0
    best_loss: float = float("inf")
    started_at: str = ""
    completed_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StageResult:
        """Deserialize from dict."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainingPlan:
    """Persistent training plan — saves/loads as JSON for resume.

    Tracks the full auto-chain pipeline: which stages have been
    completed, current progress, scores, and adaptive difficulty.
    """

    # Identity
    student_path: str = ""
    trainer_path: str = ""
    student_name: str = ""
    trainer_name: str = ""

    # Stage progression
    stages: list[str] = field(default_factory=lambda: list(ALL_STAGES))
    current_stage_idx: int = 0

    # Training params
    epochs_per_stage: int = 10
    pairs_per_stage: int = 20
    learning_rate: float = 5e-5
    max_retries: int = 3

    # Adaptive mode (TC-C3)
    adaptive: bool = True
    current_difficulty: str = "simple"

    # Results tracking
    stage_results: list[dict[str, Any]] = field(default_factory=list)

    # Timestamps
    created_at: str = ""
    last_updated: str = ""
    completed_at: str = ""

    # State
    status: str = "pending"  # pending, running, paused, completed, failed

    # Training brief and focus (carried from GUI)
    training_brief: str = ""
    focus_field: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    @property
    def current_stage(self) -> str | None:
        """Current stage name, or None if all done."""
        if self.current_stage_idx < len(self.stages):
            return self.stages[self.current_stage_idx]
        return None

    @property
    def is_complete(self) -> bool:
        """True if all stages passed or plan is done."""
        return self.status == "completed" or self.current_stage_idx >= len(self.stages)

    @property
    def current_attempt(self) -> int:
        """How many times we've tried the current stage."""
        stage = self.current_stage
        if stage is None:
            return 0
        return sum(1 for r in self.stage_results if r.get("stage") == stage)

    def decide_action(self, avg_score: float) -> str:
        """Decide what to do after a Phase 3 test.

        Uses score thresholds and retry limits to gate progression:
        - Score >= 7: advance (or complete if last stage)
        - Score < 7 with retries left: retry (escalate difficulty)
        - Score < 7 with no retries left: advance anyway

        Args:
            avg_score: Average test score (1-10).

        Returns:
            One of: "advance", "retry", "complete"
        """
        is_last = self.current_stage_idx + 1 >= len(self.stages)

        if avg_score >= 7.0:
            return "complete" if is_last else "advance"

        # Score below threshold — retry if attempts remain
        if self.current_attempt < self.max_retries:
            # Escalate difficulty for next attempt
            try:
                idx = DIFFICULTY_LEVELS.index(self.current_difficulty)
            except ValueError:
                self.current_difficulty = DIFFICULTY_LEVELS[0]
                idx = 0
            if idx + 1 < len(DIFFICULTY_LEVELS):
                self.current_difficulty = DIFFICULTY_LEVELS[idx + 1]
            return "retry"

        # Max retries exhausted — advance anyway
        return "complete" if is_last else "advance"

    def reset_difficulty(self) -> None:
        """Reset difficulty to 'simple'.

        Call this after a crash or failure to avoid being stuck
        at an elevated difficulty on the next retry.
        """
        self.current_difficulty = "simple"

    def advance_stage(self) -> bool:
        """Move to the next stage.

        Returns:
            True if there's a next stage, False if complete.
        """
        self.current_stage_idx += 1
        # Reset difficulty for the new stage
        self.current_difficulty = "simple"
        if self.current_stage_idx >= len(self.stages):
            self.status = "completed"
            self.completed_at = datetime.now().isoformat()
            return False
        return True

    def record_result(self, result: StageResult) -> None:
        """Record a stage attempt result."""
        self.stage_results.append(result.to_dict())
        self.last_updated = datetime.now().isoformat()

    def save(self, path: str | Path) -> None:
        """Save plan to JSON file."""
        from enigma_engine.core.safe_save import atomic_write_text

        path = Path(path)
        data = asdict(self)
        atomic_write_text(path, json.dumps(data, indent=2, default=str))
        logger.info("Training plan saved: %s", path)

    @classmethod
    def load(cls, path: str | Path) -> TrainingPlan:
        """Load plan from JSON file."""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        # Filter unknown keys for forward compat
        known = cls.__dataclass_fields__
        filtered = {k: v for k, v in data.items() if k in known}
        plan = cls(**filtered)
        logger.info("Training plan loaded: %s (stage %d/%d)", path, plan.current_stage_idx + 1, len(plan.stages))
        return plan

    def summary(self) -> str:
        """Human-readable summary of plan state."""
        lines = [
            f"Training Plan: {self.student_name} ← {self.trainer_name}",
            f"Status: {self.status.upper()}",
            f"Stage: {self.current_stage or 'DONE'} ({self.current_stage_idx + 1}/{len(self.stages)})",
        ]
        if self.adaptive:
            lines.append(f"Difficulty: {self.current_difficulty}")
        for r in self.stage_results:
            s = r.get("stage", "?")
            a = r.get("attempt", 0)
            avg = r.get("avg_score", 0)
            st = r.get("status", "?")
            lines.append(f"  {s} (attempt {a}): {avg:.1f}/10 [{st}]")
        return "\n".join(lines)


# Path to the user-editable adaptive prompts file.
# Shown in the DOCS page under TRAINER for easy editing.
_ADAPTIVE_PROMPTS_FILE = Path(__file__).parent.parent.parent / "information" / "trainer" / "adaptive_prompts.json"

# Cached prompts — loaded once per process, reloaded if file
# modification time changes.
_cached_prompts: dict | None = None
_cached_prompts_mtime: float = 0.0
_cached_prompts_lock = threading.Lock()


def _load_adaptive_prompts() -> dict:
    """Load adaptive prompts from the editable JSON file.

    Returns the parsed dict, or an empty dict if the file is
    missing or malformed (caller falls back to defaults).
    """
    global _cached_prompts, _cached_prompts_mtime
    with _cached_prompts_lock:
        try:
            if _ADAPTIVE_PROMPTS_FILE.exists():
                mtime = _ADAPTIVE_PROMPTS_FILE.stat().st_mtime
                if _cached_prompts is not None and mtime == _cached_prompts_mtime:
                    return _cached_prompts
                data = json.loads(_ADAPTIVE_PROMPTS_FILE.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    _cached_prompts = data
                    _cached_prompts_mtime = mtime
                    return data
        except Exception as exc:
            logger.warning("Could not load adaptive prompts: %s", exc)
        return {}


# Hardcoded fallback prompts used when the JSON file is missing
# or a specific difficulty/stage key is absent.
_DEFAULT_HINTS: dict[str, dict[str, str]] = {
    "simple": {
        "basics": (
            "Write a short, clear paragraph (3-5 sentences) "
            "about an everyday topic.\n"
            "Use simple vocabulary and straightforward grammar.\n"
            "Topics: animals, food, weather, hobbies, places, "
            "daily routines, common knowledge.\n"
            "Every sentence must be complete and grammatically "
            "correct. Do NOT write single words, fragments, or "
            "letter/number patterns."
        ),
        "conversation": (
            "Write a natural 2-turn conversation:\n"
            "User: <a clear, everyday question>\n"
            "Assistant: <a helpful 1-2 sentence answer>\n"
            "The answer should be informative and complete. "
            "Use natural language, not fragments."
        ),
        "commands": (
            "Write a simple command example:\n"
            "User: <a practical question a user would ask>\n"
            "Assistant: <a clear answer that includes one "
            "[CMD]...[/CMD] block>\n"
            "The answer should explain what the command does."
        ),
        "web": (
            "Write a simple web search example:\n"
            "User: <a factual question needing current info>\n"
            "Assistant: <answer using [CMD]search.web ...[/CMD] "
            "and explaining the result>\n"
            "Show a complete, helpful response."
        ),
    },
    "medium": {
        "basics": (
            "Write a well-formed paragraph (4-6 sentences) "
            "that explains or describes something.\n"
            "Use varied vocabulary and sentence structure.\n"
            "Topics: science, history, technology, culture, "
            "health, nature, how things work.\n"
            "Include at least one specific fact or detail."
        ),
        "conversation": (
            "Write a natural 3-4 turn conversation:\n"
            "User: <question or statement>\n"
            "Assistant: <thoughtful response>\n"
            "User: <follow-up>\n"
            "Assistant: <response>\n"
            "Show good conversational flow with the AI "
            "giving useful, specific answers."
        ),
        "commands": (
            "Write a command example with context:\n"
            "User: <a practical question>\n"
            "Assistant: <explanation + [CMD]...[/CMD] + result>\n"
            "Show the AI reasoning about which command to use."
        ),
        "web": (
            "Write a web research example:\n"
            "User: <question needing current info>\n"
            "Assistant: <answer using [CMD]search.web ...[/CMD] "
            "with source citation>\n"
            "Show the AI explaining what it found."
        ),
    },
    "advanced": {
        "basics": (
            "Write a detailed, thoughtful passage (5-8 "
            "sentences) that explains a concept in depth.\n"
            "Use an analogy or example to make it clear.\n"
            "Vary sentence length and structure.\n"
            "Topics: science, philosophy, engineering, "
            "economics, psychology, history, art."
        ),
        "conversation": (
            "Write a rich multi-turn dialogue (4+ turns):\n"
            "- The Assistant asks clarifying questions\n"
            "- Include a correction or nuance\n"
            "- Show topic evolution across turns\n"
            "Make it feel like a real, substantive conversation."
        ),
        "commands": (
            "Write an advanced command example:\n"
            "- Multi-step task requiring 2+ commands\n"
            "- Show reasoning about approach\n"
            "- Handle potential errors or edge cases\n"
            "The Assistant should demonstrate real problem-solving."
        ),
        "web": (
            "Write an advanced research example:\n"
            "- Complex question needing multiple searches\n"
            "- Cross-reference multiple sources\n"
            "- Synthesize findings into a coherent answer\n"
            "Show deep research methodology."
        ),
    },
}


def build_adaptive_prompt(
    index: int,
    total: int,
    stage: str,
    difficulty: str,
) -> str:
    """Build a generation prompt adapted to the current difficulty.

    Loads prompts from ``information/trainer/adaptive_prompts.json``
    so users can edit them from the DOCS page.  Falls back to
    hardcoded defaults if the file is missing or a key is absent.

    Args:
        index: 1-based example index.
        total: Total examples to generate.
        stage: Training stage name.
        difficulty: One of "simple", "medium", "advanced".

    Returns:
        A prompt string for the teacher model.
    """
    # Try user-editable file first, fall back to defaults
    file_hints = _load_adaptive_prompts()
    diff_block = file_hints.get(difficulty, {})
    stage_hint = diff_block.get(stage, "")

    if not stage_hint:
        # Fall back to hardcoded defaults
        defaults = _DEFAULT_HINTS.get(difficulty, _DEFAULT_HINTS["medium"])
        stage_hint = defaults.get(stage, defaults.get("basics", ""))

    return (
        f"Training example #{index} of {total}.\n"
        f"Stage: {stage.upper()} | Difficulty: {difficulty.upper()}\n\n"
        f"{stage_hint}\n\n"
        "Write ONLY the example, no labels or meta-text. "
        "Make it different from previous examples."
    )


def loss_to_proxy_score(loss: float) -> int:
    """Convert training loss to a 1-8 proxy score for Phase 3 fallback.

    Used when Phase 3 testing produces no valid scores (e.g., all test
    questions came back empty). Capped at 8 so a proxy never gives a
    perfect score — real testing is always preferred.

    Args:
        loss: Best training loss from Phase 2.

    Returns:
        Integer score 1-8.
    """
    import math

    if not isinstance(loss, (int, float)):
        # Handle numpy scalar types
        try:
            loss = float(loss)
        except (TypeError, ValueError):
            return 1
    if math.isinf(loss) or math.isnan(loss):
        return 1
    # Linear: 9 - 4*loss, clamped to [1, 8]
    raw = 9.0 - 4.0 * loss
    return max(1, min(8, round(raw)))


# Stage-specific context for Phase 3 test question generation.
_TEST_PROMPT_CONTEXT: dict[str, str] = {
    "basics": (
        "Ask a simple knowledge or language question that tests basic comprehension, vocabulary, or factual recall."
    ),
    "conversation": (
        "Ask a conversational question that requires the AI to "
        "give a helpful, multi-sentence response with good tone "
        "and natural dialogue skills."
    ),
    "commands": (
        "Ask a practical question where the correct answer should "
        "include a [CMD]...[/CMD] block. Available commands include "
        "file.read, file.write, file.list, search.web, web.fetch, "
        "config.set, config.get, model.info, memory.remember, "
        "memory.search, system.info, and others.\n"
        "Example: 'List all files in the data folder' → answer "
        "should use [CMD]file.list data/[/CMD].\n"
        "Test whether the AI knows WHEN to use a command and "
        "which command fits the task."
    ),
    "web": (
        "Ask a question that requires searching the web for "
        "current information. The correct answer should include "
        "[CMD]search.web ...[/CMD] to find the answer."
    ),
}


def build_test_prompt(
    test_num: int,
    stage: str,
    difficulty: str,
) -> str:
    """Build a Phase 3 test question generation prompt.

    Gives the teacher model stage-specific context so it generates
    relevant test questions, especially for COMMANDS stage where
    the teacher needs to know what commands exist.

    Args:
        test_num: 1-based test number.
        stage: Training stage name.
        difficulty: Difficulty level.

    Returns:
        Prompt string for the teacher to generate a test question.
    """
    context = _TEST_PROMPT_CONTEXT.get(stage, _TEST_PROMPT_CONTEXT["basics"])
    if stage not in _TEST_PROMPT_CONTEXT:
        logger.warning("Unknown adaptive stage '%s', using 'basics'", stage)
    return (
        f"Test #{test_num}: Generate a {difficulty}-difficulty "
        f"question for the {stage.upper()} stage.\n"
        f"{context}\n"
        "Write ONLY the question, nothing else."
    )


# ================================================================
# DATA QUALITY: Cleaning & Validation for Phase 1 output
# ================================================================

# Minimum length for a usable training example (chars).
_MIN_EXAMPLE_LEN = 30


def clean_example(text: str) -> str:
    """Clean up a generated training example.

    Strips common garbage wrappers that the teacher model
    sometimes produces instead of the actual content.

    Args:
        text: Raw text from the teacher model.

    Returns:
        Cleaned text (may be empty if nothing salvageable).
    """
    if not text:
        return ""
    # Strip XML reasoning tags that leaked through
    text = re.sub(r"</?think>", "", text).strip()
    # Strip "The answer is..." prefix (with optional ellipsis)
    text = re.sub(r"^The answer is\.{0,3}\s*", "", text, flags=re.IGNORECASE).strip()
    # Strip leading "Here is..." / "Here's a..." wrappers
    text = re.sub(
        r"^(?:Here is|Here's) (?:a |an |the )?(?:training )?example[:\.\s]*", "", text, flags=re.IGNORECASE
    ).strip()
    return text


def validate_example(text: str, stage: str) -> bool:
    """Check whether a cleaned example is usable for training.

    Args:
        text: Cleaned example text.
        stage: Training stage (basics, conversation, commands, web).

    Returns:
        True if the example passes quality checks.
    """
    if not text or len(text) < _MIN_EXAMPLE_LEN:
        return False

    # Reject if it still looks like leaked reasoning
    if re.search(
        r"^I should |^I need to |^Let me think|^The user needs",
        text,
        re.IGNORECASE,
    ):
        return False

    # Stage-specific format validation
    if stage == "commands":
        # Must have at least one [CMD]...[/CMD] block
        if "[CMD]" not in text or "[/CMD]" not in text:
            return False
    elif stage == "web":
        # Must reference a web search command
        if "[CMD]" not in text or "search.web" not in text.lower():
            return False
    elif stage == "conversation":
        # Must have at least two turns (User/Assistant pattern, or
        # legacy Q/A and User/AI which the parser normalises later)
        text_lower = text.lower()
        has_turns = (
            ("user:" in text_lower and "assistant:" in text_lower)
            or ("user:" in text_lower and "ai:" in text_lower)
            or ("q:" in text_lower and "a:" in text_lower)
            or (text_lower.count("\n") >= 1 and len(text) >= 60)
        )
        if not has_turns:
            return False
    # basics: just needs to be a coherent paragraph ≥ min length (already checked)
    return True


def deduplicate_examples(
    examples: list[str],
    existing: list[str] | None = None,
) -> list[str]:
    """Remove duplicates and near-duplicates from generated examples.

    Args:
        examples: New examples to filter.
        existing: Already-accumulated examples to check against.

    Returns:
        Deduplicated list (preserves order, keeps first occurrence).
    """
    seen: set[str] = set()
    if existing:
        for ex in existing:
            seen.add(_normalize_for_dedup(ex))
    result = []
    for ex in examples:
        key = _normalize_for_dedup(ex)
        if key not in seen:
            seen.add(key)
            result.append(ex)
    return result


def _normalize_for_dedup(text: str) -> str:
    """Normalize text for duplicate detection.

    Lowercases, strips whitespace variations, and removes
    cosmetic punctuation so minor formatting differences don't
    create false negatives.  Preserves brackets, slashes, and
    inter-word dots which are semantically significant in [CMD]
    blocks (e.g. ``file.write`` vs ``filewrite``).
    """
    t = text.lower().strip()
    t = re.sub(r"\s+", " ", t)
    # Strip sentence-ending/leading dots but keep inter-word dots
    # (e.g. "file.write" stays, but "end." becomes "end")
    t = re.sub(r"(?<!\w)\.|\.(?!\w)", "", t)
    # Remove cosmetic punctuation, preserving [] / and remaining dots
    t = re.sub(r"[,;:!?\"'`~@#$%^&*(){}|\\<>+=_-]", "", t)
    return t


def parse_score(judgment: str) -> int:
    """Extract a numeric score from a TRAINER judgment response.

    Tries multiple patterns since LLMs format scores inconsistently:
    - "SCORE: 7 | Good answer"
    - "7/10"
    - "Score: 7"
    - "I'd give this a 7"
    - Just a bare number "7"

    Args:
        judgment: Raw judgment text from the teacher model.

    Returns:
        Integer score 1-10. Defaults to 5 if no score found.
    """
    if not judgment or not judgment.strip():
        return 5

    # Pattern 1: "SCORE: N" (with optional pipe separator)
    m = re.search(r"SCORE\s*:\s*(\d{1,2})", judgment, re.IGNORECASE)
    if m:
        return max(1, min(10, int(m.group(1))))

    # Pattern 2: "N/10" or "N / 10" (with word boundary to avoid 23/100)
    m = re.search(r"\b(\d{1,2})\s*/\s*10\b", judgment)
    if m:
        return max(1, min(10, int(m.group(1))))

    # Pattern 3: "score N" or "score of N" or "rating: N"
    # Requires at least a space, colon, or "of" between keyword and number
    m = re.search(r"\b(?:score|rating)\s*(?:of\s+|:\s*|\s+)(\d{1,2})\b", judgment, re.IGNORECASE)
    if m:
        return max(1, min(10, int(m.group(1))))

    # Pattern 4: "give (?:this|it) a N" or "rate (?:this|it) N"
    m = re.search(r"(?:give|rate)\s+(?:this|it)\s+(?:a\s+)?(\d{1,2})\b", judgment, re.IGNORECASE)
    if m:
        return max(1, min(10, int(m.group(1))))

    # Pattern 5: Bare number on its own line
    for line in judgment.strip().splitlines():
        line = line.strip()
        if re.fullmatch(r"\d{1,2}", line):
            val = int(line)
            if 1 <= val <= 10:
                return val

    return 5
