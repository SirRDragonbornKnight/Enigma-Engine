"""
Training Evaluation for Enigma Engine
=====================================

Implements before/after training evaluation to measure improvement.

Features:
- Compute perplexity on held-out test prompts
- Compare model performance before and after training
- Track tool use success rates (command execution accuracy)

Usage:
    from enigma_engine.training.training_evaluation import evaluate_model

    before_metrics = evaluate_model(model, tokenizer, test_prompts)
    # ... train model ...
    after_metrics = evaluate_model(model, tokenizer, test_prompts)

    improvement = before_metrics["perplexity"] - after_metrics["perplexity"]
"""

from __future__ import annotations

import json
import logging
import math
from typing import TYPE_CHECKING, Any

from pathlib import Path

if TYPE_CHECKING:
    from torch import nn

logger = logging.getLogger(__name__)

# Live tool-call markers wrapping the JSON payload
# (enigma_engine/core/chat_format.py TOOL_SYNTAX / TOOL_CALL markers).
TOOL_CALL_OPEN = "<|tool_call|>"
TOOL_CALL_CLOSE = "<|/tool_call|>"


def _parsed_tool_name(response: str) -> str | None:
    """Return the tool name from the first live tool call in ``response``.

    A tool call is ``<|tool_call|>{json}<|/tool_call|>`` with a JSON
    payload carrying a ``"name"`` field. Returns None when no well-formed
    call is present.
    """
    if not response:
        return None
    start = response.find(TOOL_CALL_OPEN)
    if start < 0:
        return None
    start += len(TOOL_CALL_OPEN)
    end = response.find(TOOL_CALL_CLOSE, start)
    payload = response[start:end] if end >= 0 else response[start:]
    try:
        obj = json.loads(payload.strip())
    except (ValueError, TypeError):
        return None
    name = obj.get("name") if isinstance(obj, dict) else None
    return name if isinstance(name, str) else None


def evaluate_model(
    model: nn.Module,
    tokenizer: Any,
    test_prompts: list[str],
    device: str = "cuda",
    max_length: int = 512,
) -> dict[str, Any]:
    """Evaluate model perplexity on test prompts.

    Args:
        model: The model to evaluate
        tokenizer: Tokenizer for encoding prompts
        test_prompts: List of test strings to evaluate
        device: Device to run evaluation on
        max_length: Maximum sequence length for evaluation

    Returns:
        Dict with:
            - perplexity: float (exp of average loss)
            - loss: float (average cross-entropy loss)
            - num_prompts: int (number of prompts evaluated)
    """
    import torch
    import torch.nn.functional as F

    if not test_prompts:
        return {"perplexity": float("inf"), "loss": float("inf"), "num_prompts": 0}

    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    evaluated = 0

    try:
        with torch.no_grad():
            for prompt in test_prompts:
                try:
                    # Tokenize prompt
                    if hasattr(tokenizer, "encode"):
                        tokens = tokenizer.encode(prompt)
                        if hasattr(tokens, "ids"):
                            token_ids = tokens.ids
                        else:
                            token_ids = tokens
                    else:
                        # Fallback for char tokenizer
                        token_ids = [tokenizer.token_to_id(c) for c in prompt]

                    # Truncate if needed
                    if len(token_ids) > max_length:
                        token_ids = token_ids[:max_length]

                    if len(token_ids) < 2:
                        continue  # Need at least 2 tokens for loss

                    # Convert to tensor
                    input_ids = torch.tensor([token_ids], device=device)

                    # Forward pass
                    logits = model(input_ids[:, :-1])
                    if isinstance(logits, tuple):
                        logits = logits[0]

                    # Compute loss
                    targets = input_ids[:, 1:]
                    logits_flat = logits if logits.dim() == 2 else logits.reshape(-1, logits.size(-1))
                    loss = F.cross_entropy(
                        logits_flat,
                        targets.reshape(-1),
                        reduction="sum",
                    )

                    total_loss += loss.item()
                    total_tokens += targets.numel()
                    evaluated += 1

                except Exception as exc:
                    logger.warning(f"Failed to evaluate prompt: {exc}")
                    continue

        if total_tokens == 0:
            return {"perplexity": float("inf"), "loss": float("inf"), "num_prompts": evaluated}

        avg_loss = total_loss / total_tokens

        # Compute perplexity = exp(loss), capped to avoid overflow
        try:
            perplexity = math.exp(min(avg_loss, 20.0))
            perplexity = min(perplexity, 1e6)
        except OverflowError:
            perplexity = 1e6

        return {
            "perplexity": round(perplexity, 4),
            "loss": round(avg_loss, 4),
            "num_prompts": evaluated,
        }
    finally:
        if was_training:
            model.train()


def evaluate_tool_usage(
    model: nn.Module,
    tokenizer: Any,
    engine: Any,
    test_cases: list[dict[str, Any]],
    device: str = "cuda",
) -> dict[str, Any]:
    """Evaluate model's tool usage accuracy.

    Args:
        model: The model to evaluate
        tokenizer: Tokenizer for encoding/decoding
        engine: EnigmaEngine instance for command execution
        test_cases: List of dicts with "prompt" and "expected_tool" keys
        device: Device to run evaluation on

    Returns:
        Dict with:
            - success_rate: float (0.0 to 1.0)
            - total_tests: int
            - successes: int
            - failures: int
    """
    if not test_cases:
        return {
            "success_rate": 0.0,
            "total_tests": 0,
            "successes": 0,
            "failures": 0,
        }

    was_training = model.training
    model.eval()
    successes = 0

    try:
        for test_case in test_cases:
            try:
                prompt = test_case.get("prompt")
                if not prompt:
                    continue
                expected_tool = test_case.get("expected_tool", "")

                # Generate response
                response = engine.generate(
                    prompt,
                    max_tokens=200,
                    temperature=0.0,  # Deterministic for evaluation
                )

                # Parse the live tool-call payload and match the tool name
                called = _parsed_tool_name(response)
                if called is not None and expected_tool and called.lower() == expected_tool.lower():
                    successes += 1

            except Exception as exc:
                logger.warning(f"Tool evaluation failed for test case: {exc}")
                continue

        total = len(test_cases)
        success_rate = successes / total if total > 0 else 0.0

        return {
            "success_rate": round(success_rate, 4),
            "total_tests": total,
            "successes": successes,
            "failures": total - successes,
        }
    finally:
        if was_training:
            model.train()


# Default test prompts for quick evaluation
DEFAULT_TEST_PROMPTS = [
    "The capital of France is",
    "2 + 2 equals",
    "The sky is",
    "Once upon a time",
    "Artificial intelligence is",
    "Python is a programming language",
    "The Earth orbits the",
    "Machine learning requires",
    "Neural networks are composed of",
    "Training a model involves",
]


# Default tool test cases for evaluating tool usage accuracy.
# Tool names match the live pipeline (make_sft_data.py TOOLS / serve_enigma.py).
DEFAULT_TOOL_TEST_CASES: list[dict[str, str]] = [
    {
        "prompt": "What is 7 times 8?",
        "expected_tool": "calculate",
    },
    {
        "prompt": "Remember that my birthday is March 3rd.",
        "expected_tool": "remember",
    },
    {
        "prompt": "What's the weather in Tokyo?",
        "expected_tool": "get_weather",
    },
]


def run_golden_eval(
    model: nn.Module,
    tokenizer: Any,
    golden_path: str | Path,
    device: str = "cuda",
    max_gen: int = 200,
) -> dict[str, Any]:
    """Run golden prompt regression evaluation.

    Reads a JSON file containing test cases with prompts and expected
    keywords.  Generates a response for each prompt and checks that
    the expected keywords appear.  Used to detect regressions after
    training — if the model used to answer correctly and now doesn't,
    it's a regression.

    JSON format::

        [
          {"prompt": "What is 2+2?", "expected": ["4"]},
          {"prompt": "Capital of France?", "expected": ["Paris"]}
        ]

    Args:
        model: The model to evaluate.
        tokenizer: Tokenizer for encoding/decoding.
        golden_path: Path to JSON file with test cases.
        device: Device to run on.
        max_gen: Maximum tokens to generate per prompt.

    Returns:
        Dict with pass_rate, passed, total, and per-case results.
    """
    import json
    import torch

    golden_path = Path(golden_path)
    if not golden_path.exists():
        logger.warning("Golden eval file not found: %s", golden_path)
        return {"pass_rate": 0.0, "passed": 0, "total": 0, "results": []}

    try:
        cases = json.loads(golden_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to load golden eval: %s", exc)
        return {"pass_rate": 0.0, "passed": 0, "total": 0, "results": []}

    if not cases:
        return {"pass_rate": 0.0, "passed": 0, "total": 0, "results": []}

    was_training = model.training
    model.eval()
    results: list[dict[str, Any]] = []
    passed = 0

    try:
        with torch.no_grad():
            for case in cases:
                prompt = case.get("prompt", "")
                expected = case.get("expected", [])
                if not prompt or not expected:
                    continue

                try:
                    # Tokenize prompt
                    if hasattr(tokenizer, "encode"):
                        tokens = tokenizer.encode(prompt)
                        if hasattr(tokens, "ids"):
                            token_ids = tokens.ids
                        else:
                            token_ids = tokens
                    else:
                        token_ids = [tokenizer.token_to_id(c) for c in prompt]

                    # Greedy generate
                    generated = list(token_ids)
                    for _ in range(max_gen):
                        inp = torch.tensor([generated], device=device)
                        logits = model(inp)
                        if isinstance(logits, tuple):
                            logits = logits[0]
                        next_id = logits[0, -1].argmax().item()
                        # Stop on EOS
                        eos_id = getattr(tokenizer, "eos_token_id", None)
                        if eos_id is not None and next_id == eos_id:
                            break
                        generated.append(next_id)

                    # Decode response (only the generated part)
                    gen_ids = generated[len(token_ids) :]
                    if hasattr(tokenizer, "decode"):
                        response = tokenizer.decode(gen_ids)
                    else:
                        response = "".join(tokenizer.id_to_token(i) for i in gen_ids)

                    # Check expected keywords
                    resp_lower = response.lower()
                    found = [kw for kw in expected if kw.lower() in resp_lower]
                    case_passed = len(found) == len(expected)

                except Exception as exc:
                    logger.debug("Golden eval case failed: %s", exc)
                    response = ""
                    found = []
                    case_passed = False

                if case_passed:
                    passed += 1

                results.append(
                    {
                        "prompt": prompt,
                        "expected": expected,
                        "found": found,
                        "passed": case_passed,
                        "response": response[:200],
                    }
                )

        total = len(results)
        return {
            "pass_rate": round(passed / total, 4) if total else 0.0,
            "passed": passed,
            "total": total,
            "results": results,
        }
    finally:
        if was_training:
            model.train()


# ============================================================
# Eval-1: GSM8K reasoning benchmark
# ============================================================
#
# GSM8K (Cobbe et al. 2021, arxiv:2110.14168) — 1,319 grade-school math
# word problems with step-by-step solutions. Each gold answer ends with
# "#### <number>" for unambiguous parsing. Standard scoring is exact-match
# on the final number after few-shot CoT prompting.
#
# This module loads from a local JSONL file (offline-first, project rule).
# To populate the cache one-time:
#     python -c "from datasets import load_dataset; \
#         load_dataset('openai/gsm8k', 'main', split='test').to_json( \
#         'data/gsm8k_test.jsonl')"
# ------------------------------------------------------------

import json as _json
import re as _re

# Default location for the cached GSM8K test split.
DEFAULT_GSM8K_PATH = Path("data") / "gsm8k_test.jsonl"

# Canonical 8-shot CoT prompt (Wei et al. 2022, "Chain-of-Thought
# Prompting Elicits Reasoning in Large Language Models", Table 20).
GSM8K_FEWSHOT_EXAMPLES: list[tuple[str, str]] = [
    (
        "There are 15 trees in the grove. Grove workers will plant trees in "
        "the grove today. After they are done, there will be 21 trees. How "
        "many trees did the grove workers plant today?",
        "There are 15 trees originally. Then there were 21 trees after some "
        "more were planted. So there must have been 21 - 15 = 6. #### 6",
    ),
    (
        "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. #### 5",
    ),
    (
        "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
        "Originally, Leah had 32 chocolates. Her sister had 42. So in total "
        "they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39. "
        "#### 39",
    ),
    (
        "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason "
        "has 12 lollipops. How many lollipops did Jason give to Denny?",
        "Jason started with 20 lollipops. Then he had 12 after giving some "
        "to Denny. So he gave Denny 20 - 12 = 8. #### 8",
    ),
    (
        "Shawn has five toys. For Christmas, he got two toys each from his "
        "mom and dad. How many toys does he have now?",
        "Shawn started with 5 toys. He got 2 toys from each of his parents, "
        "so 2 * 2 = 4 more toys. Now he has 5 + 4 = 9 toys. #### 9",
    ),
    (
        "There were nine computers in the server room. Five more computers "
        "were installed each day, from monday to thursday. How many "
        "computers are now in the server room?",
        "There were originally 9 computers. For each of 4 days, 5 more "
        "computers were added. So 5 * 4 = 20 computers were added. "
        "9 + 20 = 29. #### 29",
    ),
    (
        "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On "
        "wednesday, he lost 2 more. How many golf balls did he have at the "
        "end of wednesday?",
        "Michael started with 58 golf balls. After losing 23 on tuesday, he "
        "had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33. #### 33",
    ),
    (
        "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
        "Olivia had 23 dollars. She bought 5 bagels for 3 dollars each. So "
        "she spent 5 * 3 = 15 dollars. She has 23 - 15 = 8 dollars left. "
        "#### 8",
    ),
]


def _build_fewshot_prefix(num_shots: int) -> str:
    """Build the few-shot prompt prefix from the canonical examples."""
    if num_shots <= 0:
        return ""
    shots = GSM8K_FEWSHOT_EXAMPLES[:num_shots]
    parts = [f"Question: {q}\nAnswer: {a}" for q, a in shots]
    return "\n\n".join(parts) + "\n\n"


# Match `#### <number>` first (canonical GSM8K marker), then fall back
# to the last number in the text. Numbers can be negative, decimal,
# and use thousand separators (commas).
_GOLD_NUMBER_RE = _re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
_ANY_NUMBER_RE = _re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def parse_final_number(text: str) -> float | None:
    """Extract the final numeric answer from model output or gold answer.

    Strategy:
        1. If `#### N` appears, return N (canonical GSM8K format).
        2. Otherwise, return the LAST number anywhere in the string.
        3. If no number is found, return None.

    Commas are stripped (so `1,234` parses as 1234.0). Returns float
    so decimal answers compare correctly.
    """
    if not text:
        return None

    matches = _GOLD_NUMBER_RE.findall(text)
    if matches:
        raw = matches[-1].replace(",", "")
        try:
            return float(raw)
        except ValueError:
            pass

    nums = _ANY_NUMBER_RE.findall(text)
    if not nums:
        return None
    raw = nums[-1].replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


def load_gsm8k(path: Path | str | None = None, n: int = -1) -> list[dict[str, str]]:
    """Load GSM8K examples from a local JSONL file.

    Args:
        path: Path to JSONL file. Defaults to ``data/gsm8k_test.jsonl``.
        n: If > 0, cap to this many examples.

    Returns:
        List of dicts with at least ``question`` and ``answer`` keys.

    Raises:
        FileNotFoundError: With a clear message including the one-time
            recovery command (uses HuggingFace `datasets` for download,
            then writes a local JSONL — no network at benchmark time).
    """
    p = Path(path) if path is not None else DEFAULT_GSM8K_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"GSM8K test set not found at {p}. To populate the local "
            f"cache once (requires `pip install datasets` + network):\n"
            f'  python -c "from datasets import load_dataset; '
            f"load_dataset('openai/gsm8k', 'main', split='test')."
            f"to_json('{p.as_posix()}')\""
        )

    items: list[dict[str, str]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = _json.loads(line)
            except _json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and "question" in obj and "answer" in obj:
                items.append({"question": str(obj["question"]), "answer": str(obj["answer"])})
            if n > 0 and len(items) >= n:
                break
    return items


def run_gsm8k_benchmark(
    engine: Any,
    examples: list[dict[str, str]],
    *,
    num_shots: int = 8,
    max_gen: int = 256,
    temperature: float = 0.0,
    on_progress=None,
) -> dict[str, Any]:
    """Run GSM8K benchmark against an engine with a ``.generate()`` method.

    Args:
        engine: Object with ``generate(prompt, max_gen=...)`` -> str.
            Typically an :class:`EnigmaEngine`. Tests can pass a stub.
        examples: List of ``{"question", "answer"}`` dicts. The gold
            answer must contain ``#### <number>``.
        num_shots: Number of few-shot CoT examples to prepend (0..8).
        max_gen: Max tokens to generate per question.
        temperature: Sampling temperature. Default 0.0 (greedy) is the
            standard reproducible setting for GSM8K reporting.
        on_progress: Optional callable ``(idx, total, correct)``.

    Returns:
        Dict with ``total``, ``correct``, ``accuracy``, ``results``
        (per-item details, capped to first 200 chars of output).
    """
    total = len(examples)
    if total == 0:
        return {"total": 0, "correct": 0, "accuracy": 0.0, "results": []}

    prefix = _build_fewshot_prefix(num_shots)
    correct = 0
    results: list[dict[str, Any]] = []

    for idx, ex in enumerate(examples):
        question = ex.get("question", "")
        gold_text = ex.get("answer", "")
        gold_num = parse_final_number(gold_text)

        prompt = f"{prefix}Question: {question}\nAnswer:"
        try:
            output = engine.generate(prompt, max_gen=max_gen, temperature=temperature)
            if not isinstance(output, str):
                output = str(output)
        except Exception as exc:
            logger.debug("GSM8K generate failed on item %d: %s", idx, exc)
            output = ""

        pred_num = parse_final_number(output)
        item_correct = gold_num is not None and pred_num is not None and abs(pred_num - gold_num) < 1e-6
        if item_correct:
            correct += 1

        results.append(
            {
                "question": question,
                "gold": gold_num,
                "predicted": pred_num,
                "correct": item_correct,
                "output": output[:200],
            }
        )

        if on_progress is not None:
            try:
                on_progress(idx + 1, total, correct)
            except Exception:
                pass

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "results": results,
    }
