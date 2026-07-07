"""Rule-based reward helpers for reasoning-oriented RL training.

These scorers are deterministic and avoid training a neural reward model
in the loop for reasoning tasks.
"""

from __future__ import annotations

import ast
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable


# Spans of arithmetic characters (digits, operators, parens). A pure
# regex cannot balance parentheses, so candidates are validated by
# _safe_eval_arithmetic (which parses with ast and DOES handle parens).
_ARITH_SPAN_RE = re.compile(r"[-(]*\s*\d[\d.\s+\-*/()]*")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?")
_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def _extract_tag_content(text: str, tag: str) -> str:
    pattern = re.compile(
        rf"<{tag}>\s*(.*?)\s*</{tag}>",
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def _extract_answer_text(response: str) -> str:
    tagged_answer = _extract_tag_content(response, "answer")
    if tagged_answer:
        return tagged_answer

    think_close = re.search(r"</think>", response, re.IGNORECASE)
    if think_close:
        trailing = response[think_close.end() :].strip()
        if trailing:
            return trailing

    lines = [line.strip() for line in response.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _safe_eval_arithmetic(expr: str) -> float | None:
    try:
        node = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    def _eval(n: ast.AST) -> float:
        if isinstance(n, ast.Expression):
            return _eval(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            val = _eval(n.operand)
            return val if isinstance(n.op, ast.UAdd) else -val
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left = _eval(n.left)
            right = _eval(n.right)
            if isinstance(n.op, ast.Add):
                return left + right
            if isinstance(n.op, ast.Sub):
                return left - right
            if isinstance(n.op, ast.Mult):
                return left * right
            if right == 0:
                raise ZeroDivisionError
            return left / right
        raise ValueError("unsupported expression")

    try:
        return _eval(node)
    except (ValueError, ZeroDivisionError):
        return None


def _infer_arith_result(prompt: str) -> float | None:
    """Evaluate the arithmetic expression a prompt asks about, or None.

    Handles parentheses ("(2+3)*4" -> 20). Each candidate span is trimmed
    from the end until it parses, so trailing text and unmatched close
    parens are tolerated; a span must contain a binary operator to count.
    """
    for match in _ARITH_SPAN_RE.finditer(prompt):
        # Cap the span: real arithmetic prompts are short, and the
        # end-trim loop below is quadratic in the candidate length.
        candidate = match.group(0).strip()[:128]
        while candidate:
            # A binary operator must sit between operands, not just a
            # leading unary sign.
            if any(op in candidate[1:] for op in "+-*/"):
                value = _safe_eval_arithmetic(candidate)
                if value is not None:
                    return value
            candidate = candidate[:-1].strip()
    return None


def _extract_last_number(text: str) -> float | None:
    matches = _NUMBER_RE.findall(text)
    if not matches:
        return None
    token = matches[-1]
    if "/" in token:
        num, den = token.split("/", 1)
        try:
            den_f = float(den)
            if den_f == 0:
                return None
            return float(num) / den_f
        except ValueError:
            return None
    try:
        return float(token)
    except ValueError:
        return None


def _math_correct(
    prompt: str,
    response: str,
    *,
    ground_truth: str | float | int | None,
    tolerance: float,
) -> bool | None:
    answer_text = _extract_answer_text(response)
    if not answer_text:
        return False

    pred_num = _extract_last_number(answer_text)

    if ground_truth is not None:
        if pred_num is not None:
            gt_num = _extract_last_number(str(ground_truth))
            if gt_num is not None:
                return abs(pred_num - gt_num) <= tolerance
        return answer_text.strip().lower() == str(ground_truth).strip().lower()

    expected = _infer_arith_result(prompt)
    if expected is None:
        return None
    if pred_num is None:
        return False
    return abs(pred_num - expected) <= tolerance


def format_reward(response: str) -> float:
    """Return 1.0 when response includes a usable reasoning+answer format."""
    think_open = re.search(r"<think>", response, re.IGNORECASE)
    think_close = re.search(r"</think>", response, re.IGNORECASE)
    if not think_open or not think_close or think_open.start() >= think_close.start():
        return 0.0

    has_answer_tag = bool(_extract_tag_content(response, "answer"))
    trailing = response[think_close.end() :].strip()
    answer_text = trailing if trailing else ""
    if not has_answer_tag and not answer_text:
        return 0.0

    if has_answer_tag and not _extract_tag_content(response, "answer"):
        return 0.0

    if not has_answer_tag:
        answer_text = _extract_answer_text(response)
    return 1.0 if bool(answer_text) else 0.0


def math_reward(
    prompt: str,
    response: str,
    *,
    ground_truth: str | float | int | None = None,
    tolerance: float = 1e-6,
) -> float:
    """Return binary math accuracy reward using explicit or inferred target."""
    result = _math_correct(
        prompt,
        response,
        ground_truth=ground_truth,
        tolerance=tolerance,
    )
    if result is None:
        return 0.0
    return 1.0 if result else 0.0


def code_reward(
    response: str,
    *,
    tests: list[tuple[str, str]] | None = None,
    timeout_sec: int = 5,
) -> float:
    """Execute extracted Python code in a subprocess and score pass ratio."""
    match = _CODE_BLOCK_RE.search(response)
    code = match.group(1).strip() if match else ""
    if not code:
        return 0.0

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        encoding="utf-8",
    ) as handle:
        handle.write(code)
        tmp_path = Path(handle.name)

    try:
        if not tests:
            run = subprocess.run(
                [sys.executable, str(tmp_path)],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            return 1.0 if run.returncode == 0 else 0.0

        passes = 0
        for stdin_text, expected_substring in tests:
            run = subprocess.run(
                [sys.executable, str(tmp_path)],
                input=stdin_text,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
            if run.returncode == 0 and expected_substring in run.stdout:
                passes += 1
        return passes / len(tests)
    except (OSError, subprocess.SubprocessError, TimeoutError):
        return 0.0
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


def llm_judge_reward(
    prompt: str,
    response: str,
    *,
    judge_fn: Callable[[str, str], float | bool] | None = None,
) -> float:
    """Optional yes/no judge hook for non-math reasoning tasks."""
    if judge_fn is None:
        return 0.0
    try:
        score = judge_fn(prompt, response)
    except Exception:
        return 0.0

    if isinstance(score, bool):
        return 1.0 if score else 0.0
    try:
        value = float(score)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, value))


def reasoning_reward(
    prompt: str,
    response: str,
    *,
    ground_truth: str | float | int | None = None,
    judge_fn: Callable[[str, str], float | bool] | None = None,
) -> float:
    """Combined deterministic reward for reasoning GRPO.

    If math can be scored (explicit ground truth or arithmetic prompt), use a
    weighted blend of math accuracy + format. If not, use judge+format when a
    judge is provided, otherwise format-only fallback.
    """
    format_score = format_reward(response)
    math_result = _math_correct(
        prompt,
        response,
        ground_truth=ground_truth,
        tolerance=1e-6,
    )
    if math_result is not None:
        math_score = 1.0 if math_result else 0.0
        return round(0.8 * math_score + 0.2 * format_score, 6)

    judge_score = llm_judge_reward(prompt, response, judge_fn=judge_fn)
    if judge_score > 0.0:
        return round(0.7 * judge_score + 0.3 * format_score, 6)

    return format_score
