"""Safe arithmetic evaluator -- the honest fix for a from-scratch model that
can't compute in-weights.

Enigma's BPE tokenizer splits numbers inconsistently ('56'->['5','6'] but
'15'->['15']), so a 182M model can't learn digit-wise arithmetic; trained on
math it only emits confident-wrong answers. The production-standard answer
(used by every serious LLM) is a calculator TOOL: the model routes arithmetic
to this evaluator and reports the exact result.

evaluate() parses the expression to an AST and walks a strict whitelist of
numeric operations -- NO eval()/exec(), no names, no calls, no attribute
access, so a malicious "expression" can't run code. Raises CalcError on
anything outside the whitelist or on math errors (div by zero, overflow).
"""

from __future__ import annotations

import ast
import operator

__all__ = ["evaluate", "CalcError"]


class CalcError(ValueError):
    """Expression was not safe arithmetic, or the math failed."""


# Whitelisted binary / unary operators -> their implementations.
_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Guard against pathological exponents that would hang / eat memory before
# any result is produced (2**10**9 etc.). Bounds the base and exponent.
_MAX_POW_EXP = 1000
_MAX_POW_BASE = 10**12


def _eval(node: ast.AST) -> float | int:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise CalcError(f"non-numeric constant: {node.value!r}")
        return node.value
    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise CalcError(f"unsupported operator: {type(node.op).__name__}")
        left, right = _eval(node.left), _eval(node.right)
        if op is operator.pow and (abs(right) > _MAX_POW_EXP or abs(left) > _MAX_POW_BASE):
            raise CalcError("exponent too large")
        try:
            return op(left, right)
        except ZeroDivisionError as exc:
            raise CalcError("division by zero") from exc
        except (OverflowError, ValueError) as exc:
            raise CalcError(str(exc)) from exc
    if isinstance(node, ast.UnaryOp):
        op = _UNARYOPS.get(type(node.op))
        if op is None:
            raise CalcError(f"unsupported unary operator: {type(node.op).__name__}")
        return op(_eval(node.operand))
    raise CalcError(f"unsupported expression element: {type(node).__name__}")


def evaluate(expression: str) -> float | int:
    """Evaluate a pure-arithmetic expression. Raises CalcError on anything
    that isn't safe arithmetic (names, calls, attributes) or on math errors."""
    if not isinstance(expression, str) or not expression.strip():
        raise CalcError("empty expression")
    if len(expression) > 256:
        raise CalcError("expression too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise CalcError(f"could not parse: {exc.msg}") from exc
    result = _eval(tree.body)
    # Python pow can leave the reals ((-8) ** 0.5 -> complex); a calculator
    # that answers "2.83j" isn't honest arithmetic. Refuse instead.
    if not isinstance(result, (int, float)):
        raise CalcError("result is not a real number")
    # Normalize whole-valued floats to int for clean display (6.0 -> 6).
    if isinstance(result, float) and result.is_integer():
        return int(result)
    return result


def format_result(value: float | int) -> str:
    """Human-friendly result string: exact ints as-is, floats rounded to a
    reasonable precision without trailing-zero noise."""
    if isinstance(value, int):
        return str(value)
    return f"{value:.10g}"
