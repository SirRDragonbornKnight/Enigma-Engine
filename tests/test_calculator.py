"""The safe calculator must compute correct arithmetic AND refuse anything
that isn't arithmetic (no code execution through a crafted 'expression')."""

import pytest

from enigma_engine.core.calculator import CalcError, evaluate, format_result


@pytest.mark.parametrize(
    "expr, want",
    [
        ("7 * 8", 56),
        ("15 + 27", 42),
        ("100 / 4", 25),
        ("9 - 3", 6),
        ("2 ** 10", 1024),
        ("(3 + 4) * 5", 35),
        ("17 % 5", 2),
        ("17 // 5", 3),
        ("-8 + 3", -5),
        ("6 / 4", 1.5),
        ("2 * 3 + 4 * 5", 26),
    ],
)
def test_correct_arithmetic(expr, want):
    assert evaluate(expr) == want


def test_whole_float_normalizes_to_int():
    assert evaluate("10 / 2") == 5
    assert isinstance(evaluate("10 / 2"), int)


def test_division_by_zero():
    with pytest.raises(CalcError):
        evaluate("1 / 0")


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('echo hi')",
        "os.system('x')",
        "open('f')",
        "1 + name",
        "[1, 2, 3]",
        "lambda: 1",
        "(1).__class__",
        "print(1)",
        "",
        "   ",
    ],
)
def test_rejects_non_arithmetic(expr):
    with pytest.raises(CalcError):
        evaluate(expr)


def test_huge_exponent_refused_fast():
    with pytest.raises(CalcError):
        evaluate("2 ** 10 ** 9")


def test_complex_result_refused():
    # Python pow leaves the reals on negative base ** fractional exponent;
    # the calculator must refuse rather than report a complex number.
    with pytest.raises(CalcError):
        evaluate("(-8) ** 0.5")


def test_format_result():
    assert format_result(56) == "56"
    assert format_result(1.5) == "1.5"
    assert format_result(evaluate("10 / 2")) == "5"
