"""Exact, bounded arithmetic for calculations proposed by the model."""
import ast
import math
import operator
import re
from fractions import Fraction

LETTERS = "ABCDE"
NUMBER = re.compile(
    r"\s*([+\-−]?\d{1,15}(?:[.,]\d{1,12})?)"
    r"(?:\s*/\s*([+\-−]?\d{1,15}))?\s*([%°]|[a-zA-Z€$²³/]{1,8})?\s*"
)


def complete_options(problem):
    options = problem.get("options")
    return (isinstance(options, dict) and set(options) == set(LETTERS)
            and all(isinstance(value, str) and value.strip() for value in options.values()))


def numeric(text):
    match = NUMBER.fullmatch(text)
    if not match:
        return None
    numerator, denominator, unit = match.groups()
    try:
        value = Fraction(numerator.replace("−", "-").replace(",", "."))
        if denominator:
            value /= int(denominator.replace("−", "-"))
        return value, unit or ""
    except (ValueError, ZeroDivisionError):
        return None


def numeric_options(problem):
    if not complete_options(problem):
        return None
    parsed = [numeric(problem["options"][letter]) for letter in LETTERS]
    if any(item is None for item in parsed) or len({unit for _, unit in parsed}) != 1:
        return None
    return [value for value, _ in parsed]


def calculate(expression):
    """Interpret a small expression tree; model text never runs as Python."""
    if not expression or len(expression) > 160:
        return None
    try:
        root = ast.parse(expression, mode="eval")
        if sum(1 for _ in ast.walk(root)) > 60:
            return None
        return _visit(root.body, expression)
    except (SyntaxError, ValueError, ArithmeticError, RecursionError):
        return None


def _visit(node, expression):
    if isinstance(node, ast.Constant) and type(node.value) in {int, float}:
        literal = ast.get_source_segment(expression, node)
        if not re.fullmatch(r"(?:\d+(?:\.\d*)?|\.\d+)", literal):
            raise ValueError
        value = Fraction(literal)
    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _visit(node.operand, expression) * (-1 if isinstance(node.op, ast.USub) else 1)
    elif isinstance(node, ast.BinOp):
        left, right = _visit(node.left, expression), _visit(node.right, expression)
        if isinstance(node.op, ast.Pow):
            if right.denominator != 1 or abs(right) > 12:
                raise ValueError
            value = left ** int(right)
        else:
            operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
                          ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
                          ast.Mod: operator.mod}
            op = operations.get(type(node.op))
            if op is None:
                raise ValueError
            value = Fraction(op(left, right))
    elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        args = [_visit(arg, expression) for arg in node.args]
        name = node.func.id
        if name in {"gcd", "lcm", "comb", "perm", "factorial"}:
            if any(arg.denominator != 1 or not 0 <= arg <= 100 for arg in args):
                raise ValueError
            ints = [int(arg) for arg in args]
            expected = 1 if name == "factorial" else 2
            if len(ints) != expected:
                raise ValueError
            value = Fraction(getattr(math, name)(*ints))
        elif name in {"min", "max"} and args:
            value = min(args) if name == "min" else max(args)
        else:
            raise ValueError
    else:
        raise ValueError
    if max(value.numerator.bit_length(), value.denominator.bit_length()) > 128:
        raise ValueError
    return value


def match_option(value, options):
    if value is None or options is None:
        return None
    matches = [letter for letter, option in zip(LETTERS, options) if option == value]
    return matches[0] if len(matches) == 1 else None
