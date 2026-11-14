"""Model-free fallback guess: one letter without calling any model.

Two facts about the training problems, measured in experiments/priors_analysis.py:

1. With no information about the options, D is the most frequent key (about 25%).
2. When the five options are numbers, the key is rarely an extreme value: the
   smallest is right about 9% of the time, the largest about 12%, the median
   about 31%. If the options are printed in sorted order this adds nothing to
   the letter prior (median = C ties with D). If they are printed out of order
   the letter carries no information and the median value is right about 37% of
   the time, against 18% for D.

So: the median value when five distinct numeric options are out of order, D in
every other case. Standard library only, deterministic, no data files.
"""
from __future__ import annotations

import re

LETTERS = "ABCDE"
DEFAULT = "D"

# A number, optionally a fraction, optionally followed by a short unit.
_NUMBER = re.compile(
    r"^\s*([-−–]?\s*\d+(?:[.,]\d+)?)(?:\s*/\s*(\d+))?\s*(?:%|°|[a-zA-Z€$²³/]{1,6}\.?)?\s*$"
)


def _number(text) -> float | None:
    match = _NUMBER.match(str(text))
    if not match:
        return None
    value = float(
        re.sub(r"\s", "", match.group(1)).replace("−", "-").replace("–", "-").replace(",", ".")
    )
    if match.group(2):
        denominator = float(match.group(2))
        if denominator == 0:
            return None
        value /= denominator
    return value


def guess(problem) -> str:
    """Return one letter A-E for a problem dict; never raises."""
    try:
        options = problem.get("options") or {}
        if sorted(options) != list(LETTERS):
            return DEFAULT
        values = [_number(options[letter]) for letter in LETTERS]
        if any(value is None for value in values) or len(set(values)) != 5:
            return DEFAULT
        ordered = sorted(values)
        if values == ordered or values == ordered[::-1]:
            return DEFAULT
        return LETTERS[values.index(ordered[2])]
    except Exception:
        return DEFAULT
