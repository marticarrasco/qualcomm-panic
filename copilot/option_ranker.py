"""Score multiple-choice distractors using a small train-fitted linear model."""
import json
import math
import re
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path

from .arithmetic import LETTERS, complete_options, numeric_options

WEIGHTS = Path(__file__).with_name("option_weights.json")
VERSION = 1
_DIGITS = re.compile(r"\d+")
_QUESTION_NUMBER = re.compile(r"^\s*\d{1,2}\s*[.)]\s+")


def normalize(text):
    return unicodedata.normalize("NFKC", text).casefold().strip()


def features(problem):
    if not complete_options(problem):
        return None
    options = [normalize(problem["options"][letter]) for letter in LETTERS]
    text = _QUESTION_NUMBER.sub("", normalize(problem.get("text", "")))
    numbers = {int(n) for n in _DIGITS.findall(text) if len(n) <= 10}
    values = numeric_options(problem)
    distinct = values is not None and len(set(values)) == 5
    sorted_order = distinct and (values == sorted(values) or values == sorted(values, reverse=True))
    tokens = [set(re.findall(r"[^\W_]+", opt)) for opt in options]
    lengths = [len(opt) for opt in options]
    duplicates = Counter(options)
    if distinct:
        order = sorted(values)
        span = float(order[-1] - order[0])
        center = float(order[2])
    out = []
    for i, opt in enumerate(options):
        feat = {f"letter_{LETTERS[i]}": 1.0}
        opt_numbers = {int(n) for n in _DIGITS.findall(opt) if len(n) <= 10}
        other_tokens = [tokens[j] for j in range(5) if j != i]
        feat["relative_length"] = max(-2.0, min(2.0, math.log(
            (lengths[i] + 1) / ((sum(lengths) - lengths[i]) / 4 + 1))))
        feat["option_duplicate"] = float(duplicates[opt] > 1)
        feat["shares_words"] = sum(len(tokens[i] & words) / max(1, len(tokens[i] | words))
                                   for words in other_tokens) / 4
        feat["any_number_in_question"] = float(bool(opt_numbers & numbers))
        feat["verbatim"] = float(len(opt) > 3 and opt in text)
        if distinct:
            val = values[i]
            rank = order.index(val)
            feat[f"rank_{rank}"] = 1.0
            mode = "ordered" if sorted_order else "shuffled"
            feat[f"{mode}_rank_{rank}"] = 1.0
            feat[f"{mode}_letter_{LETTERS[i]}"] = 1.0
            feat["center"] = abs(float(val) - center) / span
            neighbours = [abs(float(val - x)) / span for x in values if x != val]
            feat["nearest"] = min(neighbours)
            feat["mean_distance"] = sum(neighbours) / 4
            feat["negative"] = float(val < 0)
            feat["integral"] = float(val.denominator == 1)
            if feat["integral"]:
                n = int(val)
                feat["multiple5"] = float(n % 5 == 0)
                feat["even"] = float(n % 2 == 0)
                feat["zero"] = float(n == 0)
        out.append(feat)
    return out


@lru_cache(maxsize=1)
def load_weights():
    with WEIGHTS.open(encoding="utf-8") as source:
        data = json.load(source)
    if data["version"] != VERSION:
        raise ValueError("option model version mismatch")
    return data["weights"]


def probabilities(problem, weights=None):
    if weights is None:
        weights = load_weights()
    rows = features(problem)
    if rows is None:
        return None
    scores = [sum(value * weights.get(key, 0.0) for key, value in row.items()) for row in rows]
    best = max(scores)
    exps = [math.exp(score - best) for score in scores]
    total = sum(exps)
    return dict(zip(LETTERS, (x / total for x in exps)))
