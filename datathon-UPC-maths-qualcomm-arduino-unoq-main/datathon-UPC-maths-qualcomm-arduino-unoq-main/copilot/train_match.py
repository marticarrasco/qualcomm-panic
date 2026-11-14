"""Transfer an option value only from a nearly identical train question."""
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from .arithmetic import complete_options, numeric


def normalized(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(text.split())


def stem(text):
    # Drop the ordinal printed in front of some questions; preserve all math,
    # including signs and numbers in the body.
    return re.sub(r"^\s*\d{1,2}\s*[.)]\s+", "", normalized(text)).strip()


def grams(text):
    text = " ".join(re.findall(r"[^\W_]+", unicodedata.normalize("NFKD", stem(text))))
    return Counter(text[i:i + 3] for i in range(len(text) - 2))


def cosine(a, b):
    if not a or not b:
        return 0.0
    return sum(v * b.get(k, 0) for k, v in a.items()) / math.sqrt(
        sum(v * v for v in a.values()) * sum(v * v for v in b.values()))


def option_key(text):
    normalized_text = normalized(text)
    parsed = numeric(normalized_text)
    if parsed:
        value, unit = parsed
        return f"number:{value}:{unit}"
    return "text:" + normalized_text


class LexicalMatch:
    """Search same-language train neighbors only when the option values agree."""

    def __init__(self, train_path):
        self.buckets = defaultdict(list)
        with Path(train_path).open(encoding="utf-8-sig") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                key, values = self._key(row)
                if row.get("text_only") is True and key and row.get("label") in values:
                    text = row.get("text") or ""
                    # Compute grams only for the few rows in the query's bucket.
                    # Keeping every Counter in memory is wasteful on the UNO Q.
                    self.buckets[key].append((row["id"], text, values[row["label"]]))

    @staticmethod
    def _key(problem):
        if not complete_options(problem):
            return None, {}
        values = {letter: option_key(value) for letter, value in problem["options"].items()}
        if len(set(values.values())) != 5:
            return None, {}
        return (problem.get("language"), tuple(sorted(values.values()))), values

    def answer(self, problem, exclude_id=None, minimum=0.80):
        key, values = self._key(problem)
        candidates = self.buckets.get(key, ())
        if not candidates or len(candidates) > 64:
            return None
        query = grams(problem.get("text") or "")
        if len(query) < 16:
            return None
        excluded = {problem.get("id"), exclude_id}
        hits = {option for source, text, option in candidates
                if source not in excluded and cosine(query, grams(text)) >= minimum}
        if len(hits) != 1:
            return None
        value = hits.pop()
        return next(letter for letter, candidate in values.items() if candidate == value)
