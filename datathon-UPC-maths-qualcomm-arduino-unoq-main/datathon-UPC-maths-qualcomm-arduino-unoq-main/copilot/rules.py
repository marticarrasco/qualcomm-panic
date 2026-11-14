"""Solve explicit arithmetic without a language model."""
import re
import unicodedata

from .arithmetic import calculate, match_option, numeric_options

ORDINAL = re.compile(r"^\s*\d{1,2}\s*[.)]\s+")


def _arithmetic(text):
    text = (text.strip().replace("−", "-").replace("×", "*").replace("÷", "/"))
    match = re.fullmatch(r"(?:(?:calculate|compute|what is)\s+)?"
                         r"([-+*/()\d .]{3,90})\s*(?:=\s*)?\?\.?", text)
    if match and any(c in match[1] for c in "+-*/"):
        return calculate(match[1].strip())
    return None


def answer(problem):
    values = numeric_options(problem)
    if values is None:
        return None
    text = ORDINAL.sub("", unicodedata.normalize("NFKC", problem.get("text", "")))
    text = " ".join(text.casefold().split())
    return match_option(_arithmetic(text), values)
