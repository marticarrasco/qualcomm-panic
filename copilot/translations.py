"""Reuse one solve across input translations when their answer values agree."""
from .arithmetic import complete_options
from .train_match import option_key

# Choose a single supplied translation before inference. This order is fixed
# and does not use any validation or input labels.
LANGUAGES = ("eng", "zho", "spa", "esp", "fra", "deu", "por", "ita", "cat")
LANGUAGE_PRIORITY = {language: i for i, language in enumerate(LANGUAGES)}


def _options(row):
    if (row.get("text_only") is not True or not row.get("id")
            or not complete_options(row)):
        return None
    keys = {letter: option_key(value) for letter, value in row["options"].items()}
    return keys if len(set(keys.values())) == 5 else None


def _quality(row):
    language = row.get("language", "")
    text = row.get("text", "")
    return LANGUAGE_PRIORITY.get(language, len(LANGUAGES)), len(text), language


class TranslationBatch:
    """Use only translations already present in the prediction input, without labels."""

    def __init__(self, problems, solve):
        self.solve = solve
        self.representatives = {}
        self.representative_options = {}
        self.cached = {}
        for row in problems:
            options = _options(row)
            if options is None:
                continue
            key = row["id"], tuple(sorted(options.values()))
            if key not in self.representatives or _quality(row) < _quality(self.representatives[key]):
                self.representatives[key] = row
                self.representative_options[key] = options

    def __call__(self, problem, **kwargs):
        options = _options(problem)
        if options is None:
            return self.solve(problem, **kwargs)
        key = problem["id"], tuple(sorted(options.values()))
        representative = self.representatives[key]
        if key in self.cached:
            value, mode = self.cached[key]
            answer = next(letter for letter, option in options.items() if option == value)
            return {"answer": answer, "status": f"reused_{mode}", "seconds": 0.0,
                    "prompt_tokens": 0, "completion_tokens": 0}
        result = self.solve(representative, **kwargs)
        value = self.representative_options[key][result["answer"]]
        self.cached[key] = value, result["status"]
        result["answer"] = next(letter for letter, option in options.items() if option == value)
        return result
