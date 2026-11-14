"""Replay fallback and routing choices using recorded model outputs."""
import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from copilot.arithmetic import numeric_options
from copilot.option_ranker import probabilities
from copilot.pipeline import UNCERTAIN_THRESHOLD
from copilot.priors import guess
from copilot.train_match import LexicalMatch


def evaluate(rows, recorded, train_index):
    counts = Counter()
    groups = defaultdict(list)
    for problem in rows:
        key = problem["id"], problem["language"]
        if recorded is not None and key not in recorded:
            continue
        target = problem["label"]
        prior = guess(problem)
        probs = probabilities(problem)
        rank = max(probs, key=probs.get)
        near = train_index.answer(problem)
        model = recorded.get(key) if recorded else None
        output = model["selected_answer"] if model else None
        if not model or model["status"] not in {"ok", "calculated"}:
            output = None
        guesses = {
            "always_d": "D", "prior": prior, "ranker": rank, "lex_ranker": near or rank,
        }
        if recorded is not None:
            guesses.update({
                "observed": model["selected_answer"],
                "lex_rank_non_numeric_model": near or (output if not numeric_options(problem) else None) or rank,
                "lex_rank_uncertain_model": near or (
                    output if max(probs.values()) < UNCERTAIN_THRESHOLD else None) or rank,
            })
        votes = {name: choice == target for name, choice in guesses.items()}
        counts.update(votes)
        groups[problem["id"]].append(votes)
    if not groups:
        raise ValueError("no evaluation rows")
    return {
        "rows": sum(map(len, groups.values())), "ids": len(groups), "correct": dict(counts),
        "mean_per_id_pct": {name: round(100 * statistics.mean(
            statistics.mean(row[name] for row in group) for group in groups.values()), 2)
                            for name in counts},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--train", type=Path, default=Path("data/train.jsonl"))
    p.add_argument("--recorded", type=Path)
    args = p.parse_args()
    rows = [json.loads(l) for l in args.input.read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    recorded = None
    if args.recorded:
        with args.recorded.open(encoding="utf-8") as source:
            recorded = {(r["problem_id"], r["language"]): r for r in csv.DictReader(source)
                        if r["selected_answer"]}
    print(json.dumps(evaluate(rows, recorded, LexicalMatch(args.train)), indent=2))


if __name__ == "__main__":
    main()
