"""Compare text solver policies on identical cohort keys and group translations by ID."""
import argparse
import csv
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from copilot.option_ranker import probabilities
from copilot.priors import guess


def read(path):
    with path.open(encoding="utf-8") as source:
        return {(r["problem_id"], r["language"]): r for r in csv.DictReader(source)
                if r["selected_answer"]}


def compare_delta(values, seed=42):
    rng = random.Random(seed)
    if not values:
        return None
    draws = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(2000))
    return [round(100 * draws[i], 2) for i in (49, 1949)]


def run(cohort, baseline, candidate, allow_partial=False):
    subset = {key for key in cohort if key in candidate and (baseline is None or key in baseline)}
    if subset != set(cohort) and not allow_partial:
        raise ValueError(f"missing results for {len(cohort) - len(subset)} cohort rows")
    if not subset:
        raise ValueError("no paired rows")
    counts = Counter()
    ids = defaultdict(list)
    seconds = []
    estimations = []
    baseline_estimates = []
    status = Counter()
    for key in sorted(subset):
        problem = cohort[key]
        label = problem["label"]
        result = candidate[key]
        pick = result["selected_answer"]
        if result["correct_answer"] != label:
            raise ValueError(f"candidate label mismatch at {key}")
        prior = guess(problem)
        rank = probabilities(problem)
        rank = max(rank, key=rank.get) if rank else prior
        base = baseline[key] if baseline else None
        if base and base["correct_answer"] != label:
            raise ValueError(f"baseline label mismatch at {key}")
        baseline_guess = base["selected_answer"] if base else prior
        if base and base.get("input_tokens") and base.get("output_tokens"):
            cost = int(base["input_tokens"]) / 11.0 + int(base["output_tokens"]) / 5.6
            baseline_estimates.append(min(cost, 57.0))
            if cost > 57:
                baseline_guess = prior
        elif base and base["status"] == "prior":
            baseline_estimates.append(0.0)
        votes = (pick == label, baseline_guess == label, prior == label, rank == label, label == "D")
        ids[key[0]].append(votes)
        for name, hit in zip(("candidate", "baseline_57s", "prior", "option_ranker", "always_d"), votes):
            counts[name] += hit
        status[result["status"]] += 1
        seconds.append(float(result["elapsed_seconds"]))
        if result["arduino_estimated_seconds"]:
            estimations.append(float(result["arduino_estimated_seconds"]))
    per_id = [tuple(statistics.mean(values) for values in zip(*group)) for group in ids.values()]
    diffs = [v[0] - v[1] for v in per_id]
    def latency(values):
        return ({"mean": round(statistics.mean(values), 2), "max": round(max(values), 2),
                 "p95": round(sorted(values)[int(.95 * (len(values) - 1))], 2),
                 "over_60": sum(v > 60 for v in values), "covered": len(values)} if values else None)
    report = {
        "rows": len(subset), "planned": len(cohort), "distinct_problem_ids": len(ids),
        "right": dict(counts), "status": dict(status),
        "accuracy_per_id_pct": {name: round(100 * statistics.mean(v[i] for v in per_id), 2)
                                for i, name in enumerate(counts)},
        "paired_delta_baseline_id_pct_95_interval": compare_delta(diffs) if baseline else None,
        "elapsed_laptop_s": latency(seconds), "uno_q_estimate_s": latency(estimations),
        "baseline_uno_q_estimate_s": latency(baseline_estimates),
    }
    if baseline is None:
        report["right"].pop("baseline_57s")
        report["accuracy_per_id_pct"].pop("baseline_57s")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--allow-partial", action="store_true")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]
    cohort = {(r["id"], r["language"]): r for r in rows}
    report = run(cohort, read(args.baseline) if args.baseline else None, read(args.candidate),
                 args.allow_partial)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
