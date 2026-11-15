#!/usr/bin/env python3
"""Error analysis of one eval.py run: accuracy by topic, level, language and more.

    python3 experiments/error_analysis.py results/eval-<time>-<name>.csv
    python3 experiments/error_analysis.py results/eval-<time>-<name>.csv --markdown > analysis.md

Joins the run's per-problem CSV with the validation file to add what the CSV
lacks: the topic (tutor/topics.py keyword rules on the English text of the same
problem id), whether the problem needs its picture, and the kind of options.
Every table gives counts next to percentages, and the same rows scored with
"always D", so a group that is merely easy to guess does not look like skill.

Standard library only. No problem text is printed.
"""
import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from copilot.priors import _number  # noqa: E402
from tutor.topics import NAMES, classify  # noqa: E402

LETTERS = "ABCDE"


def option_kind(options):
    if sorted(options or {}) != list(LETTERS):
        return "no options"
    values = [_number(options[letter]) for letter in LETTERS]
    if any(value is None for value in values):
        return "text options"
    ordered = sorted(values)
    return "numbers, in order" if values in (ordered, ordered[::-1]) else "numbers, out of order"


def load_problems(paths):
    by_key, english = {}, {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    problem = json.loads(line)
                    by_key[(problem["id"], problem.get("language", ""))] = problem
                    if problem.get("language") == "eng":
                        english[problem["id"]] = problem
    return by_key, english


def interval(right, n):
    """Wilson 95% interval, in percent."""
    if not n:
        return 0.0, 0.0
    p, z = right / n, 1.96
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return 100 * (centre - half), 100 * (centre + half)


def table(title, rows, key, markdown, order=None, min_n=1):
    groups = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    names = order or sorted(groups, key=lambda name: (-len(groups[name]), str(name)))
    lines = []
    for name in names:
        group = groups.get(name, [])
        if len(group) < min_n:
            continue
        right = sum(r["right"] for r in group)
        d = sum(r["label"] == "D" for r in group)
        low, high = interval(right, len(group))
        lines.append((str(name), len(group), right, 100 * right / len(group), low, high, 100 * d / len(group)))
    if markdown:
        print(f"\n### {title}\n")
        print("| Group | Problems | Right | Accuracy | 95% interval | Always D |")
        print("| --- | --- | --- | --- | --- | --- |")
        for name, n, right, acc, low, high, d in lines:
            print(f"| {name} | {n} | {right} | {acc:.1f}% | {low:.0f} to {high:.0f}% | {d:.1f}% |")
    else:
        print(f"\n{title}")
        for name, n, right, acc, low, high, d in lines:
            print(f"  {name:34} {right:5}/{n:<5} {acc:5.1f}%  [{low:4.0f},{high:4.0f}]   always D {d:5.1f}%")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv")
    p.add_argument("--problems", nargs="+", default=["data/val.jsonl"])
    p.add_argument("--markdown", action="store_true")
    p.add_argument("--min-language", type=int, default=40, help="hide languages with fewer rows")
    args = p.parse_args()

    by_key, english = load_problems(args.problems)
    csv.field_size_limit(10 ** 9)
    rows = []
    with open(args.csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("status") == "interrupted":
                continue
            problem = by_key.get((r["problem_id"], r["language"]), {})
            source = english.get(r["problem_id"], problem)
            rows.append({
                "id": r["problem_id"], "language": r["language"], "level": int(r["level"] or 0),
                "label": r["correct_answer"], "answer": r["selected_answer"],
                "right": r["selected_answer"] == r["correct_answer"], "status": r["status"],
                "seconds": float(r["arduino_estimated_seconds"] or 0),
                "topic": NAMES[classify(source.get("text", ""))]["eng"] if source else "?",
                "picture": "needs the picture" if not problem.get("text_only", True) else "text only",
                "options": option_kind(problem.get("options")),
            })
    if not rows:
        sys.exit("no rows")

    n, right = len(rows), sum(r["right"] for r in rows)
    d = sum(r["label"] == "D" for r in rows)
    low, high = interval(right, n)
    seconds = sorted(r["seconds"] for r in rows)
    head = (f"{n} rows, {len({r['id'] for r in rows})} problems, {len({r['language'] for r in rows})} languages. "
            f"Right: {right} = {100 * right / n:.1f}% (95% interval {low:.1f} to {high:.1f}%). "
            f"Always D: {d} = {100 * d / n:.1f}%. "
            f"Estimated board time: mean {sum(seconds) / n:.1f} s, 95th percentile {seconds[int(0.95 * (n - 1))]:.1f} s, "
            f"worst {seconds[-1]:.1f} s, over 60 s: {sum(s > 60 for s in seconds)}.")
    print(("## Error analysis\n\n" if args.markdown else "") + head)

    table("By how the answer was produced (status)", rows, lambda r: r["status"], args.markdown)
    table("By level (0 is the youngest)", rows, lambda r: r["level"], args.markdown, order=list(range(8)))
    table("By topic (keyword rules, about 70% accurate)", rows, lambda r: r["topic"], args.markdown)
    table("By whether the problem needs its picture", rows, lambda r: r["picture"], args.markdown)
    table("By kind of options", rows, lambda r: r["options"], args.markdown)
    table(f"By language (at least {args.min_language} rows)", rows, lambda r: r["language"], args.markdown,
          min_n=args.min_language)
    table("By correct letter", rows, lambda r: r["label"], args.markdown, order=list(LETTERS))
    table("By answered letter", rows, lambda r: r["answer"], args.markdown, order=list(LETTERS))


if __name__ == "__main__":
    main()
