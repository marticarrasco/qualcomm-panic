#!/usr/bin/env python3
"""Score the five letters with log-probabilities, to test fixes for letter bias.

    python3 experiments/letter_scores.py collect --tag qwen0.8b          # ask the server
    python3 experiments/letter_scores.py report results_local/scores-qwen0.8b.json

collect asks for one token per prompt and stores the log-probability of each
letter A-E, for the original option order and for the 4 other rotations of
the options (the option texts move, the letters stay). report compares:

  raw        argmax of the original order (what a one-letter grammar answers)
  prior      subtract each letter's average log-probability, fitted on train
  rotate-N   sum each option's score over N rotations, so every option visits
             N letter positions and position bias cancels

Only problems with five text options are used. The question comes before the
options, so on the board each extra rotation re-reads only the options.
"""
import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from copilot.client import post
from copilot.config import CONFIG

LETTERS = "ABCDE"
SYSTEM = ("You solve multiple-choice math competition problems for school students. "
          "Read the problem and the options, then reply with the letter of the correct option.")
FLOOR = -20.0  # a letter missing from the top tokens


def prompt(problem, rotation):
    """Rotation r puts original option (j + r) % 5 at letter position j."""
    lines = [problem["text"].strip()]
    for j, letter in enumerate(LETTERS):
        lines.append(f"{letter}) {problem['options'][LETTERS[(j + rotation) % 5]]}")
    return "\n".join(lines) + "\nAnswer:"


def letter_logprobs(url, problem, rotation):
    response = post(url, {"messages": [{"role": "system", "content": SYSTEM},
                                       {"role": "user", "content": prompt(problem, rotation)}],
                          "max_tokens": 1, "temperature": 0, "seed": int(CONFIG["SEED"]),
                          "logprobs": True, "top_logprobs": 40, "cache_prompt": True}, timeout=300)
    top = response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    found = {}
    for item in top:
        token = item["token"].strip()
        if token in LETTERS and len(token) == 1 and token not in found:
            found[token] = item["logprob"]
    return [found.get(letter, FLOOR) for letter in LETTERS], response.get("timings", {}).get("prompt_n", 0)


def load(path, limit):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rows = [r for r in rows if len(r.get("options") or {}) == 5 and r["language"] == "eng"]
    return rows[:limit] if limit else rows


def collect(args):
    url = f"http://127.0.0.1:{args.port}/v1/chat/completions"
    out = {"tag": args.tag, "problems": []}
    for split, path, limit in (("train", args.train, args.train_limit), ("val", args.val, 0)):
        problems = load(path, limit)
        for n, problem in enumerate(problems, 1):
            scores, read = [], []
            for rotation in range(5):
                logprobs, tokens = letter_logprobs(url, problem, rotation)
                scores.append(logprobs)
                read.append(tokens)
            out["problems"].append({"id": problem["id"], "split": split, "label": problem["label"],
                                    "text_only": problem["text_only"], "level": problem["level"],
                                    "scores": scores, "read_tokens": read})
            print(f"\r{split}: {n}/{len(problems)}", end="", file=sys.stderr, flush=True)
        print(file=sys.stderr)
    os.makedirs("results_local", exist_ok=True)
    path = f"results_local/scores-{args.tag}.json"
    with open(path, "w") as f:
        json.dump(out, f)
    print(f"Wrote {path}")
    report_file(path)


def option_scores(problem, rotations, prior):
    """Sum, per ORIGINAL option, the (prior-corrected) score it got at each position."""
    total = [0.0] * 5
    for rotation in rotations:
        for j in range(5):
            total[(j + rotation) % 5] += problem["scores"][rotation][j] - prior[j]
    return total


def accuracy(problems, rotations, prior):
    hits, picks = 0, Counter()
    for problem in problems:
        total = option_scores(problem, rotations, prior)
        pick = LETTERS[max(range(5), key=lambda i: total[i])]
        picks[pick] += 1
        hits += pick == problem["label"]
    return hits, picks


def report_file(path):
    with open(path) as f:
        data = json.load(f)
    train = [p for p in data["problems"] if p["split"] == "train"]
    prior = [sum(p["scores"][0][j] for p in train) / max(len(train), 1) for j in range(5)]
    zero = [0.0] * 5
    print(f"\n{data['tag']}: letter prior fitted on {len(train)} train problems: "
          + " ".join(f"{l}:{v:.2f}" for l, v in zip(LETTERS, prior)))
    groups = [("val, text-only", [p for p in data["problems"] if p["split"] == "val" and p["text_only"]]),
              ("val, needs image", [p for p in data["problems"] if p["split"] == "val" and not p["text_only"]]),
              ("train, text-only", [p for p in train if p["text_only"]]),
              ("train, needs image", [p for p in train if not p["text_only"]])]
    methods = [("raw", [0], zero), ("prior", [0], prior), ("rotate-2", [0, 1], zero),
               ("rotate-3", [0, 1, 2], zero), ("rotate-5", [0, 1, 2, 3, 4], zero),
               ("rotate-5+prior", [0, 1, 2, 3, 4], prior)]
    for name, problems in groups:
        if not problems:
            continue
        truth = Counter(p["label"] for p in problems)
        top = truth.most_common(1)[0]
        print(f"\n{name}: {len(problems)} problems, always {top[0]} = {100 * top[1] / len(problems):.1f}%")
        for method, rotations, pr in methods:
            hits, picks = accuracy(problems, rotations, pr)
            print(f"  {method:15} {hits:4}/{len(problems)} = {100 * hits / len(problems):5.1f}%   picks "
                  + " ".join(f"{l}:{picks[l]}" for l in LETTERS))
    reads = sorted(p["read_tokens"][0] for p in data["problems"])
    print(f"\nprompt tokens actually read, first rotation: median {reads[len(reads) // 2]}, max {reads[-1]}")
    later = sorted(t for p in data["problems"] for t in p["read_tokens"][1:])
    print(f"prompt tokens actually read, later rotations: median {later[len(later) // 2]}, max {later[-1]}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--tag", required=True)
    c.add_argument("--port", default=CONFIG["LLM_PORT"])
    c.add_argument("--train", default="data/train.jsonl")
    c.add_argument("--val", default="data/val.jsonl")
    c.add_argument("--train-limit", type=int, default=300)
    r = sub.add_parser("report")
    r.add_argument("path")
    args = p.parse_args()
    if args.command == "collect":
        collect(args)
    else:
        report_file(args.path)


if __name__ == "__main__":
    main()
