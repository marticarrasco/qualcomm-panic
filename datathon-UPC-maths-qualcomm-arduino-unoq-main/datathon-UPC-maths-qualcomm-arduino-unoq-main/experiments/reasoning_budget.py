#!/usr/bin/env python3
"""Reasoning that fits the board's 60 seconds: which prompt finishes most often, and how right is it?

    python3 experiments/reasoning_budget.py --prompts brief,terse,shots
    python3 experiments/reasoning_budget.py --prompts shots --port 8084 --read-tps 4 --write-tps 2.5

For each problem the model may write only as many tokens as the UNO Q could
write in the time left after reading the problem (the system prompt is the
same for every problem, so the server has it cached and it costs nothing).
An answer counts as "finished" when the text reaches 'Answer: X' in budget.
Unfinished answers are scored three ways: fallback letter, the last number
in the text matched to an option ("salvage"), and nothing else.

Only English problems with five text options are used.
"""
import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from copilot.client import post
from copilot.config import CONFIG
from copilot.llm import format_problem

ANSWER_RE = re.compile(r"answer\s*[:：]\s*\**\s*\(?([A-E])\b", re.IGNORECASE)
NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?(?:/\d+)?")
LIMIT = float(CONFIG["TIME_LIMIT_SECONDS"])
MARGIN = 5.0
FALLBACK = CONFIG["FALLBACK_ANSWER"]

BRIEF = ("You solve multiple-choice math competition problems for school students. "
         "Solve the problem step by step, briefly, then finish with 'Answer: X', "
         "where X is the letter of the correct option.")
TERSE = ("You solve multiple-choice math competition problems for school students. "
         "Write only the key calculations, one per line, with no explanations, no headings "
         "and without repeating the problem. Use at most 6 short lines. "
         "Then finish with 'Answer: X', where X is the letter of the correct option.")
# Worked examples live in the system prompt: the server caches it, so on the board
# they are read once, not once per problem. The problems are made up (not M3Kang).
SHOTS = TERSE + """

Example 1
Anna has 3 bags with 4 apples in each bag. She eats 2 apples. How many apples does she have left?
A) 8
B) 9
C) 10
D) 12
E) 14
3 x 4 = 12
12 - 2 = 10
Answer: C

Example 2
A rectangle has a perimeter of 30 cm. Its length is twice its width. What is its area?
A) 36 cm²
B) 45 cm²
C) 50 cm²
D) 54 cm²
E) 60 cm²
2(w + 2w) = 30, so w = 5
length = 10
area = 5 x 10 = 50
Answer: C

Example 3
How many two-digit numbers have digits that add up to 5?
A) 4
B) 5
C) 6
D) 9
E) 10
14, 23, 32, 41, 50
count = 5
Answer: B"""

PROMPTS = {"brief": BRIEF, "terse": TERSE, "shots": SHOTS}


def salvage(text, options):
    """Match the last numbers of unfinished reasoning to exactly one option value."""
    numbers = [n.replace(",", ".") for n in NUMBER_RE.findall(text)]
    for number in reversed(numbers[-3:]):
        hits = [letter for letter, value in options.items()
                if number in [n.replace(",", ".") for n in NUMBER_RE.findall(value)]]
        if len(hits) == 1:
            return hits[0]
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", nargs="+", default=["data/val.jsonl"])
    p.add_argument("--prompts", default="brief,terse,shots")
    p.add_argument("--port", default=CONFIG["LLM_PORT"])
    p.add_argument("--read-tps", type=float, default=11.0, help="board prompt tokens per second")
    p.add_argument("--write-tps", type=float, default=5.6, help="board output tokens per second")
    p.add_argument("--text-only", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--tag", default="")
    args = p.parse_args()
    base = f"http://127.0.0.1:{args.port}"

    problems = []
    for path in args.input:
        with open(path, encoding="utf-8") as f:
            problems += [json.loads(line) for line in f if line.strip()]
    problems = [x for x in problems if x["language"] == "eng" and len(x.get("options") or {}) == 5
                and (x["text_only"] or not args.text_only)]
    if args.limit:
        problems = problems[:args.limit]

    os.makedirs("results_local", exist_ok=True)
    out_path = f"results_local/budget-{time.strftime('%Y%m%d-%H%M%S')}{'-' + args.tag if args.tag else ''}.csv"
    rows = []
    with open(out_path, "w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["prompt", "id", "level", "text_only", "label", "budget", "written", "finished",
                         "answer", "salvaged", "output"])
        for name in args.prompts.split(","):
            for n, problem in enumerate(problems, 1):
                user = format_problem(problem)
                tokens = len(post(base + "/tokenize", {"content": user}, timeout=60)["tokens"]) + 12
                budget = max(int((LIMIT - MARGIN - tokens / args.read_tps) * args.write_tps), 1)
                response = post(base + "/v1/chat/completions", {
                    "messages": [{"role": "system", "content": PROMPTS[name]}, {"role": "user", "content": user}],
                    "max_tokens": budget, "temperature": 0, "seed": int(CONFIG["SEED"]), "cache_prompt": True,
                }, timeout=600)
                text = response["choices"][0]["message"]["content"]
                written = response.get("timings", {}).get("predicted_n", 0)
                found = ANSWER_RE.findall(text)
                row = {"prompt": name, "problem": problem, "finished": bool(found),
                       "answer": found[-1].upper() if found else None,
                       "salvaged": None if found else salvage(text, problem["options"]), "written": written}
                rows.append(row)
                writer.writerow([name, problem["id"], problem["level"], problem["text_only"], problem["label"],
                                 budget, written, int(row["finished"]), row["answer"] or "", row["salvaged"] or "",
                                 text])
                out.flush()
                print(f"\r{name}: {n}/{len(problems)}", end="", file=sys.stderr, flush=True)
            print(file=sys.stderr)

    truth = Counter(x["label"] for x in problems)
    print(f"\n{len(problems)} problems; always {FALLBACK}: {100 * truth[FALLBACK] / len(problems):.1f}%; "
          f"board speed assumed {args.read_tps:g} read / {args.write_tps:g} write tokens per second")
    print(f"{'prompt':7} {'finished':>9} {'right if finished':>18} {'salvaged':>9} {'right if salvaged':>18} "
          f"{'total, fallback ' + FALLBACK:>18} {'total, salvage':>15} {'mean tokens':>12}")
    for name in args.prompts.split(","):
        mine = [r for r in rows if r["prompt"] == name]
        done = [r for r in mine if r["finished"]]
        saved = [r for r in mine if r["salvaged"]]
        right_done = sum(r["answer"] == r["problem"]["label"] for r in done)
        right_saved = sum(r["salvaged"] == r["problem"]["label"] for r in saved)
        fall = sum((r["answer"] or FALLBACK) == r["problem"]["label"] for r in mine)
        with_salvage = sum((r["answer"] or r["salvaged"] or FALLBACK) == r["problem"]["label"] for r in mine)
        print(f"{name:7} {len(done):5}/{len(mine):<3} {right_done:8}/{len(done):<3} "
              f"{100 * right_done / max(len(done), 1):4.0f}% {len(saved):9} {right_saved:8}/{len(saved):<3} "
              f"{100 * right_saved / max(len(saved), 1):4.0f}% {fall:8}/{len(mine)} {100 * fall / len(mine):4.1f}% "
              f"{with_salvage:6}/{len(mine)} {100 * with_salvage / len(mine):4.1f}% "
              f"{sum(r['written'] for r in mine) / len(mine):12.0f}")
        for level in sorted({r["problem"]["level"] for r in mine}):
            group = [r for r in mine if r["problem"]["level"] == level]
            ok = sum((r["answer"] or r["salvaged"] or FALLBACK) == r["problem"]["label"] for r in group)
            print(f"          level {level}: {ok}/{len(group)} with salvage, "
                  f"{sum(r['finished'] for r in group)} finished")
    print(f"\nDetails: {out_path}")


if __name__ == "__main__":
    main()
