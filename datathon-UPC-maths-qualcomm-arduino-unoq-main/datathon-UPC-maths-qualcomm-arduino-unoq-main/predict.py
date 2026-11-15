#!/usr/bin/env python3
"""Official entry point: answer every problem with one letter, A to E.

    python3 predict.py --input problems.jsonl --output predictions.csv
    python3 predict.py --input problems.jsonl --output predictions.csv --rag 1

Input: one JSON object per line with at least "id" and "text", plus
"options" ({"A": ..., "E": ...}) and "language" when available.
Output: a CSV with id, language, answer, seconds, status.

Rules: runs offline, at most 60 seconds per problem, deterministic.
"""
import argparse
import csv
import json
import os
import sys

from copilot.config import CONFIG
from copilot.pipeline import batch_solver


def wait_for_model(seconds=180):
    """After a reboot the model takes a while to load. Starting before it is ready would
    turn the first answers into fallbacks, so wait here, outside the per-problem time."""
    import time
    import urllib.request
    url = f"http://127.0.0.1:{CONFIG['LLM_PORT']}/health"
    deadline = time.monotonic() + seconds
    told = False
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status == 200:
                    return True
        except OSError:
            pass
        if not told:
            print("Waiting for the model server to be ready...", file=sys.stderr)
            told = True
        time.sleep(2)
    print("The model server did not answer: continuing, problems will get the fallback answer.", file=sys.stderr)
    return False


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--rag", type=int, default=0, metavar="K", help="add K similar solved problems")
    p.add_argument("--index", default="data/index.db")
    args = p.parse_args()

    index = None
    if args.rag:
        if not os.path.exists(args.index):
            sys.exit(f"{args.index} not found. Build it first: python3 build_index.py")
        from copilot.rag import Index
        index = Index(args.index)

    wait_for_model()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.input, encoding="utf-8-sig") as f:
        problems = [json.loads(line) for line in f if line.strip()]
    solve = batch_solver(problems)

    with open(args.output, "w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["id", "language", "answer", "seconds", "status"])
        for n, problem in enumerate(problems, 1):
            try:
                result = solve(problem, index=index, k=args.rag)
            except Exception as error:  # one odd row must never stop the run: answer and go on
                result = {"answer": CONFIG["FALLBACK_ANSWER"], "seconds": 0.0,
                          "status": f"error: {type(error).__name__}"}
            if result.get("answer") not in ("A", "B", "C", "D", "E"):
                result["answer"] = CONFIG["FALLBACK_ANSWER"]
            writer.writerow([problem.get("id", n), problem.get("language", ""), result["answer"],
                             result["seconds"], result["status"]])
            out.flush()
            print(f"[{n}/{len(problems)}] {problem.get('id', n)} -> {result['answer']} "
                  f"in {result['seconds']:.1f} s ({result['status']})", file=sys.stderr)


if __name__ == "__main__":
    main()
