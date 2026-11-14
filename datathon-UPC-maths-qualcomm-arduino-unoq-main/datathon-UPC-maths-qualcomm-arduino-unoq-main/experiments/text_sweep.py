#!/usr/bin/env python3
"""Paired text-only prompt and token-budget experiment on a local llama-server.

Run from the project root: python -m experiments.text_sweep --profiles short,value
"""
import argparse
import csv
import json
import statistics
import time
from collections import Counter
from pathlib import Path

from copilot.arithmetic import complete_options
from copilot.config import CONFIG
from copilot.priors import DEFAULT, guess
from copilot.text_solver import PROMPTS, infer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/eval_sets/val_100.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profiles", default="short,value")
    parser.add_argument("--port", type=int, default=int(CONFIG["LLM_PORT"]))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=192)
    parser.add_argument("--budgets", default="64,96,128,192")
    parser.add_argument("--read-tps", type=float, default=11.0)
    parser.add_argument("--write-tps", type=float, default=5.6)
    parser.add_argument("--deadline", type=float, default=55)
    parser.add_argument("--skip-prior", action="store_true")
    parser.add_argument("--rag", type=Path, help="include one example from a train embedding index")
    args = parser.parse_args()
    profiles = args.profiles.split(",")
    if any(name not in PROMPTS for name in profiles):
        parser.error(f"profiles must be from {', '.join(PROMPTS)}")
    budgets = [int(x) for x in args.budgets.split(",")]
    if min(budgets) <= 0 or min(args.tokens, args.read_tps, args.write_tps, args.deadline) <= 0:
        parser.error("token caps, rates and deadline must be positive")
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]
    rows = [row for row in rows if row.get("text_only") is True and complete_options(row)]
    rows = rows[args.skip:]
    rows = rows[:args.limit] if args.limit else rows
    args.output.parent.mkdir(parents=True, exist_ok=True)
    url = f"http://127.0.0.1:{args.port}/v1/chat/completions"
    from copilot.client import get
    if get(f"http://127.0.0.1:{args.port}/health")["status"] != "ok":
        raise RuntimeError("llama-server is not ready")
    examples_by_id = {}
    if args.rag:
        from copilot.rag import Index
        index = Index(args.rag)
        for row in rows:
            began = time.perf_counter()
            hits = index.search(row, exclude_id=row["id"])
            examples_by_id[(row["id"], row["language"])] = (
                [example for _, example in hits], time.perf_counter() - began)
    fields = ["profile", "id", "language", "level", "label", "prior", "answer", "status",
              "prompt_n", "full_prompt_n", "cached_n", "output_n", "seconds", "retrieval_s", "board_s",
              *[f"cap_{cap}" for cap in budgets], "output"]
    with args.output.open("x", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for profile in profiles:
            correct, counts, times = Counter(), Counter(), []
            for i, problem in enumerate(rows):
                prior = guess(problem)
                start = time.perf_counter()
                examples, retrieval_s = examples_by_id.get((problem["id"], problem["language"]), ((), 0.0))
                if args.skip_prior and prior != DEFAULT:
                    answer, status, raw, timings = prior, "prior", "", {}
                else:
                    answer, status, raw, timings = infer(
                        problem, start + 57, url, profile, args.tokens, examples)
                actual = time.perf_counter() - start
                prefill, generated = timings.get("prompt_n"), timings.get("predicted_n")
                board = ((prefill / args.read_tps + generated / args.write_tps)
                         if prefill is not None and generated is not None else
                         0.0 if status == "prior" else None)
                pick = answer if answer and board is not None and board <= args.deadline else prior
                rec = dict(profile=profile, id=problem["id"], language=problem["language"],
                           level=problem["level"], label=problem["label"], prior=prior,
                           answer=answer or prior, status=status, prompt_n=prefill,
                           full_prompt_n=timings.get("full_prompt_n"), cached_n=timings.get("cached_n"),
                           output_n=generated, seconds=round(actual + retrieval_s, 3),
                           retrieval_s=round(retrieval_s, 3),
                           board_s=round(board, 2) if board is not None else None, output=raw)
                for cap in budgets:
                    rec[f"cap_{cap}"] = pick if generated is not None and generated <= cap else prior
                    correct[cap] += rec[f"cap_{cap}"] == problem["label"]
                correct["prior"] += prior == problem["label"]
                counts[status] += 1
                if board is not None:
                    times.append(board)
                writer.writerow(rec)
                output.flush()
                if (i + 1) % 10 == 0 or i + 1 == len(rows):
                    print(profile, f"{i+1}/{len(rows)}", dict(correct), dict(counts),
                          f"mean board estimate {statistics.mean(times):.1f}s" if times else "",
                          flush=True)


if __name__ == "__main__":
    main()
