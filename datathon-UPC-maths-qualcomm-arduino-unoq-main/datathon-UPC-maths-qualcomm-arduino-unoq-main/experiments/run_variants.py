#!/usr/bin/env python3
"""Compare answering strategies on the same problems, without touching copilot/.

    python3 experiments/run_variants.py                       # all variants, train+val
    python3 experiments/run_variants.py --variants letter,think250 --limit 40

No retrieval is used, so train problems are fair test problems here.
Laptops are much faster than the UNO Q, so latency is ESTIMATED for the board
from token counts (ARDUINO_*_TOKENS_PER_SECOND in config.env), not measured.
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from copilot.client import post
from copilot.config import CONFIG
from copilot.llm import ANSWER_RE, SYSTEM as MARTI_SYSTEM, URL, format_problem

READ_TPS = float(CONFIG.get("ARDUINO_PROMPT_TOKENS_PER_SECOND", 11))
WRITE_TPS = float(CONFIG.get("ARDUINO_OUTPUT_TOKENS_PER_SECOND", 6))
TIME_LIMIT = float(CONFIG["TIME_LIMIT_SECONDS"])
SEED = int(CONFIG["SEED"])

LETTER_SYSTEM = ("You solve multiple-choice math competition problems for school students. "
                 "Read the problem and the options, then reply with the letter of the correct option.")
THINK_SYSTEM = ("You solve multiple-choice math competition problems for school students. "
                "Solve the problem step by step, briefly, then finish with 'Answer: X', "
                "where X is the letter of the correct option.")
FORCE = "So which option is correct? Reply with the letter only."


def chat(messages, **extra):
    response = post(URL, {"messages": messages, "temperature": 0, "seed": SEED,
                          "cache_prompt": True, **extra}, timeout=300)
    timings = response.get("timings", {})
    return (response["choices"][0]["message"]["content"].strip(),
            timings.get("prompt_n", 0), timings.get("predicted_n", 0))


def board_seconds(read, written):
    return read / READ_TPS + written / WRITE_TPS


def letter(problem):
    """The starter kit baseline: one token, forced to A-E by a grammar."""
    messages = [{"role": "system", "content": LETTER_SYSTEM},
                {"role": "user", "content": format_problem(problem) + "\nAnswer:"}]
    text, read, written = chat(messages, grammar="root ::= [A-E]", max_tokens=1)
    return text, read, written, "ok"


def marti(problem):
    """The prompt in copilot/llm.py at e957f46: free text, 128 tokens, regex parse, else fallback."""
    messages = [{"role": "system", "content": MARTI_SYSTEM},
                {"role": "user", "content": format_problem(problem) + "\nAnswer:"}]
    text, read, written = chat(messages, max_tokens=128)
    found = ANSWER_RE.findall(text)
    if found:
        return found[-1].upper(), read, written, "ok"
    return CONFIG["FALLBACK_ANSWER"], read, written, "invalid_response"


def think(budget):
    """Reason for up to `budget` tokens, then always get a letter.

    If the reasoning already ends in 'Answer: X', use it. Otherwise ask again
    with the reasoning in the context and a grammar that forces one letter.
    The server has the conversation cached, so the second call only reads the
    short follow-up. Board time counts the first prompt once, plus the follow-up.
    """
    def run(problem):
        messages = [{"role": "system", "content": THINK_SYSTEM},
                    {"role": "user", "content": format_problem(problem)}]
        text, read, written = chat(messages, max_tokens=budget)
        found = ANSWER_RE.findall(text)
        if found:
            return found[-1].upper(), read, written, "ok"
        messages += [{"role": "assistant", "content": text}, {"role": "user", "content": FORCE}]
        answer, read2, written2 = chat(messages, grammar="root ::= [A-E]", max_tokens=1)
        # read2 is what the server really re-read; with a warm cache it is the follow-up only
        return answer, read + read2, written + written2, "forced"
    return run


VARIANTS = {"letter": letter, "marti": marti,
            "think128": think(128), "think200": think(200), "think280": think(280)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", nargs="+", default=["data/train.jsonl", "data/val.jsonl"])
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--limit", type=int, default=0, help="0 means every problem")
    args = p.parse_args()

    problems = []
    for path in args.input:
        with open(path, encoding="utf-8") as f:
            problems += [json.loads(line) for line in f if line.strip()]
    if args.limit:
        problems = problems[:args.limit]

    os.makedirs("results", exist_ok=True)
    out_path = f"results/variants-{time.strftime('%Y%m%d-%H%M%S')}.csv"
    rows = []
    with open(out_path, "w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["variant", "id", "language", "level", "has_options", "label", "answer",
                         "correct", "status", "read_tokens", "written_tokens", "board_seconds_est"])
        for name in args.variants.split(","):
            run = VARIANTS[name]
            for n, problem in enumerate(problems, 1):
                answer, read, written, status = run(problem)
                row = {"variant": name, "correct": answer == problem["label"], "answer": answer,
                       "label": problem["label"], "status": status,
                       "has_options": bool(problem.get("options")),
                       "seconds": board_seconds(read, written)}
                rows.append(row)
                writer.writerow([name, problem["id"], problem.get("language", ""), problem.get("level", ""),
                                 row["has_options"], problem["label"], answer, int(row["correct"]),
                                 status, read, written, f"{row['seconds']:.1f}"])
                out.flush()
                print(f"\r{name}: {n}/{len(problems)}", end="", file=sys.stderr, flush=True)
            print(file=sys.stderr)

    print(f"\n{len(problems)} problems. Board seconds are estimates "
          f"({READ_TPS:g} tokens/s read, {WRITE_TPS:g} written).")
    print(f"{'variant':10} {'all':>12} {'with options':>14} {'no options':>12} "
          f"{'mean s':>7} {'worst s':>8} {'>60 s':>6}  status / letters")
    for name in args.variants.split(","):
        mine = [r for r in rows if r["variant"] == name]

        def acc(group):
            return f"{sum(r['correct'] for r in group)}/{len(group)} {100 * sum(r['correct'] for r in group) / max(len(group), 1):.0f}%"
        seconds = [r["seconds"] for r in mine]
        letters = Counter(r["answer"] for r in mine)
        print(f"{name:10} {acc(mine):>12} {acc([r for r in mine if r['has_options']]):>14} "
              f"{acc([r for r in mine if not r['has_options']]):>12} "
              f"{sum(seconds) / len(seconds):7.1f} {max(seconds):8.1f} "
              f"{sum(s > TIME_LIMIT for s in seconds):6}  {dict(Counter(r['status'] for r in mine))} "
              + " ".join(f"{l}:{letters[l]}" for l in "ABCDE"))
    truth = Counter(r["label"] for r in rows if r["variant"] == args.variants.split(",")[0])
    print("true letters: " + " ".join(f"{l}:{truth[l]}" for l in "ABCDE"))
    print(f"\nDetails: {out_path}")


if __name__ == "__main__":
    main()
