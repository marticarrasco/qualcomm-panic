#!/usr/bin/env python3
"""Does the vision server help on problems that need the picture?

    .venv/bin/python experiments/vision_eval.py --limit 100
    .venv/bin/python experiments/vision_eval.py --describe 3   # print what the model sees

Laptop only: it reads the images from the M3Kang parquet with pyarrow.
Compares, on the same image-dependent validation problems:
  text    one letter from the text server, no picture, no options
  vision  one letter from the vision server (port VISION_PORT), whole picture
The vision server must be started by hand (README, "Let the model see the image").
"""
import argparse
import base64
import csv
import glob
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from copilot.client import post
from copilot.config import CONFIG
from copilot.llm import format_problem

TEXT_URL = f"http://127.0.0.1:{CONFIG['LLM_PORT']}/v1/chat/completions"
VISION_URL = f"http://127.0.0.1:{CONFIG['VISION_PORT']}/v1/chat/completions"
SYSTEM = ("You solve multiple-choice math competition problems for school students. "
          "Read the problem and the options, then reply with the letter of the correct option.")
SHARD = "~/.cache/huggingface/hub/datasets--qualcomm--M3Kang/snapshots/*/data/high_quality-00000-of-00004.parquet"


def load_images(parquet, keys):
    import pyarrow.parquet as pq
    meta = pq.read_table(parquet, columns=["id", "language"]).to_pylist()
    wanted = [i for i, row in enumerate(meta) if (row["id"], row["language"]) in keys]
    table = pq.read_table(parquet, columns=["id", "language", "image"]).take(wanted)
    return {(row["id"], row["language"]): row["image"]["bytes"] for row in table.to_pylist()}


def ask(url, content, **extra):
    response = post(url, {"messages": [{"role": "system", "content": SYSTEM},
                                       {"role": "user", "content": content}],
                          "temperature": 0, "seed": int(CONFIG["SEED"]), **extra}, timeout=300)
    return response["choices"][0]["message"]["content"].strip(), response.get("timings", {})


def image_part(raw):
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(raw).decode()}}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="data/full_noocr/val.jsonl")
    p.add_argument("--parquet", default="")
    p.add_argument("--language", default="eng")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--describe", type=int, default=0, help="print the model's description of N images, then stop")
    args = p.parse_args()

    with open(args.input, encoding="utf-8") as f:
        problems = [json.loads(line) for line in f if line.strip()]
    problems = [x for x in problems if x["language"] == args.language and not x["text_only"]]
    problems = problems[:args.describe or args.limit or None]
    parquet = args.parquet or sorted(glob.glob(os.path.expanduser(SHARD)))[-1]
    images = load_images(parquet, {(x["id"], x["language"]) for x in problems})

    if args.describe:
        for problem in problems:
            text, timings = ask(VISION_URL, [image_part(images[(problem["id"], problem["language"])]),
                                             {"type": "text", "text": "Describe everything you see in this picture, "
                                                                      "including any text and the options A to E."}],
                                max_tokens=200)
            print(f"\n=== {problem['id']} (read {timings.get('prompt_n')} tokens)\n{problem['text'][:200]}\n--- model sees:\n{text}")
        return

    os.makedirs("results", exist_ok=True)
    out_path = f"results/vision-{time.strftime('%Y%m%d-%H%M%S')}.csv"
    rows = []
    with open(out_path, "w", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["id", "level", "label", "text_answer", "vision_answer", "vision_read_tokens"])
        for n, problem in enumerate(problems, 1):
            prompt = format_problem(problem) + "\nAnswer:"
            text_answer, _ = ask(TEXT_URL, prompt, grammar="root ::= [A-E]", max_tokens=1)
            seen = format_problem(problem, sees_image=True) + "\nAnswer:"
            vision_answer, timings = ask(VISION_URL, [image_part(images[(problem["id"], problem["language"])]),
                                                      {"type": "text", "text": seen}],
                                         grammar="root ::= [A-E]", max_tokens=1)
            rows.append((problem, text_answer, vision_answer, timings.get("prompt_n", 0)))
            writer.writerow([problem["id"], problem["level"], problem["label"], text_answer, vision_answer,
                             timings.get("prompt_n", 0)])
            out.flush()
            print(f"\r{n}/{len(problems)}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)

    n = len(rows)
    truth = Counter(r[0]["label"] for r in rows)
    top, top_count = truth.most_common(1)[0]
    print(f"{n} image-dependent problems ({args.language})")
    print(f"  always {top}:  {top_count}/{n} = {100 * top_count / n:.1f}%")
    for name, col in (("text", 1), ("vision", 2)):
        hits = sum(r[col] == r[0]["label"] for r in rows)
        letters = Counter(r[col] for r in rows)
        print(f"  {name:8} {hits}/{n} = {100 * hits / n:.1f}%   answers " + " ".join(f"{l}:{letters[l]}" for l in "ABCDE"))
    tokens = sorted(r[3] for r in rows)
    print(f"  vision prompt tokens: median {tokens[n // 2]}, max {tokens[-1]}")
    print("  true letters " + " ".join(f"{l}:{truth[l]}" for l in "ABCDE"))
    print(f"Details: {out_path}")


if __name__ == "__main__":
    main()
