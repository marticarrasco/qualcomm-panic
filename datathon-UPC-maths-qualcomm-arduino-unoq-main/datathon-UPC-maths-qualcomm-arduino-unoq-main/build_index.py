#!/usr/bin/env python3
"""Embed the training problems into a SQLite index for retrieval (RAG).

    python3 build_index.py                       # data/train.jsonl -> data/index.db
    python3 build_index.py --text-only --languages eng

Only train problems with all five OCR options are indexed. Run it once on
the board after copying the training data.
"""
import argparse
import json
import os
import tempfile
import time

from copilot import rag
from copilot.arithmetic import complete_options


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="data/train.jsonl")
    p.add_argument("--db", default="data/index.db")
    p.add_argument("--languages", default="", help="e.g. eng,cat (default: all in the file)")
    p.add_argument("--only-with-options", action="store_true", help="legacy flag; always enabled")
    p.add_argument("--text-only", action="store_true")
    args = p.parse_args()

    languages = set(filter(None, args.languages.split(",")))
    with open(args.input, encoding="utf-8-sig") as f:
        problems = [json.loads(line) for line in f if line.strip()]
    problems = [x for x in problems
                if (not languages or x["language"] in languages)
                and complete_options(x)
                and (not args.text_only or x.get("text_only") is True)]
    start = time.perf_counter()
    if not problems:
        p.error("no matching training problems with complete OCR options")
    directory = os.path.dirname(os.path.abspath(args.db))
    fd, staged = tempfile.mkstemp(prefix=".embed-", suffix=".db", dir=directory)
    os.close(fd)
    try:
        for done in rag.build(staged, problems):
            rate = done / (time.perf_counter() - start)
            print(f"\r{done}/{len(problems)} problems embedded, {rate:.1f}/s", end="", flush=True)
        os.replace(staged, args.db)
    finally:
        if os.path.exists(staged):
            os.unlink(staged)
    size = os.path.getsize(args.db) / 1e6
    print(f"\nWrote {args.db} ({size:.1f} MB) in {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
