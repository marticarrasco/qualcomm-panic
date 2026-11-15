#!/usr/bin/env python3
"""Compare bake-off runs of reasoning_budget.py against the Qwen3.5 0.8B Q8_0 reference.

    python3 experiments/model_bakeoff_compare.py results_local/budget-*-bakeoff-*.csv

Board seconds are reconstructed from the budget column:
read time = 55 - budget / write_tps, write time = written / write_tps
(write_tps is passed per file as name=tps via --write, otherwise parsed from the run log).
"""
import csv
import math
import re
import sys
from collections import defaultdict

csv.field_size_limit(10 ** 9)
REF = "results_local/budget-20261114-191250-08b.csv"
FALLBACK = "D"


def load(path):
    with open(path, newline="") as f:
        return {r["id"]: r for r in csv.DictReader(f) if r["prompt"] == "shots"}


def right(r):
    return (r["answer"] or FALLBACK) == r["label"]


def speeds():
    out = {}
    try:
        for line in open("results_local/bakeoff-run.log"):
            m = re.match(r"== (\S+) read ([\d.]+) write ([\d.]+)", line)
            if m:
                out[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    except FileNotFoundError:
        pass
    return out


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    # --ref FILE: use another reference run (e.g. the 0.8B Q8_0 rerun on the current 209-problem val set).
    # Comparisons are always paired on the problem ids present in both runs.
    ref_path = REF
    if "--ref" in sys.argv:
        at = sys.argv.index("--ref")
        ref_path = sys.argv[at + 1]
        del sys.argv[at:at + 2]
    ref = load(ref_path)
    sp = speeds()
    print(f"reference file: {ref_path}")
    print(f"reference: n={len(ref)} finished={sum(r['finished'] == '1' for r in ref.values())} "
          f"right={sum(right(r) for r in ref.values())}")
    for path in sys.argv[1:]:
        run = load(path)
        tag = re.search(r"bakeoff-(.+)\.csv", path).group(1)
        ids = [i for i in ref if i in run]
        rows = [run[i] for i in ids]
        done = [r for r in rows if r["finished"] == "1"]
        ok_done = sum(r["answer"] == r["label"] for r in done)
        ok = sum(right(r) for r in rows)
        ref_ok = sum(right(ref[i]) for i in ids)
        both = sum(right(run[i]) and right(ref[i]) for i in ids)
        only_new = sum(right(run[i]) and not right(ref[i]) for i in ids)
        only_ref = sum(right(ref[i]) and not right(run[i]) for i in ids)
        neither = len(ids) - both - only_new - only_ref
        written = [int(r["written"]) for r in rows]
        line = (f"\n{tag}: n={len(ids)} finished {len(done)} right-if-finished {ok_done}/{len(done)} "
                f"right overall {ok}/{len(ids)} ({100 * ok / len(ids):.1f}%) ref on same {ref_ok} | "
                f"both {both} only-new {only_new} only-ref {only_ref} neither {neither} "
                f"McNemar exact p={mcnemar_exact(only_new, only_ref):.3f} | mean tokens {sum(written) / len(rows):.0f}")
        if tag in sp:
            _, w = sp[tag]
            secs = [55 - int(r["budget"]) / w + int(r["written"]) / w for r in rows]
            line += f" | est board s mean {sum(secs) / len(secs):.1f} worst {max(secs):.1f}"
        print(line)
        answers = defaultdict(int)
        for r in done:
            answers[r["answer"]] += 1
        print("   answer letters when finished:", dict(sorted(answers.items())))
        by = defaultdict(list)
        for r in rows:
            by[r["level"]].append(r)
        print("   by level (right/n, finished; ref right): " + "; ".join(
            f"L{lv}: {sum(right(r) for r in g)}/{len(g)}, {sum(r['finished'] == '1' for r in g)} fin; "
            f"ref {sum(right(ref[r['id']]) for r in g)}" for lv, g in sorted(by.items())))


if __name__ == "__main__":
    main()
