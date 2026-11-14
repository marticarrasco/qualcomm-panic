#!/usr/bin/env python3
"""Is the time left after a fast first answer worth a second, different attempt?

    python3 experiments/second_pass.py run --set val      # second passes for the validation first pass
    python3 experiments/second_pass.py run --set train    # fresh sample: first pass, then second passes
    python3 experiments/second_pass.py report --set val   # tables only, no model calls
    python3 experiments/second_pass.py report --set train

Pass 1 is the shipped terse reasoning (system prompt SHOTS of reasoning_budget.py).
Its board time is (problem tokens)/READ + written/WRITE; the system prompt is cached.
A second pass gets only the output tokens the board could write in what is left of
TOTAL seconds after reading the text that is new for that pass:

  rev     same question, options reversed and re-lettered (new text: the options)
  check   problem + "Proposed answer: <value>", verify in at most 4 lines (new: that suffix)
  elim    problem + "cross out the options that cannot be right" (new: that suffix)
  checkc  the check asked as a follow-up turn after the first reasoning (new: the question)

A second pass is only run when pass 1 reached 'Answer: X' and at least MIN_TOKENS fit.
Standard library only. Resumable: rows already in the output file are not asked again.
"""
import argparse
import csv
import json
import os
import random
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from copilot.client import post
from copilot.llm import format_problem
from copilot.priors import _number, guess
from experiments.reasoning_budget import ANSWER_RE, SHOTS

READ, WRITE = 11.0, 5.6
TOTAL = 57.0          # the deadline of copilot/reasoner.py
PASS1_LIMIT = 55.0    # what reasoning_budget.py gives the first pass
OVERHEAD = 12         # chat template tokens around a message, as in reasoning_budget.py
MIN_TOKENS = 20
LETTERS = "ABCDE"
SEED = 42
VARIANTS = ["rev", "check", "elim", "checkc"]
VAL_PASS1 = "results_local/budget-20261114-191250-08b.csv"
FIELDS = ["id", "variant", "label", "read_tokens", "budget", "written", "finished", "answer", "output"]

CHECK = ("Proposed answer: {value}\nCheck it by substitution or by recomputing, in at most 4 short lines. "
         "Then finish with 'Answer: X', the letter of the correct option.")
ELIM = ("Cross out the options that cannot be right, one line each. "
        "Then finish with 'Answer: X', the letter of the option that is left.")
CHECKC = ("Check that result by substitution or by recomputing a different way, in at most 4 short lines. "
          "Then finish with 'Answer: X', the letter of the correct option.")


def snapshot_path(which):
    return f"results_local/second_pass-{which}-problems.jsonl"


def load_problems(which, create=False):
    """The problems exactly as sent to the model, from our own snapshot.

    data/*.jsonl was replaced during this work (14 Nov 20:16), so the JSONL is only read
    once, to create the snapshot. Validation is restricted to the 147 ids of the existing
    first pass; the train sample is 150 problems drawn with a fixed seed, none of those ids.
    """
    path = snapshot_path(which)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    if not create:
        sys.exit(f"{path} is missing: run 'second_pass.py run --set {which}' first")
    with open("data/val.jsonl" if which == "val" else "data/train.jsonl", encoding="utf-8") as f:
        problems = [json.loads(line) for line in f if line.strip()]
    problems = [x for x in problems if x["language"] == "eng" and len(x.get("options") or {}) == 5]
    first = {r["id"] for r in read_rows(VAL_PASS1) if r["prompt"] == "shots"}
    if which == "val":
        problems = [x for x in problems if x["id"] in first]
    else:
        problems = random.Random(SEED).sample([x for x in problems if x["id"] not in first], 150)
    os.makedirs("results_local", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for x in problems:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    return problems


def read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def tokens(base, text):
    return len(post(base + "/tokenize", {"content": text}, timeout=60)["tokens"])


def chat(base, messages, max_tokens):
    response = post(base + "/v1/chat/completions", {
        "messages": messages, "max_tokens": max_tokens, "temperature": 0, "seed": SEED, "cache_prompt": True,
    }, timeout=600)
    text = response["choices"][0]["message"]["content"] or ""
    return text, response.get("timings", {}).get("predicted_n", 0)


def reversed_problem(problem):
    """Options in reverse order with new letters: the new A is the old E."""
    return dict(problem, options={LETTERS[i]: problem["options"][LETTERS[4 - i]] for i in range(5)})


def build(variant, problem, first):
    """Return (messages, new text the board has to read, letter map back to the original)."""
    system = {"role": "system", "content": SHOTS}
    user = format_problem(problem)
    same = {letter: letter for letter in LETTERS}
    if variant == "rev":
        turned = format_problem(reversed_problem(problem))
        new = turned[len(problem["text"].strip()):]
        return ([system, {"role": "user", "content": turned}], new,
                {LETTERS[i]: LETTERS[4 - i] for i in range(5)})
    if variant == "check":
        suffix = "\n" + CHECK.format(value=problem["options"][first["answer"]])
        return [system, {"role": "user", "content": user + suffix}], suffix, same
    if variant == "elim":
        suffix = "\n" + ELIM
        return [system, {"role": "user", "content": user + suffix}], suffix, same
    if variant == "checkc":
        return ([system, {"role": "user", "content": user}, {"role": "assistant", "content": first["output"]},
                 {"role": "user", "content": CHECKC}], CHECKC, same)
    raise ValueError(variant)


def pass1_seconds(first):
    return int(first["problem_tokens"]) / READ + int(first["written"]) / WRITE


def run(args):
    base = f"http://127.0.0.1:{args.port}"
    problems = load_problems(args.set, create=True)
    tokens_path = f"results_local/second_pass-{args.set}-pass1.csv"
    out_path = f"results_local/second_pass-{args.set}.csv"

    # Pass 1: validation answers come from the existing run, train answers are made here the same way.
    known = {r["id"]: r for r in read_rows(tokens_path)}
    old = {r["id"]: r for r in read_rows(VAL_PASS1) if r["prompt"] == "shots"} if args.set == "val" else {}
    fields1 = ["id", "label", "problem_tokens", "budget", "written", "finished", "answer", "output"]
    with open(tokens_path, "a", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fields1)
        if not known:
            writer.writeheader()
        for n, problem in enumerate(problems, 1):
            user = format_problem(problem)
            if problem["id"] in known:
                if args.verify and tokens(base, user) + OVERHEAD != int(known[problem["id"]]["problem_tokens"]):
                    sys.exit(f"{problem['id']}: the snapshot is not the text the first pass was run on")
                continue
            count = tokens(base, user) + OVERHEAD
            if args.set == "val":
                row = {k: old[problem["id"]][k] for k in ("id", "label", "budget", "written", "finished",
                                                          "answer", "output")}
            else:
                budget = max(int((PASS1_LIMIT - count / READ) * WRITE), 1)
                text, written = chat(base, [{"role": "system", "content": SHOTS},
                                            {"role": "user", "content": user}], budget)
                found = ANSWER_RE.findall(text)
                row = {"id": problem["id"], "label": problem["label"], "budget": budget, "written": written,
                       "finished": int(bool(found)), "answer": found[-1].upper() if found else "", "output": text}
            row["problem_tokens"] = count
            writer.writerow(row)
            out.flush()
            known[problem["id"]] = row
            print(f"\rpass 1: {n}/{len(problems)}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)

    done = {(r["id"], r["variant"]) for r in read_rows(out_path)}
    with open(out_path, "a", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=FIELDS)
        if not done:
            writer.writeheader()
        for variant in args.variants.split(","):
            for n, problem in enumerate(problems, 1):
                first = known[problem["id"]]
                if (problem["id"], variant) in done or not int(first["finished"]):
                    continue
                messages, new, back = build(variant, problem, first)
                read = tokens(base, new) + OVERHEAD
                budget = int((TOTAL - pass1_seconds(first) - read / READ) * WRITE)
                row = {"id": problem["id"], "variant": variant, "label": problem["label"], "read_tokens": read,
                       "budget": budget, "written": 0, "finished": 0, "answer": "", "output": ""}
                if budget >= MIN_TOKENS:
                    text, written = chat(base, messages, budget)
                    found = ANSWER_RE.findall(text)
                    row.update(written=written, finished=int(bool(found)), output=text,
                               answer=back[found[-1].upper()] if found else "")
                writer.writerow(row)
                out.flush()
                print(f"\r{variant}: {n}/{len(problems)}", end="", file=sys.stderr, flush=True)
            print(file=sys.stderr)
    print(f"Wrote {tokens_path} and {out_path}")


# ---------------------------------------------------------------- report

def quantiles(values):
    values = sorted(values)
    pick = lambda q: values[min(int(q * len(values)), len(values) - 1)]
    return (f"min {values[0]:.1f}  p10 {pick(.1):.1f}  p25 {pick(.25):.1f}  median {pick(.5):.1f}  "
            f"p75 {pick(.75):.1f}  p90 {pick(.9):.1f}  max {values[-1]:.1f}  mean {sum(values) / len(values):.1f}")


def is_extreme(problem, letter):
    """True/False when the five options are distinct numbers, None when they are not."""
    values = [_number(problem["options"][x]) for x in LETTERS]
    if any(v is None for v in values) or len(set(values)) != 5:
        return None
    return values[LETTERS.index(letter)] in (min(values), max(values))


def rules(problem, a1, second):
    """Answers of every decision rule. a1 is the pass-1 letter or None; second is a row or None."""
    prior = guess(problem)
    base = a1 or prior                      # pass 1, the prior (D here) when it did not finish
    a2 = second["answer"] if second and int(second["finished"]) else None
    out = {"keep pass 1": base, "take pass 2 when it finishes": a2 or base,
           "agree, else prior": base, "agree, else D": base, "disagree: prefer non-extreme": base}
    if a1 and a2 and a1 != a2:
        out["agree, else prior"] = prior
        out["agree, else D"] = "D"
        e1, e2 = is_extreme(problem, a1), is_extreme(problem, a2)
        out["disagree: prefer non-extreme"] = a2 if (e1 and e2 is False) else a1
    return out


def report(args):
    problems = load_problems(args.set)
    first = {r["id"]: r for r in read_rows(f"results_local/second_pass-{args.set}-pass1.csv")}
    second = {(r["id"], r["variant"]): r for r in read_rows(f"results_local/second_pass-{args.set}.csv")}
    problems = [p for p in problems if p["id"] in first]
    n = len(problems)
    label = {p["id"]: p["label"] for p in problems}
    a1 = {p["id"]: (first[p["id"]]["answer"] or None) for p in problems}
    fires = {p["id"]: guess(p) != "D" for p in problems}
    t1 = {p["id"]: pass1_seconds(first[p["id"]]) for p in problems}

    print(f"=== {args.set}: {n} English problems with five options, as sent to the model "
          f"({snapshot_path(args.set)}) ===")
    print(f"always D {sum(v == 'D' for v in label.values())}, prior only "
          f"{sum(guess(p) == p['label'] for p in problems)}, pass 1 finished "
          f"{sum(v is not None for v in a1.values())}, mean written "
          f"{sum(int(first[i]['written']) for i in first) / n:.0f}")
    print(f"model alone, D when unfinished {sum((a1[i] or 'D') == label[i] for i in label)}; "
          f"'keep pass 1' below is the model, else the prior, on every problem; "
          f"'chain' puts the prior first where it fires ({sum(fires.values())} problems)")

    print("\n1. Budget accounting (pass 1 = problem tokens/11 + written/5.6, out of 57 s)")
    left = [TOTAL - t1[p["id"]] for p in problems]
    print("   pass 1 seconds, all:      ", quantiles(list(t1.values())))
    print("   seconds left, all:        ", quantiles(left))
    done = [TOTAL - t1[p["id"]] for p in problems if a1[p["id"]]]
    print(f"   seconds left, finished {len(done)}:", quantiles(done))
    print("   problems where a second pass of N output tokens fits (pass 1 finished), by new text to read:")
    print(f"   {'N tokens':>9} " + " ".join(f"{'+' + str(r) + ' read':>10}" for r in (0, 30, 45, 60)))
    for want in (20, 40, 60, 80, 100, 120, 150, 200):
        cells = [sum(a1[p["id"]] is not None and (TOTAL - t1[p["id"]] - r / READ) * WRITE >= want
                     for p in problems) for r in (0, 30, 45, 60)]
        print(f"   {want:9} " + " ".join(f"{c:10}" for c in cells))
    for variant in VARIANTS:
        mine = [r for (i, v), r in second.items() if v == variant and i in first]
        if mine:
            ran = [r for r in mine if int(r["budget"]) >= MIN_TOKENS]
            print(f"   {variant:7} new text {sum(int(r['read_tokens']) for r in mine) / len(mine):5.1f} tokens "
                  f"mean; fits (>= {MIN_TOKENS} tokens) {len(ran)}; mean budget "
                  f"{sum(int(r['budget']) for r in ran) / max(len(ran), 1):.0f}, mean written "
                  f"{sum(int(r['written']) for r in ran) / max(len(ran), 1):.0f}, reached Answer "
                  f"{sum(int(r['finished']) for r in ran)}")

    names = ["keep pass 1", "take pass 2 when it finishes", "agree, else prior", "agree, else D",
             "disagree: prefer non-extreme"]
    summary = []
    for variant in VARIANTS:
        rows = {p["id"]: second.get((p["id"], variant)) for p in problems}
        if not any(rows.values()):
            continue
        ran = [p for p in problems if rows[p["id"]] and int(rows[p["id"]]["budget"]) >= MIN_TOKENS]
        fin = [p for p in ran if int(rows[p["id"]]["finished"])]
        print(f"\n2-3. Variant '{variant}': second pass ran on {len(ran)}, reached Answer on {len(fin)}")
        right2 = sum(rows[p["id"]]["answer"] == p["label"] for p in fin)
        right1 = sum(a1[p["id"]] == p["label"] for p in fin)
        print(f"   on those {len(fin)}: pass 2 right {right2}, pass 1 right {right1}; pass 2 letters "
              f"{dict(sorted(Counter(rows[p['id']]['answer'] for p in fin).items()))}")
        agree = [p for p in fin if rows[p["id"]]["answer"] == a1[p["id"]]]
        differ = [p for p in fin if rows[p["id"]]["answer"] != a1[p["id"]]]
        unfinished = [p for p in ran if p not in fin]
        for name, group in (("agree", agree), ("disagree", differ), ("pass 2 unfinished", unfinished)):
            ok1 = sum(a1[p["id"]] == p["label"] for p in group)
            ok2 = sum(rows[p["id"]]["answer"] == p["label"] for p in group)
            okp = sum(guess(p) == p["label"] for p in group)
            print(f"   {name:18} n={len(group):3}  pass 1 right {ok1:3} = {100 * ok1 / max(len(group), 1):3.0f}%"
                  f"   pass 2 right {ok2:3}   prior right {okp:3}")
        print(f"   {'rule':30} {'all ' + str(n):>8} {'ran ' + str(len(ran)):>8}   fixed/broken vs keep   "
              f"{'chain':>6}  fixed/broken vs chain")
        answers = {p["id"]: rules(p, a1[p["id"]], rows[p["id"]]) for p in problems}
        for name in names:
            alone = {p["id"]: answers[p["id"]][name] for p in problems}
            keep = {p["id"]: answers[p["id"]]["keep pass 1"] for p in problems}
            chain = {p["id"]: guess(p) if fires[p["id"]] else alone[p["id"]] for p in problems}
            chain0 = {p["id"]: guess(p) if fires[p["id"]] else keep[p["id"]] for p in problems}
            score = lambda d, group=problems: sum(d[p["id"]] == p["label"] for p in group)
            fixed = sum(alone[i] == label[i] != keep[i] for i in label)
            broken = sum(keep[i] == label[i] != alone[i] for i in label)
            cfixed = sum(chain[i] == label[i] != chain0[i] for i in label)
            cbroken = sum(chain0[i] == label[i] != chain[i] for i in label)
            print(f"   {name:30} {score(alone):8} {score(alone, ran):8}   +{fixed:2} / -{broken:2}"
                  f"{'':14} {score(chain):6}  +{cfixed:2} / -{cbroken:2}")
            summary.append((variant, name, score(alone), score(chain), cfixed, cbroken))
        # Board seconds of the chain: the prior answers at 0 s, the rest run pass 1 (+ pass 2 where it ran).
        both, single = [], []
        for p in problems:
            if fires[p["id"]]:
                both.append(0.0)
                single.append(0.0)
                continue
            extra = 0.0
            if p in ran:
                extra = int(rows[p["id"]]["read_tokens"]) / READ + int(rows[p["id"]]["written"]) / WRITE
            single.append(min(t1[p["id"]], TOTAL))
            both.append(min(t1[p["id"]] + extra, TOTAL))
        calls = [s for s in single if s]
        print(f"   board seconds in the chain: pass 1 only mean {sum(single) / n:.1f} (model calls "
              f"{sum(calls) / len(calls):.1f}), worst {max(single):.1f}; with this second pass mean "
              f"{sum(both) / n:.1f} (model calls {sum(b for b in both if b) / len(calls):.1f}), "
              f"worst {max(both):.1f}; problems over 50 s: {sum(s > 50 for s in single)} -> "
              f"{sum(b > 50 for b in both)}")

    if summary:
        base = sum((guess(p) if fires[p["id"]] else (a1[p["id"]] or "D")) == p["label"] for p in problems)
        print(f"\n4. Inside the shipped chain (prior where it fires, else model, else D): baseline {base}/{n}")
        for variant, name, alone, chain, fixed, broken in summary:
            if name != "keep pass 1":
                print(f"   {variant:7} {name:30} {chain:3}/{n}  ({chain - base:+d}: fixed {fixed}, broken {broken})")
        print("   With n near 150 a difference under about 6 problems is noise.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["run", "report"])
    p.add_argument("--set", choices=["val", "train"], default="val")
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--port", default="8081")
    p.add_argument("--verify", action="store_true",
                   help="check that each snapshot problem has the token count saved with its first pass")
    args = p.parse_args()
    (run if args.command == "run" else report)(args)


if __name__ == "__main__":
    main()
