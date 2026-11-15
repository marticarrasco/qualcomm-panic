"""Probe: unknown + pieces + one search line, instead of a whole program.

The model names the unknown and its range, writes each fact of the problem as one Python condition
(as comment lines), and ends with a single print line that searches the range. A grammar forces that
shape, so the answer is short and stops by itself. The print line runs in the same sandbox as the
other probes, and the printed value is matched against the options.

    python3 cond_probe.py --input data/demo/train.jsonl --out results/cond-demo.jsonl --shots 3
    python3 cond_probe.py --url http://127.0.0.1:8081 ...        # on the board

Standard library only. The examples are made up: they are not from M3Kang or from the practice set.
"""
import argparse
import ast
import json
import os
import random
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pot_rerun import parse_num, run  # noqa: E402  (the tolerant sandbox runner)

LETTERS = "ABCDE"
SYSTEM = (
    "You turn math problems into Python. Reply in exactly this form:\n"
    "# unknown: the unknown and its range\n"
    "# piece: one fact of the problem as a Python condition (one line per fact)\n"
    "print(...)  one line of Python that searches the range and prints the answer\n"
    "Use range, len, sum, min, max, str and comprehensions. No imports."
)
INSTRUCTION = "Write the unknown, each fact as a Python condition, and one print line that finds the answer."
SHOTS = [
    ("Tom thinks of a number. He doubles it, adds 7 and gets 45. What is his number?",
     "# unknown: x = Tom's number, from 1 to 100\n"
     "# piece: 2 * x + 7 == 45\n"
     "print([x for x in range(1, 101) if 2 * x + 7 == 45])"),
    ("How many whole numbers from 1 to 200 are divisible by 3 but not by 6?",
     "# unknown: how many n from 1 to 200\n"
     "# piece: n % 3 == 0\n"
     "# piece: n % 6 != 0\n"
     "print(len([n for n in range(1, 201) if n % 3 == 0 and n % 6 != 0]))"),
    ("En una granja hi ha gallines i conills. En total hi ha 20 caps i 56 potes. Quants conills hi ha?",
     "# unknown: g = gallines, c = conills, from 0 to 20\n"
     "# piece: g + c == 20\n"
     "# piece: 2 * g + 4 * c == 56\n"
     "print([c for g in range(21) for c in range(21) if g + c == 20 and 2 * g + 4 * c == 56])"),
]
GRAMMAR = (
    'root ::= "# unknown: " line piece piece? piece? piece? piece? "print(" line\n'
    'piece ::= "# piece: " line\n'
    'line ::= [^\\n]+ "\\n"\n'
)
COUNT_QUESTION = re.compile(r"\b(how many|quants|quantes)\b", re.IGNORECASE)

# The same idea without the pieces: only the search line.
SYSTEM_LINE = (
    "You turn math problems into Python. Reply with exactly one line: print(...) that searches a range "
    "with a comprehension and prints the answer. Use range, len, sum, min, max, str. No imports."
)
INSTRUCTION_LINE = "Write one print line that finds the answer."
GRAMMAR_LINE = 'root ::= "print(" [^\\n]+ "\\n"\n'


def post(url, payload, timeout):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def read_printed(printed):
    """("number", value) or ("list", distinct numbers, how many items), or None."""
    if printed is None:
        return None
    text = printed.strip()
    try:
        obj = ast.literal_eval(text)
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        value = parse_num(text)  # "1/5" from a Fraction, "48 "
        return ("number", value) if value is not None else None
    def number(x):
        try:
            return float(x)
        except OverflowError:  # an integer too large to be an option
            return None

    if isinstance(obj, bool):
        return None
    if isinstance(obj, (int, float)):
        value = number(obj)
        return ("number", value) if value is not None else None
    if isinstance(obj, (list, tuple, set, frozenset)):
        numbers = sorted({number(x) for x in obj
                          if isinstance(x, (int, float)) and not isinstance(x, bool) and number(x) is not None})
        return ("list", numbers, len(obj))
    return None


def decide(printed, nums, text):
    """Return (letters that match, how). A list is read as its only value, as a count when the
    question asks how many, or as the set of values that satisfy the conditions."""
    parsed = read_printed(printed)
    if parsed is None:
        return [], "nothing"

    def hits(value):
        return [k for k, v in nums.items() if abs(value - v) < 1e-6]

    if parsed[0] == "number":
        return hits(parsed[1]), "number"
    _kind, numbers, size = parsed
    if size == 0:
        return [], "empty list"
    tries = []
    if len(numbers) == 1:
        tries.append((hits(numbers[0]), "single value"))
    if COUNT_QUESTION.search(text):
        tries.append((hits(float(size)), "count"))
    if len(numbers) > 1:
        tries.append(([k for k, v in nums.items() if any(abs(v - n) < 1e-6 for n in numbers)], "options in list"))
    for letters, how in tries:
        if len(letters) == 1:
            return letters, how
    return (tries[0] if tries else ([], "list"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8091")
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--languages", default="eng,cat")
    p.add_argument("--shots", type=int, default=3, help="worked examples as earlier turns (0 to 3)")
    p.add_argument("--max-tokens", type=int, default=140)
    p.add_argument("--limit", type=int, default=0, help="ask at most this many problem ids (0 = all)")
    p.add_argument("--form", default="pieces", choices=("pieces", "line"),
                   help="pieces: unknown, facts and the search line; line: only the search line")
    args = p.parse_args()
    system, instruction, grammar = SYSTEM, INSTRUCTION, GRAMMAR
    shots = SHOTS
    if args.form == "line":
        system, instruction, grammar = SYSTEM_LINE, INSTRUCTION_LINE, GRAMMAR_LINE
        shots = [(q, a.splitlines()[-1]) for q, a in SHOTS]
    langs = set(args.languages.split(","))
    with open(args.input, encoding="utf-8") as f:
        problems = [json.loads(line) for line in f if line.strip()]
    problems = [x for x in problems if x["language"] in langs]
    if args.limit:  # the same sample of ids as pot_probe.py --limit
        ids = sorted({x["id"] for x in problems if x["text_only"] and len(x.get("options") or {}) == 5
                      and all(parse_num((x.get("options") or {}).get(k, "")) is not None for k in LETTERS)})
        random.Random(42).shuffle(ids)
        keep = set(ids[:args.limit])
        problems = [x for x in problems if x["id"] in keep]

    prefix = [{"role": "system", "content": system}]
    for question, answer in shots[:args.shots]:
        prefix += [{"role": "user", "content": question + "\n" + instruction},
                   {"role": "assistant", "content": answer}]

    t0 = time.time()
    rows = []
    with open(args.out, "w", encoding="utf-8") as out:
        for problem in problems:
            options = problem.get("options") or {}
            nums = {k: parse_num(options.get(k, "")) for k in LETTERS}
            numeric = len(options) == 5 and all(v is not None for v in nums.values())
            row = {"id": problem["id"], "language": problem["language"], "level": problem["level"],
                   "text_only": problem["text_only"], "label": problem["label"]}
            if not numeric or not problem["text_only"]:
                row["status"] = "skipped"
                rows.append(row)
                out.write(json.dumps(row) + "\n")
                continue
            messages = prefix + [{"role": "user", "content": problem["text"].strip() + "\n" + instruction}]
            resp = post(args.url + "/v1/chat/completions", {
                "messages": messages, "grammar": grammar, "max_tokens": args.max_tokens,
                "temperature": 0, "seed": 42, "cache_prompt": True}, 300)
            code = resp["choices"][0]["message"]["content"].strip()
            t = resp.get("timings", {})
            printed, status = run(code)
            matches, how = decide(printed, nums, problem["text"])
            row.update({"status": status, "printed": (printed or "")[:200], "matches": matches, "how": how,
                        "finish": resp["choices"][0].get("finish_reason"),
                        "read_n": t.get("prompt_n") or 0, "cache_n": t.get("cache_n") or 0,
                        "predicted_n": t.get("predicted_n") or 0, "code": code})
            rows.append(row)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    asked = [r for r in rows if r["status"] != "skipped"]
    ran = [r for r in asked if r["status"] == "ok" and r["how"] != "nothing"]
    matched = [r for r in asked if len(r["matches"]) == 1]
    right = [r for r in matched if r["matches"][0] == r["label"]]
    print(f"asked {len(asked)} of {len(rows)} rows, form '{args.form}', {args.shots} examples")
    print(f"the line ran and printed something usable: {len(ran)}; one option matched: {len(matched)}; "
          f"right: {len(right)} ({100 * len(right) / max(len(matched), 1):.0f}% of matches, "
          f"{100 * len(right) / max(len(asked), 1):.1f}% of asked)")
    for lang in sorted({r["language"] for r in asked}):
        a = [r for r in asked if r["language"] == lang]
        m = [r for r in a if len(r["matches"]) == 1]
        print(f"  {lang}: asked {len(a)}, matched {len(m)}, right {sum(r['matches'][0] == r['label'] for r in m)}")
    how = {}
    for r in matched:
        ok = r["matches"][0] == r["label"]
        how.setdefault(r["how"], [0, 0])
        how[r["how"]][0] += 1
        how[r["how"]][1] += ok
    print("  how the answer was read (matched, right):", {k: tuple(v) for k, v in how.items()})
    status = {}
    for r in asked:
        key = r["status"] if r["status"] in ("ok", "timeout") else r["status"].split(":")[0][:24]
        status[key] = status.get(key, 0) + 1
    print("  status:", status, "| cut at max_tokens:", sum(r["finish"] == "length" for r in asked))
    later = asked[1:] or asked
    est = [r["read_n"] / 11 + r["predicted_n"] / 5.5 for r in later]
    print(f"  tokens after the first request: reused {sum(r['cache_n'] for r in later) / len(later):.0f}, "
          f"read {sum(r['read_n'] for r in later) / len(later):.0f}, written {sum(r['predicted_n'] for r in later) / len(later):.0f} "
          f"(max {max(r['predicted_n'] for r in later)}); estimated board time mean {sum(est) / len(est):.0f} s, max {max(est):.0f} s")
    print(f"  laptop time {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
