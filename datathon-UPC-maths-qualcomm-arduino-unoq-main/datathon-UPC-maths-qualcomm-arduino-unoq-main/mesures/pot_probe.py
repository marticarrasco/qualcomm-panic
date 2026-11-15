"""Probe: can the model write a short brute-force Python program whose output is one of the options?
The program runs in a child process with a whitelist of names, no imports, and a time limit.
Usage: python -I pot_probe.py --input demo.jsonl --out pot.jsonl [--url ...] [--max-tokens 160]"""
import argparse
import ast
import json
import math
import re
import subprocess
import sys
import time
import urllib.request

LETTERS = "ABCDE"
SYSTEM = ("You write short Python programs that solve math problems by brute force. "
          "Reply with Python code only. The program must print one number: the answer.")

# Made-up examples (not from M3Kang or from the practice set): three brute-force idioms, one in Catalan.
FEWSHOT = [
    ("How many whole numbers from 1 to 200 are divisible by 3 but not by 6?",
     "print(sum(1 for n in range(1, 201) if n % 3 == 0 and n % 6 != 0))"),
    ("Using each of the digits 2, 5 and 8 exactly once, how many three-digit numbers greater than 500 can be written?",
     "count = 0\nfor p in permutations([2, 5, 8]):\n    n = p[0] * 100 + p[1] * 10 + p[2]\n    if n > 500:\n        count += 1\nprint(count)"),
    ("En Pau pensa un nombre. El multiplica per 4, hi resta 6 i obté 30. Quin nombre ha pensat?",
     "for x in range(1, 1000):\n    if 4 * x - 6 == 30:\n        print(x)\n        break"),
]

RUNNER = r'''
import ast, sys, math, itertools, fractions
src = sys.stdin.read()
tree = ast.parse(src)
BAD_NODES = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.With, ast.AsyncWith, ast.Try,
             ast.Raise, ast.ClassDef, ast.AsyncFunctionDef, ast.Await, ast.Yield, ast.YieldFrom, ast.Delete)
ALLOWED = {"range", "len", "sum", "min", "max", "abs", "int", "float", "str", "set", "list", "tuple", "dict",
           "sorted", "reversed", "enumerate", "zip", "map", "filter", "any", "all", "pow", "divmod", "round",
           "print", "bool", "math", "itertools", "Fraction", "True", "False", "None", "ord", "chr", "frozenset",
           "permutations", "combinations", "product", "gcd", "factorial", "sqrt", "comb", "isqrt", "lcm"}
defined = set()
for node in ast.walk(tree):
    if isinstance(node, BAD_NODES):
        sys.exit("blocked: " + type(node).__name__)
    if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
        sys.exit("blocked: dunder attribute")
    if isinstance(node, ast.FunctionDef):
        defined.add(node.name)
        for a in node.args.args:
            defined.add(a.arg)
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
        defined.add(node.id)
    if isinstance(node, ast.arg):
        defined.add(node.arg)
    if isinstance(node, ast.comprehension):
        for n in ast.walk(node.target):
            if isinstance(n, ast.Name):
                defined.add(n.id)
for node in ast.walk(tree):
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in ALLOWED and node.id not in defined:
        sys.exit("blocked: name " + node.id)
safe = {k: getattr(__builtins__, k) for k in ("range", "len", "sum", "min", "max", "abs", "int", "float", "str",
        "set", "list", "tuple", "dict", "sorted", "reversed", "enumerate", "zip", "map", "filter", "any", "all",
        "pow", "divmod", "round", "print", "bool", "ord", "chr", "frozenset")}
env = {"__builtins__": safe, "math": math, "itertools": itertools, "Fraction": fractions.Fraction,
       "permutations": itertools.permutations, "combinations": itertools.combinations, "product": itertools.product,
       "gcd": math.gcd, "factorial": math.factorial, "sqrt": math.sqrt, "comb": math.comb, "isqrt": math.isqrt,
       "lcm": math.lcm}
exec(compile(tree, "<model>", "exec"), env)
'''


def post(url, payload, timeout):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def parse_num(val):
    s = str(val).strip().replace(",", ".").rstrip("%").strip()
    s = s.replace("−", "-").replace("–", "-").replace("一", "-")
    if s in ("O", "o"):
        s = "0"
    if re.match(r"^[-+]?\d+(\.\d+)?$", s):
        return float(s)
    m = re.match(r"^([-+]?\d+)\s*/\s*(\d+)$", s)
    if m and float(m.group(2)) != 0:
        return float(m.group(1)) / float(m.group(2))
    return None


def extract_code(text):
    m = re.search(r"```(?:python)?\s*\n(.*?)(?:```|$)", text, re.S)
    code = m.group(1) if m else text
    lines = [l for l in code.splitlines() if not re.match(r"\s*(import |from \S+ import )", l)]
    return "\n".join(lines).strip()


def run(code, timeout=5):
    try:
        r = subprocess.run([sys.executable, "-I", "-X", "utf8", "-c", RUNNER], input=code, capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    if r.returncode != 0:
        return None, ((r.stderr or "").strip().splitlines() or ["error"])[-1][:80]
    out = r.stdout.strip().splitlines()
    return (out[-1].strip() if out else None), "ok"


def to_value(printed):
    if printed is None:
        return None
    v = parse_num(printed)
    if v is not None:
        return v
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:\s*/\s*\d+)?", printed)
    return parse_num(m.group(0)) if m else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8091")
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--languages", default="eng,cat")
    p.add_argument("--max-tokens", type=int, default=160)
    p.add_argument("--with-options", action="store_true", help="show the options to the model")
    p.add_argument("--fewshot", action="store_true", help="three made-up worked examples as earlier turns")
    p.add_argument("--limit", type=int, default=0, help="ask at most this many problem ids (0 = all)")
    args = p.parse_args()
    langs = set(args.languages.split(","))
    with open(args.input, encoding="utf-8") as f:
        problems = [json.loads(line) for line in f if line.strip()]
    problems = [x for x in problems if x["language"] in langs]
    if args.limit:  # a fixed sample of problem ids, every language of each id
        import random
        ids = sorted({x["id"] for x in problems if x["text_only"] and len(x.get("options") or {}) == 5
                      and all(parse_num((x.get("options") or {}).get(k, "")) is not None for k in LETTERS)})
        random.Random(42).shuffle(ids)
        keep = set(ids[:args.limit])
        problems = [x for x in problems if x["id"] in keep]
    t_start = time.time()

    rows = []
    with open(args.out, "w", encoding="utf-8") as out:
        for problem in problems:
            options = problem.get("options") or {}
            nums = {k: parse_num(options.get(k, "")) for k in LETTERS}
            numeric = len(options) == 5 and all(v is not None for v in nums.values())
            row = {"id": problem["id"], "language": problem["language"], "level": problem["level"],
                   "text_only": problem["text_only"], "label": problem["label"], "numeric": numeric}
            if not numeric or not problem["text_only"]:
                row["status"] = "skipped"
                rows.append(row)
                out.write(json.dumps(row) + "\n")
                continue
            user = problem["text"].strip()
            if args.with_options:
                user += "\nOptions: " + ", ".join(str(options[k]) for k in LETTERS)
            if args.fewshot:
                messages = [{"role": "system", "content": SYSTEM}]
                for q, a in FEWSHOT:
                    messages += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
                messages.append({"role": "user", "content": user})
            else:
                user += "\nWrite a short Python program that computes the answer and prints it."
                messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
            start = time.perf_counter()
            resp = post(args.url + "/v1/chat/completions", {
                "messages": messages,
                "max_tokens": args.max_tokens, "temperature": 0, "seed": 42, "cache_prompt": True}, 300)
            text = resp["choices"][0]["message"]["content"]
            t = resp.get("timings", {})
            code = extract_code(text)
            printed, status = run(code)
            value = to_value(printed)
            matches = [k for k, v in nums.items() if value is not None and abs(value - v) < 1e-6]
            row.update({"status": status, "printed": printed, "value": value, "matches": matches,
                        "finish": resp["choices"][0].get("finish_reason"),
                        "prompt_n": (t.get("prompt_n") or 0) + (t.get("cache_n") or 0),
                        "read_n": t.get("prompt_n") or 0, "cache_n": t.get("cache_n") or 0,
                        "predicted_n": t.get("predicted_n"), "laptop_s": round(time.perf_counter() - start, 2),
                        "code": code})
            rows.append(row)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    asked = [r for r in rows if r["status"] != "skipped"]
    ran = [r for r in asked if r["status"] == "ok" and r["value"] is not None]
    matched = [r for r in asked if len(r.get("matches") or []) == 1]
    right = [r for r in matched if r["matches"][0] == r["label"]]
    print(f"asked {len(asked)} (text-only, numeric options) of {len(rows)} rows")
    print(f"program ran and printed a number: {len(ran)}; exactly one option matched: {len(matched)}; "
          f"of those right: {len(right)} ({100 * len(right) / max(len(matched), 1):.0f}% precision, "
          f"{100 * len(right) / max(len(asked), 1):.1f}% of asked)")
    print("status:", dict((s, sum(1 for r in asked if r["status"].startswith(s))) for s in ("ok", "timeout", "blocked")),
          "| other errors:", sum(1 for r in asked if not r["status"].startswith(("ok", "timeout", "blocked"))))
    print("finish reasons:", {f: sum(1 for r in asked if r.get("finish") == f) for f in {r.get("finish") for r in asked}})
    if asked:
        est = [r["read_n"] / 11 + (r["predicted_n"] or 0) / 5.5 for r in asked[1:]] or [0]
        print(f"tokens: prompt mean {sum(r['prompt_n'] for r in asked) / len(asked):.0f} "
              f"(reused from cache: mean {sum(r['cache_n'] for r in asked[1:]) / max(len(asked) - 1, 1):.0f}, "
              f"really read: mean {sum(r['read_n'] for r in asked[1:]) / max(len(asked) - 1, 1):.0f}), generated mean "
              f"{sum(r['predicted_n'] or 0 for r in asked) / len(asked):.0f}, max {max(r['predicted_n'] or 0 for r in asked)}; "
              f"estimated board time (read/11 + G/5.5, after the first request): mean {sum(est) / len(est):.0f} s, "
              f"max {max(est):.0f} s, over 57 s: {sum(e > 57 for e in est)}")
        for lang in sorted({r["language"] for r in asked}):
            a = [r for r in asked if r["language"] == lang]
            m = [r for r in a if len(r.get("matches") or []) == 1]
            ok = [r for r in m if r["matches"][0] == r["label"]]
            print(f"  {lang}: asked {len(a)}, unique match {len(m)}, right {len(ok)}")
        print(f"laptop time: {time.time() - t_start:.0f} s")
    for r in right:
        print(f"  RIGHT {r['id']} {r['language']} L{r['level']}: printed {r['printed']!r}")
    for r in matched:
        if r not in right:
            print(f"  wrong {r['id']} {r['language']} L{r['level']}: printed {r['printed']!r} -> {r['matches'][0]}, truth {r['label']}")


if __name__ == "__main__":
    main()
