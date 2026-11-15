"""Test improved procedural prompting (few-shot with instruction + ```python fences + flat code + smart runner).
Usage: python -I pot_probe_v2.py --input ../datathon-UPC-maths-qualcomm-arduino-unoq/data/demo/train.jsonl --out pot-demo-v2.jsonl"""
import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request

LETTERS = "ABCDE"
SYSTEM = (
    "You solve math problems by writing a short, flat Python script (no functions, no comments, no input). "
    "Always reply with a ```python code block that prints one number: the final answer."
)

INSTRUCTION_ENG = "Write a short Python program (no def, no comments) that prints the answer."
INSTRUCTION_CAT = "Write a short Python program (no def, no comments) that prints the answer."

FEWSHOT = [
    (
        "How many integers from 1 to 200 are multiples of 3 but not of 6?\n" + INSTRUCTION_ENG,
        "```python\nans = sum(1 for n in range(1, 201) if n % 3 == 0 and n % 6 != 0)\nprint(ans)\n```",
    ),
    (
        "En Pau pensa un nombre d'1 a 999. El multiplica per 4, hi resta 6 i obté 30. Quin és?\n" + INSTRUCTION_CAT,
        "```python\nfor x in range(1, 1000):\n    if 4 * x - 6 == 30:\n        print(x)\n        break\n```",
    ),
    (
        "Using digits 2, 5, 8 once each, how many 3-digit numbers > 500 can be formed?\n" + INSTRUCTION_ENG,
        "```python\ncount = 0\nfor a, b, c in permutations([2, 5, 8]):\n    if a * 100 + b * 10 + c > 500:\n        count += 1\nprint(count)\n```",
    ),
    (
        "Quin és el nombre més petit > 10 que deixa residu 1 en dividir-lo per 2, 3 i 5?\n" + INSTRUCTION_CAT,
        "```python\nfor n in range(11, 1000):\n    if n % 2 == 1 and n % 3 == 1 and n % 5 == 1:\n        print(n)\n        break\n```",
    ),
]

RUNNER = r'''
import ast, sys, math, itertools, fractions
src = sys.stdin.read()
try:
    tree = ast.parse(src)
except SyntaxError:
    lines = src.splitlines()
    tree = None
    while lines:
        lines.pop()
        try:
            tree = ast.parse("\n".join(lines))
            break
        except SyntaxError:
            continue
    if tree is None or not lines:
        sys.exit("SyntaxError")
BAD_NODES = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.With, ast.AsyncWith,
             ast.AsyncFunctionDef, ast.Await, ast.Delete)
ALLOWED = {"range", "len", "sum", "min", "max", "abs", "int", "float", "str", "set", "list", "tuple", "dict",
           "sorted", "reversed", "enumerate", "zip", "map", "filter", "any", "all", "pow", "divmod", "round",
           "print", "bool", "math", "itertools", "Fraction", "True", "False", "None", "ord", "chr", "frozenset",
           "permutations", "combinations", "product", "gcd", "factorial", "sqrt", "comb", "isqrt", "lcm",
           "__name__", "Exception", "ValueError", "ZeroDivisionError", "isinstance", "self", "object",
           "combinations_with_replacement", "floor", "ceil"}
defined = set()
for node in ast.walk(tree):
    if isinstance(node, BAD_NODES):
        sys.exit("blocked: " + type(node).__name__)
    if isinstance(node, ast.Attribute) and node.attr.startswith("_") and node.attr != "__init__":
        sys.exit("blocked: dunder attribute")
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        defined.add(node.name)
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
        defined.add(node.id)
    if isinstance(node, ast.arg):
        defined.add(node.arg)
    if isinstance(node, ast.ExceptHandler) and node.name:
        defined.add(node.name)
for node in ast.walk(tree):
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id not in ALLOWED and node.id not in defined:
        sys.exit("blocked: name " + node.id)
import builtins
real_sum = builtins.sum
def smart_sum(iterable, start=0):
    items = [int(x) if isinstance(x, str) and x.strip().lstrip("-").isdigit() else x for x in iterable]
    return real_sum(items, start)
safe = {k: getattr(builtins, k) for k in ("range", "len", "min", "max", "abs", "int", "float", "str",
        "set", "list", "tuple", "dict", "sorted", "reversed", "enumerate", "zip", "map", "filter", "any", "all",
        "pow", "divmod", "round", "print", "bool", "ord", "chr", "frozenset", "Exception", "ValueError",
        "ZeroDivisionError", "isinstance", "object", "__build_class__")}
safe["sum"] = smart_sum
env = {"__builtins__": safe, "__name__": "__main__", "math": math, "itertools": itertools,
       "Fraction": fractions.Fraction, "permutations": itertools.permutations,
       "combinations": itertools.combinations, "product": itertools.product,
       "combinations_with_replacement": itertools.combinations_with_replacement,
       "gcd": math.gcd, "factorial": math.factorial, "sqrt": math.sqrt, "comb": math.comb, "isqrt": math.isqrt,
       "lcm": math.lcm, "floor": math.floor, "ceil": math.ceil}
printed = []
real_print = print
def capture(*args, **kwargs):
    printed.append(" ".join(str(a) for a in args))
safe["print"] = capture
last = tree.body[-1] if tree.body else None
if isinstance(last, ast.Expr) and not (isinstance(last.value, ast.Call) and getattr(last.value.func, "id", "") == "print"):
    tree.body[-1] = ast.Assign(targets=[ast.Name(id="_last_value", ctx=ast.Store())], value=last.value)
    ast.fix_missing_locations(tree)
exec(compile(tree, "<model>", "exec"), env)
if printed:
    real_print(printed[-1])
elif env.get("_last_value") is not None:
    real_print(env["_last_value"])
'''


def post(url, payload, timeout=180):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def parse_num(val):
    s = str(val).strip().replace(",", ".").rstrip("%").strip()
    s = s.replace("−", "-").replace("–", "-").replace("一", "-")
    if s in ("O", "o"):
        s = "0"
    if re.match(r"^[-+]?\d+(\.\d+)?$", s):
        try:
            return float(s)
        except OverflowError:
            return None
    m = re.match(r"^([-+]?\d+)\s*/\s*(\d+)$", s)
    if m and float(m.group(2)) != 0:
        return float(m.group(1)) / float(m.group(2))
    return None


def extract_code(text):
    m = re.search(r"```(?:python)?\s*\n(.*?)(?:```|$)", text, re.S)
    code = m.group(1) if m else text
    lines = [l for l in code.splitlines() if not re.match(r"\s*(import |from \S+ import )", l)]
    return "\n".join(lines).strip()


def run(code, timeout=4):
    try:
        r = subprocess.run([sys.executable, "-I", "-X", "utf8", "-c", RUNNER], input=code, capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    if r.returncode != 0:
        return None, ((r.stderr or "").strip().splitlines() or ["error"])[-1][:80]
    out = r.stdout.strip().splitlines()
    return (out[-1].strip() if out else None), "ok"


def to_value(printed, question_text=""):
    if printed is None:
        return None
    # If question asks for last digit and printed is a huge integer, take its last digit
    if re.search(r"(last digit|última xifra|darrera xifra)", question_text, re.I) and printed.strip().isdigit() and len(printed.strip()) > 3:
        return float(printed.strip()[-1])
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
    p.add_argument("--max-tokens", type=int, default=140)
    args = p.parse_args()

    with open(args.input, encoding="utf-8") as f:
        problems = [json.loads(line) for line in f if line.strip()]

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

            instr = INSTRUCTION_CAT if problem["language"] == "cat" else INSTRUCTION_ENG
            user = problem["text"].strip() + "\n" + instr
            messages = [{"role": "system", "content": SYSTEM}]
            for q, a in FEWSHOT:
                messages += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
            messages.append({"role": "user", "content": user})

            start = time.perf_counter()
            resp = post(args.url + "/v1/chat/completions", {
                "messages": messages,
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "seed": 42,
                "cache_prompt": True,
            })
            text = resp["choices"][0]["message"]["content"]
            t = resp.get("timings", {})
            code = extract_code(text)
            printed, status = run(code)
            value = to_value(printed, problem["text"])
            matches = [k for k, v in nums.items() if value is not None and abs(val - v if (val := value) is not None else 1) < 1e-6]
            row.update({
                "status": status, "printed": printed, "value": value, "matches": matches,
                "finish": resp["choices"][0].get("finish_reason"),
                "prompt_n": (t.get("prompt_n") or 0) + (t.get("cache_n") or 0),
                "read_n": t.get("prompt_n") or 0, "cache_n": t.get("cache_n") or 0,
                "predicted_n": t.get("predicted_n"), "laptop_s": round(time.perf_counter() - start, 2),
                "code": code,
            })
            rows.append(row)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    asked = [r for r in rows if r["status"] != "skipped"]
    ran = [r for r in asked if r["status"] == "ok" and r["value"] is not None]
    matched = [r for r in asked if len(r.get("matches") or []) == 1]
    right = [r for r in matched if r["matches"][0] == r["label"]]
    print(f"asked {len(asked)} of {len(rows)} rows")
    print(f"program ran & printed number: {len(ran)}; unique match: {len(matched)}; right: {len(right)}")
    print("finish reasons:", {f: sum(1 for r in asked if r.get("finish") == f) for f in {r.get("finish") for r in asked}})
    if asked:
        est = [r["read_n"] / 11 + (r["predicted_n"] or 0) / 5.5 for r in asked[1:]] or [0]
        print(f"tokens: prompt mean {sum(r['prompt_n'] for r in asked)/len(asked):.0f} "
              f"(cached mean {sum(r['cache_n'] for r in asked[1:])/max(len(asked)-1,1):.0f}, "
              f"read mean {sum(r['read_n'] for r in asked[1:])/max(len(asked)-1,1):.0f}), "
              f"gen mean {sum(r['predicted_n'] or 0 for r in asked)/len(asked):.0f}; "
              f"est board time: mean {sum(est)/len(est):.1f}s, max {max(est):.1f}s, >57s: {sum(e > 57 for e in est)}")
        for lang in ("eng", "cat"):
            a = [r for r in asked if r["language"] == lang]
            m = [r for r in a if len(r.get("matches") or []) == 1]
            ok = [r for r in m if r["matches"][0] == r["label"]]
            print(f"  {lang}: asked {len(a)}, unique match {len(m)}, right {len(ok)}")
    print(f"elapsed laptop time: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
