"""Re-execute the programs stored by pot_probe.py with a more tolerant runner (no model needed):
`if __name__ == "__main__":` is accepted, classes and try/except are allowed, and when nothing was
printed the value of the last expression statement is used.
Usage: python -I pot_rerun.py <pot.jsonl> <data.jsonl> <out.jsonl>"""
import json
import re
import subprocess
import sys

LETTERS = "ABCDE"

RUNNER = r'''
import ast, sys, math, itertools, fractions
src = sys.stdin.read()
try:
    tree = ast.parse(src)
except SyntaxError:
    # a program cut at the token limit: drop trailing lines until it parses
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
safe = {k: getattr(builtins, k) for k in ("range", "len", "sum", "min", "max", "abs", "int", "float", "str",
        "set", "list", "tuple", "dict", "sorted", "reversed", "enumerate", "zip", "map", "filter", "any", "all",
        "pow", "divmod", "round", "print", "bool", "ord", "chr", "frozenset", "Exception", "ValueError",
        "ZeroDivisionError", "isinstance", "object", "__build_class__")}
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
    # keep the value of a final bare expression, e.g. a call to solve() that returns the answer
    tree.body[-1] = ast.Assign(targets=[ast.Name(id="_last_value", ctx=ast.Store())], value=last.value)
    ast.fix_missing_locations(tree)
exec(compile(tree, "<model>", "exec"), env)
if printed:
    real_print(printed[-1])
elif env.get("_last_value") is not None:
    real_print(env["_last_value"])
'''


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
    with open(sys.argv[1], encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    with open(sys.argv[2], encoding="utf-8") as f:
        data = {(r["id"], r["language"]): r for r in map(json.loads, (l for l in f if l.strip()))}
    before = after = before_ok = after_ok = 0
    with open(sys.argv[3], "w", encoding="utf-8") as out:
        for r in rows:
            if r["status"] != "skipped":
                old = r.get("matches") or []
                before += len(old) == 1
                before_ok += len(old) == 1 and old[0] == r["label"]
                options = data[(r["id"], r["language"])].get("options") or {}
                nums = {k: parse_num(options.get(k, "")) for k in LETTERS}
                printed, status = run(r["code"])
                value = to_value(printed)
                r.update({"status": status, "printed": printed, "value": value,
                          "matches": [k for k, v in nums.items() if value is not None and v is not None and abs(value - v) < 1e-6]})
                after += len(r["matches"]) == 1
                after_ok += len(r["matches"]) == 1 and r["matches"][0] == r["label"]
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
    asked = sum(1 for r in rows if r["status"] != "skipped")
    print(f"asked {asked}: unique matches {before} -> {after}; right {before_ok} -> {after_ok}")


if __name__ == "__main__":
    main()
