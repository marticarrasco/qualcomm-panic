"""Generate and verify synthetic reasoning procedures, flat Python PoT scripts, Socratic hints, and topics
for M3Kang text_only English problems (with automatic projection to Catalan).

Reads HELMCODE_API_KEY from environment variable (never hardcoded).
Usage:
    python -I build_synthetic_dataset.py --limit 20 --workers 8
    python -I build_synthetic_dataset.py --limit 0 --workers 10
"""
import argparse
import ast
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter

API_URL = "https://api.helmcode.com/v1/chat/completions"
MODEL = "deepseek-v4-flash"
LETTERS = "ABCDE"
VALID_TOPICS = {"combinatorics", "number_theory", "arithmetic", "algebra", "geometry", "logic"}

RUNNER = r'''
import ast, sys, math, itertools, fractions
src = sys.stdin.read()
tree = ast.parse(src)
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
exec(compile(tree, "<model>", "exec"), env)
if printed:
    real_print(printed[-1])
'''

SYSTEM_PROMPT = """You are an expert mathematical competition solver creating a clean, verified training dataset for a compact Edge AI model.
Given a Kangaroo Math problem (in English and Catalan), its 5 options (A-E), and the ground-truth correct option, return ONLY a valid JSON object with these exact keys:
{
  "topic": one of ["combinatorics", "number_theory", "arithmetic", "algebra", "geometry", "logic"],
  "steps_en": "2 to 3 concise sentences in English explaining the mathematical procedure to solve it, ending with -> Answer: <letter> (<value>)",
  "steps_ca": "2 to 3 concise sentences in Catalan explaining the mathematical procedure to solve it, ending with -> Resposta: <letter> (<value>)",
  "hint_en": "One short Socratic hint in English guiding the first step WITHOUT revealing the final answer value or letter",
  "hint_ca": "One short Socratic hint in Catalan guiding the first step WITHOUT revealing the final answer value or letter",
  "code": "2 to 6 lines of flat top-level Python code (NO def, NO comments #, NO imports, NO input()). Pre-imported globals available: range, sum, min, max, abs, gcd, lcm, factorial, comb, isqrt, permutations, combinations, product, Fraction. The code MUST compute the answer genuinely from the problem's numbers (never hardcode the answer literal!) and end with print(ans)."
}
Return raw JSON only, with no markdown fences around the JSON."""


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


def is_nontrivial_code(code_str, target_str):
    """Ensure the code computes the value rather than simply doing `print(target)`."""
    try:
        tree = ast.parse(code_str)
    except SyntaxError:
        return False
    compute_ops = (ast.BinOp, ast.Compare, ast.For, ast.While, ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.If)
    has_compute = any(isinstance(n, compute_ops) for n in ast.walk(tree))
    if not has_compute:
        # Check if there is a non-print function call like sum/len/gcd/max/min
        calls = [n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        if not any(c != "print" for c in calls):
            return False
    return True


def run_code(code_str, timeout=4):
    lines = [l for l in code_str.splitlines() if not re.match(r"\s*(import |from \S+ import |#)", l)]
    cleaned = "\n".join(lines).strip()
    try:
        r = subprocess.run([sys.executable, "-I", "-X", "utf8", "-c", RUNNER], input=cleaned,
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, "timeout", cleaned
    if r.returncode != 0:
        err = ((r.stderr or "").strip().splitlines() or ["error"])[-1][:120]
        return None, err, cleaned
    out = r.stdout.strip().splitlines()
    return (out[-1].strip() if out else None), "ok", cleaned


def verify_code(code_str, options, label):
    printed, status, cleaned = run_code(code_str)
    if status != "ok" or printed is None:
        return False, printed, status, cleaned
    target_str = str(options[label]).strip()
    if not is_nontrivial_code(cleaned, target_str):
        return False, printed, "trivial_hardcode", cleaned

    target_num = parse_num(target_str)
    printed_num = parse_num(printed)
    if target_num is not None and printed_num is not None:
        if abs(target_num - printed_num) < 1e-5:
            return True, printed, "ok_num", cleaned
        return False, printed, f"mismatch: got {printed}, expected {target_str}", cleaned

    if printed.strip().lower() == target_str.lower():
        return True, printed, "ok_str", cleaned
    return False, printed, f"mismatch: got {printed}, expected {target_str}", cleaned


def call_helmcode(api_key, messages, max_tokens=3000, timeout=60):
    req = urllib.request.Request(
        API_URL,
        data=json.dumps({
            "model": MODEL,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": max_tokens,
            "reasoning_effort": "low",
        }).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "openai-python/1.50.0",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        msg = data["choices"][0]["message"]
        return (msg.get("content") or "").strip()


def extract_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    raw = m.group(0) if m else text
    return json.loads(raw)


def solve_one(api_key, eng_row, cat_row):
    pid = eng_row["id"]
    opts = eng_row["options"]
    label = eng_row["label"]
    ans_val = opts[label]
    opts_str = "\n".join(f"{k}) {opts[k]}" for k in LETTERS)
    cat_text = cat_row["text"] if cat_row else ""

    user_prompt = (
        f"Problem ID: {pid} (Level {eng_row['level']})\n"
        f"English statement:\n{eng_row['text']}\n\n"
        f"Catalan statement:\n{cat_text}\n\n"
        f"Options:\n{opts_str}\n"
        f"Correct Option: {label}) {ans_val}\n\n"
        f"Produce the JSON with topic, steps_en, steps_ca, hint_en, hint_ca, and flat Python code that prints {ans_val!r}."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(1, 4):
        try:
            raw = call_helmcode(api_key, messages, max_tokens=3000 + (attempt - 1) * 1500)
            parsed = extract_json(raw)
            code = parsed.get("code", "")
            ok, printed, reason, cleaned_code = verify_code(code, opts, label)
            if ok:
                topic = parsed.get("topic", "arithmetic").lower().strip()
                if topic not in VALID_TOPICS:
                    topic = "arithmetic"
                return {
                    "id": pid,
                    "level": eng_row["level"],
                    "year": eng_row["year"],
                    "label": label,
                    "answer_value": ans_val,
                    "options": opts,
                    "text_en": eng_row["text"],
                    "text_ca": cat_text,
                    "topic": topic,
                    "steps_en": parsed.get("steps_en", "").strip(),
                    "steps_ca": parsed.get("steps_ca", "").strip(),
                    "hint_en": parsed.get("hint_en", "").strip(),
                    "hint_ca": parsed.get("hint_ca", "").strip(),
                    "code": cleaned_code,
                    "printed": printed,
                    "verified": True,
                    "verify_status": reason,
                    "attempts": attempt,
                }
            # Prepare retry feedback
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": (
                    f"Your Python code failed verification ({reason}). "
                    f"Remember: no imports, no def, no comments, do not hardcode {ans_val!r}, "
                    f"and make sure print() outputs {ans_val!r}. Return ONLY the corrected JSON."
                ),
            })
        except Exception as e:
            if attempt == 3:
                return {"id": pid, "verified": False, "error": str(e)[:120]}
            time.sleep(1.0)

    return {
        "id": pid,
        "level": eng_row["level"],
        "label": label,
        "answer_value": ans_val,
        "text_en": eng_row["text"],
        "text_ca": cat_text,
        "topic": parsed.get("topic", "arithmetic") if "parsed" in locals() else "arithmetic",
        "steps_en": parsed.get("steps_en", "") if "parsed" in locals() else "",
        "steps_ca": parsed.get("steps_ca", "") if "parsed" in locals() else "",
        "hint_en": parsed.get("hint_en", "") if "parsed" in locals() else "",
        "hint_ca": parsed.get("hint_ca", "") if "parsed" in locals() else "",
        "code": cleaned_code if "cleaned_code" in locals() else "",
        "printed": printed if "printed" in locals() else None,
        "verified": False,
        "verify_status": reason if "reason" in locals() else "failed",
        "attempts": 2,
    }


def main():
    default_data = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
    p = argparse.ArgumentParser()
    p.add_argument("--train", default=os.path.join(default_data, "train.jsonl"))
    p.add_argument("--out", default=os.path.join(default_data, "synthetic_procedures.jsonl"))
    p.add_argument("--limit", type=int, default=20, help="0 means all 684 problems")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    api_key = os.environ.get("HELMCODE_API_KEY", "").strip()
    if not api_key:
        sys.exit("ERROR: Set HELMCODE_API_KEY environment variable first.")

    rows = [json.loads(line) for line in open(args.train, encoding="utf-8") if line.strip()]
    eng_rows = [r for r in rows if r["language"] == "eng" and r["text_only"] and len(r.get("options") or {}) == 5]
    cat_by_id = {r["id"]: r for r in rows if r["language"] == "cat"}

    done_ids = set()
    kept_lines = []
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    if d.get("verified") and d["id"] not in done_ids:
                        done_ids.add(d["id"])
                        kept_lines.append(json.dumps(d, ensure_ascii=False) + "\n")
        with open(args.out, "w", encoding="utf-8") as f:
            f.writelines(kept_lines)

    todo = [r for r in eng_rows if r["id"] not in done_ids]
    if args.limit > 0:
        todo = todo[:args.limit]

    print(f"Total eligible (eng, text_only, 5 options): {len(eng_rows)} | Already verified: {len(done_ids)} | To process now: {len(todo)}")
    if not todo:
        return

    lock = threading.Lock()
    t0 = time.time()
    stats = Counter()

    with open(args.out, "a", encoding="utf-8") as out_f:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            future_to_pid = {
                ex.submit(solve_one, api_key, r, cat_by_id.get(r["id"])): r["id"]
                for r in todo
            }
            for idx, fut in enumerate(concurrent.futures.as_completed(future_to_pid), 1):
                res = fut.result()
                with lock:
                    out_f.write(json.dumps(res, ensure_ascii=False) + "\n")
                    out_f.flush()
                    if res.get("verified"):
                        stats["verified"] += 1
                    else:
                        stats["unverified"] += 1
                    elapsed = time.time() - t0
                    print(f"[{idx}/{len(todo)}] {res['id']} -> verified={res.get('verified')} ({res.get('verify_status', res.get('error'))}) "
                          f"| ok={stats['verified']}/{idx} ({100*stats['verified']/idx:.1f}%) | {elapsed:.1f}s", flush=True)

    print(f"\nDone in {time.time() - t0:.1f}s! Summary: {dict(stats)}")


if __name__ == "__main__":
    main()
