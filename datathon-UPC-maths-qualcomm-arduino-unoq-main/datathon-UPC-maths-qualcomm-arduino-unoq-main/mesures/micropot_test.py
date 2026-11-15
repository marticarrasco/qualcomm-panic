"""Run the investigation's Micro-PoT (section 5.1, try_micro_pot) exactly as written, on a data file.
Usage: python -I micropot_test.py --input demo.jsonl [--url http://127.0.0.1:8091]"""
import argparse
import ast
import json
import operator
import re
import urllib.request

LETTERS = "ABCDE"
SYSTEM = (  # the investigation's SYSTEM (it adds "carefully")
    "You solve multiple-choice math competition problems for school students. "
    "Read the problem and the options carefully, then reply with the letter of the correct option."
)
_VISUAL_WORDS = re.compile(
    r"\b(figure|diagram|picture|shown|shaded|drawn|arrow|grid|table below|graph|"
    r"figura|diagrama|dibuix|ombrejat|mostra|fletxa|taula)\b",
    re.IGNORECASE,
)


def _parse_num(val):
    s = val.strip().replace(",", ".").rstrip("%").strip()
    s = s.replace("−", "-").replace("–", "-")
    if re.match(r"^[-+]?\d+(\.\d+)?$", s):
        return float(s)
    m = re.match(r"^([-+]?\d+)\s*/\s*(\d+)$", s)
    if m and float(m.group(2)) != 0:
        return float(m.group(1)) / float(m.group(2))
    return None


def classify_options(options):
    if not options or len(options) < 5:
        return "non_num", {}
    nums = {k: _parse_num(options.get(k, "")) for k in LETTERS}
    if all(v is not None for v in nums.values()):
        seq = [nums[k] for k in LETTERS]
        if len(set(seq)) == 5:
            if seq == sorted(seq):
                return "inc_num", nums
            if seq == sorted(seq, reverse=True):
                return "dec_num", nums
            return "unord_num", nums
    return "non_num", {}


_SAFE_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
    ast.Pow: lambda a, b: a ** b if abs(b) <= 12 else 0,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _safe_eval_expr(expr_str):
    def _eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
            return _SAFE_OPS[type(node.op)](_eval(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
            return _SAFE_OPS[type(node.op)](_eval(node.left), _eval(node.right))
        raise ValueError("Unsupported AST node")
    tree = ast.parse(expr_str.strip(), mode="eval")
    return _eval(tree.body)


def post(url, payload, timeout):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8091")
    p.add_argument("--input", required=True)
    args = p.parse_args()
    with open(args.input, encoding="utf-8") as f:
        problems = [json.loads(line) for line in f if line.strip()]

    stats = {"rows": len(problems), "numeric": 0, "eligible": 0, "evaluated": 0, "bare_number": 0,
             "unique_match": 0, "correct": 0, "gen_tokens": 0, "prompt_tokens": 0}
    shown = 0
    for problem in problems:
        kind, nums = classify_options(problem.get("options") or {})
        if not nums:
            continue
        stats["numeric"] += 1
        text = problem.get("text", "")
        if len(text) > 380 or _VISUAL_WORDS.search(text) or not problem.get("text_only", True):
            continue
        stats["eligible"] += 1
        prompt = (f"{text.strip()}\n"
                  "Write ONLY the arithmetic expression (using numbers and +, -, *, /, parentheses) "
                  "that computes the answer:\nExpression: ")
        resp = post(args.url + "/v1/chat/completions", {
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            "grammar": 'root ::= [0-9+\\-*/(). ]+ "\\n"', "max_tokens": 16, "temperature": 0, "seed": 42,
            "cache_prompt": True}, 120)
        expr = resp["choices"][0]["message"]["content"].strip()
        t = resp.get("timings", {})
        stats["gen_tokens"] += t.get("predicted_n", 0)
        stats["prompt_tokens"] += t.get("prompt_n", 0)
        value, verdict = None, "no value"
        try:
            value = _safe_eval_expr(expr)
            stats["evaluated"] += 1
            bare = re.fullmatch(r"[\d. ]+", expr) is not None
            stats["bare_number"] += bare
            matches = [k for k, v in nums.items() if abs(value - v) < 1e-6]
            if len(matches) == 1:
                stats["unique_match"] += 1
                ok = matches[0] == problem["label"]
                stats["correct"] += ok
                verdict = f"match {matches[0]} ({'RIGHT' if ok else 'wrong, truth ' + problem['label']})" + (" [bare number]" if bare else "")
            else:
                verdict = "no option matches" + (" [bare number]" if bare else "")
        except Exception as e:
            verdict = f"not evaluable ({type(e).__name__})"
        if shown < 24:
            shown += 1
            print(f"  {problem['id']:15} {problem['language']}  expr={expr!r:28} value={value}  -> {verdict}")
    print("\n", json.dumps(stats))
    e = max(stats["eligible"], 1)
    print(f"eligible {stats['eligible']}/{stats['rows']} rows; evaluable {stats['evaluated']}; "
          f"bare numbers {stats['bare_number']}; unique option match {stats['unique_match']}; "
          f"right {stats['correct']} -> precision {100 * stats['correct'] / max(stats['unique_match'], 1):.0f}% of matches, "
          f"{100 * stats['correct'] / e:.1f}% of eligible rows; mean generated tokens {stats['gen_tokens'] / e:.1f}, "
          f"mean prompt tokens read {stats['prompt_tokens'] / e:.0f}")


if __name__ == "__main__":
    main()
