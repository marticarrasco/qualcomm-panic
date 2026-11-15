"""Ask the small language model to solve a problem via cached bilingual Program-of-Thought (PoT)
or direct 1-token letter selection."""
import re
import subprocess
import sys

from .config import CONFIG
from .client import post

URL = f"http://127.0.0.1:{CONFIG['LLM_PORT']}/v1/chat/completions"

# Legacy 1-token system prompt kept for backward compatibility with explorer/server.py
SYSTEM = ("You solve multiple-choice math competition problems for school students. "
          "Read the problem and the options, then reply with the letter of the correct option.")

# Cached bilingual Program-of-Thought (PoT) prompt.
# CRITICAL: Keep POT_SYSTEM + POT_SHOTS strictly under 512 tokens (currently 496 tokens)
# so llama-server's default ubatch=512 saves the Gated DeltaNet checkpoint at the end of
# the few-shot prefix and reuses cache_n=496 tokens on every problem.
POT_SYSTEM = (
    "You are a Python math solver. Reply ONLY with a ```python block of 2 to 6 flat lines "
    "that computes the answer and ends with print(ans). No def, no imports, no comments, no text outside the block."
)
POT_SUFFIX = "\nWrite 2 to 6 flat lines of Python ending with print(ans):"

POT_SHOTS = [
    (
        "How many integers from 10 to 99 have a digit sum equal to 9?" + POT_SUFFIX,
        "```python\nans = sum(1 for n in range(10, 100) if sum(int(d) for d in str(n)) == 9)\nprint(ans)\n```",
    ),
    (
        "En Marc té 9 caramels i en Tomàs en té 17. Quants caramels ha de donar en Tomàs a en Marc perquè tots dos en tinguin la mateixa quantitat?" + POT_SUFFIX,
        "```python\ntotal = 9 + 17\nans = total // 2 - 9\nprint(ans)\n```",
    ),
    (
        "Find the smallest positive integer n such that 2*n + 5 is divisible by 7 and n > 10." + POT_SUFFIX,
        "```python\nfor n in range(11, 500):\n    if (2 * n + 5) % 7 == 0:\n        print(n)\n        break\n```",
    ),
    (
        "Quants nombres enters de l'1 al 100 són múltiples de 3 o de 5, però no de tots dos alhora?" + POT_SUFFIX,
        "```python\nans = sum(1 for n in range(1, 101) if (n % 3 == 0) ^ (n % 5 == 0))\nprint(ans)\n```",
    ),
]

RUNNER = r'''
import ast, sys, math, itertools, fractions
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)
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
           "permutations", "combinations", "product", "prod", "gcd", "factorial", "sqrt", "comb", "isqrt", "lcm",
           "__name__", "Exception", "ValueError", "ZeroDivisionError", "isinstance", "self", "object",
           "combinations_with_replacement", "floor", "ceil", "pi"}
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
real_int = builtins.int
real_str = builtins.str
def smart_sum(iterable, start=0):
    items = [real_int(x) if isinstance(x, str) and x.strip().lstrip("-").isdigit() else x for x in iterable]
    return real_sum(items, start)
class _SmartStr(str):
    def __contains__(self, x):
        return super().__contains__(real_str(x) if isinstance(x, int) else x)
    def __getitem__(self, k):
        return _SmartStr(super().__getitem__(k))
    def __sub__(self, o):
        if isinstance(o, str) and self.lstrip("-").isdigit() and o.lstrip("-").isdigit():
            return real_int(self) - real_int(o)
        return NotImplemented
def smart_str(x=""):
    return _SmartStr(real_str(x))
class _PermTuple(tuple):
    def _num(self):
        if self and all(isinstance(x, int) and not isinstance(x, bool) and 0 <= x <= 9 for x in self):
            return real_int("".join(real_str(x) for x in self))
        if self and all(isinstance(x, str) and x.isdigit() for x in self):
            return real_int("".join(self))
        return None
    def __lt__(self, other):
        if isinstance(other, int):
            v = self._num()
            return v < other if v is not None else False
        return super().__lt__(other)
    def __le__(self, other):
        if isinstance(other, int):
            v = self._num()
            return v <= other if v is not None else False
        return super().__le__(other)
    def __gt__(self, other):
        if isinstance(other, int):
            v = self._num()
            return v > other if v is not None else False
        return super().__gt__(other)
    def __ge__(self, other):
        if isinstance(other, int):
            v = self._num()
            return v >= other if v is not None else False
        return super().__ge__(other)
    def __mod__(self, other):
        if isinstance(other, int):
            v = self._num()
            if v is not None:
                return v % other
        return NotImplemented
    def __eq__(self, other):
        if isinstance(other, int):
            v = self._num()
            return v == other if v is not None else False
        return super().__eq__(other)
    def __ne__(self, other):
        if isinstance(other, int):
            v = self._num()
            return v != other if v is not None else True
        return super().__ne__(other)
    def __hash__(self):
        return super().__hash__()
def smart_int(x=0, *args):
    if not args and isinstance(x, (tuple, list)):
        v = _PermTuple(x)._num()
        if v is not None:
            return v
    return real_int(x, *args)
def _wrap_iter(fn):
    def _wrapped(*a, **kw):
        for t in fn(*a, **kw):
            yield _PermTuple(t)
    return _wrapped
class _SmartIntProd(int):
    def __new__(cls, val, items):
        obj = super().__new__(cls, val)
        obj._items = items
        return obj
    def __iter__(self):
        for x in self._items:
            yield _PermTuple((x,))
def smart_product(*args, **kwargs):
    if len(args) == 1 and not kwargs:
        items = list(args[0])
        if all(isinstance(x, int) and not isinstance(x, bool) for x in items):
            return _SmartIntProd(math.prod(items), items)
        if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in items):
            return math.prod(items)
        return _wrap_iter(itertools.product)(items)
    return _wrap_iter(itertools.product)(*args, **kwargs)
safe = {k: getattr(builtins, k) for k in ("range", "len", "min", "max", "abs", "float",
        "set", "list", "tuple", "dict", "sorted", "reversed", "enumerate", "zip", "map", "filter", "any", "all",
        "pow", "divmod", "round", "print", "bool", "ord", "chr", "frozenset", "Exception", "ValueError",
        "ZeroDivisionError", "isinstance", "object", "__build_class__")}
safe["sum"] = smart_sum
safe["int"] = smart_int
safe["str"] = smart_str
env = {"__builtins__": safe, "__name__": "__main__", "math": math, "itertools": itertools,
       "Fraction": fractions.Fraction, "permutations": _wrap_iter(itertools.permutations),
       "combinations": _wrap_iter(itertools.combinations), "product": smart_product, "prod": math.prod,
       "combinations_with_replacement": _wrap_iter(itertools.combinations_with_replacement),
       "gcd": math.gcd, "factorial": math.factorial, "sqrt": math.sqrt, "comb": math.comb, "isqrt": math.isqrt,
       "lcm": math.lcm, "floor": math.floor, "ceil": math.ceil, "pi": math.pi}
printed = []
real_print = print
def capture(*args, **kwargs):
    printed.append(" ".join(real_str(a) for a in args))
safe["print"] = capture
last = tree.body[-1] if tree.body else None
if isinstance(last, ast.Expr) and not (isinstance(last.value, ast.Call) and getattr(last.value.func, "id", "") == "print"):
    tree.body[-1] = ast.Assign(targets=[ast.Name(id="_last_value", ctx=ast.Store())], value=last.value)
    ast.fix_missing_locations(tree)
exec(compile(tree, "<model>", "exec"), env)
if printed:
    if len(printed) > 1 and printed[0] != printed[-1]:
        real_print(printed[0] + "\t" + printed[-1])
    else:
        real_print(printed[-1])
elif env.get("_last_value") is not None:
    real_print(env["_last_value"])
'''


def format_problem(problem, sees_image=False):
    lines = [problem["text"].strip()]
    options = problem.get("options") or {}
    for letter in "ABCDE":
        if letter in options:
            lines.append(f"{letter}) {options[letter]}")
    if not options:
        lines.append("(The options are in the picture.)" if sees_image else
                     "(The options are in a picture that you can't see. Pick the most likely letter.)")
    return "\n".join(lines)


def build_messages(problem, examples=(), image_b64=None):
    """image_b64 is a PNG as base64 text. Only a server started with --mmproj can see it."""
    messages = [{"role": "system", "content": SYSTEM}]
    for ex in examples:  # solved problems retrieved with RAG, if any
        messages.append({"role": "user", "content": format_problem(ex) + "\nAnswer:"})
        messages.append({"role": "assistant", "content": ex["label"]})
    text = format_problem(problem, sees_image=bool(image_b64)) + "\nAnswer:"
    if image_b64:
        content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + image_b64}},
                   {"type": "text", "text": text}]
    else:
        content = text
    messages.append({"role": "user", "content": content})
    return messages


def answer_letter(problem, examples=(), timeout=60, image_b64=None, url=URL):
    """Return (letter, timings). The grammar makes any other output impossible."""
    response = post(url, {
        "messages": build_messages(problem, examples, image_b64),
        "grammar": "root ::= [A-E]",
        "max_tokens": 1,
        "temperature": 0,
        "seed": int(CONFIG["SEED"]),
        "cache_prompt": True,
    }, timeout=timeout)
    letter = response["choices"][0]["message"]["content"].strip()
    return letter, response.get("timings", {})


def parse_num(val):
    """Parse an option or printed string as a float, handling OCR quirks ('O'->'0', '一'->'-', spaces)."""
    import math
    s = str(val).strip().replace(",", ".").rstrip("%").rstrip("°").strip()
    s = s.replace("−", "-").replace("–", "-").replace("一", "-")
    if s in ("O", "o"):
        s = "0"
    if re.match(r"^\d{1,3}(\s+\d{3})+$", s):
        s = s.replace(" ", "")
    if re.match(r"^[-+]?\d+(\.\d+)?$", s):
        try:
            v = float(s)
            return v if math.isfinite(v) else None
        except OverflowError:
            return None
    m = re.match(r"^([-+]?\d+)\s*/\s*(\d+)$", s)
    if m and float(m.group(2)) != 0:
        try:
            v = float(m.group(1)) / float(m.group(2))
            return v if math.isfinite(v) else None
        except OverflowError:
            return None
    return None


def has_numeric_options(problem, min_numeric=5):
    options = problem.get("options") or {}
    if len(options) < 5:
        return False
    return sum(1 for v in options.values() if parse_num(v) is not None) >= min_numeric


def extract_code(reply):
    m = re.search(r"```(?:python)?\s*(.*?)```", reply, re.S)
    code = m.group(1) if m else reply
    lines = []
    for line in code.splitlines():
        if re.match(r"\s*(import |from \S+ import |#)", line):
            continue
        if line.strip().startswith("```"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def run_code(code, timeout=2.0):
    if not code:
        return None, "empty"
    try:
        r = subprocess.run(
            [sys.executable, "-I", "-X", "utf8", "-c", RUNNER],
            input=code,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None, "code_timeout"
    if r.returncode != 0:
        err = ((r.stderr or "").strip().splitlines() or ["code_error"])[-1]
        return None, err[:80]
    out = r.stdout.strip().splitlines()
    return (out[-1].strip() if out else None), "ok"


def _match_single_option(printed, options, problem_text=""):
    if printed is None or not options:
        return None
    s_print = printed.strip()
    if re.search(r"(last digit|units digit|última xifra|darrera xifra)", problem_text, re.I) and s_print.lstrip("-").isdigit() and len(s_print) >= 2:
        ld = float(s_print[-1])
        ld_hits = [k for k, v in options.items() if parse_num(v) == ld]
        if len(ld_hits) == 1:
            return ld_hits[0]
    p_num = parse_num(s_print)
    if p_num is None:
        m = re.search(r"[-+]?\d+(?:\.\d+)?(?:\s*/\s*\d+)?", s_print)
        if m:
            p_num = parse_num(m.group(0))
    if p_num is not None:
        exact = []
        close = []
        for k, v in options.items():
            o_num = parse_num(v)
            if o_num is not None:
                if abs(p_num - o_num) <= 1e-6:
                    exact.append(k)
                elif abs(p_num - o_num) <= 1e-2:
                    close.append(k)
        if len(exact) == 1:
            return exact[0]
        if len(exact) == 0 and len(close) == 1:
            return close[0]
    for k, v in options.items():
        if str(v).strip().lower() == s_print.lower():
            return k
    return None


def match_option(printed, options, problem_text=""):
    if printed is None or not options:
        return None
    if "\t" in printed:
        first_p, last_p = printed.split("\t", 1)
        hit = _match_single_option(last_p, options, problem_text)
        if hit is not None:
            return hit
        return _match_single_option(first_p, options, problem_text)
    return _match_single_option(printed, options, problem_text)


_POT_STYLE = {}


def pot_style(url=URL):
    """"finetuned" or "fewshot". POT_STYLE=auto asks the server which model it has loaded,
    so the prompt always matches the model that is really running on the board."""
    style = (CONFIG.get("POT_STYLE") or "auto").strip().lower()
    if style in ("finetuned", "fewshot"):
        return style
    if url not in _POT_STYLE:
        name = CONFIG.get("LLM_FILE", "")
        try:
            import json
            import urllib.request
            with urllib.request.urlopen(url.split("/v1/")[0] + "/props", timeout=5) as response:
                name = json.load(response).get("model_path") or name
        except Exception:
            pass  # server too old or not reachable: trust config.env
        _POT_STYLE[url] = "finetuned" if "m3kang-pot" in name.lower() else "fewshot"
    return _POT_STYLE[url]


def build_pot_messages(problem, url=URL):
    if pot_style(url) == "finetuned":
        from .pot import build_messages as build_ft_messages
        return build_ft_messages(problem, style="finetuned")
    messages = [{"role": "system", "content": POT_SYSTEM}]
    for q, a in POT_SHOTS:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": problem["text"].strip() + POT_SUFFIX})
    return messages


def solve_pot(problem, timeout=60, url=URL):
    """Generate a short flat Python script, execute it safely, and match its output to A-E.
    If the first greedy script fails to uniquely match any option and sufficient wall-clock
    budget remains (> 22 s), retry using the cached prompt with diversity sampling.
    Returns (matched_letter_or_None, timings, meta_dict).
    """
    import time
    t0 = time.perf_counter()
    messages = build_pot_messages(problem, url)
    base_seed = int(CONFIG["SEED"])
    options = problem.get("options") or {}
    prob_text = problem.get("text", "")

    matched = None
    total_prompt_n = 0
    first_printed = None
    pending_one_letter = None
    last_meta = {"code": "", "printed": None, "run_status": "none", "tries": 0}

    for attempt, (temp, seed) in enumerate(((0, base_seed), (0.45, base_seed + 1), (0.45, base_seed + 2)), 1):
        remaining = timeout - (time.perf_counter() - t0)
        if attempt > 1 and remaining < 22.0:
            break
        payload = {
            "messages": messages,
            "max_tokens": 140,
            "temperature": temp,
            "seed": seed,
            "cache_prompt": True,
        }
        if temp > 0:
            payload["top_p"] = 0.9
        # Leave room for running the program (up to 2 s) inside the caller's budget.
        response = post(url, payload, timeout=max(remaining - 2.5, 1.0))
        reply = response["choices"][0]["message"]["content"]
        timings = response.get("timings", {})
        if attempt == 1:
            total_prompt_n = timings.get("prompt_n", 0)
        code = extract_code(reply)
        printed, run_status = run_code(code, timeout=2.0)
        cand = match_option(printed, options, prob_text)
        if attempt == 1:
            first_printed = (printed or "").split("\t")[-1].strip() if printed is not None else None
        last_meta = {"code": code, "printed": printed, "run_status": run_status, "tries": attempt}
        if cand in ("A", "B", "C", "D", "E"):
            cur_p = (printed or "").split("\t")[-1].strip()
            if attempt > 1 and cur_p == "1" and first_printed in (None, "", "0"):
                if pending_one_letter == cand:
                    matched = cand
                    break
                pending_one_letter = cand
                continue
            matched = cand
            break

    return matched, {"prompt_n": total_prompt_n}, last_meta
