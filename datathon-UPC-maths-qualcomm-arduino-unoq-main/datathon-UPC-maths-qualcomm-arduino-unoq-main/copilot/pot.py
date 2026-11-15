"""Program of thought: the model writes a short Python program, and Python computes the answer.

The program runs in a separate process with a closed list of names, no imports,
a memory limit and a time limit. Only the standard library is used.
"""
import re
import subprocess
import sys

LETTERS = "ABCDE"

# Runs the program read from stdin. Kept as a string so it starts as its own process with -I.
RUNNER = r'''
import ast, sys, math, itertools, fractions
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)
try:
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (768 << 20, 768 << 20))
except Exception:
    pass
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


def parse_num(val):
    """Read an option or a printed result as a number, or None. Also undoes common OCR slips."""
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


def numeric_options(problem):
    """{letter: number} when the problem has five options and all are numbers, else None."""
    options = problem.get("options") or {}
    nums = {k: parse_num(options.get(k, "")) for k in LETTERS}
    if len(options) == 5 and all(v is not None for v in nums.values()):
        return nums
    return None


def extract_code(text):
    """The program inside a ```python block, or the whole reply. Import lines are dropped."""
    m = re.search(r"```(?:python)?\s*\n(.*?)(?:```|$)", text, re.S)
    code = m.group(1) if m else text
    lines = [line for line in code.splitlines() if not re.match(r"\s*(import |from \S+ import )", line)]
    return "\n".join(lines).strip()


def run(code, timeout=4.0):
    """Return (last printed line or None, status). status is "ok", "timeout" or an error."""
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
    text = printed.strip()
    if text in ("True", "False"):
        return None
    v = parse_num(text)
    if v is not None:
        return v
    m = re.search(r"[-+]?\d+(?:\.\d+)?(?:\s*/\s*\d+)?", text)
    return parse_num(m.group(0)) if m else None


def match(nums, value):
    """Letters whose option equals the value."""
    if value is None:
        return []
    return [k for k, v in nums.items() if abs(value - v) < 1e-6]


def _match_one(nums, printed):
    letters = match(nums, to_value(printed))
    text = (printed or "").strip()
    if not letters and text.isdigit() and len(text) > 1 and all(v in range(10) for v in nums.values()):
        letters = match(nums, float(text[-1]))
    return letters


def match_printed(nums, printed):
    """Letters whose option equals what the program printed."""
    if printed and "\t" in printed:
        first_p, last_p = printed.split("\t", 1)
        letters = _match_one(nums, last_p)
        if not letters:
            letters = _match_one(nums, first_p)
        return letters
    return _match_one(nums, printed)


def run_and_match(code, problem, timeout=4.0):
    """Run a program for a problem. Return (letters that match, printed, status)."""
    nums = numeric_options(problem)
    printed, status = run(code, timeout=timeout)
    if printed and "\t" in printed:
        first_p, last_p = printed.split("\t", 1)
        if nums is None:
            return [], last_p, status
        letters = _match_one(nums, last_p)
        if len(letters) == 1:
            return letters, last_p, status
        first_letters = _match_one(nums, first_p)
        if len(first_letters) == 1:
            return first_letters, first_p, status
        return letters, last_p, status
    if nums is None:
        return [], printed, status
    return match_printed(nums, printed), printed, status


# ---------------------------------------------------------------------------
# Prompts. Two styles, chosen with POT_STYLE in config.env:
#   fewshot    the stock model, shown three solved examples as earlier turns
#   finetuned  the model trained in finetune/, which needs no examples
# The model never sees the options: it has to compute the answer, not pick one.

SYSTEM_FEWSHOT = (
    "You solve math problems by writing a short, flat Python script (no functions, no comments, no input). "
    "Always reply with a ```python code block that prints one number: the final answer."
)
INSTRUCTION = "Write a short Python program (no def, no comments) that prints the answer."
FEWSHOT = [
    ("How many whole numbers from 1 to 200 are divisible by 3 but not by 6?",
     "```python\nprint(sum(1 for n in range(1, 201) if n % 3 == 0 and n % 6 != 0))\n```"),
    ("En Pau pensa un nombre d'1 a 999. El multiplica per 4, hi resta 6 i obté 30. Quin nombre ha pensat?",
     "```python\nfor x in range(1, 1000):\n    if 4 * x - 6 == 30:\n        print(x)\n        break\n```"),
    ("Quina és la suma de les xifres del nombre més petit més gran que 10 que deixa residu 1 en dividir-lo "
     "per 2, 3 i 5?",
     "```python\nfor n in range(11, 1000):\n    if n % 2 == 1 and n % 3 == 1 and n % 5 == 1:\n"
     "        print(sum(int(d) for d in str(n)))\n        break\n```"),
]
SYSTEM_FINETUNED = "Write a short Python program that prints the answer to the math problem."


def build_messages(problem, style="fewshot"):
    text = problem["text"].strip()
    if style == "finetuned":
        return [{"role": "system", "content": SYSTEM_FINETUNED}, {"role": "user", "content": text}]
    messages = [{"role": "system", "content": SYSTEM_FEWSHOT}]
    for question, answer in FEWSHOT:
        messages += [{"role": "user", "content": question + "\n" + INSTRUCTION},
                     {"role": "assistant", "content": answer}]
    messages.append({"role": "user", "content": text + "\n" + INSTRUCTION})
    return messages
