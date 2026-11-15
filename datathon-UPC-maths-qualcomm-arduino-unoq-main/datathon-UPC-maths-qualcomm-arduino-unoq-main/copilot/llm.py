"""Ask the small language model to solve a problem via Program-of-Thought (PoT)
or direct/reasoned letter selection."""
import re
import subprocess
import sys

from .config import CONFIG
from .client import post
from .pot import RUNNER

URL = f"http://127.0.0.1:{CONFIG['LLM_PORT']}/v1/chat/completions"

# Legacy reasoning system prompt kept for backward compatibility with ANSWER_MODE=direct/text
SYSTEM = ("You solve multiple-choice math competition problems for school students. "
          "Work out the problem before choosing an option. Reply in at most two short lines: "
          "first give the key calculation or reason, then finish with exactly 'Answer: X', "
          "where X is the letter of the correct option (A, B, C, D, or E). "
          "Do not choose an option just because of its letter.")

ANSWER_RE = re.compile(r"(?:answer|resposta)\s*[:：]\s*([A-E])\b", re.IGNORECASE)

# Cached bilingual Program-of-Thought (PoT) prompt for stock models.
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


class InvalidAnswerError(ValueError):
    """An unparseable model answer, with any token usage the server reported."""

    def __init__(self, message, timings=None, output=None):
        super().__init__(message)
        self.timings = timings or {}
        self.output = output


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
        messages.append({"role": "user", "content": format_problem(ex)})
        messages.append({"role": "assistant", "content": f"Answer: {ex['label']}"})
    text = format_problem(problem, sees_image=bool(image_b64)) + "\nAnswer:"
    if image_b64:
        content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + image_b64}},
                   {"type": "text", "text": text}]
    else:
        content = text
    messages.append({"role": "user", "content": content})
    return messages


def answer_letter(problem, examples=(), timeout=60, image_b64=None, url=URL):
    """Return the final answer letter and timings, or raise ValueError if it is missing."""
    response = post(url, {
        "messages": build_messages(problem, examples, image_b64),
        "max_tokens": 128,
        "temperature": 0,
        "seed": int(CONFIG["SEED"]),
        "cache_prompt": True,
    }, timeout=timeout)
    raw_content = response["choices"][0]["message"]["content"]
    content = raw_content.strip()
    matches = ANSWER_RE.findall(content)
    if matches:
        letter = matches[-1].upper()
    elif re.fullmatch(r"[A-E][.!]?", content, re.IGNORECASE):
        letter = content[0].upper()
    else:
        raise InvalidAnswerError("model did not return an answer in the expected format",
                                 response.get("timings", {}), raw_content)
    return letter, response.get("timings", {}), raw_content


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
    total_predicted_n = 0
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
        total_predicted_n += timings.get("predicted_n", 0)
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

    return matched, {"prompt_n": total_prompt_n, "predicted_n": total_predicted_n}, last_meta
