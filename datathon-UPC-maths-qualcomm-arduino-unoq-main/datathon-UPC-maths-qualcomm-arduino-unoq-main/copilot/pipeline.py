"""The copilot pipeline: one problem in, one letter out.

predict.py and eval.py both call solve(), so whatever runs here is what the
official evaluation on the Arduino UNO Q board measures.

Strategy:
1. If image_b64 is explicitly passed (Explorer vision mode), use answer_letter().
2. If the problem requires a figure (text_only == False) or has non-numeric options,
   immediately return FALLBACK ('D') in 0.00 s (avoiding ungrounded guesses and timeouts).
3. Otherwise, run cached bilingual 4-shot Program-of-Thought (solve_pot):
   - 496-token few-shot prefix stays strictly under llama-server's ubatch=512 limit,
     reusing the Gated DeltaNet checkpoint (cache_n=496) on every problem.
   - Executes the generated Python snippet in an isolated AST-checked subprocess.
   - If the printed value uniquely matches one of options A-E, return that letter;
     otherwise return FALLBACK ('D').
"""
import socket
import time
import urllib.error

from .config import CONFIG
from .llm import URL, answer_letter, has_numeric_options, solve_pot

TIME_LIMIT = float(CONFIG["TIME_LIMIT_SECONDS"])
FALLBACK = CONFIG["FALLBACK_ANSWER"]
MARGIN = 3.0  # seconds kept free for Python and for writing the answer


def solve(problem, index=None, k=0, exclude_id=None, image_b64=None, llm_url=URL,
          time_limit=TIME_LIMIT):
    """Return {"answer", "seconds", "status", ...} for one problem."""
    start = time.perf_counter()
    result = {"answer": FALLBACK, "status": "ok", "prompt_tokens": 0, "retrieved": "", "examples": []}
    try:
        examples = []
        if index is not None and k > 0:
            hits = index.search(problem, k=k, exclude_id=exclude_id)
            examples = [p for _score, p in hits]
            result["examples"] = hits
            result["retrieved"] = ";".join(f"{p['id']}:{s:.2f}" for s, p in hits)

        remaining = time_limit - (time.perf_counter() - start) - MARGIN
        if remaining <= 1.0:
            result["status"] = "timeout"
        elif image_b64 is not None:
            letter, timings = answer_letter(problem, examples, timeout=max(remaining, 1.0),
                                            image_b64=image_b64, url=llm_url)
            if letter in ("A", "B", "C", "D", "E"):
                result["answer"] = letter
            result["prompt_tokens"] = timings.get("prompt_n", 0)
        elif problem.get("text_only", True) and has_numeric_options(problem):
            matched, timings, meta = solve_pot(problem, timeout=max(remaining, 1.0), url=llm_url)
            if matched in ("A", "B", "C", "D", "E"):
                result["answer"] = matched
            result["prompt_tokens"] = timings.get("prompt_n", 0)
            result["pot"] = meta
    except (socket.timeout, TimeoutError):
        result["status"] = "timeout"
    except urllib.error.URLError as error:
        if isinstance(error.reason, (socket.timeout, TimeoutError)):
            result["status"] = "timeout"
        else:
            result["status"] = f"error: {error.reason}"
    except Exception as error:
        result["status"] = f"error: {type(error).__name__}"
    result["seconds"] = round(time.perf_counter() - start, 2)
    return result
