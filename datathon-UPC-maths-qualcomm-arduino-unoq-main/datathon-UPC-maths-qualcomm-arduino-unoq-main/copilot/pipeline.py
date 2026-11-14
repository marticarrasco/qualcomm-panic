"""The baseline copilot: one problem in, one letter out.

This is the file to change first. predict.py and eval.py both call solve(),
so whatever you build here is exactly what the official run measures.
"""
import socket
import time
import urllib.error

from .config import CONFIG
from .llm import URL, answer_letter

TIME_LIMIT = float(CONFIG["TIME_LIMIT_SECONDS"])
FALLBACK = CONFIG["FALLBACK_ANSWER"]
MARGIN = 3.0  # seconds kept free for Python and for writing the answer


def solve(problem, index=None, k=0, exclude_id=None, image_b64=None, llm_url=URL,
          time_limit=TIME_LIMIT):
    """Return {"answer", "seconds", "status", ...} for one problem.

    index and k turn on retrieval: the k most similar solved problems are
    added to the prompt as examples. Each example costs reading time.
    image_b64 and llm_url send the problem image to a vision server.
    """
    start = time.perf_counter()
    result = {"answer": FALLBACK, "status": "ok", "prompt_tokens": None,
              "completion_tokens": None, "model_output": None, "retrieved": "", "examples": []}
    try:
        examples = []
        if index is not None and k > 0:
            hits = index.search(problem, k=k, exclude_id=exclude_id)
            examples = [p for _score, p in hits]
            result["examples"] = hits
            result["retrieved"] = ";".join(f"{p['id']}:{s:.2f}" for s, p in hits)
        remaining = time_limit - (time.perf_counter() - start) - MARGIN
        letter, timings, model_output = answer_letter(
            problem, examples, timeout=max(remaining, 1.0), image_b64=image_b64, url=llm_url)
        result["answer"] = letter
        result["model_output"] = model_output
        result["prompt_tokens"] = timings.get("prompt_n")
        result["completion_tokens"] = timings.get("predicted_n")
    except (socket.timeout, TimeoutError):
        result["status"] = "timeout"
    except urllib.error.URLError as error:
        if isinstance(error.reason, (socket.timeout, TimeoutError)):
            result["status"] = "timeout"
        else:
            result["status"] = f"error: {error.reason}"
    except ValueError as error:
        timings = getattr(error, "timings", {})
        result["model_output"] = getattr(error, "output", None)
        result["prompt_tokens"] = timings.get("prompt_n")
        result["completion_tokens"] = timings.get("predicted_n")
        result["status"] = "invalid_response"
    result["seconds"] = round(time.perf_counter() - start, 2)
    return result
