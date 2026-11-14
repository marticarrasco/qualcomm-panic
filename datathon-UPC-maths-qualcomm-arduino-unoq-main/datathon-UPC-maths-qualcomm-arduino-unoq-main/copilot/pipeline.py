"""Shared solver for the official predictor and the evaluation runner."""
import socket
import sys
import time
import urllib.error
from functools import lru_cache
from pathlib import Path

from . import reasoner, text_solver
from .arithmetic import complete_options, numeric_options
from .config import CONFIG, ROOT
from .llm import URL, answer_letter
from .priors import guess, median_choice

TIME_LIMIT = float(CONFIG["TIME_LIMIT_SECONDS"])
FALLBACK = CONFIG["FALLBACK_ANSWER"]
MARGIN = 3.0  # seconds kept free for Python and for writing the answer
ANSWER_MODE = CONFIG.get("ANSWER_MODE", "direct")
TEXT_PROFILE = CONFIG.get("TEXT_PROFILE", "brief")
TEXT_MAX_TOKENS = int(CONFIG.get("TEXT_MAX_TOKENS", 160))
TEXT_MEDIAN_BYPASS = CONFIG.get("TEXT_MEDIAN_BYPASS", "1") == "1"
TEXT_FALLBACK = CONFIG.get("TEXT_FALLBACK", "prior")
TEXT_TRAIN = CONFIG.get("TEXT_TRAIN", "")
TEXT_RAG_K = int(CONFIG.get("TEXT_RAG_K", "3"))
TEXT_RAG_INDEX = CONFIG.get("TEXT_RAG_INDEX", "data/index-text.db")
TEXT_RAG_SAME_LANGUAGE = CONFIG.get("TEXT_RAG_SAME_LANGUAGE", "1") == "1"
TEXT_RAG_MIN_SCORE = float(CONFIG.get("TEXT_RAG_MIN_SCORE", "0"))
TEXT_RAG_MAX_CHARS = int(CONFIG.get("TEXT_RAG_MAX_CHARS", "320"))
TEXT_ROUTE = CONFIG.get("TEXT_ROUTE", "all")
# Numeric options are a weak spot for the letter model and a strong spot for
# the train-fitted ranker, so they can be routed without a model call.
TEXT_NUMERIC_ROUTE = CONFIG.get("TEXT_NUMERIC_ROUTE", "model")
UNCERTAIN_THRESHOLD = 0.29
TEXT_BOARD_BUDGET = (float(CONFIG["ARDUINO_PROMPT_TOKENS_PER_SECOND"]),
                     float(CONFIG["ARDUINO_OUTPUT_TOKENS_PER_SECOND"]),
                     float(CONFIG.get("TEXT_TARGET_SECONDS", 54)))


@lru_cache(maxsize=1)
def _train_index():
    from .train_match import LexicalMatch
    if TEXT_TRAIN:
        path = Path(ROOT) / TEXT_TRAIN
        if path.is_file():
            return LexicalMatch(path)
    return None


def prepare():
    """Load immutable retrieval data once, before the timed question loop."""
    if ANSWER_MODE == "text" and TEXT_TRAIN:
        _train_index()
    if ANSWER_MODE == "text" and TEXT_PROFILE == "retrieved" and TEXT_RAG_K > 0:
        if _text_rag_index() is None:
            print(f"warning: {TEXT_RAG_INDEX} not found; solving without retrieved examples",
                  file=sys.stderr)
    if ANSWER_MODE == "text" and (TEXT_FALLBACK == "ranker" or TEXT_ROUTE != "all"):
        from .option_ranker import load_weights
        load_weights()


def batch_solver(problems):
    prepare()
    if ANSWER_MODE == "text" and CONFIG.get("TEXT_TRANSLATIONS") == "1":
        from .translations import TranslationBatch
        return TranslationBatch(problems, solve)
    return solve


@lru_cache(maxsize=1)
def _text_rag_index():
    path = Path(ROOT) / TEXT_RAG_INDEX
    if path.is_file():
        from .rag import Index
        return Index(path)
    return None


def _solve_text(problem, deadline, url, exclude_id, retrieve, text_rag=False):
    if TEXT_ROUTE == "model":
        # Diagnostic mode: no rules, ranker, prior or retrieval. Every problem
        # goes to Qwen; a missing answer falls back only to satisfy the A-E format.
        answer, status, text, timings = text_solver.infer(
            problem, deadline, url, TEXT_PROFILE, TEXT_MAX_TOKENS, (),
            board_budget=TEXT_BOARD_BUDGET)
        return answer or FALLBACK, status, text, timings
    flag = problem.get("text_only")
    if flag is not None and str(flag).strip().casefold() in {"false", "0", "no"}:
        return reasoner.answer(problem, deadline, url)
    prior = guess(problem)
    if not complete_options(problem):
        return prior, "prior", "", {}
    from .rules import answer as solve_rule
    if letter := solve_rule(problem):
        return letter, "exact_math", "", {}
    index = _train_index() if TEXT_TRAIN else None
    if index and (letter := index.answer(problem, exclude_id)):
        return letter, "train_match", "", {}
    if TEXT_NUMERIC_ROUTE == "ranker" and numeric_options(problem) is not None:
        from .option_ranker import probabilities
        choices = probabilities(problem)
        return max(choices, key=choices.get), "ranker", "", {}
    median = median_choice(problem)
    if TEXT_MEDIAN_BYPASS and median is not None:
        return median, "prior", "", {}
    if TEXT_FALLBACK == "ranker" or TEXT_ROUTE != "all":
        from .option_ranker import probabilities
        choices = probabilities(problem)
        best = max(choices, key=choices.get)
        if TEXT_ROUTE == "ranker" or (TEXT_ROUTE == "uncertain" and choices[best] >= UNCERTAIN_THRESHOLD):
            return best, "ranker", "", {}
        if TEXT_FALLBACK == "ranker":
            prior = best
    answer, status, text, timings = text_solver.infer(
        problem, deadline, url, TEXT_PROFILE, TEXT_MAX_TOKENS, retrieve(),
        board_budget=TEXT_BOARD_BUDGET,
        example_max_chars=TEXT_RAG_MAX_CHARS if text_rag else 0)
    return answer or prior, status, text, timings


def solve(problem, index=None, k=0, exclude_id=None, image_b64=None, llm_url=URL,
          time_limit=TIME_LIMIT):
    """Return {"answer", "seconds", "status", ...} for one problem.

    index and k turn on retrieval: the k most similar solved problems are
    added to the prompt as examples. Each example costs reading time.
    image_b64 and llm_url send the problem image to a vision server.
    """
    budget_mode = ANSWER_MODE == "budget" and image_b64 is None
    text_mode = ANSWER_MODE == "text" and image_b64 is None
    start = time.perf_counter()
    result = {"answer": FALLBACK, "status": "ok", "prompt_tokens": None,
              "completion_tokens": None, "model_output": None, "retrieved": "", "examples": []}
    use_text_rag = (text_mode and index is None and k == 0
                    and TEXT_PROFILE == "retrieved" and TEXT_RAG_K > 0
                    and _text_rag_index() is not None)
    def retrieve():
        active_index = index
        active_k = k
        same_language = False
        min_score = 0.0
        if active_index is None and ANSWER_MODE == "text" and TEXT_PROFILE == "retrieved":
            active_index = _text_rag_index()
            active_k = TEXT_RAG_K
            same_language = TEXT_RAG_SAME_LANGUAGE
            min_score = TEXT_RAG_MIN_SCORE
        if active_index is not None and active_k > 0:
            try:
                hits = active_index.search(problem, k=active_k,
                                           exclude_id=exclude_id or problem.get("id"),
                                           timeout=min(1.0, max(0.05, time_limit - MARGIN)),
                                           same_language=same_language,
                                           min_score=min_score)
            except OSError:
                hits = []
            result["examples"] = hits
            result["retrieved"] = ";".join(f"{p['id']}:{s:.2f}" for s, p in hits)
            return [p for _, p in hits]
        return []

    try:
        if text_mode:
            letter, result["status"], model_output, timings = _solve_text(
                problem, start + time_limit - MARGIN, llm_url, exclude_id, retrieve,
                text_rag=use_text_rag)
        elif budget_mode:
            letter, result["status"], model_output, timings = reasoner.answer(
                problem, deadline=start + time_limit - MARGIN, url=llm_url)
        else:
            examples = retrieve()
            remaining = time_limit - (time.perf_counter() - start) - MARGIN
            if remaining <= 0:
                raise TimeoutError
            letter, timings, model_output = answer_letter(
                problem, examples, timeout=remaining, image_b64=image_b64, url=llm_url)
        result["answer"] = letter
        result["model_output"] = model_output
        result["prompt_tokens"] = timings.get("prompt_n")
        result["completion_tokens"] = timings.get("predicted_n")
        result["full_prompt_tokens"] = timings.get("full_prompt_n")
        result["cached_prompt_tokens"] = timings.get("cached_n")
        if result["status"] in {"prior", "ranker", "train_match", "exact_math"}:
            result["prompt_tokens"] = result["completion_tokens"] = 0
    except (socket.timeout, TimeoutError):
        result["status"] = "timeout"
    except urllib.error.URLError as error:
        if isinstance(error.reason, (socket.timeout, TimeoutError)):
            result["status"] = "timeout"
        else:
            result["status"] = f"error: {error.reason}"
    except OSError:
        result["status"] = "model_error"
    except ValueError as error:
        timings = getattr(error, "timings", {})
        result["model_output"] = getattr(error, "output", None)
        result["prompt_tokens"] = timings.get("prompt_n")
        result["completion_tokens"] = timings.get("predicted_n")
        result["status"] = "invalid_response"
    result["seconds"] = round(time.perf_counter() - start, 2)
    return result
