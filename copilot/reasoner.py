"""Reason inside the time limit: short working, a hard deadline, and a fallback chain.

Why this exists. Measured on the M3Kang validation problems (see experiments/):
asking Qwen3.5 0.8B for one letter is at chance, with or without correcting its
letter bias. When it reasons and reaches its own 'Answer: X' it is right far more
often, but free reasoning rarely finishes in the tokens the UNO Q can write in 60 s.

So this module:
1. keeps the worked examples in the system prompt so llama-server can cache
   them across problems;
2. streams the answer and stops at a wall-clock deadline;
3. answers without the model where a model-free prior is better (copilot/priors.py),
   and falls back to that prior when the reasoning does not reach 'Answer: X'.
"""
import re
import socket
import time
import urllib.error
from contextlib import closing

from .config import CONFIG
from .llm import URL, format_problem
from .priors import guess, median_choice
from .stream import chat

ANSWER_RE = re.compile(r"(?:answer|resposta|respuesta|réponse|antwort)\s*[:：]\s*\**\s*\(?([A-E])\b",
                       re.IGNORECASE)
MAX_TOKENS = 400  # more than the board can write in 60 s; the deadline is what stops it

# The examples are made up, not M3Kang problems.
SYSTEM = """You solve multiple-choice math competition problems for school students. \
Write only the key calculations, one per line, with no explanations, no headings \
and without repeating the problem. Use at most 6 short lines. \
Then finish with 'Answer: X', where X is the letter of the correct option.

Example 1
Anna has 3 bags with 4 apples in each bag. She eats 2 apples. How many apples does she have left?
A) 8
B) 9
C) 10
D) 12
E) 14
3 x 4 = 12
12 - 2 = 10
Answer: C

Example 2
A rectangle has a perimeter of 30 cm. Its length is twice its width. What is its area?
A) 36 cm²
B) 45 cm²
C) 50 cm²
D) 54 cm²
E) 60 cm²
2(w + 2w) = 30, so w = 5
length = 10
area = 5 x 10 = 50
Answer: C

Example 3
How many two-digit numbers have digits that add up to 5?
A) 4
B) 5
C) 6
D) 9
E) 10
14, 23, 32, 41, 50
count = 5
Answer: B"""

def stream_reasoning(problem, deadline, url=URL):
    """Return (text, timings, hit_deadline). deadline is a time.perf_counter() value."""
    payload = {"messages": [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": format_problem(problem)}],
               "max_tokens": MAX_TOKENS, "temperature": 0, "seed": int(CONFIG["SEED"]),
               "cache_prompt": True, "stream": True, "timings_per_token": True}
    text, timings, hit_deadline = "", {}, False
    try:
        with closing(chat(url, payload, deadline)) as events:
            for chunk in events:
                timings = chunk.get("timings") or timings
                for choice in chunk.get("choices", []):
                    text += (choice.get("delta") or {}).get("content") or ""
                if ANSWER_RE.search(text):
                    break  # closing the connection stops the generation
                if time.perf_counter() >= deadline:
                    hit_deadline = True
                    break
    except (socket.timeout, TimeoutError):
        hit_deadline = True
    except urllib.error.URLError as error:
        if not isinstance(error.reason, (socket.timeout, TimeoutError)):
            raise
        hit_deadline = True
    return text, timings, hit_deadline


def answer(problem, deadline, url=URL):
    """Return (letter, status, text, timings).

    Order, from the validation measurements (147 English problems with options):
    1. 'prior': no usable options, or five numeric options printed out of order.
       There the median value beats the model (15 vs 8 right of 36), at no cost.
    2. 'ok': the model reasoned and reached 'Answer: X' before the deadline
       (24 of 84 right where always-D gets 17).
    3. 'fallback': the reasoning did not finish. Matching its last number to an
       option was tried and was worse than the prior, so it is not used.
    """
    options = problem.get("options") or {}
    prior = guess(problem)
    if len(options) < 2 or median_choice(problem) is not None:
        return prior, "prior", "", {}
    text, timings, _hit = stream_reasoning(problem, deadline, url)
    found = ANSWER_RE.findall(text)
    if found:
        return found[-1].upper(), "ok", text, timings
    return prior, "fallback", text, timings
