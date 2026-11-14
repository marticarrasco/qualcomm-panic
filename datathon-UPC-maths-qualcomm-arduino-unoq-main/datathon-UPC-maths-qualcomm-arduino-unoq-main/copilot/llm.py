"""Ask the small language model to solve a problem and return its answer letter."""
import re

from .config import CONFIG
from .client import post

URL = f"http://127.0.0.1:{CONFIG['LLM_PORT']}/v1/chat/completions"

# Keep the system prompt stable: llama-server caches the start of the prompt,
# so text that never changes is only read once.
SYSTEM = ("You solve multiple-choice math competition problems for school students. "
          "Work out the problem before choosing an option. Reply in at most two short lines: "
          "first give the key calculation or reason, then finish with exactly 'Answer: X', "
          "where X is the letter of the correct option (A, B, C, D, or E). "
          "Do not choose an option just because of its letter.")

ANSWER_RE = re.compile(r"(?:answer|resposta)\s*[:：]\s*([A-E])\b", re.IGNORECASE)


class InvalidAnswerError(ValueError):
    """An unparseable model answer, with any token usage the server reported."""

    def __init__(self, message, timings=None):
        super().__init__(message)
        self.timings = timings or {}


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
    content = response["choices"][0]["message"]["content"].strip()
    matches = ANSWER_RE.findall(content)
    if matches:
        letter = matches[-1].upper()
    elif re.fullmatch(r"[A-E][.!]?", content, re.IGNORECASE):
        letter = content[0].upper()
    else:
        raise InvalidAnswerError("model did not return an answer in the expected format",
                                 response.get("timings", {}))
    return letter, response.get("timings", {})
