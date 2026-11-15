"""The baseline request of copilot/llm.py (same SYSTEM, format, grammar, temperature, seed),
plus the log probabilities of A..E. Rows are compatible with the repo's analyze_logprobs.py.
Usage: python -I dump_lp.py --url http://127.0.0.1:8091 --input x.jsonl --out y.jsonl [--languages eng]"""
import argparse
import json
import math
import random
import re
import sys
import time
import urllib.request

LETTERS = "ABCDE"
LETTER_SET = set(LETTERS)
NUMBER = re.compile(r"^[-+−]?\d+(?:[.,]\d+)?$")

# verbatim from copilot/llm.py
SYSTEM = ("You solve multiple-choice math competition problems for school students. "
          "Read the problem and the options, then reply with the letter of the correct option.")


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


def build_messages(problem):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": format_problem(problem) + "\nAnswer:"}]


def post(url, payload, timeout):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def option_kind(options):  # verbatim rule from the repo's dump_logprobs.py
    if not options:
        return "none"
    values = [str(options.get(letter, "")).strip() for letter in LETTERS]
    if not all(NUMBER.match(v) for v in values):
        return "other"
    numbers = [float(v.replace("−", "-").replace(",", ".")) for v in values]
    if all(a < b for a, b in zip(numbers, numbers[1:])):
        return "num_inc"
    if all(a > b for a, b in zip(numbers, numbers[1:])):
        return "num_dec"
    return "num_unsorted"


def ask(url, messages, top, timeout):
    response = post(url + "/v1/chat/completions", {
        "messages": messages, "grammar": "root ::= [A-E]", "max_tokens": 1, "temperature": 0,
        "seed": 42, "cache_prompt": True, "logprobs": True, "top_logprobs": top}, timeout)
    choice = response["choices"][0]
    letter = (choice.get("message", {}).get("content") or "").strip()
    content = (choice.get("logprobs") or {}).get("content") or []
    candidates = (content[0].get("top_logprobs") or []) if content else []
    return letter, candidates, response.get("timings", {})


def letter_logprobs(candidates):
    found = {}
    for c in candidates:
        token, value = c.get("token"), c.get("logprob")
        if token in LETTER_SET and value is not None:
            found[token] = max(found.get(token, -math.inf), float(value))
    return found


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8091")
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--languages", default="")
    p.add_argument("--top", type=int, default=100)
    p.add_argument("--include-empty", action="store_true")
    args = p.parse_args()

    with open(args.input, encoding="utf-8") as f:
        problems = [json.loads(line) for line in f if line.strip()]
    languages = set(filter(None, args.languages.split(",")))
    if languages:
        problems = [x for x in problems if x.get("language") in languages]
    random.Random(42).shuffle(problems)  # the order eval.py uses

    # The investigation's content-free prompt, for contextual calibration
    dummy = {"text": "Problem: N/A", "options": {k: "N/A" for k in LETTERS}}
    letter, cands, _ = ask(args.url, build_messages(dummy), args.top, 120)
    null = letter_logprobs(cands)
    print("null prompt:", letter, {k: round(v, 3) for k, v in sorted(null.items())},
          "| top-5 tokens:", [c.get("token") for c in cands[:5]], flush=True)

    t0 = time.time()
    n_ok, hits = 0, 0
    with open(args.out, "w", encoding="utf-8") as out:
        out.write(json.dumps({"null_logprobs": null}) + "\n")
        for i, problem in enumerate(problems, 1):
            options = problem.get("options") or {}
            row = {"id": problem["id"], "language": problem.get("language", ""), "level": problem.get("level"),
                   "year": problem.get("year"), "text_only": problem.get("text_only"), "label": problem.get("label"),
                   "has_options": bool(options), "option_kind": option_kind(options)}
            if not options and not args.include_empty:
                row["status"] = "no_options"
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue
            start = time.perf_counter()
            letter, cands, timings = ask(args.url, build_messages(problem), args.top, 300)
            found = letter_logprobs(cands)
            top5 = letter_logprobs(cands[:5])
            row.update({"answer": letter,
                        "logprobs": {l: (round(found[l], 4) if l in found else None) for l in LETTERS},
                        "letters_found": len(found), "letters_in_top5": len(top5),
                        "letter_mass": round(sum(math.exp(v) for v in found.values()), 4),
                        "top_token": cands[0].get("token") if cands else None,
                        "seconds": round(time.perf_counter() - start, 2),
                        "prompt_n": timings.get("prompt_n"), "cache_n": timings.get("cache_n"),
                        "prompt_per_second": timings.get("prompt_per_second"), "status": "ok"})
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_ok += 1
            hits += letter == row["label"]
            if i % 50 == 0:
                print(f"  {i}/{len(problems)}  running accuracy {100 * hits / n_ok:.1f}%  "
                      f"{time.time() - t0:.0f} s", flush=True)
    print(f"done: {n_ok} asked, baseline {hits}/{n_ok} = {100 * hits / max(n_ok, 1):.1f}%, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
