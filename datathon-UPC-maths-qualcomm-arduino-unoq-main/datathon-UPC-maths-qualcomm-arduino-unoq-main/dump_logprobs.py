#!/usr/bin/env python3
"""Save what the model thinks of all five letters, not only the one it picks.

    python3 dump_logprobs.py --limit 3                        # smoke test, half a minute
    python3 dump_logprobs.py --languages eng --name val-eng   # about 35 minutes
    python3 analyze_logprobs.py results/logprobs-val-eng.jsonl

It sends the baseline request (the same prompt, grammar, temperature and seed
as copilot/llm.py) and also asks for "logprobs". The log probabilities come
from the same forward pass, so a problem takes as long as in predict.py.

Each row goes to results/logprobs-<name>.jsonl: the id, the label, what the
options look like, and the log probability of A to E. The problem text is never
written. With that file, analyze_logprobs.py tries letter-bias corrections and
answer priors in seconds, with no board and no model.

Problems without options are not sent to the model (it can't see what the
letters stand for), unless you add --include-empty to measure that too.

Stop it and start it again whenever you like: finished problems are skipped.
Don't run eval.py or the explorer at the same time: the server answers one
request at a time, so the timings would be wrong.
"""
import argparse
import json
import math
import os
import random
import re
import socket
import statistics
import sys
import time
import urllib.error

from copilot.client import post
from copilot.config import CONFIG
from copilot.llm import URL, build_messages

LETTERS = "ABCDE"
LETTER_SET = set(LETTERS)
NUMBER = re.compile(r"^[-+−]?\d+(?:[.,]\d+)?$")
MAX_ERRORS_IN_A_ROW = 3


def option_kind(options):
    """What the options look like, without their text. The answer priors depend on it:
    when five numbers are in increasing order, the answer is rarely the first or the last."""
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


def ask(problem, top, timeout):
    """The baseline request of copilot/llm.py, plus the probabilities of the answer token.

    The grammar only limits the token that is picked. The candidates are the model's
    top tokens before the grammar, so A to E are read from that list.
    """
    response = post(URL, {
        "messages": build_messages(problem),
        "grammar": "root ::= [A-E]",
        "max_tokens": 1,
        "temperature": 0,
        "seed": int(CONFIG["SEED"]),
        "cache_prompt": True,
        "logprobs": True,
        "top_logprobs": top,
    }, timeout=timeout)
    choice = response["choices"][0]
    letter = (choice.get("message", {}).get("content") or "").strip()
    content = (choice.get("logprobs") or {}).get("content") or []
    candidates = (content[0].get("top_logprobs") or []) if content else []
    return letter, candidates, response.get("timings", {}), choice


def letter_logprobs(candidates):
    """{"A": -1.2, ...} for the letters found among the candidates. Only the bare
    letter counts, as in the grammar: " A" or "A)" are other tokens."""
    found = {}
    for candidate in candidates:
        token, value = candidate.get("token"), candidate.get("logprob")
        if token in LETTER_SET and value is not None:
            found[token] = max(found.get(token, -math.inf), float(value))
    return found


def load_done(path, include_empty):
    """Keys that already have a usable row, so a restart skips them."""
    done = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("status") == "ok" or (row.get("status") == "no_options" and not include_empty):
                    done.add(f"{row['id']}|{row.get('language', '')}")
    return done


def is_timeout(error):
    return isinstance(error, (socket.timeout, TimeoutError)) or (
        isinstance(error, urllib.error.URLError) and isinstance(error.reason, (socket.timeout, TimeoutError)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="data/val.jsonl")
    p.add_argument("--limit", type=int, default=0, help="0 means every problem (the default)")
    p.add_argument("--languages", default="", help="e.g. eng,cat")
    p.add_argument("--name", default="", help="output is results/logprobs-<name>.jsonl")
    p.add_argument("--top", type=int, default=100, help="how many of the model's top tokens to ask for")
    p.add_argument("--timeout", type=float, default=120, help="seconds to wait for one problem")
    p.add_argument("--include-empty", action="store_true",
                   help="also ask the model about problems without options")
    args = p.parse_args()

    with open(args.input, encoding="utf-8") as f:
        problems = [json.loads(line) for line in f if line.strip()]
    languages = set(filter(None, args.languages.split(",")))
    if languages:
        problems = [x for x in problems if x.get("language") in languages]
    random.Random(42).shuffle(problems)  # the same order as eval.py, so --limit 50 is the same 50
    if args.limit:
        problems = problems[:args.limit]
    if not problems:
        sys.exit("No problems matched. Check --input and --languages.")

    stem = os.path.splitext(os.path.basename(args.input))[0]
    name = args.name or "-".join([stem] + sorted(languages))
    os.makedirs("results", exist_ok=True)
    path = f"results/logprobs-{name}.jsonl"
    done = load_done(path, args.include_empty)
    todo = [x for x in problems if f"{x['id']}|{x.get('language', '')}" not in done]
    to_ask = sum(1 for x in todo if x.get("options") or args.include_empty)
    print(f"{len(problems)} problems, {len(problems) - len(todo)} already in {path}, "
          f"{to_ask} to ask the model", flush=True)

    asked, seconds, hits, with_label, errors_in_a_row = 0, [], 0, 0, 0
    with open(path, "a", encoding="utf-8") as out:
        for problem in todo:
            options = problem.get("options") or {}
            row = {"id": problem["id"], "language": problem.get("language", ""),
                   "level": problem.get("level"), "year": problem.get("year"),
                   "text_only": problem.get("text_only"), "label": problem.get("label"),
                   "has_options": bool(options), "option_kind": option_kind(options)}
            if not options and not args.include_empty:
                row["status"] = "no_options"
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue

            asked += 1
            start = time.perf_counter()
            try:
                letter, candidates, timings, choice = ask(problem, args.top, args.timeout)
            except urllib.error.HTTPError as error:
                body = error.read().decode(errors="replace")[:500]
                sys.exit(f"The server refused the request (HTTP {error.code}): {body}\n"
                         "If it complains about top_logprobs, try a smaller --top, such as 20.")
            except (urllib.error.URLError, OSError, ValueError, KeyError, IndexError) as error:
                reason = getattr(error, "reason", error)
                if isinstance(reason, ConnectionRefusedError):
                    sys.exit(f"No model server at {URL}. Check: systemctl status math-llm")
                row["status"] = "timeout" if is_timeout(error) else f"error: {reason!r}"
                row["seconds"] = round(time.perf_counter() - start, 2)
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                print(f"[{asked}/{to_ask}] {row['id']} {row['language']}: {row['status']}", flush=True)
                errors_in_a_row += 1
                if errors_in_a_row >= MAX_ERRORS_IN_A_ROW:
                    sys.exit(f"{MAX_ERRORS_IN_A_ROW} failures in a row: stopping. Finished rows are kept in {path}.")
                continue
            errors_in_a_row = 0
            elapsed = time.perf_counter() - start

            if asked == 1:  # show what the server really sends, once
                if not candidates:
                    sys.exit("The server sent no log probabilities. The answer has these fields: "
                             f"{sorted(choice)}. Nothing was written for this problem.")
                shown = ", ".join(f"{c.get('token')!r} {float(c.get('logprob') or 0):.2f}" for c in candidates[:8])
                print(f"First answer, the model's top tokens: {shown}", flush=True)

            found = letter_logprobs(candidates)
            row.update({
                "answer": letter,
                "logprobs": {l: (round(found[l], 4) if l in found else None) for l in LETTERS},
                "letters_found": len(found),
                "letter_mass": round(sum(math.exp(v) for v in found.values()), 4),
                "top_token": candidates[0].get("token") if candidates else None,
                "seconds": round(elapsed, 2),
                "prompt_n": timings.get("prompt_n"),
                "cache_n": timings.get("cache_n"),
                "prompt_per_second": timings.get("prompt_per_second"),
                "status": "ok",
            })
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()

            seconds.append(elapsed)
            if row["label"]:
                with_label += 1
                hits += letter == row["label"]
            eta = statistics.mean(seconds) * (to_ask - asked) / 60
            print(f"[{asked}/{to_ask}] {row['id']} {row['language']}: {letter} (truth {row['label']}) "
                  f"{elapsed:.1f} s, read {row['prompt_n']} tokens, letters {len(found)}/5, "
                  f"mass {row['letter_mass']:.2f}, about {eta:.0f} min left", flush=True)
            if asked == 1 and len(found) < 5:
                print(f"  Only {len(found)} of the 5 letters are in the model's top {args.top} tokens. "
                      "The analysis gives the missing ones a floor value. If most rows say "
                      "'letters 5/5', that's fine; if not, stop and raise --top.", flush=True)

    print("\n" + "=" * 60)
    if seconds:
        print(f"Asked {asked} problems: mean {statistics.mean(seconds):.1f} s, worst {max(seconds):.1f} s")
    if with_label:
        print(f"The model's letter was right in {hits}/{with_label} = {100 * hits / with_label:.1f}% of them")
    print(f"Rows: {path}\nNext: python3 analyze_logprobs.py {path}")


if __name__ == "__main__":
    main()
