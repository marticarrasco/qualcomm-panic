#!/usr/bin/env python3
"""Pure model evaluation: every problem is answered by the language model alone.

No rules, no option ranker, no median/prior and no retrieval. The model receives
the problem text and its A-E options and must reply with a short reasoning plus
an "Answer: X" line. Output is capped at 128 tokens.

    python eval_model_only.py --limit 8          # smoke test
    python eval_model_only.py                    # all English text-only problems

Defaults: data/eval_sets/val_text_only_all.jsonl, language eng, text_only true.
"""
import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime

from copilot.client import get, post
from copilot.config import CONFIG
from copilot.llm import format_problem

SYSTEM_TWO = (
    "You solve multiple-choice math competition problems. "
    "Reason in exactly two short sentences: the first gives the key calculation or "
    "deduction, the second checks the result against the options. Then write a final "
    "line exactly 'Answer: X', where X is the letter A, B, C, D or E of the correct "
    "option. Do not restate the question and do not add any other text.\n\n"
    "Example:\n"
    "7 pens cost $21. How much do 5 pens cost? A) 10 B) 15 C) 20 D) 25 E) 30\n"
    "Each pen costs $3, so 5 pens cost $15. This matches option B.\n"
    "Answer: B"
)

SYSTEM_ONE = (
    "You solve multiple-choice math competition problems. "
    "Reason in one short sentence giving the key calculation or deduction. Then write a "
    "final line exactly 'Answer: X', where X is the letter A, B, C, D or E of the correct "
    "option. Do not restate the question and do not add any other text.\n\n"
    "Example:\n"
    "7 pens cost $21. How much do 5 pens cost? A) 10 B) 15 C) 20 D) 25 E) 30\n"
    "Each pen costs $3, so 5 pens cost $15, which is option B.\n"
    "Answer: B"
)

SYSTEM_STRICT = (
    "You answer multiple-choice math competition questions. "
    "For each question: write ONE short sentence of reasoning (maximum 25 words), then a "
    "new line exactly 'Answer: X' where X is the correct option letter A, B, C, D or E. "
    "Never write more than one sentence. Never repeat the question. No other text.\n\n"
    "Example 1:\n"
    "7 pens cost $21. How much do 5 pens cost? A) 10 B) 15 C) 20 D) 25 E) 30\n"
    "One pen costs $3, so five pens cost $15, option B.\n"
    "Answer: B\n\n"
    "Example 2:\n"
    "A square has perimeter 36. What is its area? A) 18 B) 36 C) 72 D) 81 E) 144\n"
    "The side is 36/4=9, so the area is 9*9=81, option D.\n"
    "Answer: D"
)

PROMPTS = {"two": SYSTEM_TWO, "one": SYSTEM_ONE, "strict": SYSTEM_STRICT}
# Repeated after the problem, where a small model pays the most attention.
PROMPT_SUFFIX = {
    "strict": "\n\nOne sentence of reasoning (max 25 words), then 'Answer: X'.",
}

def one_line_grammar(chars):
    """Force one reasoning line, then a parseable answer.

    Without a constraint a 0.8B model tends to write four or five sentences and
    run out of the 128-token budget before it reaches 'Answer: X'.
    """
    chars = max(20, min(1000, int(chars)))
    return f'root ::= work "Answer: " [A-E]\nwork ::= [^\\n]{{0,{chars}}} "\\n"\n'

ANSWER_RE = re.compile(
    r"(?i)(?:final\s+)?answer\s*[:：]\s*\**\s*\(?([A-E])\b")
ANSWER_IS_RE = re.compile(
    r"(?i)\banswer\s+is\s*\**\s*\(?([A-E])\b")
OPTION_RE = re.compile(
    r"(?i)\b(?:option|choice|letter)\s*\(?([A-E])\b")
TRAILING_RE = re.compile(r"(?s)\b([A-E])\b[.!]?\s*$")
SENTENCE_RE = re.compile(r"[.!?]+(?=\s|$)")

FIELDS = [
    "eval_id", "evaluated_at", "problem_id", "language", "level",
    "selected_answer", "correct_answer", "is_correct", "status",
    "reasoning_sentences", "reasoning", "model_output", "finish_reason",
    "prompt_tokens", "output_tokens", "elapsed_seconds", "prompt_chars",
]


def is_text_only(problem):
    value = problem.get("text_only", False)
    return value is True or value == 1 or (
        isinstance(value, str) and value.strip().lower() in {"true", "1", "yes"})


def load_problems(path, languages, only_text):
    with open(path, encoding="utf-8-sig") as source:
        problems = [json.loads(line) for line in source if line.strip()]
    if only_text:
        problems = [problem for problem in problems if is_text_only(problem)]
    if languages:
        problems = [problem for problem in problems if problem.get("language") in languages]
    return problems


def parse_answer(output):
    """Return the answer letter, or None when the reply has no usable letter."""
    for pattern in (ANSWER_RE, ANSWER_IS_RE, OPTION_RE):
        matches = pattern.findall(output)
        if matches:
            return matches[-1].upper()
    tail = output.strip()[-40:]
    if match := TRAILING_RE.search(tail):
        return match[1].upper()
    return None


def split_reasoning(output, answer):
    """Text before the final answer line, used to count how many sentences were written."""
    if not answer:
        return output.strip()
    marker = re.search(r"(?i)(?:final\s+)?answer\s*[:：]", output)
    if marker:
        return output[:marker.start()].strip()
    return ""


def ask(problem, url, system, max_tokens, timeout, grammar=None, suffix=""):
    payload = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": format_problem(problem) + suffix},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": int(CONFIG["SEED"]),
        "cache_prompt": True,
    }
    if grammar:
        payload["grammar"] = grammar
    response = post(url, payload, timeout=timeout)
    choice = response["choices"][0]
    return {
        "output": (choice.get("message") or {}).get("content") or "",
        "finish_reason": choice.get("finish_reason") or "",
        "timings": response.get("timings") or {},
    }


def summarize(rows):
    if not rows:
        return {}
    hits = [bool(row["is_correct"]) for row in rows]
    seconds = [row["elapsed_seconds"] for row in rows]
    prompt_tokens = [row["prompt_tokens"] for row in rows
                     if row["prompt_tokens"] is not None]
    output_tokens = [row["output_tokens"] for row in rows
                     if row["output_tokens"] is not None]
    by_level = defaultdict(list)
    for row in rows:
        by_level[row["level"]].append(bool(row["is_correct"]))
    return {
        "correct": sum(hits),
        "total": len(rows),
        "accuracy_percent": round(100 * sum(hits) / len(rows), 2),
        "mean_elapsed_seconds": round(statistics.mean(seconds), 2),
        "mean_prompt_tokens": round(statistics.mean(prompt_tokens), 1) if prompt_tokens else None,
        "mean_output_tokens": round(statistics.mean(output_tokens), 1) if output_tokens else None,
        "total_output_tokens": sum(output_tokens),
        "status": dict(Counter(row["status"] for row in rows)),
        "sentences": dict(Counter(row["reasoning_sentences"] for row in rows)),
        "predicted_letters": dict(Counter(row["selected_answer"] for row in rows)),
        "true_letters": dict(Counter(row["correct_answer"] for row in rows)),
        "by_level": {
            str(level): {"correct": sum(values), "total": len(values),
                         "accuracy_percent": round(100 * sum(values) / len(values), 2)}
            for level, values in sorted(by_level.items())
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="data/eval_sets/val_text_only_all.jsonl")
    parser.add_argument("--languages", default="eng", help="comma-separated, default eng")
    parser.add_argument("--all-languages", action="store_true",
                        help="ignore --languages")
    parser.add_argument("--all-text", action="store_true",
                        help="do not filter text_only")
    parser.add_argument("--limit", type=int, default=0, help="0 means every problem")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--prompt", choices=sorted(PROMPTS), default="two",
                        help="one or two reasoning sentences")
    parser.add_argument("--grammar", action="store_true",
                        help="constrain the reply to one reasoning line plus 'Answer: X'")
    parser.add_argument("--work-chars", type=int, default=240,
                        help="maximum characters in the constrained reasoning line")
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--name", default="model-only-eng")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    languages = set() if args.all_languages else set(filter(None, args.languages.split(",")))
    problems = load_problems(args.input, languages, not args.all_text)
    problems = problems[args.offset:]
    if args.limit:
        problems = problems[:args.limit]
    if not problems:
        sys.exit("no problems selected")

    url = f"http://127.0.0.1:{CONFIG['LLM_PORT']}/v1/chat/completions"
    try:
        models = get(f"http://127.0.0.1:{CONFIG['LLM_PORT']}/v1/models", timeout=10)
        served = ", ".join(model.get("id", "?") for model in models.get("data", []))
    except OSError as error:
        sys.exit(f"model server is not reachable at {url}: {error}")

    os.makedirs(args.results_dir, exist_ok=True)
    now = datetime.now().astimezone()
    stamp = now.strftime("%Y%m%d-%H%M%S-%f")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", args.name).strip("-._") or "eval"
    base = os.path.join(args.results_dir, f"model-only-{stamp}-{slug}")
    csv_path, manifest_path = base + ".csv", base + ".json"

    manifest = {
        "eval_id": os.path.basename(base),
        "name": args.name,
        "status": "running",
        "started_at": now.isoformat(timespec="microseconds"),
        "finished_at": None,
        "input": args.input,
        "languages": sorted(languages) or "all",
        "text_only": not args.all_text,
        "limit": args.limit,
        "offset": args.offset,
        "max_tokens": args.max_tokens,
        "prompt": args.prompt,
        "grammar": one_line_grammar(args.work_chars) if args.grammar else None,
        "prompt_suffix": PROMPT_SUFFIX.get(args.prompt, ""),
        "served_models": served,
        "llm_port": int(CONFIG["LLM_PORT"]),
        "planned_problems": len(problems),
        "completed_problems": 0,
        "system_prompt": PROMPTS[args.prompt],
        "summary": None,
    }
    rows = []

    with open(csv_path, "x", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=FIELDS)
        writer.writeheader()
        out.flush()
        for number, problem in enumerate(problems, 1):
            started = time.perf_counter()
            output, finish, status = "", "", "model_error"
            timings = {}
            try:
                reply = ask(problem, url, PROMPTS[args.prompt], args.max_tokens, args.timeout,
                            manifest["grammar"], PROMPT_SUFFIX.get(args.prompt, ""))
                output, timings = reply["output"], reply["timings"]
                finish = reply["finish_reason"]
                status = "ok"
            except (OSError, ValueError, KeyError) as error:
                output, status = f"ERROR: {error}", "model_error"

            answer = parse_answer(output) if status == "ok" else None
            if status == "ok" and answer is None:
                status = "no_answer"
            reasoning = split_reasoning(output, answer)
            sentences = len(SENTENCE_RE.findall(reasoning)) if reasoning else 0
            correct = problem.get("label", "")
            row = {
                "eval_id": manifest["eval_id"],
                "evaluated_at": datetime.now().astimezone().isoformat(timespec="microseconds"),
                "problem_id": problem.get("id", ""),
                "language": problem.get("language", ""),
                "level": problem.get("level", ""),
                "selected_answer": answer or "",
                "correct_answer": correct,
                "is_correct": int(answer == correct) if answer else 0,
                "status": status,
                "reasoning_sentences": sentences,
                "reasoning": reasoning,
                "model_output": output,
                "finish_reason": finish,
                "prompt_tokens": timings.get("prompt_n"),
                "output_tokens": timings.get("predicted_n"),
                "elapsed_seconds": round(time.perf_counter() - started, 2),
                "prompt_chars": len(format_problem(problem)),
            }
            rows.append(row)
            writer.writerow(row)
            out.flush()
            os.fsync(out.fileno())

            manifest["completed_problems"] = len(rows)
            manifest["summary"] = summarize(rows)
            temporary = manifest_path + ".tmp"
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            os.replace(temporary, manifest_path)

            accuracy = manifest["summary"]["accuracy_percent"]
            print(f"[{number}/{len(problems)}] {row['problem_id']}: "
                  f"{row['selected_answer'] or '-'} (truth {correct}) "
                  f"{row['elapsed_seconds']:.1f} s, {sentences} sentence(s), "
                  f"running accuracy {accuracy:.1f}%", flush=True)

    manifest["status"] = "completed"
    manifest["finished_at"] = datetime.now().astimezone().isoformat(timespec="microseconds")
    manifest["summary"] = summarize(rows)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    summary = manifest["summary"]
    print("\n" + "=" * 60)
    print(f"Accuracy: {summary['correct']}/{summary['total']} = "
          f"{summary['accuracy_percent']:.1f}%  (chance 20.0%)")
    print(f"Status: {summary['status']}")
    print(f"Reasoning sentences in reply: {summary['sentences']}")
    print(f"Letters predicted/true: " + "  ".join(
        f"{letter}:{summary['predicted_letters'].get(letter, 0)}/"
        f"{summary['true_letters'].get(letter, 0)}" for letter in "ABCDE"))
    print(f"Mean time: {summary['mean_elapsed_seconds']:.1f} s")
    print(f"\nDetails: {csv_path}\nRun manifest: {manifest_path}")


if __name__ == "__main__":
    main()
