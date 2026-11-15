#!/usr/bin/env python3
"""Measure accuracy and latency on the board, and save durable per-problem results.

    python3 eval.py --limit 50                    # baseline on 50 validation problems
    python3 eval.py --limit 50 --rag 1 --name rag1
    python3 eval.py --languages cat --limit 100

Each run gets a uniquely named CSV with one row per problem and a JSON manifest
that tracks progress and whether the run completed or was interrupted.
"""
import argparse
import csv
import json
import os
import random
import re
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from datetime import datetime

from copilot.config import CONFIG
from copilot.pipeline import TIME_LIMIT, batch_solver


PROMPT_RATE = float(CONFIG.get("ARDUINO_PROMPT_TOKENS_PER_SECOND", 11))
OUTPUT_RATE = float(CONFIG.get("ARDUINO_OUTPUT_TOKENS_PER_SECOND", 6))

DETAIL_FIELDS = [
    "eval_id", "evaluated_at", "problem_id", "language", "level", "has_options",
    "selected_answer", "correct_answer", "is_correct", "model_output", "input_tokens", "output_tokens",
    "full_input_tokens", "cached_input_tokens", "elapsed_seconds", "arduino_estimated_seconds", "arduino_read_estimated_seconds",
    "arduino_write_estimated_seconds", "arduino_read_tokens_per_second",
    "arduino_write_tokens_per_second", "status", "retrieved",
]


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def table(title, groups):
    print(f"\n{title}")
    for key in sorted(groups, key=str):
        hits = groups[key]
        print(f"  {str(key):12} {sum(hits):4}/{len(hits):<4} {100 * sum(hits) / len(hits):5.1f}%")


def token_count(value):
    """Return a reported token count, or None when the server did not report one."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def is_text_only(problem):
    value = problem.get("text_only", False)
    return value is True or value == 1 or (isinstance(value, str) and value.strip().lower() in {"true", "1", "yes"})


def detail_row(problem, result, eval_id, evaluated_at):
    input_tokens = token_count(result.get("prompt_tokens"))
    output_tokens = token_count(result.get("completion_tokens"))
    read_seconds = round(input_tokens / PROMPT_RATE, 2) if input_tokens is not None else None
    write_seconds = round(output_tokens / OUTPUT_RATE, 2) if output_tokens is not None else None
    estimated_seconds = (round(read_seconds + write_seconds, 2)
                         if read_seconds is not None and write_seconds is not None else None)
    answer = result.get("answer", "")
    correct_answer = problem.get("label", "")
    is_correct = answer == correct_answer if answer and correct_answer else None
    return {
        "eval_id": eval_id,
        "evaluated_at": evaluated_at,
        "problem_id": problem.get("id", ""),
        "language": problem.get("language", ""),
        "level": problem.get("level", ""),
        "has_options": bool(problem.get("options")),
        "selected_answer": answer,
        "correct_answer": correct_answer,
        "is_correct": int(is_correct) if is_correct is not None else None,
        "model_output": result.get("model_output"),
        "input_tokens": input_tokens,
        "full_input_tokens": token_count(result.get("full_prompt_tokens")),
        "cached_input_tokens": token_count(result.get("cached_prompt_tokens")),
        "output_tokens": output_tokens,
        "elapsed_seconds": result.get("seconds", ""),
        "arduino_estimated_seconds": estimated_seconds,
        "arduino_read_estimated_seconds": read_seconds,
        "arduino_write_estimated_seconds": write_seconds,
        "arduino_read_tokens_per_second": PROMPT_RATE,
        "arduino_write_tokens_per_second": OUTPUT_RATE,
        "status": result.get("status", ""),
        "retrieved": result.get("retrieved", ""),
    }


def sync_csv_row(writer, out, row):
    writer.writerow(row)
    out.flush()
    os.fsync(out.fileno())


def save_manifest(path, manifest):
    """Atomically checkpoint run status so partial results remain interpretable."""
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as out:
        json.dump(manifest, out, ensure_ascii=False, indent=2)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    # A concurrent viewer can briefly hold the destination open on Windows.
    for attempt in range(8):
        try:
            os.replace(temporary_path, path)
            break
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.02 * (attempt + 1))


def unique_paths(results_dir, file_stamp, name_slug):
    suffix = 0
    while True:
        extra = f"-{suffix:02d}" if suffix else ""
        base = f"eval-{file_stamp}-{name_slug}{extra}"
        csv_path = os.path.join(results_dir, base + ".csv")
        manifest_path = os.path.join(results_dir, base + ".json")
        if not os.path.exists(csv_path) and not os.path.exists(manifest_path):
            return csv_path, manifest_path, base
        suffix += 1


def update_progress(manifest, rows, written_rows):
    manifest["written_rows"] = written_rows
    manifest["completed_problems"] = len(rows)
    manifest["correct_count"] = sum(bool(row["is_correct"]) for row in rows)
    manifest["accuracy_percent"] = (
        round(100 * manifest["correct_count"] / len(rows), 2) if rows else None
    )
    manifest["total_input_tokens"] = sum(
        row["input_tokens"] for row in rows if row["input_tokens"] is not None
    )
    manifest["total_output_tokens"] = sum(
        row["output_tokens"] for row in rows if row["output_tokens"] is not None
    )
    manifest["mean_elapsed_seconds"] = (
        round(statistics.mean(row["elapsed_seconds"] for row in rows), 2) if rows else None
    )
    estimates = [row["arduino_estimated_seconds"] for row in rows
                 if row["arduino_estimated_seconds"] is not None]
    manifest["mean_arduino_estimated_seconds"] = (
        round(statistics.mean(estimates), 2) if estimates else None
    )
    manifest["estimated_count"] = len(estimates)
    manifest["p95_arduino_estimated_seconds"] = (
        sorted(estimates)[int(0.95 * (len(estimates) - 1))] if estimates else None
    )
    manifest["worst_arduino_estimated_seconds"] = max(estimates) if estimates else None
    manifest["over_time_limit"] = sum(row["elapsed_seconds"] > TIME_LIMIT for row in rows)


def append_ablation(log_path, stamp, name, commit, args, rows, status):
    new = not os.path.exists(log_path)
    seconds = [row["elapsed_seconds"] for row in rows]
    accuracy = (100 * sum(bool(row["is_correct"]) for row in rows) / len(rows)
                if rows else None)
    labels = []
    if args.text_only:
        labels.append("text-only")
    if status != "completed":
        labels.append(status)
    log_name = f"{name} [{', '.join(labels)}]" if labels else name
    with open(log_path, "a", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        if new:
            writer.writerow(["time", "name", "commit", "input", "languages", "n", "rag_k",
                             "accuracy", "mean_s", "worst_s", "over_limit"])
        writer.writerow([
            stamp, log_name, commit, args.input, args.languages or "all", len(rows), args.rag,
            f"{accuracy:.1f}" if accuracy is not None else "",
            f"{statistics.mean(seconds):.1f}" if seconds else "",
            f"{max(seconds):.1f}" if seconds else "",
            sum(value > TIME_LIMIT for value in seconds),
        ])
        out.flush()
        os.fsync(out.fileno())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="data/val.jsonl")
    p.add_argument("--limit", type=int, default=50, help="0 means every problem")
    p.add_argument("--languages", default="", help="e.g. eng,cat")
    p.add_argument("--text-only", action="store_true",
                   help="evaluate only problems marked text_only in the input JSONL")
    p.add_argument("--rag", type=int, default=0, metavar="K")
    p.add_argument("--index", default="data/index.db")
    p.add_argument("--name", default="", help="a label for this experiment")
    p.add_argument("--results-dir", default="results", help="directory for this run and ablation log")
    args = p.parse_args()

    with open(args.input, encoding="utf-8-sig") as source:
        problems = [json.loads(line) for line in source if line.strip()]
    if args.text_only:
        problems = [problem for problem in problems if is_text_only(problem)]
    languages = set(filter(None, args.languages.split(",")))
    if languages:
        problems = [problem for problem in problems if problem.get("language") in languages]
    random.Random(42).shuffle(problems)  # same sample each run, so runs compare
    if args.limit:
        problems = problems[:args.limit]

    index = None
    if args.rag:
        from copilot.rag import Index
        index = Index(args.index)
    solve = batch_solver(problems)

    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)
    now = datetime.now().astimezone()
    evaluated_at = now.isoformat(timespec="microseconds")
    stamp = now.strftime("%Y%m%d-%H%M%S-%f")
    name = args.name or (f"rag{args.rag}" if args.rag else "baseline")
    name_slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-._") or "eval"
    detail_path, manifest_path, eval_id = unique_paths(results_dir, stamp, name_slug)
    ablation_path = os.path.join(results_dir, "ablations.csv")

    manifest = {
        "schema_version": 1,
        "eval_id": eval_id,
        "name": name,
        "status": "running",
        "started_at": evaluated_at,
        "finished_at": None,
        "git_commit": git_commit(),
        "input": args.input,
        "languages": args.languages or "all",
        "text_only": args.text_only,
        "limit": args.limit,
        "rag_k": args.rag,
        "answer_mode": CONFIG.get("ANSWER_MODE", "direct"),
        "llm_alias": CONFIG.get("LLM_ALIAS"),
        "text_profile": CONFIG.get("TEXT_PROFILE") if CONFIG.get("ANSWER_MODE") == "text" else None,
        "text_max_tokens": int(CONFIG["TEXT_MAX_TOKENS"])
        if CONFIG.get("ANSWER_MODE") == "text" else None,
        "text_route": CONFIG.get("TEXT_ROUTE") if CONFIG.get("ANSWER_MODE") == "text" else None,
        "text_numeric_route": CONFIG.get("TEXT_NUMERIC_ROUTE")
        if CONFIG.get("ANSWER_MODE") == "text" else None,
        "text_rag_k": int(CONFIG.get("TEXT_RAG_K", "0"))
        if CONFIG.get("ANSWER_MODE") == "text" else None,
        "text_rag_same_language": CONFIG.get("TEXT_RAG_SAME_LANGUAGE") == "1"
        if CONFIG.get("ANSWER_MODE") == "text" else None,
        "text_rag_min_score": float(CONFIG.get("TEXT_RAG_MIN_SCORE", "0"))
        if CONFIG.get("ANSWER_MODE") == "text" else None,
        "text_rag_max_chars": int(CONFIG.get("TEXT_RAG_MAX_CHARS", "0"))
        if CONFIG.get("ANSWER_MODE") == "text" else None,
        "text_rag_grammar": CONFIG.get("TEXT_RAG_GRAMMAR") == "1"
        if CONFIG.get("ANSWER_MODE") == "text" else None,
        "text_translations": CONFIG.get("TEXT_TRANSLATIONS") == "1",
        "text_fallback": CONFIG.get("TEXT_FALLBACK") if CONFIG.get("ANSWER_MODE") == "text" else None,
        "text_median_bypass": CONFIG.get("TEXT_MEDIAN_BYPASS") == "1"
        if CONFIG.get("ANSWER_MODE") == "text" else None,
        "llm_port": int(CONFIG["LLM_PORT"]),
        "planned_problems": len(problems),
        "written_rows": 0,
        "completed_problems": 0,
        "correct_count": 0,
        "accuracy_percent": None,
        "total_input_tokens": 0,
        "total_output_tokens": 0,
        "mean_elapsed_seconds": None,
        "mean_arduino_estimated_seconds": None,
        "arduino_prompt_tokens_per_second": PROMPT_RATE,
        "arduino_output_tokens_per_second": OUTPUT_RATE,
        "details_csv": os.path.basename(detail_path),
    }
    rows = []
    written_rows = 0
    interrupted = False

    with open(detail_path, "x", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=DETAIL_FIELDS)
        writer.writeheader()
        out.flush()
        os.fsync(out.fileno())
        save_manifest(manifest_path, manifest)

        for n, problem in enumerate(problems, 1):
            started = time.perf_counter()
            try:
                result = solve(problem, index=index, k=args.rag, exclude_id=problem.get("id"))
            except KeyboardInterrupt:
                # Keep a trace of the in-flight problem as well as all prior rows.
                result = {
                    "answer": "", "status": "interrupted", "seconds": round(time.perf_counter() - started, 2),
                    "prompt_tokens": None, "completion_tokens": None, "retrieved": "",
                }
                partial = detail_row(problem, result, eval_id, evaluated_at)
                sync_csv_row(writer, out, partial)
                written_rows += 1
                manifest["interrupted_problem_id"] = problem.get("id", "")
                update_progress(manifest, rows, written_rows)
                save_manifest(manifest_path, manifest)
                interrupted = True
                print(f"Interrupted while solving {problem.get('id', '')}; partial results saved.", flush=True)
                break

            row = detail_row(problem, result, eval_id, evaluated_at)
            sync_csv_row(writer, out, row)
            rows.append(row)
            written_rows += 1
            manifest["last_problem_id"] = problem.get("id", "")
            update_progress(manifest, rows, written_rows)
            save_manifest(manifest_path, manifest)

            running = 100 * sum(bool(r["is_correct"]) for r in rows) / len(rows)
            input_tokens = row["input_tokens"]
            output_tokens = row["output_tokens"]
            print(f"[{n}/{len(problems)}] {problem.get('id', '')} {problem.get('language', '')}: "
                  f"{result.get('answer', '')} (truth {problem.get('label', '')}) "
                  f"{result.get('seconds', 0):.1f} s, tokens in/out "
                  f"{input_tokens if input_tokens is not None else 'n/a'}/"
                  f"{output_tokens if output_tokens is not None else 'n/a'}, "
                  f"running accuracy {running:.1f}%", flush=True)

    status = "interrupted" if interrupted else "completed"
    manifest["status"] = status
    manifest["finished_at"] = datetime.now().astimezone().isoformat(timespec="microseconds")
    update_progress(manifest, rows, written_rows)
    save_manifest(manifest_path, manifest)

    if rows:
        hits = [bool(row["is_correct"]) for row in rows]
        seconds = [row["elapsed_seconds"] for row in rows]
        floor = sum(row["correct_answer"] == CONFIG["FALLBACK_ANSWER"] for row in rows)
        over = sum(value > TIME_LIMIT for value in seconds)
        accuracy = 100 * sum(hits) / len(hits)

        print("\n" + "=" * 60)
        print(f"Accuracy: {sum(hits)}/{len(hits)} = {accuracy:.1f}%")
        print(f"  floor, always {CONFIG['FALLBACK_ANSWER']}: {100 * floor / len(rows):.1f}%   "
              "random guess: 20.0%")
        print(f"Latency: mean {statistics.mean(seconds):.1f} s, median {statistics.median(seconds):.1f} s, "
              f"worst {max(seconds):.1f} s, over {TIME_LIMIT:.0f} s: {over}")
        print(f"Status: {dict(Counter(row['status'] for row in rows))}")
        by_id = defaultdict(list)
        for row in rows:
            by_id[row["problem_id"]].append(bool(row["is_correct"]))
        macro = statistics.mean(statistics.mean(values) for values in by_id.values())
        print(f"Mean accuracy per problem ID: {100 * macro:.1f}% ({len(by_id)} IDs)")
        estimates = sorted(row["arduino_estimated_seconds"] for row in rows
                           if row["arduino_estimated_seconds"] is not None)
        if estimates:
            p95 = estimates[int(0.95 * (len(estimates) - 1))]
            print(f"UNO Q estimate: mean {statistics.mean(estimates):.1f} s, p95 {p95:.1f} s, "
                  f"worst {estimates[-1]:.1f} s ({len(estimates)}/{len(rows)} rows with timing)")

        for field, title in (("language", "By language"), ("level", "By level"),
                             ("has_options", "By whether OCR found the options")):
            groups = defaultdict(list)
            for row in rows:
                groups[row[field]].append(bool(row["is_correct"]))
            table(title, groups)

        predicted = Counter(row["selected_answer"] for row in rows)
        truth = Counter(row["correct_answer"] for row in rows)
        print("\nLetters   " + "  ".join(f"{letter}:{predicted[letter]:>3}/{truth[letter]:<3}"
                                         for letter in "ABCDE"))
        print("          (predicted/true; a model that loves one letter shows up here)")
    else:
        print("\nNo problems completed; the run metadata and partial CSV are saved.")

    append_ablation(ablation_path, stamp, name, manifest["git_commit"], args, rows, status)
    print(f"\nDetails: {detail_path}\nRun manifest: {manifest_path}\nAblation log: {ablation_path}")


if __name__ == "__main__":
    main()
