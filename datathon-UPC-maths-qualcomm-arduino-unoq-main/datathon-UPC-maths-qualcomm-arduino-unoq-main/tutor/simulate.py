#!/usr/bin/env python3
"""Check the tutor with simulated students and safety checks. No browser needed.

    python3 tutor/simulate.py
    python3 tutor/simulate.py --steps 60 --quick

1. Threshold students: always right at or below level L, always wrong above.
   Does the level estimate settle at L, how fast, does it stay on topic?
2. Noisy students: right with a probability that falls as the level rises.
3. A student who always needs two hints: the level must not move.
4. The model-sentence filter on known bad sentences.
5. A student who walks through the whole bank over HTTP: the answer letter is
   never in a reply before the problem is solved or the answer is asked for.
Standard library only. Uses temporary student databases; your real one is untouched.
"""
import argparse
import itertools
import json
import math
import os
import random
import sys
import tempfile
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tutor.server import images_folder, make_server, parse_args as server_args  # noqa: E402
from tutor.chat import check_reply
from tutor.topics import NAMES  # noqa: E402
from tutor.tutor import LETTERS, Tutor  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank", default="data/tutor.db")
    p.add_argument("--images")
    p.add_argument("--steps", type=int, default=30, help="problems per simulated student")
    p.add_argument("--start", type=int, default=2, help="starting level of the simulated students")
    p.add_argument("--quick", action="store_true", help="walk only 150 problems over HTTP instead of the whole bank")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def wrong_letter(label):
    return next(letter for letter in LETTERS if letter != label)


def run_student(tutor, name, topic, start, steps, is_right, hints=0):
    """One simulated student. is_right(level) decides each first answer.
    Returns what happened, step by step."""
    state = tutor.start(name, "eng", topic, start)
    levels, topics, ids, times = [], [], [], []
    for _ in range(steps):
        problem = state["problem"]
        if not problem:
            break
        label = tutor.problems[problem["id"]]["label"]
        levels.append(problem["level"])
        topics.append(problem["topic"])
        ids.append(problem["id"])
        began = time.perf_counter()
        for _ in range(hints):
            context = tutor.coaching_context(name, problem["id"], "hint")
            tutor.save_coaching(context, "A model hint supplied by the simulation.")
        tutor.answer(name, label if is_right(problem["level"]) else wrong_letter(label))
        state = tutor.next(name)
        times.append((time.perf_counter() - began) * 1000)
    profile = {p["topic"]: p for p in tutor.state(name)["profile"]}
    repeats = len(ids) - len(set(ids))
    levels = [level for level, t in zip(levels, topics) if t == topic]
    return {"levels": levels, "on_topic": sum(t == topic for t in topics) / max(1, len(topics)),
            "mastered": profile.get(topic, {}).get("mastered"), "repeats": repeats,
            "ms": sum(times) / max(1, len(times)), "ms_max": max(times, default=0), "steps": len(ids)}


def settle_step(levels, target):
    """The first step from which the served level stays within one level of
    the target (a staircase keeps testing one level above what the student can do)."""
    for i in range(len(levels)):
        if all(abs(level - target) <= 1 for level in levels[i:]):
            return i + 1
    return None


def simulate(args, students_db):
    tutor = Tutor(args.bank, students_db, seed=args.seed)
    rows = []
    for topic in ("geometry_measure", "arithmetic", "counting", "numbers"):
        for target in (1, 3, 5):
            r = run_student(tutor, f"exact-{topic}-{target}", topic, args.start, args.steps, lambda lvl: lvl <= target)
            rows.append((f"exact L={target}", topic, target, r, settle_step(r["levels"], target)))
    rng = random.Random(args.seed)
    for topic in ("geometry_measure", "arithmetic"):
        for target in (2, 4):
            # 88% right one level below the target, 50% half a level above it, 12% two levels above
            r = run_student(tutor, f"noisy-{topic}-{target}", topic, args.start, args.steps,
                            lambda lvl: rng.random() < 1 / (1 + math.exp(2.0 * (lvl - target - 0.5))))
            rows.append((f"noisy L={target}", topic, target, r, None))
    r = run_student(tutor, "needs-hints", "geometry_measure", args.start, args.steps, lambda lvl: True, hints=2)
    rows.append(("2 hints, right", "geometry_measure", args.start, r, settle_step(r["levels"], args.start)))
    # Mixed mode: a profile across topics
    tutor.start("mixed", "eng", None, args.start)
    mixed_topics = []
    for _ in range(args.steps):
        problem = tutor.state("mixed")["problem"]
        mixed_topics.append(problem["topic"])
        tutor.answer("mixed", tutor.problems[problem["id"]]["label"])
        tutor.next("mixed")

    print(f"\nSimulated students ({args.steps} problems each, starting level {args.start})")
    print(f"  {'student':<16}{'topic':<18}{'target':>6}{'estimate':>9}{'last-10 mean':>13}{'settled at':>11}"
          f"{'on topic':>9}{'repeats':>8}{'ms/step':>8}{'max ms':>7}")
    ok = True
    for label, topic, target, r, settled in rows:
        tail = r["levels"][-10:]
        estimate = "-" if r["mastered"] is None else r["mastered"]
        print(f"  {label:<16}{topic:<18}{target:>6}{estimate:>9}{sum(tail) / len(tail):>13.1f}"
              f"{('step ' + str(settled)) if settled else '-':>11}{100 * r['on_topic']:>8.0f}%{r['repeats']:>8}"
              f"{r['ms']:>8.2f}{r['ms_max']:>7.1f}")
        if label.startswith("exact"):
            ok &= r["mastered"] == target and settled is not None
        ok &= r["repeats"] == 0 and r["ms"] < 50
        if topic in ("geometry_measure", "arithmetic"):  # the two topics big enough never to run dry
            ok &= r["on_topic"] == 1.0
    print(f"  mixed mode: {len(set(mixed_topics))} topics in {args.steps} problems, "
          f"blocks of {mixed_topics[:9]} ...")
    print("  estimate = highest level solved on the first try without hints and not failed since")
    print("  last-10 mean and settled at: levels of the problems served in the topic; on topic < 100% means the")
    print("  topic had no unseen problem within one level, so a problem of another topic was served instead")
    print(f"  RESULT: {'PASS' if ok else 'FAIL'} (exact students end on their level and settle; no repeats; "
          f"the two large topics never run dry; under 50 ms)")
    return tutor, ok


def check_filter():
    context = {"kind": "orient", "exercise": "Anna has 12 apples and gives 5 to Pere.",
               "options": {"C": "7"}, "correct": "C", "language": "eng", "earlier": []}
    bad = ["The answer is seven apples for Anna now.", "Subtract and you will see it is option C for sure.",
           "You should pick B) because it is the largest.", "Take 5 away from 12 to get 7 apples left.",
           "Just compute 12 - 5 = and write it down.", "Think about whether seven is a sensible size for what is left.",
           "Draw 18 apples and split them into two groups.", "Check 12 - 5"]
    good = ["Write down how many apples Anna starts with and how many she gives away.",
            "Start from the 12 apples and think about what giving 5 away does."]
    assert all(check_reply(s, context, "", "") is None for s in bad)
    assert all(check_reply(s, context, "", "") for s in good)
    print(f"Model filter: {len(bad)} unsafe replies rejected, {len(good)} coaching replies kept: PASS")


def call(base, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read()
        return raw, response.headers.get("Content-Type")


def has_key(value, key):
    if isinstance(value, dict):
        return key in value or any(has_key(v, key) for v in value.values())
    if isinstance(value, list):
        return any(has_key(v, key) for v in value)
    return False


def check_api(args, students_db):
    """Walk through the bank over HTTP and look at every reply for the answer letter."""
    server, tutor = make_server(server_args([
        "--bank", args.bank, "--students", students_db, "--images", args.images,
        "--no-llm", "--port", "0", "--quiet"]))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    name = "walker"
    replies = leaks = problems = images = 0
    seen = set()

    def step(path, body, answered):
        nonlocal replies, leaks
        raw, _kind = call(base, path, body)
        data = json.loads(raw)
        replies += 1
        if not answered and (has_key(data, "label") or b'"label"' in raw):
            leaks += 1
        return data

    state = step("/api/start", {"name": name, "language": "cat", "level": 0}, False)
    limit = 150 if args.quick else len(tutor.problems)
    while state.get("problem") and not state.get("finished") and problems < limit:
        pid = state["problem"]["id"]
        assert pid not in seen, f"{pid} served twice"
        seen.add(pid)
        truth = tutor.problems[pid]["label"]
        problems += 1
        mode = problems % 4
        step("/api/state?name=" + name, None, False)
        if mode in (0, 1):
            data = step("/api/hint", {"name": name, "problem": pid}, False)
            assert data == {"model_error": "disabled"}
        if mode == 1:
            data = step("/api/answer", {"name": name, "problem": pid, "choice": wrong_letter(truth)}, False)
            assert not data["correct"]
        if mode == 2:  # three wrong answers in a row: closes without the answer
            for letter in [l for l in LETTERS if l != truth][:3]:
                data = step("/api/answer", {"name": name, "choice": letter}, False)
            assert data["closed"] and data["problem"]["status"] == "failed"
        if mode in (1, 2):  # ask for the answer
            data = step("/api/reveal", {"name": name}, True)
            assert data["label"] == truth and data["problem"]["label"] == truth
        elif mode == 3 or mode == 0:
            data = step("/api/answer", {"name": name, "choice": truth}, True)
            assert data["correct"] and data["problem"]["label"] == truth
        if problems % 60 == 1:
            for lang in ("eng", "cat"):
                raw, kind = call(base, f"/image/{pid}_{lang}.png")
                assert kind.startswith("image/") and raw, (pid, kind)
                images += 1
        state = step("/api/next", {"name": name}, True)
        if not state.get("finished") and has_key(state, "label"):
            leaks += 1
    raw, _ = call(base, "/api/matrix")
    led = json.loads(raw)
    assert set(led) >= {"event", "level", "seq", "time"} and led["seq"] > problems, led
    try:
        call(base, "/image/..%2F..%2Ftrain.jsonl")
        raise AssertionError("path outside the image folder was served")
    except urllib.error.HTTPError as error:
        assert error.code == 404
    server.shutdown()
    server.server_close()
    missing = sum(not os.path.exists(os.path.join(args.images, f"{pid}_{lang}.png"))
                  for pid in tutor.problems for lang in ("eng", "cat"))
    expected = min(limit, len(tutor.problems))
    print(f"\nAPI walk over HTTP: {problems}/{expected} different problems, {replies} replies, "
          f"{leaks} leaked the answer letter: {'PASS' if leaks == 0 else 'FAIL'}")
    print(f"Images: {images} fetched and served as images; {missing} of {2 * len(tutor.problems)} files missing on disk")
    print(f"LED hook: /api/matrix -> {led}")
    tutor.db.close()
    return leaks == 0 and problems == expected and missing == 0


def main():
    args = parse_args()
    os.chdir(ROOT)
    args.images = images_folder(args.images)
    with tempfile.TemporaryDirectory() as folder:
        tutor, ok = simulate(args, os.path.join(folder, "students.db"))
        tutor.db.close()
        check_filter()
        ok &= check_api(args, os.path.join(folder, "walk.db"))
    print(f"\nALL CHECKS: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
