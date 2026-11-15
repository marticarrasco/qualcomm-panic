"""New low-level (0-3) problems with a verified program, written by a teacher model, to add to the training pairs.

Nothing from M3Kang or the practice set is sent: the teacher only gets a topic, a level and a style description.
A problem is kept only if
  1. its program runs in the copilot's sandbox and prints exactly the stated answer, which is exactly one option,
  2. a second, separate call that sees only the Catalan statement and the options picks the same option,
  3. the program computes something (no bare print of the answer), and
  4. the statement is not close to a practice-set, val or already kept statement.

Usage, from the repository root, with the key in the environment (never in a file of the repo):
    $env:HELMCODE_API_KEY = "..."
    <venv python> -X utf8 finetune/gen_synth_lowlevel.py --calls 4          # a small trial
    <venv python> -X utf8 finetune/gen_synth_lowlevel.py                    # all topics, resumes
    <venv python> -X utf8 finetune/gen_synth_lowlevel.py --pack             # write the Kaggle zip with v3 + these
"""
import argparse
import ast
import concurrent.futures
import json
import os
import random
import re
import sys
import threading
import time
import urllib.request
import zipfile
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))  # the repository root
sys.path.insert(0, ROOT)
from copilot import pot  # noqa: E402

API_URL = "https://api.helmcode.com/v1/chat/completions"
RAW = os.path.join(ROOT, "data", "synth-raw.jsonl")          # every candidate, with why it was kept or dropped
KEPT = os.path.join(ROOT, "data", "synth-lowlevel.jsonl")    # the kept ones, in the format of pot_pairs.jsonl
ZIP = os.path.join(ROOT, "data", "kaggle_pot_dataset_v5.zip")
PER_CALL = 6
PER_JOB = 2   # of the problems kept from one call, how many go to the training file

TOPICS = [
    "jumps forwards and backwards on a number line, or steps up and down a staircase",
    "a snail, frog or climber that advances and slides back until it first reaches a height",
    "positions in a queue or a race: people ahead, behind and between",
    "ages of siblings now and some years ago or from now",
    "coins and notes: paying an amount, change, number of coins",
    "buying several items with a budget, how many can be bought or how much is left",
    "a digital clock: counting times with a digit property, or minutes between two times",
    "days of the week and dates: what day it is after some days, how many Sundays",
    "legs and heads of animals in a farm (two unknowns)",
    "sharing sweets or stickers equally, with a remainder or after giving some away",
    "counting numbers in a range with a divisibility property",
    "counting two-digit or three-digit numbers with a property of their digits",
    "sum or product of the digits of a number, or the largest/smallest number with given digits",
    "a number pattern or sequence: find the term in a given position",
    "repeating pattern of coloured beads or letters: which or how many in the first n",
    "perimeter and area of rectangles and squares described in words",
    "a rectangle cut into equal squares or tiles, how many tiles",
    "cubes: faces, edges, vertices, small cubes in a block, painted faces",
    "fence posts, trees along a road, cuts needed to make pieces",
    "pages of a book: digits used to number them, or pages read per day",
    "mystery number: I think of a number, do operations, get a result",
    "two numbers given their sum and difference, or sum and product",
    "balance scales described in words: weights of objects",
    "even and odd numbers, consecutive numbers with a given sum",
    "multiples and common multiples: two lights or bells that coincide",
    "fractions of a quantity: half, a third, a quarter of a collection",
    "time, speed and distance with whole numbers, trains or cyclists meeting",
    "handshakes, matches in a tournament, pairs chosen from a group",
    "arrangements: ways to order a few objects with a restriction",
    "choosing outfits or menus: counting combinations",
    "a calculation with missing digits or symbols standing for digits",
    "the last digit of a product or a power, remainders",
    "filling and emptying: litres in tanks, bottles, taps",
    "measuring: lengths in cm and m, weights in g and kg, converting units",
    "logic with counts: some children like A, some like B, some both",
    "a grid or a chessboard described in words: counting cells, rows and columns",
    "dice and dominoes: sums of faces, opposite faces add up to 7",
    "money saved every week, when a target is reached, who has more",
    "lift or building floors, moving up and down",
    "the largest or smallest result of an expression using given numbers once",
    "prime numbers and divisors of small numbers",
    "percentages and simple ratios with whole-number results",
]

GEN_SYSTEM = "You write original math competition problems for school students and solve each with a short Python program."

GEN_PROMPT = """Write {n} new, different multiple-choice math problems in the style of the Math Kangaroo competition.

Topic: {topic}.
Difficulty: {level_text}
Each problem must have a single whole-number or decimal answer, be fully stated in words (no picture, no table),
take 2 to 4 sentences, and use a different setting from the others (animals, shops, school, sports, games, travel...).
Vary the numbers and what is asked; avoid textbook one-step questions.

For each problem give:
- "text_en": the statement in English.
- "text_ca": the same statement in natural Catalan.
- "code": a short flat Python program that computes the answer from the data of the statement and prints it.
  Rules for the code: no imports, no def, no comments, no input; 3 to 10 lines; named intermediate variables or an
  explicit search loop (for x in range(...): if ...: print(x); break); the last thing it does is print one number.
  Available without importing: math, permutations, combinations, product, gcd, Fraction.
- "answer": the number the program prints, as a string.
- "options": a list of five different numbers as strings, in increasing order, one of them the answer, the others
  plausible mistakes. Plain numbers only, no units.

Two examples of the code style:
start = 0
for jump in range(1, 11):
    if jump % 2 == 1:
        start += 3
    else:
        start -= 1
print(start)

ahead = 6
behind = 2 * ahead
total = ahead + 1 + behind
print(total)

Reply with a JSON array of {n} objects with exactly those five keys and nothing else."""

LEVELS = {
    0: "level 0, for 8-9 year olds: small numbers, one or two steps, careful reading needed.",
    1: "level 1, for 9-10 year olds: small numbers, two steps, careful reading needed.",
    2: "level 2, for 10-11 year olds: two or three steps, some need trying out cases.",
    3: "level 3, for 11-12 year olds: two or three steps, some need trying out cases or a small search.",
}

SOLVE_SYSTEM = "You solve multiple-choice math problems carefully."
SOLVE_PROMPT = """Solve each problem and pick the correct option. If a problem is ambiguous, has no correct option or
has more than one, answer "X" for it.

{problems}

Reply with a JSON array of {n} strings, one per problem in order, each "A", "B", "C", "D", "E" or "X", and nothing else."""

lock = threading.Lock()


def load(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def chat(model, system, user, temperature, key):
    body = {"model": model, "temperature": temperature, "max_tokens": 8000,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    last = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(API_URL, data=json.dumps(body).encode(), headers={
                "Content-Type": "application/json", "Authorization": "Bearer " + key, "User-Agent": "synth-gen/1.0"})
            with urllib.request.urlopen(req, timeout=240) as resp:
                return json.loads(resp.read())["choices"][0]["message"]["content"] or ""
        except Exception as error:   # noqa: BLE001
            last = error
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"API call failed three times: {last!r}"[:300])


def json_array(text):
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        return []
    try:
        data = json.loads(text[start:end + 1])
    except ValueError:
        return []
    return data if isinstance(data, list) else []


def words(text):
    return set(re.sub(r"[^a-z0-9àèéíïòóúüç]+", " ", re.sub(r"^\s*\d+\s*[.)]\s*", "", text.lower())).split())


def computes(code):
    """True when the program does more than print a literal."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    ops = sum(isinstance(n, (ast.BinOp, ast.Compare, ast.AugAssign, ast.For, ast.While)) for n in ast.walk(tree))
    return ops >= 1 and len([line for line in code.strip().splitlines() if line.strip()]) >= 2


def check(item):
    """Shape and sandbox checks. Returns (row or None, reason)."""
    try:
        text_en, text_ca = item["text_en"].strip(), item["text_ca"].strip()
        code, answer = item["code"].strip(), str(item["answer"]).strip()
        opts = [str(o).strip() for o in item["options"]]
    except (KeyError, AttributeError, TypeError):
        return None, "bad shape"
    if len(opts) != 5 or len(set(opts)) != 5 or len(text_en) < 40 or len(text_ca) < 40:
        return None, "bad shape"
    options = dict(zip("ABCDE", opts))
    row = {"text": text_en, "options": options, "text_only": True}
    nums = pot.numeric_options(row)
    if nums is None or len(set(nums.values())) != 5:
        return None, "options not five different numbers"
    target = pot.parse_num(answer)
    labels = [k for k, v in nums.items() if target is not None and abs(v - target) < 1e-9]
    if len(labels) != 1:
        return None, "answer is not one of the options"
    if "import" in code or "def " in code or "#" in code or "input(" in code:
        return None, "code breaks the style rules"
    if not computes(code):
        return None, "code only prints"
    letters, printed, status = pot.run_and_match(code, row)
    if letters != labels or printed is None or len(printed.strip().splitlines()) != 1:
        return None, f"program does not print the answer ({status})"
    return {"text_en": text_en, "text_ca": text_ca, "options": options, "label": labels[0], "code": code}, "ok"


def one_call(job, args, key, seen):
    topic_i, level, variant = job
    rng = random.Random(f"{topic_i}-{level}-{variant}")
    reply = chat(args.model, GEN_SYSTEM, GEN_PROMPT.format(n=PER_CALL, topic=TOPICS[topic_i], level_text=LEVELS[level]),
                 0.9, key)
    cands = []
    for item in json_array(reply):
        row, reason = check(item) if isinstance(item, dict) else (None, "bad shape")
        cands.append({"job": list(job), "reason": reason, "item": item, "row": row})
    good = [c for c in cands if c["row"]]
    if good:   # the independent check, on the Catalan statement
        block = "\n\n".join(f"Problem {i + 1}. {c['row']['text_ca']}\n" +
                            "\n".join(f"{k}) {v}" for k, v in c["row"]["options"].items()) for i, c in enumerate(good))
        picks = json_array(chat(args.solver, SOLVE_SYSTEM, SOLVE_PROMPT.format(problems=block, n=len(good)), 0, key))
        for i, c in enumerate(good):
            pick = str(picks[i]).strip().upper()[:1] if i < len(picks) else "?"
            if pick != c["row"]["label"]:
                c["reason"] = f"independent solver says {pick}, program says {c['row']['label']}"
    with lock:
        for c in cands:
            if c["reason"] == "ok":
                w = words(c["row"]["text_en"])
                close = max((len(w & s) / max(len(w | s), 1) for s in seen), default=0)
                if close >= 0.6:
                    c["reason"] = "too close to another statement"
                else:
                    seen.append(w)
                    c["number"] = rng.randint(1, 20)
        with open(RAW, "a", encoding="utf-8") as f:
            f.writelines(json.dumps({k: v for k, v in c.items() if k != "row"} | {"level": level, "kept": c["row"] if c["reason"] == "ok" else None},
                                    ensure_ascii=False) + "\n" for c in cands)
        f = None
    return Counter(c["reason"].split(" (")[0] for c in cands)


def write_kept():
    rows, n, per_job = [], Counter(), Counter()
    for r in load(RAW):
        k = r.get("kept")
        if not k:
            continue
        per_job[tuple(r["job"])] += 1
        if per_job[tuple(r["job"])] > PER_JOB:   # the problems of one call look alike: keep a few of each
            continue
        level = r["level"]
        n[level] += 1
        pid = f"syn-{level}_{n[level]:03d}"
        for lang, text in (("eng", k["text_en"]), ("cat", k["text_ca"])):
            rows.append({"id": pid, "language": lang, "level": level, "text": f"{r.get('number', 1)}. {text}",
                         "options": k["options"], "label": k["label"], "code": k["code"]})
    with open(KEPT, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    return rows, n


def pack():
    synth, n = write_kept()
    bad = 0
    for r in synth:   # the last word belongs to the sandbox, on the final rows
        letters, _, _ = pot.run_and_match(r["code"], r)
        bad += letters != [r["label"]]
    assert bad == 0, f"{bad} synthetic rows fail the sandbox"
    base = load(os.path.join(ROOT, "data", "pot_pairs.jsonl"))
    assert not any(r["id"].startswith("syn-") for r in base)
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("pot_pairs.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in base + synth))
        z.write(os.path.join(ROOT, "data", "demo", "train.jsonl"), "demo.jsonl")
        z.write(os.path.join(ROOT, "data", "val.jsonl"), "val.jsonl")
        z.write(os.path.join(ROOT, "copilot", "pot.py"), "pot.py")
    print(f"synthetic problems by level: {dict(sorted(n.items()))}; rows: M3Kang {len(base)} + synthetic {len(synth)}")
    print("wrote", ZIP)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--solver", default="deepseek-v4-flash")
    ap.add_argument("--calls", type=int, default=0, help="stop after this many generation calls (0: all)")
    ap.add_argument("--variants", type=int, default=1, help="calls per topic and level")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--pack", action="store_true")
    args = ap.parse_args()
    if args.pack:
        return pack()
    key = os.environ.get("HELMCODE_API_KEY", "").strip()
    assert key, "set HELMCODE_API_KEY in the environment"
    done = {tuple(r["job"]) for r in load(RAW)}
    seen = [words(r["text"]) for name in ("demo/train.jsonl", "val.jsonl")
            for r in load(os.path.join(ROOT, "data", name)) if r["language"] == "eng"]
    seen += [words(r["kept"]["text_en"]) for r in load(RAW) if r.get("kept")]
    jobs = [(t, level, v) for v in range(args.variants) for t in range(len(TOPICS)) for level in LEVELS]
    jobs = [j for j in jobs if j not in done]
    random.Random(7).shuffle(jobs)
    if args.calls:
        jobs = jobs[:args.calls]
    total, t0 = Counter(), time.time()
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(one_call, j, args, key, seen) for j in jobs]
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                total += fut.result()
            except Exception as error:   # noqa: BLE001
                total["call failed"] += 1
                print("  call failed:", repr(error)[:200], flush=True)
            if i % 10 == 0 or i == len(futures):
                print(f"calls {i}/{len(futures)}, kept {total['ok']} ({time.time() - t0:.0f} s)", flush=True)
    print("this run:", dict(total.most_common()))
    rows, n = write_kept()
    print(f"kept in total: {len(rows) // 2} problems, by level {dict(sorted(n.items()))} -> {KEPT}")


if __name__ == "__main__":
    main()
