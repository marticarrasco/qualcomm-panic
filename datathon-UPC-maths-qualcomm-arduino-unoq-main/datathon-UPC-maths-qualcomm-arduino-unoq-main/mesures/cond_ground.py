"""Does the code use the numbers of the statement? A check that needs no answer key.
Usage: python -I cond_ground.py <data.jsonl> <run.jsonl> [<run.jsonl> ...]"""
import json
import re
import sys

TRIVIAL = {0, 1, 2, 10, 100, 1000}


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def statement_numbers(text):
    text = re.sub(r"^\s*\d+\.\s*", "", text)  # the problem number
    return {int(x) for x in re.findall(r"(?<![\w.])\d+(?![\w.]*\d)", text)}


def code_numbers(code):
    lines = [l for l in code.splitlines() if not l.strip().startswith("#")]
    return {int(x) for x in re.findall(r"(?<![\w.])\d+", "\n".join(lines))}


def repeated_pieces(code):
    pieces = [l.strip() for l in code.splitlines() if l.strip().startswith("# piece:")]
    return len(pieces) != len(set(pieces)) or any(a != b and b.startswith(a[9:], 9) and len(b) > 2 * len(a)
                                                  for a in pieces for b in pieces)


data = {(r["id"], r["language"]): r for r in load(sys.argv[1])}
for path in sys.argv[2:]:
    rows = [r for r in load(path) if r["status"] != "skipped"]
    print(f"\n{path.split(chr(92))[-1]}: {len(rows)} answers")
    groups = {"uses every number of the statement": [], "misses some number": []}
    alien = 0
    for r in rows:
        text = data[(r["id"], r["language"])]["text"]
        want = statement_numbers(text) - TRIVIAL
        have = code_numbers(r.get("code") or "")
        key = "uses every number of the statement" if want <= have else "misses some number"
        groups[key].append(r)
        alien += bool(have - statement_numbers(text) - TRIVIAL - {x + 1 for x in statement_numbers(text)})
    for name, group in groups.items():
        matched = [r for r in group if len(r.get("matches") or []) == 1]
        right = [r for r in matched if r["matches"][0] == r["label"]]
        print(f"  {name:36} {len(group):3} answers, {len(matched):2} matched, {len(right):2} right")
    print(f"  code with a number that is not in the statement: {alien}")
    if any("# piece:" in (r.get("code") or "") for r in rows):
        print(f"  answers that repeat a piece: {sum(repeated_pieces(r['code']) for r in rows)}")
