"""Check teacher-written programs against the M3Kang labels, and keep the ones that are right.

A program is kept only if, run in the copilot's own sandbox (copilot/pot.py), it prints a
number that equals exactly one option and that option is the labelled answer. Each kept
program is paired with the problem text in every language of data/train.jsonl.

Usage, on the laptop, from the repository folder:
  python finetune/build_pairs.py --programs data/pot_programs.jsonl --out data/pot_pairs.jsonl

The programs file has one {"id": ..., "code": ...} per line. A later line for the same id
replaces an earlier one. Both files hold M3Kang content: don't publish them.
"""
import argparse
import json
import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from copilot.pot import numeric_options, run_and_match  # noqa: E402


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/train.jsonl")
    p.add_argument("--programs", default="data/pot_programs.jsonl")
    p.add_argument("--out", default="data/pot_pairs.jsonl")
    p.add_argument("--failed", default="data/pot_failed.jsonl", help="ids to try again, without the label")
    p.add_argument("--only", default="", help="check only the ids in this programs file, to save time")
    args = p.parse_args()

    rows = load(args.data)
    by_id = {}
    for r in rows:
        by_id.setdefault(r["id"], []).append(r)
    programs = {}
    for r in load(args.programs):
        programs[r["id"]] = r["code"].strip()
    only = {r["id"] for r in load(args.only)} if args.only else None

    kept, failed, reasons = [], [], Counter()
    for pid, code in programs.items():
        versions = by_id.get(pid)
        if not versions:
            reasons["unknown id"] += 1
            continue
        ref = next((r for r in versions if r["language"] == "eng"), versions[0])
        if numeric_options(ref) is None:
            reasons["options are not five numbers"] += 1
            continue
        if only is not None and pid not in only:
            continue
        letters, printed, status = run_and_match(code, ref, timeout=10)
        if status != "ok":
            reason = "error: " + status
        elif printed is None:
            reason = "printed nothing"
        elif not letters:
            reason = "matches no option"
        elif len(letters) > 1:
            reason = "matches several options"
        elif letters[0] != ref["label"]:
            reason = "matches another option"
        else:
            reason = "kept"
        reasons[reason.split(":")[0]] += 1
        if reason == "kept":
            kept.append(pid)
        else:
            failed.append({"id": pid, "reason": reason, "printed": printed})

    if only is None:
        n_rows = 0
        with open(args.out, "w", encoding="utf-8") as out:
            for pid in kept:
                for r in by_id[pid]:
                    if numeric_options(r) is None or not r.get("text_only"):
                        continue
                    out.write(json.dumps({"id": pid, "language": r["language"], "level": r["level"],
                                          "text": r["text"].strip(), "code": programs[pid]}, ensure_ascii=False) + "\n")
                    n_rows += 1
        with open(args.failed, "w", encoding="utf-8") as out:
            for r in failed:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"programs {len(programs)}; kept {len(kept)} problems, {n_rows} rows -> {args.out}")
    else:
        print(f"checked {len(kept) + len(failed)} programs; right {len(kept)}")
    print("reasons:", dict(reasons))
    for r in failed:
        print(f"  FAILED {r['id']}: {r['reason']} (printed {r['printed']!r})")


if __name__ == "__main__":
    main()
