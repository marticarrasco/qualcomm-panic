"""Write analysis/PROBLEM_CATEGORIES.md summarizing the final classification."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tutor.topics import NAMES, TOPICS  # noqa: E402


def main():
    recs = [json.loads(l) for l in open(ROOT / "data" / "train_problem_categories.jsonl", encoding="utf-8")]
    dist = Counter(r["topic"] for r in recs)
    src = Counter(r["source"] for r in recs)
    by_level = defaultdict(Counter)
    for r in recs:
        by_level[r["level"]][r["topic"]] += 1

    lines = [
        "# Problem types in `data/train.jsonl`",
        "",
        f"One entry per unique problem ({len(recs)} problems; the {sum(1 for _ in open(ROOT / 'data' / 'train.jsonl', encoding='utf-8')):,} train rows repeat them across languages).",
        "Vocabulary matches `tutor/topics.py`, so a category here is also a tutor topic.",
        "",
        "## How it was produced",
        "",
        "1. **Keyword pass** (`laptop/classify_problems.py`): each problem's English row is scored with the `tutor/topics.py` keyword rules. Problems with a high score *and* a wide margin over the runner-up are kept as-is (`source: keyword`).",
        "2. **Manual review**: every ambiguous or weakly-scored problem (~75% of the set) plus a 60-problem audit sample of the strong ones was classified by hand from the English text (`source: review` / `source: audit`). Manual verdicts override keywords.",
        "",
        f"Sources: {', '.join(f'{k} {v}' for k, v in sorted(src.items()))}.",
        "",
        "## Distribution",
        "",
        "| topic | problems | % |",
        "|---|---:|---:|",
    ]
    for t in TOPICS:
        n = dist.get(t, 0)
        lines.append(f"| {t} ({NAMES[t]['cat']}) | {n} | {100 * n / len(recs):.1f}% |")
    lines.append(f"| **total** | **{len(recs)}** | 100% |")

    lines += ["", "## By level", ""]
    header = "| level | " + " | ".join(TOPICS) + " |"
    sep = "|---" * (len(TOPICS) + 1) + "|"
    lines += [header, sep]
    for lvl in sorted(by_level):
        row = [str(by_level[lvl].get(t, 0)) for t in TOPICS]
        lines.append(f"| {lvl} | " + " | ".join(row) + " |")

    out = ROOT / "analysis" / "PROBLEM_CATEGORIES.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    for t in TOPICS:
        print(f"  {t:20s} {dist.get(t, 0)}")


if __name__ == "__main__":
    main()
