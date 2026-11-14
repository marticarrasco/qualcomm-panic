"""First-pass problem-type classification for data/train.jsonl.

One row per unique problem ID (1,466). The English row is used as the
canonical text (every train problem has one). Classification reuses
tutor/topics.py keyword rules; each problem also gets a confidence margin
(top score minus runner-up score) so ambiguous cases can be reviewed by hand.

Output: data/train_problem_categories.jsonl
  id, level, year, text_only, text, options, topic, scores, margin, needs_review
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tutor.topics import TOPICS, scores  # noqa: E402

MIN_SCORE = 4.0   # below this the keywords barely fired
MIN_MARGIN = 2.0  # below this two topics are too close to call


def canonical(rows):
    """The English row of a problem (all train problems have one)."""
    for r in rows:
        if r["language"] == "eng":
            return r
    return rows[0]


def main():
    by_id = {}
    with open(ROOT / "data" / "train.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            by_id.setdefault(r["id"], []).append(r)

    out_path = ROOT / "data" / "train_problem_categories.jsonl"
    dist = {}
    review = []
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for pid in sorted(by_id):
            r = canonical(by_id[pid])
            sc = scores(r["text"])
            ranked = sorted(TOPICS, key=lambda t: (-sc[t], TOPICS.index(t)))
            top, second = ranked[0], ranked[1]
            margin = sc[top] - sc[second]
            needs = sc[top] < MIN_SCORE or margin < MIN_MARGIN
            rec = {
                "id": pid,
                "level": r["level"],
                "year": r["year"],
                "text_only": r["text_only"],
                "text": r["text"],
                "options": r["options"],
                "topic": top,
                "scores": {t: sc[t] for t in TOPICS},
                "margin": margin,
                "needs_review": needs,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            dist[top] = dist.get(top, 0) + 1
            if needs:
                review.append(pid)

    print(f"problems: {len(by_id)}  review needed: {len(review)}")
    for t in TOPICS:
        print(f"  {t:20s} {dist.get(t, 0)}")


if __name__ == "__main__":
    main()
