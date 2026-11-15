"""Validate the review batch JSONs produced so far (ids, order, vocabulary).

Accepts either a single batch_XX.json or chunk files batch_XXa/b/c.json
(concatenated in name order) and compares the concatenated id list with the
headers of batch_XX.txt.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VALID = {"arithmetic", "fractions", "counting", "numbers", "geometry_measure",
         "geometry_spatial", "logic", "measures", "patterns", "algebra"}
done = 0
for k in range(1, 8):
    root = ROOT / "analysis" / "review"
    files = sorted(root.glob(f"batch_{k:02d}.json")) or sorted(root.glob(f"batch_{k:02d}[a-z].json"))
    if not files:
        print(f"batch_{k:02d}: pending")
        continue
    items = [obj for f in files for obj in json.loads(f.read_text(encoding="utf-8"))]
    exp = [l[2:].split()[0] for l in (root / f"batch_{k:02d}.txt").read_text(encoding="utf-8").splitlines() if l.startswith("# ")]
    got = [i["id"] for i in items]
    topics = set(i["topic"] for i in items)
    ok = got == exp and len(items) == len(exp) and topics <= VALID
    done += 1
    src = "+".join(f.name for f in files)
    print(f"batch_{k:02d} [{src}]: {len(items)} items, ids_match={got == exp}, "
          f"bad_topics={sorted(topics - VALID)}, OK={ok}")
print(f"batches validated: {done}/7")
