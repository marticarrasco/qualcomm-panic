"""Cross-check my batch_06 verdicts against the subagent's chunk files."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent / "review"
chunks = sorted(root.glob("batch_06[a-c].json"))
items = [o for f in chunks for o in json.loads(f.read_text(encoding="utf-8"))]
exp = [l[2:].split()[0] for l in (root / "batch_06.txt").read_text(encoding="utf-8").splitlines() if l.startswith("# ")]
mine = {o["id"]: o["topic"] for o in json.loads((root / "batch_06.json").read_text(encoding="utf-8"))}

texts, cur = {}, None
for l in (root / "batch_06.txt").read_text(encoding="utf-8").splitlines():
    if l.startswith("# "):
        cur = l[2:].split()[0]
    elif cur:
        texts.setdefault(cur, " ".join(l.split()))

print("chunks:", [f.name for f in chunks], "ids_match:", [i["id"] for i in items] == exp)
agree = sum(1 for i in items if i["id"] in mine and mine[i["id"]] == i["topic"])
dis = [(i["id"], mine[i["id"]], i["topic"]) for i in items if i["id"] in mine and mine[i["id"]] != i["topic"]]
print(f"agree: {agree}/{len(items)}  disagreements: {len(dis)}")
for pid, m, s in dis:
    print(f"  {pid}: mine={m} sub={s} | {texts.get(pid, '')[:110]}")
