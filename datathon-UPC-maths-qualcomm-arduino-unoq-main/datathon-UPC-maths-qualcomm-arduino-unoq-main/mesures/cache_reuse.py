"""How much of the fixed prefix was reused in each request of a cond_probe.py run?
Usage: python -I cache_reuse.py <run.jsonl> [...]"""
import json
import sys
from collections import Counter

for path in sys.argv[1:]:
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    asked = [r for r in rows if r["status"] != "skipped"][1:]  # the first request fills the cache
    if not asked:
        continue
    reuse = Counter(r["cache_n"] for r in asked)
    top = max(reuse)
    full = [r for r in asked if r["cache_n"] == top]
    part = [r for r in asked if r["cache_n"] != top]
    print(f"\n{path.split(chr(92))[-1]}: {len(asked)} requests after the first")
    print(f"  tokens reused -> requests: {dict(sorted(reuse.items(), reverse=True))}")
    if part:
        new_full = [r["read_n"] for r in full]
        total_part = [r["read_n"] + r["cache_n"] - top for r in part]
        print(f"  prefix fully reused in {len(full)}; not in {len(part)}")
        print(f"  new tokens of the problem when reused: mean {sum(new_full) / max(len(new_full), 1):.0f}, max {max(new_full or [0])}; "
              f"when not reused: mean {sum(total_part) / len(total_part):.0f}, min {min(total_part)}")
        order = "".join("F" if r["cache_n"] == top else "-" for r in asked)
        print(f"  in order (F = fully reused): {order}")
