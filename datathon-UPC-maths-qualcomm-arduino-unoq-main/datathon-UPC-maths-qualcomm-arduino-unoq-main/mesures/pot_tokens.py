"""How long are the programs the model wrote, and what would a lower max_tokens have cost?
Usage: python -I pot_tokens.py <pot-rerun.jsonl> [...]"""
import json
import sys

for path in sys.argv[1:]:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    asked = [r for r in rows if r["status"] != "skipped"]
    matched = [r for r in asked if len(r.get("matches") or []) == 1]
    right = [r for r in matched if r["matches"][0] == r["label"]]
    n = sorted(r.get("predicted_n") or 0 for r in asked)
    print(f"\n{path}")
    print(f"  asked {len(asked)}: generated tokens median {n[len(n) // 2]}, "
          f"over 80: {sum(x > 80 for x in n)}, over 120: {sum(x > 120 for x in n)}, cut at 160: "
          f"{sum(1 for r in asked if r.get('finish') == 'length')}")
    rn = sorted(r.get("predicted_n") or 0 for r in right)
    print(f"  right answers ({len(right)}): tokens {rn}")
    print(f"  right answers that fit in 80 tokens: {sum(x <= 80 for x in rn)}; in 120: {sum(x <= 120 for x in rn)}; in 160: {len(rn)}")
    wn = sorted(r.get("predicted_n") or 0 for r in matched if r not in right)
    print(f"  wrong matches ({len(wn)}): tokens {wn}")
    for cap in (80, 120, 160):
        est = [(r.get("read_n", r.get("prompt_n", 0))) / 11 + min(r.get("predicted_n") or 0, cap) / 5.5 for r in asked]
        print(f"  cap {cap}: estimated board time mean {sum(est) / len(est):.0f} s, max {max(est):.0f} s")
