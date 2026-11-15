"""Show what the model wrote in a cond_probe.py run. Usage: python -I cond_peek.py <cond.jsonl> [N]"""
import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
asked = [r for r in rows if r["status"] != "skipped"]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 12
pieces = [r["code"].count("# piece:") for r in asked]
print(f"pieces per answer: mean {sum(pieces) / len(pieces):.1f}; answers with a print line: "
      f"{sum('print(' in r['code'] for r in asked)}/{len(asked)}")
for r in asked[:n]:
    verdict = "no match"
    if len(r["matches"]) == 1:
        verdict = "RIGHT" if r["matches"][0] == r["label"] else f"wrong ({r['matches'][0]}, truth {r['label']})"
    print(f"\n[{r['id']} {r['language']}] {r['status']} | printed {r['printed'][:60]!r} | {r['how']} | {verdict} | {r['predicted_n']} tokens")
    print("    " + r["code"][:520].replace("\n", "\n    "))
