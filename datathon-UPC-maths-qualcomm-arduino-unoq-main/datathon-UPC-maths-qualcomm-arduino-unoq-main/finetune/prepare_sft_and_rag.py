"""Convert verified synthetic_procedures.jsonl into:
1. data/sft_train_pot.jsonl      (Bilingual Eng+Cat pure Program-of-Thought SFT dataset)
2. data/sft_train_plan_pot.jsonl (Bilingual Eng+Cat 1-line # Plan + Python PoT SFT dataset)
3. data/sft_train_multitask.jsonl(Bilingual PoT + Socratic Hints + Step-by-Step Explanations)
4. data/train_enriched.jsonl     (Original data/train.jsonl enriched with topic, steps, hint, code for RAG)

Usage:
    python -I prepare_sft_and_rag.py
"""
import json
import os
from collections import Counter

POT_SYSTEM = (
    "You are a Python math solver. Reply ONLY with a ```python block of 2 to 6 flat lines "
    "that computes the answer and ends with print(ans). No def, no imports, no comments, no text outside the block."
)
POT_SUFFIX = "\nWrite 2 to 6 flat lines of Python ending with print(ans):"

PLAN_SYSTEM = (
    "You are a mathematical competition solver. Reply ONLY with a ```python block starting with one "
    "# Plan: comment explaining the key equation or counting rule, followed by 2 to 6 flat lines of Python "
    "that compute the answer and end with print(ans)."
)

TUTOR_HINT_SYSTEM = (
    "You are a Socratic math tutor on an Arduino UNO Q. Give ONE short hint in the student's language "
    "that guides their first step WITHOUT revealing the final answer or option letter."
)

TUTOR_STEPS_SYSTEM = (
    "You are a math competition tutor. Explain the step-by-step procedure clearly in 2 to 3 concise sentences "
    "in the student's language, ending with the correct option letter and value."
)


def format_opts(opts):
    return "\n".join(f"{k}) {opts[k]}" for k in "ABCDE" if k in (opts or {}))


def first_sentence(text):
    s = text.split("->")[0].strip()
    parts = [p.strip() for p in s.split(".") if p.strip()]
    return (parts[0] + ".") if parts else s[:140]


def main():
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../data"))
    synth_path = os.path.join(data_dir, "synthetic_procedures.jsonl")
    train_path = os.path.join(data_dir, "train.jsonl")

    verified = {}
    with open(synth_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if d.get("verified") and d.get("code"):
                verified[d["id"]] = d

    print(f"Loaded {len(verified)} verified synthetic procedures from {synth_path}")
    topics = Counter(d.get("topic", "unknown") for d in verified.values())
    levels = Counter(d.get("level", -1) for d in verified.values())
    print(f"By topic:  {dict(topics)}")
    print(f"By level:  {dict(sorted(levels.items()))}")

    pot_rows = []
    plan_rows = []
    multitask_rows = []

    for pid, d in sorted(verified.items()):
        opts_block = format_opts(d.get("options") or {})
        code = d["code"].strip()

        for lang, text_key, steps_key, hint_key in (
            ("eng", "text_en", "steps_en", "hint_en"),
            ("cat", "text_ca", "steps_ca", "hint_ca"),
        ):
            stmt = (d.get(text_key) or "").strip()
            if not stmt:
                continue
            steps = (d.get(steps_key) or "").strip()
            hint = (d.get(hint_key) or "").strip()
            prob_with_opts = f"{stmt}\n{opts_block}" if opts_block else stmt

            # 1. Pure PoT SFT row (exact match with inference prompt)
            pot_item = {
                "id": f"{pid}|{lang}|pot",
                "level": d["level"],
                "topic": d.get("topic", "arithmetic"),
                "messages": [
                    {"role": "system", "content": POT_SYSTEM},
                    {"role": "user", "content": stmt + POT_SUFFIX},
                    {"role": "assistant", "content": f"```python\n{code}\n```"},
                ],
            }
            pot_rows.append(pot_item)
            multitask_rows.append(pot_item)

            # 2. Plan + PoT SFT row
            plan_line = first_sentence(steps)
            plan_item = {
                "id": f"{pid}|{lang}|plan_pot",
                "level": d["level"],
                "topic": d.get("topic", "arithmetic"),
                "messages": [
                    {"role": "system", "content": PLAN_SYSTEM},
                    {"role": "user", "content": prob_with_opts + POT_SUFFIX},
                    {"role": "assistant", "content": f"```python\n# Plan: {plan_line}\n{code}\n```"},
                ],
            }
            plan_rows.append(plan_item)

            # 3. Socratic hint & step-by-step rows for multitask tutor
            if hint:
                multitask_rows.append({
                    "id": f"{pid}|{lang}|hint",
                    "level": d["level"],
                    "topic": d.get("topic", "arithmetic"),
                    "messages": [
                        {"role": "system", "content": TUTOR_HINT_SYSTEM},
                        {"role": "user", "content": prob_with_opts},
                        {"role": "assistant", "content": hint},
                    ],
                })
            if steps:
                multitask_rows.append({
                    "id": f"{pid}|{lang}|steps",
                    "level": d["level"],
                    "topic": d.get("topic", "arithmetic"),
                    "messages": [
                        {"role": "system", "content": TUTOR_STEPS_SYSTEM},
                        {"role": "user", "content": prob_with_opts},
                        {"role": "assistant", "content": steps},
                    ],
                })

    out_pot = os.path.join(data_dir, "sft_train_pot.jsonl")
    out_plan = os.path.join(data_dir, "sft_train_plan_pot.jsonl")
    out_multi = os.path.join(data_dir, "sft_train_multitask.jsonl")

    for path, rows in ((out_pot, pot_rows), (out_plan, plan_rows), (out_multi, multitask_rows)):
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(rows):4d} rows -> {path}")

    # 4. Enrich train.jsonl for RAG
    out_enriched = os.path.join(data_dir, "train_enriched.jsonl")
    enriched_count = 0
    total_train = 0
    with open(train_path, encoding="utf-8") as fin, open(out_enriched, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            total_train += 1
            row = json.loads(line)
            d = verified.get(row["id"])
            if d:
                is_cat = row["language"] == "cat"
                row["topic"] = d.get("topic", "arithmetic")
                row["steps"] = d.get("steps_ca" if is_cat else "steps_en", "")
                row["hint"] = d.get("hint_ca" if is_cat else "hint_en", "")
                row["code"] = d.get("code", "")
                enriched_count += 1
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {total_train} rows ({enriched_count} enriched with verified procedures) -> {out_enriched}")


if __name__ == "__main__":
    main()
