#!/usr/bin/env python3
"""Can the small vision model get anything useful out of the FIGURE of a problem?

    .venv/bin/python experiments/vision_figure_eval.py describe --n 80          # 3 prompt wordings on cropped figures
    .venv/bin/python experiments/vision_figure_eval.py answer                   # text / +crop / +strip / +description
    .venv/bin/python experiments/vision_figure_eval.py report

Laptop only. Needs a vision server on port 8083 (see the task notes / README) and pillow+numpy+scipy.
Everything is cached in results_local/figure_eval/*.jsonl, so a rerun only asks what is missing.
"""
import argparse
import base64
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from copilot.client import post
from copilot.llm import format_problem
from experiments.figure_crop import crop_figure, load_images, load_problems, to_jpeg

OUT = "results_local/figure_eval"
SOLVE = ("You solve multiple-choice math competition problems for school students. Write only the key calculations, "
         "one per line, with no explanations. Use at most 6 short lines. Then finish with 'Answer: X', where X is "
         "the letter of the correct option.")
TERSE = "You transcribe figures from school math problems. Reply in plain text, at most 3 short lines, no preamble, no markdown."
DESCRIBE = {
    "list": (None, "List the numbers, labels and shapes in this figure."),
    "terse": (TERSE, "Line 1: every number and label written in the figure. Line 2: the shapes and how many of each. "
                     "Line 3: how they are arranged."),
    "short": (None, "In one sentence: what does this math figure show? Include every number and label you can read."),
}
MIN_SIDE = 448   # small crops are enlarged to this, so they use the whole image-token budget


def image_part(raw):
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(raw).decode()}}


def chat(url, system, content, max_tokens):
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
    r = post(url, {"messages": messages, "temperature": 0, "seed": 42, "max_tokens": max_tokens,
                   "cache_prompt": False}, timeout=600)
    t = r.get("timings", {})
    return {"out": r["choices"][0]["message"]["content"].strip(), "prompt_n": t.get("prompt_n"),
            "predicted_n": t.get("predicted_n"), "prompt_ms": round(t.get("prompt_ms", 0)),
            "predicted_ms": round(t.get("predicted_ms", 0))}


class Cache:
    def __init__(self, name):
        os.makedirs(OUT, exist_ok=True)
        self.path = f"{OUT}/{name}.jsonl"
        self.rows = {}
        if os.path.exists(self.path):
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    self.rows[(row["id"], row["variant"])] = row

    def get(self, pid, variant, make):
        if (pid, variant) not in self.rows:
            row = {"id": pid, "variant": variant, **make()}
            self.rows[(pid, variant)] = row
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return self.rows[(pid, variant)]


def answer_set(n_train):
    """English image-dependent problems that have OCR'd options: all of val, then the first n_train of train."""
    val = [x for x in load_problems(("data/val.jsonl",)) if x["options"]]
    train = [x for x in load_problems(("data/train.jsonl",)) if x["options"]][:n_train]
    return val + train


def figure_bytes(raw):
    crop, box = crop_figure(raw, min_side=MIN_SIDE)
    return (to_jpeg(crop), True) if crop is not None else (raw, False)   # no figure found: send the strip


def parse_letter(text):
    found = re.findall(r"Answer:\s*\**\(?([A-E])\b", text)
    return found[-1] if found else ""


def cmd_describe(args, url):
    problems = load_problems(("data/val.jsonl",))[:args.n]
    if args.answer_set:
        seen = {x["id"] for x in problems}
        problems += [x for x in answer_set(args.train) if x["id"] not in seen]
    images = load_images({(x["id"], x["language"]) for x in problems})
    cache = Cache("describe" + args.tag)
    for name in args.prompts.split(","):
        system, prompt = DESCRIBE[name]
        base = chat(url, system, prompt, 1)["prompt_n"]          # text tokens of the prompt alone
        for n, problem in enumerate(problems, 1):
            raw, found = figure_bytes(images[(problem["id"], problem["language"])])
            row = cache.get(problem["id"], name, lambda: {**chat(url, system, [image_part(raw), {"type": "text", "text": prompt}],
                                                                 args.max_tokens), "text_tokens": base, "cropped": found})
            print(f"\r{name} {n}/{len(problems)}", end="", file=sys.stderr, flush=True)
        rows = [cache.rows[(x["id"], name)] for x in problems]
        image_tokens = sorted(r["prompt_n"] - r["text_tokens"] for r in rows)
        print(f"\n{name}: image tokens median {image_tokens[len(rows) // 2]} min {image_tokens[0]} max {image_tokens[-1]}; "
              f"output tokens mean {sum(r['predicted_n'] for r in rows) / len(rows):.0f}; "
              f"hit the {args.max_tokens}-token cap {sum(r['predicted_n'] >= args.max_tokens for r in rows)}/{len(rows)}")


def cmd_answer(args, url):
    problems = answer_set(args.train)
    images = load_images({(x["id"], x["language"]) for x in problems})
    cache = Cache("answer" + args.tag)
    descriptions = Cache("describe" + args.tag).rows
    for n, problem in enumerate(problems, 1):
        raw = images[(problem["id"], problem["language"])]
        text = format_problem(problem)
        crop, found = figure_bytes(raw)
        for variant in args.variants.split(","):
            if variant == "text":
                content = text
            elif variant == "crop":
                content = [image_part(crop), {"type": "text", "text": text}]
            elif variant == "strip":
                content = [image_part(raw), {"type": "text", "text": text}]
            else:   # desc:<prompt name> = two calls: describe the crop, then solve from text only
                d = descriptions.get((problem["id"], variant.split(":")[1]))
                if d is None:
                    continue
                content = text + "\n\nWhat the figure shows (automatic description, may be wrong):\n" + d["out"]
            row = cache.get(problem["id"], variant, lambda: {**chat(url, SOLVE, content, 200), "label": problem["label"]})
        print(f"\r{n}/{len(problems)}", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    cmd_report(args, url)


def cmd_report(args, url):
    problems = answer_set(args.train)
    rows = Cache("answer" + args.tag).rows
    variants = sorted({v for _, v in rows})
    done = [x for x in problems if all((x["id"], v) in rows for v in variants)]
    n = len(done)
    truth = Counter(x["label"] for x in done)
    top, count = truth.most_common(1)[0]
    print(f"{n} image-dependent English problems with OCR'd options "
          f"({sum(x['id'] in {p['id'] for p in load_problems()} for x in done)} val + rest train)")
    print("true letters " + " ".join(f"{l}:{truth[l]}" for l in "ABCDE"))
    print(f"{'always ' + top:22} {count}/{n} = {100 * count / n:.1f}%")
    for v in variants:
        got = [(parse_letter(rows[(x["id"], v)]["out"]), x["label"], rows[(x["id"], v)]) for x in done]
        hits = sum(a == b for a, b, _ in got)
        blank = sum(a == "" for a, _, _ in got)
        letters = Counter(a for a, _, _ in got)
        read = sorted(r["prompt_n"] for _, _, r in got)[n // 2]
        wrote = sum(r["predicted_n"] for _, _, r in got) / n
        print(f"{v:22} {hits}/{n} = {100 * hits / n:.1f}%  unparseable {blank}  of parsed {hits}/{n - blank}"
              f"  letters " + " ".join(f"{l}:{letters[l]}" for l in "ABCDE") + f"  read~{read} wrote~{wrote:.0f} tok")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["describe", "answer", "report"])
    p.add_argument("--port", type=int, default=8083)
    p.add_argument("--n", type=int, default=80, help="describe: first N image-dependent val problems")
    p.add_argument("--answer-set", action="store_true", help="describe: also the problems used by 'answer'")
    p.add_argument("--train", type=int, default=30, help="answer: add this many train problems to the 60 val ones")
    p.add_argument("--prompts", default="list,terse,short")
    p.add_argument("--variants", default="text,crop,strip,desc:short")
    p.add_argument("--max-tokens", type=int, default=60)
    p.add_argument("--tag", default="", help="suffix for the cache files, e.g. -64 for another --image-max-tokens")
    args = p.parse_args()
    url = f"http://127.0.0.1:{args.port}/v1/chat/completions"
    {"describe": cmd_describe, "answer": cmd_answer, "report": cmd_report}[args.command](args, url)


if __name__ == "__main__":
    main()
