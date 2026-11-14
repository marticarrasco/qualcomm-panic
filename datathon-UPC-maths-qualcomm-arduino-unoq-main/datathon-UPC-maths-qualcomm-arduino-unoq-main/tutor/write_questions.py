#!/usr/bin/env python3
"""Ask the on-board language model for one guiding question per problem.

The tutor shows this question on rung 1 of the hint ladder, above the topic
tip: a question about THIS problem that points at the key idea or the first
step, without solving it. The model writes it ahead of time, so the page shows
it instantly; every question passes check_question() (no answer words, no option
letter, no correct value, no number that is not in the problem) or is not shown.

    python3 tutor/write_questions.py --sample 30          # try 30 random problems, print them
    python3 tutor/write_questions.py                      # every problem, English and Catalan
    python3 tutor/write_questions.py --languages eng      # one language only

Needs the language model on 8081 (the same Qwen3.5 0.8B as the board). The
run is resumable: problems that already have a question are skipped. Output:
data/tutor_questions.db (local only, like the bank: it is derived from M3Kang).
Standard library only.
"""
import argparse
import json
import os
import random
import sqlite3
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tutor.topics import clean  # noqa: E402
from tutor.tutor import check_question  # noqa: E402

SYSTEM = {
    "eng": ("You are a patient maths coach for school students. You see a maths competition problem. "
            "Do NOT solve it. Do NOT state any result, and do not use any number that is not in the problem. "
            "Ask the student ONE short guiding question (at most 25 words) about a first step or an in-between quantity "
            "that leads towards the answer. Do not repeat the problem's own question. Reply with the question only."),
    "cat": ("Ets un entrenador de matemàtiques pacient per a alumnes d'escola. Veus un problema d'un concurs de matemàtiques. "
            "NO el resolguis. NO diguis cap resultat, i no facis servir cap nombre que no sigui al problema. "
            "Fes a l'alumne UNA sola pregunta curta (com a màxim 25 paraules) sobre un primer pas o una quantitat intermèdia "
            "que porti cap a la resposta. No repeteixis la pregunta del problema. Respon només amb la pregunta, en català correcte."),
}
PICTURE = {
    "eng": " The problem has a picture that you cannot see: do not describe it, ask about what the question needs.",
    "cat": " El problema té un dibuix que no pots veure: no el descriguis, pregunta sobre el que demana el problema.",
}
# Made-up problems, so no dataset text is copied into the prompt
EXAMPLES = {
    "eng": [
        ("A frog is at the bottom of a 10 m well. Each day it climbs 3 m and each night it slips back 2 m. "
         "On which day does it get out?",
         "How high is the frog at the end of each day, and what is different about the day it reaches the top?"),
        ("The sum of three consecutive whole numbers is 72. What is the largest of them?",
         "If you call the middle number n, how can you write the other two numbers using n?"),
        ("[picture] How many triangles are there in the figure?",
         "Can you count the triangles one size at a time, starting with the smallest ones?"),
    ],
    "cat": [
        ("Una granota és al fons d'un pou de 10 m. Cada dia puja 3 m i cada nit rellisca 2 m. Quin dia en surt?",
         "A quina alçada és la granota al final de cada dia, i què canvia el dia que arriba a dalt?"),
        ("La suma de tres nombres enters consecutius és 72. Quin és el més gran?",
         "Si el nombre del mig es diu n, com pots escriure els altres dos nombres amb n?"),
        ("[dibuix] Quants triangles hi ha a la figura?",
         "Pots comptar els triangles d'una mida cada vegada, començant pels més petits?"),
    ],
}
# The first try is deterministic; the others vary a little, used only if the first is rejected
TRIES = [{"temperature": 0, "seed": 42}, {"temperature": 0.7, "seed": 1}, {"temperature": 0.7, "seed": 2}]
SCHEMA = """CREATE TABLE IF NOT EXISTS questions (
    id TEXT, language TEXT, status TEXT, text TEXT, tries INTEGER, seconds REAL, model TEXT, rejected TEXT,
    PRIMARY KEY (id, language));"""


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank", default="data/tutor.db")
    p.add_argument("--out", default="data/tutor_questions.db")
    p.add_argument("--llm", default="http://127.0.0.1:8081/v1/chat/completions")
    p.add_argument("--languages", nargs="+", default=["eng", "cat"], choices=["eng", "cat"])
    p.add_argument("--sample", type=int, default=0, help="only this many random problems, printed for a check by hand")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--redo", action="store_true", help="write again even if a question exists")
    return p.parse_args(argv)


def ask(url, language, text, text_only, attempt):
    messages = [{"role": "system", "content": SYSTEM[language] + ("" if text_only else PICTURE[language])}]
    for problem, question in EXAMPLES[language]:
        messages += [{"role": "user", "content": problem}, {"role": "assistant", "content": question}]
    messages.append({"role": "user", "content": text if text_only else ("[picture] " if language == "eng" else "[dibuix] ") + text})
    body = {"messages": messages, "max_tokens": 60, "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False}, **attempt}
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=180) as response:
        data = json.loads(response.read())
    return data["choices"][0]["message"].get("content") or "", data.get("model", "")


def main():
    args = parse_args()
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    bank = sqlite3.connect(f"file:{args.bank}?mode=ro", uri=True)
    problems = {pid: (bool(text_only), label) for pid, text_only, label in bank.execute("SELECT id, text_only, label FROM problems")}
    texts = {(pid, lang): text for pid, lang, text in bank.execute("SELECT id, language, text FROM texts")}
    options = {(pid, lang): json.loads(o) for pid, lang, o in bank.execute("SELECT id, language, options FROM options")}
    out = sqlite3.connect(args.out)
    out.executescript(SCHEMA)
    ids = sorted(problems)
    if args.sample:
        ids = random.Random(args.seed).sample(ids, args.sample)
    done = {(pid, lang) for pid, lang in out.execute("SELECT id, language FROM questions WHERE status = 'ok'")}
    jobs = [(pid, lang) for pid in ids for lang in args.languages if args.redo or (pid, lang) not in done]
    print(f"{len(jobs)} questions to write ({len(ids)} problems, {', '.join(args.languages)})", flush=True)
    started, counts = time.time(), {"ok": 0, "rejected": 0, "error": 0}
    for n, (pid, lang) in enumerate(jobs, 1):
        text_only, label = problems[pid]
        text = clean(texts.get((pid, lang)) or texts[(pid, "eng")])
        correct = (options.get((pid, lang)) or options.get((pid, "eng")) or {}).get(label)
        status, question, rejected, model, t0 = "rejected", None, [], "", time.time()
        for tries, attempt in enumerate(TRIES, 1):
            try:
                raw, model = ask(args.llm, lang, text, text_only, attempt)
            except (OSError, ValueError, KeyError, IndexError) as error:
                status, rejected = "error", rejected + [str(error)[:200]]
                break
            question = check_question(raw, text, correct)
            if question:
                status = "ok"
                break
            rejected.append(raw.strip()[:300])
        out.execute("INSERT OR REPLACE INTO questions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (pid, lang, status, question, tries, round(time.time() - t0, 2), model, json.dumps(rejected, ensure_ascii=False)))
        out.commit()
        counts[status] += 1
        if args.sample:
            print(f"\n[{pid} {lang} {'text' if text_only else 'picture'}] {text[:260]}")
            print(f"  -> {status} after {tries}: {question or ''}")
            for raw in rejected:
                print(f"     rejected: {raw}")
        elif n % 50 == 0 or n == len(jobs):
            rate = (time.time() - started) / n
            print(f"{n}/{len(jobs)} {counts} {rate:.1f} s each, about {rate * (len(jobs) - n) / 60:.0f} min left", flush=True)
    print(f"done: {counts}")


if __name__ == "__main__":
    main()
