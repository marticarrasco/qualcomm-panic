"""The tutor's logic: which problem comes next, and what help is given.

No web code here. The language model is never the source of truth: the right
answer comes from the dataset label, similar problems come from the bank
built by build_bank.py, and every decision is a plain rule.

    tutor = Tutor("data/tutor.db", "data/tutor_students.db")
    tutor.start("Aina", "cat")          # -> state with the first problem
    tutor.answer("Aina", "C")           # -> right or wrong, never the label when wrong
    tutor.hint("Aina")                  # -> the next rung of the hint ladder
    tutor.reveal("Aina")                # -> the label, counted as not solved
    tutor.next("Aina")                  # -> the next problem
    tutor.summary("Aina")               # -> the teacher view

The rules, in one place:
  solved first try with at most one hint  -> same topic, one level up
  solved with one wrong try or 2+ hints    -> same topic, same level
  not solved, answer shown, or 2+ wrong    -> same topic, one level down,
                                              the closest problem to the one missed
If the topic has no unseen problem within one level of the target, the next
problem comes from another topic at the student's level (never a repeat).
A problem closes without showing the answer when only two letters are left.
Standard library only.
"""
import hashlib
import json
import math
import os
import random
import re
import sqlite3
import threading
import time
import urllib.request
from array import array

from .topics import NAMES, TIPS, TOPICS, clean

LETTERS = "ABCDE"
MAX_LEVEL = 7
# In mixed mode the student stays this many problems on a topic before moving on
BLOCK = 4
RUNGS = ["tip", "example", "eliminate", "nudge", "reveal"]  # rung 1..5
# How the model that wrote the rung-1 questions is named on the page
MODEL_NAMES = {"qwen3.5-2b": "Qwen3.5 2B", "qwen3.5-0.8b": "Qwen3.5 0.8B"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    name TEXT PRIMARY KEY, language TEXT, topic_choice TEXT, start_level INTEGER,
    streak INTEGER DEFAULT 0, best_streak INTEGER DEFAULT 0, created REAL);
CREATE TABLE IF NOT EXISTS topic_state (
    name TEXT, topic TEXT, level INTEGER, mastered INTEGER,
    attempts INTEGER DEFAULT 0, correct INTEGER DEFAULT 0, clean INTEGER DEFAULT 0, hints INTEGER DEFAULT 0,
    PRIMARY KEY (name, topic));
CREATE TABLE IF NOT EXISTS history (
    n INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, problem_id TEXT, topic TEXT, level INTEGER,
    status TEXT DEFAULT 'open',          -- open | solved | failed | revealed | skipped
    wrong_tries INTEGER DEFAULT 0, tried TEXT DEFAULT '', eliminated TEXT DEFAULT '',
    rungs TEXT DEFAULT '',               -- hint rungs used, e.g. "tip,example"
    example_id TEXT, example_shown INTEGER DEFAULT 0,
    outcome TEXT,                        -- up | stay | down, set once when the problem closes
    started REAL, ended REAL);
CREATE INDEX IF NOT EXISTS history_name ON history (name, n);
CREATE TABLE IF NOT EXISTS nudges (
    id TEXT, language TEXT, status TEXT, text TEXT, PRIMARY KEY (id, language));  -- pending | ok | rejected | error
"""

NUDGE_SYSTEM = ("You are a maths teacher. A student is stuck on the problem below. "
                "Do NOT solve it and do NOT state any result or number that is not in the problem. "
                "Write ONE short sentence (at most 20 words) telling the student what to do first, "
                "for example what to write down, draw or list.")
NUDGE_LANGUAGE = {"eng": "", "cat": " Write the sentence in Catalan."}

OPTION_PATTERNS = [
    re.compile(r"answer|resposta|soluci|result is|equals|=", re.IGNORECASE),
    re.compile(r"\b[A-E]\)|\([A-E]\)"),
    re.compile(r"\b(option|opci[oó]|choice|letter|lletra)\s*[A-E]\b", re.IGNORECASE),
]


class TutorError(ValueError):
    """A request that makes no sense right now (unknown student, no open problem...)."""


def eliminate(problem_id, label, tried=""):
    """Rung 3: letters to cross out. Never the label, never one already tried,
    always leaving the right letter and at least one other. The choice depends
    only on the problem, so it is the same every time."""
    untried = [letter for letter in LETTERS if letter != label and letter not in tried]
    untried.sort(key=lambda letter: hashlib.sha256(f"{problem_id}|{letter}".encode()).hexdigest())
    return sorted(untried[:max(0, min(2, len(untried) - 1))])


def check_nudge(sentence, problem_text, correct_value=None):
    """Return the model's sentence if it is safe to show, else None.

    Safe means: one short sentence, no "answer", no option letter, no option
    value we know to be correct, and no number that is not already in the problem
    (a new number would be a result the model worked out, right or wrong)."""
    sentence = re.sub(r"<think>.*?</think>", "", sentence or "", flags=re.DOTALL).strip()
    sentence = sentence.splitlines()[0].strip().strip('"') if sentence else ""
    words = sentence.split()
    if not 4 <= len(words) <= 26:
        return None
    if any(pattern.search(sentence) for pattern in OPTION_PATTERNS):
        return None
    if correct_value:
        value = str(correct_value).strip().lower()
        if value and re.search(r"(?<![\w.,])" + re.escape(value) + r"(?![\w])", sentence.lower()):
            return None
    known = set(re.findall(r"\d+", problem_text))
    if any(number not in known for number in re.findall(r"\d+", sentence)):
        return None
    return sentence


def check_question(text, problem_text, correct_value=None):
    """Return the model's guiding question if it is safe to show, else None.

    The same rules as check_nudge (short, no answer words, no option letter, no
    correct value, no number that is not in the problem), and it must be one
    question: it ends with "?" and asks only once."""
    text = re.sub(r"\$([^$]{1,20})\$", r"\1", text or "")  # "$x$" -> "x": the page shows plain text
    sentence = check_nudge(text, problem_text, correct_value)
    if not sentence or not sentence.endswith("?") or sentence.count("?") > 2:
        return None
    if re.search(r"\b(solve|resol|calculate the answer|the result|el resultat|cannot see|can't see|no puc veure)\b|¿",
                 sentence, re.IGNORECASE):
        return None
    # A question that only repeats the problem's own question does not help
    words = lambda s: {w for w in re.findall(r"\w+", s.lower()) if len(w) > 3}
    asked = [part for part in re.split(r"(?<=[.?!])\s+", problem_text) if part.strip().endswith("?")]
    mine = words(sentence)
    for part in asked or [problem_text.split(".")[-1]]:
        theirs = words(part)
        if mine and theirs and len(mine & theirs) / len(mine | theirs) >= 0.6:
            return None
    # Words from the made-up examples in the prompt must not leak into a real problem
    for borrowed in ("frog", "granota", "well", "pou", "consecutive", "consecutius"):
        if re.search(rf"\b{borrowed}\b", sentence, re.IGNORECASE) and not re.search(rf"\b{borrowed}\b", problem_text, re.IGNORECASE):
            return None
    return sentence


def _locked(method):
    def wrapper(self, *args, **kwargs):
        with self.lock:
            return method(self, *args, **kwargs)
    wrapper.__name__, wrapper.__doc__ = method.__name__, method.__doc__
    return wrapper


class Tutor:
    def __init__(self, bank="data/tutor.db", students="data/tutor_students.db", llm_url=None, seed=0,
                 questions="data/tutor_questions.db", inline_problems=False):
        """llm_url: the chat completions URL for the optional rung 4, or None to leave that rung out.
        questions: the guiding questions written ahead of time by tutor/write_questions.py
        (shown on rung 1); if the file is missing, rung 1 is the topic tip alone."""
        self.llm_url = llm_url
        self.questions_path = questions
        self.inline_problems = inline_problems
        self._questions = None
        self.seed = seed
        self.lock = threading.RLock()
        self._load_bank(bank)
        self.db = sqlite3.connect(students, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(SCHEMA)
        # A nudge that was being written when the server stopped will never arrive
        self.db.execute("DELETE FROM nudges WHERE status = 'pending'")
        self.db.commit()

    # ---------- the bank, read once into memory ----------

    def _load_bank(self, path):
        db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        self.problems, self.pool, self.vectors = {}, {}, {}
        for pid, level, year, number, text_only, label, topic in db.execute(
                "SELECT id, level, year, number, text_only, label, topic FROM problems ORDER BY id"):
            self.problems[pid] = {"id": pid, "level": level, "year": year, "number": number,
                                  "text_only": bool(text_only), "label": label, "topic": topic}
            self.pool.setdefault((topic, level), []).append(pid)
        for pid, blob in db.execute("SELECT id, vec FROM vectors"):
            vec = array("f")
            vec.frombytes(blob)
            self.vectors[pid] = vec
        self.neighbours = {}
        for pid, kind, other in db.execute("SELECT id, kind, other_id FROM neighbours ORDER BY id, kind, rank"):
            self.neighbours.setdefault((pid, kind), []).append(other)
        self.duplicates = {}
        for pid, other in db.execute("SELECT id, other_id FROM duplicates"):
            self.duplicates.setdefault(pid, set()).add(other)
        self.texts = {(pid, lang): text for pid, lang, text in db.execute("SELECT id, language, text FROM texts")}
        self.options = {(pid, lang): json.loads(options)
                        for pid, lang, options in db.execute("SELECT id, language, options FROM options")}
        db.close()

    def _question(self, problem_id, language):
        """The model's guiding question for this problem, or None. Read from the file on
        each request, so questions written while the server runs appear at once."""
        if self._questions is None:
            if not self.questions_path or not os.path.exists(self.questions_path):
                return None
            self._questions = sqlite3.connect(f"file:{self.questions_path}?mode=ro", uri=True, check_same_thread=False)
        try:
            row = self._questions.execute("SELECT text, model FROM questions WHERE id = ? AND language = ? AND status = 'ok'",
                                          (problem_id, language)).fetchone()
        except sqlite3.Error:
            return None
        return (row[0], MODEL_NAMES.get(row[1], row[1] or "")) if row else None

    def info(self):
        counts = {topic: [len(self.pool.get((topic, level), ())) for level in range(MAX_LEVEL + 1)] for topic in TOPICS}
        return {"problems": len(self.problems), "levels": list(range(MAX_LEVEL + 1)),
                "topics": [{"key": t, "names": NAMES[t], "count": sum(counts[t]), "per_level": counts[t]}
                           for t in TOPICS if sum(counts[t]) or not self.inline_problems],
                "nudge": bool(self.llm_url)}

    # ---------- small helpers ----------

    def _student(self, name):
        row = self.db.execute("SELECT * FROM students WHERE name = ?", (name,)).fetchone()
        if not row:
            raise TutorError("unknown student")
        return row

    def _current(self, name):
        return self.db.execute("SELECT * FROM history WHERE name = ? ORDER BY n DESC LIMIT 1", (name,)).fetchone()

    def _seen(self, name):
        """Every problem this student must not get again: served, shown as an
        example, or a near-duplicate of one of those."""
        seen = set()
        for pid, example in self.db.execute("SELECT problem_id, example_id FROM history WHERE name = ?", (name,)):
            for one in (pid, example):
                if one:
                    seen.add(one)
                    seen |= self.duplicates.get(one, set())
        return seen

    def _topic_level(self, student, topic):
        """The level to serve next. A topic never tried starts where the
        student is in the topics already tried, else at the chosen starting level."""
        levels = {row["topic"]: row["level"] for row in self.db.execute(
            "SELECT topic, level FROM topic_state WHERE name = ?", (student["name"],))}
        if topic in levels:
            return levels[topic]
        return round(sum(levels.values()) / len(levels)) if levels else student["start_level"]

    def _similarity(self, a, b):
        va, vb = self.vectors[a], self.vectors[b]
        return math.sumprod(va, vb) if hasattr(math, "sumprod") else sum(x * y for x, y in zip(va, vb))

    # ---------- choosing the next problem ----------

    def _pick(self, name, topic, level, anchor=None, kind="same_level", reach=1):
        """An unseen problem of this topic, at this level if there is one, else
        up to `reach` levels away. With an anchor (the problem just done) the
        most similar candidate wins; without one the choice is a seeded draw."""
        seen = self._seen(name)
        step = -1 if kind == "easier" else 1
        order = [level]
        for distance in range(1, reach + 1):
            order += [level + step * distance, level - step * distance]
        for lvl in order:
            candidates = [pid for pid in self.pool.get((topic, lvl), ()) if pid not in seen]
            if not candidates:
                continue
            if anchor:
                return max(candidates, key=lambda pid: self._similarity(anchor, pid))
            return random.Random(f"{self.seed}|{name}|{len(seen)}").choice(candidates)
        return None

    def _mixed_topic(self, name, last):
        """Mixed mode: stay BLOCK problems on a topic, then go to the least practised one."""
        if last:
            recent = [row["topic"] for row in self.db.execute(
                "SELECT topic FROM history WHERE name = ? ORDER BY n DESC LIMIT ?", (name, BLOCK))]
            if len(recent) < BLOCK or any(topic != last["topic"] for topic in recent):
                return last["topic"]
        attempts = {row["topic"]: row["attempts"] for row in self.db.execute(
            "SELECT topic, attempts FROM topic_state WHERE name = ?", (name,))}
        others = [topic for topic in TOPICS if not last or topic != last["topic"]]
        return min(others, key=lambda topic: (attempts.get(topic, 0), TOPICS.index(topic)))

    def _serve(self, student, last=None):
        name = student["name"]
        topic = student["topic_choice"] or self._mixed_topic(name, last)
        anchor, kind = None, "same_level"
        if last and last["topic"] == topic and last["outcome"] and last["problem_id"] in self.vectors:
            anchor = last["problem_id"]
            kind = {"down": "easier", "up": "harder"}.get(last["outcome"], "same_level")
        # Small topics run dry near a level. Then a problem of another topic at the
        # student's level is better than a repeat or one that is far too hard.
        attempts = {row["topic"]: row["attempts"] for row in self.db.execute(
            "SELECT topic, attempts FROM topic_state WHERE name = ?", (name,))}
        others = sorted((t for t in TOPICS if t != topic), key=lambda t: (attempts.get(t, 0), TOPICS.index(t)))
        for candidate, reach in [(t, 1) for t in [topic] + others] + [(t, MAX_LEVEL) for t in [topic] + others]:
            level = max(0, min(MAX_LEVEL, self._topic_level(student, candidate)))
            pid = self._pick(name, candidate, level, anchor if candidate == topic else None, kind, reach)
            if pid:
                self.db.execute("INSERT INTO history (name, problem_id, topic, level, started) VALUES (?, ?, ?, ?, ?)",
                                (name, pid, candidate, self.problems[pid]["level"], time.time()))
                return pid
        return None

    # ---------- closing a problem: the level rule ----------

    def _close(self, row, status):
        """Record how a problem ended and move the topic level. Runs once per problem."""
        name, topic, level = row["name"], row["topic"], row["level"]
        hints = len([r for r in row["rungs"].split(",") if r])
        solved = status == "solved"
        if status == "skipped" and row["wrong_tries"] == 0 and hints == 0:
            outcome = None  # looked at it and moved on: no evidence either way
        elif solved and row["wrong_tries"] == 0 and hints <= 1:
            outcome = "up"
        elif solved and row["wrong_tries"] <= 1:
            outcome = "stay"
        else:
            outcome = "down"
        self.db.execute("UPDATE history SET status = ?, outcome = ?, ended = ? WHERE n = ?",
                        (status, outcome, time.time(), row["n"]))
        student = self._student(name)
        state = self.db.execute("SELECT * FROM topic_state WHERE name = ? AND topic = ?", (name, topic)).fetchone()
        before = state["level"] if state else student["start_level"]
        mastered = state["mastered"] if state else None
        after = before
        if outcome == "up":
            after, mastered = min(MAX_LEVEL, level + 1), max(level, mastered if mastered is not None else -1)
        elif outcome == "down":
            after = max(0, level - 1)
            if mastered is not None and mastered >= level:
                mastered = level - 1
        elif outcome == "stay":
            after = level
        if outcome:
            self.db.execute(
                "INSERT INTO topic_state (name, topic, level, mastered, attempts, correct, clean, hints) "
                "VALUES (?, ?, ?, ?, 1, ?, ?, ?) ON CONFLICT (name, topic) DO UPDATE SET level = excluded.level, "
                "mastered = excluded.mastered, attempts = attempts + 1, correct = correct + excluded.correct, "
                "clean = clean + excluded.clean, hints = hints + excluded.hints",
                (name, topic, after, mastered, int(solved), int(outcome == "up"), hints))
            streak = student["streak"] + 1 if solved else 0
            self.db.execute("UPDATE students SET streak = ?, best_streak = MAX(best_streak, ?) WHERE name = ?",
                            (streak, streak, name))
        return {"outcome": outcome, "level_before": before, "level_after": after}

    # ---------- what the page is allowed to see ----------

    def _example_for(self, row, seen):
        pid = row["problem_id"]
        banned = self.duplicates.get(pid, set()) | {pid}
        options = [(other, True) for other in self.neighbours.get((pid, "easier"), ())]
        options += [(other, False) for other in self.neighbours.get((pid, "same_level"), ())]
        options = [(other, easier) for other, easier in options if other not in banned]
        fresh = [pair for pair in options if pair[0] not in seen]
        return (fresh or options or [(None, False)])[0]

    def _hints(self, row, language):
        """The hints already given for this problem, ready to draw."""
        out = []
        for kind in [r for r in row["rungs"].split(",") if r]:
            hint = {"rung": RUNGS.index(kind) + 1, "kind": kind}
            if kind == "tip":
                hint["topic_name"] = NAMES[row["topic"]][language]
                hint["text"] = TIPS[row["topic"]][language]
                question = self._question(row["problem_id"], language)
                if question:
                    hint["question"], hint["question_model"] = question
            elif kind == "example":
                other = self.problems[row["example_id"]]
                hint.update(self._material(other["id"], language), level=other["level"],
                            easier=other["level"] < row["level"],
                            answer=other["label"] if row["example_shown"] else None)
            elif kind == "eliminate":
                hint["letters"] = list(row["eliminated"])
            elif kind == "nudge":
                nudge = self.db.execute("SELECT text FROM nudges WHERE id = ? AND status = 'ok' ORDER BY language = ? DESC",
                                        (row["problem_id"], language)).fetchone()
                hint["text"] = nudge["text"] if nudge else ""
            out.append(hint)
        return out

    def _material(self, problem_id, language):
        if self.inline_problems:
            return {"text": self.texts[(problem_id, language)], "options": self.options[(problem_id, language)]}
        return {"image": f"/image/{problem_id}_{language}.png"}

    def _public(self, row, language):
        """One problem as the browser may see it. The label is only included
        once the student has solved it or asked to see the answer."""
        if not row:
            return None
        problem = self.problems[row["problem_id"]]
        shown = {
            "id": problem["id"], **self._material(problem["id"], language),
            "topic": row["topic"], "topic_name": NAMES[row["topic"]][language], "level": row["level"],
            "text_only": problem["text_only"], "status": row["status"], "closed": row["status"] != "open",
            "tries": row["wrong_tries"], "tried": list(row["tried"]), "eliminated": list(row["eliminated"]),
            "hints": self._hints(row, language),
            "nudge_possible": bool(self.llm_url) and problem["text_only"],
        }
        if row["status"] in ("solved", "revealed"):
            shown["label"] = problem["label"]
        return shown

    @_locked
    def chat_context(self, name, problem_id):
        student = self._student(name)
        row = self._current(name)
        if not row or row["problem_id"] != problem_id or row["status"] != "open":
            raise TutorError("no open problem for this message")
        stat = self.db.execute("SELECT attempts, correct FROM topic_state WHERE name = ? AND topic = ?",
                               (name, row["topic"])).fetchone()
        language = student["language"]
        options = self.options.get((problem_id, language)) or self.options.get((problem_id, "eng")) or {}
        tried = row["tried"]
        return {
            "problem": problem_id, "topic": row["topic"], "language": language,
            "exercise": clean(self.texts.get((problem_id, language)) or self.texts.get((problem_id, "eng"), "")),
            "correct_value": options.get(self.problems[problem_id]["label"]),
            "last_wrong": options.get(tried[-1]) if tried else None,
            "text_only": self.problems[problem_id]["text_only"],
            "tried": tried, "support": bool(stat and stat["attempts"] >= 2 and stat["correct"] * 2 < stat["attempts"]),
        }

    def _profile(self, name, language):
        rows = self.db.execute("SELECT * FROM topic_state WHERE name = ?", (name,)).fetchall()
        rows.sort(key=lambda r: TOPICS.index(r["topic"]))
        return [{"topic": r["topic"], "topic_name": NAMES[r["topic"]][language], "level": r["level"],
                 "mastered": r["mastered"], "attempts": r["attempts"], "correct": r["correct"],
                 "clean": r["clean"], "hints": r["hints"],
                 "accuracy": round(r["correct"] / r["attempts"], 3) if r["attempts"] else None} for r in rows]

    @_locked
    def state(self, name):
        student = self._student(name)
        return {"student": {k: student[k] for k in ("name", "language", "topic_choice", "start_level",
                                                    "streak", "best_streak")},
                "problem": self._public(self._current(name), student["language"]),
                "profile": self._profile(name, student["language"])}

    # ---------- the steps of a session ----------

    @_locked
    def start(self, name, language="eng", topic=None, level=None):
        """Create or resume a student. A returning student keeps their profile and open problem."""
        name = " ".join(str(name or "").split())[:40]
        if not name:
            raise TutorError("a name is needed")
        language = language if language in ("eng", "cat") else "eng"
        topic = topic if topic in TOPICS else None
        existing = self.db.execute("SELECT * FROM students WHERE name = ?", (name,)).fetchone()
        if existing:
            start_level = existing["start_level"] if level is None else max(0, min(MAX_LEVEL, int(level)))
            self.db.execute("UPDATE students SET language = ?, topic_choice = ?, start_level = ? WHERE name = ?",
                            (language, topic, start_level, name))
        else:
            start_level = 1 if level is None else max(0, min(MAX_LEVEL, int(level)))
            self.db.execute("INSERT INTO students (name, language, topic_choice, start_level, created) "
                            "VALUES (?, ?, ?, ?, ?)", (name, language, topic, start_level, time.time()))
        current = self._current(name)
        same_topic_choice = existing and existing["topic_choice"] == topic
        keep = current and current["status"] == "open" and (topic is None or current["topic"] == topic or same_topic_choice)
        if not keep:
            if current and current["status"] == "open":
                self._close(current, "skipped")
            self._serve(self._student(name), self._current(name))
        self.db.commit()
        return self.state(name)

    @_locked
    def answer(self, name, choice):
        """Check one answer. A wrong answer never comes back with the label."""
        row = self._current(name)
        if not row or row["status"] != "open":
            raise TutorError("no open problem")
        if len(choice) != 1 or choice not in LETTERS or choice in row["tried"] or choice in row["eliminated"]:
            raise TutorError("that letter cannot be chosen")
        label = self.problems[row["problem_id"]]["label"]
        result = {"correct": choice == label, "closed": False, "events": []}
        if choice == label:
            result.update(self._close(row, "solved"), closed=True)
            result["events"] = [("correct", {}), ("level", {"level": result["level_after"]})]
        else:
            tried = row["tried"] + choice
            self.db.execute("UPDATE history SET wrong_tries = wrong_tries + 1, tried = ? WHERE n = ?", (tried, row["n"]))
            result["events"] = [("wrong", {})]
            if len(LETTERS) - len(set(tried) | set(row["eliminated"])) <= 2:
                # One more wrong guess would give the answer away, so stop here
                result.update(self._close(self._current(name), "failed"), closed=True)
                result["events"].append(("level", {"level": result["level_after"]}))
        self.db.commit()
        result.update(self.state(name))
        return result

    @_locked
    def hint(self, name):
        """Climb one rung. Returns {"hint": None} when only "show me the answer" is left."""
        row = self._current(name)
        if not row or row["status"] != "open":
            raise TutorError("no open problem")
        student = self._student(name)
        rungs = [r for r in row["rungs"].split(",") if r]
        problem = self.problems[row["problem_id"]]
        kind = None
        if "tip" not in rungs:
            kind = "tip"
            self._nudge_start(problem["id"], student["language"])
        if not kind and "example" not in rungs:
            example, _easier = self._example_for(row, self._seen(name))
            if example:
                kind = "example"
                self.db.execute("UPDATE history SET example_id = ? WHERE n = ?", (example, row["n"]))
        if not kind and "eliminate" not in rungs:
            letters = eliminate(problem["id"], problem["label"], row["tried"])
            if letters:
                kind = "eliminate"
                self.db.execute("UPDATE history SET eliminated = ? WHERE n = ?", ("".join(letters), row["n"]))
        if not kind and "nudge" not in rungs and self._nudge_ready(problem["id"]):
            kind = "nudge"
        if kind:
            self.db.execute("UPDATE history SET rungs = ? WHERE n = ?", (",".join(rungs + [kind]), row["n"]))
            self.db.commit()
        state = self.state(name)
        hint = next((h for h in state["problem"]["hints"] if h["kind"] == kind), None)
        return {"hint": hint, "events": [("hint", {"rung": hint["rung"]})] if hint else [], **state}

    @_locked
    def example_answer(self, name):
        """The answer of the example problem in rung 2 (not of the student's own problem)."""
        row = self._current(name)
        if not row or not row["example_id"]:
            raise TutorError("no example to show")
        self.db.execute("UPDATE history SET example_shown = 1 WHERE n = ?", (row["n"],))
        self.db.commit()
        return {"answer": self.problems[row["example_id"]]["label"], "events": [], **self.state(name)}

    @_locked
    def reveal(self, name):
        """Rung 5: show the answer because the student asked. Counts as not solved."""
        row = self._current(name)
        if not row:
            raise TutorError("no problem")
        result = {"events": []}
        if row["status"] == "open":
            result.update(self._close(row, "revealed"))
            result["events"] = [("hint", {"rung": 5}), ("level", {"level": result["level_after"]})]
        elif row["status"] in ("failed", "skipped"):
            self.db.execute("UPDATE history SET status = 'revealed' WHERE n = ?", (row["n"],))
            result["events"] = [("hint", {"rung": 5})]
        self.db.commit()
        result.update(self.state(name))
        result["label"] = self.problems[row["problem_id"]]["label"]
        return result

    @_locked
    def next(self, name, topic="keep"):
        """Move on. topic: "keep", a topic key, or None for mixed practice."""
        student = self._student(name)
        if topic != "keep":
            self.db.execute("UPDATE students SET topic_choice = ? WHERE name = ?",
                            (topic if topic in TOPICS else None, name))
            student = self._student(name)
        row = self._current(name)
        if row and row["status"] == "open":
            self._close(row, "skipped")
        pid = self._serve(student, self._current(name))
        self.db.commit()
        state = self.state(name)
        state["events"] = [("thinking", {"level": state["problem"]["level"]})] if pid else [("idle", {})]
        state["finished"] = pid is None
        return state

    # ---------- rung 4: one sentence from the small model, never waited for ----------

    def _nudge_ready(self, problem_id):
        return bool(self.db.execute("SELECT 1 FROM nudges WHERE id = ? AND status = 'ok'", (problem_id,)).fetchone())

    def _nudge_start(self, problem_id, language):
        if not self.llm_url or not self.problems[problem_id]["text_only"]:
            return  # the model cannot see the figure, so it must not talk about it
        if self.db.execute("SELECT 1 FROM nudges WHERE id = ? AND language = ?", (problem_id, language)).fetchone():
            return
        self.db.execute("INSERT INTO nudges VALUES (?, ?, 'pending', NULL)", (problem_id, language))
        threading.Thread(target=self._nudge_work, args=(problem_id, language), daemon=True).start()

    def _nudge_work(self, problem_id, language):
        text = clean(self.texts.get((problem_id, language)) or self.texts[(problem_id, "eng")])
        status, sentence = "error", None
        try:
            request = urllib.request.Request(self.llm_url, data=json.dumps({
                "messages": [{"role": "system", "content": NUDGE_SYSTEM + NUDGE_LANGUAGE[language]},
                             {"role": "user", "content": text}],
                "max_tokens": 48, "temperature": 0, "cache_prompt": True,
                "chat_template_kwargs": {"enable_thinking": False},
            }).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=180) as response:
                raw = json.loads(response.read())["choices"][0]["message"].get("content") or ""
            label = self.problems[problem_id]["label"]
            options = self.options.get((problem_id, language)) or self.options.get((problem_id, "eng")) or {}
            sentence = check_nudge(raw, text, options.get(label))
            status = "ok" if sentence else "rejected"
            sentence = sentence or raw[:300]
        except (OSError, ValueError, KeyError, IndexError):
            pass
        with self.lock:
            self.db.execute("UPDATE nudges SET status = ?, text = ? WHERE id = ? AND language = ?",
                            (status, sentence, problem_id, language))
            self.db.commit()

    # ---------- the teacher view ----------

    @_locked
    def students(self):
        return [{"name": r["name"], "language": r["language"], "problems": r["n"], "solved": r["solved"] or 0}
                for r in self.db.execute(
                    "SELECT s.name, s.language, COUNT(h.n) AS n, SUM(h.status = 'solved') AS solved "
                    "FROM students s LEFT JOIN history h ON h.name = s.name GROUP BY s.name ORDER BY s.created DESC")]

    @_locked
    def summary(self, name, language=None):
        student = self._student(name)
        language = language if language in ("eng", "cat") else student["language"]
        profile = self._profile(name, language)
        closed = self.db.execute("SELECT * FROM history WHERE name = ? AND outcome IS NOT NULL ORDER BY n", (name,)).fetchall()
        attempts = sum(p["attempts"] for p in profile)
        correct = sum(p["correct"] for p in profile)
        available = [p for p in profile if p["attempts"] >= 2 and
                     self._pick(name, p["topic"], p["level"])]
        weak, strong = self._practice_topics(available)
        recent = [{"id": r["problem_id"], "topic_name": NAMES[r["topic"]][language], "level": r["level"],
                   "status": r["status"], "wrong_tries": r["wrong_tries"],
                   "hints": len([x for x in r["rungs"].split(",") if x]), "outcome": r["outcome"],
                   "seconds": round(r["ended"] - r["started"], 1) if r["ended"] else None} for r in closed[-15:]]
        return {"student": student["name"], "language": language, "profile": profile,
                "totals": {"attempts": attempts, "correct": correct,
                           "accuracy": round(correct / attempts, 3) if attempts else None,
                           "hints": sum(p["hints"] for p in profile),
                           "streak": student["streak"], "best_streak": student["best_streak"]},
                "practise_next": self._advice(profile, language), "recent": recent[::-1],
                "recommendations": [
                    {"kind": kind, "topic": p["topic"], "topic_name": p["topic_name"], "level": p["level"]}
                    for kind, p in (("review", weak), ("challenge", strong)) if p is not None
                ]}

    def _practice_topics(self, tried):
        if not tried:
            return None, None
        weak = min(tried, key=lambda p: (p["accuracy"], -p["hints"] / p["attempts"]))
        strong = max(tried, key=lambda p: (p["mastered"] if p["mastered"] is not None else -1, p["accuracy"]))
        return weak, strong if strong is not weak and strong["accuracy"] >= 0.6 else None

    def _advice(self, profile, language):
        """One line for the teacher, worked out from the profile alone."""
        weak, strong = self._practice_topics([p for p in profile if p["attempts"] >= 2])
        untried = [NAMES[t][language] for t in TOPICS if t not in {p["topic"] for p in profile}]
        if not weak:
            return {"eng": "Not enough answers yet: do a few more problems first.",
                    "cat": "Encara no hi ha prou respostes: fes uns quants problemes més."}[language]
        per = weak["hints"] / weak["attempts"]
        if language == "cat":
            line = (f"Practica «{weak['topic_name']}» al nivell {weak['level']}: {weak['correct']} de "
                    f"{weak['attempts']} resolts, {per:.1f} pistes per problema.")
            if strong:
                line += f" Punt fort: «{strong['topic_name']}» (nivell {strong['level']})."
            if untried:
                line += " Encara no provats: " + ", ".join(untried[:3]) + "."
        else:
            line = (f"Practise \"{weak['topic_name']}\" at level {weak['level']}: {weak['correct']} of "
                    f"{weak['attempts']} solved, {per:.1f} hints per problem.")
            if strong:
                line += f" Strongest: \"{strong['topic_name']}\" (level {strong['level']})."
            if untried:
                line += " Not tried yet: " + ", ".join(untried[:3]) + "."
        return line
