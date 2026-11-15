"""The tutor's logic: which problem comes next, and what help is given.

No web code here. The language model is never the source of truth: the right
answer comes from the dataset label, similar problems come from the bank
built by build_bank.py, and every decision is a plain rule.

    tutor = Tutor("data/tutor.db", "data/tutor_students.db")
    tutor.start("Aina", "cat")          # -> state with the first problem
    tutor.answer("Aina", "C")           # -> right or wrong, never the label when wrong
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
import json
import math
import random
import sqlite3
import threading
import time
from array import array
from pathlib import Path

from .topics import NAMES, TOPICS, clean

LETTERS = "ABCDE"
MAX_LEVEL = 7
# In mixed mode the student stays this many problems on a topic before moving on
BLOCK = 4
HINT_STEPS = ("orient", "apply", "check")

SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    name TEXT PRIMARY KEY, language TEXT, topic_choice TEXT, start_level INTEGER,
    streak INTEGER DEFAULT 0, best_streak INTEGER DEFAULT 0, created REAL, finished INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS topic_state (
    name TEXT, topic TEXT, level INTEGER, mastered INTEGER,
    attempts INTEGER DEFAULT 0, correct INTEGER DEFAULT 0, clean INTEGER DEFAULT 0, hints INTEGER DEFAULT 0,
    PRIMARY KEY (name, topic));
CREATE TABLE IF NOT EXISTS history (
    n INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT, problem_id TEXT, topic TEXT, level INTEGER,
    status TEXT DEFAULT 'open',          -- open | solved | failed | revealed | skipped
    wrong_tries INTEGER DEFAULT 0, tried TEXT DEFAULT '', eliminated TEXT DEFAULT '',
    rungs TEXT DEFAULT '',               -- hint rungs used, e.g. "orient,apply"
    outcome TEXT,                        -- up | stay | down, set once when the problem closes
    started REAL, ended REAL, chat_help INTEGER DEFAULT 0, transfer_from INTEGER);
CREATE INDEX IF NOT EXISTS history_name ON history (name, n);
CREATE TABLE IF NOT EXISTS model_help (
    history_n INTEGER, language TEXT, kind TEXT, text TEXT,
    PRIMARY KEY (history_n, language, kind));
"""

class TutorError(ValueError):
    """A request that cannot be applied to the current exercise."""


def _locked(method):
    def wrapper(self, *args, **kwargs):
        with self.lock:
            return method(self, *args, **kwargs)
    wrapper.__name__, wrapper.__doc__ = method.__name__, method.__doc__
    return wrapper


class Tutor:
    def __init__(self, bank="data/tutor.db", students="data/tutor_students.db", seed=0, model_enabled=True):
        self.model_enabled = model_enabled
        self.seed = seed
        self.lock = threading.RLock()
        self._load_bank(bank)
        self.db = sqlite3.connect(students, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(SCHEMA)
        for table, column, definition in (
            ("students", "finished", "INTEGER DEFAULT 0"),
            ("history", "chat_help", "INTEGER DEFAULT 0"),
            ("history", "transfer_from", "INTEGER"),
        ):
            if column not in {row["name"] for row in self.db.execute(f"PRAGMA table_info({table})")}:
                self.db.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
        self.db.commit()

    # ---------- the bank, read once into memory ----------

    def _load_bank(self, path):
        db = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
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

    def info(self):
        counts = {topic: [len(self.pool.get((topic, level), ())) for level in range(MAX_LEVEL + 1)] for topic in TOPICS}
        return {"problems": len(self.problems), "levels": list(range(MAX_LEVEL + 1)),
                "topics": [{"key": t, "names": NAMES[t], "count": sum(counts[t]), "per_level": counts[t]}
                           for t in TOPICS if sum(counts[t])],
                "model": self.model_enabled}

    # ---------- small helpers ----------

    def _student(self, name):
        row = self.db.execute("SELECT * FROM students WHERE name = ?", (name,)).fetchone()
        if not row:
            raise TutorError("unknown student")
        return row

    def _current(self, name, expected=None):
        row = self.db.execute("SELECT * FROM history WHERE name = ? ORDER BY n DESC LIMIT 1", (name,)).fetchone()
        if expected is not None and (not row or row["problem_id"] != expected):
            raise TutorError("this problem has changed")
        return row

    @staticmethod
    def _hint_count(row):
        return len([r for r in row["rungs"].split(",") if r]) + row["chat_help"]

    def _seen(self, name):
        """Every distinct exercise already served to this student."""
        return {row[0] for row in self.db.execute("SELECT problem_id FROM history WHERE name = ?", (name,))}

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
                alternatives = [pid for pid in candidates if pid not in self.duplicates.get(anchor, ())]
                return max(alternatives or candidates, key=lambda pid: self._similarity(anchor, pid))
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

    def _transfer_target(self, row):
        if (not row or row["transfer_from"] is not None or row["status"] not in ("solved", "revealed", "failed")
                or (row["status"] == "solved" and not row["wrong_tries"] and not self._hint_count(row))):
            return None
        problem = self.problems.get(row["problem_id"])
        if not problem:
            return None
        seen = self._seen(row["name"]) | self.duplicates.get(problem["id"], set())
        kinds = ("easier", "same_level") if row["outcome"] == "down" else ("same_level",)
        for kind in kinds:
            for pid in self.neighbours.get((problem["id"], kind), ()):
                candidate = self.problems.get(pid)
                if (pid not in seen and candidate and candidate["topic"] == row["topic"]
                        and candidate["text_only"] == problem["text_only"]):
                    return pid
        return None

    def _serve(self, student, last=None, allow_transfer=False):
        name = student["name"]
        topic = student["topic_choice"] or self._mixed_topic(name, last)
        if allow_transfer and (not student["topic_choice"] or student["topic_choice"] == last["topic"]):
            transfer = self._transfer_target(last)
            if transfer:
                problem = self.problems[transfer]
                self.db.execute("INSERT INTO history (name, problem_id, topic, level, started, transfer_from) "
                                "VALUES (?, ?, ?, ?, ?, ?)",
                                (name, transfer, problem["topic"], problem["level"], time.time(), last["n"]))
                return transfer
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
        hints = self._hint_count(row)
        solved = status == "solved"
        independent = solved and not row["wrong_tries"] and not hints
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
            after = min(MAX_LEVEL, level + 1)
            if independent:
                mastered = max(level, mastered if mastered is not None else -1)
        elif outcome == "down":
            after = max(0, level - 1)
            if mastered is not None and mastered >= level:
                mastered = level - 1 if level else None
        elif outcome == "stay":
            after = level
        if outcome:
            self.db.execute(
                "INSERT INTO topic_state (name, topic, level, mastered, attempts, correct, clean, hints) "
                "VALUES (?, ?, ?, ?, 1, ?, ?, ?) ON CONFLICT (name, topic) DO UPDATE SET level = excluded.level, "
                "mastered = excluded.mastered, attempts = attempts + 1, correct = correct + excluded.correct, "
                "clean = clean + excluded.clean, hints = hints + excluded.hints",
                (name, topic, after, mastered, int(solved), int(independent), hints))
            streak = student["streak"] + 1 if solved else 0
            self.db.execute("UPDATE students SET streak = ?, best_streak = MAX(best_streak, ?) WHERE name = ?",
                            (streak, streak, name))
        return {"outcome": outcome, "level_before": before, "level_after": after}

    # ---------- what the page is allowed to see ----------

    def _model_help(self, row, language):
        return dict(self.db.execute(
            "SELECT kind, text FROM model_help WHERE history_n = ? AND language = ?",
            (row["n"], language)))

    def _transfer_status(self, row):
        if row["status"] == "open":
            return "open"
        if row["status"] == "solved":
            return "supported" if row["wrong_tries"] or self._hint_count(row) else "independent"
        return "not_solved"

    def _public(self, row, language):
        if not row or row["problem_id"] not in self.problems:
            return None
        problem = self.problems[row["problem_id"]]
        help_text = self._model_help(row, language)
        hints = [{"kind": kind, "text": help_text[kind]} for kind in HINT_STEPS if kind in help_text]
        is_open = row["status"] == "open"
        shown = {
            "id": problem["id"], "image": f"/image/{problem['id']}_{language}.png",
            "topic": row["topic"], "topic_name": NAMES[row["topic"]][language], "level": row["level"],
            "text_only": problem["text_only"], "status": row["status"], "closed": not is_open,
            "tries": row["wrong_tries"], "tried": list(row["tried"]), "eliminated": list(row["eliminated"]),
            "hints": hints, "next_hint": next((k for k in HINT_STEPS if k not in help_text), None) if is_open else None,
        }
        if row["tried"] and is_open:
            shown["coaching"] = help_text.get("wrong-" + row["tried"][-1])
        if row["status"] in ("solved", "revealed"):
            shown["label"] = problem["label"]
        if row["transfer_from"] is not None:
            shown["transfer"] = self._transfer_status(row)
        if self._transfer_target(row):
            shown["transfer_available"] = True
        return shown

    @_locked
    def coaching_context(self, name, problem_id, mode):
        student = self._student(name)
        row = self._current(name, problem_id)
        if student["finished"] or not row or row["problem_id"] not in self.problems:
            raise TutorError("no problem for this request")
        problem_id = row["problem_id"]
        problem = self.problems[problem_id]
        language = student["language"]
        saved = self._model_help(row, language)
        if row["status"] != "open":
            raise TutorError("no open problem for this request")
        if mode == "hint":
            kind = next((k for k in HINT_STEPS if k not in saved), None)
            if kind is None:
                raise TutorError("all hints have been used")
        elif mode == "feedback":
            if not row["tried"]:
                raise TutorError("no wrong choice to review")
            kind = "wrong-" + row["tried"][-1]
        else:
            kind = "chat"
        options = self.options.get((problem_id, language)) or {}
        if not isinstance(options, dict):
            options = {}
        earlier = [saved[k] for k in HINT_STEPS if k in saved]
        if row["tried"]:
            recent_feedback = saved.get("wrong-" + row["tried"][-1])
            if recent_feedback:
                earlier.append(recent_feedback)
        return {
            "name": name, "history_n": row["n"], "problem": problem_id, "status": row["status"],
            "topic": NAMES[row["topic"]][language], "language": language, "kind": kind, "cached": saved.get(kind),
            "exercise": clean(self.texts.get((problem_id, language), "")),
            "correct": problem["label"], "options": options, "diagram": not problem["text_only"],
            "tried": row["tried"], "earlier": earlier,
        }

    @_locked
    def save_coaching(self, context, text):
        """Commit an accepted model reply only if it still belongs to the current attempt."""
        name, language, kind = (context[k] for k in ("name", "language", "kind"))
        student = self._student(name)
        row = self._current(name, context["problem"])
        if (not row or student["finished"] or row["n"] != context["history_n"]
                or row["status"] != context["status"] or row["tried"] != context["tried"]
                or student["language"] != language):
            raise TutorError("this problem has changed")
        events = []
        with self.db:
            if kind != "chat":
                self.db.execute("INSERT OR IGNORE INTO model_help VALUES (?, ?, ?, ?)",
                                (row["n"], language, kind, text))
                text = self._model_help(row, language)[kind]
            if kind == "chat":
                self.db.execute("UPDATE history SET chat_help = 1 WHERE n = ?", (row["n"],))
            elif kind in HINT_STEPS and kind not in row["rungs"].split(","):
                rungs = ",".join(filter(None, (row["rungs"], kind)))
                self.db.execute("UPDATE history SET rungs = ? WHERE n = ?", (rungs, row["n"]))
                events.append(("hint", {"rung": HINT_STEPS.index(kind) + 1}))
        return {"text": text, "kind": kind, "events": events, **self.state(name)}

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
                "problem": None if student["finished"] else self._public(self._current(name), student["language"]),
                "profile": self._profile(name, student["language"])}

    # ---------- the steps of a session ----------

    @_locked
    def start(self, name, language="eng", topic=None, level=None, resume=False):
        """Create or resume a student. A returning student keeps their profile and open problem."""
        name = " ".join(str(name or "").split())[:40]
        if not name:
            raise TutorError("a name is needed")
        language = language if language in ("eng", "cat") else "eng"
        topic = topic if topic in TOPICS else None
        existing = self.db.execute("SELECT * FROM students WHERE name = ?", (name,)).fetchone()
        if existing and resume:
            with self.db:
                self.db.execute("UPDATE students SET language = ? WHERE name = ?", (language, name))
            return self.state(name)
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
        keep = (current and current["problem_id"] in self.problems and not existing["finished"]
                and (same_topic_choice or current["topic"] == topic or (topic is None and current["status"] == "open")))
        if not keep:
            if current and current["status"] == "open":
                self._close(current, "skipped")
            pid = self._serve(self._student(name), self._current(name))
            self.db.execute("UPDATE students SET finished = ? WHERE name = ?", (pid is None, name))
        self.db.commit()
        return self.state(name)

    @_locked
    def answer(self, name, choice, problem=None):
        """Check one answer. A wrong answer never comes back with the label."""
        row = self._current(name, problem)
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
    def reveal(self, name, problem=None):
        """Show the answer because the student asked. Counts as not solved."""
        row = self._current(name, problem)
        if not row or self._student(name)["finished"] or row["problem_id"] not in self.problems:
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
    def next(self, name, topic="keep", problem=None):
        """Move on. topic: "keep", a topic key, or None for mixed practice."""
        student = self._student(name)
        row = self._current(name, problem)
        if student["finished"]:
            return {**self.state(name), "events": [], "finished": True}
        if topic != "keep":
            self.db.execute("UPDATE students SET topic_choice = ? WHERE name = ?",
                            (topic if topic in TOPICS else None, name))
            student = self._student(name)
        if row and row["status"] == "open":
            self._close(row, "skipped")
        pid = self._serve(student, self._current(name), allow_transfer=topic == "keep" and row is not None)
        self.db.execute("UPDATE students SET finished = ? WHERE name = ?", (pid is None, name))
        self.db.commit()
        state = self.state(name)
        state["events"] = [("thinking", {"level": state["problem"]["level"]})] if pid else [("idle", {})]
        state["finished"] = pid is None
        return state

    # ---------- the teacher view ----------

    @_locked
    def students(self):
        return [{"name": r["name"], "language": r["language"], "problems": r["n"], "solved": r["solved"] or 0}
                for r in self.db.execute(
                    "SELECT s.name, s.language, COUNT(h.n) AS n, SUM(h.status = 'solved') AS solved "
                    "FROM students s LEFT JOIN history h ON h.name = s.name AND h.outcome IS NOT NULL "
                    "GROUP BY s.name ORDER BY s.created DESC")]

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
                   "hints": self._hint_count(r), "outcome": r["outcome"],
                   "seconds": round(r["ended"] - r["started"], 1) if r["ended"] else None} for r in closed[-15:]]
        transfers = self.db.execute(
            "SELECT * FROM history WHERE name = ? AND transfer_from IS NOT NULL AND outcome IS NOT NULL",
            (name,)).fetchall()
        return {"student": student["name"], "language": language, "profile": profile,
                "transfer": {"attempted": len(transfers),
                             "independent": sum(self._transfer_status(r) == "independent" for r in transfers)},
                "totals": {"attempts": attempts, "correct": correct,
                           "accuracy": round(correct / attempts, 3) if attempts else None,
                           "hints": sum(p["hints"] for p in profile),
                           "streak": student["streak"], "best_streak": student["best_streak"]},
                "practise_next": self._advice(profile, language, weak, strong), "recent": recent[::-1],
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

    def _advice(self, profile, language, weak, strong):
        """Explain the practice recommendations shown alongside this text."""
        tried = {p["topic"] for p in profile}
        bank_topics = {topic for topic, _ in self.pool}
        untried = [NAMES[t][language] for t in TOPICS if t in bank_topics and t not in tried]
        if not weak:
            if any(p["attempts"] >= 2 for p in profile):
                return {"eng": "Choose another topic to find more practice at your level.",
                        "cat": "Tria un altre tema per trobar més pràctica al teu nivell."}[language]
            return {"eng": "Do a few more problems to get a practice recommendation.",
                    "cat": "Fes uns quants problemes més per obtenir una recomanació de pràctica."}[language]
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
