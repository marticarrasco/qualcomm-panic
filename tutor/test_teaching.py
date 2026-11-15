"""Regression tests for model-generated feedback and attempt state."""
import json
import sqlite3
import tempfile
import threading
import time
import unittest
from array import array
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from tutor.build_bank import SCHEMA
from tutor.chat import Chat
from tutor.server import make_server, parse_args
from tutor.simulate import call


HINT_ENG = "Draw one mark for each row and compare the groups."
HINT_CAT = "Dibuixa les files i marca la quantitat que es repeteix."


class FakeModel(BaseHTTPRequestHandler):
    def do_POST(self):
        self.server.prompts.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        if self.server.gate is not None:
            self.server.gate.wait(3)
        text = self.server.replies.pop(0)
        result = {"choices": [{"finish_reason": "stop", "message": {"content": text}}]}
        self.send_response(200)
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())

    def log_message(self, *args):
        pass


class TeachingTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        bank = Path(folder.name) / "bank.db"

        with sqlite3.connect(bank) as conn:
            conn.executescript(SCHEMA)
            for pid in ("rows", "tiles"):
                conn.execute("INSERT INTO problems VALUES (?, 1, 2026, 1, 1, 'C', 'arithmetic')", (pid,))
                conn.execute("INSERT INTO vectors VALUES (?, ?)", (pid, array("f", [1]).tobytes()))
                for lang in ("eng", "cat"):
                    question = "Three rows have four tiles each. How many tiles are there?"
                    if lang == "cat":
                        question = "Tres files tenen quatre rajoles cadascuna. Quantes rajoles hi ha en total?"
                    conn.execute("INSERT INTO texts VALUES (?, ?, ?)", (pid, lang, question))
                    conn.execute("INSERT INTO options VALUES (?, ?, ?)", (pid, lang, '{"A": "7", "C": "12"}'))
            for first, other in (("rows", "tiles"), ("tiles", "rows")):
                conn.execute("INSERT INTO neighbours VALUES (?, 'same_level', 0, ?, 0.9, 1)", (first, other))
        conn.close()

        self.model = ThreadingHTTPServer(("127.0.0.1", 0), FakeModel)
        self.model.replies = []
        self.model.prompts = []
        self.model.gate = None
        self.run_server(self.model)
        self.args = parse_args([
            "--bank", str(bank), "--students", str(Path(folder.name) / "students.db"),
            "--port", "0", "--quiet", "--llm", f"http://127.0.0.1:{self.model.server_port}/v1/chat/completions",
        ])
        self.start_tutor()

    def run_server(self, server):
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

    def start_tutor(self):
        server, tutor = make_server(self.args)
        self.addCleanup(tutor.db.close)
        self.run_server(server)
        self.base = f"http://127.0.0.1:{server.server_port}"

    def api(self, path, data=None):
        return json.loads(call(self.base, path, data)[0])

    def start(self, language="cat"):
        return self.api("/api/start", {"name": "Pupil", "language": language, "topic": "arithmetic"})["problem"]

    def post(self, route, pid, **data):
        return self.api("/api/" + route, {"name": "Pupil", "problem": pid, **data})

    def finish_job(self, request):
        for _ in range(200):
            result = self.api("/api/chat?job=" + request["job"])
            if not result.get("pending"):
                return result
            time.sleep(0.015)
        self.fail("model job never completed")

    def test_model_generates_hints_and_preserves_their_language(self):
        problem = self.start()
        apply_hint = "Agrupa les marques de cada fila i compara la mida de cada grup."
        check_hint = "Comprova que has comptat totes les files del dibuix."
        self.model.replies += [HINT_CAT, apply_hint, check_hint, HINT_ENG]
        result = self.finish_job(self.post("hint", problem["id"]))
        self.assertEqual(result["problem"]["hints"], [{"kind": "orient", "text": HINT_CAT}])
        self.assertEqual(result["problem"]["next_hint"], "apply")
        self.assertNotIn("label", result["problem"])
        prompt = self.model.prompts[0]["messages"][1]["content"]
        self.assertNotIn("Clau oficial", prompt)
        for kind, text in (("apply", apply_hint), ("check", check_hint)):
            result = self.finish_job(self.post("hint", problem["id"]))
            self.assertEqual(result["problem"]["hints"][-1], {"kind": kind, "text": text})
        self.assertIsNone(result["problem"]["next_hint"])
        with self.assertRaises(HTTPError):
            self.post("hint", problem["id"])
        self.assertIn(HINT_CAT, self.model.prompts[1]["messages"][1]["content"])

        english = self.start("eng")
        self.assertEqual(english["id"], problem["id"])
        self.assertEqual(english["hints"], [])
        result = self.finish_job(self.post("hint", english["id"]))
        self.assertEqual(result["problem"]["hints"][0]["text"], HINT_ENG)
        self.assertEqual(self.start()["hints"][0]["text"], HINT_CAT)

        self.start_tutor()
        self.assertEqual(self.start()["hints"][0]["text"], HINT_CAT)

    def test_wrong_choice_correction_and_follow_up(self):
        problem = self.start("eng")
        wrong = self.post("answer", problem["id"], choice="A")
        self.assertNotIn("label", wrong["problem"])
        self.model.replies += [HINT_ENG]
        feedback = self.finish_job(self.post("feedback", problem["id"]))
        self.assertEqual(feedback["problem"]["coaching"], HINT_ENG)
        prompt = self.model.prompts[0]["messages"][1]["content"]
        self.assertIn("Incorrect result proposed by the pupil: 7", prompt)

        solved = self.post("answer", problem["id"], choice="C")
        self.assertTrue(solved["problem"]["transfer_available"])
        follow_up = self.post("next", problem["id"])["problem"]
        self.assertEqual(follow_up["transfer"], "open")
        self.assertNotEqual(follow_up["id"], problem["id"])
        with self.assertRaises(HTTPError):
            self.post("answer", problem["id"], choice="C")
        result = self.post("answer", follow_up["id"], choice="C")
        self.assertEqual(result["problem"]["transfer"], "independent")
        self.assertEqual(self.api("/api/summary?name=Pupil")["transfer"], {"attempted": 1, "independent": 1})
        self.assertIsNone(self.post("next", follow_up["id"])["problem"])
        self.assertIsNone(self.api("/api/state?name=Pupil")["problem"])

    def test_model_retries_a_leak_and_rejection_does_not_consume_hint(self):
        problem = self.start("eng")
        self.model.replies += ["The answer is option C.", "Now you should pick C) here."]
        rejected = self.finish_job(self.post("hint", problem["id"]))
        self.assertEqual(rejected, {"model_error": "rejected"})
        state = self.api("/api/state?name=Pupil")
        self.assertEqual(state["problem"]["next_hint"], "orient")
        self.assertEqual(state["problem"]["hints"], [])

        self.model.replies += ["The answer is twelve tiles altogether.", HINT_ENG]
        saved = self.finish_job(self.post("hint", problem["id"]))
        self.assertEqual(saved["text"], HINT_ENG)
        self.assertEqual(len(self.model.prompts), 4)

    def test_finished_problem_rejects_further_model_requests(self):
        problem = self.start("eng")
        self.post("answer", problem["id"], choice="C")
        with self.assertRaises(HTTPError):
            self.post("hint", problem["id"])
        self.assertEqual(len(self.model.prompts), 0)

    def test_connection_refused_reports_the_model_is_offline(self):
        context = {"language": "eng", "kind": "orient", "diagram": False, "cached": None,
                   "topic": "Arithmetic", "exercise": "Count the tiles in three rows.", "earlier": [], "tried": ""}
        chat = Chat("http://localhost:8081/v1/chat/completions", None)
        with patch("tutor.chat.urllib.request.urlopen", side_effect=URLError(ConnectionRefusedError())):
            chat._run("test-job", context, "", "", None)
        self.assertEqual(chat.status("test-job"), {"model_error": "offline"})

    def test_late_model_reply_does_not_change_the_next_attempt(self):
        problem = self.start("eng")
        self.model.replies += [HINT_ENG]
        self.model.gate = threading.Event()
        request = self.post("chat", problem["id"], message="What can I draw?")
        following = self.post("next", problem["id"])["problem"]
        self.model.gate.set()
        self.assertEqual(self.finish_job(request), {"model_error": "stale"})
        self.assertEqual(self.api("/api/state?name=Pupil")["problem"]["id"], following["id"])

    def test_chat_help_counts_only_when_returned(self):
        problem = self.start("eng")
        self.model.replies += [HINT_ENG]
        result = self.finish_job(self.post("chat", problem["id"], message="What should I draw?"))
        self.assertEqual(result["text"], HINT_ENG)
        answered = self.post("answer", problem["id"], choice="C")
        self.assertEqual(answered["profile"][0]["clean"], 0)
        self.assertEqual(answered["profile"][0]["hints"], 1)
