"""Run with python -m unittest experiments.test_text_solver."""
import json
import tempfile
import threading
import time
import unittest
from fractions import Fraction
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from copilot import pipeline, reasoner
from copilot.arithmetic import calculate, numeric_options
from copilot.rules import answer as exact_answer
from copilot.text_solver import decode, infer
from copilot.train_match import LexicalMatch
from copilot.translations import TranslationBatch


def problem(text, values=("1", "2", "3", "4", "5"), **metadata):
    return {"text": text, "options": dict(zip("ABCDE", values)), "language": "eng", **metadata}


class TextSolverTest(unittest.TestCase):
    def test_exact_calculation_only_when_unambiguous(self):
        row = problem("Calculate 2 * (3 + 1) / 2?")
        self.assertEqual(exact_answer(row), "D")
        self.assertEqual(calculate("gcd(12, 9) + 1/3"), Fraction(10, 3))
        self.assertEqual(calculate("0.100000000000000001 - 0.1"), Fraction(1, 10**18))
        self.assertIsNone(calculate("9**(9**9)"))
        self.assertIsNone(calculate("open('not a calculation')"))
        self.assertIsNone(calculate("1/0"))
        self.assertEqual(decode("Calc: 4", numeric_options(row)), (None, "fallback"))
        self.assertEqual(decode("Calc: 4\n", numeric_options(row)), ("D", "calculated"))
        self.assertEqual(decode("**Resposta**: B", numeric_options(row)), ("B", "ok"))
        row["options"]["A"] = "4"
        self.assertIsNone(exact_answer(row))

    def test_neighbour_uses_answer_value_and_excludes_same_id(self):
        question = "What is the remainder when the number 13 is divided by 5?"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            clean = problem("14. " + question, id="train", label="C", text_only=True)
            invalid = problem(question, id="bad", label="A", text_only=True)
            invalid["options"]["D"] = " "
            path.write_text("\n".join(json.dumps(x) for x in (clean, invalid)), encoding="utf-8")
            index = LexicalMatch(path)
            query = problem(question, ("3", "2", "1", "4", "5"), id="eval")
            self.assertEqual(index.answer(query), "A")
            self.assertIsNone(index.answer(query, exclude_id="train"))
            query["options"]["E"] = "6"
            self.assertIsNone(index.answer(query))

    def test_median_rule_is_used_even_when_it_returns_d(self):
        row = problem("A question", ("5", "1", "4", "3", "2"), id="median-d", text_only=True)
        with patch.object(pipeline, "TEXT_TRAIN", ""), \
                patch.object(pipeline, "TEXT_MEDIAN_BYPASS", True), \
                patch.object(pipeline, "ANSWER_MODE", "text"), \
                patch("copilot.text_solver.infer", side_effect=AssertionError("model called")):
            result = pipeline.solve(row, time_limit=60)
        self.assertEqual((result["answer"], result["status"]), ("D", "prior"))

    def test_model_only_route_skips_all_shortcuts(self):
        row = problem("A question", ("5", "1", "4", "3", "2"), id="model-only")
        with patch.object(pipeline, "TEXT_ROUTE", "model"), \
                patch.object(pipeline, "TEXT_TRAIN", ""), \
                patch.object(pipeline, "TEXT_PROFILE", "shots"), \
                patch("copilot.text_solver.infer",
                      return_value=("C", "ok", "Answer: C", {})) as infer:
            result = pipeline.solve(row, time_limit=60)
        self.assertEqual((result["answer"], result["status"]), ("C", "ok"))
        infer.assert_called_once()

    def test_numeric_route_uses_ranker_without_model(self):
        row = problem("A numeric question", ("5", "1", "4", "3", "2"),
                      id="numeric-route", text_only=True)
        with patch.object(pipeline, "TEXT_TRAIN", ""), \
                patch.object(pipeline, "TEXT_NUMERIC_ROUTE", "ranker"), \
                patch.object(pipeline, "TEXT_MEDIAN_BYPASS", True), \
                patch.object(pipeline, "ANSWER_MODE", "text"), \
                patch("copilot.text_solver.infer", side_effect=AssertionError("model called")):
            result = pipeline.solve(row, time_limit=60)
        self.assertEqual(result["status"], "ranker")
        self.assertIn(result["answer"], "ABCDE")

    def test_budget_mode_uses_median_rule_even_when_it_returns_d(self):
        row = problem("A question", ("5", "1", "4", "3", "2"))
        with patch.object(reasoner, "stream_reasoning",
                          side_effect=AssertionError("model called")):
            letter, status, text, timings = reasoner.answer(row, deadline=time.perf_counter() + 10)
        self.assertEqual((letter, status, text, timings), ("D", "prior", "", {}))

    def test_deadline_cancels_even_when_server_keeps_writing(self):
        started = threading.Event()
        stopped = threading.Event()

        class SlowServer(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                self.send_response(200)
                self.end_headers()
                started.set()
                try:
                    for _ in range(200):
                        self.wfile.write(b'data: {"choices":[{"delta":{"content":"."}}]}\n\n')
                        self.wfile.flush()
                        time.sleep(.02)
                except OSError:
                    stopped.set()

        with ThreadingHTTPServer(("127.0.0.1", 0), SlowServer) as server:
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                start = time.perf_counter()
                url = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
                letter, status, _, _ = infer(problem("Test"), start + .25, url)
                self.assertTrue(started.is_set())
                self.assertIsNone(letter)
                self.assertEqual(status, "timeout")
                self.assertLess(time.perf_counter() - start, .75)
                self.assertTrue(stopped.wait(.5))
            finally:
                server.shutdown()
                worker.join()

    def test_batch_reuses_value_not_letter_and_only_with_same_options(self):
        source = problem("Question in English", id="one", text_only=True)
        translated = problem("Longer translated question", ("3", "2", "1", "4", "5"),
                             id="one", text_only=True)
        translated["language"] = "cat"
        incompatible = problem("Not the same options", ("9", "2", "1", "4", "5"),
                               id="one", text_only=True)
        calls = []

        def solve(row, **kwargs):
            calls.append(row)
            return {"answer": "C", "status": "ok", "seconds": 1.0}

        batch = TranslationBatch([translated, source, incompatible], solve)
        self.assertEqual(batch(translated)["answer"], "A")
        self.assertEqual(batch(source)["status"], "reused_ok")
        self.assertEqual(batch(incompatible)["answer"], "C")
        self.assertEqual(calls, [source, incompatible])


if __name__ == "__main__":
    unittest.main()
