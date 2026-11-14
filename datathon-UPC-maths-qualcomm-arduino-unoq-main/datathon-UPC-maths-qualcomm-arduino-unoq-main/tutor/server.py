#!/usr/bin/env python3
"""The maths tutor: a web page that guides a student without giving answers away.

    python3 tutor/server.py                      # http://127.0.0.1:8000
    python3 tutor/server.py --host 0.0.0.0       # reachable from phones and the LED app
    python3 tutor/server.py --no-llm             # leave out the optional model sentence

Build the bank first with tutor/build_bank.py. Standard library only.

API (JSON). The answer letter of the current problem is only ever sent after
the student has solved it or asked for it with /api/reveal.
    GET  /api/info                          topics, levels, problem counts
    POST /api/start   {name, language, topic?, level?}
    POST /api/answer  {name, choice}        choice is A-E
    POST /api/hint    {name}                next rung of the hint ladder
    POST /api/example {name}                answer of the example problem in rung 2
    POST /api/reveal  {name}                show the answer (counts as not solved)
    POST /api/next    {name, topic?}        next problem; topic "" means mixed
    GET  /api/state?name=                   where the student is now
    GET  /api/summary?name=&language=       teacher view for one student
    GET  /api/students                      everyone who has a profile
    GET  /api/matrix                        last event for the LED matrix (see matrix.py)
    GET  /image/<id>_<language>.png         the problem picture
"""
import argparse
import json
import os
import re
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tutor import matrix  # noqa: E402
from tutor.tutor import Tutor, TutorError  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
IMAGE_KEY = re.compile(r"^(.+)_(eng|cat)\.png$")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank", default="data/tutor.db", help="problem bank from build_bank.py")
    p.add_argument("--students", default="data/tutor_students.db", help="student profiles (created if missing)")
    p.add_argument("--images", default="data/full_noocr/images", help="folder with <id>_<language>.png")
    p.add_argument("--llm", default="http://127.0.0.1:8081/v1/chat/completions", help="language model for the optional rung 4")
    p.add_argument("--no-llm", action="store_true", help="never call the language model")
    p.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to open it to your network")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--quiet", action="store_true", help="do not log each request")
    return p.parse_args(argv)


def image_type(data):
    """The files are named .png but hold JPEG bytes, so look at the bytes."""
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"


def make_handler(tutor, images, quiet=False):
    def emit(result):
        """Send the events of one step to the LED matrix hook, and keep them out of the reply."""
        for event, info in result.pop("events", []):
            matrix.show(event, **info)
        return result

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            if not quiet:
                sys.stderr.write("  " + fmt % args + "\n")

        def send(self, code, body, kind="application/json", cache="no-store"):
            data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", cache)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path, _, query = self.path.partition("?")
            params = {k: v[0] for k, v in urllib.parse.parse_qs(query).items()}
            try:
                if path == "/":
                    with open(os.path.join(HERE, "index.html"), "rb") as f:
                        return self.send(200, f.read(), "text/html; charset=utf-8")
                if path == "/api/info":
                    return self.send(200, tutor.info())
                if path == "/api/matrix":
                    return self.send(200, matrix.last())
                if path == "/api/students":
                    return self.send(200, tutor.students())
                if path == "/api/state":
                    return self.send(200, tutor.state(params.get("name", "")))
                if path == "/api/summary":
                    return self.send(200, tutor.summary(params.get("name", ""), params.get("language")))
                if path.startswith("/image/"):
                    match = IMAGE_KEY.match(urllib.parse.unquote(path[len("/image/"):]))
                    # Only known problem ids reach the disk, so a request can't name any other file
                    if match and match.group(1) in tutor.problems:
                        file = os.path.join(images, f"{match.group(1)}_{match.group(2)}.png")
                        if os.path.exists(file):
                            with open(file, "rb") as f:
                                data = f.read()
                            return self.send(200, data, image_type(data), cache="max-age=86400")
            except TutorError as error:
                return self.send(400, {"error": str(error)})
            return self.send(404, {"error": "not found"})

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                name = " ".join(str(body.get("name") or "").split())[:40]
                if self.path == "/api/start":
                    state = tutor.start(name, body.get("language", "eng"), body.get("topic") or None, body.get("level"))
                    if state["problem"]:
                        matrix.show("thinking", level=state["problem"]["level"])
                    return self.send(200, state)
                if self.path == "/api/answer":
                    return self.send(200, emit(tutor.answer(name, str(body.get("choice", "")).upper())))
                if self.path == "/api/hint":
                    return self.send(200, emit(tutor.hint(name)))
                if self.path == "/api/example":
                    return self.send(200, emit(tutor.example_answer(name)))
                if self.path == "/api/reveal":
                    return self.send(200, emit(tutor.reveal(name)))
                if self.path == "/api/next":
                    topic = body["topic"] or None if "topic" in body else "keep"
                    return self.send(200, emit(tutor.next(name, topic)))
            except TutorError as error:
                return self.send(400, {"error": str(error)})
            except (ValueError, TypeError) as error:
                return self.send(400, {"error": f"bad request: {error}"})
            return self.send(404, {"error": "not found"})

    return Handler


def make_server(args):
    """Build the server without starting it (simulate.py uses this too)."""
    tutor = Tutor(args.bank, args.students, llm_url=None if args.no_llm else args.llm)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(tutor, args.images, args.quiet))
    return server, tutor


def main():
    args = parse_args()
    os.chdir(ROOT)
    if not os.path.exists(args.bank):
        sys.exit(f"{args.bank} is missing: run python3 tutor/build_bank.py first")
    server, tutor = make_server(args)
    found = sum(os.path.exists(os.path.join(args.images, f"{pid}_eng.png")) for pid in tutor.problems)
    print(f"{len(tutor.problems)} problems in the bank, {found} with an English image in {args.images}")
    print(f"Optional model sentence (rung 4): {'off' if args.no_llm else args.llm}")
    print(f"Open http://{args.host}:{args.port} (Ctrl+C to stop)", flush=True)
    matrix.show("idle")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
