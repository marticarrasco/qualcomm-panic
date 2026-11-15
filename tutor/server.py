#!/usr/bin/env python3
"""The maths tutor: a web page that guides a student without giving answers away.

    python3 tutor/server.py                      # http://127.0.0.1:8000
    python3 tutor/server.py --host 0.0.0.0       # reachable from phones and the LED app
    python3 tutor/server.py --no-llm             # practise without model assistance

Build the bank first with tutor/build_bank.py. Standard library only.

API (JSON). The answer letter of the current problem is only ever sent after
the student has solved it or asked for it with /api/reveal.
    GET  /api/info                          topics, levels, problem counts
    POST /api/start   {name, language, topic?, level?, resume?}
    POST /api/answer  {name, choice, problem?}        choice is A-E
    POST /api/hint    {name, problem?}         generate the next hint
    POST /api/feedback {name, problem?}        generate feedback on the latest wrong choice
    POST /api/reveal  {name, problem?}                show the answer (counts as not solved)
    POST /api/next    {name, topic?, problem?}        next problem; topic "" means mixed
    POST /api/chat    {name, problem, message, last_hint?}
    GET  /api/chat?job=                    pending or completed conversation turn
    GET  /api/state?name=                   where the student is now
    GET  /api/summary?name=&language=       teacher view for one student
    GET  /api/students                      everyone who has a profile
    GET  /api/matrix                        last event for the LED matrix (see matrix.py)
    GET  /image/<id>_<language>.png         the problem picture
    GET  /static/<file>                     fonts and other page assets from tutor/static
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
from tutor.chat import Chat, validate_message  # noqa: E402
from tutor.local_model import MODEL_URL, start_windows_model  # noqa: E402
from tutor.tutor import Tutor, TutorError  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
IMAGE_KEY = re.compile(r"^(.+)_(eng|cat)\.png$")
# Fonts and page assets in tutor/static. Only plain file names in known folders are served.
STATIC_KEY = re.compile(
    r"^(?:[A-Za-z0-9_-]+\.(?:ttf|woff2|svg|css|js|txt)|"
    r"fonts/[A-Za-z0-9_-]+\.(?:ttf|woff2)|logos/[A-Za-z0-9_-]+\.(?:svg|png))$"
)
STATIC_TYPES = {"ttf": "font/ttf", "woff2": "font/woff2", "svg": "image/svg+xml",
                "png": "image/png",
                "css": "text/css; charset=utf-8", "js": "text/javascript; charset=utf-8",
                "txt": "text/plain; charset=utf-8"}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank", default="data/tutor.db", help="problem bank from build_bank.py")
    p.add_argument("--students", default="data/tutor_students.db", help="student database")
    p.add_argument("--images", help="folder with <id>_<language>.png")
    p.add_argument("--llm", help=f"chat completions endpoint (default: {MODEL_URL})")
    p.add_argument("--vision", action="store_true", help="send the figure to a multimodal llama server")
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


def make_handler(tutor, images, quiet=False, chat=None, vision=False):
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
                if path.startswith("/static/"):
                    key = urllib.parse.unquote(path[len("/static/"):])
                    file = os.path.join(HERE, "static", key)
                    if STATIC_KEY.match(key) and os.path.isfile(file):
                        with open(file, "rb") as f:
                            return self.send(200, f.read(), STATIC_TYPES[key.rsplit(".", 1)[1]], cache="max-age=86400")
                if path == "/api/info":
                    return self.send(200, tutor.info())
                if path == "/api/matrix":
                    return self.send(200, matrix.last())
                if path == "/api/students":
                    return self.send(200, tutor.students())
                if path == "/api/chat" and chat:
                    return self.send(200, emit(chat.status(params.get("job", ""))))
                name = params.get("name", "")
                if path == "/api/state":
                    return self.send(200, tutor.state(name))
                if path == "/api/summary":
                    return self.send(200, tutor.summary(name, params.get("language")))
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
                size = int(self.headers.get("Content-Length", 0))
                if not 0 <= size <= 4096:
                    return self.send(413, {"error": "request too large"})
                body = json.loads(self.rfile.read(size) or b"{}")
                if not isinstance(body, dict):
                    return self.send(400, {"error": "expected a JSON object"})
                name = " ".join(str(body.get("name") or "").split())[:40]
                if self.path == "/api/start":
                    state = tutor.start(name, body.get("language", "eng"), body.get("topic") or None, body.get("level"), resume=body.get("resume") is True)
                    if state["problem"]:
                        matrix.show("thinking", level=state["problem"]["level"])
                    return self.send(200, state)
                problem = body.get("problem")
                if self.path == "/api/answer":
                    return self.send(200, emit(tutor.answer(name, str(body.get("choice", "")).upper(), problem)))
                if self.path == "/api/reveal":
                    return self.send(200, emit(tutor.reveal(name, problem)))
                if self.path == "/api/next":
                    topic = body["topic"] or None if "topic" in body else "keep"
                    return self.send(200, emit(tutor.next(name, topic, problem)))
                mode = self.path.removeprefix("/api/")
                if chat and self.path in ("/api/chat", "/api/hint", "/api/feedback"):
                    message = validate_message(body.get("message")) if mode == "chat" else ""
                    context = tutor.coaching_context(name, problem, mode)
                    image = None
                    if vision and context["diagram"] and chat.url and not context["cached"]:
                        file = os.path.join(images, f"{context['problem']}_{context['language']}.png")
                        try:
                            with open(file, "rb") as source:
                                data = source.read()
                            image = image_type(data), data
                        except OSError:
                            pass
                    return self.send(200, emit(chat.start(context, message, body.get("last_hint"), image)))
            except TutorError as error:
                return self.send(400, {"error": str(error)})
            except (ValueError, TypeError) as error:
                return self.send(400, {"error": f"bad request: {error}"})
            return self.send(404, {"error": "not found"})

    return Handler


def images_folder(path):
    if path:
        return path
    return "data/full_noocr/images" if os.path.isdir("data/full_noocr/images") else "data/images"


def make_server(args):
    """Build the server without starting it (simulate.py uses this too)."""
    url = None if args.no_llm else (args.llm or MODEL_URL)
    tutor = Tutor(args.bank, args.students, model_enabled=bool(url))
    handler = make_handler(tutor, images_folder(args.images), args.quiet,
                           Chat(url, tutor.save_coaching), args.vision)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    return server, tutor


def main():
    args = parse_args()
    os.chdir(ROOT)
    if not os.path.exists(args.bank):
        sys.exit(f"{args.bank} is missing: run python3 tutor/build_bank.py --lexical first")
    server, tutor = make_server(args)
    images = images_folder(args.images)
    print(f"{len(tutor.problems)} problems in the bank")
    missing = 0
    for language in ("cat", "eng"):
        found = sum(os.path.isfile(os.path.join(images, f"{pid}_{language}.png")) for pid in tutor.problems)
        print(f"{language}: {found}/{len(tutor.problems)} images in {images}")
        missing += len(tutor.problems) - found
    if missing:
        server.server_close()
        tutor.db.close()
        sys.exit(f"{missing} exercise images are missing. Prepare the bilingual images or set --images.")
    print(f"Coaching model: {'disabled' if args.no_llm else (args.llm or MODEL_URL)}")
    browser_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    model_process = None
    try:
        if os.name == "nt" and not (args.no_llm or args.llm or args.vision):
            model_process = start_windows_model("data/tutor_model.log")
        print(f"Open http://{browser_host}:{server.server_port} (Ctrl+C to stop)", flush=True)
        matrix.show("idle")
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        tutor.db.close()
        if model_process and model_process.poll() is None:
            model_process.terminate()
            model_process.wait(timeout=5)


if __name__ == "__main__":
    main()
