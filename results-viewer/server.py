#!/usr/bin/env python3
"""Result viewer: browse the CSV files written by eval.py in results/.

    python3 results-viewer/server.py
    python3 results-viewer/server.py --port 8001

Then open http://127.0.0.1:8001 in a browser. Standard library only.
You can also open results-viewer/index.html directly in a browser and drop a
CSV file onto it, in which case no server is needed.
"""
import argparse
import json
import os
import socket
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS = os.path.normpath(os.path.join(HERE, "..", "results"))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=DEFAULT_RESULTS, help="folder with the eval CSVs")
    p.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to open it to your network")
    p.add_argument("--port", type=int, default=8001)
    return p.parse_args()


def make_handler(results_dir):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, code, body, content_type):
            data = body if isinstance(body, bytes) else body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = self.path.split("?", 1)[0].split("#", 1)[0]
            if path in ("/", "/index.html"):
                self.send_index()
            elif path == "/api/results":
                self.send_list()
            elif path.startswith("/results/"):
                self.send_csv(os.path.basename(path[len("/results/"):]))
            else:
                self.send(404, "not found", "text/plain; charset=utf-8")

        def send_index(self):
            with open(os.path.join(HERE, "index.html"), "rb") as fh:
                self.send(200, fh.read(), "text/html; charset=utf-8")

        def send_list(self):
            files = []
            try:
                for name in os.listdir(results_dir):
                    if not name.lower().endswith(".csv"):
                        continue
                    info = os.stat(os.path.join(results_dir, name))
                    files.append({"name": name, "kb": info.st_size / 1024, "mtime": info.st_mtime})
            except FileNotFoundError:
                pass
            files.sort(key=lambda f: (-f["mtime"], f["name"]))
            self.send(200, json.dumps({"files": files}), "application/json; charset=utf-8")

        def send_csv(self, name):
            path = os.path.join(results_dir, name)
            if not os.path.isfile(path):
                self.send(404, "not found", "text/plain; charset=utf-8")
                return
            with open(path, "rb") as fh:
                self.send(200, fh.read(), "text/csv; charset=utf-8")

    return Handler


def already_running(host, port):
    """SO_REUSEADDR lets a second bind succeed on Windows, so probe instead."""
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def main():
    args = parse_args()
    if already_running(args.host, args.port):
        print(f"A viewer is already running at http://{args.host}:{args.port}")
        print("Just open that address in the browser, or use --port for another one.")
        raise SystemExit(1)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.results))
    print(f"Results from: {args.results}")
    print(f"Open http://{args.host}:{args.port} in a browser")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
