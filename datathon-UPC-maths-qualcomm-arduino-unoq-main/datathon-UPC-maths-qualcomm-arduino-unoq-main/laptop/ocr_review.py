#!/usr/bin/env python3
"""Review OCR options for M3Kang problems in a local browser.

Example:
    .venv\\Scripts\\python.exe laptop\\ocr_review.py \\
        --parquet data\\textonly_100.parquet

The server binds to 127.0.0.1 by default. Review state and any edits are kept
in data/ocr_review.json. Use the page's export button to write reviewed copies
of train.jsonl and val.jsonl without overwriting the originals.
"""
import argparse
import base64
import json
import os
import sys
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
PAGE = Path(__file__).with_name("ocr_review.html")
LETTERS = "ABCDE"


def key_of(row):
    return f"{row['id']}|{row.get('language', 'eng')}"


def load_jsonl(paths):
    result = {}
    for path in paths:
        with open(path, encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    row = json.loads(line)
                    result[key_of(row)] = row
    return result


def options_of(value):
    if not isinstance(value, dict):
        raise ValueError("options must be an object")
    cleaned = {}
    for letter in LETTERS:
        option = value.get(letter, "")
        if not isinstance(option, str):
            raise ValueError(f"option {letter} must be text")
        option = option.strip()
        if option:
            cleaned[letter] = option
    return cleaned


class ReviewApp:
    def __init__(self, parquet_path, input_paths, state_path):
        import pyarrow.parquet as pq

        self.state_path = state_path
        self.problems = load_jsonl(input_paths)
        self.images = {}
        parquet = pq.read_table(parquet_path, columns=["id", "language", "image"])
        for row in parquet.to_pylist():
            key = f"{row['id']}|{row.get('language') or 'eng'}"
            image = row["image"] or {}
            if key in self.problems and image.get("bytes"):
                self.images[key] = image["bytes"]
        self.order = [key for key in self.problems if key in self.images]
        self.reviews = {"version": 1, "reviews": {}}
        if state_path.exists():
            with open(state_path, encoding="utf-8") as saved:
                loaded = json.load(saved)
            if isinstance(loaded, dict) and isinstance(loaded.get("reviews"), dict):
                self.reviews = loaded
        if not self.order:
            raise SystemExit("No input problems with images match this Parquet file.")

    def review_for(self, key):
        saved = self.reviews["reviews"].get(key, {})
        options = saved.get("options", self.problems[key].get("options") or {})
        return {
            "reviewed": bool(saved.get("reviewed", False)),
            "corrected": bool(saved.get("corrected", False)),
            "options": options_of(options),
            "updated": saved.get("updated"),
        }

    def listing(self):
        items = []
        for i, key in enumerate(self.order):
            row = self.problems[key]
            review = self.review_for(key)
            items.append({
                "key": key,
                "id": row["id"],
                "language": row.get("language", "eng"),
                "text_only": row.get("text_only", False),
                "reviewed": review["reviewed"],
                "corrected": review["corrected"],
                "has_options": bool(review["options"]),
                "number": i + 1,
            })
        return items

    def public_problem(self, key):
        row = self.problems[key]
        review = self.review_for(key)
        raw = self.images[key]
        if raw.startswith(b"\x89PNG"):
            mime = "image/png"
        elif raw.startswith(b"\xff\xd8"):
            mime = "image/jpeg"
        elif raw.startswith((b"GIF87a", b"GIF89a")):
            mime = "image/gif"
        else:
            mime = "application/octet-stream"
        return {
            "key": key,
            "id": row["id"],
            "language": row.get("language", "eng"),
            "level": row.get("level"),
            "year": row.get("year"),
            "text_only": row.get("text_only", False),
            "text": row.get("text", ""),
            "options": review["options"],
            "original_options": options_of(self.problems[key].get("options") or {}),
            "reviewed": review["reviewed"],
            "corrected": review["corrected"],
            "image": f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}",
        }

    def save(self, key, options):
        original = options_of(self.problems[key].get("options") or {})
        edited = options_of(options)
        self.reviews["reviews"][key] = {
            "reviewed": True,
            "corrected": edited != original,
            "options": edited,
            "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self.write_state()
        return self.review_for(key)

    def mark_pending(self, key, options):
        self.reviews["reviews"][key] = {
            "reviewed": False,
            "corrected": options_of(options) != options_of(self.problems[key].get("options") or {}),
            "options": options_of(options),
            "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self.write_state()
        return self.review_for(key)

    def write_state(self):
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        with open(temp, "w", encoding="utf-8", newline="\n") as target:
            json.dump(self.reviews, target, ensure_ascii=False, indent=2)
            target.write("\n")
        os.replace(temp, self.state_path)

    def export(self):
        outputs = []
        for source in self.input_paths:
            target = source.with_name(f"{source.stem}_reviewed{source.suffix}")
            count = 0
            with open(source, encoding="utf-8") as src, open(target, "w", encoding="utf-8", newline="\n") as dst:
                for line in src:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    saved = self.reviews["reviews"].get(key_of(row), {})
                    if saved.get("reviewed"):
                        row["options"] = options_of(saved.get("options", {}))
                        count += 1
                    dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            outputs.append({"file": str(target), "reviewed": count})
        return {"files": outputs}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", default="data/textonly_100.parquet",
                        help="Parquet file containing the problem images")
    parser.add_argument("--input", nargs="+", default=["data/train.jsonl", "data/val.jsonl"],
                        help="JSONL files containing the OCR output")
    parser.add_argument("--state", default="data/ocr_review.json",
                        help="where review status and corrections are saved")
    parser.add_argument("--host", default="127.0.0.1",
                        help="local only by default; use 0.0.0.0 only on a trusted network")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def main():
    args = parse_args()
    os.chdir(ROOT)
    input_paths = [Path(p).resolve() for p in args.input]
    app = ReviewApp(Path(args.parquet).resolve(), input_paths, Path(args.state).resolve())
    app.input_paths = input_paths

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *values):
            sys.stderr.write("  " + fmt % values + "\n")

        def send(self, code, body, content_type="application/json; charset=utf-8"):
            data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if parsed.path == "/":
                return self.send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            if parsed.path == "/api/problems":
                return self.send(200, {"items": app.listing(), "state": str(app.state_path)})
            if parsed.path == "/api/problem":
                key = query.get("key", [""])[0]
                if key in app.problems and key in app.images:
                    return self.send(200, app.public_problem(key))
            if parsed.path == "/api/export":
                return self.send(200, app.export())
            return self.send(404, {"error": "not found"})

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
                if self.path == "/api/export":
                    return self.send(200, app.export())
                key = body.get("key")
                if key not in app.problems or key not in app.images:
                    return self.send(404, {"error": "unknown problem"})
                if self.path == "/api/save":
                    review = app.save(key, body.get("options", {}))
                elif self.path == "/api/pending":
                    review = app.mark_pending(key, body.get("options", {}))
                else:
                    return self.send(404, {"error": "not found"})
                return self.send(200, {"key": key, **review})
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                return self.send(400, {"error": str(error)})

    print(f"Loaded {len(app.order)} problems with images.")
    print(f"Review state: {app.state_path}")
    print(f"Open http://{args.host}:{args.port} (Ctrl+C to stop).", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
