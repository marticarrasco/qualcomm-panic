#!/usr/bin/env python3
"""Chat: talk freely with the small language model on the board.

    python3 chat/server.py
    python3 chat/server.py --host 0.0.0.0     # open it to your network

Then open http://127.0.0.1:8001 in a browser on the board, or forward the
port to your laptop:

    ssh -L 8001:127.0.0.1:8001 arduino@<board-ip>
    adb forward tcp:8001 tcp:8001

Standard library only. The page keeps the conversation and sends all of it
with every message, so this server holds no state.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from copilot.client import get  # noqa: E402
from copilot.config import CONFIG  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LLM = f"http://127.0.0.1:{CONFIG['LLM_PORT']}"

# Keep the system prompt stable: llama-server caches the start of the prompt
SYSTEM = ("You are a friendly assistant running on an Arduino UNO Q board. "
          "Answer briefly and clearly, in the language the user writes in.")

# The model reads LLM_CTX tokens at most (4096). Leave room for the reply, and
# count about 3 characters per token, which is safe for European languages.
MAX_REPLY_TOKENS = 512
HISTORY_CHARS = (int(CONFIG["LLM_CTX"]) - MAX_REPLY_TOKENS - 200) * 3
# Seconds without a single token before giving up
SILENCE_LIMIT = 120


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to open it to your network")
    p.add_argument("--port", type=int, default=8001)
    return p.parse_args()


def info():
    try:
        up = get(LLM + "/health", timeout=2).get("status") == "ok"
    except (OSError, ValueError):
        up = False
    return {"llm": up, "model": CONFIG["LLM_ALIAS"], "context": int(CONFIG["LLM_CTX"])}


def clean(messages):
    """Keep only user and assistant text, and drop the oldest turns until it fits."""
    turns = [{"role": m["role"], "content": m["content"]} for m in messages
             if isinstance(m, dict) and m.get("role") in ("user", "assistant")
             and isinstance(m.get("content"), str) and m["content"].strip()]
    while len(turns) > 1 and sum(len(m["content"]) for m in turns) > HISTORY_CHARS:
        turns.pop(0)
    # A conversation must start with the user
    while turns and turns[0]["role"] != "user":
        turns.pop(0)
    if turns:
        turns[-1]["content"] = turns[-1]["content"][:HISTORY_CHARS]
    return turns


def stream_reply(turns):
    """Yield the reply piece by piece, then one dict with the timings."""
    payload = {
        "messages": [{"role": "system", "content": SYSTEM}] + turns,
        "max_tokens": MAX_REPLY_TOKENS,
        "temperature": 0.7,
        "cache_prompt": True,
        "stream": True,
    }
    request = urllib.request.Request(
        LLM + "/v1/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    started = time.time()
    timings = {}
    with urllib.request.urlopen(request, timeout=SILENCE_LIMIT) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            timings = event.get("timings") or timings
            for choice in event.get("choices", []):
                piece = (choice.get("delta") or {}).get("content")
                if piece:
                    yield piece
    yield {"done": True, "seconds": round(time.time() - started, 1),
           "tokens": timings.get("predicted_n"),
           "tokens_per_second": round(timings["predicted_per_second"], 1)
           if timings.get("predicted_per_second") else None}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("  " + fmt % args + "\n")

    def send(self, code, body, kind="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.partition("?")[0]
        if path == "/":
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                return self.send(200, f.read(), "text/html; charset=utf-8")
        if path == "/api/info":
            return self.send(200, info())
        return self.send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/chat":
            return self.send(404, {"error": "not found"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            turns = clean(body.get("messages") or [])
        except (ValueError, AttributeError, TypeError):
            return self.send(400, {"error": "bad request"})
        if not turns or turns[-1]["role"] != "user":
            return self.send(400, {"error": "the last message must be from the user"})

        # One JSON object per line, written as the model produces it. There is
        # no Content-Length: the reply ends when the connection closes.
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        reply = stream_reply(turns)
        try:
            for piece in reply:
                self.write_line(piece if isinstance(piece, dict) else {"delta": piece})
        except (BrokenPipeError, ConnectionError):
            pass  # the page pressed Stop: closing the generator below stops the model too
        except (OSError, ValueError, urllib.error.URLError) as error:
            try:
                self.write_line({"error": f"model server: {error}"})
            except OSError:
                pass
        finally:
            reply.close()

    def write_line(self, value):
        self.wfile.write(json.dumps(value, ensure_ascii=False).encode() + b"\n")
        self.wfile.flush()


def main():
    args = parse_args()
    state = info()
    print(f"Language model {state['model']}: {'up' if state['llm'] else 'DOWN'} at {LLM}")
    print(f"Open http://{args.host}:{args.port} (Ctrl+C to stop)", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
