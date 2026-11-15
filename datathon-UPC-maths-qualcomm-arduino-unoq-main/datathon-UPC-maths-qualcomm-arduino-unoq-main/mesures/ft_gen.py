"""Generate programs with a GGUF on the laptop, for the rows the copilot asks (text only, five numeric options).

Starts its own llama-server on port 8091 with the board's flags (CPU only) and stops it at the end.
It never touches the servers on 8081 and 8082. Only the standard library is used.

Usage, from the Datathon folder:
  <venv python> -X utf8 mesures/ft_gen.py --model <file.gguf> --style finetuned --tag ft2
  <venv python> -X utf8 mesures/ft_gen.py --model <file.gguf> --style finetuned --tag ft2 --samples 4
Output: mesures/ft-<tag>-<set>.jsonl (greedy) or mesures/ft-<tag>-<set>-samples.jsonl. They hold M3Kang ids: don't publish.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "datathon-UPC-maths-qualcomm-arduino-unoq"))
sys.path.insert(0, ROOT)
from copilot import pot  # noqa: E402

PORT = 8091
BASE = f"http://127.0.0.1:{PORT}"


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def post(path, payload, timeout=300):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def healthy():
    try:
        with urllib.request.urlopen(BASE + "/health", timeout=3) as resp:
            return json.loads(resp.read()).get("status") == "ok"
    except Exception:
        return False


def loose_options(problem):
    """{letter: number} when the five options are numbers once a unit is dropped ("8cm", "37,5 km/h", "20 kg")."""
    import re
    options = problem.get("options") or {}
    if len(options) != 5:
        return None
    nums = {}
    for k in pot.LETTERS:
        raw = str(options.get(k, "")).strip()
        v = pot.parse_num(raw)
        if v is None:
            m = re.match(r"^([-+−]?\d+(?:[.,]\d+)?)\s*(?:[a-zA-Zº°%€$²³][a-zA-Z0-9º°%€$²³/. ]{0,10})$", raw)
            v = pot.parse_num(m.group(1)) if m else None
        if v is None:
            return None
        nums[k] = v
    return nums if len(set(nums.values())) == 5 else None


def asked_rows(rows, gate):
    """default: what the copilot asks today. figure: rows it skips because they have a figure.
    units: text-only rows it skips because the options carry a unit."""
    if gate == "default":
        return [r for r in rows if r.get("text_only") and pot.numeric_options(r) is not None]
    if gate == "figure":
        return [r for r in rows if not r.get("text_only") and pot.numeric_options(r) is not None]
    return [r for r in rows if r.get("text_only") and pot.numeric_options(r) is None and loose_options(r) is not None]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gate", default="default", choices=["default", "figure", "units"])
    p.add_argument("--model", required=True)
    p.add_argument("--style", default="finetuned", choices=["finetuned", "fewshot"])
    p.add_argument("--tag", required=True)
    p.add_argument("--sets", default="val,demo")
    p.add_argument("--samples", type=int, default=0, help="extra sampled programs per row (temperature 0.45, top_p 0.9)")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--port", type=int, default=8091, help="never 8081 or 8082: those are not ours")
    args = p.parse_args()
    global PORT, BASE
    assert args.port not in (8081, 8082, 8083)
    PORT, BASE = args.port, f"http://127.0.0.1:{args.port}"

    if healthy():
        sys.exit(f"something already answers on port {PORT}: not starting a second server")
    log = open(os.path.join(HERE, f"ft-server-{args.tag}.log"), "w")
    proc = subprocess.Popen(
        ["llama-server", "-m", args.model, "--host", "127.0.0.1", "--port", str(PORT), "--threads", str(args.threads),
         "--ctx-size", "4096", "--parallel", "1", "--cache-ram", "0", "--ctx-checkpoints", "2", "--jinja",
         "--reasoning", "off", "--reasoning-budget", "0", "--device", "none", "-ngl", "0", "--no-webui", "--offline"],
        stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(180):
            if proc.poll() is not None:
                sys.exit("llama-server stopped: see the log")
            if healthy():
                break
            time.sleep(1)
        else:
            sys.exit("llama-server did not become healthy")

        files = {"val": os.path.join(ROOT, "data", "val.jsonl"), "demo": os.path.join(ROOT, "data", "demo", "train.jsonl")}
        for name in args.sets.split(","):
            rows = asked_rows(load(files[name]), args.gate)
            suffix = ("-samples" if args.samples else "") + ("" if args.gate == "default" else "-" + args.gate)
            out_path = os.path.join(HERE, f"ft-{args.tag}-{name}{suffix}.jsonl")
            done = set()
            if os.path.exists(out_path):
                done = {(r["id"], r["language"], r["seed"]) for r in load(out_path)}
            runs = [(0.0, 42)] if not args.samples else [(0.45, 43 + i) for i in range(args.samples)]
            t0 = time.time()
            with open(out_path, "a", encoding="utf-8") as out:
                for n, r in enumerate(rows, 1):
                    for temp, seed in runs:
                        if (r["id"], r["language"], seed) in done:
                            continue
                        payload = {"messages": pot.build_messages(r, args.style), "max_tokens": 140,
                                   "temperature": temp, "seed": seed, "cache_prompt": True}
                        if temp > 0:
                            payload["top_p"] = 0.9
                        resp = post("/v1/chat/completions", payload)
                        t = resp.get("timings", {})
                        out.write(json.dumps({
                            "id": r["id"], "language": r["language"], "level": r["level"], "label": r["label"],
                            "seed": seed, "temperature": temp, "reply": resp["choices"][0]["message"]["content"],
                            "finish": resp["choices"][0].get("finish_reason"), "read_n": t.get("prompt_n") or 0,
                            "cache_n": t.get("cache_n") or 0, "predicted_n": t.get("predicted_n") or 0,
                        }, ensure_ascii=False) + "\n")
                        out.flush()
                    if n % 40 == 0:
                        print(f"  {args.tag} {name}: {n}/{len(rows)} ({time.time() - t0:.0f} s)", flush=True)
            print(f"{args.tag} {name}: {len(rows)} rows -> {out_path} ({time.time() - t0:.0f} s)", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


if __name__ == "__main__":
    main()
