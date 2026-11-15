"""Start the installed Windows model for the tutor's default local endpoint."""
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from copilot.config import CONFIG


MODEL_PORT = 8081
MODEL_HOST = "127.0.0.1"
MODEL_URL = f"http://{MODEL_HOST}:{MODEL_PORT}/v1/chat/completions"


def find_llama():
    executable = shutil.which("llama-server")
    if executable:
        return executable
    local_apps = os.environ.get("LOCALAPPDATA")
    if local_apps:
        installs = Path(local_apps) / "Programs" / "llama.cpp"
        return max(installs.glob("*/llama-server.exe"), key=lambda path: path.stat().st_mtime, default=None)
    return None


def start_windows_model(log_path):
    """Reuse a listener on 8081 or start llama-server with the project's base model."""
    try:
        with socket.create_connection((MODEL_HOST, MODEL_PORT), timeout=1):
            return None
    except OSError:
        pass

    executable = find_llama()
    model = Path(CONFIG["MODELS_DIR"]).expanduser() / CONFIG["LLM_FILE"]
    if not executable or not model.is_file():
        print(f"Install llama-server and place the model at {model}, or supply --llm.", file=sys.stderr)
        return None

    try:
        with open(log_path, "ab") as log:
            process = subprocess.Popen(
                [str(executable), "-m", str(model), "--host", MODEL_HOST, "--port", str(MODEL_PORT),
                 "-c", CONFIG["LLM_CTX"], "-t", CONFIG["LLM_THREADS"], "-np", "1"],
                stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP,
            )
    except OSError as error:
        print(f"Could not start the local model: {error}", file=sys.stderr)
        return None

    print("Starting the local Qwen model…", flush=True)
    deadline = time.monotonic() + 30
    while process.poll() is None and time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://{MODEL_HOST}:{MODEL_PORT}/health", timeout=1):
                if process.poll() is None:
                    return process
        except OSError:
            time.sleep(0.4)

    print(f"Local model did not become ready. See {log_path}.", file=sys.stderr)
    return process if process.poll() is None else None
