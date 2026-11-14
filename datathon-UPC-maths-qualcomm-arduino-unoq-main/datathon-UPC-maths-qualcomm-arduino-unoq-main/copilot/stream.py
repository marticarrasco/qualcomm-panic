"""Deadline-aware streaming client for a local llama-server."""
import http.client
import json
import socket
import threading
import time
from urllib.parse import urlsplit


def chat(url, payload, deadline):
    """Yield SSE events, interrupting socket reads at the absolute deadline."""
    endpoint = urlsplit(url)
    if endpoint.scheme != "http" or endpoint.hostname not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("expected a local HTTP llama-server")
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        raise TimeoutError
    connection = http.client.HTTPConnection(endpoint.hostname, endpoint.port,
                                            timeout=min(remaining, 2))
    timer = None
    expired = threading.Event()
    try:
        connection.connect()
        sock = connection.sock
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError
        sock.settimeout(remaining)

        def interrupt():
            expired.set()
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

        timer = threading.Timer(remaining, interrupt)
        timer.daemon = True
        timer.start()
        path = endpoint.path + (f"?{endpoint.query}" if endpoint.query else "")
        connection.request("POST", path, json.dumps(payload),
                           {"Content-Type": "application/json", "Connection": "close"})
        with connection.getresponse() as response:
            if response.status != 200:
                raise OSError(f"llama-server HTTP {response.status}")
            for line in response:
                if time.perf_counter() >= deadline:
                    raise TimeoutError
                if line.startswith(b"data:"):
                    data = line[5:].strip()
                    if data == b"[DONE]":
                        return
                    yield json.loads(data)
            if expired.is_set() or time.perf_counter() >= deadline:
                raise TimeoutError
    except (OSError, http.client.HTTPException) as error:
        if expired.is_set() or time.perf_counter() >= deadline:
            raise TimeoutError from None
        if isinstance(error, http.client.HTTPException):
            raise OSError("llama-server disconnected") from error
        raise
    finally:
        if timer is not None:
            timer.cancel()
        connection.close()
