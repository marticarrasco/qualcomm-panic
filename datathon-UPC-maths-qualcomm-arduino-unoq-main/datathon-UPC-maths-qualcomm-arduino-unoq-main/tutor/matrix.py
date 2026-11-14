"""The hook between the tutor and the board's 8x13 LED matrix.

The server calls show() after every answer, hint and new problem:

    matrix.show("correct")
    matrix.show("level", level=3)

Events: "thinking" (a problem is on screen), "correct", "wrong", "hint",
"level" (the student's level changed or was confirmed; level is 0-7), "idle".

The LED app can either poll GET /api/matrix on the tutor server, which returns

    {"event": "correct", "level": 3, "seq": 17, "time": 1794680000.0}

where seq goes up by one on every new event (so a poller can tell a new
"correct" from the one it already drew) and level is the last level seen, or
register a function with on_event() to be called in-process on every event.
Extra details (for example "rung" for a hint) appear as extra keys.
Standard library only.
"""
import threading
import time

EVENTS = ("thinking", "correct", "wrong", "hint", "level", "idle")

_lock = threading.Lock()
_last = {"event": "idle", "level": 0, "seq": 0, "time": time.time()}
_listeners = []


def show(event, **info):
    """Record an event and pass it to any listener. Never raises into the tutor."""
    global _last
    if event not in EVENTS:
        return last()
    with _lock:
        level = info.pop("level", _last["level"])
        level = max(0, min(7, int(level))) if level is not None else _last["level"]
        _last = {"event": event, "level": level, "seq": _last["seq"] + 1, "time": time.time(), **info}
        current = dict(_last)
    for listener in list(_listeners):
        try:
            listener(dict(current))
        except Exception:  # a broken display must not break the lesson
            pass
    return current


def last():
    """The most recent event, as GET /api/matrix returns it."""
    with _lock:
        return dict(_last)


def on_event(function):
    """Call function(event_dict) on every new event, e.g. to draw on the LEDs."""
    _listeners.append(function)
