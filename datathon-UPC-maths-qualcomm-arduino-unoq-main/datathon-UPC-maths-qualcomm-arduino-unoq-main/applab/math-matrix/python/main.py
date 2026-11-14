"""LED matrix for the maths tutor.

The tutor (tutor/server.py) runs on the board's Linux side and publishes its
last event at /api/matrix. This app polls it and draws on the 8x13 matrix:
a tick, a cross, a light bulb for a hint, dots while thinking, and the
student's level as a bar. Everything shown is decided here in Python; the
sketch only draws the frame it receives over the Bridge.
"""
import json
import time
import urllib.request

import numpy as np
from arduino.app_utils import App, Bridge, Frame

# The app runs in a container: try the board itself first, then the Docker bridge.
TUTOR_URLS = ["http://127.0.0.1:8000/api/matrix", "http://172.17.0.1:8000/api/matrix",
              "http://host.docker.internal:8000/api/matrix"]
HOLD_SECONDS = 4  # how long a tick, cross or bulb stays before the level bar returns

PICTURES = {
    "correct": ["............#",
                "...........##",
                "..........##.",
                ".#.......##..",
                ".##.....##...",
                "..##...##....",
                "...##.##.....",
                "....###......"],
    "wrong":   ["..##.....##..",
                "...##...##...",
                "....##.##....",
                ".....###.....",
                ".....###.....",
                "....##.##....",
                "...##...##...",
                "..##.....##.."],
    "hint":    ["....#####....",
                "...#.....#...",
                "...#..#..#...",
                "...#..#..#...",
                "....#.#.#....",
                ".....###.....",
                ".....###.....",
                "......#......"],
    "thinking": [".............",
                 ".............",
                 ".............",
                 "..##..##..##.",
                 "..##..##..##.",
                 ".............",
                 ".............",
                 "............."],
    "idle":    [".............",
                "..#.......#..",
                "..#.......#..",
                ".............",
                ".#.........#.",
                "..#.......#..",
                "...#######...",
                "............."],
}


def picture(name):
    grid = np.zeros((8, 13), dtype=np.uint8)
    for row, line in enumerate(PICTURES[name]):
        for col, pixel in enumerate(line):
            if pixel == "#":
                grid[row, col] = 7
    return grid


def level_bar(level):
    """Levels 0-7 as a staircase: one more column pair, one step higher, per level."""
    grid = np.zeros((8, 13), dtype=np.uint8)
    level = max(0, min(7, int(level)))
    for step in range(8):
        col = 1 + step * 1.5
        height = step + 1
        brightness = 7 if step <= level else 1
        for row in range(8 - height, 8):
            grid[row, int(col)] = brightness
    return grid


def draw(grid):
    Bridge.call("draw", Frame(grid).to_board_bytes())


def read_tutor():
    for url in TUTOR_URLS:
        try:
            with urllib.request.urlopen(url, timeout=1) as response:
                return json.loads(response.read())
        except (OSError, ValueError):
            continue
    return None


state = {"seq": None, "shown_at": 0.0, "level": 0, "reachable": None}


def loop():
    data = read_tutor()
    reachable = data is not None
    if reachable != state["reachable"]:
        print("tutor reachable" if reachable else "tutor not reachable yet", flush=True)
        state["reachable"] = reachable
    if not reachable:
        draw(picture("idle"))
        time.sleep(1)
        return
    event = data.get("event") or "idle"
    if data.get("level") is not None:
        state["level"] = data["level"]
    seq = data.get("seq", (event, data.get("time")))
    if seq != state["seq"]:
        state["seq"], state["shown_at"] = seq, time.monotonic()
    fresh = time.monotonic() - state["shown_at"] < HOLD_SECONDS
    if event == "thinking":
        draw(picture("thinking"))
    elif event in ("correct", "wrong", "hint") and fresh:
        draw(picture(event))
    elif event == "idle" and state["seq"] is None:
        draw(picture("idle"))
    else:
        draw(level_bar(state["level"]))
    time.sleep(0.3)


draw(picture("idle"))
App.run(user_loop=loop)
