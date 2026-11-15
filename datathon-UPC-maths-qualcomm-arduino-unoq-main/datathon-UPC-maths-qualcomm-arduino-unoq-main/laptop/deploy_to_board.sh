#!/usr/bin/env bash
# Copy this project to the Arduino UNO Q over USB and check that it works there.
# Run it on the laptop, from the project folder, with the board plugged in:
#
#     bash laptop/deploy_to_board.sh            # code, data, tutor bank and images
#     bash laptop/deploy_to_board.sh --no-images # skip the 145 MB of images (already on the board)
#
# The fine-tuned model is not in git (794 MB). Put the file in ~/models on the laptop,
# or point POT_GGUF at it, and this script copies it to the board, checks its SHA-256
# and switches the math-llm service to it. Without the file the board keeps the stock
# model and the solver uses the few-shot prompt (POT_STYLE=auto picks the prompt).
#
# It writes to ~/qp, ~/models and ~/ArduinoApps/math-matrix on the board, and
# reinstalls the math-llm service only when the model changes.
set -euo pipefail

cd "$(dirname "$0")/.."
BOARD_DIR=/home/arduino/qp
IMAGES=data/full_noocr/images
[ -d "$IMAGES" ] || IMAGES=data/images
WITH_IMAGES=1
[ "${1:-}" = "--no-images" ] && WITH_IMAGES=0
POT_FILE=Qwen3.5-0.8B-M3Kang-PoT-v4-e1-Q8_0.gguf
POT_SHA=7c90dbbc3518241fd9b1661887abe1131a07d9d9f637c86a1a8a897b7a19cbb3
POT_GGUF="${POT_GGUF:-$HOME/models/$POT_FILE}"

step() { echo; echo "==> $*"; }

step "Board"
adb get-state >/dev/null 2>&1 || { echo "No board on USB. Plug it in and check: adb devices"; exit 1; }
adb shell 'hostname; systemctl is-active math-llm math-embed | tr "\n" " "; echo'

step "Packing"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
FILES=(copilot tutor board predict.py eval.py build_index.py config.env data/sample_problems.jsonl data/val.jsonl data/train.jsonl)
for extra in data/tutor.db data/index-text.db; do
  [ -f "$extra" ] && FILES+=("$extra") || echo "  (no $extra: skipped)"
done
if [ "$WITH_IMAGES" = 1 ] && [ -d "$IMAGES" ]; then FILES+=("$IMAGES"); fi
COPYFILE_DISABLE=1 tar cf "$TMP/qp.tar" --exclude='__pycache__' --exclude='tutor_students.db' "${FILES[@]}"
ls -lh "$TMP/qp.tar" | awk '{print "  " $5}'

step "Copying to $BOARD_DIR"
adb push "$TMP/qp.tar" /tmp/qp.tar | tail -1
adb shell "mkdir -p $BOARD_DIR/results && cd $BOARD_DIR && tar xf /tmp/qp.tar 2>/dev/null; rm -f /tmp/qp.tar; ls"

step "Fine-tuned model"
if [ -f "$POT_GGUF" ]; then
  [ "$(shasum -a 256 "$POT_GGUF" | cut -d' ' -f1)" = "$POT_SHA" ] || { echo "  $POT_GGUF does not have the expected SHA-256. Stopping."; exit 1; }
  BOARD_SHA="$(adb shell "sha256sum ~/models/$POT_FILE 2>/dev/null | cut -d' ' -f1" | tr -d '\r\n')"
  if [ "$BOARD_SHA" != "$POT_SHA" ]; then
    adb shell 'mkdir -p ~/models'
    adb push "$POT_GGUF" "/home/arduino/models/$POT_FILE" | tail -1
    BOARD_SHA="$(adb shell "sha256sum ~/models/$POT_FILE | cut -d' ' -f1" | tr -d '\r\n')"
    [ "$BOARD_SHA" = "$POT_SHA" ] || { echo "  The copy on the board is damaged. Run the script again."; exit 1; }
  fi
  echo "  $POT_FILE is on the board, SHA-256 checked"
  # Only the board's copy of config.env names the fine-tuned file: the repo keeps the stock one,
  # which board/03_download_models.sh can download.
  adb shell "cd $BOARD_DIR && sed -i -e 's|^LLM_FILE=.*|LLM_FILE=$POT_FILE|' -e 's|^LLM_SHA256=.*|LLM_SHA256=$POT_SHA|' config.env"
else
  echo "  No $POT_GGUF on this laptop: the board keeps the model it has."
fi
# The service keeps its file name; that name becomes a link to the model we want. This needs
# no sudo: llama-server runs as the arduino user, and systemd restarts it when it is killed.
SERVICE_PATH="$(adb shell "curl -s -m 5 http://127.0.0.1:8081/props | python3 -c 'import json,sys; print(json.load(sys.stdin).get(\"model_path\",\"\"))' 2>/dev/null" | tr -d '\r\n')"
LOADED="$(adb shell "readlink -f '$SERVICE_PATH' 2>/dev/null" | tr -d '\r\n')"
echo "  math-llm has loaded: ${LOADED:-nothing}"
if [ -f "$POT_GGUF" ] && [ -n "$SERVICE_PATH" ] && [ "$(basename "${LOADED:-x}")" != "$POT_FILE" ]; then
  echo "  Switching math-llm to $POT_FILE"
  adb shell "cd ~/models && if [ ! -L '$SERVICE_PATH' ]; then mv '$SERVICE_PATH' '$SERVICE_PATH.stock'; fi; ln -sfn /home/arduino/models/$POT_FILE '$SERVICE_PATH' && pkill -9 -f '[l]lama-server.*--port 8081'; for i in \$(seq 90); do sleep 2; curl -sf -m 2 http://127.0.0.1:8081/health >/dev/null && break; done; curl -s -m 2 http://127.0.0.1:8081/health; echo"
fi
adb shell "cd $BOARD_DIR && python3 -c 'from copilot.llm import pot_style; print(\"  prompt style:\", pot_style())'"

step "predict.py on the five sample problems (on the board)"
adb shell "cd $BOARD_DIR && python3 predict.py --input data/sample_problems.jsonl --output results/smoke.csv 2>&1 | tail -5"
adb shell "cd $BOARD_DIR && python3 - <<'PY'
import csv
rows = list(csv.DictReader(open('results/smoke.csv')))
worst = max(float(r['seconds']) for r in rows)
assert all(r['answer'] in 'ABCDE' and len(r['answer']) == 1 for r in rows), 'invalid answer'
assert worst < 60, 'too slow'
print(f'  {len(rows)} answers, all one letter, worst {worst:.1f} s')
PY"

step "Tutor"
# The solver's model on port 8081 is fine-tuned to write Python, so it cannot coach.
# The tutor gets its own copy of the stock model on port 8084 when the file is on the board.
COACH="$(adb shell 'for f in ~/models/Qwen_Qwen3.5-0.8B-Q8_0.gguf*; do [ -f "$f" ] && [ ! -L "$f" ] && echo "$f" && break; done' | tr -d '\r\n')"
LLAMA="$(adb shell '. ~/qp/config.env; echo ${LLAMA_CPP_DIR/#\~/$HOME}/build/bin/llama-server' | tr -d '\r\n')"
if [ -z "${TUTOR_FLAGS:-}" ]; then
  if [ -n "$COACH" ] && [ "$LOADED" = "$COACH" ]; then
    TUTOR_FLAGS=""
  elif [ -n "$COACH" ] && adb shell "[ -x $LLAMA ]" 2>/dev/null; then
    adb shell "pkill -f '[l]lama-server.*--port 8084'" >/dev/null 2>&1 || true  # its own call: the next one names the same command
    adb shell "nohup $LLAMA -m $COACH --host 127.0.0.1 --port 8084 --threads 4 --ctx-size 4096 --parallel 1 --cache-ram 0 --jinja --reasoning off --reasoning-budget 0 > $BOARD_DIR/coach.log 2>&1 < /dev/null &
for i in \$(seq 60); do curl -sf -m 2 http://127.0.0.1:8084/health >/dev/null && break; sleep 2; done; curl -s -m 2 http://127.0.0.1:8084/health; echo; free -m | awk 'NR==2 {print \"  RAM available: \" \$7 \" MB\"}'"
    TUTOR_FLAGS="--llm http://127.0.0.1:8084/v1/chat/completions"
  else
    echo "  No stock model on the board for coaching: the tutor runs without model help."
    TUTOR_FLAGS="--no-llm"
  fi
fi
echo "  tutor flags: ${TUTOR_FLAGS:-(default, port 8081)}"
adb shell "pkill -f '[t]utor/server.py'" >/dev/null 2>&1 || true
adb shell "cd $BOARD_DIR && nohup python3 tutor/server.py --host 0.0.0.0 ${TUTOR_FLAGS:-} > tutor.log 2>&1 < /dev/null &
sleep 4; head -4 $BOARD_DIR/tutor.log; curl -s -m 5 http://127.0.0.1:8000/api/matrix; echo"
echo "  Stop the coaching model before the official predict.py run, so it has the whole board:"
echo "    adb shell \"pkill -f '[l]lama-server.*--port 8084'\""

step "LED matrix app"
if [ -d applab/math-matrix ]; then
  adb shell 'mkdir -p ~/ArduinoApps/math-matrix/python ~/ArduinoApps/math-matrix/sketch'
  adb push applab/math-matrix/app.yaml /home/arduino/ArduinoApps/math-matrix/app.yaml >/dev/null
  adb push applab/math-matrix/python/main.py /home/arduino/ArduinoApps/math-matrix/python/main.py >/dev/null
  adb push applab/math-matrix/sketch/sketch.ino /home/arduino/ArduinoApps/math-matrix/sketch/sketch.ino >/dev/null
  adb push applab/math-matrix/sketch/sketch.yaml /home/arduino/ArduinoApps/math-matrix/sketch/sketch.yaml >/dev/null
  adb shell 'arduino-app-cli app restart user:math-matrix >/dev/null 2>&1 || arduino-app-cli app start user:math-matrix >/dev/null 2>&1; arduino-app-cli app ps 2>&1 | tail -1'
fi

step "Open the tutor on this laptop"
adb forward tcp:8000 tcp:8000 >/dev/null
echo "  http://127.0.0.1:8000   (works with Wi-Fi off: it goes over the USB cable)"
echo
echo "Next: a timed run on the board with the organizers' practice set (about 20 minutes):"
echo "  adb push data/demo/train.jsonl $BOARD_DIR/data/demo.jsonl"
echo "  adb shell 'cd $BOARD_DIR && python3 predict.py --input data/demo.jsonl --output results/demo-board.csv'"
