#!/usr/bin/env bash
# Copy this project to the Arduino UNO Q over USB and check that it works there.
# Run it on the laptop, from the project folder, with the board plugged in:
#
#     bash laptop/deploy_to_board.sh            # code, data, tutor bank and images
#     bash laptop/deploy_to_board.sh --no-images # skip the 145 MB of images (already on the board)
#
# It writes only to ~/qp and ~/ArduinoApps/math-matrix on the board. The model
# services and the organizers' starter repository are left alone.
set -euo pipefail

cd "$(dirname "$0")/.."
BOARD_DIR=/home/arduino/qp
IMAGES=data/full_noocr/images
[ -d "$IMAGES" ] || IMAGES=data/images
WITH_IMAGES=1
[ "${1:-}" = "--no-images" ] && WITH_IMAGES=0

step() { echo; echo "==> $*"; }

step "Board"
adb get-state >/dev/null 2>&1 || { echo "No board on USB. Plug it in and check: adb devices"; exit 1; }
adb shell 'hostname; systemctl is-active math-llm math-embed | tr "\n" " "; echo'

step "Packing"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
FILES=(copilot tutor predict.py eval.py build_index.py config.env data/sample_problems.jsonl data/val.jsonl data/train.jsonl)
for extra in data/tutor.db data/index-text.db; do
  [ -f "$extra" ] && FILES+=("$extra") || echo "  (no $extra: skipped)"
done
if [ "$WITH_IMAGES" = 1 ] && [ -d "$IMAGES" ]; then FILES+=("$IMAGES"); fi
COPYFILE_DISABLE=1 tar cf "$TMP/qp.tar" --exclude='__pycache__' --exclude='tutor_students.db' "${FILES[@]}"
ls -lh "$TMP/qp.tar" | awk '{print "  " $5}'

step "Copying to $BOARD_DIR"
adb push "$TMP/qp.tar" /tmp/qp.tar | tail -1
adb shell "mkdir -p $BOARD_DIR/results && cd $BOARD_DIR && tar xf /tmp/qp.tar 2>/dev/null; rm -f /tmp/qp.tar; ls"

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
adb shell "pkill -f 'tutor/server.py' 2>/dev/null; cd $BOARD_DIR && nohup python3 tutor/server.py --host 0.0.0.0 ${TUTOR_FLAGS:-} > tutor.log 2>&1 &
sleep 4; head -4 $BOARD_DIR/tutor.log; curl -s -m 5 http://127.0.0.1:8000/api/matrix; echo"

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
echo "Next: a timed run on the board, for the numbers in the README:"
echo "  adb shell 'cd $BOARD_DIR && python3 eval.py --languages eng --limit 50 --name board-final-50'"
