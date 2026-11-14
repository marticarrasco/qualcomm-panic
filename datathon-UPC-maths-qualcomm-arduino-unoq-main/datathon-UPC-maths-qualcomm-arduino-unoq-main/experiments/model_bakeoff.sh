#!/bin/bash
# Model bake-off: one candidate at a time on port 8086, scored with reasoning_budget.py.
# Board speeds are ESTIMATES: 11 read / 5.6 write tokens/s of Qwen3.5 0.8B Q8_0 (834.6 MB)
# scaled inversely with GGUF file size.
#   bash experiments/model_bakeoff.sh tag file expected_bytes [extra server flags...]
cd "$(dirname "$0")/.." || exit 1
tag=$1; file=$HOME/models/candidates/$2; bytes=$3; shift 3
flags=("$@"); [ ${#flags[@]} -eq 0 ] && flags=(--reasoning off --reasoning-budget 0)
[ "${flags[0]}" = "none" ] && flags=()
until [ -f "$file" ] && [ "$(stat -f %z "$file")" = "$bytes" ]; do sleep 5; done
ref=$(stat -f %z "$HOME/models/Qwen_Qwen3.5-0.8B-Q8_0.gguf")
read_tps=$(python3 -c "print(round(11*$ref/$bytes,2))"); write_tps=$(python3 -c "print(round(5.6*$ref/$bytes,2))")
pkill -f "port 8086"; sleep 2
llama-server -m "$file" --host 127.0.0.1 --port 8086 --threads 4 --ctx-size 4096 --parallel 1 --cache-ram 0 \
  --ctx-checkpoints 2 --jinja "${flags[@]}" > "$HOME/models/logs/cand-$tag.log" 2>&1 &
for i in $(seq 120); do curl -sf http://127.0.0.1:8086/health > /dev/null && break; sleep 1; done
curl -sf http://127.0.0.1:8086/health > /dev/null || { echo "$tag: server failed to start"; pkill -f "port 8086"; exit 1; }
echo "== $tag read $read_tps write $write_tps flags ${flags[*]}"
python3 experiments/reasoning_budget.py --prompts shots --port 8086 --read-tps "$read_tps" --write-tps "$write_tps" \
  --tag "bakeoff-$tag" ${LIMIT:+--limit $LIMIT} 2> /dev/null
pkill -f "port 8086"; sleep 2
