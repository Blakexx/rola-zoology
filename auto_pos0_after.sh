#!/bin/bash
# Wait for h2h sweep to hit 25 results, then run the pos_emb=0 follow-up sweep.
set -u
LOG=/tmp/auto_pos0.log
PY=/mnt/c/Users/Blake/Documents/VSCode/CLA/.venv/bin/python
ZOO=/home/blake/zoology

echo "[$(date +%H:%M:%S)] pos0 watcher started" >> "$LOG"
while true; do
  if [ -f "$ZOO/h2h_results.jsonl" ]; then
    N=$(wc -l < "$ZOO/h2h_results.jsonl")
    if [ "$N" -ge 25 ]; then
      echo "[$(date +%H:%M:%S)] h2h done (N=$N), launching pos0 sweep" >> "$LOG"
      break
    fi
  fi
  sleep 30
done

cd "$ZOO" && "$PY" run_h2h_pos0.py >> "$LOG" 2>&1
echo "[$(date +%H:%M:%S)] pos0 sweep done" >> "$LOG"
