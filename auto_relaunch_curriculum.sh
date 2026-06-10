#!/bin/bash
# Wait for the current curriculum sweep to finish (18 results), then relaunch
# the script so the new 4th condition (dot_sympeak_curr) runs via resume logic.
set -u
LOG=/tmp/auto_relaunch.log
PY=/mnt/c/Users/Blake/Documents/VSCode/CLA/.venv/bin/python
ZOO=/home/blake/zoology

echo "[$(date +%H:%M:%S)] waiting for current sweep to finish (target N=18)" >> "$LOG"
while true; do
  if [ -f "$ZOO/curriculum_results.jsonl" ]; then
    N=$(wc -l < "$ZOO/curriculum_results.jsonl")
    if [ "$N" -ge 18 ]; then
      echo "[$(date +%H:%M:%S)] current sweep done (N=$N), relaunching for dot_sympeak_curr" >> "$LOG"
      break
    fi
  fi
  sleep 30
done

cd "$ZOO" && "$PY" run_curriculum_sweep.py >> "$LOG" 2>&1
echo "[$(date +%H:%M:%S)] dot_sympeak_curr stage done" >> "$LOG"
