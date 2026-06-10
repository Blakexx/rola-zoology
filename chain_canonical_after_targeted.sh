#!/bin/bash
# Waits for fast_targeted sweep to finish 40 runs, then launches canonical_state_scan.
set -u
LOG=/tmp/chain_canonical.log
PY=/mnt/c/Users/Blake/Documents/VSCode/CLA/.venv/bin/python
ZOO=/home/blake/zoology

echo "[$(date +%H:%M:%S)] chain_canonical watcher started; waiting for fast_targeted_results.jsonl N=40" >> "$LOG"
while true; do
  if [ -f "$ZOO/fast_targeted_results.jsonl" ]; then
    N=$(wc -l < "$ZOO/fast_targeted_results.jsonl")
    if [ "$N" -ge 40 ]; then
      echo "[$(date +%H:%M:%S)] fast_targeted N=$N, launching canonical_state_scan" >> "$LOG"
      break
    fi
  fi
  sleep 20
done

cd "$ZOO" && "$PY" run_canonical_state_scan.py >> "$LOG" 2>&1
echo "[$(date +%H:%M:%S)] canonical_state_scan done" >> "$LOG"
