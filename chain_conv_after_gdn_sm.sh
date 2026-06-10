#!/bin/bash
# Waits for the gdn-state-matched sweep to finish 20 runs, then launches conv_sweep.
set -u
LOG=/tmp/chain_conv.log
PY=/mnt/c/Users/Blake/Documents/VSCode/CLA/.venv/bin/python
ZOO=/home/blake/zoology

echo "[$(date +%H:%M:%S)] chain_conv watcher started; waiting for gdn_state_matched_results.jsonl N=20" >> "$LOG"
while true; do
  if [ -f "$ZOO/gdn_state_matched_results.jsonl" ]; then
    N=$(wc -l < "$ZOO/gdn_state_matched_results.jsonl")
    if [ "$N" -ge 20 ]; then
      echo "[$(date +%H:%M:%S)] gdn-stmatched N=$N, launching conv_sweep" >> "$LOG"
      break
    fi
  fi
  sleep 30
done

cd "$ZOO" && "$PY" run_conv_sweep.py >> "$LOG" 2>&1
echo "[$(date +%H:%M:%S)] conv_sweep done" >> "$LOG"
