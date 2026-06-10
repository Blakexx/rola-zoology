#!/bin/bash
# Waits for run_canonical_state_scan.py process to exit (not just N lines),
# then launches run_zoology_canonical_scan.py.
set -u
LOG=/tmp/chain_zoology.log
PY=/mnt/c/Users/Blake/Documents/VSCode/CLA/.venv/bin/python
ZOO=/home/blake/zoology

echo "[$(date +%H:%M:%S)] chain_zoology watcher started; waiting for run_canonical_state_scan.py to exit" >> "$LOG"
while pgrep -f "run_canonical_state_scan.py" > /dev/null 2>&1; do
  sleep 30
done
echo "[$(date +%H:%M:%S)] run_canonical_state_scan.py done, launching zoology_canonical_scan" >> "$LOG"

cd "$ZOO" && "$PY" run_zoology_canonical_scan.py >> "$LOG" 2>&1
echo "[$(date +%H:%M:%S)] zoology_canonical_scan done" >> "$LOG"
