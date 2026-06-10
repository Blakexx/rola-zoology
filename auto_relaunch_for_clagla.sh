#!/bin/bash
# After current pos0 sweep finishes 15 runs, relaunch run_h2h_pos0.py so it
# picks up the 5 new cla-gla-asym configs (added after the in-memory config
# loaded). Resume logic skips the 15 done indices.
set -u
LOG=/tmp/auto_relaunch_clagla.log
PY=/mnt/c/Users/Blake/Documents/VSCode/CLA/.venv/bin/python
ZOO=/home/blake/zoology

echo "[$(date +%H:%M:%S)] watcher started — waiting for N=15" >> "$LOG"
while true; do
  if [ -f "$ZOO/h2h_pos0_results.jsonl" ]; then
    N=$(wc -l < "$ZOO/h2h_pos0_results.jsonl")
    if [ "$N" -ge 15 ]; then
      echo "[$(date +%H:%M:%S)] N=$N, relaunching for cla-gla-asym at pos_emb=0" >> "$LOG"
      break
    fi
  fi
  sleep 30
done

cd "$ZOO" && "$PY" run_h2h_pos0.py >> "$LOG" 2>&1
echo "[$(date +%H:%M:%S)] cla-gla-asym pos0 done" >> "$LOG"

# Stage: CLA-GDN config sweep (15 runs = 3 variants × 5 seeds)
echo "[$(date +%H:%M:%S)] launching CLA-GDN config sweep" >> "$LOG"
rm -f "$ZOO/cla_gdn_sweep_results.jsonl"
cd "$ZOO" && "$PY" run_cla_gdn_sweep.py >> "$LOG" 2>&1
echo "[$(date +%H:%M:%S)] CLA-GDN config sweep done" >> "$LOG"
