#!/bin/bash
# Launch curriculum sweep first, then baseline GLA. Writes to /tmp/auto_queue.log.
set -u
LOG=/tmp/auto_queue.log
PY=/mnt/c/Users/Blake/Documents/VSCode/CLA/.venv/bin/python
ZOO=/home/blake/zoology

echo "[$(date +%H:%M:%S)] auto-queue v2: curriculum-first" >> "$LOG"

# Stage 1: curriculum sweep (30 runs).
echo "[$(date +%H:%M:%S)] launching curriculum sweep" >> "$LOG"
cd "$ZOO" && "$PY" run_curriculum_sweep.py >> "$LOG" 2>&1
echo "[$(date +%H:%M:%S)] curriculum sweep done" >> "$LOG"

# Stage 2: baseline GLA (10 runs).
echo "[$(date +%H:%M:%S)] launching baseline_gla_small_task" >> "$LOG"
cd "$ZOO" && "$PY" run_sweep_robust.py zoology/experiments/baseline_gla_small_task.py >> "$LOG" 2>&1
mv "$ZOO/sweep_results.jsonl" "$ZOO/sweep_results_baseline_gla.jsonl"
echo "[$(date +%H:%M:%S)] baseline_gla done" >> "$LOG"

echo "[$(date +%H:%M:%S)] all stages complete" >> "$LOG"
