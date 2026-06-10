#!/bin/bash
# Pipeline: GLA baseline → CLA-GDN-asym + GDN baseline.
set -u
LOG=/tmp/auto_pipeline.log
PY=/mnt/c/Users/Blake/Documents/VSCode/CLA/.venv/bin/python
ZOO=/home/blake/zoology

echo "[$(date +%H:%M:%S)] === pipeline start ===" >> "$LOG"

# Stage 1: baseline GLA (10 runs: baseline-gla + inhouse-gla-norm × 5 seeds)
rm -f "$ZOO/sweep_results.jsonl"
echo "[$(date +%H:%M:%S)] launching baseline GLA" >> "$LOG"
cd "$ZOO" && "$PY" run_sweep_robust.py zoology/experiments/baseline_gla_small_task.py >> "$LOG" 2>&1
mv "$ZOO/sweep_results.jsonl" "$ZOO/sweep_results_baseline_gla.jsonl"
echo "[$(date +%H:%M:%S)] baseline GLA done" >> "$LOG"

# Stage 2: GDN compare (CLA-GDN-asym first, then baseline GDN; 5 seeds each = 10 runs)
rm -f "$ZOO/gdn_results.jsonl"
echo "[$(date +%H:%M:%S)] launching GDN compare" >> "$LOG"
cd "$ZOO" && "$PY" run_gdn_compare.py >> "$LOG" 2>&1
echo "[$(date +%H:%M:%S)] GDN compare done" >> "$LOG"

echo "[$(date +%H:%M:%S)] === pipeline complete ===" >> "$LOG"
