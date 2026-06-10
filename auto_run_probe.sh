#!/bin/bash
# Wait for curriculum sweep to hit 24 results, then run the peakiness probe.
set -u
LOG=/tmp/auto_probe.log
PY=/mnt/c/Users/Blake/Documents/VSCode/CLA/.venv/bin/python
ZOO=/home/blake/zoology

echo "[$(date +%H:%M:%S)] probe watcher started" >> "$LOG"
while true; do
  if [ -f "$ZOO/curriculum_results.jsonl" ]; then
    N=$(wc -l < "$ZOO/curriculum_results.jsonl")
    if [ "$N" -ge 24 ]; then
      echo "[$(date +%H:%M:%S)] curriculum sweep done (N=$N), launching probe" >> "$LOG"
      break
    fi
  fi
  sleep 30
done

cd "$ZOO" && "$PY" probe_router_peakiness.py > /tmp/probe_output.log 2>&1
echo "[$(date +%H:%M:%S)] probe done" >> "$LOG"

# Then run baseline GLA at 5 seeds (apples-to-apples comparison vs dot_asym_curr).
rm -f "$ZOO/sweep_results.jsonl"
echo "[$(date +%H:%M:%S)] launching baseline GLA (10 runs: baseline-gla + inhouse-gla-norm × 5 seeds)" >> "$LOG"
cd "$ZOO" && "$PY" run_sweep_robust.py zoology/experiments/baseline_gla_small_task.py >> "$LOG" 2>&1
mv "$ZOO/sweep_results.jsonl" "$ZOO/sweep_results_baseline_gla.jsonl"
echo "[$(date +%H:%M:%S)] baseline GLA done" >> "$LOG"

# Then GDN comparison: baseline GDN vs CLA-GDN+asym+curriculum, 5 seeds each.
echo "[$(date +%H:%M:%S)] launching GDN comparison (10 runs)" >> "$LOG"
rm -f "$ZOO/gdn_results.jsonl"
cd "$ZOO" && "$PY" run_gdn_compare.py >> "$LOG" 2>&1
echo "[$(date +%H:%M:%S)] GDN comparison done" >> "$LOG"
