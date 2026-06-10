#!/bin/bash
# After the GDN-noconv-pos128 test finishes (5 results), launch the pos_emb=0
# follow-up sweep (missing data: baseline-gdn no-conv, cla-gdn-asym, inhouse-gla-norm).
set -u
LOG=/tmp/auto_stage2.log
PY=/mnt/c/Users/Blake/Documents/VSCode/CLA/.venv/bin/python
ZOO=/home/blake/zoology

echo "[$(date +%H:%M:%S)] stage2 watcher started" >> "$LOG"
while true; do
  if [ -f "$ZOO/sweep_results.jsonl" ]; then
    N=$(wc -l < "$ZOO/sweep_results.jsonl")
    if [ "$N" -ge 5 ]; then
      echo "[$(date +%H:%M:%S)] stage1 done (N=$N), launching stage2 (pos_emb=0 missing models)" >> "$LOG"
      break
    fi
  fi
  sleep 30
done

# Archive stage 1 results
mv "$ZOO/sweep_results.jsonl" "$ZOO/sweep_results_gdn_noconv_pos128.jsonl"

# Stage 2: pos_emb=0 missing data
rm -f "$ZOO/h2h_pos0_results.jsonl"
cd "$ZOO" && "$PY" run_h2h_pos0.py >> "$LOG" 2>&1
echo "[$(date +%H:%M:%S)] stage2 done" >> "$LOG"
