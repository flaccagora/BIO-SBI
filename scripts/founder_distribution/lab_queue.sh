#!/bin/bash
# Lab-knowledge experiment (report section 12): informative founder priors.
#   MAP (16 workers x 1 thread): T1, T3, P3 x 4 seedings x {lab20, lab100, labw100} x 6 datasets
#   then 4 NUTS checks (4 workers x 4 threads).
# Restartable: run.py skips finished / locked jobs (delete stale .lock files after a crash).
cd "$(dirname "$0")/../.." || exit 1
LOG=/home/matteo/BIO-SBI/results/exact_gw/founder_distribution/logs
mkdir -p $LOG
MAP=""
for gt in T1_ref T3_Bdiv P3; do
  for d in S2_pA0.7 S4_pA0.5 S3_pA0.9 S1_pA1.0; do
    for m in lab100 lab20 labw100; do MAP="$MAP,$gt:$d:$m"; done
  done
done
MAP=${MAP#,}
uv run python scripts/founder_distribution/run.py mle --jobs $MAP --seeds 0,1,2,3,4,5 \
  --workers 16 --threads 1 >> $LOG/lab_map.log 2>&1
NUTS="T1_ref:S4_pA0.5:lab100,T1_ref:S2_pA0.7:lab100,P3:S4_pA0.5:lab100,T1_ref:S4_pA0.5:labw100"
uv run python scripts/founder_distribution/run.py nuts --jobs $NUTS --workers 4 --threads 4 \
  >> $LOG/lab_nuts.log 2>&1
echo "lab queue finished $(date)" >> $LOG/lab_map.log
