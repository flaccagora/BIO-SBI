#!/bin/bash
# Lab-knowledge MAP on datasets 6-11 (seeds 0-5 in lab_queue.sh), so that the lab models are
# compared with est / known on the same 12 datasets. 4 workers x 1 thread, alongside the NUTS.
cd "$(dirname "$0")/../.." || exit 1
LOG=/home/matteo/BIO-SBI/results/exact_gw/founder_distribution/logs
MAP=""
for gt in T1_ref T3_Bdiv P3; do
  for d in S2_pA0.7 S4_pA0.5 S3_pA0.9 S1_pA1.0; do
    for m in lab100 lab20 labw100; do MAP="$MAP,$gt:$d:$m"; done
  done
done
MAP=${MAP#,}
uv run python scripts/founder_distribution/run.py mle --jobs $MAP --seeds 6,7,8,9,10,11 \
  --workers 4 --threads 1 >> $LOG/lab_map_seeds.log 2>&1
