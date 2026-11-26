#!/bin/bash
# Structured truths S0 (no switching) and M (mirror roles): MLE (16 workers x 1 thread) on
# datasets 0-9, then NUTS on dataset 0 (4 workers x 4 threads): S0 free / noswitch, M free
# (the tied model has no NUTS: rows cannot be tied in the law). Local machine only;
# restartable (finished / locked jobs are skipped).
cd "$(dirname "$0")/../.." || exit 1
LOG=/home/matteo/BIO-SBI/results/exact_gw/death_free/logs
mkdir -p $LOG
uv run python scripts/death_free/run.py mle --kinds S0,M --seeds 0-9 --workers 16 \
  >> $LOG/mle_s0_m.log 2>&1
uv run python scripts/death_free/run.py nuts --jobs S0:free,S0:noswitch,M:free --workers 4 \
  --threads 4 >> $LOG/nuts_s0_m.log 2>&1
echo "queue finished $(date)" >> $LOG/nuts_s0_m.log
