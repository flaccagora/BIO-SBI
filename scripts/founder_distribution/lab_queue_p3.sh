#!/bin/bash
# Extra NUTS check of the lab-knowledge experiment: P3, mostly single cells (p_A 0.7), lab100,
# which has est / known NUTS counterparts (replaces P3:S4_pA0.5:lab100, locked). Starts when
# the MAP part of lab_queue.sh has finished; 1 worker x 4 threads.
cd "$(dirname "$0")/../.." || exit 1
LOG=/home/matteo/BIO-SBI/results/exact_gw/founder_distribution/logs
until grep -q "^done" $LOG/lab_map.log; do sleep 60; done
uv run python scripts/founder_distribution/run.py nuts --jobs P3:S2_pA0.7:lab100 --workers 1 \
  --threads 4 >> $LOG/lab_nuts_p3.log 2>&1
