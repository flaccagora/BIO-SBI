#!/usr/bin/env bash
# Wait until no exact_gw_data_size sweep is running, then fit the MLE of every finished
# run (scripts/exact_gw_mle.py: mle.json + posterior.png with the MLE) and rerun the XG2
# analysis. Log: <root>/mle_queue.log.
#   setsid nohup scripts/queue_xg2_mle.sh [results root] [workers] &
set -uo pipefail
ROOT=${1:-results/exact_gw/data_size}
WORKERS=${2:-16}
cd "$(dirname "$0")/.."
echo "$(date -Is) waiting for exact_gw_data_size sweeps to finish"
while pgrep -f "python.*bio-sbi -m \+experiment=exact_gw_data_size" > /dev/null; do sleep 120; done
echo "$(date -Is) sweeps finished: $(find "$ROOT" -name metrics.json | wc -l) runs with metrics.json"
OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/exact_gw_mle.py "$ROOT" --workers "$WORKERS"
echo "$(date -Is) MLE fits done; running analysis"
.venv/bin/python scripts/analyse_exact_gw_data_size.py "$ROOT" > "$ROOT/xg2_analysis.txt" 2>&1
echo "$(date -Is) analysis written to $ROOT/xg2_analysis.txt"
