#!/usr/bin/env bash
# Wait until no exact_gw_data_size sweep is running, then fit the MLE of every finished
# run that lacks an up-to-date mle.json (scripts/exact_gw_mle.py: mle.json + posterior.png
# with the MLE) and rerun the XG2 analysis. Log: <root>/mle_queue.log.
#   setsid nohup scripts/queue_xg2_mle.sh [results root] [workers] [prior starts] &
set -uo pipefail
ROOT=${1:-results/exact_gw/data_size}
WORKERS=${2:-16}
PRIOR_STARTS=${3:-24}
cd "$(dirname "$0")/.."
echo "$(date -Is) waiting for exact_gw_data_size sweeps to finish"
# match the python sweep processes only (launcher shells can carry the same text)
while pgrep -f "python.*bio-sbi -m \+experiment=exact_gw_data_size" > /dev/null; do sleep 120; done
# and any MLE batch still running on these runs (avoid fitting the same run twice)
while pgrep -f "python.*exact_gw_mle.py" > /dev/null; do sleep 60; done
echo "$(date -Is) sweeps finished: $(find "$ROOT" -name metrics.json | wc -l) runs with metrics.json"
OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/exact_gw_mle.py "$ROOT" --workers "$WORKERS" --prior-starts "$PRIOR_STARTS"
echo "$(date -Is) MLE fits done; running analysis"
.venv/bin/python scripts/analyse_exact_gw_data_size.py "$ROOT" > "$ROOT/xg2_analysis.txt" 2>&1
echo "$(date -Is) analysis written to $ROOT/xg2_analysis.txt"
