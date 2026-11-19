#!/usr/bin/env bash
# Ratio study (assumed / true mis-typing rate), 16 single-threaded workers, staged so that
# partial results cover every cell: H0 seeds 0-49, H1 seeds 0-24, H0 50-99, H1 25-49.
# Logs: results/exact_gw/dediff_test/logs/ratio_*.log. Resumable (finished analyses kept).
#   nohup bash scripts/dediff_test/queue_ratio.sh > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/../.."
LOG=/home/matteo/BIO-SBI/results/exact_gw/dediff_test/logs
mkdir -p "$LOG"
W=${WORKERS:-16}
run() { uv run python scripts/dediff_test/run.py ratio "$@" --workers "$W"; }
echo "ratio queue started $(date)" >> "$LOG/ratio_queue.log"
run --cells ratio_null --seeds 0:50 > "$LOG/ratio_null_a.log" 2>&1
run --cells ratio_alt --seeds 0:25 > "$LOG/ratio_alt_a.log" 2>&1
run --cells ratio_null --seeds 50:100 > "$LOG/ratio_null_b.log" 2>&1
run --cells ratio_alt --seeds 25:50 > "$LOG/ratio_alt_b.log" 2>&1
echo "ratio queue finished $(date)" >> "$LOG/ratio_queue.log"
