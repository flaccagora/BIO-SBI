#!/usr/bin/env bash
# Full dedifferentiation-test study, sequentially, 16 single-threaded workers per stage.
# Logs: results/exact_gw/dediff_test/logs/. Resumable (finished datasets are skipped).
#   nohup bash scripts/dediff_test/queue.sh > /dev/null 2>&1 &
set -u
cd "$(dirname "$0")/../.."
LOG=/home/matteo/BIO-SBI/results/exact_gw/dediff_test/logs
mkdir -p "$LOG"
W=${WORKERS:-16}
run() { uv run python scripts/dediff_test/run.py "$@" --workers "$W"; }
run null --seeds 0:200 > "$LOG/null.log" 2>&1
run alt --seeds 0:50 --m 0.0025,0.005,0.01,0.02,0.05 > "$LOG/alt_small.log" 2>&1
run typing --seeds 0:100 > "$LOG/typing.log" 2>&1
run boot --cells N1,N2,N3,N4 --ns 128,1024 --seeds 0:15 --reps 100 > "$LOG/boot.log" 2>&1
run alt --seeds 0:10 --m 0.1,0.2,0.4 > "$LOG/alt_large.log" 2>&1
echo "queue finished $(date)" >> "$LOG/queue.log"
