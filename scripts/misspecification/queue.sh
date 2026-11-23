#!/usr/bin/env bash
# Misspecification study: the whole job list in priority order (restartable: every step
# skips finished outputs). Waits for an already running run.py first.
#   setsid nohup bash scripts/misspecification/queue.sh > <results>/logs/queue.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
L=/home/matteo/BIO-SBI/results/exact_gw/misspecification/logs
mkdir -p "$L"
while pgrep -f "misspecification/run.py run" > /dev/null; do sleep 20; done
echo "$(date) fit seeds 0-4"
uv run python -u scripts/misspecification/run.py run --workers 16 --only fit --seeds 0-4 >> "$L/fit_s0-4.log" 2>&1
echo "$(date) extended models seeds 0-4"
uv run python -u scripts/misspecification/run.py run --workers 16 --only ext --seeds 0-4 >> "$L/ext_s0-4.log" 2>&1
echo "$(date) dediff seeds 0-49"
uv run python -u scripts/misspecification/run.py run --workers 16 --only dediff --dediff-seeds 0-49 >> "$L/dediff_s0-49.log" 2>&1
echo "$(date) nuts"
uv run python -u scripts/misspecification/nuts.py --workers 4 --threads 4 >> "$L/nuts.log" 2>&1
echo "$(date) fit seeds 5-9"
uv run python -u scripts/misspecification/run.py run --workers 16 --only fit --seeds 5-9 >> "$L/fit_s5-9.log" 2>&1
echo "$(date) extended models seeds 5-9"
uv run python -u scripts/misspecification/run.py run --workers 16 --only ext --seeds 5-9 >> "$L/ext_s5-9.log" 2>&1
echo "$(date) dediff seeds 50-99"
uv run python -u scripts/misspecification/run.py run --workers 16 --only dediff --dediff-seeds 50-99 >> "$L/dediff_s50-99.log" 2>&1
echo "$(date) queue done"
