#!/usr/bin/env bash
# After the queued MLE fits + analysis (mle_queue.log ends with "analysis written"),
# reorganise XG2 into <root>/<gt>/n<n>_s<seed>/ and rerun the analysis on the new layout.
#   setsid nohup scripts/queue_xg2_reorganise.sh [results root] &
set -uo pipefail
ROOT=${1:-results/exact_gw/data_size}
cd "$(dirname "$0")/.."
echo "$(date -Is) waiting for the final MLE fits and analysis"
until grep -q "analysis written" "$ROOT/mle_queue.log" 2>/dev/null; do sleep 60; done
while pgrep -f "^[^ ]*python[^ ]* [^ ]*(bio-sbi -m \+experiment=exact_gw_data_size|exact_gw_mle.py|analyse_exact_gw_data_size.py)" > /dev/null; do sleep 30; done
echo "$(date -Is) reorganising"
.venv/bin/python scripts/reorganise_xg2.py "$ROOT" || { echo "reorganise failed"; exit 1; }
echo "$(date -Is) rerunning the analysis on the new layout"
.venv/bin/python scripts/analyse_exact_gw_data_size.py "$ROOT" > "$ROOT/xg2_analysis.txt" 2>&1
echo "$(date -Is) done: $(find "$ROOT" -mindepth 3 -maxdepth 3 -name metrics.json | wc -l) runs in <gt>/n*_s*/"
