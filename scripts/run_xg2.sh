#!/usr/bin/env bash
# XG2 (experiment exact_gw_data_size): 96 runs as 4 parallel sweeps x 4 threads (16 cores),
# one curated + one prior-drawn truth per sweep; seed outermost so complete n-curves come
# first. Results: <results>/exact_gw/data_size/{a,b,c,d}.
#   scripts/run_xg2.sh [/path/to/results]
set -euo pipefail
RESULTS=${1:-results}
OUT="$RESULTS/exact_gw/data_size"
mkdir -p "$OUT"
BIN=$(dirname "$0")/../.venv/bin/bio-sbi
declare -A ARMS=([a]="T1_ref,P1" [b]="T2_toA05,P2" [c]="T3_Bdiv,P3" [d]="T4_slowA,P4")
for arm in a b c d; do
  setsid nohup "$BIN" -m +experiment=exact_gw_data_size seed=0,1,2 "gt=${ARMS[$arm]}" \
    design.clones=128,256,512,1024 hydra.sweep.dir="$OUT/$arm" \
    > "$OUT/$arm.log" 2>&1 < /dev/null &
  echo "arm $arm (${ARMS[$arm]}): pid $!"
done
