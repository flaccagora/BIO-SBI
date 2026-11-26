#!/usr/bin/env bash
# Exact-GW NUTS (XG2 preset, adjoint kernel, stopping rule) on custom slow/fast-mixing truths
# of the composition-convergence study, at most MAXJOBS cells at a time (4 threads each).
# Restart-safe: a cell whose folder or log exists is skipped.
#   setsid nohup scripts/convergence/run_nuts.sh [jobs file] [root] [max jobs] > log 2>&1 &
# Jobs file: one "label n seed" per line; truths in scripts/convergence/truths.sh.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
JOBS_FILE=${1:-$HERE/nuts_jobs.txt}
ROOT=${2:-/home/matteo/BIO-SBI/results/exact_gw/convergence/nuts}
MAXJOBS=${3:-2}
BIN=/home/matteo/BIO-SBI/.venv/bin/bio-sbi
# shellcheck source=truths.sh
source "$HERE/truths.sh"
mkdir -p "$ROOT/_logs"
cd /home/matteo/BIO-SBI || exit 1
mine() { pgrep -fc "bio-sbi -m \+experiment=exact_gw_data_size .*hydra.sweep.dir=$ROOT"; }
grep -v '^\s*#' "$JOBS_FILE" | while read -r gt n seed; do
  [ -z "${gt:-}" ] && continue
  dir="$ROOT/$gt/n${n}_s${seed}"
  log="$ROOT/_logs/${gt}_n${n}_s${seed}.log"
  if [ -e "$dir" ] || [ -e "$log" ]; then echo "$(date -Is) skip $gt $n $seed"; continue; fi
  while [ "$(mine)" -ge "$MAXJOBS" ]; do sleep 30; done
  truth="${TRUTH[$gt]}"
  setsid nohup "$BIN" -m +experiment=exact_gw_data_size "+truths.$gt=$truth" "gt=$gt" \
    "design.clones=$n" "seed=$seed" inference.method.kernel=adjoint \
    inference.method.stop.enabled=true \
    hydra.sweep.dir="$ROOT/$gt" "hydra.sweep.subdir=n${n}_s${seed}" \
    > "$log" 2>&1 < /dev/null &
  echo "$(date -Is) launched $gt $n $seed -> $dir"
  sleep 60
done
echo "$(date -Is) all jobs launched"
