#!/usr/bin/env bash
# Run the reparametrisation benchmark jobs one at a time (4 threads), skipping finished ones.
#   scripts/exact_gw_reparam/queue.sh jobs.txt
# jobs.txt: one job per line, "<cell> <arm> [--seed k] [--smc key=value ...]".
set -u
OUT=/home/matteo/BIO-SBI/results/exact_gw/reparam_bench
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=4
cd "$(dirname "$0")/../.."
while read -r cell arm rest; do
  [[ -z "${cell:-}" || "$cell" == \#* ]] && continue
  name=${arm//+/_}
  if [[ "$rest" =~ --seed\ ([0-9]+) ]]; then name="${name}_seed${BASH_REMATCH[1]}"; fi
  if [[ -f "$OUT/$cell/$name/diagnostics.json" ]]; then echo "skip $cell $name"; continue; fi
  echo "$(date +%H:%M:%S) start $cell $arm $rest"
  # shellcheck disable=SC2086
  uv run python scripts/exact_gw_reparam/run.py --cell "$cell" --arm "$arm" --out "$OUT" $rest \
    < /dev/null > /dev/null 2>&1 && echo "$(date +%H:%M:%S) done $cell $arm" \
    || echo "$(date +%H:%M:%S) FAILED $cell $arm"
done < "$1"
