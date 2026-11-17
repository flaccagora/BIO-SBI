#!/usr/bin/env bash
# Run XG2-style exact-GW cells (preset exact_gw_data_size) from a job list, one single-job
# multirun per cell, written straight into the reorganised layout <root>/<gt>/n<n>_s<seed>/.
# A job starts when the CPU used by ALL of this user's processes (measured over 10 s)
# leaves room for 4 more threads within BUDGET, and fewer than MAXJOBS cells run. Cells
# whose folder or log already exists are skipped (restart-safe).
#   setsid nohup scripts/run_exact_gw_cells.sh <jobs file> [results root] [budget] [max jobs] [extra overrides...] &
# Jobs file: one "gt n seed" per line (# comments allowed). T5_C7 gets its CLI truth.
set -uo pipefail
JOBS_FILE=$1
ROOT=${2:-results/exact_gw/data_size}
BUDGET=${3:-18}
MAXJOBS=${4:-6}
shift $(( $# < 4 ? $# : 4 ))
EXTRA=("$@")
cd "$(dirname "$0")/.."
T5='+truths.T5_C7={source:reference,seed:null,values:{A:{death:0.05,persist:0.10,to_B:0.10,div_AA:0.50,div_AB:0.15,div_BB:0.10},B:{death:0.08,persist:0.05,to_A:0.10,div_AA:0.05,div_AB:0.05,div_BB:0.67}}}'
PAT="^[^ ]*python[^ ]* [^ ]*bio-sbi -m \+experiment=exact_gw_data_size"
njobs() { pgrep -fc "$PAT"; }
busy_cores() {  # summed %CPU of all processes of this user over a 10 s window, in cores
  top -b -n 2 -d 10 -u "$(id -un)" | awk '/^top -/{i++} i==2 && $1 ~ /^[0-9]+$/ {s+=$9} END {printf "%d", s/100 + 0.5}'
}
mkdir -p "$ROOT/_logs"
grep -v '^\s*#' "$JOBS_FILE" | while read -r gt n seed; do
  [ -z "${gt:-}" ] && continue
  dir="$ROOT/$gt/n${n}_s${seed}"
  log="$ROOT/_logs/${gt}_n${n}_s${seed}.log"
  if [ -e "$dir" ] || [ -e "$log" ]; then echo "$(date -Is) skip $gt $n $seed (exists)"; continue; fi
  while :; do
    nj=$(njobs); bc=$(busy_cores)
    if [ "$nj" -lt "$MAXJOBS" ] && [ $((bc + 4)) -le "$BUDGET" ]; then break; fi
    sleep 30
  done
  extra=()
  [ "$gt" = T5_C7 ] && extra=("$T5")
  setsid nohup .venv/bin/bio-sbi -m +experiment=exact_gw_data_size "${extra[@]}" "gt=$gt" \
    "design.clones=$n" "seed=$seed" inference.method.kernel=adjoint "${EXTRA[@]}" \
    hydra.sweep.dir="$ROOT/$gt" "hydra.sweep.subdir=n${n}_s${seed}" \
    > "$log" 2>&1 < /dev/null &
  echo "$(date -Is) launched $gt $n $seed -> $dir (before: $nj cells, $bc cores busy)"
  sleep 90  # let it ramp up before measuring again
done
echo "$(date -Is) all jobs launched"
