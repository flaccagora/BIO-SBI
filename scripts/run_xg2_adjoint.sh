#!/usr/bin/env bash
# Remaining XG2 runs with the adjoint likelihood kernel (inference.method.kernel=adjoint),
# one single-job multirun per cell (sweep dir <root>/adj_<gt>_n<n>_s<seed>), 4 threads each.
# A job starts when the running exact_gw_data_size sweeps use at most BUDGET - 4 cores
# (measured over 10 s, so jobs whose last chain is still sampling free their idle
# threads) and fewer than MAXJOBS sweeps run. Cells already launched (log exists) are
# skipped, so the script can be restarted safely.
#   setsid nohup scripts/run_xg2_adjoint.sh [results root] [core budget] [max jobs] &
set -uo pipefail
ROOT=${1:-results/exact_gw/data_size}
BUDGET=${2:-18}
MAXJOBS=${3:-6}
cd "$(dirname "$0")/.."
T5='+truths.T5_C7={source:reference,seed:null,values:{A:{death:0.05,persist:0.10,to_B:0.10,div_AA:0.50,div_AB:0.15,div_BB:0.10},B:{death:0.08,persist:0.05,to_A:0.10,div_AA:0.05,div_AB:0.05,div_BB:0.67}}}'
JOBS=(  # gt n seed, largest first
  "T5_C7 1024 0" "T5_C7 1024 1" "T5_C7 1024 2"
  "P3 1024 0" "P3 1024 1" "P3 1024 2"
  "T5_C7 512 2" "P3 512 2" "P3 256 2"
)
PAT="^[^ ]*python[^ ]* [^ ]*bio-sbi -m \+experiment=exact_gw_data_size"
njobs() { pgrep -fc "$PAT"; }
busy_cores() {  # summed %CPU of the sweep processes over a 10 s window, in cores
  local pids
  pids=$(pgrep -f "$PAT" | paste -sd, -)
  [ -z "$pids" ] && { echo 0; return; }
  top -b -n 2 -d 10 -p "$pids" | awk '/^top -/{i++} i==2 && $1 ~ /^[0-9]+$/ {s+=$9} END {printf "%d", s/100 + 0.5}'
}
for job in "${JOBS[@]}"; do
  read -r gt n seed <<< "$job"
  dir="$ROOT/adj_${gt}_n${n}_s${seed}"
  if [ -e "$dir.log" ]; then echo "$(date -Is) skip $job (already launched)"; continue; fi
  while :; do
    nj=$(njobs); bc=$(busy_cores)
    if [ "$nj" -lt "$MAXJOBS" ] && [ $((bc + 4)) -le "$BUDGET" ]; then break; fi
    sleep 30
  done
  extra=()
  [ "$gt" = T5_C7 ] && extra=("$T5")
  setsid nohup .venv/bin/bio-sbi -m +experiment=exact_gw_data_size "${extra[@]}" "gt=$gt" \
    "design.clones=$n" "seed=$seed" inference.method.kernel=adjoint \
    hydra.sweep.dir="$dir" > "$dir.log" 2>&1 < /dev/null &
  echo "$(date -Is) launched $job -> $dir (pid $!; before: $nj sweeps, $bc cores busy)"
  sleep 60  # let the new job ramp up before measuring again
done
echo "$(date -Is) all jobs launched"
