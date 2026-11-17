#!/usr/bin/env bash
# Remaining XG2 runs with the adjoint likelihood kernel (inference.method.kernel=adjoint),
# one single-job multirun per cell (sweep dir <root>/adj_<gt>_n<n>_s<seed>), at most MAX
# exact_gw_data_size sweeps running at once (older sweeps count too), 4 threads each.
#   setsid nohup scripts/run_xg2_adjoint.sh [results root] [max concurrent] &
set -uo pipefail
ROOT=${1:-results/exact_gw/data_size}
MAX=${2:-4}
cd "$(dirname "$0")/.."
T5='+truths.T5_C7={source:reference,seed:null,values:{A:{death:0.05,persist:0.10,to_B:0.10,div_AA:0.50,div_AB:0.15,div_BB:0.10},B:{death:0.08,persist:0.05,to_A:0.10,div_AA:0.05,div_AB:0.05,div_BB:0.67}}}'
JOBS=(  # gt n seed, largest first
  "T5_C7 1024 0" "T5_C7 1024 1" "T5_C7 1024 2"
  "P3 1024 0" "P3 1024 1" "P3 1024 2"
  "T5_C7 512 2" "P3 512 2" "P3 256 2"
)
running() { pgrep -fc "python.*bio-sbi -m \+experiment=exact_gw_data_size"; }
for job in "${JOBS[@]}"; do
  read -r gt n seed <<< "$job"
  dir="$ROOT/adj_${gt}_n${n}_s${seed}"
  if find "$dir" -name metrics.json 2>/dev/null | grep -q .; then
    echo "$(date -Is) skip $job (done)"; continue
  fi
  while [ "$(running)" -ge "$MAX" ]; do sleep 30; done
  extra=()
  [ "$gt" = T5_C7 ] && extra=("$T5")
  setsid nohup .venv/bin/bio-sbi -m +experiment=exact_gw_data_size "${extra[@]}" "gt=$gt" \
    "design.clones=$n" "seed=$seed" inference.method.kernel=adjoint \
    hydra.sweep.dir="$dir" > "$dir.log" 2>&1 < /dev/null &
  echo "$(date -Is) launched $job -> $dir (pid $!)"
  sleep 20  # let the process appear before counting again
done
echo "$(date -Is) all jobs launched"
