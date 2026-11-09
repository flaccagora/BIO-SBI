#!/usr/bin/env bash
# Exact likelihood (GW, BH) vs NPE speed benchmark. Run it ALONE on the machine.
#
#   scripts/run_exact_benchmark.sh [OUT_DIR] [PROFILE]     # defaults: results/exact_benchmark full
#   BENCH_ONLY='^bh' scripts/run_exact_benchmark.sh        # only jobs matching a regex
#   BENCH_NO_WAIT=1 scripts/run_exact_benchmark.sh /tmp/x smoke   # smoke test, no waiting
#
# One job at a time (scripts/benchmark_exact.py run JOB, own thread count). Before each
# job it waits until the machine is free: no other bio-sbi / bio_sbi / pytest / BIO-SBI
# python process and foreign CPU use below $FOREIGN_CPU_MAX % (sum of %CPU of processes
# outside our job, from top). While the job runs it checks every 15 s and logs
# CONTENTION lines; each job is marked clean=1/0 in load.log and jobs.csv.
# Outputs: timings.csv, jobs.csv, load.log, info.md, summary.{csv,md}, npe/ (runs).
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
R=$(realpath -m "${1:-$REPO/results/exact_benchmark}")
PROFILE=${2:-full}
FOREIGN_CPU_MAX=${FOREIGN_CPU_MAX:-50}
mkdir -p "$R"
cd "$REPO"
PY=(uv run --project "$REPO" python "$REPO/scripts/benchmark_exact.py")

# production exact-GW kernel: from the package if merged, else a snapshot of origin/exact-gw
if ! uv run python -c "import bio_sbi.inference.gw_exact" 2>/dev/null; then
  if git show origin/exact-gw:src/bio_sbi/inference/gw_exact.py > "$R/gw_exact_snapshot.py" 2>/dev/null; then
    export BENCH_GW_EXACT_FILE="$R/gw_exact_snapshot.py"
    GWX="snapshot of origin/exact-gw $(git rev-parse --short origin/exact-gw)"
  else
    rm -f "$R/gw_exact_snapshot.py"; GWX="not available (gwprod jobs skipped)"
  fi
else
  GWX="bio_sbi.inference.gw_exact (installed package)"
fi

# --- contention checks ------------------------------------------------------------------
TOPF="$R/.top_snapshot"
ours() {  # pids of this runner, its ancestors (the launching shell) and all its descendants
  ps -eo pid=,ppid= | awk -v me=$$ '
    {par[$1] = $2; kids[$2] = kids[$2] " " $1}
    END {
      for (p = me; p > 1; p = par[p]) print p
      n = split(kids[me], q, " ")
      while (n > 0) { c = q[n--]; print c; m = split(kids[c], k, " "); for (i = 1; i <= m; i++) q[++n] = k[i] }
    }'
}
others() {  # other BIO-SBI-like python jobs (not ours)
  local mine; mine=" $(ours | tr '\n' ' ') "
  ps -eo pid=,args= | grep -E "[b]io-sbi|[b]io_sbi|[p]ytest|[B]IO-SBI.*python|python.*[B]IO-SBI" \
    | grep python | while read -r pid args; do
        case "$mine" in *" $pid "*) ;; *) echo "$pid ${args:0:110}";; esac
      done
}
foreign_cpu() {  # current %CPU (1 s top sample) of all processes not ours; saves the sample
  local mine; mine=" $(ours | tr '\n' ' ') "
  top -b -n 2 -d 1 -w 512 | awk -v mine="$mine" '
    /^top -/ {blk++} blk==2 && $1 ~ /^[0-9]+$/ && $12 != "top" {
      tag = (index(mine, " " $1 " ") == 0) ? "" : "[ours]"; print $9, $2, $12 tag }' > "$TOPF"
  awk '$3 !~ /\[ours\]$/ {s += $1} END {printf "%.0f", s}' "$TOPF"
}
top_procs() { sort -rn "$TOPF" 2>/dev/null | head -4 | paste -sd';'; }  # from the last sample
busy() {  # BENCH_NO_WAIT=1 (smoke tests only): never wait, still log contention
  [ "${BENCH_NO_WAIT:-0}" = 1 ] && return 1
  [ -n "$(others)" ] || [ "$(foreign_cpu)" -gt "$FOREIGN_CPU_MAX" ]
}

waited=0
while busy; do
  [ $((waited % 300)) -eq 0 ] && echo "$(date +%H:%M:%S) waiting: machine busy (foreign cpu $(foreign_cpu)%; $(others | head -2 | paste -sd'|'))" | tee -a "$R/load.log"
  sleep 30; waited=$((waited + 30))
done
echo "$(date '+%F %H:%M:%S') machine free; profile=$PROFILE out=$R; gw_exact: $GWX" | tee -a "$R/load.log"
[ -f "$R/jobs.csv" ] || echo "job,start,end,exit,clean,load_start,foreign_cpu_start,max_foreign_cpu" > "$R/jobs.csv"

if [ ! -f "$R/info.md" ]; then
  cat > "$R/info.md" <<EOF
# Exact likelihood vs NPE: speed benchmark

- Started: $(date '+%F %H:%M:%S') on $(hostname), $(nproc) cores ($(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //'))
- Branch/commit: $(git rev-parse --abbrev-ref HEAD) $(git rev-parse --short HEAD); profile: $PROFILE
- Exact GW production kernel: $GWX
- Command: scripts/run_exact_benchmark.sh $R $PROFILE
- Each job runs alone (contention checked every 15 s, see load.log; clean flag in jobs.csv).

## Question
How long does each method take per task: one likelihood evaluation, one posterior for one
dataset, and posteriors for 100 datasets (e.g. SBC)? Exact = gwexact.py / profile.py
(GW, numpy), bio_sbi gw_exact (GW, numba, data-sized grids), bhexact.py (BH, numba
DOP853); NPE = bio-sbi pipeline (simulation + training once, then amortised sampling).

## Settings
- Truth: reference rows (A .03/.03/.05/.55/.05/.29; B death .1, persist .1, to_A .8);
  BH means A 20 h, B 30 h, Gamma shape 8 (and 4).
- Designs: real_like (GW gens 6/10, BH 144/240 h; 165/101 clones), E0-4096 (GW gens
  4/6/10 x 4096 clones; law two_type for the production kernel).
- Exact MCMC cost model (derive defaults): MH/slice 4 chains x 2000 law evals; NUTS
  4 x 1000 x 10 law+gradient evals. Change with \`benchmark_exact.py derive --mh-evals
  --grad-evals\`.
- Skipped (> 5 min or memory): gwexact Jacobian at L = 2048 (~5 GB complex arrays);
  BH Jacobian only at 768x512 with 20 threads; NPE full budget only at 20 threads
  (BENCH_NPE_FULL_THREADS="4 20" adds 4).

## Results
(summary.md: derived table; timings.csv: raw; fill in conclusions here.)
EOF
fi

run() {  # job id
  local job=$1 waited=0
  while busy; do
    [ $((waited % 300)) -eq 0 ] && echo "$(date +%H:%M:%S) waiting before $job: foreign cpu $(foreign_cpu)%; $(others | head -2 | paste -sd'|')" | tee -a "$R/load.log"
    sleep 30; waited=$((waited + 30))
  done
  local load fc; load=$(cut -d' ' -f1-3 /proc/loadavg); fc=$(foreign_cpu)
  local start; start=$(date +%s)
  echo "$(date +%H:%M:%S) start $job load=$load foreign_cpu=$fc% top=$(top_procs)" | tee -a "$R/load.log"
  "${PY[@]}" run "$job" --out "$R" >> "$R/jobs.log" 2>&1 &
  local pid=$! clean=1 maxfc=$fc tick=0
  while kill -0 "$pid" 2>/dev/null; do  # watch for other jobs every 15 s while this one runs
    if [ $((tick++ % 15)) -ne 0 ]; then sleep 1; continue; fi
    local o c; o=$(others); c=$(foreign_cpu)
    [ "$c" -gt "$maxfc" ] && maxfc=$c
    if [ -n "$o" ] || [ "$c" -gt "$FOREIGN_CPU_MAX" ]; then
      clean=0
      echo "$(date +%H:%M:%S) CONTENTION during $job: foreign_cpu=$c% load=$(cut -d' ' -f1-3 /proc/loadavg) top=$(top_procs) jobs=$(echo "$o" | head -3 | paste -sd'|')" | tee -a "$R/load.log"
    fi
  done
  wait "$pid"; local status=$?
  echo "$(date +%H:%M:%S) end $job exit=$status clean=$clean wall=$(( $(date +%s) - start ))s" | tee -a "$R/load.log"
  echo "\"$job\",$start,$(date +%s),$status,$clean,\"$load\",$fc,$maxfc" >> "$R/jobs.csv"
}

T0=$(date +%s)
for job in $("${PY[@]}" jobs --profile "$PROFILE"); do
  if [ -n "${BENCH_ONLY:-}" ] && ! [[ $job =~ $BENCH_ONLY ]]; then continue; fi
  run "$job"
done
echo "$(date +%H:%M:%S) benchmark finished in $(( ($(date +%s) - T0) / 60 )) min" | tee -a "$R/load.log"
"${PY[@]}" derive --out "$R" > "$R/derive.log" 2>&1 && echo "summary: $R/summary.md" | tee -a "$R/load.log"
