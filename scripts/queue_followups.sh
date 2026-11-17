#!/usr/bin/env bash
# Exact-GW follow-ups, chained after the large-n scheduler (each stage starts when the
# previous stage has launched all its jobs; the CPU-aware scheduler keeps the total
# within the core budget):
#   1. bimodal reruns (8 chains, 1000 warmup, 32 starts)        -> data_size_reruns/bimodal
#   2. sampler noise (dataset 0, sampler seeds 1, 2)             -> data_size_reruns/sampler_noise
#   3. XG2 repeated with the flat B->A prior (108 cells)         -> data_size_flat_btoa
#   setsid nohup scripts/queue_followups.sh [results root exact_gw] &
set -uo pipefail
R=${1:-results/exact_gw}
cd "$(dirname "$0")/.."
wait_for() { until grep -q "all jobs launched" "$1" 2>/dev/null; do sleep 60; done; }
mkdir -p "$R/data_size_reruns/bimodal" "$R/data_size_reruns/sampler_noise" "$R/data_size_flat_btoa"
echo "$(date -Is) waiting for the large-n launches"
wait_for "$R/data_size/_logs_large_n_scheduler.log"
echo "$(date -Is) stage 1: bimodal reruns"
scripts/run_exact_gw_cells.sh scripts/jobs/xg2_bimodal.txt "$R/data_size_reruns/bimodal" 18 6 \
  inference.method.chains=8 inference.method.warmup=1000 inference.method.draws=500 \
  inference.method.init_starts=32 budget.posterior_samples=4000 \
  > "$R/data_size_reruns/bimodal/_scheduler.log" 2>&1
echo "$(date -Is) stage 2: sampler noise"
scripts/run_exact_gw_cells.sh scripts/jobs/xg2_sampler_noise.txt "$R/data_size_reruns/sampler_noise" 18 6 \
  data_seed=0 > "$R/data_size_reruns/sampler_noise/_scheduler.log" 2>&1
echo "$(date -Is) stage 3: flat B->A prior, 108 cells"
scripts/run_exact_gw_cells.sh scripts/jobs/xg2_flat_btoa.txt "$R/data_size_flat_btoa" 18 6 \
  prior=flat_btoa > "$R/data_size_flat_btoa/_scheduler.log" 2>&1
echo "$(date -Is) all follow-up jobs launched"
