#!/usr/bin/env bash
# R: train final candidate models at the real design (budget=full) and fit the CRC0322
# data variants (data/real/, scripts/prepare_real_data.py). Results: results/real/<model>/
# (training run) and results/real/<model>/fit_<variant>/ (bio-sbi-infer output).
set -u
cd "$(dirname "$0")/.."
T="torch_threads=4 workers=4"
declare -A MODEL=(
  [bh_rev]="simulator=bellman_harris law=reversible_ab"
  [bh_rev_s1]="simulator=bellman_harris law=reversible_ab seed=1"
  [bh_rev_founder]="simulator=bellman_harris law=reversible_ab ~law.fixed.founder"
  [bh_two]="simulator=bellman_harris law=two_type"
  [gw_rev]="simulator=galton_watson law=reversible_ab"
)
ORDER=${ORDER:-"bh_rev bh_rev_founder bh_two gw_rev bh_rev_s1"}
for m in $ORDER; do
  dir=results/real/$m
  if [ ! -f $dir/posterior.pt ]; then
    echo "== train $m start $(date +%H:%M)"
    uv run bio-sbi ${MODEL[$m]} design=real_like budget=full $T hydra.run.dir=$dir > results/real_$m.log 2>&1
    echo "== train $m exit=$? $(date +%H:%M)"
  fi
  sim=bh; [[ $m == gw* ]] && sim=gw
  variants="3d_mean 3d_sample1 3d_sample2 3d_sample3 2d"
  [[ $m == gw* ]] && variants="2d"
  for v in $variants; do
    uv run bio-sbi-infer $dir data/real/crc0322_${v}_${sim}.csv --out $dir/fit_$v --ppc 500 >> results/real_$m.log 2>&1
    echo "   fit $m $v exit=$?"
  done
done
