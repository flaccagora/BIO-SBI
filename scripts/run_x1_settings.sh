#!/usr/bin/env bash
# X1: inference methods on the decisive settings. One lane per method; settings run in
# sequence inside a lane. Usage: scripts/run_x1.sh <npe|snpe|nle|nre> [extra overrides]
# Results: results/x1/<setting>/<method>/ (5 seeds each), logs results/x1_<method>.log
set -u
method=$1; shift
cd "$(dirname "$0")/.."
case $method in
  npe)  M="inference=npe" ;;
  snpe) M="inference=npe inference.method.rounds=4 budget.sbc_cases=0" ;;
  nle)  M="inference=nle budget.sbc_cases=50" ;;
  nre)  M="inference=nre budget.sbc_cases=50" ;;
  *) echo "unknown method $method"; exit 2 ;;
esac
S="seed=0,1,2,3,4 $*"
declare -A SET=(
  [e0_free]="simulator=galton_watson law=two_type design=default budget=dev"
  [e0_fixed]="simulator=galton_watson law=reversible_ab design=default budget=dev"
  [bh_ridge]="+experiment=x5b_lifetime_prior"
  [silent_short_lifetime]="+experiment=e3a_lifetime_fixed lifetime_setting={B:30,T:18.75}"
  [loud_wrong_lifetime_prior]="+experiment=x5b_lifetime_prior ~simulator.model.lifetime_mean_bounds +simulator.model.lifetime_mean_bounds={A:[24,37.5],B:[36,56.25]}"
)
for s in ${SETTINGS:-e0_free e0_fixed bh_ridge silent_short_lifetime loud_wrong_lifetime_prior}; do
  echo "== $s $method start $(date +%H:%M)"
  uv run bio-sbi -m ${SET[$s]} $M $S hydra.sweep.dir=results/x1/$s/$method
  echo "== $s $method exit=$? done=$(find results/x1/$s/$method -name metrics.json | wc -l)"
done
