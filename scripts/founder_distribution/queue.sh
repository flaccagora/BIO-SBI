#!/bin/bash
# Local job queue of the founder-distribution simulation check (restartable: run.py skips
# finished / locked jobs; delete stale .lock files after a crash).
#   queue.sh nuts      one NUTS runner (1 job x 4 threads) over the ordered NUTS list
#   queue.sh mle       MLE runner (4 workers x 1 thread), then a second NUTS runner
#   queue.sh extra     waits until the parallel convergence experiment has finished, then
#                      a third and fourth NUTS runner (8 more threads; 16 in total)
cd "$(dirname "$0")/../.." || exit 1
LOG=/home/matteo/BIO-SBI/results/exact_gw/founder_distribution/logs
mkdir -p $LOG
NUTS="T1_ref:S2_pA0.7:est,T1_ref:S2_pA0.7:known,T1_ref:S1_pA1.0:est,T1_ref:S1_pA1.0:known,\
P3:S1_pA1.0:est,P3:S1_pA1.0:known,T1_ref:S3_pA0.9:est,T1_ref:S3_pA0.9:known,\
P3:S2_pA0.7:est,P3:S2_pA0.7:known,T1_ref:S4_pA0.5:est,T1_ref:S4_pA0.5:known,\
P3:S3_pA0.9:est,P3:S3_pA0.9:known,T3_Bdiv:S2_pA0.7:est,T3_Bdiv:S2_pA0.7:known"
MLE=""
for gt in T1_ref P3 T3_Bdiv; do
  for d in S1_pA1.0 S2_pA0.7 S4_pA0.5 S3_pA0.9; do
    for m in est known; do MLE="$MLE,$gt:$d:$m"; done
  done
done
MLE=${MLE#,}
run_nuts() {
  uv run python scripts/founder_distribution/run.py nuts --jobs $NUTS --workers 1 --threads 4 \
    >> $LOG/nuts_$1.log 2>&1
}
case $1 in
  nuts) run_nuts 1 ;;
  mle)
    uv run python scripts/founder_distribution/run.py mle --jobs $MLE --seeds 0,1,2,3,4,5 \
      --workers 4 --threads 1 >> $LOG/mle.log 2>&1
    run_nuts 2 ;;
  extra)
    while pgrep -f scripts/convergence > /dev/null; do sleep 120; done
    run_nuts 3 &
    run_nuts 4
    wait ;;
esac
