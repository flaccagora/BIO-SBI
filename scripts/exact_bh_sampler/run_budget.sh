#!/bin/bash
# sequential runs at a fixed budget (2 threads): NAME, SEED, BUDGET, DT, RUNS
W=$(git rev-parse --show-toplevel)
OUT=${OUT:-$W/scratch/out}
mkdir -p $OUT
NAME=${NAME:-small0}
SEED=${SEED:-0}
BUDGET=${BUDGET:-40000}
DT=${DT:-2.0}
for R in ${RUNS:-none stable+probit}; do
  tag=$(echo $R | tr '+' '_')
  pre=s${SEED}
  [ "$NAME" != small0 ] && pre=${NAME}_s${SEED}
  nice -n 10 env NUMBA_NUM_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=$W/src:$W/scripts/exact_bh_sampler python scripts/exact_bh_sampler/drive.py $NAME $OUT/${pre}_${tag}.json \
    threads=2 seed=$SEED dt=$DT reparam=$R max_evals=$BUDGET target_ess=100000 \
    > $OUT/${pre}_${tag}.log 2>&1
  echo "done $NAME $R $(date)"
done
