#!/bin/bash
# After the main local job list: the reduced control F70 with pi_2 free (does the model
# invent doublets?) on T1_ref (running on the remote), T3_Bdiv, T4_slowA, P3.
# base:fA+d2 and dataset 1 were dropped (deadline). All runs are stopped at 12:00.
cd "$(dirname "$0")/../.."
LOG=/home/matteo/BIO-SBI/results/exact_gw/free_founder/logs
while pgrep -f "run.py --jobs D15mix:fA\+d2,F70:fA,D15A" > /dev/null; do sleep 30; done
uv run python scripts/free_founder/run.py --jobs F70:fA+d2 --gts T1_ref,T3_Bdiv,T4_slowA,P3 \
  --seeds 0 --workers 4 --threads 4 > $LOG/local_s0_control.log 2>&1
