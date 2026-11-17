#!/usr/bin/env bash
# Stop a running Hydra multirun when it LAUNCHES job number N (so jobs 0..N-1 complete and
# job N and later are left to another sweep). Kills the sweep's process group, then removes
# job N's just-created run folder if it holds no metrics.json.
#   scripts/stop_sweep_at.sh <sweep.log> <sweep dir> <pgid> <N>
set -uo pipefail
LOG=$1 DIR=$2 PGID=$3 N=$4
until grep -q "#$N : " "$LOG"; do
  kill -0 -- "-$PGID" 2>/dev/null || { echo "sweep $PGID ended before job #$N"; exit 0; }
  sleep 5
done
kill -- "-$PGID"
echo "$(date -Is) stopped sweep $PGID at launch of job #$N"
sleep 5
for d in "$DIR"/"$N"_*; do
  [ -d "$d" ] || continue
  if [ ! -f "$d/metrics.json" ]; then rm -rf -- "$d"; echo "removed unstarted run folder $d"; fi
done
