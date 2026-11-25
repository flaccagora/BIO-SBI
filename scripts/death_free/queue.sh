#!/bin/bash
# Death-free dataset D0: MLE (16 workers x 1 thread), then NUTS (4 workers x 4 threads),
# local machine only. Restartable (finished / locked jobs are skipped; delete stale .lock
# files after a crash).
cd "$(dirname "$0")/../.." || exit 1
LOG=/home/matteo/BIO-SBI/results/exact_gw/death_free/logs
mkdir -p $LOG
uv run python scripts/death_free/run.py mle --seeds 0-9 --workers 16 >> $LOG/mle.log 2>&1
uv run python scripts/death_free/run.py nuts --jobs D0:free,D0:nodeath,orig:free --workers 4 \
  --threads 4 >> $LOG/nuts.log 2>&1
echo "queue finished $(date)" >> $LOG/nuts.log
