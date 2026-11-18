#!/usr/bin/env bash
# Wait until process PID exits, then stop the scheduler process(es) matching SCHED_PATTERN
# and start the given command (e.g. the same scheduler with more slots; it skips cells
# already launched, so in-flight runs are untouched).
#   setsid nohup scripts/bump_after_pid.sh <pid> <scheduler pattern> <command...> &
set -uo pipefail
PID=$1 PATTERN=$2
shift 2
echo "$(date -Is) waiting for pid $PID"
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
echo "$(date -Is) pid $PID ended; restarting scheduler"
pkill -f -- "^bash [^ ]*$PATTERN" || true
sleep 2
exec "$@"
