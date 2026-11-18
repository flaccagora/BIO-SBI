#!/usr/bin/env bash
# Wait until a log file contains a line matching PATTERN, then run the given command.
#   setsid nohup scripts/queue_after.sh <log> <pattern> <command...> &
set -uo pipefail
LOG=$1 PATTERN=$2
shift 2
echo "$(date -Is) waiting for '$PATTERN' in $LOG"
until grep -q "$PATTERN" "$LOG" 2>/dev/null; do sleep 60; done
echo "$(date -Is) starting: $*"
exec "$@"
