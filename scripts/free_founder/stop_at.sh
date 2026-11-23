#!/bin/bash
# Stop every free-founder run (local and remote) at the given time (default 12:00).
until [ "$(date +%H%M)" -ge "${1:-1200}" ]; do sleep 20; done
pkill -f queue_local.sh
pkill -f "scripts/free_founder/run.py"
ssh matteo@172.28.107.111 'pkill -f ff_queue_remote.sh; pkill -f scripts/free_founder/run.py'
echo "stopped at $(date)"
