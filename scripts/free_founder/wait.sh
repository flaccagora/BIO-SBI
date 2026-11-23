#!/bin/bash
# Poll the free-founder jobs (local + remote): return once >= TARGET results (json or err)
# exist in total, or after TIMEOUT seconds. Usage: wait.sh TARGET TIMEOUT
target=$1
timeout=$2
t0=$(date +%s)
LOCAL=/home/matteo/BIO-SBI/results/exact_gw/free_founder
REMOTE=matteo@172.28.107.111
count() {
  l=$(ls $LOCAL/*/*/*.json $LOCAL/*/*/*.err 2>/dev/null | wc -l)
  r=$(ssh -o ConnectTimeout=10 $REMOTE 'ls ~/bio-sbi-remote/results/free_founder/*/*/*.json ~/bio-sbi-remote/results/free_founder/*/*/*.err 2>/dev/null | wc -l')
  echo $((l + ${r:-0}))
}
while true; do
  c=$(count)
  now=$(date +%s)
  if [ "$c" -ge "$target" ] || [ $((now - t0)) -ge "$timeout" ]; then
    echo "count=$c"
    break
  fi
  sleep 60
done
ls $LOCAL/*/*/*.err 2>/dev/null
tail -3 $LOCAL/logs/*.log
ssh $REMOTE 'tail -3 ~/bio-sbi-remote/results/free_founder/logs/*.log; ls ~/bio-sbi-remote/results/free_founder/*/*/*.err 2>/dev/null'
