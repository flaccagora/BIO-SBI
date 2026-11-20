#!/usr/bin/env bash
# Launch exact-Jeffreys reruns (scripts/jeffreys/jeffreys_rerun.py) for every XG2 cell whose
# PSIS reweighting Dirichlet(1/2) -> Jeffreys has Pareto k > 0.7 (iteration 1), and a second
# iteration where the rerun's own correction still has k > 0.7. CPU-aware like
# scripts/run_exact_gw_cells.sh (starts a 4-thread job when this user's processes leave room
# within BUDGET cores, at most MAXJOBS reruns); restart-safe (a log = launched). Exits when
# all TOTAL cells have PSIS summaries and nothing is pending or running.
#   setsid nohup scripts/jeffreys/run_reruns.sh [budget] [max jobs] [total] &
set -uo pipefail
BUDGET=${1:-16}
MAXJOBS=${2:-2}
TOTAL=${3:-108}
cd "$(dirname "$0")/../.."
J=/home/matteo/BIO-SBI/results/exact_gw/data_size_jeffreys
LOGS=$J/jeffreys_rerun/_logs
mkdir -p "$LOGS"
busy_cores() {
  top -b -n 2 -d 10 -u "$(id -un)" | awk '/^top -/{i++} i==2 && $1 ~ /^[0-9]+$/ {s+=$9} END {printf "%d", s/100 + 0.5}'
}
njobs() { pgrep -fc "python[^ ]* scripts/jeffreys/jeffreys_rerun.py"; }
pending() {  # "gt n seed iter" lines to launch
  .venv/bin/python - "$J" <<'EOF'
import json, sys
from pathlib import Path
J = Path(sys.argv[1])
for f in sorted((J / "jeffreys").glob("*/n*_s*/summary.json")):
    gt, cell = f.parent.parent.name, f.parent.name
    n, s = cell[1:].split("_s")
    if json.loads(f.read_text())["pareto_k"] <= 0.7:
        continue
    r = J / "jeffreys_rerun" / gt / cell / "summary.json"
    logs = J / "jeffreys_rerun" / "_logs"
    if not (logs / f"{gt}_{cell}_iter1.log").exists():
        print(gt, n, s, 1)
    elif r.exists():
        d = json.loads(r.read_text())
        if d["pareto_k"] > 0.7 and d["iteration"] == 1 and not (logs / f"{gt}_{cell}_iter2.log").exists():
            print(gt, n, s, 2)
EOF
}
while :; do
  todo=$(pending)
  if [ -z "$todo" ]; then
    done_cells=$(ls "$J"/jeffreys/*/n*_s*/summary.json 2>/dev/null | wc -l)
    if [ "$done_cells" -ge "$TOTAL" ] && [ "$(njobs)" -eq 0 ]; then break; fi
    sleep 120; continue
  fi
  read -r gt n seed it <<< "$(echo "$todo" | head -1)"
  while :; do
    nj=$(njobs); bc=$(busy_cores)
    if [ "$nj" -lt "$MAXJOBS" ] && [ $((bc + 4)) -le "$BUDGET" ]; then break; fi
    sleep 30
  done
  log="$LOGS/${gt}_n${n}_s${seed}_iter${it}.log"
  PYTHONWARNINGS=ignore setsid nohup .venv/bin/python scripts/jeffreys/jeffreys_rerun.py \
    "$gt" "$n" "$seed" --iter "$it" > "$log" 2>&1 < /dev/null &
  echo "$(date -Is) launched $gt $n $seed iter $it (before: $nj reruns, $bc cores busy)"
  sleep 90
done
echo "$(date -Is) all reruns done"
