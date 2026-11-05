#!/usr/bin/env bash
# Run the device benchmark with no other BIO-SBI job on the machine, one job at a time.
# Waits until no other bio-sbi process is running, then runs each configuration as a
# separate single run, logging the 1-minute load average at the start of each job.
set -u
R=/home/matteo/BIO-SBI/results/gpu_bench
mkdir -p "$R"
others() { ps -eo pid=,args= | grep -E "[b]io-sbi|[p]ytest" | grep -v "device_benchmark" | grep -c "python"; }
while [ "$(others)" -gt 0 ]; do sleep 30; done
echo "machine free of other bio-sbi jobs at $(date +%H:%M:%S)" | tee "$R/load.log"
run() {  # name, overrides...
  local name=$1; shift
  while [ "$(others)" -gt 0 ]; do sleep 30; done
  echo "$(date +%H:%M:%S) start $name load=$(cut -d' ' -f1-3 /proc/loadavg) top=$(ps -eo pcpu=,user=,comm= --sort=-pcpu | head -3 | tr -s ' ' | paste -sd';')" | tee -a "$R/load.log"
  uv run bio-sbi +experiment=device_benchmark "$@" hydra.run.dir="$R/$name" > "$R/$name.log" 2>&1
  echo "$(date +%H:%M:%S) end $name exit=$?" | tee -a "$R/load.log"
}
run amortized_cpu_b128 device=cpu inference.method.training.training_batch_size=128
run amortized_cpu_b512 device=cpu inference.method.training.training_batch_size=512
run amortized_gpu_b128 device=cuda inference.method.training.training_batch_size=128
run amortized_gpu_b512 device=cuda inference.method.training.training_batch_size=512
run snpe_cpu_b128 device=cpu inference.method.rounds=4 inference.method.training.training_batch_size=128
run snpe_gpu_b128 device=cuda inference.method.rounds=4 inference.method.training.training_batch_size=128
echo "benchmark finished at $(date +%H:%M:%S)" | tee -a "$R/load.log"
