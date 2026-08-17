#!/usr/bin/env bash
# Run commands sequentially after an optional PID exits. Usage:
#   scripts/dev/queue.sh <wait_pid|0> "<cmd1>" "<cmd2>" ...
# Each command runs under caffeinate with 2 threads; logs to results/logs/queue_<n>.log
set -uo pipefail
cd "$(dirname "$0")/../.."
wait_pid="$1"; shift
if [ "$wait_pid" != "0" ]; then while kill -0 "$wait_pid" 2>/dev/null; do sleep 30; done; fi
i=0
for cmd in "$@"; do
  i=$((i+1))
  echo "[queue] $(date '+%H:%M:%S') start: $cmd" >> results/logs/queue.log
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 caffeinate -i bash -c "$cmd" > "results/logs/queue_$i.log" 2>&1
  echo "[queue] $(date '+%H:%M:%S') done ($?): $cmd" >> results/logs/queue.log
done
