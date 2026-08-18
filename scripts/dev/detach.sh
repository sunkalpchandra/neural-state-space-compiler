#!/usr/bin/env bash
# Run a command fully detached (new session, survives parent shell/IDE exit), CPU-limited, awake.
# Usage: scripts/dev/detach.sh <logfile> <cmd...>
log="$1"; shift
cd "$(dirname "$0")/../.."
mkdir -p "$(dirname "$log")"
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 perl -MPOSIX -e 'POSIX::setsid(); exec @ARGV' -- \
  nohup caffeinate -i "$@" >> "$log" 2>&1 < /dev/null &
echo "detached pid $!"
