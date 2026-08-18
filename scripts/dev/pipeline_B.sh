#!/usr/bin/env bash
# Follow-ups that need the Van der Pol compile.
set -uo pipefail
cd "$(dirname "$0")/../.."
log() { echo "[pipeline-B] $(date '+%F %T') $*" >> results/logs/pipeline.log; }
run() { log "start: $*"; OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 caffeinate -i "$@" >> results/logs/pipeline_cmds.log 2>&1; log "done($?): $*"; }
until [ -f results/compile/vanderpol/compile_report.json ]; do sleep 120; done
run python3 experiments/synthetic/make_compiled_suite.py results/compile/lorenz63 results/compile/vanderpol
run nssc benchmark --suite compiled_vs_manual --device cpu
run python3 experiments/synthetic/run_ood.py --suite synthetic_core --dataset vanderpol --param mu --values 0.5 1.5 2.0 --ic-scales 2 4
run python3 scripts/generate_report.py
log "pipeline-B complete"
