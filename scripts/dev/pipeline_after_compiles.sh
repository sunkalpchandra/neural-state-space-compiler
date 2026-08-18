#!/usr/bin/env bash
# Waits for the Lorenz-63 and Van der Pol compiles to finish, then runs the follow-up
# experiments sequentially. Idempotent: every step is registry/state-resumable.
set -uo pipefail
cd "$(dirname "$0")/../.."
log() { echo "[pipeline] $(date '+%F %T') $*" >> results/logs/pipeline.log; }
until [ -f results/compile/lorenz63/compile_report.json ] && [ -f results/compile/vanderpol/compile_report.json ]; do sleep 120; done
log "compiles done"
run() { log "start: $*"; OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 caffeinate -i "$@" >> results/logs/pipeline_cmds.log 2>&1; log "done($?): $*"; }
run python3 experiments/synthetic/make_compiled_suite.py results/compile/lorenz63 results/compile/vanderpol
run nssc benchmark --suite compiled_vs_manual --device cpu
run nssc compile --config configs/compiler/ablations/lorenz63_valmse.yaml --device cpu
run nssc compile --config configs/compiler/ablations/lorenz63_nostability.yaml --device cpu
run nssc compile --config configs/compiler/ablations/lorenz63_nocomplexity.yaml --device cpu
run nssc compile --config configs/compiler/ablations/lorenz63_rollout_only.yaml --device cpu
run python3 experiments/synthetic/run_ood.py --suite synthetic_core --dataset lorenz63 --param rho --values 20 24 32 35 --ic-scales 2 4
run python3 experiments/synthetic/run_ood.py --suite synthetic_core --dataset vanderpol --param mu --values 0.5 1.5 2.0 --ic-scales 2 4
run nssc compile --config configs/compiler/lorenz63_highdim.yaml --device cpu
run nssc compile --config configs/compiler/ablations/lorenz63_multiscale.yaml --device cpu
run nssc compile --config configs/compiler/lorenz96.yaml --device cpu
run nssc compile --config configs/compiler/eegbci.yaml --device cpu
run nssc benchmark --suite real_eegbci --device cpu
run python3 scripts/generate_report.py
log "pipeline complete"
