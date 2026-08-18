#!/usr/bin/env bash
# Follow-ups that only need the Lorenz-63 compile (runs now); pipeline_B.sh needs Van der Pol.
set -uo pipefail
cd "$(dirname "$0")/../.."
log() { echo "[pipeline-A] $(date '+%F %T') $*" >> results/logs/pipeline.log; }
run() { log "start: $*"; OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 caffeinate -i "$@" >> results/logs/pipeline_cmds.log 2>&1; log "done($?): $*"; }
run python3 experiments/synthetic/make_compiled_suite.py results/compile/lorenz63
run nssc benchmark --suite compiled_vs_manual --device cpu
run nssc compile --config configs/compiler/ablations/lorenz63_valmse.yaml --device cpu
run nssc compile --config configs/compiler/ablations/lorenz63_nostability.yaml --device cpu
run nssc compile --config configs/compiler/ablations/lorenz63_nocomplexity.yaml --device cpu
run nssc compile --config configs/compiler/ablations/lorenz63_rollout_only.yaml --device cpu
run python3 experiments/synthetic/run_ood.py --suite synthetic_core --dataset lorenz63 --param rho --values 20 24 32 35 --ic-scales 2 4
run nssc compile --config configs/compiler/lorenz63_highdim.yaml --device cpu
run nssc compile --config configs/compiler/ablations/lorenz63_multiscale.yaml --device cpu
run nssc compile --config configs/compiler/lorenz96.yaml --device cpu
run nssc compile --config configs/compiler/eegbci.yaml --device cpu
run nssc benchmark --suite real_eegbci --device cpu
log "pipeline-A complete"
