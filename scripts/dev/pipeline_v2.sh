#!/usr/bin/env bash
# Protocol-v2 re-runs and follow-ups, in dependency order. Everything is registry/state-resumable:
# baseline runs are reused by config hash, only latent runs retrain.
set -uo pipefail
cd "$(dirname "$0")/../.."
log() { echo "[v2] $(date '+%F %T') $*" >> results/logs/pipeline.log; }
run() { log "start: $*"; OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 caffeinate -i "$@" >> results/logs/pipeline_cmds.log 2>&1; log "done($?): $*"; }
run nssc benchmark --suite synthetic_core --device cpu
run nssc tables --suite synthetic_core --reference mlpae_resmlp_d3
run nssc pareto --suite synthetic_core
run nssc compile --config configs/compiler/lorenz63.yaml --device cpu
run nssc compile --config configs/compiler/vanderpol.yaml --device cpu
run nssc compile --config configs/compiler/lorenz63_highdim.yaml --device cpu
run python3 experiments/synthetic/make_compiled_suite.py results/compile/lorenz63 results/compile/vanderpol results/compile/lorenz63_highdim
run nssc benchmark --suite compiled_vs_manual --device cpu
run nssc compile --config configs/compiler/ablations/lorenz63_valmse.yaml --device cpu
run nssc compile --config configs/compiler/ablations/lorenz63_nostability.yaml --device cpu
run nssc compile --config configs/compiler/ablations/lorenz63_nocomplexity.yaml --device cpu
run nssc compile --config configs/compiler/ablations/lorenz63_rollout_only.yaml --device cpu
run nssc benchmark --suite ablation_stability_reg --device cpu
run nssc benchmark --suite baseline_rollout_control --device cpu
run python3 experiments/synthetic/run_ood.py --suite synthetic_core --dataset lorenz63 --param rho --values 20 24 32 35 --ic-scales 2 4
run python3 experiments/synthetic/run_ood.py --suite synthetic_core --dataset vanderpol --param mu --values 0.5 1.5 2.0 --ic-scales 2 4
run nssc compile --config configs/compiler/ablations/lorenz63_multiscale.yaml --device cpu
run nssc compile --config configs/compiler/lorenz96.yaml --device cpu
run nssc compile --config configs/compiler/eegbci.yaml --device cpu
run nssc benchmark --suite real_eegbci --device cpu
run python3 scripts/generate_report.py
log "pipeline-v2 complete"
