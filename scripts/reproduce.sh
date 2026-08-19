#!/usr/bin/env bash
# Reproduce results from a fresh clone. Model weights (*.pt) are NOT in git, so any target that
# needs a model retrains it; metrics, tables, figures, the registry ledger and the search states
# are committed, so `figures` and `tables` work without training.
#
#   scripts/reproduce.sh smoke       # minutes: tests + tiny end-to-end run + tiny compile
#   scripts/reproduce.sh benchmark   # hours:  synthetic_core suite (3 systems × 10 models × 5 seeds)
#   scripts/reproduce.sh compile     # hours:  Lorenz-63 / Van der Pol / high-dim compiler runs
#   scripts/reproduce.sh ablations   # hours:  objective ablations (share training runs via the registry)
#   scripts/reproduce.sh tables      # seconds: regenerate every table from results/registry.jsonl
#   scripts/reproduce.sh figures     # minutes: regenerate every figure + results/SUMMARY.md
#   scripts/reproduce.sh all
#
# Everything is resumable and idempotent: a run already in the registry with the same config hash
# and protocol version is reused, and a cached run whose weights are missing is retrained.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
DEV=${NSSC_DEVICE:-cpu}

smoke()      { pytest -q -m "not slow"; nssc smoke; nssc compile --config configs/compiler/tiny.yaml --device "$DEV"; }
benchmark()  { nssc benchmark --suite synthetic_core --device "$DEV"; tables; }
compile_()   { for c in lorenz63 vanderpol lorenz63_highdim; do nssc compile --config "configs/compiler/$c.yaml" --device "$DEV"; done; }
ablations()  { for c in configs/compiler/ablations/*.yaml; do nssc compile --config "$c" --device "$DEV"; done; }
tables()     { nssc tables --suite synthetic_core --reference mlpae_resmlp_d3
               nssc pareto --suite synthetic_core
               nssc failures --suite synthetic_core; }
figures()    { python scripts/generate_report.py; }

case "${1:-smoke}" in
  smoke) smoke ;;
  benchmark) benchmark ;;
  compile) compile_ ;;
  ablations) ablations ;;
  tables) tables ;;
  figures) figures ;;
  all) smoke; benchmark; compile_; ablations; figures ;;
  *) echo "unknown target: $1"; sed -n '2,20p' "$0"; exit 1 ;;
esac
