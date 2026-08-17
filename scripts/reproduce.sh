#!/usr/bin/env bash
# Reproduce the core results from a fresh clone.
#   scripts/reproduce.sh smoke        # seconds: tiny end-to-end + tiny compile
#   scripts/reproduce.sh compile      # hours: Lorenz-63 / Van der Pol / high-dim compiles
#   scripts/reproduce.sh benchmark    # hours: core synthetic benchmark (5 seeds)
#   scripts/reproduce.sh figures      # regenerate all figures + tables from results/
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
case "${1:-smoke}" in
  smoke)
    pytest -q -m "not slow"
    nssc smoke
    nssc compile --config configs/compiler/tiny.yaml --device cpu ;;
  compile)
    for c in lorenz63 vanderpol lorenz63_highdim; do
      nssc compile --config configs/compiler/$c.yaml --device cpu
    done ;;
  ablations)
    for c in configs/compiler/ablations/*.yaml; do nssc compile --config "$c" --device cpu; done ;;
  benchmark)
    nssc benchmark --suite synthetic_core --device cpu
    nssc tables --suite synthetic_core --reference mlpae_resmlp_d3 ;;
  figures)
    python scripts/generate_report.py ;;
  *) echo "unknown target $1"; exit 1 ;;
esac
