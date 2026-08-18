#!/usr/bin/env bash
# One-screen progress summary of background experiments.
cd "$(dirname "$0")/../.."
echo "commits: $(git log --oneline | wc -l | tr -d ' ')   registry: $(grep -c '' results/registry.jsonl 2>/dev/null) lines"
for c in results/compile/*/search_state.json; do
  python3 - "$c" <<'PY'
import json,sys; p=sys.argv[1]; s=json.load(open(p)); print(f"{p.split('/')[2]:18s} runs={len(s['runs']):3d} stages={list(s['stages'])}")
PY
done
for l in results/logs/bench_*.log; do echo "$(basename $l): $(grep -c completed $l) completed, $(grep -c ' skip' $l) skipped, $(grep -c failed $l) failed"; done
ps -Ao etime,command | grep -E "[n]ssc (compile|benchmark|train)" | sed 's#/Library.*/bin/##' | cut -c1-100
