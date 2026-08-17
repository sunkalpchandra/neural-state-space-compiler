#!/usr/bin/env bash
# Every N seconds: commit new run artifacts (one commit per run) + registry snapshot, push.
# Usage: scripts/dev/autocommit_runs.sh [interval_s]
cd "$(dirname "$0")/../.."
int="${1:-600}"
while true; do
  python3 scripts/dev/commit_runs.py >> results/logs/autocommit.log 2>&1
  if ! git diff --quiet -- results/registry.jsonl 2>/dev/null || [ -n "$(git status --porcelain results/registry.jsonl)" ]; then
    git add -f results/registry.jsonl && git commit -qm "exp: registry ledger snapshot ($(grep -c '' results/registry.jsonl) records)" >> results/logs/autocommit.log 2>&1
  fi
  git push -q origin main >> results/logs/autocommit.log 2>&1
  sleep "$int"
done
