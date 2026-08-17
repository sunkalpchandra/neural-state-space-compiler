#!/usr/bin/env bash
# Commit every changed/untracked file under given paths as its own commit.
# Usage: scripts/dev/commit_each.sh "<prefix>" path1 [path2 ...]
# Message: "<prefix>: <relative file path>"
set -euo pipefail
prefix="$1"; shift
cd "$(git rev-parse --show-toplevel)"
files=$(git status --porcelain -uall -- "$@" | awk '{print $2}' | sort -u)
n=0
for f in $files; do
  [ -e "$f" ] || { git rm -q --cached "$f" 2>/dev/null || true; git commit -qm "$prefix: remove $f" && n=$((n+1)); continue; }
  git add -- "$f"
  git commit -qm "$prefix: $f" >/dev/null && n=$((n+1))
done
echo "committed $n files"
