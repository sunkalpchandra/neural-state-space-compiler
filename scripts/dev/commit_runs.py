#!/usr/bin/env python3
"""Commit each finished run's small artifacts (metrics.json, history.json, checkpoint config/metadata,
error.json) as one commit per run. Idempotent: skips run dirs with nothing new. Weights (*.pt) are
gitignored. Message: `exp(<dataset>): <model> seed=<s> — test NRMSE@50=<v> [EXP-xxxx]`.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTS = ("metrics.json", "history.json", "error.json", "checkpoint/config.yaml", "checkpoint/metadata.json",
        "checkpoint/config.json")


def git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout


def main(limit: int | None) -> None:
    n = 0
    for mfile in sorted(ROOT.glob("results/raw/**/metrics.json")) + sorted(ROOT.glob("results/compile/**/runs/**/metrics.json")):
        run = mfile.parent
        files = [str(run / a) for a in ARTS if (run / a).exists()]
        status = git("status", "--porcelain", "--", *files)
        if not status.strip():
            continue
        try:
            m = json.load(open(mfile))
            meta = json.load(open(run / "checkpoint/metadata.json")) if (run / "checkpoint/metadata.json").exists() else {}
            exp = meta.get("experiment_id", "")
            ds = (meta.get("dataset") or {}).get("system") or (meta.get("dataset") or {}).get("source") or run.parts[-3]
            model = run.parts[-2] if run.parts[-1].startswith("seed") else run.name
            seed = meta.get("seed", run.name.replace("seed", ""))
            t = m.get("test", {})
            v = t.get("recursive/nrmse@50", t.get("recursive/nrmse_mean"))
            vs = f"{v:.4f}" if isinstance(v, (int, float)) else "n/a"
            msg = f"exp({ds}): {model} seed={seed} — test NRMSE@50={vs}" + (f" [{exp}]" if exp else "")
        except Exception:  # noqa: BLE001
            msg = f"exp: artifacts for {run.relative_to(ROOT)}"
        subprocess.run(["git", "add", "-f", *files], cwd=ROOT, check=True)
        subprocess.run(["git", "commit", "-qm", msg], cwd=ROOT, check=True)
        n += 1
        if limit and n >= limit:
            break
    print(f"committed {n} runs")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
