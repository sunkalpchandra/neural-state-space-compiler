#!/usr/bin/env python3
"""Experiments G/H — OOD initial conditions and dynamical parameters.

For every completed run in a suite (or an explicit list of experiment ids) evaluate the
frozen checkpoint on (a) shifted system parameters and (b) widened initial conditions,
and write results/tables/ood_<suite>.{md,json}.

    python experiments/synthetic/run_ood.py --suite synthetic_core --dataset lorenz63 \
        --param rho --values 20 24 32 35 --ic-scales 2 4
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from nssc.evaluation.evaluator import EvalConfig
from nssc.evaluation.ood import evaluate_ood
from nssc.utils.experiment_registry import ExperimentRegistry
from nssc.utils.io import save_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--param", default=None)
    ap.add_argument("--values", nargs="*", type=float, default=[])
    ap.add_argument("--ic-scales", nargs="*", type=float, default=[])
    ap.add_argument("--models", nargs="*", default=None, help="model tag names to include")
    ap.add_argument("--n-traj", type=int, default=20)
    ap.add_argument("--out", default="results/tables")
    a = ap.parse_args()

    recs = [r for r in ExperimentRegistry().records()
            if r["status"] == "completed" and r.get("checkpoint") and f"suite:{a.suite}" in r.get("tags", [])
            and f"ds:{a.dataset}" in r.get("tags", []) and not any(t == "baseline" for t in r.get("tags", []))]
    if a.models:
        recs = [r for r in recs if any(f"m:{m}" in r["tags"] for m in a.models)]
    ecfg = EvalConfig(context=20, horizons=(1, 5, 10, 25, 50, 100), latency=False, stability=False)
    per_model: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        name = [t[2:] for t in r["tags"] if t.startswith("m:")][0]
        res = evaluate_ood(r["checkpoint"], param_shifts={a.param: a.values} if a.param else None,
                           ic_scales=a.ic_scales, n_traj=a.n_traj, eval_cfg=ecfg)
        res["experiment_id"], res["seed"] = r["experiment_id"], r["seed"]
        per_model[name].append(res)
        print(r["experiment_id"], name, r["seed"], {c["condition"]: round(c.get("degradation_ratio", float("nan")), 2)
                                                    for c in res["conditions"]})
    conds = [c["condition"] for c in next(iter(per_model.values()))[0]["conditions"]] if per_model else []
    lines = [f"# OOD evaluation — suite `{a.suite}`, dataset `{a.dataset}` (frozen checkpoints; ID reference = test split)", "",
             "Cells: recursive NRMSE@50 mean ± std over seeds (degradation ratio vs in-distribution in parentheses)", "",
             "| model | n | ID NRMSE@50 | " + " | ".join(conds) + " |", "|---|---|---|" + "---|" * len(conds)]
    table = {}
    for name, results in sorted(per_model.items()):
        ref = [x["reference"].get("recursive/nrmse@50", float("nan")) for x in results]
        cells = [f"{np.mean(ref):.4f} ± {np.std(ref, ddof=1) if len(ref) > 1 else 0:.4f}"]
        row = {"id": (float(np.mean(ref)), float(np.std(ref)))}
        for c in conds:
            vals = [next(cc for cc in x["conditions"] if cc["condition"] == c) for x in results]
            v = [cc.get("recursive/nrmse@50", float("nan")) for cc in vals]
            d = [cc.get("degradation_ratio", float("nan")) for cc in vals]
            cells.append(f"{np.nanmean(v):.4f} ± {np.nanstd(v, ddof=1) if len(v) > 1 else 0:.4f} (×{np.nanmean(d):.2f})")
            row[c] = (float(np.nanmean(v)), float(np.nanmean(d)))
        lines.append(f"| {name} | {len(results)} | " + " | ".join(cells) + " |")
        table[name] = row
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"ood_{a.suite}_{a.dataset}.md").write_text("\n".join(lines))
    save_json({"table": table, "per_model": per_model, "args": vars(a)}, out / f"ood_{a.suite}_{a.dataset}.json")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
