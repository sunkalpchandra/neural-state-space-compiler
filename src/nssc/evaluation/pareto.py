"""Accuracy/complexity Pareto analysis over registry runs (Experiment I / §26).

    x = parameter count (or latency), y = long-horizon rollout NRMSE (mean over seeds)

Reports Pareto-efficient models per dataset, hypervolume-style dominated area (log-x), and
which candidates the compiler selected fall on the front.
"""

from __future__ import annotations

import math
from typing import Any

from nssc.evaluation.aggregate import group_runs, mean_std, pareto_front
from nssc.utils.experiment_registry import ExperimentRegistry


def pareto_points(records: list[dict[str, Any]], suite: str, metric: str = "test/recursive/nrmse@50",
                  cost: str = "params") -> dict[str, list[dict[str, Any]]]:
    groups = group_runs(records, suite=suite)
    per_ds: dict[str, list[dict[str, Any]]] = {}
    for (ds, m), recs in groups.items():
        vals = [r["metrics"].get(metric, float("nan")) for r in recs]
        mu, sd, n = mean_std(vals)
        c = recs[0].get("param_count") if cost == "params" else \
            mean_std([r["metrics"].get("test/latency/step_latency_ms_mean", float("nan")) for r in recs])[0]
        per_ds.setdefault(ds, []).append({"model": m, "cost": c, "error": mu, "error_std": sd, "n_seeds": n,
                                          "kind": "baseline" if any(t == "baseline" for t in recs[0].get("tags", []))
                                          else "latent"})
    for pts in per_ds.values():
        mask = pareto_front([(math.log10(max(p["cost"] or 1, 1)), p["error"]) for p in pts])
        for p, on in zip(pts, mask):
            p["pareto"] = bool(on)
        pts.sort(key=lambda p: (p["cost"] or 0))
    return per_ds


def dominated_area(pts: list[dict[str, Any]], ref_error: float | None = None) -> float:
    """Area (log10 cost × error) dominated by the front, relative to a reference error (max finite)."""
    front = sorted([p for p in pts if p["pareto"] and math.isfinite(p["error"])], key=lambda p: p["cost"] or 1)
    if not front:
        return float("nan")
    ref = ref_error if ref_error is not None else max(p["error"] for p in pts if math.isfinite(p["error"]))
    xmax = math.log10(max(max(p["cost"] or 1 for p in pts), 1))
    area, prev_x = 0.0, math.log10(max(front[0]["cost"] or 1, 1))
    cur = front[0]["error"]
    for p in front[1:] + [{"cost": 10 ** xmax, "error": cur}]:
        x = math.log10(max(p["cost"] or 1, 1))
        area += (x - prev_x) * max(ref - cur, 0.0)
        prev_x, cur = x, min(cur, p["error"])
    return float(area)


def pareto_markdown(per_ds: dict[str, list[dict[str, Any]]], metric: str) -> str:
    L = [f"# Pareto analysis — {metric} vs parameter count", ""]
    for ds, pts in sorted(per_ds.items()):
        L += [f"## {ds}  (dominated area {dominated_area(pts):.3f})", "",
              "| model | kind | params | error (mean ± std) | n | Pareto-efficient |", "|---|---|---|---|---|---|"]
        for p in pts:
            e = "—" if not math.isfinite(p["error"]) else f"{p['error']:.4f} ± {p['error_std']:.4f}"
            L.append(f"| {p['model']} | {p['kind']} | {p['cost']} | {e} | {p['n_seeds']} | {'**yes**' if p['pareto'] else ''} |")
        L.append("")
    return "\n".join(L)


def suite_pareto(suite: str, registry_path: str = "results/registry.jsonl",
                 metric: str = "test/recursive/nrmse@50") -> tuple[dict[str, list[dict[str, Any]]], str]:
    per_ds = pareto_points(ExperimentRegistry(registry_path).records(), suite, metric)
    return per_ds, pareto_markdown(per_ds, metric)
