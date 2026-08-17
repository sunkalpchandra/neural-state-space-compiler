"""Aggregate registry records into tables: mean ± std over seeds, CIs, paired tests, Pareto sets."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np

from nssc.utils.experiment_registry import ExperimentRegistry


def _tag(rec: dict[str, Any], prefix: str) -> str | None:
    for t in rec.get("tags", []):
        if t.startswith(prefix):
            return t[len(prefix):]
    return None


def group_runs(records: list[dict[str, Any]], suite: str | None = None
               ) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """(dataset, model_name) → list of completed records (one per seed; latest per seed wins)."""
    groups: dict[tuple[str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    for r in records:
        if r.get("status") != "completed":
            continue
        if suite and _tag(r, "suite:") != suite:
            continue
        ds, m = _tag(r, "ds:"), _tag(r, "m:")
        if ds is None or m is None:
            continue
        groups[(ds, m)][int(r["seed"])] = r
    return {k: list(v.values()) for k, v in groups.items()}


def mean_std(values: list[float]) -> tuple[float, float, int]:
    v = np.asarray([x for x in values if x is not None and math.isfinite(x)], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), 0
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0, int(v.size)


def bootstrap_ci(values: list[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
                 ) -> tuple[float, float]:
    v = np.asarray([x for x in values if math.isfinite(x)], dtype=float)
    if v.size < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    return float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def paired_test(a: list[float], b: list[float]) -> dict[str, float]:
    """Paired comparison across seeds (same seeds!): Wilcoxon signed-rank + paired t.
    With n=5 the minimum two-sided Wilcoxon p is 0.0625 — report, don't over-interpret."""
    from scipy import stats

    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    out = {"n": int(a.size), "mean_diff": float((a - b).mean()) if a.size else float("nan")}
    if a.size >= 2 and np.any(a != b):
        out["t_p"] = float(stats.ttest_rel(a, b).pvalue)
        try:
            out["wilcoxon_p"] = float(stats.wilcoxon(a, b).pvalue)
        except ValueError:
            out["wilcoxon_p"] = float("nan")
    return out


def summary_table(groups: dict[tuple[str, str], list[dict[str, Any]]], metrics: list[str]
                  ) -> list[dict[str, Any]]:
    rows = []
    for (ds, m), recs in sorted(groups.items()):
        row: dict[str, Any] = {"dataset": ds, "model": m, "n_seeds": len(recs),
                               "params": recs[0].get("param_count")}
        for k in metrics:
            vals = [r["metrics"].get(k, float("nan")) for r in recs]
            mu, sd, n = mean_std(vals)
            row[k] = mu
            row[f"{k}_std"] = sd
            row[f"{k}_ci"] = bootstrap_ci(vals) if n >= 2 else (float("nan"), float("nan"))
        rows.append(row)
    return rows


def pareto_front(points: list[tuple[float, float]]) -> list[bool]:
    """Minimise both coordinates. Returns mask of Pareto-efficient points."""
    pts = np.asarray(points, float)
    eff = np.ones(len(pts), bool)
    for i, p in enumerate(pts):
        if not np.all(np.isfinite(p)):
            eff[i] = False
            continue
        dominated = np.any(np.all(pts <= p, axis=1) & np.any(pts < p, axis=1))
        eff[i] = not dominated
    return eff.tolist()


def format_markdown(rows: list[dict[str, Any]], metrics: list[str], labels: dict[str, str] | None = None
                    ) -> str:
    labels = labels or {}
    head = ["dataset", "model", "n", "params"] + [labels.get(k, k) for k in metrics]
    L = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        cells = [r["dataset"], r["model"], str(r["n_seeds"]), str(r.get("params"))]
        for k in metrics:
            mu, sd = r[k], r[f"{k}_std"]
            if not math.isfinite(mu):
                cells.append("—")
            elif abs(mu) >= 1e3 and "params" not in k:
                cells.append(f"diverged (>{1e3:.0e})")
            elif float(mu).is_integer() and ("params" in k or "time" in k):
                cells.append(f"{mu:.0f}" + (f" ± {sd:.1f}" if r["n_seeds"] > 1 and sd > 0 else ""))
            else:
                cells.append(f"{mu:.4f} ± {sd:.4f}" if r["n_seeds"] > 1 else f"{mu:.4f}")
        L.append("| " + " | ".join(cells) + " |")
    return "\n".join(L)


def load_groups(registry_path: str = "results/registry.jsonl", suite: str | None = None):
    return group_runs(ExperimentRegistry(registry_path).records(), suite=suite)
