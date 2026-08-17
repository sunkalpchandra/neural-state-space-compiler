"""Complexity–accuracy Pareto figure."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from nssc.evaluation.aggregate import pareto_front
from nssc.visualization.style import COLORS, DOUBLE_COL, model_color

# isort: split
import matplotlib.pyplot as plt


def _num(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return f


def plot_pareto(points: list[dict[str, Any]], title: str = "Complexity vs error",
                xlabel: str = "parameters", ylabel: str = "NRMSE (recursive)", logy: bool = True,
                annotate_front: bool = True, max_annotations: int = 12) -> plt.Figure:
    """``points``: dicts with ``name``, ``params``, ``error``, optional ``family``, ``is_selected``,
    ``error_std``. Log-x parameters; Pareto front (min params, min error) drawn as a step line;
    selected candidate starred; front points annotated."""
    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.6, DOUBLE_COL * 0.45))
    xs = np.array([_num(p.get("params")) for p in points])
    ys = np.array([_num(p.get("error")) for p in points])
    ok = np.isfinite(xs) & np.isfinite(ys) & (xs > 0)
    n_bad = int((~ok).sum())
    front = np.zeros(len(points), bool)
    if ok.any():
        fm = pareto_front([(float(x), float(y)) for x, y in zip(xs[ok], ys[ok])])
        front[np.flatnonzero(ok)] = np.asarray(fm, bool)
    families_seen: set[str] = set()
    for i, p in enumerate(points):
        if not ok[i]:
            continue
        fam = str(p.get("family") or p.get("name", ""))
        c = model_color(fam) if p.get("family") else model_color(str(p.get("name", "")))
        lab = fam if fam not in families_seen else None
        families_seen.add(fam)
        err = _num(p.get("error_std"))
        if math.isfinite(err) and err > 0:
            ax.errorbar(xs[i], ys[i], yerr=err, fmt="none", ecolor=c, elinewidth=0.7, capsize=2,
                        alpha=0.7)
        ax.scatter(xs[i], ys[i], color=c, s=28 if front[i] else 18, label=lab,
                   edgecolors="black" if front[i] else "none", linewidths=0.6, zorder=3)
        if p.get("is_selected"):
            ax.scatter(xs[i], ys[i], marker="*", s=180, color=COLORS["selected"],
                       edgecolors="black", linewidths=0.6, zorder=4, label="selected")
    fi = np.flatnonzero(front)
    if fi.size:
        order = fi[np.argsort(xs[fi])]
        ax.step(xs[order], ys[order], where="post", color=COLORS["front"], lw=1.0, ls="--",
                alpha=0.8, label="Pareto front", zorder=2)
        if annotate_front:
            for i in order[:max_annotations]:
                ax.annotate(str(points[i].get("name", "")), (xs[i], ys[i]), fontsize=6,
                            xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log")
    if logy and ok.any() and np.all(ys[ok] > 0):
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    t = title if n_bad == 0 else f"{title} ({n_bad} non-finite point(s) omitted)"
    ax.set_title(t)
    h, lab = ax.get_legend_handles_labels()
    uniq: dict[str, Any] = {}
    for hh, ll in zip(h, lab):
        uniq.setdefault(ll, hh)
    ax.legend(uniq.values(), uniq.keys(), loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return fig
