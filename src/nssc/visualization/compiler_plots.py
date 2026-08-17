"""Compiler figures: score decomposition of the final ranking, stage funnel, sweeps."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np

from nssc.visualization.style import COLORS, DOUBLE_COL, model_color

# isort: split
import matplotlib.pyplot as plt

TERM_ORDER = ("recon", "one_step", "rollout", "complexity", "instability", "blowup")
TERM_WEIGHT_KEY = {"recon": "reconstruction", "one_step": "one_step", "rollout": "rollout",
                   "complexity": "complexity", "instability": "stability", "blowup": "blowup_penalty"}
TERM_COLORS = {"recon": COLORS["blue"], "one_step": COLORS["sky"], "rollout": COLORS["vermillion"],
               "complexity": COLORS["green"], "instability": COLORS["purple"],
               "blowup": COLORS["black"]}
NAN_PENALTY = 5.0  # mirrors nssc.compiler.scorer.MultiObjectiveScorer.score


def selected_name(report: dict[str, Any]) -> str:
    s = report.get("selected", {}) or {}
    return f"{s.get('encoder')}+{s.get('dynamics')}@d{s.get('latent_dim')}"


def _finite(v: Any, default: float = float("nan")) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def score_contributions(row: dict[str, Any], weights: dict[str, Any], criterion: str = "multi_objective"
                        ) -> dict[str, float]:
    """weight × term for each scorer term (NaN terms → weight × 5 penalty, as in the scorer)."""
    t = row.get("terms", {}) or {}
    out: dict[str, float] = {}
    for k in TERM_ORDER:
        val = _finite(t.get(k), NAN_PENALTY)
        if criterion == "val_mse":
            w = 0.5 if k in ("recon", "one_step") else 0.0
        elif criterion == "rollout_only":
            w = 1.0 if k == "rollout" else 0.0
        else:
            w = _finite(weights.get(TERM_WEIGHT_KEY[k]), 0.0)
        out[k] = w * val
    return out


def plot_compiler_decision(report: dict[str, Any], max_candidates: int = 20,
                           title: str | None = None) -> plt.Figure:
    """Horizontal stacked bars: score J of every final-ranking candidate decomposed into
    weighted terms; the selected candidate is highlighted."""
    ranking = list(report.get("ranking", []) or [])
    weights = dict(report.get("weights", {}) or {})
    criterion = str(weights.get("criterion", "multi_objective"))
    rows = [r for r in ranking if r.get("agg", {}).get("n_seeds", 1) > 0][:max_candidates]
    sel = selected_name(report)
    n = max(len(rows), 1)
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 0.32 * n + 1.3))
    if not rows:
        ax.text(0.5, 0.5, "no completed candidates", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig
    names = [str(r.get("name") or r.get("candidate_id")) for r in rows]
    y = np.arange(len(rows))[::-1]
    left = np.zeros(len(rows))
    for k in TERM_ORDER:
        vals = np.array([score_contributions(r, weights, criterion)[k] for r in rows])
        if not np.any(vals != 0):
            continue
        ax.barh(y, vals, left=left, color=TERM_COLORS[k], height=0.7, label=k, edgecolor="white", lw=0.3)
        left = left + vals
    for yi, r in zip(y, rows):
        J = _finite(r.get("score"))
        ax.text(left[list(y).index(yi)] + 0.02 * max(left.max(), 1e-9), yi,
                f"J={J:.2f}" if math.isfinite(J) else "J=∞", va="center", fontsize=6.5)
    ax.set_yticks(y)
    labels = []
    for name in names:
        labels.append(f"★ {name}" if name == sel else name)
    ax.set_yticklabels(labels)
    for tick, name in zip(ax.get_yticklabels(), names):
        if name == sel:
            tick.set_color(COLORS["selected"])
            tick.set_fontweight("bold")
    ax.set_xlabel("weighted score contribution (lower is better)")
    ax.set_title(title or f"Compiler decision ({criterion}): J decomposition, selected = {sel}")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), title="term")
    ax.grid(axis="y", visible=False)
    return fig


def plot_stage_funnel(report: dict[str, Any], title: str = "Search funnel") -> plt.Figure:
    """Candidates entering / surviving each search stage."""
    stages = list(report.get("stage_summaries", []) or [])
    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.55, DOUBLE_COL * 0.38))
    if not stages:
        ax.text(0.5, 0.5, "no stages", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return fig
    x = np.arange(len(stages))
    n_in = [int(s.get("n_candidates", 0)) for s in stages]
    n_out = [int(s.get("n_survivors", 0)) for s in stages]
    ax.bar(x - 0.2, n_in, width=0.4, color=COLORS["blue"], label="evaluated")
    ax.bar(x + 0.2, n_out, width=0.4, color=COLORS["orange"], label="survivors")
    for xi, a, b in zip(x, n_in, n_out):
        ax.text(xi - 0.2, a, str(a), ha="center", va="bottom", fontsize=7)
        ax.text(xi + 0.2, b, str(b), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([str(s.get("stage")) for s in stages])
    ax.set_ylabel("candidates")
    ax.set_xlabel("stage")
    nr, nf = report.get("n_runs"), report.get("n_failed")
    sub = f" — {nr} runs, {nf} failed" if nr is not None else ""
    ax.set_title(f"{title}{sub}")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    return fig


def plot_latent_dim_sweep(rows: list[dict[str, Any]], title: str = "Error vs latent dimension",
                          ylabel: str = "NRMSE (recursive, val)", logy: bool = True,
                          intrinsic_dim: int | None = None) -> plt.Figure:
    """``rows``: [{latent_dim, value, std?, family?}] → one line per family (mean ± std per dim)."""
    by_fam: dict[str, dict[int, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        try:
            d = int(r["latent_dim"])
        except (KeyError, TypeError, ValueError):
            continue
        v = _finite(r.get("value"))
        s = _finite(r.get("std"), 0.0)
        if math.isfinite(v):
            by_fam[str(r.get("family", "all"))][d].append((v, s))
    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.6, DOUBLE_COL * 0.42))
    n_bad = sum(1 for r in rows if not math.isfinite(_finite(r.get("value"))))
    for fam, dd in sorted(by_fam.items()):
        dims = sorted(dd)
        mean = np.array([np.mean([v for v, _ in dd[k]]) for k in dims])
        # std: propagate given per-row std when single row, else spread across rows
        std = np.array([np.mean([s for _, s in dd[k]]) if len(dd[k]) == 1
                        else float(np.std([v for v, _ in dd[k]], ddof=1)) for k in dims])
        c = model_color(fam)
        ax.plot(dims, mean, marker="o", color=c, label=fam)
        lo = mean - std
        if logy:  # keep the band positive without exploding the log axis
            lo = np.where(lo > 0, lo, np.abs(mean) * 0.1 + np.finfo(float).tiny)
        ax.fill_between(dims, lo, mean + std, color=c, alpha=0.2, lw=0)
    if intrinsic_dim is not None:
        ax.axvline(intrinsic_dim, color="#444444", ls=":", lw=0.9, label=f"intrinsic n={intrinsic_dim}")
    if logy and by_fam:
        ax.set_yscale("log")
    all_dims = sorted({d for dd in by_fam.values() for d in dd})
    if all_dims:
        ax.set_xticks(all_dims)
    ax.set_xlabel("latent dimension d")
    ax.set_ylabel(ylabel)
    ax.set_title(title if n_bad == 0 else f"{title} ({n_bad} non-finite omitted)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return fig


def plot_family_comparison(rows: list[dict[str, Any]], title: str = "Model family comparison",
                           ylabel: str = "NRMSE (recursive, val)", logy: bool = False) -> plt.Figure:
    """``rows``: [{family, value, std?}] → bar chart with error bars (one bar per family, in
    ascending order of value)."""
    fams: dict[str, tuple[float, float]] = {}
    for r in rows:
        v = _finite(r.get("value"))
        if math.isfinite(v):
            fams[str(r.get("family"))] = (v, _finite(r.get("std"), 0.0))
    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.55, DOUBLE_COL * 0.4))
    n_bad = len(rows) - len(fams)
    if fams:
        names = sorted(fams, key=lambda k: fams[k][0])
        vals = np.array([fams[k][0] for k in names])
        errs = np.array([fams[k][1] for k in names])
        x = np.arange(len(names))
        ax.bar(x, vals, yerr=errs, color=[model_color(k) for k in names], capsize=3,
               error_kw={"elinewidth": 0.8}, edgecolor="white", lw=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right")
        if logy and np.all(vals > 0):
            ax.set_yscale("log")
    else:
        ax.text(0.5, 0.5, "no finite values", ha="center", va="center", transform=ax.transAxes)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("family")
    ax.set_title(title if n_bad == 0 else f"{title} ({n_bad} non-finite omitted)")
    ax.grid(axis="x", visible=False)
    return fig
