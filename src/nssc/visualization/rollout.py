"""Rollout figures: true vs predicted observations, error vs horizon, one-step vs recursive."""

from __future__ import annotations

from typing import Any

import numpy as np

from nssc.visualization._common import clean, note_title, to_numpy
from nssc.visualization.style import COLORS, DOUBLE_COL, model_color

# isort: split
import matplotlib.pyplot as plt


def _2d(x: Any) -> tuple[np.ndarray, bool]:
    a, bad = clean(x)
    if a.ndim == 3:
        a = a[0]
    if a.ndim == 1:
        a = a[:, None]
    if a.ndim != 2:
        raise ValueError(f"expected (T,D), got {a.shape}")
    return a, bad


def plot_rollout_comparison(x_true: Any, x_pred: Any, context: int, dims: int | list[int] = 4,
                            x_std: Any | None = None, title: str = "Recursive rollout",
                            mode_label: str = "recursive") -> plt.Figure:
    """True (T,D) vs predicted (H,D) for the H steps after ``context``; ±2σ envelope optional.

    ``dims``: number of leading dims or explicit list of dim indices (max 4 recommended).
    """
    xt, bad1 = _2d(x_true)
    xp, bad2 = _2d(x_pred)
    T, D = xt.shape
    H = xp.shape[0]
    idx = list(range(min(D, dims))) if isinstance(dims, int) else [int(i) for i in dims]
    idx = [i for i in idx if i < D]
    xs, bad3 = (None, False)
    if x_std is not None:
        xs, bad3 = _2d(x_std)
    fig, axes = plt.subplots(len(idx), 1, sharex=True, figsize=(DOUBLE_COL, max(1.3 * len(idx), 2.2)),
                             squeeze=False)
    t_all = np.arange(T)
    t_pred = context + np.arange(H)
    for row, j in enumerate(idx):
        ax = axes[row, 0]
        ax.axvspan(0, context, color=COLORS["context"], alpha=0.25, lw=0,
                   label="context" if row == 0 else None)
        ax.plot(t_all, xt[:, j], color=COLORS["true"], lw=1.2, label="true" if row == 0 else None)
        ax.plot(t_pred, xp[:, j], color=COLORS["pred"], lw=1.2, ls="--",
                label=f"predicted ({mode_label})" if row == 0 else None)
        if xs is not None and j < xs.shape[1]:
            ax.fill_between(t_pred, xp[:, j] - 2 * xs[:, j], xp[:, j] + 2 * xs[:, j],
                            color=COLORS["envelope"], alpha=0.3, lw=0,
                            label="±2σ" if row == 0 else None)
        ax.axvline(context, color="#444444", lw=0.8, ls=":")
        ax.set_ylabel(f"$x_{{{j + 1}}}$")
    axes[-1, 0].set_xlabel("time (steps)")
    axes[0, 0].legend(loc="upper right", ncol=4)
    axes[0, 0].set_title(note_title(f"{title}: context={context}, horizon={H}", bad1 or bad2 or bad3))
    return fig


def plot_horizon_curves(curves: dict[str, Any], horizons: Any | None = None, logy: bool = True,
                        logx: bool = False, title: str = "Rollout error vs horizon",
                        ylabel: str = "NRMSE", mode_label: str = "recursive",
                        mark_horizons: Any | None = None) -> plt.Figure:
    """NRMSE vs horizon for several models. ``curves``: {name: curve} or {name: (mean, std)}.

    Values are plotted per step ``1..H`` unless ``horizons`` (len H) is given.
    """
    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.6, DOUBLE_COL * 0.42))
    any_bad = False
    for name, cv in curves.items():
        std = None
        if isinstance(cv, (tuple, list)) and len(cv) == 2 and np.ndim(cv[0]) >= 1 \
                and np.ndim(cv[1]) >= 1 and np.shape(cv[0]) == np.shape(cv[1]):
            mean, bad = clean(cv[0])
            std, bad2 = clean(cv[1])
            any_bad |= bad or bad2
        else:
            mean, bad = clean(cv)
            any_bad |= bad
        mean = np.asarray(mean, float).ravel()
        h = np.arange(1, mean.size + 1) if horizons is None else to_numpy(horizons).ravel()[: mean.size]
        c = model_color(name)
        ls = "--" if "[val]" in name or name.endswith("(val)") else "-"
        ax.plot(h, mean, color=c, label=name, lw=1.4, ls=ls)
        if std is not None:
            std = np.asarray(std, float).ravel()
            lo = mean - std
            if logy:  # keep the band positive without exploding the log axis
                lo = np.where(lo > 0, lo, np.abs(mean) * 0.1 + np.finfo(float).tiny)
            ax.fill_between(h, lo, mean + std, color=c, alpha=0.2, lw=0)
    if mark_horizons is not None:
        for hh in to_numpy(mark_horizons).ravel():
            ax.axvline(hh, color="#BBBBBB", lw=0.6, ls=":")
    if logy:
        ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel("horizon (steps)")
    ax.set_ylabel(f"{ylabel} ({mode_label})")
    ax.set_title(note_title(title, any_bad))
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return fig


def plot_one_step_vs_long_horizon(x_true: Any, x_tf_pred: Any, x_recursive_pred: Any, context: int,
                                  dim: int = 0, title: str = "One-step vs recursive prediction"
                                  ) -> plt.Figure:
    """Two panels: (left) teacher-forced one-step x̂_{t+1} vs truth; (right) recursive rollout
    from ``context``. ``x_tf_pred``: (T-1,D) aligned to targets x_{2:T}; ``x_recursive_pred``: (H,D)."""
    xt, b1 = _2d(x_true)
    tf, b2 = _2d(x_tf_pred)
    rc, b3 = _2d(x_recursive_pred)
    T = xt.shape[0]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.35), sharey=True)
    ax = axes[0]
    ax.plot(np.arange(T), xt[:, dim], color=COLORS["true"], lw=1.2, label="true")
    ax.plot(np.arange(1, 1 + tf.shape[0]), tf[:, dim], color=COLORS["orange"], lw=1.0, ls="--",
            label="one-step (teacher_forced)")
    ax.set_title("teacher_forced (one step)")
    ax.set_xlabel("time (steps)")
    ax.set_ylabel(f"$x_{{{dim + 1}}}$")
    ax.legend(loc="upper right")
    ax = axes[1]
    ax.axvspan(0, context, color=COLORS["context"], alpha=0.25, lw=0, label="context")
    ax.plot(np.arange(T), xt[:, dim], color=COLORS["true"], lw=1.2, label="true")
    ax.plot(context + np.arange(rc.shape[0]), rc[:, dim], color=COLORS["pred"], lw=1.0, ls="--",
            label="recursive")
    ax.axvline(context, color="#444444", lw=0.8, ls=":")
    ax.set_title(f"recursive (H={rc.shape[0]})")
    ax.set_xlabel("time (steps)")
    ax.legend(loc="upper right")
    fig.suptitle(note_title(title, b1 or b2 or b3))
    return fig
