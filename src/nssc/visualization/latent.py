"""Latent-space figures: stacked trajectories, phase portraits, alignment to ground truth."""

from __future__ import annotations

from typing import Any

import numpy as np

from nssc.visualization._common import clean, finite_range, note_title
from nssc.visualization.style import COLORS, DOUBLE_COL, SINGLE_COL

# isort: split
import matplotlib.pyplot as plt


def _as_batch(z: Any) -> tuple[np.ndarray, bool]:
    """(B,T,d) float array (adds batch axis for (T,d)); second value = had non-finite."""
    a, bad = clean(z)
    if a.ndim == 1:
        a = a[None, :, None]
    elif a.ndim == 2:
        a = a[None]
    elif a.ndim != 3:
        raise ValueError(f"z must be (T,d) or (B,T,d), got {a.shape}")
    return a, bad


def plot_latent_trajectories(z: Any, title: str = "Latent trajectories", t: Any | None = None,
                             max_traj: int = 8, dim_labels: list[str] | None = None) -> plt.Figure:
    """Stacked time series of z_1..z_d. ``z``: (T,d) or (B,T,d); each trajectory a faint line."""
    zb, bad = _as_batch(z)
    B, T, d = zb.shape
    tt = np.arange(T) if t is None else clean(t)[0]
    fig, axes = plt.subplots(d, 1, sharex=True, figsize=(DOUBLE_COL, max(1.1 * d, 2.0)),
                             squeeze=False)
    for j in range(d):
        ax = axes[j, 0]
        for b in range(min(B, max_traj)):
            ax.plot(tt, zb[b, :, j], color=COLORS["blue"], alpha=1.0 if B == 1 else 0.55, lw=1.0)
        ax.set_ylabel(dim_labels[j] if dim_labels else f"$z_{{{j + 1}}}$")
        ax.set_ylim(*finite_range(zb[:, :, j]))
    axes[-1, 0].set_xlabel("time (steps)" if t is None else "time")
    n = min(B, max_traj)
    axes[0, 0].set_title(note_title(f"{title} (n={n} trajectories, d={d})", bad))
    return fig


def plot_phase_portrait(z: Any, dims: tuple[int, ...] = (0, 1), color_by_time: bool = True,
                        title: str = "Latent phase portrait", max_traj: int = 6) -> plt.Figure:
    """2D or 3D phase portrait of latent trajectories along ``dims`` (2 or 3 indices).

    Colored by time with viridis (ordinal → sequential colormap) if ``color_by_time``.
    """
    zb, bad = _as_batch(z)
    B, T, d = zb.shape
    dims = tuple(int(i) for i in dims)
    if any(i >= d for i in dims):
        raise ValueError(f"dims {dims} out of range for latent_dim={d}")
    if len(dims) not in (2, 3):
        raise ValueError("dims must have length 2 or 3")
    three = len(dims) == 3
    fig = plt.figure(figsize=(SINGLE_COL * 1.15, SINGLE_COL * 1.15))
    ax = fig.add_subplot(111, projection="3d" if three else None)
    tt = np.arange(T)
    sc = None
    for b in range(min(B, max_traj)):
        coords = [zb[b, :, i] for i in dims]
        if color_by_time:
            ax.plot(*coords, color="#999999", lw=0.5, alpha=0.6)
            sc = ax.scatter(*coords, c=tt, cmap="viridis", s=4, alpha=0.9, linewidths=0)
        else:
            ax.plot(*coords, lw=0.9, alpha=0.9)
        ax.plot(*[[c[0]] for c in coords], marker="o", color=COLORS["green"], ms=4)
        ax.plot(*[[c[-1]] for c in coords], marker="s", color=COLORS["vermillion"], ms=4)
    ax.set_xlabel(f"$z_{{{dims[0] + 1}}}$")
    ax.set_ylabel(f"$z_{{{dims[1] + 1}}}$")
    if three:
        ax.set_zlabel(f"$z_{{{dims[2] + 1}}}$")
    if sc is not None:
        cb = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.08 if three else 0.02)
        cb.set_label("time (steps)")
    ax.set_title(note_title(title, bad))
    return fig


def align_latents(z: Any, z_true: Any) -> dict[str, Any]:
    """Least-squares affine map ``z_true ≈ W z + b`` on flattened (·, d) arrays.

    Returns ``{"z_aligned": (…, d_true), "W", "b", "r2": float, "r2_per_dim": array}``.
    """
    za, bad1 = clean(z)
    zt, bad2 = clean(z_true)
    shape = zt.shape
    Z = za.reshape(-1, za.shape[-1]).astype(np.float64)
    Y = zt.reshape(-1, zt.shape[-1]).astype(np.float64)
    if Z.shape[0] != Y.shape[0]:
        n = min(Z.shape[0], Y.shape[0])
        Z, Y = Z[:n], Y[:n]
    A = np.concatenate([Z, np.ones((Z.shape[0], 1))], axis=1)
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    W, b = coef[:-1], coef[-1]
    Yhat = A @ coef
    ss_res = ((Y - Yhat) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0)
    r2_dim = 1.0 - ss_res / np.where(ss_tot > 0, ss_tot, np.nan)
    r2 = 1.0 - ss_res.sum() / max(ss_tot.sum(), 1e-12)
    return {"z_aligned": Yhat.reshape(shape), "W": W, "b": b, "r2": float(r2),
            "r2_per_dim": r2_dim, "had_nonfinite": bad1 or bad2}


def plot_latent_vs_true(z: Any, z_true: Any, title: str = "Latent vs ground-truth state",
                        max_traj: int = 3) -> plt.Figure:
    """Overlay of linearly aligned latents (``z_true ≈ W z + b``) on true latents; R² in title.

    Caption warning: alignment R² is a *linear recoverability* measure, not evidence that
    latent coordinates are physical variables.
    """
    zb, _ = _as_batch(z)
    tb, _ = _as_batch(z_true)
    B = min(zb.shape[0], tb.shape[0])
    T = min(zb.shape[1], tb.shape[1])
    zb, tb = zb[:B, :T], tb[:B, :T]
    al = align_latents(zb, tb)
    za = al["z_aligned"]
    d_true = tb.shape[-1]
    fig, axes = plt.subplots(d_true, 1, sharex=True, figsize=(DOUBLE_COL, max(1.2 * d_true, 2.2)),
                             squeeze=False)
    tt = np.arange(T)
    for j in range(d_true):
        ax = axes[j, 0]
        for b in range(min(B, max_traj)):
            ax.plot(tt, tb[b, :, j], color=COLORS["true"], lw=1.2,
                    label="ground truth" if (b == 0 and j == 0) else None)
            ax.plot(tt, za[b, :, j], color=COLORS["pred"], lw=1.0, ls="--",
                    label="aligned latent (W z + b)" if (b == 0 and j == 0) else None)
        r2j = al["r2_per_dim"][j]
        ax.set_ylabel(f"true $s_{{{j + 1}}}$\n$R^2$={r2j:.3f}" if np.isfinite(r2j) else f"$s_{{{j + 1}}}$")
    axes[-1, 0].set_xlabel("time (steps)")
    axes[0, 0].legend(loc="upper right", ncol=2)
    axes[0, 0].set_title(note_title(f"{title} — linear alignment $R^2$={al['r2']:.3f} "
                                    f"(d={zb.shape[-1]} → {d_true})", al["had_nonfinite"]))
    return fig
