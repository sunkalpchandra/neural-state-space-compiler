"""Stability figures: Jacobian eigenvalue spectra, spectral radius, norm growth, vector fields."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from nssc.visualization._common import clean, note_title, to_numpy
from nssc.visualization.style import COLORS, DOUBLE_COL, SINGLE_COL

# isort: split
import matplotlib.pyplot as plt


def plot_eigenvalue_spectrum(eigvals: Any, title: str = "Latent Jacobian eigenvalues",
                             max_points: int = 20000) -> plt.Figure:
    """Scatter of complex eigenvalues (M,d) in the complex plane with the unit circle;
    points colored by local density (Gaussian-KDE-free 2-D histogram lookup)."""
    ev, bad = clean(eigvals)
    ev = np.asarray(ev).ravel()
    if ev.dtype.kind != "c":
        ev = ev.astype(complex)
    if ev.size > max_points:
        ev = ev[np.linspace(0, ev.size - 1, max_points).astype(int)]
    re, im = ev.real, ev.imag
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.15, SINGLE_COL * 1.15))
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(th), np.sin(th), color=COLORS["unit_circle"], lw=0.9, ls="--", label="unit circle")
    if ev.size:
        H, xe, ye = np.histogram2d(re, im, bins=40)
        ix = np.clip(np.searchsorted(xe, re, side="right") - 1, 0, H.shape[0] - 1)
        iy = np.clip(np.searchsorted(ye, im, side="right") - 1, 0, H.shape[1] - 1)
        dens = H[ix, iy]
        order = np.argsort(dens)
        sc = ax.scatter(re[order], im[order], c=dens[order], cmap="viridis", s=6, alpha=0.85,
                        linewidths=0)
        cb = fig.colorbar(sc, ax=ax, shrink=0.75)
        cb.set_label("local density (count)")
        rho = float(np.abs(ev).max())
        n_out = int((np.abs(ev) > 1.0).sum())
        sub = f"max|λ|={rho:.3f}; {n_out}/{ev.size} outside unit circle"
    else:
        sub = "no eigenvalues"
    ax.axhline(0, color="#BBBBBB", lw=0.5)
    ax.axvline(0, color="#BBBBBB", lw=0.5)
    ax.set_aspect("equal")
    lim = max(1.15, float(np.abs(ev).max()) * 1.05 if ev.size else 1.15)
    lim = min(lim, 5.0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Re λ")
    ax.set_ylabel("Im λ")
    ax.set_title(note_title(f"{title}\n{sub}", bad), fontsize=8.5)
    ax.legend(loc="lower left")
    return fig


def plot_spectral_radius_hist(rho: Any, title: str = "Local spectral radius", bins: int = 30
                              ) -> plt.Figure:
    """Histogram of per-point spectral radii ρ(J) with the ρ=1 stability boundary marked."""
    r, bad = clean(rho)
    r = np.asarray(r, float).ravel()
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.2, SINGLE_COL * 0.8))
    if r.size:
        ax.hist(r, bins=bins, color=COLORS["blue"], alpha=0.85, edgecolor="white", lw=0.3)
        frac = float((r > 1.0).mean())
        sub = f"mean={r.mean():.3f}, max={r.max():.3f}, frac(ρ>1)={frac:.2f}, n={r.size}"
    else:
        sub = "no data"
    ax.axvline(1.0, color=COLORS["vermillion"], lw=1.0, ls="--", label="ρ = 1")
    ax.set_xlabel("spectral radius ρ(∂F/∂z)")
    ax.set_ylabel("count")
    ax.set_title(note_title(f"{title}\n{sub}", bad))
    ax.legend(loc="upper right")
    return fig


def plot_norm_growth(norms: Any, ref_norm: float | None = None, title: str = "Latent norm growth",
                     logy: bool = True, max_lines: int = 32) -> plt.Figure:
    """||ẑ_t|| along free rollouts, ``norms``: (B,H). Median and 10–90% band plus individual lines.
    ``ref_norm``: mean norm of encoded data (dashed reference)."""
    n, bad = clean(norms, clip=1e12)
    n = np.asarray(n, float)
    if n.ndim == 1:
        n = n[None]
    B, H = n.shape
    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.6, DOUBLE_COL * 0.4))
    h = np.arange(1, H + 1)
    for b in range(min(B, max_lines)):
        ax.plot(h, n[b], color=COLORS["blue"], lw=0.5, alpha=0.3)
    if B > 1:
        med = np.median(n, 0)
        lo, hi = np.percentile(n, 10, axis=0), np.percentile(n, 90, axis=0)
        ax.plot(h, med, color=COLORS["blue"], lw=1.6, label=f"median (n={B})")
        ax.fill_between(h, lo, hi, color=COLORS["blue"], alpha=0.2, lw=0, label="10–90%")
    else:
        ax.plot(h, n[0], color=COLORS["blue"], lw=1.4, label="rollout")
    if ref_norm is not None and np.isfinite(ref_norm):
        ax.axhline(ref_norm, color=COLORS["true"], lw=0.9, ls="--", label="mean ‖E(x)‖ (data)")
        ax.axhline(10 * ref_norm, color=COLORS["vermillion"], lw=0.8, ls=":", label="10× (blow-up)")
    if logy and np.all(n > 0):
        ax.set_yscale("log")
    ax.set_xlabel("rollout step")
    ax.set_ylabel("‖ẑ_t‖₂")
    ax.set_title(note_title(title, bad))
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return fig


def _wrap_dynamics(dynamics: Any) -> Callable[[np.ndarray], np.ndarray]:
    """Turn a Dynamics module or a numpy/torch callable into ``f(z: (N,d) np) -> (N,d) np``.
    The returned function gives the *next state* F(z)."""
    import torch

    step = getattr(dynamics, "step", None)
    fn = step if callable(step) and hasattr(dynamics, "parameters") else dynamics

    def f(z: np.ndarray) -> np.ndarray:
        try:
            with torch.no_grad():
                out = fn(torch.as_tensor(np.asarray(z, np.float32)))
            return to_numpy(out)
        except Exception:  # noqa: BLE001 - fall back to a pure numpy callable
            return np.asarray(fn(np.asarray(z)), float)

    return f


def plot_vector_field(dynamics: Any, z_samples: Any, dims: tuple[int, int] = (0, 1), grid: int = 25,
                      trajectory: Any | None = None, title: str = "Latent vector field",
                      stream: bool = True, is_displacement: bool = False) -> plt.Figure:
    """Displacement field F(z) − z of the discrete latent map on the plane spanned by ``dims``.

    ``dynamics``: a ``Dynamics`` module (``step``) or callable ``f(z)->F(z)`` (numpy or torch);
    if ``is_displacement`` the callable already returns F(z) − z (e.g. an ODE vector field).
    ``z_samples``: (N,d) or (B,T,d) latents defining plot ranges; other dims held at their mean.
    ``trajectory``: optional (T,d) latent trajectory to overlay.
    """
    zs, bad = clean(z_samples)
    zs = zs.reshape(-1, zs.shape[-1]).astype(np.float64)
    d = zs.shape[1]
    i, j = int(dims[0]), int(dims[1])
    if d < 2 or i >= d or j >= d:
        raise ValueError(f"vector field needs latent_dim ≥ 2 and dims within range (d={d}, dims={dims})")
    lo = zs.min(0)
    hi = zs.max(0)
    pad = 0.15 * np.maximum(hi - lo, 1e-6)
    gx = np.linspace(lo[i] - pad[i], hi[i] + pad[i], grid)
    gy = np.linspace(lo[j] - pad[j], hi[j] + pad[j], grid)
    X, Y = np.meshgrid(gx, gy)
    pts = np.tile(zs.mean(0), (grid * grid, 1))
    pts[:, i] = X.ravel()
    pts[:, j] = Y.ravel()
    f = _wrap_dynamics(dynamics)
    nxt = np.asarray(f(pts), float).reshape(grid * grid, -1)
    disp = nxt if is_displacement else nxt - pts
    disp, bad2 = clean(disp)
    U = disp[:, i].reshape(grid, grid)
    V = disp[:, j].reshape(grid, grid)
    speed = np.hypot(U, V)
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.3, SINGLE_COL * 1.15))
    drew = False
    if stream and np.isfinite(speed).all() and speed.max() > 0:
        try:
            st = ax.streamplot(gx, gy, U, V, color=speed, cmap="viridis", density=1.1, linewidth=0.7,
                               arrowsize=0.7)
            cb = fig.colorbar(st.lines, ax=ax, shrink=0.75)
            cb.set_label("|F(z) − z|")
            drew = True
        except Exception:  # noqa: BLE001 - degenerate grids → quiver
            drew = False
    if not drew:
        q = ax.quiver(X, Y, U, V, speed, cmap="viridis", angles="xy", scale_units="xy", pivot="mid",
                      width=0.004)
        cb = fig.colorbar(q, ax=ax, shrink=0.75)
        cb.set_label("|F(z) − z|")
    ax.scatter(zs[:, i], zs[:, j], s=3, color="#999999", alpha=0.35, linewidths=0, label="encoded data")
    if trajectory is not None:
        tr, bad3 = clean(trajectory)
        tr = tr.reshape(-1, tr.shape[-1])
        ax.plot(tr[:, i], tr[:, j], color=COLORS["vermillion"], lw=1.2, label="trajectory")
        ax.plot(tr[0, i], tr[0, j], marker="o", color=COLORS["green"], ms=4)
        bad |= bad3
    ax.set_xlim(gx[0], gx[-1])
    ax.set_ylim(gy[0], gy[-1])
    ax.set_xlabel(f"$z_{{{i + 1}}}$")
    ax.set_ylabel(f"$z_{{{j + 1}}}$")
    other = "" if d == 2 else f" (other {d - 2} dims at mean)"
    ax.set_title(note_title(f"{title}: F(z) − z{other}", bad or bad2))
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    return fig
