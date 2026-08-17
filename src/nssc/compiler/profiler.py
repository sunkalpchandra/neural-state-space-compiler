"""Dataset profiler: cheap numpy statistics that steer the compiler's search.

:func:`profile_dataset` inspects a :class:`~nssc.data.dataset.TrajectoryDataset`
(raw, unnormalised) and returns a :class:`DatasetProfile` with shape / scale
summaries, intrinsic-dimension estimates (PCA, Levina–Bickel MLE, correlation
dimension), temporal structure (autocorrelation, spectrum, smoothness), noise
and non-stationarity estimates, a linear-predictability score, a crude
largest-Lyapunov proxy and a ``recommendations`` dict of boolean/list hints.

All arrays are ``(N, T, D)`` (trajectories, time, observation dim). NaNs in ``x``
(missing values) are tolerated: statistics are NaN-aware and dynamical estimates
use a linearly time-interpolated copy. Every scalar is finite or explicitly NaN.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from nssc.data.dataset import TrajectoryDataset

_STANDARD_GRID = (2, 4, 8, 16, 32)


@dataclass
class DatasetProfile:
    """Result of :func:`profile_dataset`. See module docstring for semantics."""

    # shape
    n_traj: int
    n_steps: int
    obs_dim: int
    dt: float                     # NaN if unknown
    total_samples: int
    has_missing: bool
    missing_rate: float
    sampling_rate_hz: float       # NaN if dt unknown
    # scale
    mean_min: float
    mean_max: float
    std_min: float
    std_median: float
    std_max: float
    dynamic_range: float          # max over dims of (max - min)
    # intrinsic dimensionality
    pca_dims_for_variance: dict[str, int]
    explained_variance_curve: list[float]
    mle_dim_k10: float
    mle_dim_k20: float
    correlation_dim: float
    suggested_latent_dims: list[int]
    # temporal
    autocorr: list[float]
    autocorr_time: float          # NaN if acf never drops below 1/e within max lag
    smoothness: float
    dominant_period_steps: float
    spectral_flatness: float
    # noise
    noise_std_estimate: float
    signal_std: float
    noise_ratio_estimate: float
    # stationarity
    nonstationarity_mean: float
    nonstationarity_std: float
    nonstationary_dim_fraction: float
    # linearity
    linear_predictability_r2: float
    linear_r2_at_10_steps: float
    # chaos
    lyapunov_proxy: float
    lyapunov_proxy_per_time: float
    recommendations: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        d = self.to_dict()
        rec = d.pop("recommendations")
        curve = d.pop("explained_variance_curve")
        acf = d.pop("autocorr")
        lines = ["| field | value |", "|---|---|"]
        for k, v in d.items():
            if isinstance(v, float):
                v = "nan" if not math.isfinite(v) else f"{v:.4g}"
            lines.append(f"| {k} | {v} |")
        lines.append(f"| explained_variance_curve | {[round(c, 3) for c in curve[:8]]}... |")
        lines.append(f"| autocorr | {[round(a, 3) for a in acf[:8]]}... |")
        lines += ["", "**Recommendations**", ""]
        lines += [f"- {k}: {v}" for k, v in rec.items()]
        return "\n".join(lines)


# ----------------------------------------------------------------------------- helpers
def _f(v: Any) -> float:
    """Coerce to a python float, mapping inf/None to NaN."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def _fill_nans(x: np.ndarray) -> np.ndarray:
    """Linear time-interpolation of NaNs per (trajectory, dim); all-NaN columns -> dim mean/0."""
    if not np.isnan(x).any():
        return x
    x = x.copy()
    n, t, d = x.shape
    dim_mean = np.nanmean(x.reshape(-1, d), axis=0)
    dim_mean = np.where(np.isfinite(dim_mean), dim_mean, 0.0)
    idx = np.arange(t)
    for i in range(n):
        for j in range(d):
            col = x[i, :, j]
            ok = np.isfinite(col)
            if ok.all():
                continue
            if ok.sum() == 0:
                col[:] = dim_mean[j]
            elif ok.sum() == 1:
                col[:] = col[ok][0]
            else:
                col[~ok] = np.interp(idx[~ok], idx[ok], col[ok])
    return x


def _subsample_points(x: np.ndarray, n_max: int, rng: np.random.Generator) -> np.ndarray:
    flat = x.reshape(-1, x.shape[-1])
    if flat.shape[0] > n_max:
        flat = flat[rng.choice(flat.shape[0], n_max, replace=False)]
    return flat


def _pca(flat: np.ndarray, rng: np.random.Generator, max_samples: int
         ) -> tuple[np.ndarray, dict[str, int]]:
    d = flat.shape[1]
    sub = _subsample_points(flat[None], max_samples, rng)
    sub = sub - sub.mean(0, keepdims=True)
    s = np.linalg.svd(sub, compute_uv=False) if sub.shape[0] > 0 else np.zeros(0)
    var = s**2
    total = max(var.sum(), 1e-12)
    curve = np.cumsum(var) / total
    if curve.size < d:
        curve = np.concatenate([curve, np.ones(d - curve.size)])
    curve = np.clip(curve, 0.0, 1.0)
    curve[-1] = 1.0
    dims = {}
    for thr in (0.90, 0.95, 0.99, 0.999):
        dims[f"{thr:g}"] = int(np.searchsorted(curve, thr - 1e-9) + 1)
    return curve, dims


def _mle_dim(pts: np.ndarray, k: int) -> float:
    """Levina–Bickel MLE intrinsic dimension (averaged inverse estimator)."""
    n = pts.shape[0]
    if n < k + 2:
        return float("nan")
    from scipy.spatial import cKDTree

    tree = cKDTree(pts)
    dist, _ = tree.query(pts, k=k + 1)
    dist = dist[:, 1:]  # drop self
    rk = dist[:, -1:]
    good = (rk[:, 0] > 0) & np.all(dist > 0, axis=1)
    if good.sum() < 5:
        return float("nan")
    logs = np.log(rk[good] / dist[good, : k - 1])
    inv = logs.sum(1) / (k - 1)
    inv = inv[inv > 0]
    if inv.size == 0:
        return float("nan")
    return _f(1.0 / np.mean(inv))


def _correlation_dim(pts: np.ndarray) -> float:
    """Grassberger–Procaccia slope of log C(r) vs log r over the middle range."""
    n = pts.shape[0]
    if n < 50:
        return float("nan")
    from scipy.spatial.distance import pdist

    d = pdist(pts)
    d = d[d > 0]
    if d.size < 100:
        return float("nan")
    lo, hi = np.percentile(d, [1, 50])
    if not (hi > lo > 0):
        return float("nan")
    radii = np.exp(np.linspace(np.log(lo), np.log(hi), 16))
    counts = np.array([(d <= r).mean() for r in radii])
    ok = counts > 0
    if ok.sum() < 4:
        return float("nan")
    lr, lc = np.log(radii[ok]), np.log(counts[ok])
    # fit on the middle 60% of the usable range
    m = ok.sum()
    a, b = m // 5, m - m // 5
    if b - a < 3:
        a, b = 0, m
    slope = np.polyfit(lr[a:b], lc[a:b], 1)[0]
    return _f(slope)


def _acf(xf: np.ndarray, max_lag: int) -> np.ndarray:
    """Mean autocorrelation over dims and trajectories, lags 0..max_lag."""
    xc = xf - xf.mean(1, keepdims=True)
    var = (xc**2).mean(1)  # (N, D)
    acf = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        num = (xc[:, : xf.shape[1] - lag] * xc[:, lag:]).mean(1)
        r = np.where(var > 1e-12, num / np.maximum(var, 1e-12), 1.0 if lag == 0 else 0.0)
        acf[lag] = r.mean()
    acf[0] = 1.0
    return acf


def _spectrum(xf: np.ndarray) -> tuple[float, float]:
    """Dominant period (steps) and spectral flatness of the mean power spectrum."""
    t = xf.shape[1]
    if t < 4:
        return float("nan"), float("nan")
    xc = xf - xf.mean(1, keepdims=True)
    p = np.abs(np.fft.rfft(xc, axis=1)) ** 2  # (N, F, D)
    ps = p.mean((0, 2))
    ps = ps[1:]  # drop DC
    freqs = np.fft.rfftfreq(t)[1:]
    if ps.size == 0 or ps.sum() <= 0:
        return float("nan"), float("nan")
    k = int(np.argmax(ps))
    period = 1.0 / freqs[k]
    ps_pos = np.maximum(ps, 1e-300)
    flat = float(np.exp(np.mean(np.log(ps_pos))) / np.mean(ps_pos))
    return _f(period), _f(flat)


def _ridge_fit(a: np.ndarray, b: np.ndarray, lam: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares ``b ≈ a @ W + c`` with tiny ridge; returns ``(W, c)``."""
    am, bm = a.mean(0), b.mean(0)
    ac, bc = a - am, b - bm
    scale = ac.std(0).mean() ** 2 + 1e-12
    g = ac.T @ ac + lam * scale * np.eye(a.shape[1])
    w = np.linalg.solve(g, ac.T @ bc)
    return w, bm - am @ w


def _linear_r2(xf: np.ndarray, rng: np.random.Generator, max_pairs: int) -> tuple[float, float]:
    n, t, d = xf.shape
    if t < 3:
        return float("nan"), float("nan")
    a = xf[:, :-1].reshape(-1, d)
    b = xf[:, 1:].reshape(-1, d)
    if a.shape[0] > max_pairs:
        sel = rng.choice(a.shape[0], max_pairs, replace=False)
        a_fit, b_fit = a[sel], b[sel]
    else:
        a_fit, b_fit = a, b
    try:
        w, c = _ridge_fit(a_fit, b_fit)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan")
    tot = ((b - b.mean(0)) ** 2).sum()
    if tot <= 0:
        return float("nan"), float("nan")
    r2_1 = 1.0 - ((b - (a @ w + c)) ** 2).sum() / tot
    h = 10
    if t <= h:
        return _f(r2_1), float("nan")
    z = xf[:, : t - h].reshape(-1, d)
    for _ in range(h):
        z = z @ w + c
    target = xf[:, h:].reshape(-1, d)
    tot_h = ((target - target.mean(0)) ** 2).sum()
    if tot_h <= 0 or not np.all(np.isfinite(z)):
        return _f(r2_1), float("nan")
    r2_h = 1.0 - ((target - z) ** 2).sum() / tot_h
    return _f(r2_1), _f(max(r2_h, -1e6))


def _lyapunov_proxy(xf: np.ndarray, rng: np.random.Generator, n_pts: int = 1000,
                    follow: int = 20) -> float:
    """Rosenstein-style mean log-divergence slope of nearest-neighbour pairs (per step)."""
    n, t, d = xf.shape
    follow = min(follow, t - 2)
    if follow < 2:
        return float("nan")
    from scipy.spatial import cKDTree

    pts = xf[:, : t - follow].reshape(-1, d)
    ids = np.arange(pts.shape[0])
    if pts.shape[0] > n_pts:
        ids = np.sort(rng.choice(pts.shape[0], n_pts, replace=False))
    pts = pts[ids]
    tree = cKDTree(pts)
    per_traj = t - follow
    traj, tt = np.divmod(ids, per_traj)
    # exclude temporally-adjacent points of the same trajectory (Theiler window)
    theiler = max(1, min(10, per_traj // 4))
    kq = min(pts.shape[0], 12)
    dist, nn = tree.query(pts, k=kq)
    logs = []
    for i in range(pts.shape[0]):
        cand = None
        for j, dij in zip(nn[i, 1:], dist[i, 1:]):
            if traj[j] != traj[i] or abs(tt[j] - tt[i]) > theiler:
                if dij > 0:
                    cand = j
                break
        if cand is None:
            continue
        a = xf[traj[i], tt[i]: tt[i] + follow + 1]
        b = xf[traj[cand], tt[cand]: tt[cand] + follow + 1]
        dd = np.linalg.norm(a - b, axis=1)
        if np.all(dd > 0):
            logs.append(np.log(dd))
    if len(logs) < 5:
        return float("nan")
    curve = np.mean(np.array(logs), axis=0)  # (follow+1,)
    slope = np.polyfit(np.arange(curve.size), curve, 1)[0]
    return _f(slope)


# ----------------------------------------------------------------------------- main
def profile_dataset(ds: TrajectoryDataset, max_samples: int = 20000, seed: int = 0
                    ) -> DatasetProfile:
    """Profile ``ds`` (raw data, numpy only). See module docstring.

    ``max_samples`` bounds the number of ``(t, D)`` points used by PCA / linear
    fits; MLE (≤5000), correlation dimension (≤2000) and Lyapunov (≤1000) use
    tighter internal caps.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(ds.x, dtype=np.float64)
    n, t, d = x.shape
    dt = _f(ds.metadata.get("dt", np.nan))
    if not math.isfinite(dt) and t > 1:
        diffs = np.diff(np.asarray(ds.t, dtype=np.float64))
        if diffs.size and np.allclose(diffs, diffs[0]) and diffs[0] > 0:
            dt = float(diffs[0])

    nan_mask = ~np.isfinite(x)
    if ds.mask is not None:
        nan_mask |= ~ds.mask
        x = np.where(ds.mask, x, np.nan)
    has_missing = bool(nan_mask.any())
    missing_rate = float(nan_mask.mean()) if x.size else 0.0

    flat = x.reshape(-1, d)
    with np.errstate(all="ignore"):
        mean = np.nanmean(flat, 0)
        std = np.nanstd(flat, 0)
        rng_dim = np.nanmax(flat, 0) - np.nanmin(flat, 0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    std = np.where(np.isfinite(std), std, 0.0)
    rng_dim = np.where(np.isfinite(rng_dim), rng_dim, 0.0)

    xf = _fill_nans(x)  # (N, T, D) finite

    # ---- intrinsic dimension
    curve, pca_dims = _pca(xf.reshape(-1, d), rng, max_samples)
    pts_mle = _subsample_points(xf, 5000, rng)
    mle10 = _mle_dim(pts_mle, 10)
    mle20 = _mle_dim(pts_mle, 20)
    corr_dim = _correlation_dim(_subsample_points(xf, 2000, rng))

    dmax = max(1, min(64, d))
    cands: set[int] = set()
    mle_ref = mle10 if math.isfinite(mle10) else mle20
    seeds = [pca_dims["0.95"], pca_dims["0.99"]]
    if math.isfinite(mle_ref):
        seeds.append(int(math.ceil(mle_ref)))
    for s in seeds:
        for m in (0.5, 1.0, 2.0):
            cands.add(int(min(dmax, max(1, round(s * m)))))
    cands.update(g for g in _STANDARD_GRID if g <= d)
    suggested = sorted(cands)

    # ---- temporal
    max_lag = max(1, min(200, t // 2))
    acf = _acf(xf, max_lag) if t > 1 else np.array([1.0])
    below = np.nonzero(acf < 1.0 / math.e)[0]
    ac_time = float(below[0]) if below.size else float("nan")
    xc = xf - xf.reshape(-1, d).mean(0)
    denom = (xc**2).sum(-1).mean()
    if t > 1 and denom > 0:
        smooth = _f((np.diff(xf, axis=1) ** 2).sum(-1).mean() / denom)
    else:
        smooth = float("nan")
    period, flatness = _spectrum(xf)

    # ---- noise
    if t >= 3:
        d2 = np.diff(xf, n=2, axis=1)
        noise_std = _f(d2.std() / math.sqrt(6.0))
    else:
        noise_std = float("nan")
    signal_std = _f(np.sqrt(np.mean(std**2)))
    noise_ratio = _f(noise_std / signal_std) if signal_std > 0 else float("nan")

    # ---- stationarity
    if t >= 4:
        h = t // 2
        with np.errstate(all="ignore"):
            m1, m2 = np.nanmean(x[:, :h], 1), np.nanmean(x[:, h: 2 * h], 1)   # (N, D)
            s1, s2 = np.nanstd(x[:, :h], 1), np.nanstd(x[:, h: 2 * h], 1)
        ref = np.maximum(std, 1e-12)[None, :]
        dm = np.abs(m1 - m2) / ref
        dsd = np.abs(s1 - s2) / ref
        ns_mean = _f(np.nanmean(dm))
        ns_std = _f(np.nanmean(dsd))
        ns_frac = _f(np.nanmean((np.nanmean(dm, 0) > 0.5).astype(float)))
    else:
        ns_mean = ns_std = ns_frac = float("nan")

    # ---- linearity
    r2_1, r2_10 = _linear_r2(xf, rng, max_samples)

    # ---- chaos
    lyap = _lyapunov_proxy(xf, rng)
    lyap_t = _f(lyap / dt) if math.isfinite(dt) and dt > 0 else float("nan")

    rec = {
        "candidate_latent_dims": suggested,
        # one-step R² alone is ~1 for any smooth signal at fine dt (persistence), so
        # linearity additionally requires the recursive 10-step linear R² to hold up.
        "likely_linear": bool(math.isfinite(r2_1) and r2_1 > 0.98
                              and (not math.isfinite(r2_10) or r2_10 > 0.9)),
        "likely_chaotic": bool(math.isfinite(lyap) and lyap > 0.01),
        "noisy": bool(math.isfinite(noise_ratio) and noise_ratio > 0.05),
        "long_memory": bool(math.isfinite(ac_time) and ac_time > 50)
        or (not math.isfinite(ac_time) and max_lag > 50),
        "nonstationary": bool(math.isfinite(ns_frac) and ns_frac > 0.5),
    }

    return DatasetProfile(
        n_traj=int(n), n_steps=int(t), obs_dim=int(d), dt=dt, total_samples=int(n * t),
        has_missing=has_missing, missing_rate=missing_rate,
        sampling_rate_hz=_f(1.0 / dt) if math.isfinite(dt) and dt > 0 else float("nan"),
        mean_min=_f(mean.min()), mean_max=_f(mean.max()),
        std_min=_f(std.min()), std_median=_f(np.median(std)), std_max=_f(std.max()),
        dynamic_range=_f(rng_dim.max()),
        pca_dims_for_variance=pca_dims,
        explained_variance_curve=[float(c) for c in curve[:64]],
        mle_dim_k10=mle10, mle_dim_k20=mle20, correlation_dim=corr_dim,
        suggested_latent_dims=suggested,
        autocorr=[float(a) for a in acf], autocorr_time=ac_time, smoothness=smooth,
        dominant_period_steps=period, spectral_flatness=flatness,
        noise_std_estimate=noise_std, signal_std=signal_std, noise_ratio_estimate=noise_ratio,
        nonstationarity_mean=ns_mean, nonstationarity_std=ns_std,
        nonstationary_dim_fraction=ns_frac,
        linear_predictability_r2=r2_1, linear_r2_at_10_steps=r2_10,
        lyapunov_proxy=lyap, lyapunov_proxy_per_time=lyap_t,
        recommendations=rec,
    )
