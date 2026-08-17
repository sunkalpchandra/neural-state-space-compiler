"""Reconstruction / prediction / rollout error metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor

DEFAULT_HORIZONS: tuple[int, ...] = (1, 5, 10, 25, 50, 100, 250, 500)


def _np(x: Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().double().numpy()
    return np.asarray(x, dtype=np.float64)


def mse(pred, target) -> float:
    p, t = _np(pred), _np(target)
    return float(np.mean((p - t) ** 2))


def rmse(pred, target) -> float:
    return float(np.sqrt(mse(pred, target)))


def nrmse(pred, target, sigma: float | np.ndarray | None = None) -> float:
    """RMSE divided by the (per-dim, then averaged) std of the target.

    ``sigma`` may be provided from the *training* set to avoid test-set leakage.
    """
    p, t = _np(pred), _np(target)
    if sigma is None:
        sigma = t.reshape(-1, t.shape[-1]).std(axis=0)
    sigma = np.asarray(sigma, dtype=np.float64)
    per_dim = np.sqrt(np.mean((p - t) ** 2, axis=tuple(range(p.ndim - 1))))
    return float(np.mean(per_dim / (sigma + 1e-12)))


def r2_score(pred, target) -> float:
    p, t = _np(pred), _np(target)
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - t.reshape(-1, t.shape[-1]).mean(0)) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-12))


def horizon_curve(pred, target, sigma=None) -> np.ndarray:
    """Per-step NRMSE along the horizon axis. pred/target: (B, H, D) → (H,)."""
    p, t = _np(pred), _np(target)
    if sigma is None:
        sigma = t.reshape(-1, t.shape[-1]).std(axis=0)
    err = np.sqrt(np.mean((p - t) ** 2, axis=0))  # (H, D)
    return np.mean(err / (np.asarray(sigma) + 1e-12), axis=1)


def rollout_errors(pred, target, horizons: Sequence[int] = DEFAULT_HORIZONS, sigma=None
                   ) -> dict[str, float]:
    """NRMSE accumulated up to each horizon k (mean over steps 1..k) plus per-step at k.

    Keys: ``nrmse@k`` (cumulative), ``nrmse_step@k`` (instantaneous). Horizons
    beyond the available length are skipped.
    """
    curve = horizon_curve(pred, target, sigma)
    H = len(curve)
    out: dict[str, float] = {}
    for k in horizons:
        if k <= H:
            out[f"nrmse@{k}"] = float(curve[:k].mean())
            out[f"nrmse_step@{k}"] = float(curve[k - 1])
    return out


def rollout_divergence_time(pred, target, threshold: float = 1.0, sigma=None) -> float:
    """First horizon step at which per-step NRMSE exceeds ``threshold`` (H+1 if never)."""
    curve = horizon_curve(pred, target, sigma)
    idx = np.nonzero(curve > threshold)[0]
    return float(idx[0] + 1) if len(idx) else float(len(curve) + 1)
