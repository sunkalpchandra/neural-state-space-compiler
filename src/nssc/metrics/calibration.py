"""Calibration metrics for probabilistic rollouts (Gaussian predictive distributions)."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor


def _np(x):
    return x.detach().cpu().double().numpy() if isinstance(x, torch.Tensor) else np.asarray(x, float)


def gaussian_nll(mean, var, target) -> float:
    m, v, t = _np(mean), np.maximum(_np(var), 1e-8), _np(target)
    return float(np.mean(0.5 * (np.log(2 * math.pi * v) + (t - m) ** 2 / v)))


def coverage(mean, std, target, z: float = 1.96) -> float:
    """Fraction of targets inside mean ± z·std (nominal 95% for z=1.96)."""
    m, s, t = _np(mean), _np(std), _np(target)
    inside = np.abs(t - m) <= z * s
    return float(inside.mean())


def expected_calibration_error_regression(mean, std, target,
                                          levels: tuple[float, ...] = (0.5, 0.68, 0.8, 0.9, 0.95, 0.99)
                                          ) -> dict[str, float]:
    """Mean |empirical coverage − nominal| across confidence levels + per-level coverage."""
    from scipy.stats import norm

    out = {}
    errs = []
    for lv in levels:
        zq = norm.ppf(0.5 + lv / 2)
        c = coverage(mean, std, target, z=zq)
        out[f"coverage@{lv}"] = c
        errs.append(abs(c - lv))
    out["ece"] = float(np.mean(errs))
    return out


def sharpness(std: Tensor | np.ndarray) -> float:
    return float(np.mean(_np(std)))
