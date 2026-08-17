"""Uncertainty envelopes for long-horizon rollouts.

For ``GaussianDynamics`` (``is_stochastic``) we propagate Monte-Carlo samples
through the latent transition and decode each sample, giving a predictive
distribution over observations at every horizon step. For deterministic
dynamics we fall back to an *ensemble-of-initial-perturbations* envelope
(perturb the encoded initial latent by ``eps``), which is clearly labelled as
such in the returned dict (``method``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from nssc.metrics.calibration import coverage, expected_calibration_error_regression, gaussian_nll
from nssc.models.latent_model import LatentModel


@torch.no_grad()
def probabilistic_rollout(model: LatentModel, x_context: Tensor, horizon: int, n_samples: int = 32,
                          eps: float = 0.01) -> dict[str, Any]:
    """Returns dict with ``mean`` (B,H,D), ``std`` (B,H,D), ``samples`` (S,B,H,D), ``method``."""
    model.eval()
    z = model.encode(x_context)
    z0 = z[:, -1]
    dyn = model.dynamics
    if getattr(dyn, "is_stochastic", False) and hasattr(dyn, "rollout_samples"):
        zs = dyn.rollout_samples(z0, horizon, n_samples=n_samples)  # (S,B,H,d)
        method = "gaussian_transition_mc"
    else:
        S = n_samples
        z0s = z0.unsqueeze(0) + eps * torch.randn(S, *z0.shape, device=z0.device)
        zs = torch.stack([dyn.rollout(z0s[s], horizon) for s in range(S)])
        method = f"initial_perturbation_ensemble(eps={eps})"
    Sn, B, H, d = zs.shape
    xs = model.decode(zs.reshape(Sn * B, H, d)).reshape(Sn, B, H, -1)
    return {"mean": xs.mean(0), "std": xs.std(0, unbiased=False) + 1e-8, "samples": xs,
            "latent_samples": zs, "method": method}


@torch.no_grad()
def evaluate_uncertainty(model: LatentModel, x: Tensor, context: int, horizon: int,
                         n_samples: int = 32, batch_size: int = 32) -> dict[str, Any]:
    """Calibration of the predictive envelope on held-out ``x`` (N,T,D)."""
    H = min(horizon, x.shape[1] - context)
    means, stds, tgts = [], [], []
    method = ""
    for i in range(0, x.shape[0], batch_size):
        xb = x[i : i + batch_size]
        r = probabilistic_rollout(model, xb[:, :context], H, n_samples)
        method = r["method"]
        means.append(r["mean"])
        stds.append(r["std"])
        tgts.append(xb[:, context : context + H])
    m, s, t = torch.cat(means), torch.cat(stds), torch.cat(tgts)
    out: dict[str, Any] = {"method": method, "horizon": H,
                           "nll": gaussian_nll(m, s**2, t), "coverage95": coverage(m, s, t),
                           "sharpness": float(s.mean())}
    out.update(expected_calibration_error_regression(m, s, t))
    # calibration along the horizon: coverage@95 per step
    inside = (torch.abs(t - m) <= 1.96 * s).float().mean(dim=(0, 2))
    out["coverage95_curve"] = inside.cpu().numpy().tolist()
    out["std_curve"] = s.mean(dim=(0, 2)).cpu().numpy().tolist()
    out["error_curve"] = torch.sqrt(((m - t) ** 2).mean(dim=(0, 2))).cpu().numpy().tolist()
    # correlation between predicted std and actual abs error across horizon (should be > 0)
    e = np.asarray(out["error_curve"])
    sd = np.asarray(out["std_curve"])
    out["std_error_corr"] = float(np.corrcoef(e, sd)[0, 1]) if len(e) > 2 and e.std() > 0 and sd.std() > 0 else float("nan")
    return out
