"""Aggregate stability report for a LatentModel on held-out data."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor

from nssc.models.latent_model import LatentModel
from nssc.stability.lyapunov import largest_lyapunov_exponent
from nssc.stability.spectral import spectral_radius_stats


@dataclass
class StabilityReport:
    spectral: dict[str, float] = field(default_factory=dict)
    lyapunov: dict[str, float] = field(default_factory=dict)
    norm_growth: dict[str, float] = field(default_factory=dict)
    rollout: dict[str, float] = field(default_factory=dict)
    instability_score: float = 0.0
    verdict: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@torch.no_grad()
def latent_norm_growth(model: LatentModel, x: Tensor, horizon: int) -> dict[str, float]:
    """Compare ||ẑ_t|| along a free rollout to the norm distribution of encoded data."""
    z_data = model.encode(x)
    ref = z_data.norm(dim=-1)  # (B, T)
    z0 = z_data[:, 0]
    z_roll = model.dynamics.rollout(z0, horizon)
    n = z_roll.norm(dim=-1)  # (B, H)
    finite = torch.isfinite(n).all(dim=1)
    ratio_end = (n[:, -1] / (ref.mean() + 1e-8))
    return {
        "ref_norm_mean": float(ref.mean()),
        "ref_norm_max": float(ref.max()),
        "rollout_norm_end_mean": float(n[finite, -1].mean()) if finite.any() else float("inf"),
        "rollout_norm_max": float(n[finite].max()) if finite.any() else float("inf"),
        "norm_ratio_end": float(ratio_end[finite].mean()) if finite.any() else float("inf"),
        "frac_nonfinite": float(1 - finite.float().mean()),
        "frac_blowup": float(((n > 10 * ref.max()).any(dim=1) | ~finite).float().mean()),
        "frac_collapse": float((n[:, -1] < 1e-3 * ref.mean()).float().mean()) if finite.any() else 0.0,
    }


def analyze_stability(model: LatentModel, x: Tensor, horizon: int = 200, dt: float = 1.0,
                      lyapunov_steps: int = 200, max_points: int = 128) -> StabilityReport:
    """``x``: (B, T, D) held-out observations (already normalised like training data)."""
    model.eval()
    z = model.encode(x)
    rep = StabilityReport()
    rep.spectral = spectral_radius_stats(model.dynamics, z.detach(), max_points=max_points)
    B = min(16, z.shape[0])
    rep.lyapunov = largest_lyapunov_exponent(model.dynamics, z[:B, 0].detach(),
                                             n_steps=lyapunov_steps, dt=dt)
    rep.norm_growth = latent_norm_growth(model, x, horizon)
    # scalar instability score in [0, ∞): blow-up fraction dominates, then expanding spectra
    ng = rep.norm_growth
    score = (2.0 * ng["frac_blowup"] + 1.0 * ng["frac_collapse"]
             + 0.5 * max(0.0, rep.spectral["rho_max"] - 1.0)
             + 0.25 * max(0.0, np.log(max(ng["norm_ratio_end"], 1e-8))) if np.isfinite(ng["norm_ratio_end"]) else 5.0)
    rep.instability_score = float(score)
    if ng["frac_blowup"] > 0.5:
        rep.verdict = "explodes"
    elif ng["frac_collapse"] > 0.5:
        rep.verdict = "collapses"
    elif rep.spectral["rho_max"] > 1.5:
        rep.verdict = "locally_expanding"
    else:
        rep.verdict = "stable"
    return rep
