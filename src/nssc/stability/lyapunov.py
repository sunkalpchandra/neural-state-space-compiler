"""Largest Lyapunov exponent of a discrete latent map via tangent-vector renormalisation.

    λ_max ≈ (1/(K·Δt)) Σ_k log ||J_k v_k|| / ||v_k||

Positive λ indicates local exponential divergence (chaos or instability); for a
learned model of Lorenz-63 in latent time units one expects λ·(1/dt) ≈ 0.9 if the
latent dynamics faithfully reproduce the attractor.
"""

from __future__ import annotations

import torch
from torch import Tensor

from nssc.dynamics.base import Dynamics


def largest_lyapunov_exponent(dynamics: Dynamics, z0: Tensor, n_steps: int = 500, dt: float = 1.0,
                              n_transient: int = 50, seed: int = 0) -> dict[str, float]:
    """``z0``: (B, d) initial latents. Returns mean/std over batch of λ_max (per unit ``dt``)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    z = z0.detach().clone()
    v = torch.randn(z.shape, generator=g).to(z)
    v = v / v.norm(dim=-1, keepdim=True)
    log_growth = torch.zeros(z.shape[0], device=z.device)
    was_training = dynamics.training
    dynamics.eval()
    with torch.no_grad():
        for k in range(n_transient + n_steps):
            J = dynamics.jacobian(z)  # (B, d, d)
            v = torch.einsum("bij,bj->bi", J, v)
            nrm = v.norm(dim=-1, keepdim=True) + 1e-30
            v = v / nrm
            if k >= n_transient:
                log_growth += torch.log(nrm.squeeze(-1))
            z = dynamics.step(z)
            if not torch.isfinite(z).all():
                break
    dynamics.train(was_training)
    lam = log_growth / (n_steps * dt)
    lam = lam[torch.isfinite(lam)]
    return {"lyapunov_max_mean": float(lam.mean()) if lam.numel() else float("nan"),
            "lyapunov_max_std": float(lam.std(unbiased=False)) if lam.numel() > 1 else 0.0}
