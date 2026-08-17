"""Local linearisation spectra of latent dynamics."""

from __future__ import annotations

import torch
from torch import Tensor

from nssc.dynamics.base import Dynamics


@torch.no_grad()
def _eig(J: Tensor) -> Tensor:
    return torch.linalg.eigvals(J.detach().cpu().double())  # (B, d) complex; MPS has no float64


def jacobian_spectrum(dynamics: Dynamics, z: Tensor, max_points: int = 256) -> dict[str, Tensor]:
    """Eigenvalues of ∂F/∂z at up to ``max_points`` latent states.

    ``z``: (N, d) or (B, T, d). Returns dict with ``eigvals`` (M, d) complex,
    ``spectral_radius`` (M,), ``points`` (M, d).
    """
    flat = z.reshape(-1, z.shape[-1])
    if flat.shape[0] > max_points:
        idx = torch.linspace(0, flat.shape[0] - 1, max_points).long()
        flat = flat[idx]
    was_training = dynamics.training
    dynamics.eval()
    J = dynamics.jacobian(flat)  # (M, d, d)
    dynamics.train(was_training)
    eig = _eig(J.detach())
    rho = eig.abs().max(dim=-1).values.float()
    return {"eigvals": eig, "spectral_radius": rho, "points": flat.detach(), "jacobians": J.detach()}


def spectral_radius_stats(dynamics: Dynamics, z: Tensor, max_points: int = 256) -> dict[str, float]:
    s = jacobian_spectrum(dynamics, z, max_points)
    rho = s["spectral_radius"]
    return {
        "rho_mean": float(rho.mean()),
        "rho_max": float(rho.max()),
        "rho_min": float(rho.min()),
        "rho_std": float(rho.std(unbiased=False)) if rho.numel() > 1 else 0.0,
        "frac_expanding": float((rho > 1.0).float().mean()),
        "max_real_eig": float(s["eigvals"].real.max()),
    }
