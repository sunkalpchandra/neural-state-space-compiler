import math

import torch

from nssc.dynamics import build_dynamics
from nssc.stability import jacobian_spectrum, largest_lyapunov_exponent, spectral_radius_stats


def _linear_with_A(A):
    d = A.shape[0]
    dyn = build_dynamics("linear", latent_dim=d)
    with torch.no_grad():
        dyn.A.copy_(A)
    return dyn


def test_spectral_radius_of_linear_map():
    A = torch.diag(torch.tensor([0.5, 0.9, 1.2]))
    dyn = _linear_with_A(A)
    z = torch.randn(10, 3)
    s = spectral_radius_stats(dyn, z)
    assert abs(s["rho_max"] - 1.2) < 1e-5 and abs(s["rho_min"] - 1.2) < 1e-5
    assert s["frac_expanding"] == 1.0
    spec = jacobian_spectrum(dyn, z, max_points=4)
    assert spec["eigvals"].shape == (4, 3)


def test_lyapunov_of_linear_expansion():
    # x' = 2x  → λ = log 2 per step, regardless of direction after transient
    A = torch.diag(torch.tensor([2.0, 0.5]))
    dyn = _linear_with_A(A)
    z0 = torch.randn(4, 2)
    lam = largest_lyapunov_exponent(dyn, z0, n_steps=100, n_transient=20)
    assert abs(lam["lyapunov_max_mean"] - math.log(2)) < 1e-3


def test_lyapunov_contracting_is_negative():
    A = torch.diag(torch.tensor([0.5, 0.25]))
    lam = largest_lyapunov_exponent(_linear_with_A(A), torch.randn(3, 2), n_steps=50)
    assert lam["lyapunov_max_mean"] < 0
