"""Regression tests against textbook values of the synthetic systems (Gate A)."""

import numpy as np
import pytest

import nssc.data  # noqa: F401  (populate registry)
from nssc.data.integrators import integrate
from nssc.utils.registry import SYSTEMS


def _benettin_lyapunov(system, x0, dt, n_steps, eps=1e-8, renorm_every=10):
    """Largest Lyapunov exponent via two nearby trajectories with periodic renormalisation."""
    p = system.params
    f = lambda t, x: system.f(t, x, p)  # noqa: E731
    x = np.array([x0, x0 + np.array([eps, 0, 0])[: len(x0)]])
    total = 0.0
    n_ren = 0
    for _ in range(n_steps // renorm_every):
        x = integrate(f, x, dt, renorm_every + 1)[:, -1]
        d = np.linalg.norm(x[1] - x[0])
        total += np.log(d / eps)
        n_ren += 1
        x[1] = x[0] + (x[1] - x[0]) * eps / d
    return total / (n_ren * renorm_every * dt)


@pytest.mark.slow
def test_lorenz63_largest_lyapunov_exponent():
    sysm = SYSTEMS.build("lorenz63")
    x0 = sysm.simulate(1, 2, dt=0.01, seed=0, transient=2000)[0, -1]
    lam = _benettin_lyapunov(sysm, x0, dt=0.01, n_steps=60000)
    assert abs(lam - 0.905) < 0.1, lam  # literature: λ1 ≈ 0.9056


def test_lorenz63_attractor_statistics():
    x = SYSTEMS.build("lorenz63").simulate(20, 2000, dt=0.01, seed=0, transient=1000)
    assert abs(x[..., 2].mean() - 23.5) < 2.0  # mean z ≈ 23.5 on the attractor
    assert 7.0 < x[..., 0].std() < 9.0


def test_vanderpol_limit_cycle_period():
    # μ=1: period ≈ 6.66 time units
    sysm = SYSTEMS.build("vanderpol", params={"mu": 1.0})
    x = sysm.simulate(1, 4000, dt=0.01, seed=0, transient=2000)[0, :, 0]
    zc = np.nonzero((x[:-1] < 0) & (x[1:] >= 0))[0]
    period = np.diff(zc).mean() * 0.01
    assert abs(period - 6.66) < 0.1, period


def test_lotka_volterra_conserved_quantity():
    sysm = SYSTEMS.build("lotka_volterra")
    x = sysm.simulate(3, 3000, dt=0.005, seed=0, transient=0)
    V = sysm.energy(x)  # V = δx − γ ln x + βy − α ln y is conserved
    rel = V.std(axis=1) / np.abs(V.mean(axis=1))
    assert np.all(rel < 1e-3), rel
