"""Unit tests for nssc.data.integrators and nssc.data.systems."""

from __future__ import annotations

import numpy as np
import pytest

import nssc.data  # noqa: F401  (registers systems)
from nssc.data.integrators import euler, rk4
from nssc.data.systems.kuramoto import Kuramoto, observe_sin_cos, order_parameter
from nssc.utils.registry import SYSTEMS

ALL_KEYS = [
    "harmonic", "damped_oscillator", "pendulum", "lorenz63", "lorenz96", "vanderpol",
    "lotka_volterra", "fitzhugh_nagumo", "coupled_oscillators", "kuramoto", "gray_scott",
]


def test_registry_has_all_systems():
    for k in ALL_KEYS:
        assert k in SYSTEMS


@pytest.mark.parametrize("key", ALL_KEYS)
def test_shape_finite_deterministic(key):
    s = SYSTEMS.build(key)
    z = s.simulate(3, 50, seed=1)
    assert z.shape == (3, 50, s.state_dim)
    assert z.dtype == np.float64
    assert np.all(np.isfinite(z))
    z2 = SYSTEMS.build(key).simulate(3, 50, seed=1)
    np.testing.assert_array_equal(z, z2)
    z3 = s.simulate(3, 50, seed=2)
    assert not np.allclose(z, z3)


def test_rk4_energy_conservation_harmonic():
    s = SYSTEMS.build("harmonic")
    z = s.simulate(16, 1000, dt=0.01, seed=0)
    e = s.energy(z)  # (N, T)
    rel = np.abs(e - e[:, :1]) / e[:, :1]
    assert rel.max() < 1e-4


def test_rk4_more_accurate_than_euler():
    s = SYSTEMS.build("harmonic")
    x0 = np.array([[1.0, 0.0]])
    dt, n = 0.05, 200

    def f(t, x):
        return s.f(t, x, s.params)

    zr, ze = rk4(f, x0, dt, n), euler(f, x0, dt, n)
    tt = np.arange(n) * dt
    exact = np.cos(tt)
    assert np.abs(zr[0, :, 0] - exact).max() < 1e-5
    assert np.abs(ze[0, :, 0] - exact).max() > 1e-2


def test_rk4_substeps_match_finer_dt():
    s = SYSTEMS.build("lorenz63")
    z_coarse = s.simulate(2, 100, dt=0.05, substeps=5, seed=0, transient=0)
    z_fine = s.simulate(2, 496, dt=0.01, seed=0, transient=0)
    np.testing.assert_allclose(z_coarse[:, :100], z_fine[:, ::5], atol=1e-8)


def test_pendulum_energy_conserved_when_undamped():
    s = SYSTEMS.build("pendulum", params={"gamma": 0.0})
    z = s.simulate(8, 1000, dt=0.01)
    e = s.energy(z)
    assert np.abs(e - e[:, :1]).max() / e.max() < 1e-4


def test_lorenz63_attractor_bounds():
    z = SYSTEMS.build("lorenz63").simulate(8, 2000, seed=3)
    assert np.abs(z[..., 0]).max() < 30
    assert np.abs(z[..., 1]).max() < 40
    assert (z[..., 2] > 0).all() and z[..., 2].max() < 60


def test_lorenz63_param_override_and_fixed_points():
    s = SYSTEMS.build("lorenz63", params={"rho": 20.0})
    assert s.params["rho"] == 20.0 and s.params["sigma"] == 10.0
    fp = s.fixed_points()
    assert fp.shape == (3, 3)
    assert np.allclose(s.f(0.0, fp, s.params), 0.0, atol=1e-10)
    z_a = s.simulate(2, 100, seed=0)
    z_b = s.simulate(2, 100, seed=0, params={"rho": 28.0})
    assert not np.allclose(z_a, z_b)
    with pytest.raises(KeyError):
        SYSTEMS.build("lorenz63", params={"nope": 1.0})


def test_lotka_volterra_positive_and_conserved():
    s = SYSTEMS.build("lotka_volterra")
    z = s.simulate(8, 2000, seed=0)
    assert (z > 0).all()
    v = s.energy(z)
    assert np.abs(v - v[:, :1]).max() < 1e-3


def test_vanderpol_settles_on_limit_cycle():
    z = SYSTEMS.build("vanderpol").simulate(4, 2000, seed=0)
    amp = np.abs(z[:, 1000:, 0]).max(axis=1)
    assert np.all((amp > 1.8) & (amp < 2.2))


def test_lorenz96_state_dim_follows_N():
    s = SYSTEMS.build("lorenz96", params={"N": 12})
    assert s.state_dim == 12
    z = s.simulate(2, 100, seed=0)
    assert z.shape[-1] == 12 and np.isfinite(z).all()


def test_coupled_oscillators_energy_conserved():
    s = SYSTEMS.build("coupled_oscillators", params={"N": 5, "c": 0.0})
    assert s.state_dim == 10
    z = s.simulate(4, 500, dt=0.02)
    e = s.energy(z)
    assert np.abs(e - e[:, :1]).max() / e.max() < 1e-4


def test_kuramoto_frequencies_deterministic_and_sync():
    a, b = Kuramoto(), Kuramoto()
    np.testing.assert_array_equal(a.omega, b.omega)
    assert not np.allclose(a.omega, Kuramoto(params={"freq_seed": 1}).omega)
    z = Kuramoto(params={"K": 6.0}).simulate(4, 400, seed=0)
    r = order_parameter(z)
    assert r[:, -1].mean() > 0.9
    obs = observe_sin_cos(z)
    assert obs.shape == (4, 400, 16)
    assert np.allclose(obs[..., :8] ** 2 + obs[..., 8:] ** 2, 1.0)


def test_gray_scott_bounded_and_nontrivial():
    s = SYSTEMS.build("gray_scott")
    assert s.state_dim == 64
    z = s.simulate(2, 300, seed=0)
    assert z.min() > -1e-6 and z.max() < 1.0 + 1e-6
    assert z[..., 32:].sum(-1).min() > 1.0  # v has not died out
    assert z.std(axis=1).mean() > 1e-3        # dynamics not frozen
    with pytest.raises(ValueError):
        SYSTEMS.build("gray_scott", params={"Du": 1.0})


def test_fitzhugh_nagumo_oscillates():
    z = SYSTEMS.build("fitzhugh_nagumo").simulate(4, 1000, seed=0)
    assert z[:, 500:, 0].max() > 1.0 and z[:, 500:, 0].min() < -1.0
