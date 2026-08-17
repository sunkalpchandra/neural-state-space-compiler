"""Unit tests for nssc.data.observation."""

from __future__ import annotations

import numpy as np
import pytest

from nssc.data.observation import (
    OBS_MAPS,
    IdentityObservation,
    LinearObservation,
    ObservationMap,
    ObservationPipeline,
    PolynomialObservation,
    RandomMLPObservation,
    RedundantObservation,
    add_noise,
    irregular_subsample,
    mask_missing,
)
from nssc.utils.seeding import rng

Z = rng(0).normal(size=(4, 20, 3))


@pytest.mark.parametrize("m,D", [
    (IdentityObservation(), 3),
    (LinearObservation(16, seed=1), 16),
    (LinearObservation(2, seed=1, orthogonal=True), 2),
    (RandomMLPObservation(32, hidden=[16, 16], seed=2), 32),
    (PolynomialObservation(degree=2), 3 + 6),
    (PolynomialObservation(degree=2, obs_dim=5, seed=0), 5),
    (RedundantObservation(repeats=4, seed=0), 12),
    (ObservationPipeline([RandomMLPObservation(8, seed=0), LinearObservation(6, seed=1)]), 6),
])
def test_shape_finite_deterministic_roundtrip(m, D):
    x = m(Z)
    assert x.shape == (4, 20, D)
    assert np.isfinite(x).all()
    cfg = m.to_config()
    m2 = ObservationMap.from_config(cfg)
    assert m2 == m
    np.testing.assert_allclose(m2(Z), x)
    m3 = ObservationMap.from_config(cfg)  # fresh object, same seed
    np.testing.assert_allclose(m3(Z), x)


def test_seed_changes_map():
    a, b = LinearObservation(8, seed=0)(Z), LinearObservation(8, seed=1)(Z)
    assert not np.allclose(a, b)
    a, b = RandomMLPObservation(8, seed=0)(Z), RandomMLPObservation(8, seed=1)(Z)
    assert not np.allclose(a, b)


def test_orthogonal_linear_is_isometry():
    m = LinearObservation(3, seed=0, orthogonal=True)
    W = m.matrix(3)
    np.testing.assert_allclose(W.T @ W, np.eye(3), atol=1e-10)


def test_all_kinds_registered():
    assert set(OBS_MAPS) == {"identity", "linear", "mlp", "polynomial", "redundant", "pipeline"}


def test_add_noise_and_mask():
    g = rng(0)
    y = add_noise(Z, 0.1, g)
    assert y.shape == Z.shape and not np.allclose(y, Z)
    assert np.allclose(add_noise(Z, 0.0, g), Z)
    xm, mask = mask_missing(Z, 0.3, rng(0))
    assert mask.shape == Z.shape and mask.dtype == bool
    assert np.isnan(xm[~mask]).all() and np.isfinite(xm[mask]).all()
    assert 0.2 < 1 - mask.mean() < 0.4
    xm2, mask2 = mask_missing(Z, 0.3, rng(0))
    np.testing.assert_array_equal(mask, mask2)


def test_irregular_subsample():
    t = np.arange(20) * 0.1
    tk, xk = irregular_subsample(t, Z, 0.5, rng(0))
    assert tk[0] == 0.0 and len(tk) == xk.shape[1] and xk.shape[0] == 4 and xk.shape[2] == 3
    assert 3 <= len(tk) < 20
    np.testing.assert_array_equal(xk[:, 0], Z[:, 0])
