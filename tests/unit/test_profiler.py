"""Unit tests for :mod:`nssc.compiler.profiler`."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from nssc.compiler import DatasetProfile, profile_dataset
from nssc.data.builder import build_dataset
from nssc.data.dataset import TrajectoryDataset

LORENZ = {"system": "lorenz63", "n_traj": 20, "n_steps": 400, "dt": 0.01, "transient": 1000,
          "seed": 0}


def _all_finite_or_nan(d: dict) -> None:
    for k, v in d.items():
        if isinstance(v, float):
            assert not math.isinf(v), k
        elif isinstance(v, list):
            assert all(not math.isinf(float(a)) for a in v), k
        elif isinstance(v, dict):
            _all_finite_or_nan(v)


@pytest.fixture(scope="module")
def harmonic_profile() -> DatasetProfile:
    ds = build_dataset({"system": "harmonic", "n_traj": 8, "n_steps": 200, "dt": 0.05, "seed": 0})
    return profile_dataset(ds)


@pytest.fixture(scope="module")
def lorenz_profile() -> DatasetProfile:
    return profile_dataset(build_dataset(LORENZ))


def test_harmonic_is_linear_2d(harmonic_profile: DatasetProfile) -> None:
    p = harmonic_profile
    assert p.n_traj == 8 and p.n_steps == 200 and p.obs_dim == 2 and p.dt == 0.05
    assert p.sampling_rate_hz == pytest.approx(20.0)
    assert p.pca_dims_for_variance["0.95"] == 2
    assert p.recommendations["likely_linear"] is True
    assert p.linear_predictability_r2 > 0.98
    assert abs(p.lyapunov_proxy) < 0.02
    assert p.recommendations["likely_chaotic"] is False
    assert not p.has_missing and p.missing_rate == 0.0
    assert p.recommendations["noisy"] is False
    _all_finite_or_nan(p.to_dict())


def test_lorenz_is_chaotic_low_dim(lorenz_profile: DatasetProfile) -> None:
    p = lorenz_profile
    assert 1.5 <= p.mle_dim_k10 <= 3.5
    assert 1.5 <= p.mle_dim_k20 <= 3.5
    assert p.recommendations["likely_chaotic"] is True
    assert 0.3 <= p.lyapunov_proxy_per_time <= 2.0
    assert p.recommendations["likely_linear"] is False
    assert p.linear_r2_at_10_steps < 0.9
    assert p.pca_dims_for_variance["0.99"] <= 3
    assert 2 in p.suggested_latent_dims
    _all_finite_or_nan(p.to_dict())


def test_lorenz_mlp_observation_suggests_low_dims() -> None:
    cfg = dict(LORENZ, observation={"type": "mlp", "obs_dim": 32, "seed": 0})
    p = profile_dataset(build_dataset(cfg))
    assert p.obs_dim == 32
    assert p.pca_dims_for_variance["0.95"] < 32
    assert {2, 4, 8}.issubset(set(p.suggested_latent_dims))
    assert p.suggested_latent_dims == sorted(set(p.suggested_latent_dims))
    assert all(1 <= k <= 32 for k in p.suggested_latent_dims)
    assert p.mle_dim_k10 < 6
    assert len(p.explained_variance_curve) == 32
    assert p.explained_variance_curve[-1] == pytest.approx(1.0)


def test_missing_values_handled() -> None:
    p = profile_dataset(build_dataset(dict(LORENZ, missing_rate=0.1)))
    assert p.has_missing
    assert 0.05 < p.missing_rate < 0.15
    d = p.to_dict()
    _all_finite_or_nan(d)
    for k in ("std_median", "mle_dim_k10", "linear_predictability_r2", "lyapunov_proxy",
              "noise_std_estimate", "autocorr_time"):
        assert math.isfinite(d[k]), k


def test_to_dict_json_and_markdown(harmonic_profile: DatasetProfile) -> None:
    s = json.dumps(harmonic_profile.to_dict(), default=float)
    back = json.loads(s)
    assert back["obs_dim"] == 2 and "recommendations" in back
    md = harmonic_profile.to_markdown()
    assert "linear_predictability_r2" in md and "Recommendations" in md


def test_small_T_and_D1_and_no_dt() -> None:
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(size=(3, 64, 1)), axis=1).astype(np.float32)
    ds = TrajectoryDataset(x=x, t=np.arange(64, dtype=float), metadata={})
    p = profile_dataset(ds)
    assert p.obs_dim == 1 and p.n_steps == 64
    assert p.dt == pytest.approx(1.0)  # inferred from t
    assert p.suggested_latent_dims == [1]
    assert len(p.autocorr) == 33
    _all_finite_or_nan(p.to_dict())


def test_determinism_under_seed(harmonic_profile: DatasetProfile) -> None:
    ds = build_dataset({"system": "harmonic", "n_traj": 8, "n_steps": 200, "dt": 0.05, "seed": 0})
    again = profile_dataset(ds, seed=0).to_dict()
    assert json.dumps(again, default=float) == json.dumps(harmonic_profile.to_dict(), default=float)
