import numpy as np
import torch

from nssc.metrics import (
    coverage,
    expected_calibration_error_regression,
    gaussian_nll,
    horizon_curve,
    mse,
    nrmse,
    r2_score,
    rollout_divergence_time,
    rollout_errors,
)


def test_mse_nrmse_basic():
    t = torch.randn(4, 10, 3)
    assert mse(t, t) == 0.0
    assert nrmse(t, t) == 0.0
    assert r2_score(t, t) > 0.999
    p = t + 1.0
    assert abs(mse(p, t) - 1.0) < 1e-6


def test_nrmse_with_sigma_scaling():
    t = np.zeros((2, 5, 1))
    p = np.ones((2, 5, 1))
    assert abs(nrmse(p, t, sigma=np.array([2.0])) - 0.5) < 1e-9


def test_rollout_errors_and_divergence():
    B, H, D = 3, 20, 2
    t = np.zeros((B, H, D))
    p = np.cumsum(np.ones((B, H, D)) * 0.1, axis=1)  # error grows linearly
    curve = horizon_curve(p, t, sigma=np.ones(D))
    assert curve.shape == (H,) and np.all(np.diff(curve) > 0)
    e = rollout_errors(p, t, horizons=(1, 5, 10, 100), sigma=np.ones(D))
    assert "nrmse@10" in e and "nrmse@100" not in e
    assert e["nrmse@1"] < e["nrmse@10"]
    assert rollout_divergence_time(p, t, threshold=1.0, sigma=np.ones(D)) == 11.0
    assert rollout_divergence_time(t, t, sigma=np.ones(D)) == H + 1


def test_calibration_perfect_gaussian():
    rng = np.random.default_rng(0)
    m = np.zeros((1000, 1, 1))
    s = np.ones_like(m)
    t = rng.normal(size=m.shape)
    c = coverage(m, s, t)
    assert 0.93 < c < 0.97
    ece = expected_calibration_error_regression(m, s, t)
    assert ece["ece"] < 0.05
    assert np.isfinite(gaussian_nll(m, s**2, t))


def test_forecast_latency_is_measured_the_same_way_for_both_harnesses():
    """R-49: the comparable cost metric must exist with the same key shape on both paths."""
    import torch

    from nssc.baselines import build_baseline
    from nssc.baselines.evaluate import evaluate_forecaster
    from nssc.evaluation import EvalConfig, evaluate_model
    from nssc.models.builder import build_latent_model

    x = torch.randn(4, 40, 3)
    lm = build_latent_model({"latent_dim": 2, "encoder": {"name": "mlp", "kwargs": {"hidden_dims": [8]}},
                             "decoder": {"name": "mlp", "kwargs": {"hidden_dims": [8]}},
                             "dynamics": {"name": "linear"}}, 3)
    a = evaluate_model(lm, x, EvalConfig(context=10, horizons=(1, 5, 10), latency=True,
                                         latency_horizon=10, stability=False))
    b = evaluate_forecaster(build_baseline("gru", 3, hidden=8), x, context=10, horizons=(1, 5, 10),
                            device=torch.device("cpu"), latency=True, latency_horizon=10)
    assert a["latency/horizon"] == b["latency/horizon"] == 10
    for k in ("latency/forecast10_latency_ms_mean",):
        assert k in a and k in b and a[k] > 0 and b[k] > 0
