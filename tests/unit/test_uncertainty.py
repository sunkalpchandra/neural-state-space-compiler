import torch

from nssc.models.builder import build_latent_model
from nssc.uncertainty import evaluate_uncertainty, probabilistic_rollout


def _model(dyn):
    return build_latent_model({"latent_dim": 3, "encoder": {"name": "mlp", "kwargs": {"hidden_dims": [16]}},
                               "decoder": {"name": "mlp", "kwargs": {"hidden_dims": [16]}},
                               "dynamics": dyn}, obs_dim=5)


def test_gaussian_rollout_shapes_and_calibration_keys():
    m = _model({"name": "gaussian", "kwargs": {"base": "residual_mlp", "base_kwargs": {"hidden_dims": [16]}}})
    x = torch.randn(4, 30, 5)
    r = probabilistic_rollout(m, x[:, :10], 8, n_samples=6)
    assert r["mean"].shape == (4, 8, 5) and r["samples"].shape == (6, 4, 8, 5)
    assert r["method"].startswith("gaussian")
    ev = evaluate_uncertainty(m, x, context=10, horizon=8, n_samples=6)
    for k in ("nll", "coverage95", "ece", "coverage95_curve", "std_error_corr"):
        assert k in ev
    assert len(ev["coverage95_curve"]) == 8


def test_deterministic_fallback_is_labelled():
    m = _model({"name": "linear"})
    r = probabilistic_rollout(m, torch.randn(2, 10, 5), 5, n_samples=4)
    assert "initial_perturbation" in r["method"] and torch.isfinite(r["std"]).all()
