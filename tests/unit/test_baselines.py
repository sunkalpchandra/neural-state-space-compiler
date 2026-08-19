"""Unit tests for nssc.baselines (sequence-model comparators + trivial baselines)."""

from __future__ import annotations

import pytest
import torch

from nssc.baselines import (
    BASELINES,
    BaselineTrainer,
    BaselineTrainerConfig,
    LatentModelForecaster,
    build_baseline,
    evaluate_forecaster,
)
from nssc.baselines.run import load_forecaster_checkpoint, load_preset, run_baseline_experiment
from nssc.data.builder import build_dataset
from nssc.data.dataset import make_loaders

D = 5
KEYS = ["persistence", "mean", "gru", "lstm", "tcn", "transformer", "ssm"]
SMALL = {"gru": dict(hidden=8), "lstm": dict(hidden=8), "tcn": dict(channels=8, n_layers=2),
         "transformer": dict(d_model=8, n_heads=2, n_layers=1, max_len=64),
         "ssm": dict(d_model=8, d_state=4, n_layers=1)}
TINY_DS = {"system": "harmonic", "n_traj": 6, "n_steps": 64, "dt": 0.05, "seed": 0}


def test_registry_keys():
    assert set(KEYS) <= set(BASELINES.keys())


@pytest.mark.parametrize("key", KEYS)
def test_build_forecast_recursive(key):
    m = build_baseline(key, D, **SMALL.get(key, {}))
    x = torch.randn(3, 12, D)
    y = m.forecast(x, 7)
    assert y.shape == (3, 7, D) and torch.isfinite(y).all()
    assert m.predict_next(x).shape == (3, D)
    assert m.predict_next_sequence(x).shape == (3, 12, D)
    assert m.num_parameters() >= 0


@pytest.mark.parametrize("key", KEYS)
def test_direct_mode(key):
    m = build_baseline(key, D, mode="direct", direct_horizon=4, **SMALL.get(key, {}))
    x = torch.randn(2, 10, D)
    assert m.predict_direct(x).shape == (2, 4, D)
    y = m.forecast(x, 10)  # 3 blocks of 4, truncated
    assert y.shape == (2, 10, D) and torch.isfinite(y).all()


def test_persistence_exact():
    m = build_baseline("persistence", D)
    x = torch.randn(4, 9, D)
    y = m.forecast(x, 6)
    assert torch.equal(y, x[:, -1:].expand(-1, 6, -1))
    assert m.num_parameters() == 0


def test_mean_baseline_fit():
    m = build_baseline("mean", D)
    x = torch.randn(4, 9, D)
    m.fit(x)
    assert torch.allclose(m.forecast(x, 3)[:, 0], x.reshape(-1, D).mean(0).expand(4, -1))


@pytest.mark.parametrize("key", ["gru", "lstm"])
def test_rnn_carried_state_matches_naive(key):
    m = build_baseline(key, D, hidden=16, n_layers=2).eval()
    x = torch.randn(3, 10, D)
    with torch.no_grad():
        fast = m.forecast(x, 6)
        ctx = x
        naive = []
        for _ in range(6):
            nxt = m.predict_next(ctx)
            naive.append(nxt)
            ctx = torch.cat([ctx, nxt[:, None]], 1)
        naive = torch.stack(naive, 1)
    assert torch.allclose(fast, naive, atol=1e-5)


def test_transformer_causal():
    m = build_baseline("transformer", D, d_model=16, n_heads=2, n_layers=2, max_len=32).eval()
    x = torch.randn(2, 12, D)
    with torch.no_grad():
        a = m.predict_next_sequence(x)
        x2 = x.clone()
        x2[:, 8:] += 5.0
        b = m.predict_next_sequence(x2)
    assert torch.allclose(a[:, :8], b[:, :8], atol=1e-5)
    assert not torch.allclose(a[:, 8:], b[:, 8:])


@pytest.mark.parametrize("key", ["tcn", "ssm"])
def test_conv_ssm_causal(key):
    m = build_baseline(key, D, **SMALL[key]).eval()
    x = torch.randn(2, 12, D)
    with torch.no_grad():
        a = m.predict_next_sequence(x)
        x2 = x.clone()
        x2[:, 8:] += 5.0
        b = m.predict_next_sequence(x2)
    assert torch.allclose(a[:, :8], b[:, :8], atol=1e-5)


def _tiny_loaders():
    ds = build_dataset(TINY_DS)
    splits = ds.split(seed=0)
    train_n, stats = splits["train"].normalize()
    val_n, _ = splits["val"].normalize(stats)
    return make_loaders({"train": train_n, "val": val_n}, context=8, horizon=8, batch_size=16, stride=4)


def test_trainer_reduces_loss():
    loaders = _tiny_loaders()
    m = build_baseline("gru", 2, hidden=16)
    cfg = BaselineTrainerConfig(epochs=6, lr=5e-3, context=8, rollout_horizon=3, log_every=100)
    fit = BaselineTrainer(m, cfg, device=torch.device("cpu")).fit(loaders["train"], loaders["val"])
    hist = fit["history"]
    assert len(hist) == 6
    assert hist[-1]["train/total"] < hist[0]["train/total"]
    assert "train/rollout" in hist[-1] and hist[-1]["horizon"] == 3


def test_trainer_direct_mode():
    loaders = _tiny_loaders()
    m = build_baseline("gru", 2, hidden=16, mode="direct", direct_horizon=8)
    cfg = BaselineTrainerConfig(epochs=2, context=8, rollout_weight=0.0)
    fit = BaselineTrainer(m, cfg, device=torch.device("cpu")).fit(loaders["train"])
    assert "train/direct" in fit["history"][-1] and "train/rollout" not in fit["history"][-1]


EVAL_KEYS = ["teacher_forced/mse", "teacher_forced/nrmse", "recursive/horizon", "recursive/context",
             "recursive/nrmse@1", "recursive/nrmse_step@1", "recursive/nrmse@5",
             "recursive/nrmse_step@5", "recursive/nrmse@10", "recursive/nrmse_step@10",
             "recursive/nrmse_mean", "recursive/divergence_time", "curves", "params/total",
             "latent_dim", "mode", "latency/step_latency_ms_mean", "latency/step_latency_ms_std",
             "latency/step_latency_ms_min", "latency/horizon",
             "latency/forecast20_latency_ms_mean", "latency/forecast20_latency_ms_std",
             "latency/forecast20_latency_ms_min"]


def test_evaluate_forecaster_keys():
    m = build_baseline("persistence", D)
    x = torch.randn(4, 30, D)
    out = evaluate_forecaster(m, x, context=10, horizons=(1, 5, 10, 50), device=torch.device("cpu"))
    assert set(out) == set(EVAL_KEYS)
    assert out["recursive/horizon"] == 20 and out["latent_dim"] is None
    assert "recursive/nrmse@50" not in out
    assert len(out["curves"]["recursive_nrmse"]) == 20
    md = build_baseline("gru", D, hidden=8, mode="direct", direct_horizon=5)
    out = evaluate_forecaster(md, x, context=10, horizons=(1, 5, 10), latency=False)
    assert "direct/nrmse@5" in out and "direct/nrmse@10" not in out and out["mode"] == "direct"


def test_evaluate_teacher_forced_chunking():
    m = build_baseline("transformer", D, d_model=8, n_heads=2, n_layers=1, max_len=16).eval()
    x = torch.randn(2, 40, D)
    out = evaluate_forecaster(m, x, context=8, horizons=(1, 5), latency=False)
    assert torch.isfinite(torch.tensor(out["teacher_forced/nrmse"]))


def test_latent_wrapper():
    from nssc.models.builder import build_latent_model

    lm = build_latent_model({"latent_dim": 3, "encoder": {"name": "mlp", "kwargs": {"hidden_dims": [8]}},
                             "dynamics": "linear"}, obs_dim=D)
    f = LatentModelForecaster(lm)
    x = torch.randn(2, 12, D)
    assert f.forecast(x, 4).shape == (2, 4, D)
    assert f.predict_next_sequence(x).shape == (2, 12, D)
    assert f.latent_dim == 3 and f.num_parameters() == lm.num_parameters()["total"]


def test_presets_load():
    for k in ("gru", "lstm", "tcn", "transformer", "ssm", "persistence"):
        for s in ("small", "medium", "large"):
            kw = load_preset(k, s)
            assert isinstance(kw, dict)
    assert load_preset("gru", "small")["hidden"] == 32


def test_run_baseline_experiment(tmp_path, tmp_registry):
    cfg = {"dataset": TINY_DS,
           "model": {"baseline": "gru", "kwargs": {"hidden": 8}, "mode": "recursive"},
           "training": {"epochs": 2, "rollout_horizon": 2, "log_every": 100},
           "windows": {"context": 8, "horizon": 8, "stride": 4, "batch_size": 16},
           "eval": {"context": 8, "horizons": [1, 5, 10], "latency": False},
           "seed": 0, "tags": ["unit"], "output_dir": str(tmp_path / "run")}
    res = run_baseline_experiment(cfg, registry=tmp_registry, device=torch.device("cpu"), log=None)
    assert res["status"] == "completed", res.get("traceback")
    assert res["model"] == "baseline:gru"
    for split in ("val", "test"):
        assert "recursive/nrmse@10" in res["metrics"][split]
    assert (tmp_path / "run" / "metrics.json").exists()
    assert (tmp_path / "run" / "checkpoint" / "model.pt").exists()
    recs = tmp_registry.records()
    assert len(recs) == 1 and recs[0]["status"] == "completed"
    assert recs[0]["model"] == "baseline:gru" and recs[0]["param_count"] > 0
    assert "test/recursive/nrmse@10" in recs[0]["metrics"] and "test/params/total" in recs[0]["metrics"]
    for k in ("experiment_id", "git_commit", "config_hash", "dataset", "seed", "train_time_s",
              "hardware", "checkpoint"):
        assert k in recs[0]
    model, meta = load_forecaster_checkpoint(res["checkpoint"])
    assert model.num_parameters() == recs[0]["param_count"] and meta["seed"] == 0


def test_run_baseline_experiment_failure_recorded(tmp_path, tmp_registry):
    cfg = {"dataset": TINY_DS, "model": {"baseline": "nope"},
           "windows": {"context": 8, "horizon": 8}, "output_dir": str(tmp_path / "bad")}
    res = run_baseline_experiment(cfg, registry=tmp_registry, log=None)
    assert res["status"] == "failed"
    assert tmp_registry.records()[0]["status"] == "failed"


def test_suite_lookup_hash_matches_registered_hash(tmp_path):
    """Regression: the suite runner must find completed baseline runs on resume."""
    import torch

    from nssc.baselines.run import baseline_config_hash, run_baseline_experiment
    from nssc.utils.experiment_registry import ExperimentRegistry

    cfg = {"dataset": {"system": "harmonic", "n_traj": 4, "n_steps": 40, "dt": 0.05, "seed": 0},
           "model": {"baseline": "gru", "size": "small"}, "windows": {"context": 8, "horizon": 8, "stride": 4,
                                                                      "batch_size": 8},
           "training": {"epochs": 1, "context": 8, "log_every": 100}, "eval": {"context": 8, "horizons": [1, 5],
                                                                              "latency": False},
           "seed": 0, "output_dir": str(tmp_path / "r")}
    reg = ExperimentRegistry(tmp_path / "reg.jsonl")
    res = run_baseline_experiment(dict(cfg), registry=reg, device=torch.device("cpu"), log=None)
    assert res["status"] == "completed"
    assert reg.get(res["experiment_id"])["config_hash"] == baseline_config_hash(cfg)
