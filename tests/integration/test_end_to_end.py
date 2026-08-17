"""dataset → train → evaluate → checkpoint → registry on the tiny smoke config."""

from pathlib import Path

import torch

from nssc.experiment import run_experiment
from nssc.training import load_checkpoint
from nssc.utils.config import load_config
from nssc.utils.experiment_registry import ExperimentRegistry

ROOT = Path(__file__).resolve().parents[2]


def _cfg(tmp_path, **over):
    cfg = load_config(ROOT / "configs/experiments/smoke.yaml")
    cfg["dataset"]["_file"] = str(ROOT / cfg["dataset"]["_file"])
    cfg["output_dir"] = str(tmp_path / "run")
    for k, v in over.items():
        cfg.set_path(k, v)
    return cfg


def test_smoke_pipeline_completes_and_registers(tmp_path):
    reg = ExperimentRegistry(tmp_path / "registry.jsonl")
    res = run_experiment(_cfg(tmp_path), registry=reg, device=torch.device("cpu"), log=None)
    assert res["status"] == "completed", res.get("traceback")
    s = res["summary"]
    for k in ("val/recon/nrmse", "test/recursive/nrmse@10", "test/params/total",
              "test/stability/rho_max"):
        assert k in s and s[k] == s[k]  # present and not NaN
    rec = reg.get(res["experiment_id"])
    assert rec["status"] == "completed" and rec["param_count"] > 0 and rec["checkpoint"]
    # checkpoint round trip reproduces predictions
    model, meta = load_checkpoint(rec["checkpoint"])
    assert meta["experiment_id"] == res["experiment_id"]
    x = torch.randn(2, 12, model.obs_dim)
    with torch.no_grad():
        a = model.rollout(x[:, :8], 4)[0]
        b = model.rollout(x[:, :8], 4)[0]
    assert torch.allclose(a, b)
    assert (Path(res["output_dir"]) / "metrics.json").exists()


def test_failed_run_is_recorded_not_raised(tmp_path):
    reg = ExperimentRegistry(tmp_path / "registry.jsonl")
    cfg = _cfg(tmp_path, **{"model.dynamics": {"name": "does_not_exist"}})
    res = run_experiment(cfg, registry=reg, device=torch.device("cpu"), log=None)
    assert res["status"] == "failed"
    assert reg.get(res["experiment_id"])["status"] == "failed"
    assert (Path(res["output_dir"]) / "error.json").exists()


def test_pca_linear_closed_form_pipeline(tmp_path):
    cfg = _cfg(tmp_path, **{"model.encoder": "pca", "model.decoder": "pca",
                            "model.dynamics": "linear", "training.epochs": 1})
    res = run_experiment(cfg, registry=ExperimentRegistry(tmp_path / "r.jsonl"),
                         device=torch.device("cpu"), log=None)
    assert res["status"] == "completed", res.get("traceback")
    # harmonic oscillator is exactly linear in 2-D → PCA(2)+DMD should be very accurate
    assert res["summary"]["test/recursive/nrmse@10"] < 0.2
