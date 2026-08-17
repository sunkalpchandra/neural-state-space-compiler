"""Dashboard backend: sources → load → trajectory / field / stability / counterfactual / static."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from nssc.experiment import run_experiment  # noqa: E402
from nssc.utils.config import load_config  # noqa: E402
from nssc.utils.experiment_registry import ExperimentRegistry  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def _cfg(tmp_path, **over):
    cfg = load_config(ROOT / "configs/experiments/smoke.yaml")
    cfg["dataset"]["_file"] = str(ROOT / cfg["dataset"]["_file"])
    cfg["output_dir"] = str(tmp_path / "run")
    for k, v in over.items():
        cfg.set_path(k, v)
    return cfg


def _all_finite(obj) -> bool:
    if isinstance(obj, dict):
        return all(_all_finite(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return all(_all_finite(v) for v in obj)
    if isinstance(obj, float):
        return math.isfinite(obj)
    return True


@pytest.fixture(scope="module")
def dash(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("dash")
    reg = ExperimentRegistry(tmp / "registry.jsonl")
    res = run_experiment(_cfg(tmp), registry=reg, device=torch.device("cpu"), log=None)
    assert res["status"] == "completed", res.get("traceback")

    from dashboard import app as dapp

    dapp.REGISTRY_PATHS[:] = [tmp / "registry.jsonl"]
    dapp.COMPILE_ROOT = tmp / "compile"          # empty → no compile dirs
    dapp._CACHE.clear()
    dapp._STATE["current"] = None
    client = TestClient(dapp.app)
    return {"client": client, "exp_id": res["experiment_id"], "app": dapp}


def test_sources_lists_completed_run(dash):
    r = dash["client"].get("/api/sources")
    assert r.status_code == 200
    j = r.json()
    ids = [e["id"] for e in j["experiments"]]
    assert dash["exp_id"] in ids
    e = next(e for e in j["experiments"] if e["id"] == dash["exp_id"])
    assert e["load_id"] == dash["exp_id"] and e["checkpoint"]
    assert "test/recon/nrmse" in e["metrics"]
    assert j["compiles"] == []


def test_load_and_summary(dash):
    c = dash["client"]
    r = c.post("/api/load", json={"experiment_id": dash["exp_id"]})
    assert r.status_code == 200, r.text
    s = r.json()
    assert s["obs_dim"] == 2 and s["latent_dim"] == 2 and s["n_traj"] >= 1 and s["T"] > 8
    assert s["params"]["total"] > 0 and s["model_name"] == "mlp+residual_mlp@d2"
    assert s["is_stochastic"] is False and s["has_z_true"] is True
    assert len(s["dims"]) == s["obs_dim"] and len(s["latent_labels"]) == s["latent_dim"]
    assert c.get("/api/summary").json()["experiment_id"] == dash["exp_id"]


def test_load_unknown_is_404(dash):
    r = dash["client"].post("/api/load", json={"experiment_id": "EXP-9999"})
    assert r.status_code == 404
    assert dash["client"].post("/api/load", json={}).status_code == 400


def test_trajectory_shapes(dash):
    c = dash["client"]
    c.post("/api/load", json={"experiment_id": dash["exp_id"]})
    r = c.get("/api/trajectory", params={"idx": 0, "context": 8, "horizon": 20})
    assert r.status_code == 200, r.text
    t = r.json()
    T, D, d = t["T"], len(t["dims"]), 2
    assert np.asarray(t["x"]).shape == (T, D)
    assert np.asarray(t["z"]).shape == (T, d)
    assert np.asarray(t["x_hat"]).shape == (20, D)
    assert np.asarray(t["z_hat"]).shape == (20, d)
    assert np.asarray(t["std"]).shape == (20, D) and t["std_method"]
    assert len(t["nrmse"]) == 20 and len(t["t"]) == T
    assert t["context"] == 8 and t["horizon"] == 20
    assert _all_finite(t)
    # horizon is clamped to what the trajectory can support
    r = c.get("/api/trajectory", params={"idx": 0, "context": 8, "horizon": 10_000})
    assert r.status_code == 200 and r.json()["horizon"] == T - 8
    assert c.get("/api/trajectory", params={"idx": 999}).status_code == 400


def test_field(dash):
    c = dash["client"]
    c.post("/api/load", json={"experiment_id": dash["exp_id"]})
    r = c.get("/api/field", params={"dims": "0,1", "grid": 11})
    assert r.status_code == 200, r.text
    f = r.json()
    assert f["grid"] == 11 and len(f["x"]) == 11 and len(f["y"]) == 11
    assert np.asarray(f["u"]).shape == (11, 11) and np.asarray(f["v"]).shape == (11, 11)
    assert len(f["traj"]["x"]) == len(f["traj"]["y"]) > 0
    assert _all_finite(f)
    assert c.get("/api/field", params={"dims": "0,7"}).status_code == 400
    assert c.get("/api/field", params={"dims": "bad"}).status_code == 400


def test_stability(dash):
    c = dash["client"]
    c.post("/api/load", json={"experiment_id": dash["exp_id"]})
    r = c.get("/api/stability", params={"idx": 0})
    assert r.status_code == 200, r.text
    s = r.json()
    M = s["n_points"]
    assert 1 <= M <= 128
    assert np.asarray(s["real"]).shape == (M, 2) and np.asarray(s["imag"]).shape == (M, 2)
    assert len(s["spectral_radius"]) == M and len(s["t_index"]) == M
    assert _all_finite(s["real"]) and _all_finite(s["imag"]) and _all_finite(s["spectral_radius"])
    assert s["metrics"]["verdict"] is not None and s["metrics"]["rho_max"] is not None


def test_counterfactual(dash):
    c = dash["client"]
    c.post("/api/load", json={"experiment_id": dash["exp_id"]})
    t = c.get("/api/trajectory", params={"idx": 0, "context": 8, "horizon": 12}).json()
    z0 = t["z"][7]
    r = c.post("/api/counterfactual", json={"idx": 0, "context": 8, "horizon": 12, "z0": z0})
    assert r.status_code == 200, r.text
    cf = r.json()
    assert np.asarray(cf["x_hat"]).shape == (12, len(cf["dims"]))
    assert np.asarray(cf["x_hat_original"]).shape == (12, len(cf["dims"]))
    assert np.asarray(cf["z_hat"]).shape == (12, 2)
    assert _all_finite(cf)
    # identical z0 → identical rollout, zero divergence
    assert cf["delta_z0"] == pytest.approx(0.0, abs=1e-6)
    assert max(cf["divergence"]) == pytest.approx(0.0, abs=1e-5)
    # a perturbed z0 changes the decoded trajectory
    z1 = [z0[0] + 0.5, z0[1] - 0.5]
    cf2 = c.post("/api/counterfactual", json={"idx": 0, "context": 8, "horizon": 12, "z0": z1}).json()
    assert cf2["delta_z0"] > 0 and max(cf2["divergence"]) > 0
    # wrong latent size
    r = c.post("/api/counterfactual", json={"idx": 0, "context": 8, "horizon": 12, "z0": [0.0]})
    assert r.status_code == 400


def test_compile_report_null_for_registry_run(dash):
    c = dash["client"]
    c.post("/api/load", json={"experiment_id": dash["exp_id"]})
    r = c.get("/api/compile_report")
    assert r.status_code == 200 and r.json() is None


def test_index_and_static(dash):
    c = dash["client"]
    r = c.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "<html" in r.text and "app.js" in r.text and "plotly" in r.text
    assert c.get("/static/app.js").status_code == 200
    assert c.get("/static/style.css").status_code == 200


def test_routes_before_load_are_409():
    from dashboard import app as dapp

    dapp._STATE["current"] = None
    c = TestClient(dapp.app)
    assert c.get("/api/trajectory").status_code == 409
    assert c.get("/api/compile_report").json() is None


def _write_compile_dir(root: Path, checkpoint: str, exp_metrics: dict) -> Path:
    """A minimal compile dir whose report points at an existing checkpoint."""
    from nssc.compiler.report import CompileReport

    d = root / "toy"
    d.mkdir(parents=True)
    agg = {k: v for k, v in exp_metrics.items() if k.startswith("val/")}
    agg["n_seeds"] = 1
    row = {"candidate_id": "mlp+residual_mlp@d2-abc123", "name": "mlp+residual_mlp@d2-abc123", "rank": 1,
           "score": 0.1, "agg": agg, "rollout_key": "val/recursive/nrmse@10",
           "terms": {"recon": 0.0, "one_step": 0.0, "rollout": 0.0, "complexity": 0.0,
                     "instability": 0.1, "blowup": 0.0}}
    rep = CompileReport(selected={"encoder": "mlp", "decoder": "mlp", "dynamics": "residual_mlp",
                                  "latent_dim": 2},
                        selected_metrics=agg, ranking=[row],
                        stage_summaries=[{"stage": "screen", "n_candidates": 1, "n_survivors": 1}],
                        profile={}, weights={"criterion": "weighted", "rollout": 1.0},
                        reasons=["only candidate"], n_runs=1, n_failed=0, wall_time_s=1.0,
                        dataset={"system": "harmonic"}, checkpoint=checkpoint,
                        rollout_key="val/recursive/nrmse@10")
    rep.save(d / "compile_report.json")
    return d


def test_compile_dir_source(dash, tmp_path):
    dapp, c = dash["app"], dash["client"]
    rec = next(e for e in c.get("/api/sources").json()["experiments"] if e["id"] == dash["exp_id"])
    ckpt_meta = dapp.load_json(Path(rec["checkpoint"]) / "metadata.json")
    d = _write_compile_dir(tmp_path / "compile", rec["checkpoint"], ckpt_meta["metrics_summary"])
    old = dapp.COMPILE_ROOT
    dapp.COMPILE_ROOT = tmp_path / "compile"
    try:
        src = c.get("/api/sources").json()
        assert len(src["compiles"]) == 1 and src["compiles"][0]["has_checkpoint"]
        r = c.post("/api/load", json={"compile_dir": str(d)})
        assert r.status_code == 200, r.text
        assert r.json()["has_compile_report"] is True and r.json()["model_name"] == "mlp+residual_mlp@d2"
        rep = c.get("/api/compile_report").json()
        assert rep["selected"]["latent_dim"] == 2 and rep["ranking"][0]["rank"] == 1
        assert c.get("/api/trajectory", params={"idx": 0, "context": 8, "horizon": 5}).status_code == 200
    finally:
        dapp.COMPILE_ROOT = old
        dapp._STATE["current"] = None
