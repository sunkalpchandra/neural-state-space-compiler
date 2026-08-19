from pathlib import Path

import torch

from nssc.evaluation.reevaluate import EVAL_PROTOCOL, reevaluate_record, reevaluate_suite
from nssc.experiment import run_experiment
from nssc.utils.config import load_config
from nssc.utils.experiment_registry import ExperimentRegistry

ROOT = Path(__file__).resolve().parents[2]


def _run(tmp_path, reg):
    cfg = load_config(ROOT / "configs/experiments/smoke.yaml")
    cfg["dataset"]["_file"] = str(ROOT / cfg["dataset"]["_file"])
    cfg["output_dir"] = str(tmp_path / "run")
    cfg["tags"] = ["suite:t", "ds:x", "m:y"]
    return run_experiment(cfg, registry=reg, device=torch.device("cpu"), log=None)


def test_reevaluate_reproduces_metrics_and_appends_a_row(tmp_path):
    reg = ExperimentRegistry(tmp_path / "r.jsonl")
    res = _run(tmp_path, reg)
    before = reg.get(res["experiment_id"])["metrics"]["test/recursive/nrmse@10"]
    out = reevaluate_record(reg.get(res["experiment_id"]), reg, device=torch.device("cpu"))
    assert out["status"] == "ok"
    after = reg.get(res["experiment_id"])["metrics"]
    assert abs(after["test/recursive/nrmse@10"] - before) < 1e-9   # deterministic given the checkpoint
    assert after["eval/eval_protocol"] == EVAL_PROTOCOL
    assert len(reg.records()) == 1 and len(open(reg.path).read().splitlines()) == 3  # append-only


def test_reevaluate_suite_skips_missing_checkpoints(tmp_path):
    reg = ExperimentRegistry(tmp_path / "r.jsonl")
    res = _run(tmp_path, reg)
    (Path(res["checkpoint"]) / "model.pt").unlink()
    counts = reevaluate_suite(suite="t", registry_path=str(reg.path), device=torch.device("cpu"), log=None)
    assert counts["skipped"] == 1 and counts["ok"] == 0
