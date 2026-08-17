from pathlib import Path

import torch

from nssc.evaluation.evaluator import EvalConfig
from nssc.evaluation.ood import evaluate_ood
from nssc.experiment import run_experiment
from nssc.utils.config import load_config
from nssc.utils.experiment_registry import ExperimentRegistry

ROOT = Path(__file__).resolve().parents[2]


def test_ood_param_and_ic_shift(tmp_path):
    cfg = load_config(ROOT / "configs/experiments/smoke.yaml")
    cfg["dataset"]["_file"] = str(ROOT / cfg["dataset"]["_file"])
    cfg["output_dir"] = str(tmp_path / "run")
    res = run_experiment(cfg, registry=ExperimentRegistry(tmp_path / "r.jsonl"),
                         device=torch.device("cpu"), log=None)
    out = evaluate_ood(res["checkpoint"], param_shifts={"omega": [1.5]}, ic_scales=[2.0], n_traj=4,
                       eval_cfg=EvalConfig(context=8, horizons=(1, 5, 10), latency=False, stability=False))
    names = [c["condition"] for c in out["conditions"]]
    assert names == ["param:omega=1.5", "ic_scale=2.0"]
    assert all(c["status"] == "ok" for c in out["conditions"])
    assert out["ood/degradation_ratio_mean"] > 0
