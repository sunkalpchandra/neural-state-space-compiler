"""Known benchmark values (from results/registry.jsonl, suite synthetic_core) with tolerances.

These recompute the cheap, deterministic rows so a silent change in data generation, splits,
normalisation or metrics is caught. Values were produced by the registered runs (5 seeds).
"""

import numpy as np
import pytest
import torch

from nssc.baselines import build_baseline
from nssc.baselines.evaluate import evaluate_forecaster
from nssc.experiment import prepare_data
from nssc.utils.config import load_yaml


@pytest.fixture(scope="module")
def lorenz_test():
    splits, stats, raw = prepare_data(load_yaml("configs/datasets/lorenz63.yaml"))
    return torch.from_numpy(splits["test"].x), raw.obs_dim


def test_persistence_lorenz63_nrmse50(lorenz_test):
    x, D = lorenz_test
    m = build_baseline("persistence", D)
    r = evaluate_forecaster(m, x, context=20, horizons=(1, 5, 10, 25, 50, 100, 250), sigma=np.ones(D),
                            device=torch.device("cpu"), latency=False)
    # registry (EXP-0065..0069 and duplicates EXP-0245..0249): 1.2950 ± 0.0000
    assert abs(r["recursive/nrmse@50"] - 1.2950) < 0.005, r["recursive/nrmse@50"]
    assert abs(r["recursive/nrmse@250"] - 1.532) < 0.01


def test_lorenz63_dataset_shape_and_split_sizes():
    splits, stats, raw = prepare_data(load_yaml("configs/datasets/lorenz63.yaml"))
    assert raw.x.shape == (100, 500, 3)
    assert {k: v.n_traj for k, v in splits.items()} == {"train": 70, "val": 15, "test": 15}
    assert np.allclose(stats["mean"], [-0.3, -0.3, 23.6], atol=1.5)  # attractor means
