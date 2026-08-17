import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    np.random.seed(0)  # noqa: NPY002
    yield


@pytest.fixture
def tmp_registry(tmp_path):
    from nssc.utils.experiment_registry import ExperimentRegistry

    return ExperimentRegistry(tmp_path / "registry.jsonl")
