"""Every YAML under configs/ must load; dataset configs must build a small dataset."""

from pathlib import Path

import pytest

from nssc.data.builder import build_dataset
from nssc.utils.config import load_config

ROOT = Path(__file__).resolve().parents[2]
ALL = sorted(p for p in (ROOT / "configs").rglob("*.yaml"))
DATASETS = sorted((ROOT / "configs/datasets").glob("*.yaml"))


@pytest.mark.parametrize("path", ALL, ids=lambda p: str(p.relative_to(ROOT)))
def test_yaml_loads(path):
    cfg = load_config(path)
    assert isinstance(cfg, dict) and len(cfg) > 0


@pytest.mark.parametrize("path", DATASETS, ids=lambda p: p.stem)
def test_synthetic_dataset_configs_build_small(path):
    cfg = load_config(path).to_dict()
    if "source" in cfg:  # real data: needs network
        pytest.skip("real-data source")
    cfg["n_traj"], cfg["n_steps"] = 2, 32
    cfg["transient"] = min(int(cfg.get("transient") or 0), 50)
    ds = build_dataset(cfg)
    assert ds.x.shape[0] == 2 and ds.x.shape[1] == 32
