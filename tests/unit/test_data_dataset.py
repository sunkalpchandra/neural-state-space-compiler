"""Unit tests for nssc.data.dataset and nssc.data.builder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from nssc.data.builder import build_dataset, resolve_config
from nssc.data.dataset import TrajectoryDataset, WindowDataset, make_loaders, n_windows

ROOT = Path(__file__).resolve().parents[2]
TINY = ROOT / "configs" / "datasets" / "tiny_smoke.yaml"


@pytest.fixture(scope="module")
def tiny() -> TrajectoryDataset:
    return build_dataset(TINY)


def test_build_tiny(tiny):
    assert tiny.x.shape == (8, 64, 2) and tiny.x.dtype == np.float32
    assert tiny.z_true.shape == (8, 64, 2) and tiny.t.shape == (64,)
    assert np.isfinite(tiny.x).all()
    md = tiny.metadata
    assert md["system"] == "harmonic" and md["dt"] == 0.05 and len(md["version"]) == 12
    assert md["observation"]["type"] == "identity"


def test_build_deterministic_and_version_tracks_config():
    cfg = {"system": "harmonic", "n_traj": 4, "n_steps": 32, "seed": 0}
    a, b = build_dataset(cfg), build_dataset(cfg)
    np.testing.assert_array_equal(a.x, b.x)
    assert a.metadata["version"] == b.metadata["version"]
    c = build_dataset({**cfg, "noise_std": 0.1})
    assert c.metadata["version"] != a.metadata["version"]
    assert not np.allclose(c.x, a.x)
    assert resolve_config(cfg)["dt"] == 0.05


def test_build_with_obs_noise_missing():
    ds = build_dataset({"system": "lorenz63", "n_traj": 3, "n_steps": 20, "transient": 10,
                        "observation": {"type": "mlp", "obs_dim": 16, "seed": 1},
                        "noise_std": 0.05, "missing_rate": 0.2})
    assert ds.x.shape == (3, 20, 16) and ds.z_true.shape == (3, 20, 3)
    assert ds.mask is not None and ds.mask.shape == ds.x.shape
    assert np.isnan(ds.x[~ds.mask]).all()
    assert 0.1 < 1 - ds.mask.mean() < 0.3


def test_split_trajectory_level(tiny):
    sp = tiny.split(seed=0)
    assert set(sp) == {"train", "val", "test"}
    assert sum(v.n_traj for v in sp.values()) == tiny.n_traj
    assert sp["train"].n_traj == 4  # tiny_smoke split fractions (0.5, 0.25, 0.25)
    idx = [set(v.metadata["traj_idx"]) for v in sp.values()]
    assert not (idx[0] & idx[1]) and not (idx[0] & idx[2]) and not (idx[1] & idx[2])
    for v in sp.values():
        assert v.n_steps == tiny.n_steps and v.z_true is not None
    tr = sp["train"]
    np.testing.assert_array_equal(tr.x, tiny.x[np.array(tr.metadata["traj_idx"])])


def test_normalize_stats_from_train_only(tiny):
    sp = tiny.split(seed=0)
    tr_n, stats = sp["train"].normalize()
    np.testing.assert_allclose(tr_n.x.mean(axis=(0, 1)), 0.0, atol=1e-5)
    np.testing.assert_allclose(tr_n.x.std(axis=(0, 1)), 1.0, atol=1e-4)
    val_n, stats2 = sp["val"].normalize(stats)
    assert stats2 is stats
    np.testing.assert_allclose(val_n.x, (sp["val"].x - stats["mean"]) / stats["std"], rtol=1e-6)
    assert not np.allclose(val_n.x.mean(axis=(0, 1)), 0.0, atol=1e-6)
    assert val_n.metadata["normalized"] is True and sp["val"].x.dtype == np.float32


def test_save_load_roundtrip(tiny, tmp_path):
    p = tiny.save(tmp_path / "ds.npz")
    ds = TrajectoryDataset.load(p)
    np.testing.assert_array_equal(ds.x, tiny.x)
    np.testing.assert_array_equal(ds.z_true, tiny.z_true)
    np.testing.assert_array_equal(ds.t, tiny.t)
    assert ds.metadata == tiny.metadata


def test_to_torch(tiny):
    d = tiny.to_torch("cpu")
    assert d["x"].shape == (8, 64, 2) and d["x"].dtype == torch.float32
    assert d["z_true"].shape == (8, 64, 2) and d["t"].shape == (64,)


@pytest.mark.parametrize("ctx,hor,stride", [(8, 4, 1), (10, 10, 5), (32, 32, 1), (30, 30, 7)])
def test_window_dataset_count_formula(tiny, ctx, hor, stride):
    w = WindowDataset(tiny, ctx, hor, stride)
    L = ctx + hor
    expected = tiny.n_traj * ((tiny.n_steps - L) // stride + 1)
    assert len(w) == expected == n_windows(tiny.n_traj, tiny.n_steps, L, stride)
    item = w[len(w) - 1]
    assert item["x"].shape == (L, 2) and item["z_true"].shape == (L, 2)
    assert int(item["start"]) + L <= tiny.n_steps
    torch.testing.assert_close(item["x"], torch.as_tensor(tiny.x[int(item["traj"]),
                                                                 int(item["start"]):int(item["start"]) + L]))


def test_window_too_long_gives_empty(tiny):
    assert len(WindowDataset(tiny, 60, 10)) == 0
    assert n_windows(8, 64, 70, 1) == 0


def test_make_loaders(tiny):
    sp = tiny.split(seed=0)
    loaders = make_loaders(sp, context=8, horizon=4, batch_size=16, stride=4)
    batch = next(iter(loaders["train"]))
    assert batch["x"].shape[1:] == (12, 2) and batch["x"].shape[0] <= 16
    total = sum(b["x"].shape[0] for b in loaders["val"])
    assert total == len(WindowDataset(sp["val"], 8, 4, 4))
