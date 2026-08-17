"""Unit tests for real-data loaders (nssc.data.real). Network tests are opt-in.

Set ``NSSC_NETWORK_TESTS=1`` (and have ``mne`` installed) to run the EEGBCI download path.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from nssc.data.builder import build_dataset
from nssc.data.dataset import TrajectoryDataset
from nssc.data.real import REAL_SOURCES, is_real_source
from nssc.data.real.eegbci import (
    _segment,
    cache_path,
    resolve_eegbci_config,
    subject_split_indices,
    validate_subject_split,
)

ROOT = Path(__file__).resolve().parents[2]
TINY = ROOT / "configs" / "datasets" / "eegbci_tiny.yaml"


def _fake_eeg(n_per_subject: int = 4, subjects=(1, 2, 3), T: int = 16, C: int = 3
              ) -> TrajectoryDataset:
    subj = [s for s in subjects for _ in range(n_per_subject)]
    x = np.random.default_rng(0).standard_normal((len(subj), T, C)).astype(np.float32)
    split = {"by": "subject", "train_subjects": [1], "val_subjects": [2], "test_subjects": [3]}
    md = {"source": "fake", "subject_of_segment": subj, "run_of_segment": [3] * len(subj),
          "per_traj_keys": ["subject_of_segment", "run_of_segment"],
          "split_indices": subject_split_indices(subj, split), "config": {"split": split}}
    return TrajectoryDataset(x=x, t=np.arange(T, dtype=float), metadata=md)


def test_split_honours_split_indices_and_reindexes_per_traj_metadata():
    ds = _fake_eeg()
    parts = ds.split(seed=123, fractions=(0.2, 0.4, 0.4))  # args must be ignored
    assert {k: v.n_traj for k, v in parts.items()} == {"train": 4, "val": 4, "test": 4}
    assert set(parts["train"].metadata["subject_of_segment"]) == {1}
    assert set(parts["val"].metadata["subject_of_segment"]) == {2}
    assert set(parts["test"].metadata["subject_of_segment"]) == {3}
    np.testing.assert_array_equal(parts["val"].x, ds.x[4:8])
    assert parts["val"].metadata["traj_idx"] == [4, 5, 6, 7]
    assert "split_indices" not in parts["val"].metadata  # subsets are not re-splittable
    # random trajectory split still works when no fixed indices are given
    ds2 = TrajectoryDataset(x=ds.x, t=ds.t, metadata={})
    p2 = ds2.split(fractions=(0.5, 0.25, 0.25))
    assert sum(v.n_traj for v in p2.values()) == ds2.n_traj


def test_split_indices_leakage_detected():
    ds = _fake_eeg()
    ds.metadata["split_indices"] = {"train": [0, 1], "val": [1, 2], "test": [5]}
    with pytest.raises(ValueError, match="leakage"):
        ds.split()


def test_subject_split_validation():
    ok = {"train_subjects": [1, 2], "val_subjects": [3], "test_subjects": [4]}
    validate_subject_split([1, 2, 3, 4], ok)
    with pytest.raises(ValueError, match="only one of"):
        validate_subject_split([1, 2, 3], {"train_subjects": [1, 2], "val_subjects": [2],
                                           "test_subjects": [3]})
    with pytest.raises(ValueError, match="not in 'subjects'"):
        validate_subject_split([1, 2], ok)
    with pytest.raises(ValueError, match="non-empty"):
        validate_subject_split([1], {"train_subjects": [], "val_subjects": [1]})
    idx = subject_split_indices([1, 1, 2, 3, 3], ok)
    assert idx == {"train": [0, 1, 2], "val": [3, 4], "test": []}


def test_resolve_config_defaults_hash_and_cache_path(tmp_path):
    cfg = {"source": "eegbci", "subjects": [1, 2], "runs": [3], "channels": 8,
           "split": {"train_subjects": [1], "val_subjects": [2], "test_subjects": []},
           "cache_dir": str(tmp_path)}
    r = resolve_eegbci_config(cfg)
    assert r["segment_stride_seconds"] == r["segment_seconds"] == 8
    assert r["split"]["by"] == "subject" and r["resample_hz"] == 64
    p = cache_path(r)
    assert p.parent == tmp_path and p.name.startswith("eegbci_") and p.suffix == ".npz"
    r2 = resolve_eegbci_config({**cfg, "resample_hz": 32})
    assert cache_path(r2) != p  # any preprocessing change → new version
    with pytest.raises(ValueError, match="only split.by"):
        resolve_eegbci_config({**cfg, "split": {"by": "random"}})
    with pytest.raises(ValueError, match="1..109"):
        resolve_eegbci_config({**cfg, "subjects": [0]})
    with pytest.raises(ValueError):
        resolve_eegbci_config({**cfg, "resample_hz": 1000})


def test_segment_windows():
    d = np.arange(20, dtype=np.float32).reshape(10, 2)
    s = _segment(d, length=4, stride=4)
    assert s.shape == (2, 4, 2) and s[1, 0, 0] == 8
    assert _segment(d, length=4, stride=2).shape == (4, 4, 2)
    assert _segment(d, length=11, stride=1).shape == (0, 11, 2)


def test_dispatch_and_sources():
    assert is_real_source({"source": "eegbci"}) and not is_real_source({"system": "lorenz63"})
    assert "eegbci" in REAL_SOURCES and "motion_cmu_mocap" in REAL_SOURCES
    with pytest.raises(NotImplementedError):
        build_dataset({"source": "motion_cmu_mocap"})
    with pytest.raises(KeyError, match="unknown real data source"):
        build_dataset({"source": "nope"})


@pytest.mark.slow
def test_eegbci_tiny_network():
    """Real download + preprocessing; opt-in via NSSC_NETWORK_TESTS=1."""
    if not os.environ.get("NSSC_NETWORK_TESTS"):
        pytest.skip("set NSSC_NETWORK_TESTS=1 to run network tests")
    pytest.importorskip("mne")
    ds = build_dataset(TINY)
    assert ds.x.ndim == 3 and ds.x.shape[1:] == (128, 8) and ds.x.dtype == np.float32
    assert np.isfinite(ds.x).all() and ds.z_true is None
    assert ds.metadata["fs"] == 32.0 and len(ds.metadata["subject_of_segment"]) == ds.n_traj
    parts = ds.split()
    for name, subj in (("train", 1), ("val", 2), ("test", 3)):
        assert set(parts[name].metadata["subject_of_segment"]) == {subj}
    ds2 = build_dataset(TINY)  # cached
    assert ds2.metadata["cache_hit"] and ds2.metadata["version"] == ds.metadata["version"]
