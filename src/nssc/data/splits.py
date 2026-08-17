"""Trajectory-level (never timestep-level) dataset splits and OOD parameter ranges."""

from __future__ import annotations

from typing import Any

import numpy as np

from nssc.utils.seeding import rng as make_rng


def trajectory_split(n_traj: int, fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
                     seed: int = 0) -> dict[str, np.ndarray]:
    """Random disjoint trajectory index sets ``{'train','val','test'}`` covering ``range(n_traj)``.

    Rounding never drops trajectories: the remainder goes to train. With very
    few trajectories val/test may be empty; the split is deterministic in ``seed``.
    """
    if n_traj < 1:
        raise ValueError("n_traj must be >= 1")
    if len(fractions) != 3 or any(f < 0 for f in fractions) or abs(sum(fractions) - 1) > 1e-6:
        raise ValueError(f"fractions must be 3 non-negative values summing to 1, got {fractions}")
    perm = make_rng(seed).permutation(n_traj)
    n_val = int(round(fractions[1] * n_traj))
    n_test = int(round(fractions[2] * n_traj))
    n_train = n_traj - n_val - n_test
    if n_train < 1:
        raise ValueError("split leaves no training trajectories")
    out = {
        "train": np.sort(perm[:n_train]),
        "val": np.sort(perm[n_train:n_train + n_val]),
        "test": np.sort(perm[n_train + n_val:]),
    }
    check_no_leakage(out["train"], out["val"], out["test"])
    return out


def check_no_leakage(train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray) -> None:
    """Raise ``ValueError`` if any trajectory index appears in more than one split."""
    a, b, c = (set(np.asarray(s).tolist()) for s in (train_idx, val_idx, test_idx))
    if a & b or a & c or b & c:
        raise ValueError("split leakage: overlapping trajectory indices between splits")


def param_range_split(train_range: tuple[float, float], test_range: tuple[float, float],
                      n_train: int, n_test: int, seed: int = 0, name: str = "param"
                      ) -> dict[str, Any]:
    """Sample per-trajectory scalar parameter values for OOD experiments.

    Train values are ``U(train_range)``, test values ``U(test_range)``; the ranges
    should be disjoint (raises if they overlap). Returns
    ``{'name', 'train': (n_train,), 'test': (n_test,), 'train_range', 'test_range'}``.
    Ranges must come from the dataset config, never from code defaults.
    """
    lo_a, hi_a = train_range
    lo_b, hi_b = test_range
    if max(lo_a, lo_b) < min(hi_a, hi_b):
        raise ValueError(f"OOD ranges overlap: train {train_range} vs test {test_range}")
    g = make_rng(seed)
    return {
        "name": name,
        "train": g.uniform(lo_a, hi_a, size=n_train),
        "test": g.uniform(lo_b, hi_b, size=n_test),
        "train_range": tuple(train_range),
        "test_range": tuple(test_range),
    }
