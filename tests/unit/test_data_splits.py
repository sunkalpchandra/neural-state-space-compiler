"""Unit tests for nssc.data.splits."""

from __future__ import annotations

import numpy as np
import pytest

from nssc.data.splits import check_no_leakage, param_range_split, trajectory_split


@pytest.mark.parametrize("n", [1, 2, 7, 10, 100, 101])
def test_disjoint_and_cover(n):
    s = trajectory_split(n, seed=0)
    allidx = np.concatenate([s["train"], s["val"], s["test"]])
    assert sorted(allidx.tolist()) == list(range(n))
    assert len(set(allidx.tolist())) == n
    assert len(s["train"]) >= 1


def test_fractions_and_determinism():
    s = trajectory_split(100, fractions=(0.7, 0.15, 0.15), seed=3)
    assert (len(s["train"]), len(s["val"]), len(s["test"])) == (70, 15, 15)
    s2 = trajectory_split(100, fractions=(0.7, 0.15, 0.15), seed=3)
    for k in s:
        np.testing.assert_array_equal(s[k], s2[k])
    s3 = trajectory_split(100, seed=4)
    assert not np.array_equal(s["train"], s3["train"])


def test_bad_fractions():
    with pytest.raises(ValueError):
        trajectory_split(10, fractions=(0.5, 0.5, 0.5))


def test_check_no_leakage():
    check_no_leakage([0, 1], [2], [3])
    with pytest.raises(ValueError):
        check_no_leakage([0, 1], [1], [3])
    with pytest.raises(ValueError):
        check_no_leakage([0], [2], [2])


def test_param_range_split():
    r = param_range_split((0.5, 2.0), (2.5, 4.0), n_train=20, n_test=5, seed=0, name="mu")
    assert r["train"].shape == (20,) and r["test"].shape == (5,)
    assert (r["train"] >= 0.5).all() and (r["train"] <= 2.0).all()
    assert (r["test"] >= 2.5).all() and (r["test"] <= 4.0).all()
    with pytest.raises(ValueError):
        param_range_split((0.5, 2.0), (1.5, 4.0), 1, 1)
