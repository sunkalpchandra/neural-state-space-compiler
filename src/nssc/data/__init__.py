"""Data subsystem: synthetic systems, observation maps, splits, datasets, builder."""

from nssc.data import systems  # noqa: F401  (registers SYSTEMS)
from nssc.data.builder import build_dataset, resolve_config
from nssc.data.dataset import TrajectoryDataset, WindowDataset, make_loaders, n_windows
from nssc.data.integrators import euler, integrate, rk4
from nssc.data.observation import (
    IdentityObservation,
    LinearObservation,
    ObservationMap,
    ObservationPipeline,
    PolynomialObservation,
    RandomMLPObservation,
    RedundantObservation,
    add_noise,
    irregular_subsample,
    mask_missing,
)
from nssc.data.splits import check_no_leakage, param_range_split, trajectory_split
from nssc.data.systems.base import DynamicalSystem

__all__ = [
    "DynamicalSystem",
    "IdentityObservation",
    "LinearObservation",
    "ObservationMap",
    "ObservationPipeline",
    "PolynomialObservation",
    "RandomMLPObservation",
    "RedundantObservation",
    "TrajectoryDataset",
    "WindowDataset",
    "add_noise",
    "build_dataset",
    "check_no_leakage",
    "euler",
    "integrate",
    "irregular_subsample",
    "make_loaders",
    "mask_missing",
    "n_windows",
    "param_range_split",
    "resolve_config",
    "rk4",
    "trajectory_split",
]
