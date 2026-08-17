"""Real-world data sources (Tier 3). Heavy deps (mne) are imported lazily inside builders.

Dispatch by ``cfg['source']`` via :func:`build_real_dataset`; sources register in
:data:`REAL_SOURCES` as ``name -> callable(cfg) -> TrajectoryDataset``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nssc.data.dataset import TrajectoryDataset

REAL_SOURCES: dict[str, Callable[[dict[str, Any]], TrajectoryDataset]] = {}


def _eegbci(cfg: dict[str, Any]) -> TrajectoryDataset:
    from nssc.data.real.eegbci import build_eegbci

    return build_eegbci(cfg)


def _motion(cfg: dict[str, Any]) -> TrajectoryDataset:
    from nssc.data.real.motion import build_motion

    return build_motion(cfg)


REAL_SOURCES["eegbci"] = _eegbci
REAL_SOURCES["motion_cmu_mocap"] = _motion


def is_real_source(cfg: dict[str, Any]) -> bool:
    """True when the dataset config selects a real-world source (``source`` key present)."""
    return isinstance(cfg, dict) and cfg.get("source") is not None


def build_real_dataset(cfg: dict[str, Any]) -> TrajectoryDataset:
    src = cfg.get("source")
    if src not in REAL_SOURCES:
        raise KeyError(f"unknown real data source {src!r}; available: {sorted(REAL_SOURCES)}")
    return REAL_SOURCES[src](cfg)


__all__ = ["REAL_SOURCES", "build_real_dataset", "is_real_source"]
