"""Sequence-model baselines (comparators for the central hypothesis) and trivial baselines.

Importing this package populates ``nssc.utils.registry.BASELINES`` with keys
``persistence, mean, gru, lstm, tcn, transformer, ssm``.
"""

from __future__ import annotations

from typing import Any

from nssc.baselines import persistence, rnn, ssm, tcn, transformer  # noqa: F401  (register)
from nssc.baselines.base import SequenceForecaster
from nssc.baselines.evaluate import evaluate_forecaster
from nssc.baselines.latent_wrapper import LatentModelForecaster
from nssc.baselines.persistence import MeanBaseline, PersistenceBaseline
from nssc.baselines.rnn import GRUForecaster, LSTMForecaster
from nssc.baselines.ssm import SSMForecaster
from nssc.baselines.tcn import TCNForecaster
from nssc.baselines.trainer import BaselineTrainer, BaselineTrainerConfig
from nssc.baselines.transformer import TransformerForecaster
from nssc.utils.registry import BASELINES

__all__ = [
    "BASELINES", "SequenceForecaster", "PersistenceBaseline", "MeanBaseline", "GRUForecaster",
    "LSTMForecaster", "TCNForecaster", "TransformerForecaster", "SSMForecaster",
    "LatentModelForecaster", "BaselineTrainer", "BaselineTrainerConfig", "evaluate_forecaster",
    "build_baseline", "run_baseline_experiment", "load_preset",
]


def build_baseline(key: str, obs_dim: int, **kw: Any) -> SequenceForecaster:
    """``BASELINES.build(key, obs_dim=obs_dim, **kw)`` (kw includes ``mode``/``direct_horizon``)."""
    return BASELINES.build(key, obs_dim=obs_dim, **kw)


def __getattr__(name: str) -> Any:  # lazy: run.py imports nssc.experiment (heavier)
    if name in ("run_baseline_experiment", "load_preset"):
        from nssc.baselines import run as _run

        return getattr(_run, name)
    raise AttributeError(name)
