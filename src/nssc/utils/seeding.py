"""Deterministic seeding across python, numpy and torch."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed python, numpy, torch (cpu + all accelerators).

    ``deterministic=True`` also asks torch to use deterministic kernels where
    available; some ops may fall back or warn on MPS.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def rng(seed: int) -> np.random.Generator:
    """Isolated numpy generator (preferred over global state inside generators)."""
    return np.random.default_rng(seed)
