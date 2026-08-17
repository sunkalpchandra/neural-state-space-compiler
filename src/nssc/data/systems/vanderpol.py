"""Van der Pol relaxation oscillator, ``d = 2``."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from nssc.data.systems.base import DynamicalSystem
from nssc.utils.registry import SYSTEMS


@SYSTEMS.register("vanderpol")
class VanDerPol(DynamicalSystem):
    """``x' = v, v' = mu (1 - x^2) v - x``; limit cycle for ``mu > 0``."""

    name: ClassVar[str] = "vanderpol"
    state_dim = 2
    default_params: ClassVar[dict[str, Any]] = {"mu": 1.0}
    default_dt: ClassVar[float] = 0.05
    default_transient: ClassVar[int] = 0

    def f(self, t: float, x: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        mu = params["mu"]
        return np.stack([x[:, 1], mu * (1.0 - x[:, 0] ** 2) * x[:, 1] - x[:, 0]], axis=1)

    def sample_initial(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.uniform(-3.0, 3.0, size=(n, 2))

    def fixed_points(self) -> np.ndarray:
        return np.zeros((1, 2))
