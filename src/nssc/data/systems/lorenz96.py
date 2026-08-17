"""Lorenz-96 chaotic lattice, ``d = N`` (cyclic)."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from nssc.data.systems.base import DynamicalSystem
from nssc.utils.registry import SYSTEMS


@SYSTEMS.register("lorenz96")
class Lorenz96(DynamicalSystem):
    """``x_i' = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F`` on ``N`` cyclic sites."""

    name: ClassVar[str] = "lorenz96"
    default_params: ClassVar[dict[str, Any]] = {"F": 8.0, "N": 8}
    default_dt: ClassVar[float] = 0.01
    default_transient: ClassVar[int] = 1000

    def __init__(self, params: dict[str, Any] | None = None, dt: float | None = None) -> None:
        super().__init__(params, dt)
        self.state_dim = int(self.params["N"])

    def f(self, t: float, x: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        F = params["F"]
        xp1 = np.roll(x, -1, axis=1)
        xm1 = np.roll(x, 1, axis=1)
        xm2 = np.roll(x, 2, axis=1)
        return (xp1 - xm2) * xm1 - x + F

    def sample_initial(self, rng: np.random.Generator, n: int) -> np.ndarray:
        x0 = np.full((n, self.state_dim), float(self.params["F"]))
        x0 += 0.01 * rng.normal(size=(n, self.state_dim))
        return x0

    def fixed_points(self) -> np.ndarray:
        return np.full((1, self.state_dim), float(self.params["F"]))
