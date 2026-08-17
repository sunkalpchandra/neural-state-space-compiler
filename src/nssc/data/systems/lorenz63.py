"""Lorenz-63 chaotic attractor, ``d = 3``."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from nssc.data.systems.base import DynamicalSystem
from nssc.utils.registry import SYSTEMS


@SYSTEMS.register("lorenz63")
class Lorenz63(DynamicalSystem):
    """``x' = s(y-x), y' = x(r-z) - y, z' = xy - bz``; sigma=10, rho=28, beta=8/3."""

    name: ClassVar[str] = "lorenz63"
    state_dim = 3
    default_params: ClassVar[dict[str, Any]] = {"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0}
    default_dt: ClassVar[float] = 0.01
    default_transient: ClassVar[int] = 1000

    def f(self, t: float, x: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        s, r, b = params["sigma"], params["rho"], params["beta"]
        dx = s * (x[:, 1] - x[:, 0])
        dy = x[:, 0] * (r - x[:, 2]) - x[:, 1]
        dz = x[:, 0] * x[:, 1] - b * x[:, 2]
        return np.stack([dx, dy, dz], axis=1)

    def sample_initial(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.normal(size=(n, 3)) + np.array([0.0, 0.0, 25.0])

    def fixed_points(self) -> np.ndarray:
        r, b = self.params["rho"], self.params["beta"]
        if r <= 1:
            return np.zeros((1, 3))
        c = np.sqrt(b * (r - 1))
        return np.array([[0.0, 0.0, 0.0], [c, c, r - 1], [-c, -c, r - 1]])
