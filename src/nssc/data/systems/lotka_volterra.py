"""Lotka-Volterra predator-prey, ``d = 2`` (positive states)."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from nssc.data.systems.base import DynamicalSystem
from nssc.utils.registry import SYSTEMS


@SYSTEMS.register("lotka_volterra")
class LotkaVolterra(DynamicalSystem):
    """``x' = a x - b x y, y' = d x y - g y``. Conserved ``V = d x - g ln x + b y - a ln y``."""

    name: ClassVar[str] = "lotka_volterra"
    state_dim = 2
    default_params: ClassVar[dict[str, Any]] = {
        "alpha": 1.5, "beta": 1.0, "delta": 1.0, "gamma": 3.0,
    }
    default_dt: ClassVar[float] = 0.01

    def f(self, t: float, x: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        a, b, d, g = params["alpha"], params["beta"], params["delta"], params["gamma"]
        u, v = x[:, 0], x[:, 1]
        return np.stack([a * u - b * u * v, d * u * v - g * v], axis=1)

    def sample_initial(self, rng: np.random.Generator, n: int) -> np.ndarray:
        # around the interior fixed point (gamma/delta, alpha/beta) = (3, 1.5)
        return rng.uniform(0.5, 4.0, size=(n, 2))

    def energy(self, x: np.ndarray) -> np.ndarray:
        p = self.params
        u, v = x[..., 0], x[..., 1]
        return p["delta"] * u - p["gamma"] * np.log(u) + p["beta"] * v - p["alpha"] * np.log(v)

    def fixed_points(self) -> np.ndarray:
        p = self.params
        return np.array([[0.0, 0.0], [p["gamma"] / p["delta"], p["alpha"] / p["beta"]]])
