"""Nonlinear (optionally damped) pendulum, state ``(theta, omega)``, ``d = 2``."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from nssc.data.systems.base import DynamicalSystem
from nssc.utils.registry import SYSTEMS


@SYSTEMS.register("pendulum")
class Pendulum(DynamicalSystem):
    """``theta' = omega, omega' = -(g/L) sin(theta) - gamma omega``."""

    name: ClassVar[str] = "pendulum"
    state_dim = 2
    default_params: ClassVar[dict[str, Any]] = {"g_over_l": 1.0, "gamma": 0.0}
    default_dt: ClassVar[float] = 0.05

    def f(self, t: float, x: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        gl, gam = params["g_over_l"], params["gamma"]
        return np.stack([x[:, 1], -gl * np.sin(x[:, 0]) - gam * x[:, 1]], axis=1)

    def sample_initial(self, rng: np.random.Generator, n: int) -> np.ndarray:
        th = rng.uniform(-np.pi, np.pi, size=n)
        om = rng.uniform(-2.0, 2.0, size=n)
        return np.stack([th, om], axis=1)

    def energy(self, x: np.ndarray) -> np.ndarray:
        """``E = omega^2/2 + (g/L)(1 - cos theta)``; conserved when ``gamma = 0``."""
        gl = self.params["g_over_l"]
        return 0.5 * x[..., 1] ** 2 + gl * (1.0 - np.cos(x[..., 0]))

    def fixed_points(self) -> np.ndarray:
        return np.array([[0.0, 0.0], [np.pi, 0.0]])
