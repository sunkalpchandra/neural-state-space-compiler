"""Harmonic and damped harmonic oscillators (state ``(x, v)``, ``d = 2``)."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from nssc.data.systems.base import DynamicalSystem
from nssc.utils.registry import SYSTEMS


@SYSTEMS.register("harmonic")
class HarmonicOscillator(DynamicalSystem):
    """``x' = v, v' = -omega^2 x``. Energy ``E = v^2/2 + omega^2 x^2/2`` conserved."""

    name: ClassVar[str] = "harmonic"
    state_dim = 2
    default_params: ClassVar[dict[str, Any]] = {"omega": 1.0}
    default_dt: ClassVar[float] = 0.05

    def f(self, t: float, x: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        w2 = params["omega"] ** 2
        return np.stack([x[:, 1], -w2 * x[:, 0]], axis=1)

    def sample_initial(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.uniform(-2.0, 2.0, size=(n, 2))

    def energy(self, x: np.ndarray) -> np.ndarray:
        w2 = self.params["omega"] ** 2
        return 0.5 * x[..., 1] ** 2 + 0.5 * w2 * x[..., 0] ** 2

    def fixed_points(self) -> np.ndarray:
        return np.zeros((1, 2))


@SYSTEMS.register("damped_oscillator")
class DampedOscillator(DynamicalSystem):
    """``x' = v, v' = -omega^2 x - 2 zeta omega v`` (``zeta > 0`` decays)."""

    name: ClassVar[str] = "damped_oscillator"
    state_dim = 2
    default_params: ClassVar[dict[str, Any]] = {"omega": 1.0, "zeta": 0.1}
    default_dt: ClassVar[float] = 0.05

    def f(self, t: float, x: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        w, z = params["omega"], params["zeta"]
        return np.stack([x[:, 1], -(w**2) * x[:, 0] - 2.0 * z * w * x[:, 1]], axis=1)

    def sample_initial(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.uniform(-2.0, 2.0, size=(n, 2))

    def energy(self, x: np.ndarray) -> np.ndarray:
        w2 = self.params["omega"] ** 2
        return 0.5 * x[..., 1] ** 2 + 0.5 * w2 * x[..., 0] ** 2

    def fixed_points(self) -> np.ndarray:
        return np.zeros((1, 2))
