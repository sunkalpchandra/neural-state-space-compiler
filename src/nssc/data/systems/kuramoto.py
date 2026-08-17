"""Kuramoto phase oscillators, ``d = N`` (states are phases in radians).

Natural frequencies ``omega_i ~ N(0, 1) * omega_std`` are drawn once from a
generator seeded with ``params['freq_seed']`` so the system is deterministic
given its config. Observe with :func:`observe_sin_cos` (``D = 2N``) to avoid
``2*pi`` wrap discontinuities.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from nssc.data.systems.base import DynamicalSystem
from nssc.utils.registry import SYSTEMS
from nssc.utils.seeding import rng as make_rng


def observe_sin_cos(theta: np.ndarray) -> np.ndarray:
    """Map phases ``(..., N)`` to ``(..., 2N)`` = ``[cos theta, sin theta]``."""
    return np.concatenate([np.cos(theta), np.sin(theta)], axis=-1)


def order_parameter(theta: np.ndarray) -> np.ndarray:
    """Kuramoto order parameter ``r = |mean_j exp(i theta_j)|``; ``(..., N) -> (...)``."""
    return np.abs(np.mean(np.exp(1j * theta), axis=-1))


@SYSTEMS.register("kuramoto")
class Kuramoto(DynamicalSystem):
    """``theta_i' = omega_i + (K/N) sum_j sin(theta_j - theta_i)``."""

    name: ClassVar[str] = "kuramoto"
    default_params: ClassVar[dict[str, Any]] = {
        "N": 8, "K": 2.0, "omega_std": 1.0, "freq_seed": 0,
    }
    default_dt: ClassVar[float] = 0.05

    def __init__(self, params: dict[str, Any] | None = None, dt: float | None = None) -> None:
        super().__init__(params, dt)
        self.state_dim = int(self.params["N"])
        self.omega = self.natural_frequencies(self.params)
        self._omega_key = (int(self.params["freq_seed"]), int(self.params["N"]),
                           float(self.params["omega_std"]))

    @staticmethod
    def natural_frequencies(params: dict[str, Any]) -> np.ndarray:
        """Deterministic ``(N,)`` natural frequencies from ``freq_seed``."""
        g = make_rng(int(params["freq_seed"]))
        return params["omega_std"] * g.normal(size=int(params["N"]))

    def f(self, t: float, x: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        key = (int(params["freq_seed"]), int(params["N"]), float(params["omega_std"]))
        if key != self._omega_key:
            self._omega_key, self.omega = key, self.natural_frequencies(params)
        omega = self.omega
        K, n = params["K"], int(params["N"])
        diff = x[:, None, :] - x[:, :, None]  # (N_batch, i, j) = theta_j - theta_i
        return omega[None, :] + (K / n) * np.sin(diff).sum(axis=2)

    def sample_initial(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.uniform(0.0, 2.0 * np.pi, size=(n, self.state_dim))

    def energy(self, x: np.ndarray) -> np.ndarray:
        return order_parameter(x)
