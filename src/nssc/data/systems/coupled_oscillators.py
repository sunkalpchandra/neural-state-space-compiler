"""Chain of ``N`` linearly coupled harmonic oscillators, ``d = 2N``.

State layout: ``x = (q_1..q_N, p_1..p_N)``. Each mass has spring ``k`` to
ground and coupling ``kc`` to its chain neighbours (open boundary), plus
optional damping ``c``.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from nssc.data.systems.base import DynamicalSystem
from nssc.utils.registry import SYSTEMS


@SYSTEMS.register("coupled_oscillators")
class CoupledOscillators(DynamicalSystem):
    """``q' = p, p' = -K q - c p`` with tridiagonal stiffness ``K``."""

    name: ClassVar[str] = "coupled_oscillators"
    default_params: ClassVar[dict[str, Any]] = {"N": 4, "k": 1.0, "kc": 0.5, "c": 0.0}
    default_dt: ClassVar[float] = 0.05

    def __init__(self, params: dict[str, Any] | None = None, dt: float | None = None) -> None:
        super().__init__(params, dt)
        self.n_osc = int(self.params["N"])
        self.state_dim = 2 * self.n_osc

    def stiffness(self, params: dict[str, Any] | None = None) -> np.ndarray:
        """Tridiagonal stiffness matrix ``(N, N)``."""
        p = params or self.params
        n, k, kc = int(p["N"]), p["k"], p["kc"]
        K = np.zeros((n, n))
        idx = np.arange(n)
        K[idx, idx] = k
        if n > 1:
            K[idx[:-1], idx[1:]] = -kc
            K[idx[1:], idx[:-1]] = -kc
            K[idx[:-1], idx[:-1]] += kc
            K[idx[1:], idx[1:]] += kc
        return K

    def f(self, t: float, x: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        n = int(params["N"])
        q, p = x[:, :n], x[:, n:]
        K = self.stiffness(params)
        dp = -q @ K.T - params["c"] * p
        return np.concatenate([p, dp], axis=1)

    def sample_initial(self, rng: np.random.Generator, n: int) -> np.ndarray:
        return rng.uniform(-1.0, 1.0, size=(n, self.state_dim))

    def energy(self, x: np.ndarray) -> np.ndarray:
        n = self.n_osc
        q, p = x[..., :n], x[..., n:]
        K = self.stiffness()
        return 0.5 * np.sum(p**2, axis=-1) + 0.5 * np.einsum("...i,ij,...j->...", q, K, q)

    def fixed_points(self) -> np.ndarray:
        return np.zeros((1, self.state_dim))
