"""Gray-Scott 1-D reaction-diffusion (method of lines), ``d = 2N``.

State layout ``x = (u_1..u_N, v_1..v_N)`` on a periodic grid of ``N`` points with
spacing ``h``. Integrated with forward Euler (default) using a 3-point Laplacian;
stability requires ``dt * max(Du, Dv) / h^2 < 0.5``.

Defaults ``(F, k) = (0.03, 0.055)`` give sustained, non-stationary pulsating
patterns in 1-D on 32 points (the 2-D "spots" pair (0.035, 0.065) decays to the
trivial state ``u=1, v=0`` in 1-D on this grid).
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from nssc.data.systems.base import DynamicalSystem
from nssc.utils.registry import SYSTEMS


@SYSTEMS.register("gray_scott")
class GrayScott1D(DynamicalSystem):
    """``u_t = Du u_xx - u v^2 + F(1-u)``, ``v_t = Dv v_xx + u v^2 - (F+k) v``."""

    name: ClassVar[str] = "gray_scott"
    default_params: ClassVar[dict[str, Any]] = {
        "N": 32, "Du": 0.16, "Dv": 0.08, "F": 0.03, "k": 0.055, "h": 1.0,
    }
    default_dt: ClassVar[float] = 1.0
    default_transient: ClassVar[int] = 500
    default_substeps: ClassVar[int] = 1
    integrator: ClassVar[str] = "euler"

    def __init__(self, params: dict[str, Any] | None = None, dt: float | None = None) -> None:
        super().__init__(params, dt)
        self.n_grid = int(self.params["N"])
        self.state_dim = 2 * self.n_grid
        dmax = max(self.params["Du"], self.params["Dv"])
        if self.dt * dmax / self.params["h"] ** 2 >= 0.5:
            raise ValueError("gray_scott: Euler unstable, need dt*D/h^2 < 0.5")

    @staticmethod
    def _laplacian(a: np.ndarray, h: float) -> np.ndarray:
        return (np.roll(a, -1, axis=1) - 2.0 * a + np.roll(a, 1, axis=1)) / h**2

    def f(self, t: float, x: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        n = int(params["N"])
        u, v = x[:, :n], x[:, n:]
        h, F, k = params["h"], params["F"], params["k"]
        uvv = u * v * v
        du = params["Du"] * self._laplacian(u, h) - uvv + F * (1.0 - u)
        dv = params["Dv"] * self._laplacian(v, h) + uvv - (F + k) * v
        return np.concatenate([du, dv], axis=1)

    def sample_initial(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Homogeneous ``u=1, v=0`` with a random localised perturbation blob."""
        N = self.n_grid
        u = np.ones((n, N))
        v = np.zeros((n, N))
        centre = rng.integers(0, N, size=n)
        width = max(2, N // 8)
        idx = np.arange(N)[None, :]
        dist = np.minimum(np.abs(idx - centre[:, None]), N - np.abs(idx - centre[:, None]))
        blob = (dist < width).astype(np.float64)
        u = u - 0.5 * blob + 0.02 * rng.normal(size=(n, N))
        v = v + 0.25 * blob + 0.02 * rng.normal(size=(n, N))
        return np.clip(np.concatenate([u, v], axis=1), 0.0, 1.0)

    def energy(self, x: np.ndarray) -> np.ndarray:
        """Total ``v`` mass (a characteristic, not conserved, quantity)."""
        return x[..., self.n_grid:].sum(axis=-1)
