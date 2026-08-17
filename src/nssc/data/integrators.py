"""Vectorised fixed-step ODE integrators (pure numpy).

All integrators take a vector field ``f(t, x) -> dx`` operating on a batch of
states ``x`` of shape ``(N, d)`` and return the sampled trajectory of shape
``(N, T, d)`` where ``T = n_steps`` (the initial condition is included as the
first sample). Integration happens in float64; callers cast as needed.

``substeps`` decouples the sampling interval ``dt`` from the integrator step
``dt / substeps`` for stiff or fast systems (e.g. Lorenz-63 at coarse ``dt``).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

VectorField = Callable[[float, np.ndarray], np.ndarray]


def rk4_step(f: VectorField, t: float, x: np.ndarray, h: float) -> np.ndarray:
    """One classical Runge-Kutta 4 step of size ``h``. ``x``: ``(N, d)``."""
    k1 = f(t, x)
    k2 = f(t + 0.5 * h, x + 0.5 * h * k1)
    k3 = f(t + 0.5 * h, x + 0.5 * h * k2)
    k4 = f(t + h, x + h * k3)
    return x + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def euler_step(f: VectorField, t: float, x: np.ndarray, h: float) -> np.ndarray:
    """One forward-Euler step of size ``h``. ``x``: ``(N, d)``."""
    return x + h * f(t, x)


_STEPPERS = {"rk4": rk4_step, "euler": euler_step}


def integrate(
    f: VectorField,
    x0: np.ndarray,
    dt: float,
    n_steps: int,
    substeps: int = 1,
    method: str = "rk4",
    t0: float = 0.0,
) -> np.ndarray:
    """Integrate ``dx/dt = f(t, x)`` from a batch of initial conditions.

    Parameters
    ----------
    f : callable ``(t, x) -> dx`` with ``x`` and ``dx`` of shape ``(N, d)``.
    x0 : ``(N, d)`` initial states (any float dtype; cast to float64).
    dt : sampling interval between returned samples.
    n_steps : number of returned samples ``T`` (first sample is ``x0``).
    substeps : integrator steps per sampling interval (``h = dt / substeps``).
    method : ``"rk4"`` or ``"euler"``.

    Returns
    -------
    ``(N, T, d)`` float64 array of sampled states.
    """
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")
    if substeps < 1:
        raise ValueError("substeps must be >= 1")
    step = _STEPPERS[method]
    x = np.asarray(x0, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"x0 must be (N, d), got shape {x.shape}")
    h = dt / substeps
    out = np.empty((x.shape[0], n_steps, x.shape[1]), dtype=np.float64)
    out[:, 0] = x
    t = t0
    for i in range(1, n_steps):
        for _ in range(substeps):
            x = step(f, t, x, h)
            t += h
        out[:, i] = x
    return out


def rk4(f: VectorField, x0: np.ndarray, dt: float, n_steps: int, substeps: int = 1,
        t0: float = 0.0) -> np.ndarray:
    """RK4 convenience wrapper: see :func:`integrate`. Returns ``(N, T, d)``."""
    return integrate(f, x0, dt, n_steps, substeps=substeps, method="rk4", t0=t0)


def euler(f: VectorField, x0: np.ndarray, dt: float, n_steps: int, substeps: int = 1,
          t0: float = 0.0) -> np.ndarray:
    """Forward-Euler convenience wrapper: see :func:`integrate`. Returns ``(N, T, d)``."""
    return integrate(f, x0, dt, n_steps, substeps=substeps, method="euler", t0=t0)
