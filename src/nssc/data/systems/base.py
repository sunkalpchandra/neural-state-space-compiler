"""Base class for synthetic dynamical systems.

Every system subclasses :class:`DynamicalSystem`, declares its default
parameters, and registers itself in ``nssc.utils.registry.SYSTEMS`` so
configs can reference systems by name. Simulation is vectorised over a batch of
trajectories through :mod:`nssc.data.integrators`.
"""

from __future__ import annotations

import copy
from typing import Any, ClassVar

import numpy as np

from nssc.data.integrators import integrate
from nssc.utils.seeding import rng as make_rng


class DynamicalSystem:
    """Abstract ODE system ``dx/dt = f(t, x; params)``.

    Subclasses must set the class attributes ``name``, ``state_dim``,
    ``default_params``, ``default_dt`` and implement :meth:`f` and
    :meth:`sample_initial`. ``default_transient`` (burn-in steps) and
    ``integrator`` (``"rk4"``/``"euler"``) may be overridden.
    """

    name: ClassVar[str] = "base"
    state_dim: int = 0
    default_params: ClassVar[dict[str, Any]] = {}
    default_dt: ClassVar[float] = 0.01
    default_transient: ClassVar[int] = 0
    default_substeps: ClassVar[int] = 1
    integrator: ClassVar[str] = "rk4"

    def __init__(self, params: dict[str, Any] | None = None, dt: float | None = None) -> None:
        self.params: dict[str, Any] = copy.deepcopy(dict(self.default_params))
        if params:
            unknown = set(params) - set(self.params)
            if unknown:
                raise KeyError(f"{self.name}: unknown params {sorted(unknown)}")
            self.params.update(params)
        self.dt: float = float(self.default_dt if dt is None else dt)

    # ------------------------------------------------------------------ dynamics
    def f(self, t: float, x: np.ndarray, params: dict[str, Any]) -> np.ndarray:
        """Vector field. ``x``: ``(N, d)`` -> ``(N, d)``."""
        raise NotImplementedError

    def sample_initial(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Draw ``n`` initial states -> ``(n, d)``."""
        raise NotImplementedError

    def energy(self, x: np.ndarray) -> np.ndarray | None:
        """Optional conserved/characteristic quantity. ``x``: ``(..., d)`` -> ``(...)``."""
        return None

    def fixed_points(self) -> np.ndarray | None:
        """Optional known fixed points ``(k, d)``."""
        return None

    # ---------------------------------------------------------------- simulate
    def simulate(
        self,
        n_traj: int,
        n_steps: int,
        dt: float | None = None,
        seed: int = 0,
        transient: int | None = None,
        params: dict[str, Any] | None = None,
        substeps: int | None = None,
    ) -> np.ndarray:
        """Simulate ``n_traj`` trajectories of ``n_steps`` samples.

        A ``transient`` (burn-in) of that many steps is integrated and discarded
        before recording. Returns ``(N, T, d)`` float64.
        """
        dt = self.dt if dt is None else float(dt)
        transient = self.default_transient if transient is None else int(transient)
        substeps = self.default_substeps if substeps is None else int(substeps)
        p = copy.deepcopy(self.params)
        if params:
            unknown = set(params) - set(p)
            if unknown:
                raise KeyError(f"{self.name}: unknown params {sorted(unknown)}")
            p.update(params)
        g = make_rng(seed)
        x0 = np.asarray(self.sample_initial(g, n_traj), dtype=np.float64)

        def field(t: float, x: np.ndarray) -> np.ndarray:
            return self.f(t, x, p)

        if transient > 0:
            burn = integrate(field, x0, dt, transient + 1, substeps=substeps,
                             method=self.integrator)
            x0 = burn[:, -1]
        return integrate(field, x0, dt, n_steps, substeps=substeps, method=self.integrator)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(params={self.params}, dt={self.dt})"
