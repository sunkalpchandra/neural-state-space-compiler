"""Neural ODE latent dynamics with an in-house fixed-step RK4 integrator.

    dz/dt = f_θ(z, u),   z_{t+1} = z_t + ∫_0^dt f_θ dt   (RK4, n_substeps)

Backprop goes straight through the solver (no adjoint), which is exact and
cheap for the small latent dimensions this project targets.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import Tensor

from nssc.dynamics._mlp_util import make_mlp
from nssc.dynamics.base import Dynamics
from nssc.utils.registry import DYNAMICS


def rk4_step(f: Callable[[Tensor], Tensor], z: Tensor, h: float) -> Tensor:
    k1 = f(z)
    k2 = f(z + 0.5 * h * k1)
    k3 = f(z + 0.5 * h * k2)
    k4 = f(z + h * k3)
    return z + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def euler_step(f: Callable[[Tensor], Tensor], z: Tensor, h: float) -> Tensor:
    return z + h * f(z)


_SOLVERS = {"rk4": rk4_step, "euler": euler_step}


@DYNAMICS.register("neural_ode")
class NeuralODEDynamics(Dynamics):
    def __init__(
        self,
        latent_dim: int,
        control_dim: int = 0,
        hidden_dims: Sequence[int] = (64, 64),
        act: str = "tanh",
        dt: float = 1.0,
        n_substeps: int = 4,
        solver: str = "rk4",
        **kwargs,
    ) -> None:
        super().__init__(latent_dim, control_dim)
        if solver not in _SOLVERS:
            raise KeyError(f"unknown solver '{solver}'. Available: {sorted(_SOLVERS)}")
        self.dt = dt
        self.n_substeps = n_substeps
        self.solver = solver
        self.net = make_mlp(latent_dim + control_dim, latent_dim, hidden_dims, act, zero_init_last=True)

    def vector_field(self, z: Tensor, u: Tensor | None = None) -> Tensor:
        """f_θ(z, u): (B, d) → (B, d). Useful for phase-portrait plots."""
        if self.control_dim > 0:
            if u is None:
                u = z.new_zeros(z.shape[0], self.control_dim)
            z = torch.cat([z, u], dim=-1)
        return self.net(z)

    def integrate(self, z: Tensor, t_span: float, u: Tensor | None = None, n_substeps: int | None = None) -> Tensor:
        """Integrate from z over ``t_span`` latent time units (u held constant)."""
        n = self.n_substeps if n_substeps is None else n_substeps
        h = t_span / n
        f = lambda x: self.vector_field(x, u)  # noqa: E731
        stepper = _SOLVERS[self.solver]
        for _ in range(n):
            z = stepper(f, z, h)
        return z

    def step(self, z: Tensor, u: Tensor | None = None) -> Tensor:
        return self.integrate(z, self.dt, u)

    def trajectory(self, z0: Tensor, t_end: float, n_points: int, u: Tensor | None = None) -> Tensor:
        """Dense trajectory sampled at ``n_points`` uniform times in (0, t_end]: (B, n_points, d)."""
        h = t_end / n_points
        zs = []
        z = z0
        for _ in range(n_points):
            z = self.integrate(z, h, u, n_substeps=1)
            zs.append(z)
        return torch.stack(zs, dim=1)
