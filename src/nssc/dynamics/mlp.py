"""MLP and residual-MLP latent dynamics."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from nssc.dynamics._mlp_util import make_mlp
from nssc.dynamics.base import Dynamics
from nssc.utils.registry import DYNAMICS


def _cat_control(z: Tensor, u: Tensor | None, control_dim: int) -> Tensor:
    if control_dim == 0:
        return z
    if u is None:
        u = z.new_zeros(z.shape[0], control_dim)
    return torch.cat([z, u], dim=-1)


@DYNAMICS.register("mlp")
class MLPDynamics(Dynamics):
    """z_{t+1} = f_θ([z_t, u_t]) with a plain MLP."""

    def __init__(
        self,
        latent_dim: int,
        control_dim: int = 0,
        hidden_dims: Sequence[int] = (128, 128),
        act: str = "gelu",
        **kwargs,
    ) -> None:
        super().__init__(latent_dim, control_dim)
        self.hidden_dims = tuple(hidden_dims)
        self.act = act
        self.net = make_mlp(latent_dim + control_dim, latent_dim, self.hidden_dims, act)

    def step(self, z: Tensor, u: Tensor | None = None) -> Tensor:
        return self.net(_cat_control(z, u, self.control_dim))


@DYNAMICS.register("residual_mlp")
class ResidualMLPDynamics(Dynamics):
    """z_{t+1} = z_t + dt · f_θ([z_t, u_t]).

    The last layer is zero-initialised so the model starts as the identity.
    ``stability_reg`` > 0 adds ``stability_reg * mean ||f(z)||^2`` (computed on
    the most recent batch passed through ``step``) to :meth:`extra_losses`,
    discouraging unbounded growth of the update field.
    """

    def __init__(
        self,
        latent_dim: int,
        control_dim: int = 0,
        hidden_dims: Sequence[int] = (128, 128),
        act: str = "gelu",
        dt: float = 1.0,
        stability_reg: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(latent_dim, control_dim)
        self.hidden_dims = tuple(hidden_dims)
        self.act = act
        self.dt = dt
        self.stability_reg = stability_reg
        self.net = make_mlp(latent_dim + control_dim, latent_dim, self.hidden_dims, act, zero_init_last=True)
        self._last_update_sq: Tensor | None = None

    def update(self, z: Tensor, u: Tensor | None = None) -> Tensor:
        """The residual field f_θ(z, u): (B, d) → (B, d)."""
        return self.net(_cat_control(z, u, self.control_dim))

    def step(self, z: Tensor, u: Tensor | None = None) -> Tensor:
        f = self.update(z, u)
        if self.stability_reg > 0 and torch.is_grad_enabled():
            self._last_update_sq = (f**2).sum(-1).mean()
        return z + self.dt * f

    def extra_losses(self) -> dict[str, Tensor]:
        if self.stability_reg <= 0 or self._last_update_sq is None:
            return {}
        return {"stability": self.stability_reg * self._last_update_sq}
