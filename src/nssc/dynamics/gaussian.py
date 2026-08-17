"""Gaussian (heteroscedastic) wrapper around any deterministic dynamics.

    z_{t+1} ~ N( F_base(z_t, u_t), diag(exp(logvar_θ(z_t))) )

``step`` returns the mean so the wrapped model behaves like a deterministic
dynamics for the rest of the pipeline; ``sample_step`` / ``rollout_samples`` /
``nll`` expose the stochastic interface used by the uncertainty module.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor

from nssc.dynamics._mlp_util import make_mlp
from nssc.dynamics.base import Dynamics
from nssc.utils.registry import DYNAMICS


@DYNAMICS.register("gaussian")
class GaussianDynamics(Dynamics):
    is_stochastic = True

    def __init__(
        self,
        latent_dim: int,
        control_dim: int = 0,
        base: str = "residual_mlp",
        base_kwargs: dict[str, Any] | None = None,
        var_hidden_dims: Sequence[int] = (64,),
        act: str = "gelu",
        min_logvar: float = -10.0,
        max_logvar: float = 4.0,
        init_logvar: float = -2.0,
        **kwargs,
    ) -> None:
        super().__init__(latent_dim, control_dim)
        if base == "gaussian":
            raise ValueError("GaussianDynamics cannot wrap itself")
        self.base_key = base
        self.base = DYNAMICS.build(base, latent_dim=latent_dim, control_dim=control_dim, **(base_kwargs or {}))
        self.is_linear = self.base.is_linear
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar
        self.logvar_net = make_mlp(latent_dim + control_dim, latent_dim, var_hidden_dims, act, zero_init_last=True)
        # bias the (zeroed) last layer to init_logvar so initial variance is sensible
        with torch.no_grad():
            self.logvar_net[-1].bias.fill_(init_logvar)

    # ------------------------------------------------------------ moments
    def _inp(self, z: Tensor, u: Tensor | None) -> Tensor:
        if self.control_dim == 0:
            return z
        if u is None:
            u = z.new_zeros(z.shape[0], self.control_dim)
        return torch.cat([z, u], dim=-1)

    def logvar(self, z: Tensor, u: Tensor | None = None) -> Tensor:
        """Diagonal log-variance of the transition: (B, d) → (B, d)."""
        return self.logvar_net(self._inp(z, u)).clamp(self.min_logvar, self.max_logvar)

    def moments(self, z: Tensor, u: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """(mean, logvar) of z_{t+1} | z_t."""
        return self.base.step(z, u), self.logvar(z, u)

    def step(self, z: Tensor, u: Tensor | None = None) -> Tensor:
        return self.base.step(z, u)

    def jacobian(self, z: Tensor, u: Tensor | None = None) -> Tensor:
        return self.base.jacobian(z, u)

    def extra_losses(self) -> dict[str, Tensor]:
        return self.base.extra_losses()

    # ------------------------------------------------------------ stochastic
    def sample_step(self, z: Tensor, u: Tensor | None = None) -> Tensor:
        mean, lv = self.moments(z, u)
        return mean + torch.randn_like(mean) * torch.exp(0.5 * lv)

    def nll(self, z: Tensor, z_next: Tensor, u: Tensor | None = None) -> Tensor:
        """Mean Gaussian negative log-likelihood of ``z_next`` given ``z``. Accepts (B,d) or (B,T,d)."""
        d = self.latent_dim
        z = z.reshape(-1, d)
        z_next = z_next.reshape(-1, d)
        u_flat = None if u is None else u.reshape(z.shape[0], -1)
        mean, lv = self.moments(z, u_flat)
        per_dim = 0.5 * (lv + (z_next - mean) ** 2 * torch.exp(-lv) + math.log(2 * math.pi))
        return per_dim.sum(-1).mean()

    def rollout_samples(self, z0: Tensor, horizon: int, n_samples: int = 10, u: Tensor | None = None) -> Tensor:
        """Monte-Carlo rollouts: (B, d) → (S, B, H, d)."""
        B = z0.shape[0]
        z = z0.unsqueeze(0).expand(n_samples, -1, -1).reshape(n_samples * B, -1)
        outs = []
        for k in range(horizon):
            uk = None
            if u is not None:
                uk = u[:, k].unsqueeze(0).expand(n_samples, -1, -1).reshape(n_samples * B, -1)
            z = self.sample_step(z, uk)
            outs.append(z)
        return torch.stack(outs, dim=1).reshape(n_samples, B, horizon, -1)

    def rollout_moments(self, z0: Tensor, horizon: int, n_samples: int = 10, u: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Empirical (mean, std) over ``rollout_samples``: each (B, H, d)."""
        s = self.rollout_samples(z0, horizon, n_samples, u)
        return s.mean(0), s.std(0, unbiased=n_samples > 1)


__all__ = ["GaussianDynamics"]
