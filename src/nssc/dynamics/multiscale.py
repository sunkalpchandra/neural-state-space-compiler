"""Slow/fast multi-scale latent dynamics.

The latent is split z = [z_slow, z_fast] (first ``slow_dim`` coordinates slow).

    z_slow' = z_slow + Δ_s f_s(z_slow)              (small Δ_s, or only every ``slow_every`` steps)
    z_fast' = f_f(z_fast, z_slow, u)                (residual MLP)

``mode='rate'``: slow block updates every step with rate Δ_s.
``mode='strided'``: slow block updates (with rate Δ_s·slow_every, so the mean
speed matches) only when ``t % slow_every == 0``; ``step`` accepts an optional
``t`` index and ``rollout`` threads it so the module stays stateless.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor

from nssc.dynamics._mlp_util import make_mlp
from nssc.dynamics.base import Dynamics
from nssc.utils.registry import DYNAMICS


@DYNAMICS.register("multiscale")
class MultiScaleDynamics(Dynamics):
    def __init__(
        self,
        latent_dim: int,
        control_dim: int = 0,
        slow_dim: int | None = None,
        slow_rate: float = 0.1,
        slow_every: int = 1,
        mode: str = "rate",
        hidden_dims: Sequence[int] = (64, 64),
        act: str = "gelu",
        fast_dt: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(latent_dim, control_dim)
        if mode not in ("rate", "strided"):
            raise ValueError("mode must be 'rate' or 'strided'")
        self.slow_dim = max(1, latent_dim // 2) if slow_dim is None else slow_dim
        if not 0 < self.slow_dim < latent_dim:
            raise ValueError("slow_dim must satisfy 0 < slow_dim < latent_dim")
        self.fast_dim = latent_dim - self.slow_dim
        self.slow_rate = slow_rate
        self.slow_every = max(1, slow_every)
        self.mode = mode
        self.fast_dt = fast_dt
        self.f_slow = make_mlp(self.slow_dim, self.slow_dim, hidden_dims, act)
        self.f_fast = make_mlp(latent_dim + control_dim, self.fast_dim, hidden_dims, act, zero_init_last=True)

    def split(self, z: Tensor) -> tuple[Tensor, Tensor]:
        return z[..., : self.slow_dim], z[..., self.slow_dim :]

    def _slow_update(self, zs: Tensor, t: int | None) -> Tensor:
        if self.mode == "rate":
            return zs + self.slow_rate * self.f_slow(zs)
        # strided: only advance on multiples of slow_every; t=None → treat as an update step
        if t is not None and (t % self.slow_every) != 0:
            return zs
        return zs + self.slow_rate * self.slow_every * self.f_slow(zs)

    def step(self, z: Tensor, u: Tensor | None = None, t: int | None = None) -> Tensor:
        zs, zf = self.split(z)
        inp = z if self.control_dim == 0 else torch.cat(
            [z, u if u is not None else z.new_zeros(z.shape[0], self.control_dim)], dim=-1
        )
        zf_new = zf + self.fast_dt * self.f_fast(inp)
        zs_new = self._slow_update(zs, t)
        return torch.cat([zs_new, zf_new], dim=-1)

    def rollout(self, z0: Tensor, horizon: int, u: Tensor | None = None, t0: int = 0) -> Tensor:
        zs = []
        z = z0
        for k in range(horizon):
            uk = None if u is None else u[:, k]
            z = self.step(z, uk, t=t0 + k)
            zs.append(z)
        return torch.stack(zs, dim=1)
