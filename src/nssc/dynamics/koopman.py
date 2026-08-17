"""Koopman-style dynamics: lift, apply a linear operator, read back.

    φ(z) = [z, g_θ(z)] ∈ R^m,   z_{t+1} = C K φ(z_t)

with learnable lifting ``g_θ`` (MLP), operator ``K`` (m×m) and readout ``C``.
Because ``z`` is included in ``φ`` the readout can be the exact projection onto
the first ``d`` coordinates (``exact_readout=True``, default) or a learned
linear map. The consistency loss ``||φ(z_next) - K φ(z)||^2`` regularises ``K``
towards a genuine Koopman operator on the lifted space.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from nssc.dynamics._mlp_util import make_mlp
from nssc.dynamics.base import Dynamics
from nssc.utils.registry import DYNAMICS


@DYNAMICS.register("koopman")
class KoopmanDynamics(Dynamics):
    def __init__(
        self,
        latent_dim: int,
        control_dim: int = 0,
        obs_dim_lift: int | None = None,
        hidden_dims: Sequence[int] = (64, 64),
        act: str = "gelu",
        residual: bool = False,
        exact_readout: bool = True,
        consistency_weight: float = 1.0,
        init_scale: float = 0.01,
        **kwargs,
    ) -> None:
        super().__init__(latent_dim, control_dim)
        d = latent_dim
        self.obs_dim_lift = obs_dim_lift if obs_dim_lift is not None else 4 * d
        if self.obs_dim_lift < d:
            raise ValueError("obs_dim_lift must be >= latent_dim (z is concatenated into φ)")
        self.n_extra = self.obs_dim_lift - d
        self.residual = residual
        self.exact_readout = exact_readout
        self.consistency_weight = consistency_weight

        m = self.obs_dim_lift
        self.lift = make_mlp(d, self.n_extra, hidden_dims, act) if self.n_extra > 0 else None
        if residual:
            self.K_delta = nn.Parameter(init_scale * torch.randn(m, m))
        else:
            self.K_full = nn.Parameter(torch.eye(m) + init_scale * torch.randn(m, m))
        self.B = nn.Parameter(init_scale * torch.randn(m, control_dim)) if control_dim > 0 else None
        if not exact_readout:
            C0 = torch.zeros(d, m)
            C0[:, :d] = torch.eye(d)
            self.C = nn.Parameter(C0 + init_scale * torch.randn(d, m))
        self._last_consistency: Tensor | None = None

    # ------------------------------------------------------------ pieces
    @property
    def K(self) -> Tensor:
        if self.residual:
            return torch.eye(self.obs_dim_lift, device=self.K_delta.device) + self.K_delta
        return self.K_full

    def lift_fn(self, z: Tensor) -> Tensor:
        """φ(z) = [z, g_θ(z)]: (B, d) → (B, m)."""
        if self.lift is None:
            return z
        return torch.cat([z, self.lift(z)], dim=-1)

    def readout(self, phi: Tensor) -> Tensor:
        if self.exact_readout:
            return phi[..., : self.latent_dim]
        return phi @ self.C.T

    def lifted_step(self, phi: Tensor, u: Tensor | None = None) -> Tensor:
        out = phi @ self.K.T
        if self.B is not None and u is not None:
            out = out + u @ self.B.T
        return out

    def step(self, z: Tensor, u: Tensor | None = None) -> Tensor:
        return self.readout(self.lifted_step(self.lift_fn(z), u))

    def rollout(self, z0: Tensor, horizon: int, u: Tensor | None = None) -> Tensor:
        # Re-lift each step (matches step semantics; the lifted-linear rollout is
        # available via ``rollout_lifted`` for analysis).
        return super().rollout(z0, horizon, u)

    def rollout_lifted(self, z0: Tensor, horizon: int) -> Tensor:
        """Purely linear rollout in lifted space (no re-lifting): (B, H, d)."""
        phi = self.lift_fn(z0)
        outs = []
        for _ in range(horizon):
            phi = self.lifted_step(phi)
            outs.append(self.readout(phi))
        return torch.stack(outs, dim=1)

    # ------------------------------------------------------------ analysis
    def eigenvalues(self) -> Tensor:
        return torch.linalg.eigvals(self.K.detach())

    def consistency_loss(self, z: Tensor, z_next: Tensor, u: Tensor | None = None) -> Tensor:
        """||φ(z_next) - K φ(z)||^2 averaged over the batch. Accepts (B,d) or (B,T,d).

        Also caches the value so that :meth:`extra_losses` can report it.
        """
        d = self.latent_dim
        z = z.reshape(-1, d)
        z_next = z_next.reshape(-1, d)
        u_flat = None if u is None else u.reshape(z.shape[0], -1)
        pred = self.lifted_step(self.lift_fn(z), u_flat)
        loss = ((self.lift_fn(z_next) - pred) ** 2).sum(-1).mean()
        self._last_consistency = loss
        return loss

    def extra_losses(self) -> dict[str, Tensor]:
        if self._last_consistency is None:
            return {}
        return {"koopman_consistency": self.consistency_weight * self._last_consistency}
