"""Diagonal (+ low-rank) linear SSM in the latent, driven by a nonlinear feature of z.

    z_{t+1} = diag(a) z_t + (U V^T) z_t + B σ(W z_t + c) + G u_t

``a`` is constrained to |a| < 1 either through ``tanh`` (allows negative /
oscillatory-sign modes) or ``exp(-softplus)`` (positive decays only). With
``rank=0`` and ``feature_dim=0`` this collapses to a stable diagonal linear map.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from nssc.dynamics._mlp_util import get_activation
from nssc.dynamics.base import Dynamics
from nssc.utils.registry import DYNAMICS


@DYNAMICS.register("ssm")
class SSMDynamics(Dynamics):
    def __init__(
        self,
        latent_dim: int,
        control_dim: int = 0,
        feature_dim: int | None = None,
        rank: int = 0,
        act: str = "tanh",
        param: str = "tanh",
        init_decay: float = 0.95,
        init_scale: float = 0.01,
        **kwargs,
    ) -> None:
        super().__init__(latent_dim, control_dim)
        if param not in ("tanh", "exp"):
            raise ValueError("param must be 'tanh' or 'exp'")
        d = latent_dim
        self.param = param
        self.rank = rank
        self.feature_dim = 2 * d if feature_dim is None else feature_dim
        # inverse-parameterise a ≈ init_decay
        if param == "tanh":
            a0 = torch.atanh(torch.full((d,), init_decay))
        else:
            # exp(-softplus(p)) = init_decay  →  softplus(p) = -log(init_decay)
            s = -torch.log(torch.tensor(init_decay))
            a0 = torch.full((d,), float(torch.log(torch.expm1(s))))
        self.a_raw = nn.Parameter(a0 + init_scale * torch.randn(d))
        if rank > 0:
            self.U = nn.Parameter(init_scale * torch.randn(d, rank))
            self.V = nn.Parameter(init_scale * torch.randn(d, rank))
        if self.feature_dim > 0:
            self.W = nn.Linear(d, self.feature_dim)
            self.sigma = get_activation(act)
            self.B = nn.Parameter(init_scale * torch.randn(d, self.feature_dim))
        self.G = nn.Parameter(init_scale * torch.randn(d, control_dim)) if control_dim > 0 else None

    # ---------------------------------------------------------------- pieces
    @property
    def a(self) -> Tensor:
        """Diagonal transition coefficients, |a| < 1: (d,)."""
        if self.param == "tanh":
            return torch.tanh(self.a_raw)
        return torch.exp(-F.softplus(self.a_raw))

    def linear_operator(self) -> Tensor:
        """Dense linear part diag(a) + U V^T: (d, d)."""
        A = torch.diag(self.a)
        if self.rank > 0:
            A = A + self.U @ self.V.T
        return A

    def step(self, z: Tensor, u: Tensor | None = None) -> Tensor:
        out = z * self.a
        if self.rank > 0:
            out = out + (z @ self.V) @ self.U.T
        if self.feature_dim > 0:
            out = out + self.sigma(self.W(z)) @ self.B.T
        if self.G is not None and u is not None:
            out = out + u @ self.G.T
        return out

    def eigenvalues(self) -> Tensor:
        """Eigenvalues of the linear part diag(a) + U V^T: (d,)."""
        return torch.linalg.eigvals(self.linear_operator().detach())
