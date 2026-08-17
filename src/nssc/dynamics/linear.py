"""Linear and affine latent dynamics: z' = A z (+ b).

These are the simplest, most interpretable families and the natural baseline for
the compiler (PCA + linear). ``least_squares_fit`` gives a DMD-style closed-form
initialisation from paired latents.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from nssc.dynamics.base import Dynamics
from nssc.utils.registry import DYNAMICS


@DYNAMICS.register("linear")
class LinearDynamics(Dynamics):
    """z_{t+1} = A z_t (+ B u_t).

    ``A`` is initialised near the identity (``I + 0.01 * randn``) so untrained
    rollouts are stable. ``spectral_norm_max`` adds a soft hinge penalty
    ``relu(||A||_2 - spectral_norm_max)^2`` via :meth:`extra_losses`.
    """

    is_linear = True

    def __init__(
        self,
        latent_dim: int,
        control_dim: int = 0,
        spectral_norm_max: float | None = None,
        init_scale: float = 0.01,
        **kwargs,
    ) -> None:
        super().__init__(latent_dim, control_dim)
        self.spectral_norm_max = spectral_norm_max
        self.A = nn.Parameter(torch.eye(latent_dim) + init_scale * torch.randn(latent_dim, latent_dim))
        self.B = nn.Parameter(init_scale * torch.randn(latent_dim, control_dim)) if control_dim > 0 else None

    def _control(self, z: Tensor, u: Tensor | None) -> Tensor:
        if self.B is None or u is None:
            return torch.zeros_like(z)
        return u @ self.B.T

    def step(self, z: Tensor, u: Tensor | None = None) -> Tensor:
        return z @ self.A.T + self._control(z, u)

    def jacobian(self, z: Tensor, u: Tensor | None = None) -> Tensor:
        return self.A.detach().unsqueeze(0).expand(z.shape[0], -1, -1).clone()

    def eigenvalues(self) -> Tensor:
        """Complex eigenvalues of ``A``: (d,)."""
        return torch.linalg.eigvals(self.A.detach())

    def spectral_radius(self) -> float:
        return float(self.eigenvalues().abs().max())

    def extra_losses(self) -> dict[str, Tensor]:
        if self.spectral_norm_max is None:
            return {}
        sn = torch.linalg.matrix_norm(self.A, ord=2)
        return {"spectral_norm": torch.relu(sn - self.spectral_norm_max) ** 2}

    @torch.no_grad()
    def least_squares_fit(self, z_t: Tensor, z_next: Tensor, ridge: float = 1e-6) -> None:
        """Closed-form (DMD-like) fit of ``A`` from paired latents.

        ``z_t``, ``z_next``: (N, d) or (B, T, d) (flattened). Solves
        ``min_A ||z_next - z_t A^T||^2 + ridge ||A||^2`` and copies the result into ``A``.
        """
        A = _lstsq(z_t.reshape(-1, self.latent_dim), z_next.reshape(-1, self.latent_dim), ridge)
        self.A.copy_(A.to(self.A))

    @classmethod
    def from_least_squares(cls, z_t: Tensor, z_next: Tensor, ridge: float = 1e-6, **kwargs) -> LinearDynamics:
        d = z_t.shape[-1]
        m = cls(d, **kwargs)
        m.least_squares_fit(z_t, z_next, ridge)
        return m


def _lstsq(X: Tensor, Y: Tensor, ridge: float) -> Tensor:
    """Solve Y ≈ X A^T for A (out, in) with Tikhonov regularisation."""
    X = X.double()
    Y = Y.double()
    d = X.shape[1]
    G = X.T @ X + ridge * torch.eye(d, dtype=X.dtype, device=X.device)
    A_T = torch.linalg.solve(G, X.T @ Y)  # (in, out)
    return A_T.T.float()


@DYNAMICS.register("affine")
class AffineDynamics(LinearDynamics):
    """z_{t+1} = A z_t + b (+ B u_t)."""

    def __init__(self, latent_dim: int, control_dim: int = 0, **kwargs) -> None:
        super().__init__(latent_dim, control_dim, **kwargs)
        self.b = nn.Parameter(torch.zeros(latent_dim))

    def step(self, z: Tensor, u: Tensor | None = None) -> Tensor:
        return z @ self.A.T + self.b + self._control(z, u)

    def fixed_point(self) -> Tensor:
        """z* = (I - A)^{-1} b (via lstsq for robustness)."""
        eye = torch.eye(self.latent_dim, device=self.A.device)
        return torch.linalg.lstsq(eye - self.A.detach(), self.b.detach().unsqueeze(1)).solution.squeeze(1)

    @torch.no_grad()
    def least_squares_fit(self, z_t: Tensor, z_next: Tensor, ridge: float = 1e-6) -> None:
        """Closed-form fit of ``[A | b]`` by augmenting z with a constant 1."""
        X = z_t.reshape(-1, self.latent_dim)
        Y = z_next.reshape(-1, self.latent_dim)
        X1 = torch.cat([X, torch.ones(X.shape[0], 1, dtype=X.dtype, device=X.device)], dim=1)
        W = _lstsq(X1, Y, ridge)  # (d, d+1)
        self.A.copy_(W[:, : self.latent_dim].to(self.A))
        self.b.copy_(W[:, self.latent_dim].to(self.b))
